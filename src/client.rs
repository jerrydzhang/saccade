//! The client half of the API: one `POST /api/v1/command`, the shared
//! envelope, and the failure taxonomy the CLI renders.

use crate::Command;
use crate::api::{Envelope, WireContext, WireTier};
use crate::db::StoredRecord;
use crate::store::{Context, Tier};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum ClientFail {
    /// Nothing answered at the URL
    ServerUnreachable { url: String },
    /// The server answered and refused; `code` is its error code.
    Refused { code: String, detail: String },
    Undiscovered {
        db: std::path::PathBuf,
        hint: std::path::PathBuf,
    },
    Mismatch {
        url: String,
        serves: String,
        expected: std::path::PathBuf,
    },
    /// The server's version is known and breaking against this
    /// binary's: a 0.x minor bump breaks the wire, binaries move in
    /// step, and no command write may be posted across the line.
    VersionMismatch {
        url: String,
        ours: &'static str,
        theirs: String,
    },
    /// The server's version could not be established, or no ratified
    /// compatibility rule spans the two version lines: the write
    /// refuses rather than guessing.
    VersionUnproven {
        url: String,
        ours: &'static str,
        theirs: String,
        why: String,
    },
}

impl ClientFail {
    pub fn code(&self) -> &'static str {
        match self {
            ClientFail::ServerUnreachable { .. } => "server_required",
            ClientFail::Refused { .. } => "server_refused",
            ClientFail::Undiscovered { .. } => "server_undiscovered",
            ClientFail::Mismatch { .. } => "server_mismatch",
            ClientFail::VersionMismatch { .. } => "server_version_mismatch",
            ClientFail::VersionUnproven { .. } => "server_version_unproven",
        }
    }
}

impl std::fmt::Display for ClientFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientFail::ServerUnreachable { url } => write!(
                f,
                "no server answered at {url}: start one (`sac serve`) or pass the right --server"
            ),
            ClientFail::Refused { code, detail } => write!(f, "{code}: {detail}"),
            ClientFail::Undiscovered { db, hint } => write!(
                f,
                "no server discovered for {}: no address hint at {}; start one beside that tracker (`sac serve`)",
                db.display(),
                hint.display()
            ),
            ClientFail::Mismatch {
                url,
                serves,
                expected,
            } => write!(
                f,
                "the server at {url} serves {serves}, not {}; the address hint is stale or points elsewhere",
                expected.display()
            ),
            ClientFail::VersionMismatch { url, ours, theirs } => write!(
                f,
                "the server at {url} runs sac {theirs}, this binary is sac {ours}; \
                 a 0.x minor bump is breaking - move the binaries in step before writing"
            ),
            ClientFail::VersionUnproven {
                url,
                ours,
                theirs,
                why,
            } => write!(
                f,
                "the server at {url} reports version '{theirs}', this binary is sac {ours}; \
                 {why} - no command write was sent"
            ),
        }
    }
}

#[derive(Deserialize)]
struct RecordsBody {
    records: Vec<StoredRecord>,
    /// The born task's token, present only on create replies; servers
    /// that predate it omit the field
    #[serde(default)]
    id: Option<String>,
}

#[derive(Deserialize)]
struct ErrorDetail {
    code: String,
    detail: serde_json::Value,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

/// One command's outcome: the records it landed, and the born task's
/// token when the command created one.
#[derive(Debug)]
pub struct Reply {
    pub records: Vec<StoredRecord>,
    pub id: Option<String>,
}

/// Sends one command and returns the records it landed
pub fn send(
    url: &str,
    context: &Context,
    command: Command,
    at: Option<u64>,
) -> Result<Reply, ClientFail> {
    require_compatible_peer(url)?;
    let envelope = Envelope {
        context: WireContext {
            actor: context.actor.clone(),
            tier: match context.tier {
                Tier::Human => WireTier::Human,
                Tier::Agent => WireTier::Agent,
                Tier::System => unreachable!("the API has no system tier"),
            },
            scope: context.scope.clone(),
        },
        command,
        at,
        client: Some(env!("CARGO_PKG_VERSION").to_string()),
    };
    let mut response = ureq::post(&format!("{url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(serde_json::to_value(&envelope).expect("the envelope is plain data"))
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;

    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    if status == 200 {
        let body: RecordsBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
            code: "malformed_response".into(),
            detail: e.to_string(),
        })?;
        Ok(Reply {
            records: body.records,
            id: body.id,
        })
    } else {
        let body: ErrorBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
            code: "malformed_response".into(),
            detail: e.to_string(),
        })?;
        Err(ClientFail::Refused {
            code: body.error.code,
            detail: body.error.detail.to_string(),
        })
    }
}

#[derive(Debug)]
pub struct ServerIdentity {
    pub version: String,
    pub db: Option<String>,
}

/// The version line a peer must prove before any command write posts:
/// exactly the major.minor.patch sac advertises, whole input. None
/// when the text is truncated, extended, or otherwise not a version
/// this rule can read.
fn version_line(text: &str) -> Option<(u64, u64)> {
    let mut parts = text.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let _patch = parts.next()?.parse::<u64>().ok()?;
    parts.next().is_none().then_some((major, minor))
}

/// Every command write crosses this gate before its POST: a peer on
/// the same major and minor is compatible whatever its patch, a peer
/// on the same 0.x line with a different minor is breaking (the wire
/// may have silently dropped fields), and a peer off the major line
/// or without a readable version is refused rather than guessed at.
fn require_compatible_peer(url: &str) -> Result<(), ClientFail> {
    let ours = env!("CARGO_PKG_VERSION");
    let id = identity(url)?;
    let theirs = version_line(&id.version);
    let mine = version_line(ours).expect("CARGO_PKG_VERSION is major.minor.patch");
    let Some((major, minor)) = theirs else {
        return Err(ClientFail::VersionUnproven {
            url: url.to_string(),
            ours,
            theirs: id.version,
            why: "the version does not read as major.minor[.patch]".into(),
        });
    };
    if major != mine.0 {
        return Err(ClientFail::VersionUnproven {
            url: url.to_string(),
            ours,
            theirs: id.version,
            why: format!(
                "no ratified compatibility rule spans the {major} and {} version lines",
                mine.0
            ),
        });
    }
    if minor != mine.1 {
        return Err(ClientFail::VersionMismatch {
            url: url.to_string(),
            ours,
            theirs: id.version,
        });
    }
    Ok(())
}

pub fn identity(url: &str) -> Result<ServerIdentity, ClientFail> {
    let mut response = ureq::get(&format!("{url}/api/v1/version"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    if status != 200 {
        return Err(ClientFail::Refused {
            code: "malformed_response".into(),
            detail: format!("the identity route answered status {status}"),
        });
    }
    let body: serde_json::Value = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
    })?;
    Ok(ServerIdentity {
        version: body["version"].as_str().unwrap_or_default().to_string(),
        db: body["db"].as_str().map(str::to_string),
    })
}

/// The hint beside the expected database, verified to serve it -
/// missing, unreachable, or mismatched refuses; there is no fallback.
pub fn discover(expected_db: &std::path::Path) -> Result<String, ClientFail> {
    let hint = crate::paths::address_at(expected_db);
    let url = std::fs::read_to_string(&hint)
        .map_err(|_| ClientFail::Undiscovered {
            db: expected_db.to_path_buf(),
            hint: hint.clone(),
        })?
        .trim()
        .to_string();
    if url.is_empty() {
        return Err(ClientFail::Undiscovered {
            db: expected_db.to_path_buf(),
            hint,
        });
    }
    let id = identity(&url)?;
    let expected = canonical_db(expected_db);
    if id.db.as_deref() != Some(expected.to_string_lossy().as_ref()) {
        return Err(ClientFail::Mismatch {
            url,
            serves: id.db.unwrap_or_else(|| "an unnamed database".into()),
            expected,
        });
    }
    Ok(url)
}

fn canonical_db(db_path: &std::path::Path) -> std::path::PathBuf {
    std::fs::canonicalize(db_path).unwrap_or_else(|_| db_path.to_path_buf())
}

pub fn records(url: &str) -> Result<Vec<StoredRecord>, ClientFail> {
    let mut response = ureq::get(&format!("{url}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    if status != 200 {
        return Err(error_of(&text));
    }
    #[derive(Deserialize)]
    struct RecordsBody {
        records: Vec<StoredRecord>,
    }
    let body: RecordsBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
    })?;
    Ok(body.records)
}

/// The handshake's client half: warn once when this binary and the
/// server disagree. A fetch that fails stays quiet - the send that
/// follows reports an unreachable server without our duplication.
pub fn handshake(url: &str) {
    if let Ok(id) = identity(url) {
        let mine = env!("CARGO_PKG_VERSION");
        if id.version != mine {
            eprintln!(
                "warning: server runs sac {}, this binary is sac {mine}; a mismatch may refuse",
                id.version
            );
        }
    }
}

/// POST one JSON body and read the reply: 200 with a parsed body, or
/// the typed error envelope every surface shares.
fn post_json<T: serde::de::DeserializeOwned>(
    url: &str,
    path: &str,
    body: serde_json::Value,
) -> Result<T, ClientFail> {
    let mut response = ureq::post(&format!("{url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(body)
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    if status == 200 {
        serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
            code: "malformed_response".into(),
            detail: e.to_string(),
        })
    } else {
        Err(error_of(&text))
    }
}

fn error_of(text: &str) -> ClientFail {
    #[derive(Deserialize)]
    struct ErrorDetail {
        code: String,
        detail: serde_json::Value,
    }
    #[derive(Deserialize)]
    struct ErrorBody {
        error: ErrorDetail,
    }
    let body: ErrorBody = serde_json::from_str(text).unwrap_or(ErrorBody {
        error: ErrorDetail {
            code: "malformed_response".into(),
            detail: serde_json::Value::String(bound(text)),
        },
    });
    ClientFail::Refused {
        code: body.error.code,
        detail: bound(&body.error.detail.to_string()),
    }
}

/// A refusal's detail stays readable: an HTML 404 from a server that
/// does not know a route is noise, not a message.
fn bound(detail: &str) -> String {
    const LIMIT: usize = 240;
    let flat: String = detail
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect();
    if flat.len() <= LIMIT {
        return flat;
    }
    let cut = flat
        .char_indices()
        .nth(LIMIT)
        .map(|(i, _)| i)
        .unwrap_or(LIMIT);
    format!("{}...", &flat[..cut])
}

// -- the native integration's client half -------------------------------

/// What the native integration learns when its session attaches.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttachReply {
    pub registered: bool,
    /// The birth record position: the machine identity channel
    pub incarnation: usize,
    /// The incarnation's dense public number, the human-facing i-N
    pub num: usize,
    pub records: Vec<StoredRecord>,
}

/// Attach a native session: an existing exact native session resolves
/// its incarnation, a fresh one registers - the machinery's own act.
pub fn attach(
    url: &str,
    native_id: &str,
    session_file: &std::path::Path,
) -> Result<AttachReply, ClientFail> {
    post_json(
        url,
        "/api/v1/native/attach",
        serde_json::json!({
            "native_id": native_id,
            "session_file": session_file.display().to_string(),
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

/// One sent message's facts: its stable birth position and the dense
/// m-N, plus the sender's resolved identity.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MessageFacts {
    pub seq: usize,
    pub num: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SendMessageReply {
    pub records: Vec<StoredRecord>,
    pub message: MessageFacts,
    pub from: MessageFacts,
}

/// Send a durable addressed message from the declared incarnation;
/// `to` and `reply_to` are the public dense numbers every face names.
pub fn send_message(
    url: &str,
    context: &Context,
    to: usize,
    reply_to: Option<usize>,
    task: Option<usize>,
    body: &crate::Prose,
) -> Result<SendMessageReply, ClientFail> {
    post_json(
        url,
        "/api/v1/messages/send",
        serde_json::json!({
            "context": {
                "actor": context.actor.as_str(),
                "tier": match context.tier {
                    Tier::Human => "human",
                    Tier::Agent => "agent",
                    Tier::System => unreachable!("the API has no system tier"),
                },
                "scope": context.scope,
            },
            "to": to,
            "reply_to": reply_to,
            "task": task,
            "body": body.as_str(),
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

/// Record the recipient's submission: the delivery machinery's act.
pub fn submit_message(
    url: &str,
    incarnation: usize,
    message: usize,
) -> Result<Vec<StoredRecord>, ClientFail> {
    #[derive(Deserialize)]
    struct SubmitBody {
        records: Vec<StoredRecord>,
    }
    let body: SubmitBody = post_json(
        url,
        "/api/v1/messages/submit",
        serde_json::json!({
            "incarnation": incarnation,
            "message": message,
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )?;
    Ok(body.records)
}

/// One message as the mailbox read returns it: the stable birth
/// position, the dense endpoints, and the delivery state.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MailboxMessage {
    pub seq: usize,
    pub num: Option<usize>,
    pub from: MailboxEndpoint,
    pub to: MailboxEndpoint,
    pub reply_to: Option<usize>,
    pub task: Option<usize>,
    pub state: String,
    pub body: String,
    pub born_at: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MailboxEndpoint {
    pub seq: usize,
    pub num: Option<usize>,
    pub actor: String,
}

/// The reader's own mail, in birth order, any state: a read the
/// integration polls, never a subscription the tracker keeps.
pub fn mailbox(url: &str, incarnation: usize) -> Result<Vec<MailboxMessage>, ClientFail> {
    #[derive(Deserialize)]
    struct MailboxBody {
        messages: Vec<MailboxMessage>,
    }
    let mut response = ureq::get(&format!(
        "{url}/api/v1/messages/pending?incarnation={incarnation}"
    ))
    .config()
    .http_status_as_error(false)
    .build()
    .call()
    .map_err(|_| ClientFail::ServerUnreachable {
        url: url.to_string(),
    })?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    if status != 200 {
        return Err(error_of(&text));
    }
    let body: MailboxBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
    })?;
    Ok(body.messages)
}

// -- the task-free workspace controller's client half --------------------

/// What a creation provisioned: actual Git facts and the birth record
/// carrying them - the workspace's identity, never a numbering.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkspaceBirth {
    pub seq: usize,
    pub base: String,
    pub branch: String,
    pub path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CreateReply {
    pub created: bool,
    pub workspace: WorkspaceBirth,
    pub records: Vec<StoredRecord>,
}

/// Provision a NEW task-free workspace through the server's Git
/// effects; the birth lands with the facts the effect produced.
pub fn create_workspace(
    url: &str,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<CreateReply, ClientFail> {
    post_json(
        url,
        "/api/v1/workspace/create",
        serde_json::json!({
            "label": label,
            "base": base,
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChangeReply {
    pub changed: bool,
    pub incarnation: usize,
    pub num: usize,
    pub from: String,
    pub to: String,
    pub records: Vec<StoredRecord>,
}

/// Move a stopped native session's saved working directory to an
/// explicit existing one; the server refuses a live or unprovable
/// session before any mutation.
pub fn change_workspace(
    url: &str,
    incarnation: usize,
    to: &std::path::Path,
) -> Result<ChangeReply, ClientFail> {
    post_json(
        url,
        "/api/v1/workspace/change",
        serde_json::json!({
            "incarnation": incarnation,
            "to": to.display().to_string(),
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

/// Which incarnations hold a live delivery connection right now: the
/// activation decision's operational input, never a liveness proof -
/// absence is not proof a native process exited.
pub fn bridge_connections(url: &str) -> Result<Vec<usize>, ClientFail> {
    #[derive(Deserialize)]
    struct Entry {
        incarnation: usize,
    }
    #[derive(Deserialize)]
    struct BridgeBody {
        incarnations: Vec<Entry>,
    }
    let mut response = ureq::get(&format!("{url}/api/v1/native/bridge"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    if status != 200 {
        return Err(error_of(&text));
    }
    let body: BridgeBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
    })?;
    Ok(body
        .incarnations
        .into_iter()
        .map(|e| e.incarnation)
        .collect())
}

/// Which incarnation registered under a compose launch token, if any:
/// how a launching door learns the birth it caused. None when the
/// token is unknown or expired - the session may still register later.
pub fn launch_of(url: &str, token: &str) -> Result<Option<(usize, usize)>, ClientFail> {
    #[derive(Deserialize)]
    struct LaunchBody {
        incarnation: usize,
        num: usize,
    }
    let mut response = ureq::get(&format!(
        "{url}/api/v1/native/launch?token={}",
        urlencoding_lite(token)
    ))
    .config()
    .http_status_as_error(false)
    .build()
    .call()
    .map_err(|_| ClientFail::ServerUnreachable {
        url: url.to_string(),
    })?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    if status == 404 {
        return Ok(None);
    }
    if status != 200 {
        return Err(error_of(&text));
    }
    let body: LaunchBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
    })?;
    Ok(Some((body.incarnation, body.num)))
}

/// Tokens are self-generated hex: percent-encoding is defense in depth
/// for the one reserved character a query cannot carry bare.
fn urlencoding_lite(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    for b in token.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}
