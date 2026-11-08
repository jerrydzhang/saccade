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
    /// The server answered and refused; `code` is its error code, and
    /// a landed record the refusal names (a send's own m-N) rides
    /// beside it so the caller keeps the reference
    Refused {
        code: String,
        detail: String,
        message: Option<MessageFacts>,
    },
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
            ClientFail::Refused {
                code,
                detail,
                message: _,
            } => write!(f, "{code}: {detail}"),
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
    // send_json rides Content-Type: application/json - the one content
    // type the server's POST policy accepts, so the CLI never trips it
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
            message: None,
        })?;
        Ok(Reply {
            records: body.records,
            id: body.id,
        })
    } else {
        let body: ErrorBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
            code: "malformed_response".into(),
            detail: e.to_string(),
            message: None,
        })?;
        Err(ClientFail::Refused {
            code: body.error.code,
            detail: body.error.detail.to_string(),
            message: None,
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
            message: None,
        });
    }
    let body: serde_json::Value = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
        message: None,
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
        message: None,
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
            message: None,
        })
    } else {
        Err(error_of(&text))
    }
}

/// A POST whose 200 reply carries no body worth reading: the door
/// cares that the record landed, and failures surface by code.
fn post_json_unit(url: &str, path: &str, body: serde_json::Value) -> Result<(), ClientFail> {
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
        Ok(())
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
        #[serde(default)]
        message: Option<MessageFacts>,
    }
    let body: ErrorBody = serde_json::from_str(text).unwrap_or(ErrorBody {
        error: ErrorDetail {
            code: "malformed_response".into(),
            detail: serde_json::Value::String(bound(text)),
        },
        message: None,
    });
    ClientFail::Refused {
        code: body.error.code,
        detail: bound(&body.error.detail.to_string()),
        message: body.message,
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
    /// The native-submission attempt's truth; absent only from a
    /// pre-attempt server, never from this wire's own success reply
    #[serde(default)]
    pub submission: Option<SubmissionFacts>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubmissionFacts {
    pub state: String,
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
    let mut response = ureq::get(&format!("{url}/api/v1/messages?incarnation={incarnation}"))
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
        message: None,
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

/// The lifecycle controller's client half: clean and restore name the
/// exact recorded workspace path; the server runs the ordered effects
/// and lands the completed facts itself.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CleanReply {
    pub cleaned: bool,
    pub already_cleaned: bool,
    pub workspace: WorkspaceBirth,
    pub verified: Vec<String>,
    pub records: Vec<StoredRecord>,
}

pub fn clean_workspace(url: &str, path: &std::path::Path) -> Result<CleanReply, ClientFail> {
    post_json(
        url,
        "/api/v1/workspace/clean",
        serde_json::json!({
            "path": path.display().to_string(),
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RestoreReply {
    pub restored: bool,
    pub already_restored: bool,
    pub reused: bool,
    pub tip: Option<String>,
    pub workspace: WorkspaceBirth,
    pub records: Vec<StoredRecord>,
}

pub fn restore_workspace(url: &str, path: &std::path::Path) -> Result<RestoreReply, ClientFail> {
    post_json(
        url,
        "/api/v1/workspace/restore",
        serde_json::json!({
            "path": path.display().to_string(),
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReconcileEntry {
    pub seq: usize,
    pub path: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReconcileReply {
    pub applied: bool,
    pub cleaned: Vec<ReconcileEntry>,
    pub preserved: Vec<ReconcileEntry>,
    pub conflicts: Vec<ReconcileEntry>,
    pub records: Vec<StoredRecord>,
}

/// The one-time cutover verification: dry run by default, recording
/// only under `--apply`.
pub fn reconcile_workspaces(url: &str, apply: bool) -> Result<ReconcileReply, ClientFail> {
    post_json(
        url,
        "/api/v1/workspace/reconcile",
        serde_json::json!({
            "apply": apply,
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

/// The launch door's own birth: the pre-allocated native id and the
/// opaque host address, recorded before the harness's first save. The
/// association exists at launch time - no correlation token, no
/// attach wait. The harness rides the birth: native identity is
/// interpreted within it.
pub fn register_incarnation(
    url: &str,
    native_id: &str,
    address: &str,
    adopted: bool,
    harness: crate::Harness,
) -> Result<(), ClientFail> {
    post_json_unit(
        url,
        "/api/v1/incarnations/register",
        serde_json::json!({
            "native_id": native_id,
            "address": address,
            "adopted": adopted,
            "harness": harness,
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

/// The association's address moves with a relaunch: the same
/// incarnation, another terminal.
pub fn assign_host_address(
    url: &str,
    incarnation: usize,
    address: &str,
    adopted: bool,
) -> Result<(), ClientFail> {
    post_json_unit(
        url,
        "/api/v1/incarnations/address",
        serde_json::json!({
            "incarnation": incarnation,
            "address": address,
            "adopted": adopted,
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
}

// -- the board doors' client half ----------------------------------------
//
// The same envelope law as the command face: the context is built
// from the caller's store Context, the reply's failures carry the
// server's codes, and every read is a plain GET.

pub(crate) fn wire_context(context: &Context) -> crate::api::WireContext {
    crate::api::WireContext {
        actor: context.actor.clone(),
        tier: match context.tier {
            Tier::Human => crate::api::WireTier::Human,
            Tier::Agent => crate::api::WireTier::Agent,
            Tier::System => unreachable!("the API has no system tier"),
        },
        scope: context.scope.clone(),
    }
}

fn board_reply<T: serde::de::DeserializeOwned>(
    url: &str,
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<T, ClientFail> {
    let mut response = response.map_err(|_| ClientFail::ServerUnreachable {
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
        return serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
            code: "malformed_response".into(),
            detail: e.to_string(),
            message: None,
        });
    }
    let body: ErrorBody = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
        message: None,
    })?;
    Err(ClientFail::Refused {
        code: body.error.code,
        detail: body.error.detail.to_string(),
        message: None,
    })
}

fn board_post<T: serde::de::DeserializeOwned>(
    url: &str,
    route: &str,
    body: &impl serde::Serialize,
) -> Result<T, ClientFail> {
    let response = ureq::post(&format!("{url}{route}"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(serde_json::to_value(body).expect("board requests are plain data"));
    board_reply(url, response)
}

fn board_read<T: serde::de::DeserializeOwned>(url: &str, route: &str) -> Result<T, ClientFail> {
    let response = ureq::get(&format!("{url}{route}"))
        .config()
        .http_status_as_error(false)
        .build()
        .call();
    board_reply(url, response)
}

pub fn board_edit(
    url: &str,
    task: usize,
    actor: Option<&str>,
) -> Result<crate::serve::board_wire::EditReply, ClientFail> {
    let actor = actor.map(|a| query_escape(a.to_string()));
    board_read(
        url,
        &format!(
            "/api/v1/boards/edit?task={task}{}",
            actor.map(|a| format!("&actor={a}")).unwrap_or_default()
        ),
    )
}

fn query_escape(value: String) -> String {
    // ureq encodes nothing on our side; query values here are actor
    // names, so the reserved few percent-escape minimally
    value
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('?', "%3F")
        .replace('#', "%23")
}

pub fn board_checkpoint(
    url: &str,
    context: &Context,
    task: usize,
    files: &crate::serve::board_wire::BoardFiles,
    base: &str,
    message: &str,
) -> Result<crate::serve::board_wire::CheckpointReply, ClientFail> {
    board_post(
        url,
        "/api/v1/boards/checkpoint",
        &crate::serve::board_wire::CheckpointRequest {
            context: wire_context(context),
            task,
            files: files.clone(),
            base: base.to_string(),
            message: message.to_string(),
            client: Some(env!("CARGO_PKG_VERSION").to_string()),
        },
    )
}

pub fn board_commit(
    url: &str,
    context: &Context,
    task: usize,
    files: &crate::serve::board_wire::BoardFiles,
    message: &str,
) -> Result<crate::serve::board_wire::CommitReply, ClientFail> {
    board_post(
        url,
        "/api/v1/boards/commit",
        &crate::serve::board_wire::CommitRequest {
            context: wire_context(context),
            task,
            files: files.clone(),
            message: message.to_string(),
            client: Some(env!("CARGO_PKG_VERSION").to_string()),
        },
    )
}

pub fn board_discard(
    url: &str,
    context: &Context,
    task: usize,
) -> Result<crate::serve::board_wire::DiscardReply, ClientFail> {
    board_post(
        url,
        "/api/v1/boards/discard",
        &crate::serve::board_wire::DiscardRequest {
            context: wire_context(context),
            task,
            client: Some(env!("CARGO_PKG_VERSION").to_string()),
        },
    )
}

pub fn board_update(
    url: &str,
    task: usize,
    base: &str,
    files: &crate::serve::board_wire::BoardFiles,
) -> Result<crate::serve::board_wire::UpdateReply, ClientFail> {
    board_post(
        url,
        "/api/v1/boards/update",
        &crate::serve::board_wire::UpdateRequest {
            task,
            base: base.to_string(),
            files: files.clone(),
        },
    )
}

/// One revision's tree; `revision` absent means the current one.
pub fn board_get_tree(
    url: &str,
    task: usize,
    revision: Option<usize>,
) -> Result<crate::serve::board_wire::TreeReply, ClientFail> {
    board_read(
        url,
        &format!(
            "/api/v1/boards/get?task={task}{}",
            revision
                .map(|r| format!("&revision={r}"))
                .unwrap_or_default()
        ),
    )
}

/// One file out of a revision.
pub fn board_get_component(
    url: &str,
    task: usize,
    revision: Option<usize>,
    select: &str,
) -> Result<crate::serve::board_wire::GetReply, ClientFail> {
    board_read(
        url,
        &format!(
            "/api/v1/boards/get?task={task}{}&select={select}",
            revision
                .map(|r| format!("&revision={r}"))
                .unwrap_or_default()
        ),
    )
}

pub fn board_get_file(
    url: &str,
    task: usize,
    revision: Option<usize>,
    path: &str,
) -> Result<crate::serve::board_wire::GetReply, ClientFail> {
    board_read(
        url,
        &format!(
            "/api/v1/boards/get?task={task}{}&path={}",
            revision
                .map(|r| format!("&revision={r}"))
                .unwrap_or_default(),
            query_escape(path.to_string())
        ),
    )
}

pub fn board_diff(
    url: &str,
    task: usize,
    from: usize,
    to: usize,
) -> Result<crate::serve::board_wire::DiffReply, ClientFail> {
    board_read(
        url,
        &format!("/api/v1/boards/diff?task={task}&from={from}&to={to}"),
    )
}

pub fn board_log(url: &str, task: usize) -> Result<serde_json::Value, ClientFail> {
    board_read(url, &format!("/api/v1/boards/log?task={task}"))
}

pub fn board_path(url: &str, task: usize) -> Result<serde_json::Value, ClientFail> {
    board_read(url, &format!("/api/v1/boards/path/{task}"))
}

pub fn board_search(
    url: &str,
    terms: &[String],
    history: bool,
    within: Option<&str>,
) -> Result<crate::serve::board_wire::SearchReply, ClientFail> {
    board_post(
        url,
        "/api/v1/boards/search",
        &crate::serve::board_wire::SearchQuery {
            terms: terms.to_vec(),
            history,
            within: within.map(str::to_string),
        },
    )
}
