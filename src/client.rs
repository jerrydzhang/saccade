//! The client half of the API: one `POST /api/v1/command`, the shared
//! envelope, and the failure taxonomy the CLI renders.

use crate::Command;
use crate::api::{Envelope, WireContext, WireTier};
use crate::db::StoredRecord;
use crate::store::{Context, Tier};
use crate::views::WorkspaceFacts;
use serde::{Deserialize, Serialize};

/// One workspace operation's outcome, the wire's shape.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkspaceReply {
    pub established: bool,
    pub recreated: bool,
    pub workspace: WorkspaceFacts,
    pub incarnation: usize,
    pub records: Vec<StoredRecord>,
}

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
}

impl ClientFail {
    pub fn code(&self) -> &'static str {
        match self {
            ClientFail::ServerUnreachable { .. } => "server_required",
            ClientFail::Refused { .. } => "server_refused",
            ClientFail::Undiscovered { .. } => "server_undiscovered",
            ClientFail::Mismatch { .. } => "server_mismatch",
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

pub fn identity(url: &str) -> Result<ServerIdentity, ClientFail> {
    let mut response = ureq::get(&format!("{url}/api/v1/version"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|_| ClientFail::ServerUnreachable {
            url: url.to_string(),
        })?;
    let body: serde_json::Value = serde_json::from_str(&text).map_err(|e| ClientFail::Refused {
        code: "malformed_response".into(),
        detail: e.to_string(),
    })?;
    Ok(ServerIdentity {
        version: body["version"].as_str().unwrap_or_default().to_string(),
        db: body["db"].as_str().map(str::to_string),
    })
}

/// The hint beside the expected database, verified to serve it —
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
/// server disagree. A fetch that fails stays quiet — the send that
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
    format!("{}…", &flat[..cut])
}

/// The start operation: first sight establishes the workspace; a task
/// already holding a workspace is ensured and reported, nothing
/// landed. No request, comment, or ownership rides it — the prompt is
/// native input at launch, never a publication.
pub fn start_workspace(
    url: &str,
    context: &Context,
    task: usize,
    base: Option<&str>,
) -> Result<WorkspaceReply, ClientFail> {
    post_json(
        url,
        "/api/v1/workspace/start",
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
            "task": task,
            "base": base,
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
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
/// its incarnation, a fresh one registers — the machinery's own act.
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

/// The open operation: ensure the recorded environment exists (a
/// missing checkout is recreated from its branch tip), launching
/// nothing, landing nothing on a recorded workspace; an unestablished
/// task births the establishment alone.
pub fn open_workspace(url: &str, task: usize) -> Result<WorkspaceReply, ClientFail> {
    post_json(
        url,
        "/api/v1/workspace/open",
        serde_json::json!({ "task": task, "client": env!("CARGO_PKG_VERSION") }),
    )
}
