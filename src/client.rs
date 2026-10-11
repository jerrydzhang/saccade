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
}

impl ClientFail {
    pub fn code(&self) -> &'static str {
        match self {
            ClientFail::ServerUnreachable { .. } => "server_required",
            ClientFail::Refused { .. } => "server_refused",
        }
    }
}

impl std::fmt::Display for ClientFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientFail::ServerUnreachable { url } => write!(
                f,
                "no server answered at {url}: start one (`sac serve`), \
                 or pass --offline to write directly to the db"
            ),
            ClientFail::Refused { code, detail } => write!(f, "{code}: {detail}"),
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
    #[serde(default)]
    seq: Option<usize>,
    #[serde(default)]
    task: Option<usize>,
    #[serde(default)]
    workspace: Option<WorkspaceFacts>,
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
    pub seq: Option<usize>,
    pub task: Option<usize>,
    pub workspace: Option<WorkspaceFacts>,
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
            seq: body.seq,
            task: body.task,
            workspace: body.workspace,
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

/// The handshake's client half: warn once when this binary and the
/// server disagree. A fetch that fails stays quiet — the send that
/// follows reports an unreachable server without our duplication.
pub fn handshake(url: &str) {
    let Ok(mut response) = ureq::get(&format!("{url}/api/v1/version"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
    else {
        return;
    };
    let Ok(text) = response.body_mut().read_to_string() else {
        return;
    };
    let Ok(body) = serde_json::from_str::<serde_json::Value>(&text) else {
        return;
    };
    let mine = env!("CARGO_PKG_VERSION");
    if body["version"]
        .as_str()
        .is_some_and(|theirs| theirs != mine)
    {
        eprintln!(
            "warning: server runs sac {theirs}, this binary is sac {mine}; a mismatch may refuse",
            theirs = body["version"].as_str().unwrap_or_default()
        );
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
            detail: serde_json::Value::String(text.into()),
        },
    });
    ClientFail::Refused {
        code: body.error.code,
        detail: body.error.detail.to_string(),
    }
}

/// The start operation: first sight establishes the workspace, records
/// the caller's request, and assigns the incarnation ownership; a task
/// already holding a workspace is ensured and reported, nothing landed.
pub fn start_workspace(
    url: &str,
    context: &Context,
    task: usize,
    base: Option<&str>,
    request: &crate::Prose,
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
            "request": request.as_str(),
            "client": env!("CARGO_PKG_VERSION"),
        }),
    )
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
