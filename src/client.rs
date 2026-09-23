//! The client half of the API: one `POST /api/v1/command`, the shared
//! envelope, and the failure taxonomy the CLI renders.

use crate::Command;
use crate::api::{Envelope, WireContext, WireTier};
use crate::db::StoredRecord;
use crate::store::{Context, Tier};
use serde::Deserialize;

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
