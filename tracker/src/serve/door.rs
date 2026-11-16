use std::time::Instant;

use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::attempts::{AsReceived, Outcome};
use crate::db;

use super::board::board_body;
use super::fail::{ApiFail, api_failure};
use super::state::{AppState, Inner, ServerState};

/// The one Ready/Degraded spelling behind the writer doors: a
/// degraded log refuses the write, it never guesses past it.
pub(super) fn ready(guard: &mut ServerState) -> Result<&mut Inner, ApiFail> {
    match guard {
        ServerState::Ready(inner) => Ok(inner),
        ServerState::Degraded(reason, _) => Err(ApiFail::Degraded(reason.clone())),
    }
}

/// The frame the write doors share: parse the body or refuse
/// Malformed, read the caller's client off the parsed request, start
/// the clock, take the writer lock, run the operation, log the
/// attempt, fire the landing wake exactly when the write landed, and
/// answer 200 with the reply or the failure. `name` is the door's
/// name, and the attempts line carries it in its `door` field.
pub(super) fn door<Request, Reply, Client, Operation>(
    app: &AppState,
    body: Bytes,
    name: &'static str,
    client_of: Client,
    operation: Operation,
) -> Response
where
    Request: serde::de::DeserializeOwned,
    Reply: serde::Serialize,
    Client: Fn(&Request) -> Option<String>,
    Operation: FnOnce(&AppState, &mut ServerState, Request) -> Result<Reply, ApiFail>,
{
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: Request = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: client_of(&request),
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = operation(app, &mut guard, request);
    integration_outcome(app, &guard, name, None, &received, started, &outcome);
    if outcome.is_ok() {
        app.wake_landing();
    }
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

/// The board doors' variant: the body arrives under the route's size
/// limit, and the limit's rejection is the error envelope before the
/// frame begins.
pub(super) fn door_limited<Request, Reply, Client, Operation>(
    app: &AppState,
    body: Result<Bytes, BytesRejection>,
    name: &'static str,
    client_of: Client,
    operation: Operation,
) -> Response
where
    Request: serde::de::DeserializeOwned,
    Reply: serde::Serialize,
    Client: Fn(&Request) -> Option<String>,
    Operation: FnOnce(&AppState, &mut ServerState, Request) -> Result<Reply, ApiFail>,
{
    match board_body(body) {
        Ok(body) => door(app, body, name, client_of, operation),
        Err(fail) => api_failure(fail),
    }
}

/// One native integration attempt's outcome, logged like every
/// workspace operation: the attempts log is the diagnoser's memory.
pub(super) fn integration_outcome<T>(
    app: &AppState,
    guard: &ServerState,
    door: &str,
    actor: Option<String>,
    received: &AsReceived,
    started: std::time::Instant,
    outcome: &Result<T, ApiFail>,
) {
    let duration_ms = started.elapsed().as_millis() as u64;
    match outcome {
        Ok(_) => {
            let cursor = match guard {
                ServerState::Ready(inner) => db::last_seq(&inner.conn).ok().flatten(),
                ServerState::Degraded(..) => None,
            };
            app.attempts.append(
                door,
                actor.as_deref(),
                received.client.as_deref(),
                duration_ms,
                Outcome::Landed {
                    seqs: &cursor.map(|c| vec![c]).unwrap_or_default(),
                },
            );
        }
        Err(fail) => {
            let code = match fail {
                ApiFail::Reject(reject, _) => reject.code(),
                ApiFail::Degraded(_) => "degraded",
                ApiFail::Db(_) => "storage",
                ApiFail::Malformed(_) => "malformed_request",
                ApiFail::Workspace(effect) => effect.code(),
                ApiFail::NoRepo => "workspace_no_repo",
                ApiFail::Host(host) => host.wire_code(),
                ApiFail::Refused(_, code, _) => code,
            };
            let cursor = match guard {
                ServerState::Ready(inner) => db::last_seq(&inner.conn).ok().flatten(),
                ServerState::Degraded(..) => None,
            };
            app.attempts.append(
                door,
                actor.as_deref(),
                received.client.as_deref(),
                duration_ms,
                Outcome::Refused {
                    code,
                    cursor,
                    request: &received.raw,
                },
            );
        }
    }
}
