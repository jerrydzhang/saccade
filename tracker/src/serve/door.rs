use crate::attempts::{AsReceived, Outcome};
use crate::db;

use super::fail::ApiFail;
use super::state::{AppState, ServerState};

/// One native integration attempt's outcome, logged like every
/// workspace operation: the attempts log is the diagnoser's memory.
pub(super) fn integration_outcome<T>(
    app: &AppState,
    guard: &ServerState,
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
