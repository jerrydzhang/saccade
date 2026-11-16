use std::time::Instant;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::Command;
use crate::attempts::{AsReceived, Outcome};
use crate::db::{self, StoredRecord};
use crate::store::{Context, RecordId, Tier};

use super::door::{integration_outcome, ready};
use super::fail::{ApiFail, api_failure, execute_failure_response};
use super::message::{message_facts, send_operation};
use super::state::AppState;
use super::wire::{Envelope, WireTier};

pub async fn command(State(app): State<AppState>, body: Bytes) -> Response {
    let envelope: Envelope = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => {
            // a refusal that enters no journal: the line is the request's
            // only surviving shape, with no actor to name
            let cursor = app
                .snapshot()
                .ok()
                .and_then(|s| s.rows.last().map(|r| r.seq));
            app.attempts.append(
                "command",
                None,
                None,
                0,
                Outcome::Refused {
                    code: "malformed_request",
                    cursor,
                    request: &String::from_utf8_lossy(&body),
                },
            );
            let (status, code, detail) = ApiFail::Malformed(e.to_string()).parts();
            return (
                status,
                Json(json!({"error": {"code": code, "detail": detail}})),
            )
                .into_response();
        }
    };
    let context = Context {
        actor: envelope.context.actor,
        tier: match envelope.context.tier {
            WireTier::Human => Tier::Human,
            WireTier::Agent => Tier::Agent,
        },
        scope: envelope.context.scope,
    };
    let command = envelope.command;
    let received = AsReceived {
        client: envelope.client,
        raw: String::from_utf8_lossy(&body).into_owned(),
    };
    // a send command rides the send operation like the send face: the
    if let Command::AcceptBoard { task } = &command {
        let accept_task = *task;
        let accept_context = context.clone();
        let accept_command = command.clone();
        let started = Instant::now();
        let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
        let outcome: Result<Vec<StoredRecord>, ApiFail> = (|| {
            let inner = ready(&mut guard)?;
            db::record(
                &mut inner.conn,
                &accept_context,
                accept_command,
                envelope.at.unwrap_or_else(db::now_epoch),
            )
            .map(|(stored, _)| stored)
            .map_err(ApiFail::from)
        })();
        integration_outcome(&app, &guard, "command", None, &received, started, &outcome);
        drop(guard);
        if outcome.is_ok() {
            app.wake_landing();
        }
        return match outcome {
            Ok(stored) => {
                let mut reply = json!({"records": &stored});
                if let Ok(snapshot) = app.snapshot()
                    && let Some(ctx) = snapshot.world.tasks.get(accept_task.0)
                {
                    reply["revision"] = json!(ctx.commits.len());
                }
                (StatusCode::OK, Json(reply)).into_response()
            }
            Err(e) => api_failure(e),
        };
    }
    // per-recipient lock must cover its record, and the attempt's truth
    // rides the reply as data
    if let Command::SendMessage { .. } = &command {
        let send_app = app.clone();
        let send_context = context.clone();
        let send_command = command.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            send_operation(&send_app, "command", &send_context, send_command, received)
        })
        .await
        .expect("the send operation does not panic");
        return match outcome {
            Ok((stored, submission)) => {
                let (message, from) = message_facts(&app, &stored);
                let mut reply = json!({"records": &stored});
                reply["message"] = json!(message);
                reply["from"] = json!(from);
                reply["submission"] = json!(submission.facts());
                (StatusCode::OK, Json(reply)).into_response()
            }
            Err(e) => execute_failure_response(&app, e, &command),
        };
    }
    match app.execute("command", &context, command.clone(), envelope.at, received) {
        Ok(stored) => {
            let mut reply = json!({"records": &stored});
            // a create reply names what was born: the id resolved from
            // the post-write fold, matched by the birth record it landed
            if matches!(command, Command::CreateTask { .. }) {
                let birth = stored
                    .iter()
                    .find(|r| r.kind == "task_created")
                    .expect("a create lands its birth record");
                let born = app
                    .snapshot()
                    .expect("a landed write leaves a foldable world")
                    .world
                    .task_born_at(RecordId(birth.seq))
                    .expect("the birth record folds into its task");
                reply["id"] = json!(format!("t-{}", born.0));
            }
            // a board decision names its revision: the dense rK the
            // fold derives from the commit list
            if let Command::CommitBoard { task, .. } | Command::OfferBoard { task, .. } = &command
                && let Ok(snapshot) = app.snapshot()
                && let Some(ctx) = snapshot.world.tasks.get(task.0)
            {
                reply["revision"] = json!(ctx.commits.len());
            }
            (StatusCode::OK, Json(reply)).into_response()
        }
        Err(e) => execute_failure_response(&app, e, &command),
    }
}
