use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;

use crate::attempts::AsReceived;
use crate::db::{self, ExecuteFail, StoredRecord};
use crate::store::{Context, RecordId, Tier, World};
use crate::{
    Reject,
    record::{events::Command, objects::message::MessageId},
};

use super::fail::{ApiFail, api_failure, execute_failure_response};
use super::message_wire::{
    MailboxEndpoint, MailboxMessage, MailboxReply, MessageFacts, MessageSendRequest,
    SendMessageReply, SubmissionFacts,
};
use super::state::{AppState, ReadFail, ServerState, Snapshot};
use super::wire::WireTier;

// -- the send path at the sole writer -------------------------------------

/// One send's submission attempt at the sole writer, after its record
/// landed: the adapter's bounded submit to the recipient's recorded
/// address. Submitted means the host accepted the ordered submission
/// and the outcome record landed; Unrecorded means the host accepted
/// while the outcome record could not land (the message reads pending
/// though the submission is real); Refused means nothing reached the
/// session and the message stays pending - evidence of the attempt,
/// never a promise.
pub(super) enum SubmitOutcome {
    Submitted,
    Unrecorded,
    Refused { code: &'static str, detail: String },
}

impl SubmitOutcome {
    /// The reply face's summary: the state plus, on refusal, the
    /// evidence.
    pub(super) fn facts(&self) -> SubmissionFacts {
        match self {
            SubmitOutcome::Submitted => SubmissionFacts {
                state: "submitted".into(),
                code: None,
                detail: None,
            },
            SubmitOutcome::Unrecorded => SubmissionFacts {
                state: "unrecorded".into(),
                code: None,
                detail: Some(
                    "the host accepted the submission but its outcome record could not land; the message reads pending"
                        .into(),
                ),
            },
            SubmitOutcome::Refused { code, detail } => SubmissionFacts {
                state: "refused".into(),
                code: Some((*code).to_string()),
                detail: Some(detail.clone()),
            },
        }
    }
}

/// The whole send path every face shares: serialize on the
/// recipient's host address, record the message, submit through the
/// adapter, then record the outcome. The per-recipient lock covers
/// record and submit together, so concurrent sends to one recipient
/// submit in record order; the adapter's own bound contains a hung
/// core, and the lock order is always address-then-writer, never the
/// reverse, so the server cannot deadlock. Runs on a blocking thread
/// (the launch door's pattern): the adapter ask is a subprocess call.
/// The wire envelope a submission carries: machine-supplied origin on
/// one ASCII line - the dense message number, the sender's incarnation
/// number, and the reply and task segments only when they exist -
/// then a blank line and the body untouched. Wire-only: the record's
/// body and every read stay as authored; the envelope exists at the
/// submit door alone, so a recipient knows who is talking and holds
/// the id to reply to without a prior query.
fn wire_envelope(world: &World, id: MessageId) -> Option<String> {
    let mail = &world.messages.get(&id)?.message;
    let num = world.message_num(id)?;
    let from = world.incarnation_num(mail.from)?;
    let mut head = format!("[saccade message m-{num} | from i-{from}");
    if let Some(reply) = mail.reply_to {
        head += &format!(" | replying to m-{}", world.message_num(reply)?);
    }
    if let Some(task) = mail.task {
        head += &format!(" | task t-{}", task.0);
    }
    head.push(']');
    Some(format!("{head}\n\n{}", mail.body.as_str()))
}

pub(super) fn send_operation(
    app: &AppState,
    door: &str,
    context: &Context,
    command: Command,
    received: AsReceived,
) -> Result<(Vec<StoredRecord>, SubmitOutcome), ExecuteFail> {
    let Command::SendMessage { to, .. } = &command else {
        unreachable!("the send operation serves SendMessage alone");
    };
    let to = *to;
    // the recipient's id names the lock; the address is resolved
    // fresh under it, so a move between record and submit submits at
    // the new association without surrendering serialization
    let addressed = |world: &World| {
        world
            .incarnations
            .get(&to)
            .and_then(|ctx| ctx.incarnation.address().map(str::to_string))
    };
    let lock = app.submit_locks.acquire(to);
    // the guard, not the Arc: the serialization is the lock itself,
    // held across the record and the submit
    let _serialized = lock.lock().expect("the submit lock holds");
    let stored = app.execute(door, context, command, None, received)?;
    let sent = stored.first().expect("a message lands its one record").seq;
    // the freshest facts under the lock: the snapshot that resolves
    // the address holds the just-landed message, so its numbers are
    // fold-guaranteed
    let snapshot = app.snapshot().ok();
    // gone for delivery, first face: the fold knows the recipient was
    // superseded - its recorded address belongs to a newer incarnation
    // now. The refusal discloses the supersession's evidence and never
    // asks the host: the host would answer live and the delivery would
    // land in the successor's slot
    if let Some(supersession) = snapshot.as_ref().and_then(|s| s.world.superseded.get(&to)) {
        return Ok((
            stored,
            SubmitOutcome::Refused {
                code: "recipient_gone",
                detail: format!(
                    "the recipient is superseded: incarnation i-{} holds its address {} since record seq {}; delivery never lands in another session's slot",
                    supersession.by.0.0, supersession.address, supersession.at.0
                ),
            },
        ));
    }
    // an unaddressed recipient is the pre-B shape - the record stands
    // as evidence, nothing submits
    let Some(address) = snapshot.as_ref().and_then(|s| addressed(&s.world)) else {
        return Ok((
            stored,
            SubmitOutcome::Refused {
                code: "recipient_unaddressed",
                detail: "the recipient carries no host address: pre-B sessions lose inbound \
                         delivery; the sanctioned migration is exit + resume"
                    .into(),
            },
        ));
    };
    // the envelope composes from the recorded facts at the submit
    // door; the snapshot that resolved the address holds the
    // just-landed message, so its numbers are fold-guaranteed
    let message_id = MessageId(RecordId(sent));
    let wire_text = snapshot
        .as_ref()
        .and_then(|s| wire_envelope(&s.world, message_id))
        .expect("the just-landed message composes its wire envelope");
    use crate::host_contract::HostContract;
    let submission = match crate::herdr::HerdrCore::resolved() {
        Ok(host) => host.submit(&crate::host_contract::HostAddress(address), &wire_text),
        Err(e) => {
            return Ok((
                stored,
                SubmitOutcome::Refused {
                    code: "host_unavailable",
                    detail: e.to_string(),
                },
            ));
        }
    };
    match submission {
        Ok(()) => {
            // the addressee's machinery act: the outcome record is the
            // recipient's, never the sender's
            let outcome_record = Command::SubmitMessage {
                id: crate::record::objects::message::MessageId(RecordId(sent)),
                by: to,
            };
            let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
            let recorded = match &mut *guard {
                ServerState::Ready(inner) => db::record(
                    &mut inner.conn,
                    &Context::system(),
                    outcome_record,
                    db::now_epoch(),
                )
                .is_ok(),
                ServerState::Degraded(..) => false,
            };
            drop(guard);
            if recorded {
                app.wake_landing();
            }
            Ok((
                stored,
                if recorded {
                    SubmitOutcome::Submitted
                } else {
                    SubmitOutcome::Unrecorded
                },
            ))
        }
        Err(refusal) => Ok((
            stored,
            SubmitOutcome::Refused {
                code: match &refusal {
                    crate::host_contract::SubmitRefusal::AddressNotLive(_) => "recipient_not_live",
                    crate::host_contract::SubmitRefusal::HostRefused(_) => "submission_refused",
                    crate::host_contract::SubmitRefusal::Unknown(_) => "submission_unknown",
                    crate::host_contract::SubmitRefusal::Unresolved => "submission_unresolved",
                },
                detail: refusal.to_string(),
            },
        )),
    }
}

/// The message facts both send faces reply with: the birth position,
/// the dense m-N, and the sender's resolved identity.
pub(super) fn message_facts(
    app: &AppState,
    stored: &[StoredRecord],
) -> (MessageFacts, MessageFacts) {
    let sent = stored.first().expect("a message lands its one record").seq;
    let numbers = app
        .snapshot()
        .expect("a landed write leaves a foldable world")
        .world;
    let mctx = numbers
        .messages
        .get(&crate::record::objects::message::MessageId(RecordId(sent)))
        .expect("the landed message folds in");
    let message = MessageFacts {
        seq: sent,
        num: numbers.message_num(crate::record::objects::message::MessageId(RecordId(sent))),
    };
    let from = MessageFacts {
        seq: mctx.message.from.0.0,
        num: numbers.incarnation_num(mctx.message.from),
    };
    (message, from)
}
pub(super) async fn message_send(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: MessageSendRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    // the sender is the sending session's registered identity, read
    // from the declared scope - never caller-authored prose
    let Some(from) = request.context.scope.as_ref().and_then(|s| s.incarnation) else {
        return api_failure(ApiFail::Reject(
            Reject::InvalidIncarnationId,
            Some(
                "messages send as a registered incarnation: the sending session declares \
                 SACCADE_INCARNATION_SEQ"
                    .into(),
            ),
        ));
    };
    let context = Context {
        actor: request.context.actor,
        tier: match request.context.tier {
            WireTier::Human => Tier::Human,
            WireTier::Agent => Tier::Agent,
        },
        scope: request.context.scope,
    };
    // the public numbers resolve against the fold before the write:
    // a send's reference guards teach before anything lands
    let resolved = (|| {
        let snapshot = app.snapshot().map_err(|read_fail| match read_fail {
            ReadFail::Degraded(d) => ApiFail::Degraded(d.reason),
            ReadFail::Db(reason) => ApiFail::Db(reason),
        })?;
        let to = snapshot
            .world
            .incarnation_at_num(request.to)
            .ok_or(ApiFail::Reject(Reject::InvalidIncarnationId, None))?;
        let reply_to = match request.reply_to {
            Some(n) => Some(
                snapshot
                    .world
                    .message_at_num(n)
                    .ok_or(ApiFail::Reject(Reject::InvalidMessageId, None))?,
            ),
            None => None,
        };
        Ok((to, reply_to))
    })();
    let (to_id, reply_to) = match resolved {
        Ok(resolved) => resolved,
        Err(fail) => return api_failure(fail),
    };
    let command = Command::SendMessage {
        from,
        to: to_id,
        reply_to,
        task: request.task.map(crate::record::objects::task::TaskId),
        body: request.body,
    };
    // the whole send path runs off the async runtime: the adapter ask
    // is a bounded subprocess call, serialized per recipient address
    let send_app = app.clone();
    let send_context = context.clone();
    let send_command = command.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        send_operation(
            &send_app,
            "messages/send",
            &send_context,
            send_command,
            received,
        )
    })
    .await
    .expect("the send operation does not panic");
    match outcome {
        Ok((stored, submission)) => {
            let (message, from) = message_facts(&app, &stored);
            match submission {
                SubmitOutcome::Submitted | SubmitOutcome::Unrecorded => {
                    let reply = SendMessageReply {
                        records: stored,
                        message,
                        from,
                        submission: Some(submission.facts()),
                    };
                    (
                        StatusCode::OK,
                        Json(serde_json::to_value(&reply).expect("the reply is plain data")),
                    )
                        .into_response()
                }
                SubmitOutcome::Refused { code, detail } => (
                    StatusCode::CONFLICT,
                    Json(json!({
                        "error": {"code": code, "detail": detail},
                        "message": message,
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => execute_failure_response(&app, e, &command),
    }
}

#[derive(Deserialize)]
pub(super) struct MailboxQuery {
    /// The reader's own incarnation: a bare birth-record position, the
    /// machine identity channel the session attaches with
    incarnation: usize,
}

/// The mailbox read: the reader's mail in birth order, any state,
/// both directions, with the log's cursor.
fn mailbox_view(
    snapshot: &Snapshot,
    id: crate::record::objects::incarnation::IncarnationId,
) -> MailboxReply {
    let world = &snapshot.world;
    let messages: Vec<MailboxMessage> = world
        .messages
        .iter()
        .filter(|(_, mctx)| mctx.message.to == id || mctx.message.from == id)
        .map(|(mid, mctx)| {
            let sender = &mctx.message.from;
            MailboxMessage {
                seq: mid.0.0,
                num: world.message_num(*mid),
                from: MailboxEndpoint {
                    seq: sender.0.0,
                    num: world.incarnation_num(*sender),
                    actor: Some(mctx.actor.as_str().to_string()),
                },
                to: MailboxEndpoint {
                    seq: mctx.message.to.0.0,
                    num: world.incarnation_num(mctx.message.to),
                    actor: None,
                },
                reply_to: mctx.message.reply_to.and_then(|r| world.message_num(r)),
                task: mctx.message.task.map(|t| t.0),
                state: match mctx.state {
                    crate::record::objects::message::MessageState::Pending => "pending",
                    crate::record::objects::message::MessageState::Submitted => "submitted",
                }
                .to_string(),
                body: mctx.message.body.as_str().to_string(),
                born_at: mctx.born_at,
            }
        })
        .collect();
    MailboxReply {
        messages,
        cursor: snapshot.rows.last().map(|r| r.seq),
    }
}

/// The descriptive history read: a reader's own mail, in birth order,
/// any state, read-only by construction. `pending` is a recorded
/// message without a confirmed submission - evidence of its attempt,
/// never a promise of later delivery - and `submitted` means the
/// recipient's native input queue accepted it.
pub(super) async fn messages_history(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<MailboxQuery>,
) -> Response {
    let snapshot = match app.snapshot() {
        Ok(s) => s,
        Err(read_fail) => {
            let (status, code, detail) = match read_fail {
                ReadFail::Degraded(d) => ApiFail::Degraded(d.reason).parts(),
                ReadFail::Db(reason) => ApiFail::Db(reason).parts(),
            };
            return (
                status,
                Json(json!({"error": {"code": code, "detail": detail}})),
            )
                .into_response();
        }
    };
    let id = crate::record::objects::incarnation::IncarnationId(RecordId(query.incarnation));
    if !snapshot.world.incarnations.contains_key(&id) {
        return api_failure(ApiFail::Reject(Reject::InvalidIncarnationId, None));
    }
    (
        StatusCode::OK,
        Json(
            serde_json::to_value(mailbox_view(&snapshot, id))
                .expect("the mailbox view is plain data"),
        ),
    )
        .into_response()
}
