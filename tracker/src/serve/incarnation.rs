use axum::body::Bytes;
use axum::extract::State;
use axum::response::Response;

use crate::Command;
use crate::db::{self, StoredRecord};
use crate::store::{Context, RecordId, World};

use super::door::{door, ready};
use super::fail::{ApiFail, api_failure};
use super::incarnation_wire::{
    AttachReply, AttachRequest, IncarnationAddressReply, IncarnationAddressRequest,
    IncarnationRegisterReply, IncarnationRegisterRequest, PresentedTerminal,
};
use super::state::{AppState, Inner, ServerState};

// -- the native integration's doors -------------------------------------

/// Move the recorded (address, adopted) pair to what the attaching
/// process presents, through the association's own command; an equal
/// pair is silent. The emitted event is the existing
/// HostAddressAssigned - reassignment happens here, at attach
/// evidence, never at launch submission.
fn present_terminal(
    inner: &mut Inner,
    world: &World,
    incarnation: crate::IncarnationId,
    terminal: &PresentedTerminal,
) -> Result<Vec<StoredRecord>, ApiFail> {
    let ctx = world
        .incarnations
        .get(&incarnation)
        .ok_or(ApiFail::Reject(crate::Reject::InvalidIncarnationId, None))?;
    let moved = ctx.incarnation.address() != Some(terminal.address.as_str())
        || ctx.incarnation.adopted() != terminal.adopted;
    if !moved {
        return Ok(Vec::new());
    }
    let (stored, _) = db::record(
        &mut inner.conn,
        &Context::system(),
        Command::AssignHostAddress {
            incarnation,
            address: Some(terminal.address.clone()),
            adopted: terminal.adopted,
        },
        db::now_epoch(),
    )?;
    Ok(stored)
}

fn attach_operation(
    guard: &mut ServerState,
    native_id: &crate::NativeSessionId,
    saved_session: &crate::SessionLocator,
    harness: crate::Harness,
    terminal: Option<&PresentedTerminal>,
) -> Result<AttachReply, ApiFail> {
    let inner = ready(guard)?;
    let loadout = db::load(&inner.conn).map_err(|e| ApiFail::Db(e.to_string()))?;
    let world = match loadout.state {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    // an existing exact native session is the same incarnation: the
    // recorded association is harness AND native id AND saved locator;
    // a reused id under another harness, or with a different file,
    // binds other history, it never resolves
    let existing = world.incarnations.iter().find(|(_, ctx)| {
        ctx.incarnation.native_id() == Some(native_id) && ctx.incarnation.harness() == Some(harness)
    });
    if let Some((id, ctx)) = existing {
        match ctx.incarnation.saved_session() {
            // a launch-door birth precedes pi's first save: the attach
            // carries the locator and binds it, completing the birth
            None => {
                let command = Command::BindSavedSession {
                    incarnation: *id,
                    saved_session: saved_session.clone(),
                };
                let (bound, _) = db::record(
                    &mut inner.conn,
                    &Context::system(),
                    command,
                    db::now_epoch(),
                )?;
                // the presented terminal rides the completed birth: a
                // fresh birth already carries the door's address, so
                // this is silent for the ordinary launch; the record
                // exists for attach-born sessions presenting a pair
                let mut records = bound;
                if let Some(presented) = terminal {
                    records.extend(present_terminal(inner, &world, *id, presented)?);
                }
                return Ok(AttachReply {
                    registered: false,
                    incarnation: id.0.0,
                    num: world
                        .incarnation_num(*id)
                        .expect("a resident incarnation holds a number"),
                    records,
                });
            }
            Some(bound) if bound == saved_session => {}
            Some(_) => {
                return Err(ApiFail::Reject(
                    crate::Reject::SavedSessionMismatch,
                    Some(format!(
                        "native session {} is already registered as i-{} from {}; attach carries {}",
                        native_id.as_str(),
                        world
                            .incarnation_num(*id)
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| id.0.0.to_string()),
                        ctx.incarnation
                            .saved_session()
                            .map(|l| l.as_path().display().to_string())
                            .unwrap_or_default(),
                        saved_session.as_path().display(),
                    )),
                ));
            }
        }
        // the association's reassignment lands here, on exact attach
        // evidence, never at launch submission: a presented pair that
        // differs from the record moves it, an equal pair stays silent
        let records = match terminal {
            Some(presented) => present_terminal(inner, &world, *id, presented)?,
            None => Vec::new(),
        };
        return Ok(AttachReply {
            registered: false,
            incarnation: id.0.0,
            num: world
                .incarnation_num(*id)
                .expect("a resident incarnation holds a number"),
            records,
        });
    }
    // a self-registered birth carries the presented pair as its
    // association: the same attach evidence the resolve arm trusts
    let (address, adopted) = match terminal {
        Some(presented) => (Some(presented.address.clone()), presented.adopted),
        None => (None, false),
    };
    let (stored, _) = db::record(
        &mut inner.conn,
        &Context::system(),
        Command::RegisterIncarnation {
            native_id: native_id.clone(),
            saved_session: Some(saved_session.clone()),
            address,
            adopted,
            harness,
        },
        db::now_epoch(),
    )?;
    let birth = RecordId(stored[0].seq);
    let reloaded = db::load(&inner.conn).map_err(|e| ApiFail::Db(e.to_string()))?;
    let db::LoadState::Full(world) = reloaded.state else {
        return Err(ApiFail::Degraded("the log will not fold".into()));
    };
    Ok(AttachReply {
        registered: true,
        incarnation: birth.0,
        num: world
            .incarnation_num(crate::IncarnationId(birth))
            .expect("the landed registration folds in"),
        records: stored,
    })
}

pub(super) async fn incarnation_register(State(app): State<AppState>, body: Bytes) -> Response {
    door(
        &app,
        body,
        "incarnations/register",
        |r: &IncarnationRegisterRequest| r.client.clone(),
        |_app, guard, request| {
            let inner = ready(guard)?;
            let (records, incarnation, registered) = incarnation_register_operation(
                inner,
                request.native_id,
                request.address,
                request.adopted,
                request.harness,
                request.name,
            )?;
            Ok(IncarnationRegisterReply {
                incarnation,
                records,
                registered,
            })
        },
    )
}

/// Recovery is the attach-born unaddressed case only: an addressed
/// session keeps the fold's registration refusal - address moves
/// belong to the relaunch door. A launch's name lands right after,
/// on whichever incarnation the registration resolved.
fn incarnation_register_operation(
    inner: &mut Inner,
    native_id: crate::NativeSessionId,
    address: String,
    adopted: bool,
    harness: crate::Harness,
    name: Option<crate::IncarnationName>,
) -> Result<(Vec<StoredRecord>, usize, bool), ApiFail> {
    let (mut stored, incarnation, registered) =
        register_or_recover(inner, native_id, address, adopted, harness)?;
    if let Some(name) = name {
        let (named, _) = db::record(
            &mut inner.conn,
            &Context::system(),
            Command::NameIncarnation {
                incarnation: crate::IncarnationId(RecordId(incarnation)),
                name: Some(name),
            },
            db::now_epoch(),
        )?;
        stored.extend(named);
    }
    Ok((stored, incarnation, registered))
}

fn register_or_recover(
    inner: &mut Inner,
    native_id: crate::NativeSessionId,
    address: String,
    adopted: bool,
    harness: crate::Harness,
) -> Result<(Vec<StoredRecord>, usize, bool), ApiFail> {
    let loadout = db::load(&inner.conn).map_err(|e| ApiFail::Db(e.to_string()))?;
    let db::LoadState::Full(world) = loadout.state else {
        return Err(ApiFail::Degraded("the log will not fold".into()));
    };
    let known = world.incarnations.iter().find(|(_, ctx)| {
        ctx.incarnation.native_id() == Some(&native_id)
            && ctx.incarnation.harness() == Some(harness)
    });
    match known {
        Some((id, ctx))
            if ctx.incarnation.saved_session().is_some() && ctx.incarnation.address().is_none() =>
        {
            let command = Command::AssignHostAddress {
                incarnation: *id,
                address: Some(address),
                adopted,
            };
            let (stored, _) = db::record(
                &mut inner.conn,
                &Context::system(),
                command,
                db::now_epoch(),
            )?;
            Ok((stored, id.0.0, false))
        }
        _ => {
            let command = Command::RegisterIncarnation {
                native_id,
                saved_session: None,
                address: Some(address),
                adopted,
                harness,
            };
            let (stored, _) = db::record(
                &mut inner.conn,
                &Context::system(),
                command,
                db::now_epoch(),
            )?;
            let birth = stored[0].seq;
            Ok((stored, birth, true))
        }
    }
}

pub(super) async fn incarnation_address(State(app): State<AppState>, body: Bytes) -> Response {
    door(
        &app,
        body,
        "incarnations/address",
        |r: &IncarnationAddressRequest| r.client.clone(),
        |_app, guard, request| {
            let inner = ready(guard)?;
            let command = Command::AssignHostAddress {
                incarnation: crate::IncarnationId(RecordId(request.incarnation)),
                address: Some(request.address),
                adopted: request.adopted,
            };
            let (records, _) = db::record(
                &mut inner.conn,
                &Context::system(),
                command,
                db::now_epoch(),
            )
            .map_err(ApiFail::from)?;
            Ok(IncarnationAddressReply {
                assigned: true,
                records,
            })
        },
    )
}

pub(super) async fn native_attach(State(app): State<AppState>, body: Bytes) -> Response {
    // the presented terminal is the request's shape, not a judged
    // write: an empty address refuses before the attempt, unlogged
    // like the parse refusal, so the check rides ahead of the door
    if let Ok(request) = serde_json::from_slice::<AttachRequest>(&body)
        && let Some(presented) = &request.terminal
        && presented.address.is_empty()
    {
        return api_failure(ApiFail::Malformed(
            "the presented terminal address is empty".into(),
        ));
    }
    door(
        &app,
        body,
        "native/attach",
        |r: &AttachRequest| r.client.clone(),
        |_app, guard, request| {
            attach_operation(
                guard,
                &request.native_id,
                &request.session_file,
                request.harness,
                request.terminal.as_ref(),
            )
        },
    )
}
