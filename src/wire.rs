use crate::events::Event;
use crate::objects::comment::Target;
use crate::objects::proposal::{Proposal, ProposalAction, ProposalId, ProposalState};
use crate::objects::task::{TaskId, TaskState};
use crate::prose::Prose;
use crate::store::{Tier, World};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq)]
pub enum ParseFail {
    UnknownKind(String),
    UnknownTier(String),
    Malformed { kind: String, detail: String },
}

pub fn tier_of(tier: &Tier) -> &'static str {
    match tier {
        Tier::Human => "human",
        Tier::Agent => "agent",
    }
}

pub fn tier_from(s: &str) -> Result<Tier, ParseFail> {
    match s {
        "human" => Ok(Tier::Human),
        "agent" => Ok(Tier::Agent),
        _ => Err(ParseFail::UnknownTier(s.to_string())),
    }
}

#[derive(Serialize, Deserialize)]
struct CreatedPayload {
    id: usize,
    task_name: Prose,
    parent_id: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct IdPayload {
    id: usize,
}

#[derive(Serialize, Deserialize)]
struct DonePayload {
    id: usize,
    receipt: Prose,
}

#[derive(Serialize, Deserialize)]
struct CommentedPayload {
    target: Target,
    body: Prose,
}

#[derive(Serialize, Deserialize)]
struct NotedPayload {
    id: usize,
    note: Prose,
}

#[derive(Serialize, Deserialize)]
struct ProposalCreatedPayload {
    name: Prose,
    action: ProposalAction,
}

#[derive(Serialize, Deserialize)]
struct ProposalIdPayload {
    id: ProposalId,
}

#[derive(Serialize, Deserialize)]
struct ProposalNotedPayload {
    id: ProposalId,
    note: Prose,
}

/// Serializing an in-memory event cannot fail
fn pack<T: Serialize>(payload: &T) -> String {
    serde_json::to_string(payload).expect("in-memory events always serialize")
}

pub fn disassemble(event: &Event) -> (&'static str, String) {
    match event {
        Event::TaskCreated {
            id,
            name: task_name,
            parent_id,
        } => (
            "task_created",
            pack(&CreatedPayload {
                id: id.0,
                task_name: task_name.clone(),
                parent_id: parent_id.map(|p| p.0),
            }),
        ),
        Event::TaskClaimed { id } => ("task_claimed", pack(&IdPayload { id: id.0 })),
        Event::TaskDone { id, receipt } => (
            "task_done",
            pack(&DonePayload {
                id: id.0,
                receipt: receipt.clone(),
            }),
        ),
        Event::TaskDropped { id, note } => (
            "task_dropped",
            pack(&NotedPayload {
                id: id.0,
                note: note.clone(),
            }),
        ),
        Event::TaskReleased { id, note } => (
            "task_released",
            pack(&NotedPayload {
                id: id.0,
                note: note.clone(),
            }),
        ),
        Event::ProposalCreated { name, action } => (
            "proposal_created",
            pack(&ProposalCreatedPayload {
                name: name.clone(),
                action: *action,
            }),
        ),
        Event::ProposalAccepted { id } => {
            ("proposal_accepted", pack(&ProposalIdPayload { id: *id }))
        }
        Event::ProposalRejected { id, note } => (
            "proposal_rejected",
            pack(&ProposalNotedPayload {
                id: *id,
                note: note.clone(),
            }),
        ),
        Event::ProposalWithdrawn { id, note } => (
            "proposal_withdrawn",
            pack(&ProposalNotedPayload {
                id: *id,
                note: note.clone(),
            }),
        ),
        Event::Commented { target, body } => (
            "commented",
            pack(&CommentedPayload {
                target: *target,
                body: body.clone(),
            }),
        ),
    }
}

pub fn assemble(kind: &str, payload: &str) -> Result<Event, ParseFail> {
    fn malformed<T>(kind: &str, err: serde_json::Error) -> Result<T, ParseFail> {
        Err(ParseFail::Malformed {
            kind: kind.to_string(),
            detail: err.to_string(),
        })
    }

    match kind {
        "task_created" => {
            let p: CreatedPayload =
                serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::TaskCreated {
                id: TaskId(p.id),
                name: p.task_name,
                parent_id: p.parent_id.map(TaskId),
            })
        }
        "task_claimed" => {
            let p: IdPayload = serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::TaskClaimed { id: TaskId(p.id) })
        }
        "task_done" => {
            let p: DonePayload = serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::TaskDone {
                id: TaskId(p.id),
                receipt: p.receipt,
            })
        }
        "task_dropped" => {
            let p: NotedPayload = serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::TaskDropped {
                id: TaskId(p.id),
                note: p.note,
            })
        }
        "task_released" => {
            let p: NotedPayload = serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::TaskReleased {
                id: TaskId(p.id),
                note: p.note,
            })
        }
        "proposal_created" => {
            let p: ProposalCreatedPayload =
                serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::ProposalCreated {
                name: p.name,
                action: p.action,
            })
        }
        "proposal_accepted" => {
            let p: ProposalIdPayload =
                serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::ProposalAccepted { id: p.id })
        }
        "proposal_rejected" => {
            let p: ProposalNotedPayload =
                serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::ProposalRejected {
                id: p.id,
                note: p.note,
            })
        }
        "proposal_withdrawn" => {
            let p: ProposalNotedPayload =
                serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::ProposalWithdrawn {
                id: p.id,
                note: p.note,
            })
        }
        "commented" => {
            let p: CommentedPayload =
                serde_json::from_str(payload).or_else(|e| malformed(kind, e))?;
            Ok(Event::Commented {
                target: p.target,
                body: p.body,
            })
        }
        _ => Err(ParseFail::UnknownKind(kind.to_string())),
    }
}

pub fn state_of(state: &TaskState) -> &'static str {
    match state {
        TaskState::Open => "open",
        TaskState::Claimed => "claimed",
        TaskState::Done(_) => "done",
        TaskState::Dropped => "dropped",
    }
}

/// Display proposal type
pub struct ProposalView {
    pub id: usize,
    pub state: &'static str,
    pub action: &'static str,
    pub task: String,
    pub name: String,
}

pub fn view_of_proposal(id: &ProposalId, p: &Proposal, world: &World) -> ProposalView {
    ProposalView {
        id: id.0 .0,
        state: match &p.state {
            ProposalState::Open if is_stale(p, world) => "stale",
            ProposalState::Open => "open",
            ProposalState::Accepted => "accepted",
            ProposalState::Rejected(_) => "rejected",
            ProposalState::Withdrawn(_) => "withdrawn",
        },
        action: match &p.action {
            ProposalAction::Drop { .. } => "drop",
            ProposalAction::Release { .. } => "release",
        },
        task: match &p.action {
            ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => {
                format!("t-{}", task_id.0)
            }
        },
        name: p.name.as_str().to_string(),
    }
}

/// Derived staleness: would the embedded act be refused today?
/// The same probe decide uses at propose time — the quiet consumer to
/// accept's loud one. A missing target counts as stale: the act would
/// be refused.
fn is_stale(p: &Proposal, world: &World) -> bool {
    let task_id = match p.action {
        ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => task_id,
    };
    world
        .tasks
        .get(task_id.0)
        .map(|t| {
            t.state
                .validate(
                    &p.action
                        .target_event(&Prose::new("probe".into()).expect("probe is non-empty")),
                )
                .is_err()
        })
        .unwrap_or(true)
}

/// Display task type
pub struct CommentLine {
    pub seq: usize,
    pub depth: usize,
    pub actor: String,
    pub body: String,
}

pub struct ShowView {
    pub id: String,
    pub state: &'static str,
    pub parent: Option<String>,
    pub name: String,
    pub receipt: Option<String>,
}

pub fn show_of(task_id: &TaskId, world: &World) -> Option<ShowView> {
    let task = world.tasks.get(task_id.0)?;
    let receipt = match &task.state {
        TaskState::Done(r) => Some(r.as_str().to_string()),
        _ => None,
    };
    Some(ShowView {
        id: format!("t-{}", task_id.0),
        state: state_of(&task.state),
        parent: task.parent_id.map(|p| format!("t-{}", p.0)),
        name: task.name.as_str().to_string(),
        receipt,
    })
}


/// The thread as a projection: the task is the root, so a comment
/// addressing it sits at depth 1. A walk over the comment pointers.
pub fn comment_thread(world: &World, task_id: &TaskId) -> Vec<CommentLine> {
    let mut lines = Vec::new();
    for (id, comment) in &world.comments {
        let mut depth = 1;
        let mut up = comment.target;
        loop {
            match up {
                Target::Task(t) => {
                    if t != *task_id {
                        break;
                    }
                    lines.push(CommentLine {
                        seq: id.0 .0,
                        depth,
                        actor: comment.actor.clone(),
                        body: comment.body.as_str().to_string(),
                    });
                    break;
                }
                Target::Comment(addressed) => match world.comments.get(&addressed) {
                    Some(parent) => {
                        depth += 1;
                        up = parent.target;
                    }
                    // unreachable: validate refuses unknown comment ids
                    None => break,
                },
            }
        }
    }
    lines
}

pub struct TaskView {
    pub id: String,
    pub state: &'static str,
    pub parent: Option<String>,
    pub name: String,
    /// The pending-judgment mark, when an open proposal targets this task.
    /// The row is an attention cue and a pointer; the full story lives at the seq.
    pub proposal: Option<ProposalMark>,
    /// Size of the attached comment thread; renders as a mere pointer (`#`).
    pub comments: usize,
}

pub struct ProposalMark {
    pub seq: usize,
    pub verb: &'static str,
}

pub fn view_of(id: &TaskId, world: &World) -> Option<TaskView> {
    let task = world.tasks.get(id.0)?;
    Some(TaskView {
        id: format!("t-{}", id.0),
        state: state_of(&task.state),
        parent: task.parent_id.map(|p| format!("t-{}", p.0)),
        name: task.name.as_str().to_string(),
        proposal: world
            .proposals
            .iter()
            .find(|(_, p)| {
                p.state == ProposalState::Open
                    && matches!(&p.action,
                        ProposalAction::Drop { task_id } | ProposalAction::Release { task_id }
                            if *task_id == *id)
            })
            .map(|(id, p)| ProposalMark {
                seq: id.0 .0,
                verb: match p.action {
                    ProposalAction::Drop { .. } => "drop",
                    ProposalAction::Release { .. } => "release",
                },
            }),
        comments: comment_thread(world, id).len(),
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::comment::CommentId;
    use crate::prose::Prose;
    use crate::store::RecordId;

    #[test]
    fn every_event_kind_round_trips() {
        let samples = [
            Event::TaskCreated {
                id: TaskId(0),
                name: Prose::new("implement foo".into()).unwrap(),
                parent_id: None,
            },
            Event::TaskCreated {
                id: TaskId(1),
                name: Prose::new("child".into()).unwrap(),
                parent_id: Some(TaskId(0)),
            },
            Event::TaskClaimed { id: TaskId(0) },
            Event::TaskDone {
                id: TaskId(0),
                receipt: Prose::new("tests green".into()).unwrap(),
            },
            Event::TaskDropped {
                id: TaskId(0),
                note: Prose::new("scope covered elsewhere".into()).unwrap(),
            },
            Event::TaskDropped {
                id: TaskId(0),
                note: Prose::new("superseded by t-2".into()).unwrap(),
            },
            Event::TaskReleased {
                id: TaskId(0),
                note: Prose::new("run dead".into()).unwrap(),
            },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("leaning sections, owner: jerry".into()).unwrap(),
            },
            Event::Commented {
                target: Target::Comment(CommentId(RecordId(6))),
                body: Prose::new("no - pure tree, here is why".into()).unwrap(),
            },
        ];

        for event in &samples {
            let (kind, payload) = disassemble(event);
            let back = assemble(kind, &payload).unwrap_or_else(|e| panic!("{kind}: {e:?}"));
            assert_eq!(&back, event, "different: {kind}");
        }
    }

    /// This exists to prevent accidental changes to the string representation of task states
    #[test]
    fn state_strings_are_the_identifier_vocabulary() {
        assert_eq!(state_of(&TaskState::Open), "open");
        assert_eq!(state_of(&TaskState::Claimed), "claimed");
        assert_eq!(
            state_of(&TaskState::Done(Prose::new("filler".into()).unwrap())),
            "done"
        );
        assert_eq!(state_of(&TaskState::Dropped), "dropped");
    }

    #[test]
    fn unknown_kind_and_tier_are_version_skew_not_corruption() {
        assert!(matches!(
            assemble("task_zapped", "{}"),
            Err(ParseFail::UnknownKind(k)) if k == "task_zapped"
        ));
        assert!(matches!(
            tier_from("aliens"),
            Err(ParseFail::UnknownTier(t)) if t == "aliens"
        ));
    }

    #[test]
    fn malformed_payload_of_known_kind_is_loud() {
        assert!(matches!(
            assemble("task_done", "{"),
            Err(ParseFail::Malformed { .. })
        ));
        assert!(matches!(
            assemble("task_claimed", r#"{"id":"three"}"#),
            Err(ParseFail::Malformed { .. })
        ));
    }

}
