use serde::{Deserialize, Serialize};

use crate::RecordId;
use crate::events::Event;
use crate::objects::comment::{CommentId, CommentKind};
use crate::objects::incarnation::IncarnationId;
use crate::objects::workspace::WorkspaceContext;
use crate::types::actor::ActorName;
use crate::types::artifact::Artifact;
use crate::types::prose::Prose;

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct TaskId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum TaskState {
    Open,
    /// A whole-task offer stands; the current offer's identity is an
    /// envelope fact on the context, never a payload here
    Offered,
    Concluded,
    /// The human's explanation of why the undertaking stopped rides
    /// the state, the way a verdict's grounds ride its comment
    Cancelled(Prose),
}

impl TaskState {
    /// The state machine table, all state transitions must go through
    /// this. The store's reference checks decide whether an Accept or
    /// Reject names the current offer; this table admits the cell.
    pub fn transition(&self, event: &Event) -> Option<TaskState> {
        match (self, event) {
            (
                TaskState::Open,
                Event::Commented {
                    kind: CommentKind::Offer { .. },
                    ..
                },
            ) => Some(TaskState::Offered),
            (
                TaskState::Offered,
                Event::Commented {
                    kind: CommentKind::Accept { .. },
                    ..
                },
            ) => Some(TaskState::Concluded),
            (
                TaskState::Offered,
                Event::Commented {
                    kind: CommentKind::Reject { .. },
                    ..
                },
            ) => Some(TaskState::Open),
            (TaskState::Open | TaskState::Offered, Event::TaskCancelled { explanation, .. }) => {
                Some(TaskState::Cancelled(explanation.clone()))
            }
            (TaskState::Cancelled(_), Event::TaskReopened { .. }) => Some(TaskState::Open),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub state: TaskState,
    pub name: Prose,
    pub parent_id: Option<TaskId>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct TaskContext {
    pub task: Task,
    /// The log position of this task's birth record
    pub birth: RecordId,
    /// Last updated record id
    pub last_updated: RecordId,
    /// Event time of the most recent record that moved this task
    pub last_record_at: u64,
    /// Event time the standing offer landed, present while one stands
    /// or the concluded one holds
    pub offered_at: Option<u64>,
    /// The ownership ledger, earliest first: whole-task responsibility,
    /// transferred by claim, history preserved
    pub ownership: Vec<OwnerEntry>,
    /// The offer now standing, present while offered and after the
    /// accept that concluded; a reject clears it
    pub current_offer: Option<CommentId>,
    pub thread: Vec<CommentId>,
    /// Artifacts the thread holds, in record order
    pub artifacts: Vec<(RecordId, Artifact)>,
    /// The attribution that birthed the task, from the birth record
    pub birth_actor: ActorName,
    /// The task's workspace lineage, present once registered
    pub workspace: Option<WorkspaceContext>,
}

/// The owner an ownership entry names — exactly the ratified kinds.
/// An incarnation owner is a record-derived reference, never inferred
/// from declared scope: a task-scoped helper claiming without the
/// reference owns as the named external agent it is.
#[derive(Clone, Debug, PartialEq)]
pub enum Owner {
    Human(ActorName),
    External(ActorName),
    Incarnation(IncarnationId),
}

/// One holding of whole-task responsibility: who held it, the record
/// that transferred it, when. Ownership is separate from state — no
/// claim is a state transition, no state gates a claim.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnerEntry {
    pub owner: Owner,
    pub record: RecordId,
    pub at: u64,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::comment::{CommentKind, Target};
    use crate::types::prose::Prose;

    fn note() -> Event {
        Event::Commented {
            target: Target::Task(TaskId(0)),
            kind: CommentKind::Note {
                body: Prose::new("filler".into()).unwrap(),
            },
        }
    }

    fn offer() -> Event {
        Event::Commented {
            target: Target::Task(TaskId(0)),
            kind: CommentKind::Offer {
                body: Prose::new("filler".into()).unwrap(),
            },
        }
    }

    fn accept() -> Event {
        Event::Commented {
            target: Target::Comment(CommentId(RecordId(1))),
            kind: CommentKind::Accept {
                offer: CommentId(RecordId(1)),
            },
        }
    }

    fn reject() -> Event {
        Event::Commented {
            target: Target::Comment(CommentId(RecordId(1))),
            kind: CommentKind::Reject {
                offer: CommentId(RecordId(1)),
                grounds: Prose::new("filler".into()).unwrap(),
            },
        }
    }

    #[test]
    fn transition_table_admits_exactly_the_legal_cells() {
        let states = [
            TaskState::Open,
            TaskState::Offered,
            TaskState::Concluded,
            TaskState::Cancelled(Prose::new("filler".into()).unwrap()),
        ];
        let events = [
            Event::TaskCreated {
                name: Prose::new("filler".into()).unwrap(),
                parent_id: None,
            },
            Event::TaskClaimed {
                id: TaskId(0),
                incarnation: None,
            },
            Event::TaskCancelled {
                id: TaskId(0),
                explanation: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskReopened {
                id: TaskId(0),
                explanation: Prose::new("filler".into()).unwrap(),
            },
            note(),
            offer(),
            accept(),
            reject(),
        ];

        let legal = |state: &TaskState, event: &Event| {
            matches!(
                (state, event),
                (
                    TaskState::Open,
                    Event::Commented {
                        kind: CommentKind::Offer { .. },
                        ..
                    }
                ) | (
                    TaskState::Offered,
                    Event::Commented {
                        kind: CommentKind::Accept { .. },
                        ..
                    }
                ) | (
                    TaskState::Offered,
                    Event::Commented {
                        kind: CommentKind::Reject { .. },
                        ..
                    }
                ) | (
                    TaskState::Open | TaskState::Offered,
                    Event::TaskCancelled { .. }
                ) | (TaskState::Cancelled(_), Event::TaskReopened { .. })
            )
        };

        for state in &states {
            for event in &events {
                assert_eq!(
                    state.transition(event).is_some(),
                    legal(state, event),
                    "table disagrees at ({state:?}, {event:?})"
                );
            }
        }
    }
}
