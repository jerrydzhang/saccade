use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::comment::CommentKind;
use crate::objects::incarnation::IncarnationId;
use crate::objects::workspace::WorkspaceContext;
use crate::types::actor::ActorName;
use crate::types::artifact::Artifact;
use crate::types::prose::Prose;
use crate::{CommentId, ProposalId, RecordId};

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct TaskId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum TaskState {
    Open,
    Claimed,
    /// A delivered run's deposit; accept is the only door to done
    Delivered(Prose),
    Done(Prose),
    Dropped,
}

impl TaskState {
    /// The state machine table, all state transitions must go through this
    pub fn transition(&self, event: &Event) -> Option<TaskState> {
        match (self, event) {
            (TaskState::Open, Event::TaskClaimed { .. }) => Some(TaskState::Claimed),
            // an in-thread round claims a delivered task afresh; the
            // demand never reopens it
            (TaskState::Delivered(_), Event::TaskClaimed { .. }) => Some(TaskState::Claimed),
            // the old law's close, reachable only in logs written before
            // the receipts law
            (TaskState::Claimed, Event::TaskDone { receipt, .. }) => {
                Some(TaskState::Done(receipt.clone()))
            }
            (TaskState::Claimed, Event::TaskDelivered { receipt, .. }) => {
                Some(TaskState::Delivered(receipt.clone()))
            }
            // the receipt rides through the accept door
            (TaskState::Delivered(receipt), Event::TaskAccepted { .. }) => {
                Some(TaskState::Done(receipt.clone()))
            }
            (TaskState::Claimed, Event::TaskReleased { .. }) => Some(TaskState::Open),
            (TaskState::Open | TaskState::Done(_), Event::TaskDropped { .. }) => {
                Some(TaskState::Dropped)
            }
            // a demand reopens done: the fired session claims afresh
            (
                TaskState::Done(_),
                Event::Commented {
                    kind: CommentKind::Demand,
                    ..
                },
            ) => Some(TaskState::Open),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub(crate) state: TaskState,
    pub(crate) name: Prose,
    pub(crate) parent_id: Option<TaskId>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct TaskContext {
    pub task: Task,
    /// The log position of this task's birth record
    pub birth: RecordId,
    /// Last updated record id
    pub last_updated: RecordId,
    /// Event time of the claim now held, present only while claimed
    pub claimed_at: Option<u64>,
    /// Event time of the most recent record that moved this task
    pub last_record_at: u64,
    /// Event time of the delivery now held, present only while delivered
    pub delivered_at: Option<u64>,
    pub proposal: Option<ProposalId>,
    pub thread: Vec<CommentId>,
    /// Artifacts the thread holds, in record order
    pub artifacts: Vec<(RecordId, Artifact)>,
    /// Holder of the current claim, present only while the task is claimed
    pub holder: Option<ActorName>,
    /// The attribution that birthed the task, from the birth record
    pub birth_actor: ActorName,
    /// The one live run on this task, None while unbound or terminal
    pub active_incarnation: Option<IncarnationId>,
    /// The task's workspace lineage, present once provisioned
    pub workspace: Option<WorkspaceContext>,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::comment::{CommentKind, Target};
    use crate::types::prose::Prose;

    #[test]
    fn transition_table_admits_exactly_the_legal_cells() {
        let states = [
            TaskState::Open,
            TaskState::Claimed,
            TaskState::Delivered(Prose::new("filler".into()).unwrap()),
            TaskState::Done(Prose::new("filler".into()).unwrap()),
            TaskState::Dropped,
        ];
        let events = [
            Event::TaskCreated {
                name: Prose::new("filler".into()).unwrap(),
                parent_id: None,
            },
            Event::TaskClaimed { id: TaskId(0) },
            Event::TaskDone {
                id: TaskId(0),
                receipt: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskDelivered {
                id: TaskId(0),
                receipt: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskAccepted { id: TaskId(0) },
            Event::TaskDropped {
                id: TaskId(0),
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskReleased {
                id: TaskId(0),
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("filler".into()).unwrap(),
                kind: CommentKind::Demand,
            },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("filler".into()).unwrap(),
                kind: CommentKind::Note,
            },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("filler".into()).unwrap(),
                kind: CommentKind::Steer,
            },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("filler".into()).unwrap(),
                kind: CommentKind::Ask,
            },
        ];

        let legal = |state: &TaskState, event: &Event| {
            matches!(
                (state, event),
                (TaskState::Open, Event::TaskClaimed { .. })
                    | (TaskState::Delivered(_), Event::TaskClaimed { .. })
                    | (TaskState::Claimed, Event::TaskDone { .. })
                    | (TaskState::Claimed, Event::TaskDelivered { .. })
                    | (TaskState::Delivered(_), Event::TaskAccepted { .. })
                    | (TaskState::Claimed, Event::TaskReleased { .. })
                    | (
                        TaskState::Open | TaskState::Done(_),
                        Event::TaskDropped { .. }
                    )
                    | (
                        TaskState::Done(_),
                        Event::Commented {
                            kind: CommentKind::Demand,
                            ..
                        }
                    )
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
