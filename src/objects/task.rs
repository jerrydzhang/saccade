use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::incarnation::IncarnationId;
use crate::objects::workspace::WorkspaceContext;
use crate::types::actor::ActorName;
use crate::types::prose::Prose;
use crate::{CommentId, ProposalId, RecordId};

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct TaskId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum TaskState {
    Open,
    Claimed,
    Done(Prose),
    Dropped,
}

impl TaskState {
    /// The state machine table, all state transitions must go through this
    pub fn transition(&self, event: &Event) -> Option<TaskState> {
        match (self, event) {
            (TaskState::Open, Event::TaskClaimed { .. }) => Some(TaskState::Claimed),
            (TaskState::Claimed, Event::TaskDone { receipt, .. }) => {
                Some(TaskState::Done(receipt.clone()))
            }
            (TaskState::Claimed, Event::TaskReleased { .. }) => Some(TaskState::Open),
            (TaskState::Open | TaskState::Done(_), Event::TaskDropped { .. }) => {
                Some(TaskState::Dropped)
            }
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
    /// Last updated record id
    pub last_updated: RecordId,
    pub proposal: Option<ProposalId>,
    pub thread: Vec<CommentId>,
    /// Holder of the current claim, present only while the task is claimed
    pub holder: Option<ActorName>,
    /// The one live run on this task, None while unbound or terminal
    pub active_incarnation: Option<IncarnationId>,
    /// The task's workspace lineage, present once provisioned
    pub workspace: Option<WorkspaceContext>,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::prose::Prose;

    /// This test doesn't really test anything its more just a contract that at the time this test
    /// was written this is the expected behavior that shouldn't regress
    #[test]
    fn transition_table_admits_exactly_the_legal_cells() {
        let states = [
            TaskState::Open,
            TaskState::Claimed,
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
            Event::TaskDropped {
                id: TaskId(0),
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskReleased {
                id: TaskId(0),
                note: Prose::new("filler".into()).unwrap(),
            },
        ];

        let legal = |state: &TaskState, event: &Event| {
            matches!(
                (state, event),
                (TaskState::Open, Event::TaskClaimed { .. })
                    | (TaskState::Claimed, Event::TaskDone { .. })
                    | (TaskState::Claimed, Event::TaskReleased { .. })
                    | (
                        TaskState::Open | TaskState::Done(_),
                        Event::TaskDropped { .. }
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
