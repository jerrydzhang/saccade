use serde::{Deserialize, Serialize};

use crate::RecordId;
use crate::events::Event;
use crate::objects::incarnation::IncarnationId;
use crate::objects::workspace::WorkspaceContext;
use crate::types::actor::ActorName;
use crate::types::prose::Prose;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct TaskId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum TaskState {
    Open,
    /// A whole-task offer stands; the current offer's identity is an
    /// envelope fact on the context, never a payload here
    Offered,
    Concluded,
    /// The human's explanation of why the undertaking stopped rides
    /// the state
    Cancelled(Prose),
}

impl TaskState {
    /// The state machine table, all state transitions must go through
    /// this.
    pub fn transition(&self, event: &Event) -> Option<TaskState> {
        match (self, event) {
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
    /// The ownership ledger, earliest first: whole-task responsibility,
    /// transferred by claim, history preserved
    pub ownership: Vec<OwnerEntry>,
    /// The attribution that birthed the task, from the birth record
    pub birth_actor: ActorName,
    /// The task's workspace lineage, present once registered
    pub workspace: Option<WorkspaceContext>,
}

/// The owner an ownership entry names - exactly the ratified kinds.
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
/// that transferred it, when. Ownership is separate from state - no
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
    use crate::types::prose::Prose;

    fn filler() -> Prose {
        Prose::new("filler".into()).unwrap()
    }

    #[test]
    fn transition_table_admits_exactly_the_legal_cells() {
        let states = [
            TaskState::Open,
            TaskState::Offered,
            TaskState::Concluded,
            TaskState::Cancelled(filler()),
        ];
        let events = [
            Event::TaskCreated { name: filler() },
            Event::TaskClaimed {
                id: TaskId(0),
                incarnation: None,
            },
            Event::TaskCancelled {
                id: TaskId(0),
                explanation: filler(),
            },
            Event::TaskReopened {
                id: TaskId(0),
                explanation: filler(),
            },
        ];

        let legal = |state: &TaskState, event: &Event| {
            matches!(
                (state, event),
                (
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
