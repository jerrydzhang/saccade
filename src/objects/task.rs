use serde::{Deserialize, Serialize};

use crate::Reject;
use crate::events::Event;
use crate::prose::Prose;

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct TaskId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum TaskState {
    Open,
    // TODO: give claimed an identity so we can
    // detect who claimed the task not just that was claimed
    Claimed,
    Done(Prose),
    Dropped,
}

impl TaskState {
    /// The state machine table, all state transitions must go through this
    pub fn transition(&self, event: &Event) -> Result<TaskState, Reject> {
        match (self, event) {
            (TaskState::Open, Event::TaskClaimed { .. }) => Ok(TaskState::Claimed),
            (TaskState::Claimed, Event::TaskDone { receipt, .. }) => {
                Ok(TaskState::Done(receipt.clone()))
            }
            (TaskState::Claimed, Event::TaskReleased { .. }) => Ok(TaskState::Open),
            (TaskState::Open | TaskState::Done(_), Event::TaskDropped { .. }) => {
                Ok(TaskState::Dropped)
            }
            _ => Err(Reject::InvalidStateTransition),
        }
    }

    pub fn validate(&self, event: &Event) -> Result<(), Reject> {
        self.transition(event).map(|_| ())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub(crate) state: TaskState,
    pub(crate) name: Prose,
    pub(crate) parent_id: Option<TaskId>,
}

impl Task {
    pub fn apply(&self, event: &Event) -> Result<Task, Reject> {
        let new_state = self.state.transition(event)?;
        Ok(Task {
            state: new_state,
            name: self.name.clone(),
            parent_id: self.parent_id,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prose::Prose;

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
                id: TaskId(0),
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
                    state.validate(event).is_ok(),
                    legal(state, event),
                    "table disagrees at ({state:?}, {event:?})"
                );
            }
        }
    }
}
