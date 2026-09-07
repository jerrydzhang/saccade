use crate::events::Event;

#[derive(Debug)]
pub enum Reject {
    InvalidTaskId,
    InvalidParentTaskId,
    InvalidStateTransition,
    HumanOnly,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Receipt(pub(crate) String);

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Copy, Clone)]
pub struct TaskId(pub(crate) usize);

/// Why a task was dropped, pre-registered taxonomy
#[derive(Clone, Debug, PartialEq)]
pub enum AbandonReason {
    /// Task we deemed unwanted this can happen either before or after implementation
    Unwanted,
    /// Task was superseded this is distinct from unwanted since it means something has taken
    /// its place
    Superseded,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TaskState {
    Open,
    Claimed,
    Done(Receipt),
    Dropped(AbandonReason),
}

impl TaskState {
    /// The state machine table, all state transitions must go through this
    pub fn transition(&self, event: &Event) -> Result<TaskState, Reject> {
        match (self, event) {
            (TaskState::Open, Event::TaskClaimed { .. }) => Ok(TaskState::Claimed),
            (TaskState::Claimed, Event::TaskDone { receipt, .. }) => {
                Ok(TaskState::Done(receipt.clone()))
            }
            (TaskState::Open | TaskState::Done(_), Event::TaskDropped { reason, .. }) => {
                Ok(TaskState::Dropped(reason.clone()))
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
    pub(crate) id: TaskId,
    pub(crate) state: TaskState,
    pub(crate) task_name: String,
    pub(crate) parent_id: Option<TaskId>,
}

impl Task {
    pub fn apply(&self, event: &Event) -> Result<Task, Reject> {
        let new_state = self.state.transition(event)?;
        Ok(Task {
            state: new_state,
            task_name: self.task_name.clone(),
            ..*self
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transition_table_admits_exactly_the_legal_cells() {
        let states = [
            TaskState::Open,
            TaskState::Claimed,
            TaskState::Done(Receipt(String::new())),
            TaskState::Dropped(AbandonReason::Unwanted),
        ];
        let events = [
            Event::TaskCreated {
                id: TaskId(0),
                task_name: String::new(),
                parent_id: None,
            },
            Event::TaskClaimed { id: TaskId(0) },
            Event::TaskDone {
                id: TaskId(0),
                receipt: Receipt(String::new()),
            },
            Event::TaskDropped {
                id: TaskId(0),
                reason: AbandonReason::Unwanted,
                note: None,
            },
        ];

        let legal = |state: &TaskState, event: &Event| {
            matches!(
                (state, event),
                (TaskState::Open, Event::TaskClaimed { .. })
                    | (TaskState::Claimed, Event::TaskDone { .. })
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
