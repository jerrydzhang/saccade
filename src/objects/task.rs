use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::proposal::ProposalContext;
use crate::prose::Prose;
use crate::{CommentId, ProposalAction, ProposalId, ProposalState, RecordId, Reject, World};

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
pub struct TaskView {
    pub id: String,
    pub state: &'static str,
    pub parent: Option<String>,
    pub name: String,
    /// The pending-judgment mark, when an open proposal targets this task.
    /// The row is an attention cue and a pointer; the full story lives at the seq.
    pub proposal: Option<ProposalMark>,
    /// Size of the attached comment thread; renders as a mere pointer (`#`).
    pub n_comments: usize,
}

pub struct ProposalMark {
    pub seq: usize,
    pub verb: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaskContext {
    pub task: Task,
    /// Last updated record id
    pub last_updated: RecordId,
    pub proposal: Option<ProposalId>,
    pub thread: Vec<CommentId>,
}

impl TaskContext {
    pub fn view(&self, id: TaskId, proposal_ctx: Option<&ProposalContext>) -> TaskView {
        TaskView {
            id: format!("t-{}", id.0),
            state: match self.task.state {
                TaskState::Open => "open",
                TaskState::Claimed => "claimed",
                TaskState::Done(_) => "done",
                TaskState::Dropped => "dropped",
            },
            parent: self.task.parent_id.map(|p| format!("t-{}", p.0)),
            name: self.task.name.as_str().to_string(),
            proposal: match proposal_ctx {
                Some(proposal_ctx) => Some(ProposalMark {
                    seq: self.proposal.unwrap().0.0,
                    verb: match proposal_ctx.proposal.action {
                        ProposalAction::Drop { .. } => "drop",
                        ProposalAction::Release { .. } => "release",
                    },
                }),
                None => None,
            },
            n_comments: self.thread.len(),
        }
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
