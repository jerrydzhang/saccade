use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::comment::CommentId;
use crate::objects::task::TaskId;
use crate::store::RecordId;
use crate::types::actor::ActorName;
use crate::types::pointers::SessionPointer;

/// The record id of IncarnationBound is the incarnation's identity.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct IncarnationId(pub RecordId);

#[derive(Clone, Debug, PartialEq)]
pub enum IncarnationState {
    Bound,
    PromptAccepted,
    Settled,
    /// A rejected prompt terminalizes the run: nothing accepted the work.
    Interrupted,
}

impl IncarnationState {
    /// The run lifecycle table, all transitions must go through this.
    /// The prompt event is the intake acknowledgment, not the outcome:
    /// at most one per run, and settling requires acceptance.
    pub fn transition(&self, event: &Event) -> Option<IncarnationState> {
        match (self, event) {
            (IncarnationState::Bound, Event::IncarnationPromptAccepted { .. }) => {
                Some(IncarnationState::PromptAccepted)
            }
            (IncarnationState::Bound, Event::IncarnationPromptRejected { .. }) => {
                Some(IncarnationState::Interrupted)
            }
            (IncarnationState::PromptAccepted, Event::IncarnationSettled { .. }) => {
                Some(IncarnationState::Settled)
            }
            // producing a record is legal only while accepted; the state
            // itself does not move
            (accepted @ IncarnationState::PromptAccepted, Event::RecordProducedBy { .. }) => {
                Some(accepted.clone())
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IncarnationContext {
    pub task_id: TaskId,
    pub response_target: CommentId,
    pub trigger: RecordId,
    pub actor: ActorName,
    pub session: SessionPointer,
    pub state: IncarnationState,
    /// Agent records this run produced, in birth order.
    pub produced: Vec<RecordId>,
}

impl IncarnationContext {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            IncarnationState::Settled | IncarnationState::Interrupted
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::events::Event;
    use crate::objects::comment::CommentId;
    use crate::types::actor::ActorName;
    use crate::types::failure::{FailureCode, FailureEvidence};
    use crate::types::pointers::SessionPointer;

    /// This test doesn't really test anything its more just a contract that at the time this test
    /// was written this is the expected behavior that shouldn't regress
    #[test]
    fn lifecycle_table_admits_exactly_the_legal_cells() {
        let states = [
            IncarnationState::Bound,
            IncarnationState::PromptAccepted,
            IncarnationState::Settled,
            IncarnationState::Interrupted,
        ];
        let events = [
            Event::IncarnationPromptAccepted {
                id: IncarnationId(RecordId(0)),
            },
            Event::IncarnationPromptRejected {
                id: IncarnationId(RecordId(0)),
                evidence: FailureEvidence::new(FailureCode::PromptRejected, None),
            },
            Event::IncarnationSettled {
                id: IncarnationId(RecordId(0)),
            },
            // present to show the table refuses non-lifecycle events outright
            Event::IncarnationBound {
                task_id: TaskId(0),
                response_target: CommentId(RecordId(1)),
                trigger: RecordId(0),
                actor: ActorName::new("pi".into()).unwrap(),
                session: SessionPointer::new("/tmp/session".into()).unwrap(),
            },
        ];

        for state in &states {
            for event in &events {
                let legal = matches!(
                    (state, event),
                    (
                        IncarnationState::Bound,
                        Event::IncarnationPromptAccepted { .. }
                    ) | (
                        IncarnationState::Bound,
                        Event::IncarnationPromptRejected { .. }
                    ) | (
                        IncarnationState::PromptAccepted,
                        Event::IncarnationSettled { .. }
                    ) | (
                        IncarnationState::PromptAccepted,
                        Event::RecordProducedBy { .. }
                    )
                );
                assert_eq!(
                    state.transition(event).is_some(),
                    legal,
                    "table disagrees at ({state:?}, {event:?})"
                );
            }
        }
    }
}
