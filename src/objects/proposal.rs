use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::task::TaskId;
use crate::{Prose, RecordId};

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct ProposalId(pub RecordId);

#[derive(Clone, Debug, PartialEq)]
pub enum ProposalState {
    Open,
    Withdrawn(Prose),
    Rejected(Prose),
    Accepted,
}

impl ProposalState {
    /// The state machine table, all state transitions must go through this
    pub fn transition(&self, event: &Event) -> Option<ProposalState> {
        match (self, event) {
            (ProposalState::Open, Event::ProposalWithdrawn { note, .. }) => {
                Some(ProposalState::Withdrawn(note.clone()))
            }
            (ProposalState::Open, Event::ProposalRejected { note, .. }) => {
                Some(ProposalState::Rejected(note.clone()))
            }
            (ProposalState::Open, Event::ProposalAccepted { .. }) => Some(ProposalState::Accepted),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProposalAction {
    Drop { task_id: TaskId },
    Release { task_id: TaskId },
}

impl ProposalAction {
    // I have deliberately chosen to not have it return a vec of events because currently all
    // proposals only emit a single action
    pub fn target_event(&self, note: &Prose) -> Event {
        match self {
            ProposalAction::Drop { task_id } => Event::TaskDropped {
                id: *task_id,
                note: note.clone(),
            },
            ProposalAction::Release { task_id } => Event::TaskReleased {
                id: *task_id,
                note: note.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub(crate) state: ProposalState,
    pub(crate) name: Prose,
    pub(crate) action: ProposalAction,
}

impl Proposal {
    pub fn apply(&self, event: &Event) -> Option<Proposal> {
        let new_state = self.state.transition(event)?;
        Some(Proposal {
            state: new_state.clone(),
            name: self.name.clone(),
            ..*self
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProposalContext {
    pub proposal: Proposal,
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
            ProposalState::Open,
            ProposalState::Withdrawn(Prose::new("ruled out".into()).unwrap()),
            ProposalState::Rejected(Prose::new("ruled real".into()).unwrap()),
            ProposalState::Accepted,
        ];
        let events = [
            Event::ProposalCreated {
                name: Prose::new("filler".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            Event::ProposalWithdrawn {
                id: ProposalId(RecordId(0)),
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::ProposalRejected {
                id: ProposalId(RecordId(0)),
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::ProposalAccepted {
                id: ProposalId(RecordId(0)),
            },
        ];

        let legal = |state: &ProposalState, event: &Event| {
            matches!(
                (state, event),
                (ProposalState::Open, Event::ProposalWithdrawn { .. })
                    | (ProposalState::Open, Event::ProposalRejected { .. })
                    | (ProposalState::Open, Event::ProposalAccepted { .. })
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
