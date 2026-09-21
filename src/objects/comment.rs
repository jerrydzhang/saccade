use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::incarnation::IncarnationId;
use crate::objects::task::TaskId;
use crate::store::{Record, RecordId, Tier};
use crate::types::actor::ActorName;
use crate::types::prose::Prose;

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct CommentId(pub RecordId);

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Task(TaskId),
    Comment(CommentId),
}

/// Who a comment asks for a response. System is not a respondent and is
/// unrepresentable here rather than rejected at the fold.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Addressee {
    Human,
    Agent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    pub(crate) target: Target,
    pub(crate) body: Prose,
    /// The task this comment lives on, derived from its target chain
    pub(crate) root: TaskId,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseState {
    Awaiting,
    Responded { reply: CommentId },
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentAttemptState {
    Authorized { trigger: RecordId },
    InFlight { incarnation: IncarnationId },
    Spent,
}

impl AgentAttemptState {
    /// The attempt state machine table, all transitions must go through
    /// this. The event names the transition; the record supplies the
    /// record positions the event does not carry.
    pub fn transition(&self, event: &Event, record: &Record) -> Option<AgentAttemptState> {
        match (self, event) {
            // binding consumes the demand's live authorization, exactly
            (
                AgentAttemptState::Authorized { trigger },
                Event::IncarnationBound {
                    trigger: binding, ..
                },
            ) if trigger == binding => Some(AgentAttemptState::InFlight {
                incarnation: IncarnationId(record.id),
            }),
            // a reply consumes a pre-bind authorization
            (AgentAttemptState::Authorized { .. }, Event::Commented { .. }) => {
                Some(AgentAttemptState::Spent)
            }
            // a reply on an in-flight demand: the slot holds until the run ends
            (AgentAttemptState::InFlight { .. }, Event::Commented { .. }) => Some(self.clone()),
            // a terminal run frees the slot
            (
                AgentAttemptState::InFlight { .. },
                Event::IncarnationSettled { .. }
                | Event::IncarnationPromptRejected { .. }
                | Event::IncarnationCancelled { .. },
            ) => Some(AgentAttemptState::Spent),
            // an answered demand's run may still be settling, and a late
            // reply answers an ended run's demand
            (
                spent @ AgentAttemptState::Spent,
                Event::Commented { .. }
                | Event::IncarnationSettled { .. }
                | Event::IncarnationPromptRejected { .. }
                | Event::IncarnationCancelled { .. },
            ) => Some(spent.clone()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CommentState {
    Unaddressed,
    AddressedToHuman {
        response: ResponseState,
    },
    AddressedToAgent {
        response: ResponseState,
        attempt: AgentAttemptState,
    },
}

impl CommentState {
    /// The demand state machine table, all transitions must go through
    /// this. The replying tier is a cell guard, so who may answer is a
    /// table fact — the human's word ends any demand; unaddressed
    /// comments never transition.
    pub fn transition(&self, event: &Event, record: &Record) -> Option<CommentState> {
        match (self, event) {
            (
                CommentState::AddressedToHuman {
                    response: ResponseState::Awaiting,
                },
                Event::Commented { .. },
            ) if record.context.tier == Tier::Human => Some(CommentState::AddressedToHuman {
                response: ResponseState::Responded {
                    reply: CommentId(record.id),
                },
            }),
            (
                CommentState::AddressedToAgent {
                    response: ResponseState::Awaiting,
                    attempt,
                },
                Event::Commented { .. },
            ) if matches!(record.context.tier, Tier::Agent | Tier::Human) => {
                Some(CommentState::AddressedToAgent {
                    response: ResponseState::Responded {
                        reply: CommentId(record.id),
                    },
                    attempt: attempt.transition(event, record)?,
                })
            }
            (
                CommentState::AddressedToAgent { response, attempt },
                Event::IncarnationBound { .. }
                | Event::IncarnationSettled { .. }
                // only a rejected prompt or a cancel is run-ending;
                // acceptance changes the run, never the demand
                | Event::IncarnationPromptRejected { .. }
                | Event::IncarnationCancelled { .. },
            ) => Some(CommentState::AddressedToAgent {
                response: response.clone(),
                attempt: attempt.transition(event, record)?,
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommentContext {
    pub comment: Comment,
    pub actor: ActorName,
    pub tier: Tier,
    pub state: CommentState,
    /// Event time of the comment's birth record
    pub born_at: u64,
}

#[cfg(test)]
mod tables {
    use super::*;
    use crate::store::{Context, Record};
    use crate::types::failure::{FailureCode, FailureEvidence};
    use crate::types::pointers::SessionPointer;

    fn at(tier: Tier, event: Event) -> Record {
        Record {
            id: RecordId(9),
            timestamp: 1,
            context: Context {
                actor: ActorName::new("saccade bot".into()).unwrap(),
                tier,
            },
            event,
        }
    }

    fn reply(tier: Tier) -> Record {
        at(
            tier,
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("filler".into()).unwrap(),
                addressee: None,
            },
        )
    }

    fn bind(trigger: RecordId) -> Record {
        at(
            Tier::System,
            Event::IncarnationBound {
                task_id: TaskId(0),
                response_target: CommentId(RecordId(1)),
                trigger,
                actor: ActorName::new("pi".into()).unwrap(),
                session: SessionPointer::new("/tmp/session".into()).unwrap(),
            },
        )
    }

    fn settled() -> Record {
        at(
            Tier::System,
            Event::IncarnationSettled {
                id: IncarnationId(RecordId(3)),
            },
        )
    }

    fn prompt_accepted() -> Record {
        at(
            Tier::System,
            Event::IncarnationPromptAccepted {
                id: IncarnationId(RecordId(3)),
            },
        )
    }

    fn prompt_rejected() -> Record {
        at(
            Tier::System,
            Event::IncarnationPromptRejected {
                id: IncarnationId(RecordId(3)),
                evidence: FailureEvidence::new(FailureCode::PromptRejected, None),
            },
        )
    }

    fn cancelled() -> Record {
        at(
            Tier::Human,
            Event::IncarnationCancelled {
                id: IncarnationId(RecordId(3)),
            },
        )
    }

    /// This test doesn't really test anything its more just a contract that at the time this test
    /// was written this is the expected behavior that shouldn't regress
    #[test]
    fn attempt_table_admits_exactly_the_legal_cells() {
        let trigger = RecordId(1);
        let states = [
            AgentAttemptState::Authorized { trigger },
            AgentAttemptState::InFlight {
                incarnation: IncarnationId(RecordId(3)),
            },
            AgentAttemptState::Spent,
        ];
        let records = [
            reply(Tier::Human),
            reply(Tier::Agent),
            // a stale correlation must not bind
            bind(RecordId(2)),
            bind(trigger),
            settled(),
            prompt_accepted(),
            prompt_rejected(),
            cancelled(),
        ];

        for state in &states {
            for record in &records {
                let legal = match (state, &record.event) {
                    (
                        AgentAttemptState::Authorized { trigger },
                        Event::IncarnationBound {
                            trigger: binding, ..
                        },
                    ) => trigger == binding,
                    (AgentAttemptState::Authorized { .. }, Event::Commented { .. }) => true,
                    (AgentAttemptState::InFlight { .. }, Event::Commented { .. }) => true,
                    (
                        AgentAttemptState::InFlight { .. },
                        Event::IncarnationSettled { .. } | Event::IncarnationPromptRejected { .. },
                    ) => true,
                    (AgentAttemptState::InFlight { .. }, Event::IncarnationCancelled { .. }) => {
                        true
                    }
                    (
                        AgentAttemptState::Spent,
                        Event::Commented { .. }
                        | Event::IncarnationSettled { .. }
                        | Event::IncarnationPromptRejected { .. }
                        | Event::IncarnationCancelled { .. },
                    ) => true,
                    _ => false,
                };
                assert_eq!(
                    state.transition(&record.event, record).is_some(),
                    legal,
                    "table disagrees at ({state:?}, {:?})",
                    record.event
                );
            }
        }
    }

    #[test]
    fn demand_table_admits_exactly_the_legal_cells() {
        let answered = CommentId(RecordId(9));
        let trigger = RecordId(1);
        let states = [
            CommentState::Unaddressed,
            CommentState::AddressedToHuman {
                response: ResponseState::Awaiting,
            },
            CommentState::AddressedToHuman {
                response: ResponseState::Responded { reply: answered },
            },
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Authorized { trigger },
            },
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::InFlight {
                    incarnation: IncarnationId(RecordId(3)),
                },
            },
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Spent,
            },
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply: answered },
                attempt: AgentAttemptState::Spent,
            },
        ];
        let records = [
            reply(Tier::Human),
            reply(Tier::Agent),
            reply(Tier::System),
            bind(trigger),
            bind(RecordId(2)),
            settled(),
            prompt_accepted(),
            prompt_rejected(),
            cancelled(),
        ];

        for state in &states {
            for record in &records {
                let legal = match (state, &record.event) {
                    (CommentState::Unaddressed, _) => false,
                    (CommentState::AddressedToHuman { response }, Event::Commented { .. }) => {
                        *response == ResponseState::Awaiting && record.context.tier == Tier::Human
                    }
                    (CommentState::AddressedToHuman { .. }, _) => false,
                    (
                        CommentState::AddressedToAgent { response, attempt },
                        Event::Commented { .. },
                    ) => {
                        *response == ResponseState::Awaiting
                            && matches!(record.context.tier, Tier::Agent | Tier::Human)
                            && attempt.transition(&record.event, record).is_some()
                    }
                    (
                        CommentState::AddressedToAgent { attempt, .. },
                        Event::IncarnationBound { .. }
                        | Event::IncarnationSettled { .. }
                        | Event::IncarnationPromptRejected { .. }
                        | Event::IncarnationCancelled { .. },
                    ) => attempt.transition(&record.event, record).is_some(),
                    _ => false,
                };
                assert_eq!(
                    state.transition(&record.event, record).is_some(),
                    legal,
                    "table disagrees at ({state:?}, {:?})",
                    record.event
                );
            }
        }
    }
}
