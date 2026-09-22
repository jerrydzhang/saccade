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

/// What a comment is for. No variant carries an address: routing is
/// structural — the note pulls, the demand fires a run, the steer
/// reaches the live run, the ask holds a wait for its answer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommentKind {
    Note,
    Demand,
    Steer,
    Ask,
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

/// A steer's delivery: standing intent until the live run consumes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteerDelivery {
    Standing,
    Forwarded,
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
            // a prepare refusal consumes the demand's live authorization:
            // re-asking is a new comment
            (AgentAttemptState::Authorized { .. }, Event::DemandRefused { .. }) => {
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
    Note,
    /// The demand: fires a run when the task is free, queues while
    /// busy. The attempt is the run machinery's slot on it.
    Demand {
        response: ResponseState,
        attempt: AgentAttemptState,
    },
    /// The steer: forwarded to the live run at the turn boundary,
    /// consumed by it, never re-fired.
    Steer {
        delivery: SteerDelivery,
    },
    /// The ask: a blocking question; the first reply answers it and
    /// resumes the run, later replies are ordinary notes.
    Ask {
        response: ResponseState,
    },
}

impl CommentState {
    /// The comment state machine table, all transitions must go through
    /// this. The replying tier is a cell guard for demands — the human's
    /// word ends any demand — and asks answer at any judgment tier;
    /// notes and steers never transition on replies. A steer moves only
    /// on its own forward event.
    pub fn transition(&self, event: &Event, record: &Record) -> Option<CommentState> {
        match (self, event) {
            (
                CommentState::Demand {
                    response: ResponseState::Awaiting,
                    attempt,
                },
                Event::Commented { .. },
            ) if matches!(record.context.tier, Tier::Agent | Tier::Human) => {
                Some(CommentState::Demand {
                    response: ResponseState::Responded {
                        reply: CommentId(record.id),
                    },
                    attempt: attempt.transition(event, record)?,
                })
            }
            (
                CommentState::Demand { response, attempt },
                Event::IncarnationBound { .. }
                | Event::IncarnationSettled { .. }
                // only a rejected prompt, a cancel, or a refusal is
                // demand-ending; acceptance changes the run, never the demand
                | Event::IncarnationPromptRejected { .. }
                | Event::IncarnationCancelled { .. }
                | Event::DemandRefused { .. },
            ) => Some(CommentState::Demand {
                response: response.clone(),
                attempt: attempt.transition(event, record)?,
            }),
            // the run consumed the steer: standing intent, delivered once
            (CommentState::Steer { delivery: SteerDelivery::Standing }, Event::SteerForwarded { .. }) => {
                Some(CommentState::Steer {
                    delivery: SteerDelivery::Forwarded,
                })
            }
            // the first reply answers the ask; an answered ask holds —
            // later replies are ordinary notes
            (
                CommentState::Ask {
                    response: ResponseState::Awaiting,
                },
                Event::Commented { .. },
            ) if matches!(record.context.tier, Tier::Agent | Tier::Human) => Some(CommentState::Ask {
                response: ResponseState::Responded {
                    reply: CommentId(record.id),
                },
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
    /// The machinery's refusal to run this demand, when it refused
    pub refusal: Option<Refusal>,
}

/// Why the machinery refused to run a demand, and when: the asker's
/// fact, rendered on the thread the demand lives on.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub reason: Prose,
    /// Event time of the refusal record
    pub at: u64,
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
                kind: CommentKind::Note,
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

    fn forwarded() -> Record {
        at(
            Tier::System,
            Event::SteerForwarded {
                steer: CommentId(RecordId(1)),
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

    fn refused() -> Record {
        at(
            Tier::System,
            Event::DemandRefused {
                demand: CommentId(RecordId(1)),
                reason: Prose::new("the worktree is a disk-only leftover".into()).unwrap(),
            },
        )
    }

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
            refused(),
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
                    (AgentAttemptState::Authorized { .. }, Event::DemandRefused { .. }) => true,
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
            CommentState::Note,
            CommentState::Demand {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Authorized { trigger },
            },
            CommentState::Demand {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::InFlight {
                    incarnation: IncarnationId(RecordId(3)),
                },
            },
            CommentState::Demand {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Spent,
            },
            CommentState::Demand {
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
            refused(),
            forwarded(),
        ];

        for state in &states {
            for record in &records {
                let legal = match (state, &record.event) {
                    (CommentState::Note, _) => false,
                    (CommentState::Demand { response, attempt }, Event::Commented { .. }) => {
                        *response == ResponseState::Awaiting
                            && matches!(record.context.tier, Tier::Agent | Tier::Human)
                            && attempt.transition(&record.event, record).is_some()
                    }
                    (
                        CommentState::Demand { attempt, .. },
                        Event::IncarnationBound { .. }
                        | Event::IncarnationSettled { .. }
                        | Event::IncarnationPromptRejected { .. }
                        | Event::IncarnationCancelled { .. }
                        | Event::DemandRefused { .. },
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

    #[test]
    fn steer_and_ask_tables_admit_exactly_the_legal_cells() {
        let states = [
            CommentState::Steer {
                delivery: SteerDelivery::Standing,
            },
            CommentState::Steer {
                delivery: SteerDelivery::Forwarded,
            },
            CommentState::Ask {
                response: ResponseState::Awaiting,
            },
            CommentState::Ask {
                response: ResponseState::Responded {
                    reply: CommentId(RecordId(9)),
                },
            },
        ];
        let records = [
            reply(Tier::Human),
            reply(Tier::Agent),
            reply(Tier::System),
            forwarded(),
            settled(),
            bind(RecordId(1)),
        ];

        for state in &states {
            for record in &records {
                let legal = match (state, &record.event) {
                    (
                        CommentState::Steer {
                            delivery: SteerDelivery::Standing,
                        },
                        Event::SteerForwarded { .. },
                    ) => true,
                    (
                        CommentState::Ask {
                            response: ResponseState::Awaiting,
                        },
                        Event::Commented { .. },
                    ) => matches!(record.context.tier, Tier::Agent | Tier::Human),
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
