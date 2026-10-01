use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::incarnation::IncarnationId;
use crate::objects::task::TaskId;
use crate::store::{RecordId, Tier};
use crate::types::actor::ActorName;
use crate::types::pointers::GitCommit;
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommentKind {
    Note,
    Demand { base: GitCommit },
    Steer,
    Ask,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    pub target: Target,
    pub body: Prose,
    /// The task this comment lives on, derived from its target chain
    pub root: TaskId,
    pub state: CommentState,
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
pub enum CommentState {
    Note,
    /// The demand: fires a run when the task is free, queues while
    /// busy. The bound run and the refusal are the context's facts.
    Demand {
        response: ResponseState,
        base: GitCommit,
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
    /// The standing-error repair, terminal: the deposit's body left
    /// the fold's presentation, the tombstone holds its place.
    Withdrawn(Prose),
}

impl CommentState {
    /// The comment state machine table, all event-named transitions
    /// must go through this. The event alone names the transition;
    /// the reply's tier gate and its record position are the store
    /// arm's facts.
    pub fn transition(&self, event: &Event) -> Option<CommentState> {
        match (self, event) {
            (
                CommentState::Steer {
                    delivery: SteerDelivery::Standing,
                },
                Event::SteerForwarded { .. },
            ) => Some(CommentState::Steer {
                delivery: SteerDelivery::Forwarded,
            }),
            (CommentState::Note, Event::CommentWithdrawn { note, .. }) => {
                Some(CommentState::Withdrawn(note.clone()))
            }
            _ => None,
        }
    }

    /// The reply door: the first reply answers an awaiting demand or
    /// ask; every other state holds, and later replies are ordinary
    /// notes. The reply pointer is the answering record's position,
    /// which only the fold knows; the reply's tier gate lives in the
    /// store arm.
    pub fn answered(&self, reply: CommentId) -> Option<CommentState> {
        match self {
            CommentState::Demand {
                response: ResponseState::Awaiting,
                base,
            } => Some(CommentState::Demand {
                response: ResponseState::Responded { reply },
                base: base.clone(),
            }),
            CommentState::Ask {
                response: ResponseState::Awaiting,
            } => Some(CommentState::Ask {
                response: ResponseState::Responded { reply },
            }),
            _ => None,
        }
    }
}

/// The latest revision of a folded comment's body: the record that
/// swapped it in and the actor who did — disclosure is a fold fact,
/// named whenever the reviser differs from the birth author.
#[derive(Clone, Debug, PartialEq)]
pub struct Revision {
    pub record: RecordId,
    pub reviser: ActorName,
}

/// The withdrawal's disclosure pointer: the record that withdrew and
/// the actor who did. The note itself rides the state, the way a
/// receipt rides a done task.
#[derive(Clone, Debug, PartialEq)]
pub struct Withdrawal {
    pub record: RecordId,
    pub withdrawer: ActorName,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommentContext {
    pub comment: Comment,
    pub actor: ActorName,
    pub tier: Tier,
    /// Event time of the comment's birth record
    pub born_at: u64,
    /// The latest revision of the body, when it was ever revised —
    /// a pointer with its reviser, never the payload it swapped in
    pub revised: Option<Revision>,
    /// The standing-error repair's pointer, when the deposit was
    /// withdrawn — beside the revised pointer it mirrors
    pub withdrawn: Option<Withdrawal>,
    /// The machinery's refusal to run this demand, when it refused
    pub refusal: Option<Refusal>,
    /// The run this demand bound, when it bound one: written by the
    /// bind arm, never cleared — re-asking is a new comment
    pub bound: Option<IncarnationId>,
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
    use crate::types::actor::ActorName;
    use crate::types::failure::{FailureCode, FailureEvidence};
    use crate::types::pointers::SessionPointer;

    fn comment_event(kind: CommentKind) -> Event {
        Event::Commented {
            target: Target::Task(TaskId(0)),
            body: Prose::new("filler".into()).unwrap(),
            kind,
        }
    }

    fn withdrawn_event() -> Event {
        Event::CommentWithdrawn {
            id: CommentId(RecordId(1)),
            note: Prose::new("parked in the wrong place".into()).unwrap(),
        }
    }

    fn forwarded_event() -> Event {
        Event::SteerForwarded {
            steer: CommentId(RecordId(1)),
        }
    }

    fn bind(trigger: RecordId) -> Event {
        Event::IncarnationBound {
            task_id: TaskId(0),
            response_target: CommentId(RecordId(1)),
            trigger,
            actor: ActorName::new("pi".into()).unwrap(),
            session: SessionPointer::new("/tmp/session".into()).unwrap(),
        }
    }

    fn demand(response: ResponseState) -> CommentState {
        CommentState::Demand {
            response,
            base: GitCommit::new("a1b2c3".into()).unwrap(),
        }
    }

    fn ask(response: ResponseState) -> CommentState {
        CommentState::Ask { response }
    }

    fn steer(delivery: SteerDelivery) -> CommentState {
        CommentState::Steer { delivery }
    }

    fn withdrawn() -> CommentState {
        CommentState::Withdrawn(Prose::new("parked in the wrong place".into()).unwrap())
    }

    fn replied(response: CommentId) -> ResponseState {
        ResponseState::Responded { reply: response }
    }

    #[test]
    fn withdrawal_is_a_note_deposits_terminal_door() {
        assert_eq!(
            CommentState::Note.transition(&withdrawn_event()),
            Some(withdrawn())
        );
        let others = [
            demand(ResponseState::Awaiting),
            demand(replied(CommentId(RecordId(9)))),
            ask(ResponseState::Awaiting),
            ask(replied(CommentId(RecordId(9)))),
            steer(SteerDelivery::Standing),
            steer(SteerDelivery::Forwarded),
            withdrawn(),
        ];
        for state in &others {
            assert_eq!(state.transition(&withdrawn_event()), None, "{state:?}");
        }
    }

    #[test]
    fn a_steer_moves_only_on_its_own_forward() {
        assert_eq!(
            steer(SteerDelivery::Standing).transition(&forwarded_event()),
            Some(steer(SteerDelivery::Forwarded))
        );
        let others = [
            CommentState::Note,
            demand(ResponseState::Awaiting),
            ask(ResponseState::Awaiting),
            steer(SteerDelivery::Forwarded),
            withdrawn(),
        ];
        for state in &others {
            assert_eq!(state.transition(&forwarded_event()), None, "{state:?}");
        }
    }

    /// Replies, bindings, terminals, and refusals move no comment
    /// state: the reply door answers, the context records the rest.
    #[test]
    fn replies_and_machinery_events_move_no_state() {
        let states = [
            CommentState::Note,
            demand(ResponseState::Awaiting),
            demand(replied(CommentId(RecordId(9)))),
            ask(ResponseState::Awaiting),
            ask(replied(CommentId(RecordId(9)))),
            steer(SteerDelivery::Standing),
            steer(SteerDelivery::Forwarded),
            withdrawn(),
        ];
        let events = [
            comment_event(CommentKind::Note),
            comment_event(CommentKind::Demand {
                base: GitCommit::new("a1b2c3".into()).unwrap(),
            }),
            comment_event(CommentKind::Steer),
            comment_event(CommentKind::Ask),
            Event::CommentRevised {
                id: CommentId(RecordId(1)),
                body: Prose::new("filler".into()).unwrap(),
            },
            bind(RecordId(1)),
            Event::IncarnationPromptAccepted {
                id: crate::objects::incarnation::IncarnationId(RecordId(3)),
            },
            Event::IncarnationPromptRejected {
                id: crate::objects::incarnation::IncarnationId(RecordId(3)),
                evidence: FailureEvidence::new(FailureCode::PromptRejected, None),
            },
            Event::IncarnationSettled {
                id: crate::objects::incarnation::IncarnationId(RecordId(3)),
            },
            Event::IncarnationCancelled {
                id: crate::objects::incarnation::IncarnationId(RecordId(3)),
            },
            Event::DemandRefused {
                demand: CommentId(RecordId(1)),
                reason: Prose::new("the worktree is a disk-only leftover".into()).unwrap(),
            },
        ];
        for state in &states {
            for event in &events {
                assert_eq!(state.transition(event), None, "table moved {state:?}");
            }
        }
    }

    /// The reply door answers an awaiting demand or ask and nothing
    /// else: an answered one holds, the other variants never had one.
    #[test]
    fn the_reply_door_answers_awaiting_demands_and_asks_alone() {
        let reply = CommentId(RecordId(9));
        assert_eq!(
            demand(ResponseState::Awaiting).answered(reply),
            Some(demand(replied(reply)))
        );
        assert_eq!(
            ask(ResponseState::Awaiting).answered(reply),
            Some(ask(replied(reply)))
        );
        let others = [
            CommentState::Note,
            demand(replied(CommentId(RecordId(2)))),
            ask(replied(CommentId(RecordId(2)))),
            steer(SteerDelivery::Standing),
            steer(SteerDelivery::Forwarded),
            withdrawn(),
        ];
        for state in &others {
            assert_eq!(state.answered(reply), None, "{state:?}");
        }
    }
}
