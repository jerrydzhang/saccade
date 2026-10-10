use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::task::TaskId;
use crate::store::{RecordId, Tier};
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

/// The payload a comment carries. Notes and Requests keep the
/// task-or-comment targets; an Offer addresses its whole task;
/// Accept and Reject address the offer they name; a Use lives on the
/// consuming task and names the source offer. No variant carries an
/// address beyond its target: routing is structural.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommentKind {
    Note {
        body: Prose,
    },
    Request {
        body: Prose,
    },
    /// The whole-task closing account: broadcast, never a point reply
    Offer {
        body: Prose,
    },
    /// Bodyless by law: acceptance is the reference, never prose
    Accept {
        offer: CommentId,
    },
    Reject {
        offer: CommentId,
        grounds: Prose,
    },
    /// The consuming task's declared reliance on a source offer
    Use {
        offer: CommentId,
        reason: Prose,
    },
}

impl CommentKind {
    /// None for the bodyless Accept, which the revision door refuses.
    pub fn prose(&self) -> Option<&Prose> {
        match self {
            CommentKind::Note { body }
            | CommentKind::Request { body }
            | CommentKind::Offer { body } => Some(body),
            CommentKind::Accept { .. } => None,
            CommentKind::Reject { grounds, .. } => Some(grounds),
            CommentKind::Use { reason, .. } => Some(reason),
        }
    }

    pub fn offer(&self) -> Option<CommentId> {
        match self {
            CommentKind::Accept { offer }
            | CommentKind::Reject { offer, .. }
            | CommentKind::Use { offer, .. } => Some(*offer),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    pub target: Target,
    pub kind: CommentKind,
    /// The task this comment lives on, derived from its target chain
    pub root: TaskId,
    pub state: CommentState,
}

/// A comment's own lifecycle, born mirroring its variant; only the
/// note carries a door.
#[derive(Clone, Debug, PartialEq)]
pub enum CommentState {
    Note,
    Request,
    Offer,
    Accept,
    Reject,
    Use,
    /// The standing-error repair, terminal: the deposit's body left
    /// the fold's presentation, the tombstone holds its place.
    Withdrawn(Prose),
}

impl CommentState {
    /// The comment state machine table, all event-named transitions
    /// must go through this: withdrawal is the note deposits' door
    /// alone, every other variant holds.
    pub fn transition(&self, event: &Event) -> Option<CommentState> {
        match (self, event) {
            (CommentState::Note, Event::CommentWithdrawn { note, .. }) => {
                Some(CommentState::Withdrawn(note.clone()))
            }
            _ => None,
        }
    }

    pub fn of(kind: &CommentKind) -> CommentState {
        match kind {
            CommentKind::Note { .. } => CommentState::Note,
            CommentKind::Request { .. } => CommentState::Request,
            CommentKind::Offer { .. } => CommentState::Offer,
            CommentKind::Accept { .. } => CommentState::Accept,
            CommentKind::Reject { .. } => CommentState::Reject,
            CommentKind::Use { .. } => CommentState::Use,
        }
    }
}

/// The latest revision of a folded comment's prose: the record that
/// swapped it in and the actor who did — disclosure is a fold fact,
/// named whenever the reviser differs from the birth author.
#[derive(Clone, Debug, PartialEq)]
pub struct Revision {
    pub record: RecordId,
    pub reviser: ActorName,
}

/// The withdrawal's disclosure pointer: the record that withdrew and
/// the actor who did. The note itself rides the state, the way an
/// explanation rides a cancelled task.
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
    pub scope: Option<crate::store::Scope>,
    /// Event time of the comment's birth record
    pub born_at: u64,
    /// The latest revision of the prose, when it was ever revised —
    /// a pointer with its reviser, never the payload it swapped in
    pub revised: Option<Revision>,
    /// The standing-error repair's pointer, when the deposit was
    /// withdrawn — beside the revised pointer it mirrors
    pub withdrawn: Option<Withdrawal>,
}

#[cfg(test)]
mod tables {
    use super::*;
    use crate::types::prose::Prose;

    fn comment_event(kind: CommentKind) -> Event {
        Event::Commented {
            target: Target::Task(crate::TaskId(0)),
            kind,
        }
    }

    fn withdrawn_event() -> Event {
        Event::CommentWithdrawn {
            id: CommentId(RecordId(1)),
            note: Prose::new("parked in the wrong place".into()).unwrap(),
        }
    }

    fn note() -> CommentKind {
        CommentKind::Note {
            body: Prose::new("filler".into()).unwrap(),
        }
    }

    fn offer() -> CommentKind {
        CommentKind::Offer {
            body: Prose::new("filler".into()).unwrap(),
        }
    }

    fn accept() -> CommentKind {
        CommentKind::Accept {
            offer: CommentId(RecordId(1)),
        }
    }

    fn reject() -> CommentKind {
        CommentKind::Reject {
            offer: CommentId(RecordId(1)),
            grounds: Prose::new("filler".into()).unwrap(),
        }
    }

    fn use_of() -> CommentKind {
        CommentKind::Use {
            offer: CommentId(RecordId(1)),
            reason: Prose::new("filler".into()).unwrap(),
        }
    }

    fn request() -> CommentKind {
        CommentKind::Request {
            body: Prose::new("filler".into()).unwrap(),
        }
    }

    fn withdrawn() -> CommentState {
        CommentState::Withdrawn(Prose::new("parked in the wrong place".into()).unwrap())
    }

    /// Withdrawal is the note deposits' terminal door alone: every
    /// other variant holds, at the table.
    #[test]
    fn withdrawal_is_a_note_deposits_terminal_door() {
        assert_eq!(
            CommentState::Note.transition(&withdrawn_event()),
            Some(withdrawn())
        );
        let others = [
            CommentState::Request,
            CommentState::Offer,
            CommentState::Accept,
            CommentState::Reject,
            CommentState::Use,
            withdrawn(),
        ];
        for state in &others {
            assert_eq!(state.transition(&withdrawn_event()), None, "{state:?}");
        }
    }

    /// No comment's state moves on a comment birth, a revision, or a
    /// task event: births and revisions are content, task events
    /// belong to the task's table.
    #[test]
    fn births_revisions_and_task_events_move_no_comment_state() {
        let states = [
            CommentState::Note,
            CommentState::Request,
            CommentState::Offer,
            CommentState::Accept,
            CommentState::Reject,
            CommentState::Use,
            withdrawn(),
        ];
        let events = [
            comment_event(note()),
            comment_event(request()),
            comment_event(offer()),
            comment_event(accept()),
            comment_event(reject()),
            comment_event(use_of()),
            Event::CommentRevised {
                id: CommentId(RecordId(1)),
                body: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskClaimed {
                id: crate::TaskId(0),
                incarnation: None,
            },
            Event::TaskCancelled {
                id: crate::TaskId(0),
                explanation: Prose::new("filler".into()).unwrap(),
            },
        ];
        for state in &states {
            for event in &events {
                assert_eq!(state.transition(event), None, "table moved {state:?}");
            }
        }
    }

    /// The birth state mirrors the variant, one for one.
    #[test]
    fn the_birth_state_mirrors_the_variant() {
        for (kind, state) in [
            (note(), CommentState::Note),
            (request(), CommentState::Request),
            (offer(), CommentState::Offer),
            (accept(), CommentState::Accept),
            (reject(), CommentState::Reject),
            (use_of(), CommentState::Use),
        ] {
            assert_eq!(&CommentState::of(&kind), &state);
        }
    }

    /// Accept carries no prose; every other variant's prose is the
    /// revision door's swap surface, and the naming variants name
    /// exactly the offer they reference.
    #[test]
    fn prose_and_offer_surface_of_each_variant() {
        assert!(accept().prose().is_none());
        for kind in [note(), request(), offer(), reject(), use_of()] {
            assert!(kind.prose().is_some(), "{kind:?} carries prose");
        }
        assert_eq!(accept().offer(), Some(CommentId(RecordId(1))));
        assert_eq!(reject().offer(), Some(CommentId(RecordId(1))));
        assert_eq!(use_of().offer(), Some(CommentId(RecordId(1))));
        assert_eq!(note().offer(), None);
        assert_eq!(request().offer(), None);
        assert_eq!(offer().offer(), None);
    }
}
