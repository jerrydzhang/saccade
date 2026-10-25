use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::task::TaskId;
use crate::store::{RecordId, Tier};
use crate::types::actor::ActorName;
use crate::types::prose::Prose;

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct CommentId(pub RecordId);

/// The wait door's read-side addressing: a task's records, or one
/// comment's. Comments themselves carry a home task and a reference
/// mapping, never a positional address.
#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Task(TaskId),
    Comment(CommentId),
}

/// A use's selection of a conclusion: the offer named and the grounds
/// for selecting it. A birth may carry one already bound - the shape a
/// historical log migrates into - every later selection is its own
/// binding record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UseBinding {
    pub offer: CommentId,
    pub grounds: Prose,
}

/// The payload a comment carries. Every variant lives on one explicit
/// home task; an Offer addresses its whole task; Accept and Reject
/// name the offer they decide; a Use declares its fixed source task.
/// A Note and an Offer are born with their own authored title - the
/// concise whole-contribution description - beside the body; every
/// other variant is untitled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommentKind {
    Note {
        title: Prose,
        body: Prose,
    },
    /// The whole-task closing account: broadcast, never a point reply
    Offer {
        title: Prose,
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
    /// The consuming task's declared reliance on a fixed source task;
    /// the conclusion it rests on is a separate explicit binding
    Use {
        source: TaskId,
        reason: Prose,
        /// The initial binding a historical log carries at birth; a
        /// live birth is prospective and leaves it absent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bound: Option<UseBinding>,
    },
}

impl CommentKind {
    /// None for the bodyless Accept, which the revision door refuses.
    pub fn prose(&self) -> Option<&Prose> {
        match self {
            CommentKind::Note { body, .. } | CommentKind::Offer { body, .. } => Some(body),
            CommentKind::Accept { .. } => None,
            CommentKind::Reject { grounds, .. } => Some(grounds),
            CommentKind::Use { reason, .. } => Some(reason),
        }
    }

    /// The authored title, on the deposit variants alone: notes and
    /// offers carry one at birth, every other variant none.
    pub fn title(&self) -> Option<&Prose> {
        match self {
            CommentKind::Note { title, .. } | CommentKind::Offer { title, .. } => Some(title),
            _ => None,
        }
    }

    /// The offer a decision names; a use names its source task, not an
    /// offer - its selection rides the binding records.
    pub fn offer(&self) -> Option<CommentId> {
        match self {
            CommentKind::Accept { offer } | CommentKind::Reject { offer, .. } => Some(*offer),
            _ => None,
        }
    }

    /// The source task a use declares.
    pub fn source(&self) -> Option<TaskId> {
        match self {
            CommentKind::Use { source, .. } => Some(*source),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    /// The one task this comment lives on, explicit at birth
    pub home: TaskId,
    pub kind: CommentKind,
    pub state: CommentState,
}

/// A comment's own lifecycle, born mirroring its variant; the note
/// carries the withdrawal door, the use carries the binding door.
#[derive(Clone, Debug, PartialEq)]
pub enum CommentState {
    Note,
    Offer,
    Accept,
    Reject,
    Use,
    /// A use's current selection, the payload of the last binding:
    /// the offer named and the grounds for selecting it. Earlier
    /// selections stay in the log; rebinding replaces this one.
    UseBound(UseBinding),
    /// The standing-error repair, terminal: the deposit's body left
    /// the fold's presentation, the tombstone holds its place.
    Withdrawn(Prose),
}

impl CommentState {
    /// The comment state machine table, all event-named transitions
    /// must go through this: withdrawal is the note deposits' door
    /// alone, binding is the use's; every other variant holds.
    pub fn transition(&self, event: &Event) -> Option<CommentState> {
        match (self, event) {
            (CommentState::Note, Event::CommentWithdrawn { note, .. }) => {
                Some(CommentState::Withdrawn(note.clone()))
            }
            (
                CommentState::Use | CommentState::UseBound(_),
                Event::UseBound { offer, grounds, .. },
            ) => Some(CommentState::UseBound(UseBinding {
                offer: *offer,
                grounds: grounds.clone(),
            })),
            _ => None,
        }
    }

    /// The offer a bound use's current selection names.
    pub fn payload_offer(&self) -> Option<CommentId> {
        match self {
            CommentState::UseBound(payload) => Some(payload.offer),
            _ => None,
        }
    }

    pub fn of(kind: &CommentKind) -> CommentState {
        match kind {
            CommentKind::Note { .. } => CommentState::Note,
            CommentKind::Offer { .. } => CommentState::Offer,
            CommentKind::Accept { .. } => CommentState::Accept,
            CommentKind::Reject { .. } => CommentState::Reject,
            CommentKind::Use { bound, .. } => match bound {
                None => CommentState::Use,
                Some(binding) => CommentState::UseBound(binding.clone()),
            },
        }
    }
}

/// The latest revision of a folded comment's prose: the record that
/// swapped it in and the actor who did - disclosure is a fold fact,
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

/// A use's last binding's attribution: the record that selected and
/// the actor who did. The selection itself - offer and grounds - is
/// the state's payload.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    pub record: RecordId,
    pub actor: ActorName,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommentContext {
    pub comment: Comment,
    pub actor: ActorName,
    pub tier: Tier,
    pub scope: Option<crate::store::Scope>,
    /// Event time of the comment's birth record
    pub born_at: u64,
    /// The latest revision of the prose, when it was ever revised -
    /// a pointer with its reviser, never the payload it swapped in
    pub revised: Option<Revision>,
    /// The standing-error repair's pointer, when the deposit was
    /// withdrawn - beside the revised pointer it mirrors
    pub withdrawn: Option<Withdrawal>,
    /// The use's current selection, when a binding has landed; a
    /// rebinding replaces it, the log keeps every earlier one
    pub binding: Option<Binding>,
    /// The citation mapping as born: marker k names references[k-1];
    /// a revision never retargets it
    pub references: Vec<RecordId>,
}

#[cfg(test)]
mod tables {
    use super::*;
    use crate::types::prose::Prose;

    fn comment_event(kind: CommentKind) -> Event {
        Event::Commented {
            home: crate::TaskId(0),
            kind,
            references: Vec::new(),
        }
    }

    fn withdrawn_event() -> Event {
        Event::CommentWithdrawn {
            id: CommentId(RecordId(1)),
            note: Prose::new("parked in the wrong place".into()).unwrap(),
        }
    }

    fn bind_event() -> Event {
        Event::UseBound {
            id: CommentId(RecordId(1)),
            offer: CommentId(RecordId(5)),
            grounds: Prose::new("the selected account holds".into()).unwrap(),
        }
    }

    fn bound_payload() -> UseBinding {
        UseBinding {
            offer: CommentId(RecordId(5)),
            grounds: Prose::new("the selected account holds".into()).unwrap(),
        }
    }

    fn note() -> CommentKind {
        CommentKind::Note {
            title: Prose::new("a filler title".into()).unwrap(),
            body: Prose::new("filler".into()).unwrap(),
        }
    }

    fn offer() -> CommentKind {
        CommentKind::Offer {
            title: Prose::new("a filler title".into()).unwrap(),
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
            source: crate::TaskId(1),
            reason: Prose::new("filler".into()).unwrap(),
            bound: None,
        }
    }

    fn born_bound_use() -> CommentKind {
        CommentKind::Use {
            source: crate::TaskId(1),
            reason: Prose::new("filler".into()).unwrap(),
            bound: Some(UseBinding {
                offer: CommentId(RecordId(5)),
                grounds: Prose::new("filler".into()).unwrap(),
            }),
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
            CommentState::Offer,
            CommentState::Accept,
            CommentState::Reject,
            CommentState::Use,
            CommentState::UseBound(bound_payload()),
            withdrawn(),
        ];
        for state in &others {
            assert_eq!(state.transition(&withdrawn_event()), None, "{state:?}");
        }
    }

    /// Binding is the use's door alone - prospective or already bound,
    /// a binding selects again; every other variant holds.
    #[test]
    fn binding_is_the_use_door_alone() {
        let selected = || CommentState::UseBound(bound_payload());
        assert_eq!(
            CommentState::Use.transition(&bind_event()),
            Some(selected())
        );
        assert_eq!(
            CommentState::UseBound(UseBinding {
                offer: CommentId(RecordId(3)),
                grounds: Prose::new("an earlier selection".into()).unwrap(),
            })
            .transition(&bind_event()),
            Some(selected())
        );
        let others = [
            CommentState::Note,
            CommentState::Offer,
            CommentState::Accept,
            CommentState::Reject,
            withdrawn(),
        ];
        for state in &others {
            assert_eq!(state.transition(&bind_event()), None, "{state:?}");
        }
    }

    /// No comment's state moves on a comment birth, a revision, or a
    /// task event: births and revisions are content, task events
    /// belong to the task's table.
    #[test]
    fn births_revisions_and_task_events_move_no_comment_state() {
        let states = [
            CommentState::Note,
            CommentState::Offer,
            CommentState::Accept,
            CommentState::Reject,
            CommentState::Use,
            CommentState::UseBound(bound_payload()),
            withdrawn(),
        ];
        let events = [
            comment_event(note()),
            comment_event(offer()),
            comment_event(accept()),
            comment_event(reject()),
            comment_event(use_of()),
            Event::CommentRevised {
                id: CommentId(RecordId(1)),
                title: Some(Prose::new("a filler title".into()).unwrap()),
                body: Some(Prose::new("filler".into()).unwrap()),
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

    /// The birth state mirrors the variant, one for one - including
    /// the born-bound use a historical log carries.
    #[test]
    fn the_birth_state_mirrors_the_variant() {
        for (kind, state) in [
            (note(), CommentState::Note),
            (offer(), CommentState::Offer),
            (accept(), CommentState::Accept),
            (reject(), CommentState::Reject),
            (use_of(), CommentState::Use),
            (
                born_bound_use(),
                CommentState::UseBound(UseBinding {
                    offer: CommentId(RecordId(5)),
                    grounds: Prose::new("filler".into()).unwrap(),
                }),
            ),
        ] {
            assert_eq!(&CommentState::of(&kind), &state);
        }
    }

    /// Accept carries no prose; every other variant's prose is the
    /// revision door's swap surface, the decisions name exactly the
    /// offer they reference, a use names its source task, and the
    /// authored title rides the deposit variants alone.
    #[test]
    fn prose_and_offer_surface_of_each_variant() {
        assert!(accept().prose().is_none());
        for kind in [note(), offer(), reject(), use_of()] {
            assert!(kind.prose().is_some(), "{kind:?} carries prose");
        }
        assert_eq!(accept().offer(), Some(CommentId(RecordId(1))));
        assert_eq!(reject().offer(), Some(CommentId(RecordId(1))));
        assert_eq!(use_of().offer(), None);
        assert_eq!(note().offer(), None);
        assert_eq!(offer().offer(), None);
        assert_eq!(use_of().source(), Some(crate::TaskId(1)));
        assert_eq!(note().source(), None);
        assert!(note().title().is_some());
        assert!(offer().title().is_some());
        for kind in [accept(), reject(), use_of()] {
            assert!(kind.title().is_none(), "{kind:?} carries no title");
        }
    }
}
