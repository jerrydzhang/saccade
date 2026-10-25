use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::incarnation::IncarnationId;
use crate::objects::task::TaskId;
use crate::store::{RecordId, Tier};
use crate::types::actor::ActorName;
use crate::types::prose::Prose;

/// The message's own stable birth reference; the human-facing m-N is
/// its position in message birth order, derived by the fold.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct MessageId(pub RecordId);

/// A durable addressed communication between incarnations. Never a
/// comment variant, never a task-thread publication: the optional
/// task rides as context without publication, and no home task
/// exists.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    /// The sending session's registered identity
    pub from: IncarnationId,
    pub to: IncarnationId,
    /// The message this one answers, when the sender names one
    pub reply_to: Option<MessageId>,
    /// Task context riding the exchange, without publication
    pub task: Option<TaskId>,
    pub body: Prose,
}

/// Ordinary delivery bookkeeping: pending until the recipient's
/// integration hands it to its native session. Never a claim of
/// reading, agreement, completion, or task acceptance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageState {
    Pending,
    Submitted,
}

impl MessageState {
    /// The message state machine table, all event-named transitions
    /// must go through this: submission is the pending mail's door
    /// alone, and submission never repeats.
    pub fn transition(&self, event: &Event) -> Option<MessageState> {
        match (self, event) {
            (MessageState::Pending, Event::MessageSubmitted { .. }) => {
                Some(MessageState::Submitted)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MessageContext {
    pub message: Message,
    pub state: MessageState,
    /// The sender of record
    pub actor: ActorName,
    pub tier: Tier,
    /// Event time of the birth record
    pub born_at: u64,
}

#[cfg(test)]
mod tables {
    use super::*;
    use crate::types::prose::Prose;

    fn submitted_event() -> Event {
        Event::MessageSubmitted {
            id: MessageId(RecordId(1)),
            by: IncarnationId(RecordId(0)),
        }
    }

    fn any_other_events() -> Vec<Event> {
        vec![
            Event::MessageSent {
                from: IncarnationId(RecordId(0)),
                to: IncarnationId(RecordId(1)),
                reply_to: None,
                task: None,
                body: Prose::new("filler".into()).unwrap(),
            },
            Event::IncarnationRegistered {
                native_id: crate::NativeSessionId::new("a1b2c3d4".into()).unwrap(),
                saved_session: crate::SessionLocator::new("/sessions/a1b2c3d4.jsonl".into())
                    .unwrap(),
            },
            Event::TaskClaimed {
                id: TaskId(0),
                incarnation: None,
            },
            Event::CommentRevised {
                id: crate::CommentId(RecordId(0)),
                title: None,
                body: Some(Prose::new("filler".into()).unwrap()),
            },
        ]
    }

    /// Submission is the pending mail's door alone: submitted mail
    /// holds, and no other event moves a message.
    #[test]
    fn submission_is_the_pending_doors_single_step() {
        assert_eq!(
            MessageState::Pending.transition(&submitted_event()),
            Some(MessageState::Submitted)
        );
        assert_eq!(MessageState::Submitted.transition(&submitted_event()), None);
        for event in any_other_events() {
            assert_eq!(MessageState::Pending.transition(&event), None);
            assert_eq!(MessageState::Submitted.transition(&event), None);
        }
    }
}
