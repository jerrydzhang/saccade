use serde::{Deserialize, Serialize};

use crate::objects::task::TaskId;
use crate::store::RecordId;
use crate::types::pointers::{NativeSessionId, SessionLocator, WorktreePath};

/// The record position of the incarnation's birth is its identity.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct IncarnationId(pub RecordId);

/// An incarnation identifies a native agent session - never a task, a
/// workspace, or a running process. Two births carry one: the
/// workspace establishment births a historical attribution identity,
/// the registration births a native session identity. A native
/// identity has no lifecycle: turns, compaction, resume, and
/// workspace handoff retain it; fresh sessions and forks create
/// another. Historical existence asserts no current execution.
#[derive(Clone, Debug, PartialEq)]
pub enum Incarnation {
    /// The shared workspace-establishment birth: attribution only,
    /// never a binding to a real native session.
    Established {
        task: TaskId,
        workspace: WorktreePath,
    },
    /// A registered native session: the durable native id, the
    /// saved-session locator ordinary resume validates (bound by the
    /// attach when the launch door birthed the identity before pi's
    /// first save), the opaque host address the association carries -
    /// submission targets it, probes read it - and the address kind:
    /// launched (a terminal that dies with the session, so terminal
    /// liveness reads as session liveness) or adopted (the caller's
    /// own pane, which outlives the session). The harness the session
    /// runs under rides the birth; native identity is interpreted
    /// within it, and births before the field existed fold as pi.
    Native {
        native_id: NativeSessionId,
        saved_session: Option<SessionLocator>,
        address: Option<String>,
        adopted: bool,
        harness: crate::harness::Harness,
    },
}

impl Incarnation {
    /// The native identity a message sender must resolve to.
    pub fn native_id(&self) -> Option<&NativeSessionId> {
        match self {
            Incarnation::Native { native_id, .. } => Some(native_id),
            Incarnation::Established { .. } => None,
        }
    }

    /// The locator a native incarnation resumes from; unbound while a
    /// launch-door birth precedes pi's first save, and historical
    /// establishments carry none.
    pub fn saved_session(&self) -> Option<&SessionLocator> {
        match self {
            Incarnation::Native { saved_session, .. } => saved_session.as_ref(),
            Incarnation::Established { .. } => None,
        }
    }

    /// The opaque host address the association carries, when the
    /// launch recorded one; pre-contract births carry none.
    pub fn address(&self) -> Option<&str> {
        match self {
            Incarnation::Native { address, .. } => address.as_deref(),
            Incarnation::Established { .. } => None,
        }
    }

    /// Whether the carried address is the caller's own pane (adopted):
    /// such a pane outlives the session, so terminal liveness is never
    /// session liveness for it.
    pub fn adopted(&self) -> bool {
        match self {
            Incarnation::Native { adopted, .. } => *adopted,
            Incarnation::Established { .. } => false,
        }
    }

    /// The harness the native session runs under; historical
    /// establishments carry none.
    pub fn harness(&self) -> Option<crate::harness::Harness> {
        match self {
            Incarnation::Native { harness, .. } => Some(*harness),
            Incarnation::Established { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IncarnationContext {
    pub incarnation: Incarnation,
    /// Event time of the birth record
    pub born_at: u64,
}
