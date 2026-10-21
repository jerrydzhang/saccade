use serde::{Deserialize, Serialize};

use crate::objects::task::TaskId;
use crate::store::RecordId;
use crate::types::pointers::{NativeSessionId, SessionLocator, WorktreePath};

/// The record position of the incarnation's birth is its identity.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct IncarnationId(pub RecordId);

/// An incarnation identifies a native agent session — never a task, a
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
    /// A registered native session: the durable native id and the
    /// saved-session locator ordinary resume validates.
    Native {
        native_id: NativeSessionId,
        saved_session: SessionLocator,
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

    /// The locator a native incarnation resumes from; historical
    /// establishments carry none.
    pub fn saved_session(&self) -> Option<&SessionLocator> {
        match self {
            Incarnation::Native { saved_session, .. } => Some(saved_session),
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
