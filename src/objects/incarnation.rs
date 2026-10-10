use serde::{Deserialize, Serialize};

use crate::objects::task::TaskId;
use crate::store::RecordId;
use crate::types::pointers::WorktreePath;

/// The record position of the incarnation's birth is its identity.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct IncarnationId(pub RecordId);

/// A task-bound worker identity: record-derived id, one task, one
/// workspace address. No lifecycle, no session pointer — attachment
/// and its effects are the native-start slice's doors, so this slice
/// births none and the registry reads empty until that slice lands.
#[derive(Clone, Debug, PartialEq)]
pub struct Incarnation {
    pub task: TaskId,
    pub workspace: WorktreePath,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IncarnationContext {
    pub incarnation: Incarnation,
    /// Event time of the birth record
    pub born_at: u64,
}
