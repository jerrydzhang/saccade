use serde::{Deserialize, Serialize};

use crate::objects::task::TaskId;
use crate::prose::Prose;
use crate::store::RecordId;

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct CommentId(pub RecordId);

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum Target {
    Task(TaskId),
    Comment(CommentId),
}

/// An address plus prose. The task is the tree's terminal node; thread
/// shape is `comment_thread`, a walk over these pointers. The actor is
/// cached from the envelope — the only provenance the world keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    pub(crate) target: Target,
    pub(crate) body: Prose,
    pub(crate) actor: String,
}
