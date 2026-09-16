use serde::{Deserialize, Serialize};

use crate::objects::task::TaskId;
use crate::store::RecordId;
use crate::types::prose::Prose;

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct CommentId(pub RecordId);

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum Target {
    Task(TaskId),
    Comment(CommentId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comment {
    pub(crate) target: Target,
    pub(crate) body: Prose,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommentContext {
    pub comment: Comment,
    pub actor: String,
}
