use crate::RecordId;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

/// The task's working environment, born once by the establishment
/// record: the facts that record established, never a promise that
/// a directory or process still exists - existence checks live in
/// effects, not the fold. The birth record's position is the shared
/// workspace/incarnation identity.
#[derive(Clone, Debug, PartialEq)]
pub struct Workspace {
    /// The immutable commit the workspace was cut from
    pub base: GitCommit,
    /// The branch preserving advancement between checkpoints
    pub branch: GitBranch,
    /// The latest recorded clean head; born at base
    pub checkpoint: GitCommit,
    /// The workspace's absolute address
    pub worktree: WorktreePath,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceContext {
    pub workspace: Workspace,
    /// The establishment record's position: the shared
    /// workspace/incarnation identity
    pub birth: RecordId,
    /// Every head the record has named, base first; the explicit
    /// checkpoint door returns to none of them
    pub heads: Vec<GitCommit>,
}
