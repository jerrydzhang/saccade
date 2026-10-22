use serde::{Deserialize, Serialize};

use crate::RecordId;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

/// The record position of a created workspace's birth is its identity.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct WorkspaceId(pub RecordId);

/// A workspace born by its own creation record: independent of any task
/// or incarnation, its identity the birth record's position. The facts
/// are what the provisioner established, never caller-claimed values.
#[derive(Clone, Debug, PartialEq)]
pub struct CreatedWorkspace {
    /// The immutable commit the workspace was cut from
    pub base: GitCommit,
    /// The branch provisioning created
    pub branch: GitBranch,
    /// The workspace's absolute address
    pub worktree: WorktreePath,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreatedWorkspaceContext {
    pub workspace: CreatedWorkspace,
    /// Event time of the birth record
    pub born_at: u64,
}

/// The latest confirmed saved-directory change of a native incarnation:
/// the selected directory, the record that confirmed it, and its event
/// time. A machinery fact riding the incarnation, never a lifecycle
/// stage; earlier changes stay in the log.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceChange {
    pub worktree: WorktreePath,
    pub record: RecordId,
    pub at: u64,
}

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
