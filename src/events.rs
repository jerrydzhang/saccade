use crate::objects::comment::{CommentId, CommentKind, Target};
use crate::objects::incarnation::IncarnationId;
use crate::objects::task::TaskId;
use crate::types::artifact::Artifact;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};
use crate::types::prose::Prose;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    // Task events
    TaskCreated {
        name: Prose,
        parent_id: Option<TaskId>,
    },
    /// A transfer of whole-task responsibility: never a state
    /// transition, never a permission grant. The claim history
    /// accumulates on the context.
    TaskClaimed {
        id: TaskId,
        /// The record-derived incarnation claiming, when the claim
        /// rides one; absent means the named actor claims directly
        incarnation: Option<IncarnationId>,
    },
    /// The human's stop of an unfinished undertaking, with the
    /// explanation that survives in the record.
    TaskCancelled { id: TaskId, explanation: Prose },
    /// The human's explicit resumption of a cancelled undertaking,
    /// with the explanation of why it is pursued again.
    TaskReopened { id: TaskId, explanation: Prose },
    // Comment events
    /// The payload is its variant, body included; Accept carries none.
    Commented { target: Target, kind: CommentKind },
    /// A presentation repair, not a transition: the fold swaps the
    /// variant's prose and points at the latest revision record — the
    /// original bytes stay in their birth event, references never change.
    CommentRevised { id: CommentId, body: Prose },
    /// The standing-error repair: the deposit's implicit claim — this
    /// is reached, this earns residency — was false. The body leaves
    /// the fold's presentation, a tombstone stays; the log is
    /// untouched. Note deposits only.
    CommentWithdrawn { id: CommentId, note: Prose },
    /// A thread holding an artifact: name and content identity, the
    /// bytes stay in the store. Authored like a comment, any tier.
    ArtifactAdded { root: TaskId, artifact: Artifact },
    // Workspace events: the establishment is the machinery verb,
    // System-authored by role; the checkpoint is the exception any
    // tier may record
    /// The workspace-establishment birth: task reference, canonical
    /// base, branch, and the workspace's absolute path. The record's
    /// position is the shared workspace/incarnation identity.
    WorkspaceEstablished {
        task_id: TaskId,
        base: GitCommit,
        branch: GitBranch,
        worktree: WorktreePath,
    },
    TaskWorkspaceCheckpointed {
        task_id: TaskId,
        checkpoint: GitCommit,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    // Task commands
    CreateTask {
        name: Prose,
        parent_id: Option<TaskId>,
    },
    ClaimTask {
        id: TaskId,
        incarnation: Option<IncarnationId>,
    },
    CancelTask {
        id: TaskId,
        explanation: Prose,
    },
    ReopenTask {
        id: TaskId,
        explanation: Prose,
    },
    // Comment commands
    Comment {
        target: Target,
        kind: CommentKind,
    },
    ReviseComment {
        id: CommentId,
        body: Prose,
    },
    WithdrawComment {
        id: CommentId,
        note: Prose,
    },
    Artifact {
        root: TaskId,
        artifact: Artifact,
    },
    // Machinery verbs: System authorship comes from the role, never input
    EstablishWorkspace {
        task_id: TaskId,
        base: GitCommit,
        branch: GitBranch,
        worktree: WorktreePath,
    },
    CheckpointWorkspace {
        task_id: TaskId,
        checkpoint: GitCommit,
    },
}
