//! Saccade - issue tracker (idk what makes it special yet other than it's mine)

pub mod api;
pub mod client;
pub mod db;
pub mod decide;
pub mod events;
pub mod objects;
pub mod paths;
pub mod runner;
pub mod store;
pub mod types;
pub mod views;
pub mod wire;

use serde::{Deserialize, Serialize};

pub use decide::decide;
pub use events::{Command, Event};
pub use objects::comment::{Addressee, Comment, CommentId, Target};
pub use objects::proposal::{Proposal, ProposalAction, ProposalId, ProposalState};
pub use objects::task::{Task, TaskId, TaskState};
pub use store::{Context, Record, RecordId, Tier, World};
pub use types::actor::ActorName;
pub use types::failure::{FailureCode, FailureEvidence};
pub use types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
pub use types::prose::Prose;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum Reject {
    // Task
    #[error("no such task")]
    InvalidTaskId,
    #[error("no such parent task")]
    InvalidParentTaskId,
    // Proposal
    #[error("no open proposal with that id")]
    InvalidProposalId,
    #[error("that task already holds an open judgment proposal")]
    ProposalAlreadyOpen,
    #[error("no comment with that id")]
    InvalidCommentId,
    // Permissions
    #[error("human-only act")]
    HumanOnly,
    #[error("not the claim holder")]
    NotClaimHolder,
    #[error("no such incarnation")]
    InvalidIncarnationId,
    #[error("the task's run slot is taken")]
    IncarnationAlreadyActive,
    #[error("the demand lives on another task")]
    DemandNotOnTask,
    #[error("the task already holds a workspace")]
    WorkspaceAlreadyExists,
    #[error("the task has no workspace")]
    WorkspaceMissing,
    #[error("the worktree is already present")]
    WorktreeAlreadyPresent,
    // Misc
    #[error("invalid actor name")]
    InvalidActor,
    #[error("invalid state transition")]
    InvalidStateTransition,
    #[error("words are required")]
    ReasonRequired,
}
