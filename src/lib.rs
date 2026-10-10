//! Saccade - issue tracker (idk what makes it special yet other than it's mine)

pub mod api;
pub mod attempts;
pub mod client;
pub mod db;
pub mod decide;
pub mod events;
pub mod objects;
pub mod paths;
pub mod refusals;
pub mod serve;
pub mod skill;
pub mod store;
pub mod types;
pub mod views;
pub mod web;
pub mod wire;

use serde::{Deserialize, Serialize};

pub use decide::decide;
pub use events::{Command, Event};
pub use objects::comment::{Comment, CommentId, CommentKind, CommentState, Target};
pub use objects::incarnation::{Incarnation, IncarnationContext, IncarnationId};
pub use objects::task::{Owner, OwnerEntry, Task, TaskContext, TaskId, TaskState};
pub use store::{Context, Record, RecordId, Scope, Tier, World};
pub use types::actor::ActorName;
pub use types::artifact::{Artifact, ContentHash};
pub use types::pointers::{GitBranch, GitCommit, WorktreePath};
pub use types::prose::Prose;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum Reject {
    // Task
    #[error("no such task")]
    InvalidTaskId,
    #[error("no such parent task")]
    InvalidParentTaskId,
    #[error("no comment with that id")]
    InvalidCommentId,
    #[error("the referenced comment is not an offer")]
    NotAnOffer,
    #[error("that offer is not the task's current one")]
    NotCurrentOffer,
    #[error(
        "the declared task scope covers this task: a task-bound producer cannot accept its own conclusion"
    )]
    SelfAcceptance,
    #[error("no such incarnation")]
    InvalidIncarnationId,
    // Permissions
    #[error("human-only act")]
    HumanOnly,
    // Workspace
    #[error("the task already holds a workspace")]
    WorkspaceAlreadyExists,
    #[error("the task has no workspace")]
    WorkspaceMissing,
    #[error("the worktree is already present")]
    WorktreeAlreadyPresent,
    #[error("the tip rewinds the recorded checkpoint")]
    CheckpointRewind,
    // Misc
    #[error("invalid actor name")]
    InvalidActor,
    #[error("invalid state transition")]
    InvalidStateTransition,
    #[error("words are required")]
    ReasonRequired,
}

impl Reject {
    /// The stable refusal code every face cites: the wire's error body,
    /// the CLI's exit line, and the attempts log's refusal lines.
    pub fn code(&self) -> &'static str {
        match self {
            Reject::InvalidTaskId => "invalid_task_id",
            Reject::InvalidParentTaskId => "invalid_parent_task_id",
            Reject::InvalidCommentId => "invalid_comment_id",
            Reject::NotAnOffer => "not_an_offer",
            Reject::NotCurrentOffer => "not_current_offer",
            Reject::SelfAcceptance => "self_acceptance",
            Reject::InvalidIncarnationId => "invalid_incarnation_id",
            Reject::HumanOnly => "human_only",
            Reject::WorkspaceAlreadyExists => "workspace_already_exists",
            Reject::WorkspaceMissing => "workspace_missing",
            Reject::WorktreeAlreadyPresent => "worktree_already_present",
            Reject::CheckpointRewind => "checkpoint_rewind",
            Reject::InvalidActor => "invalid_actor",
            Reject::InvalidStateTransition => "invalid_state_transition",
            Reject::ReasonRequired => "reason_required",
        }
    }
}

#[cfg(test)]
mod test {
    use super::Reject;

    /// The code table is exhaustive and matches the wire's serde names,
    /// so the three faces that cite codes cannot drift apart.
    #[test]
    fn refusal_codes_match_the_wire_names() {
        let every = [
            Reject::InvalidTaskId,
            Reject::InvalidParentTaskId,
            Reject::InvalidCommentId,
            Reject::NotAnOffer,
            Reject::NotCurrentOffer,
            Reject::SelfAcceptance,
            Reject::InvalidIncarnationId,
            Reject::HumanOnly,
            Reject::WorkspaceAlreadyExists,
            Reject::WorkspaceMissing,
            Reject::WorktreeAlreadyPresent,
            Reject::CheckpointRewind,
            Reject::InvalidActor,
            Reject::InvalidStateTransition,
            Reject::ReasonRequired,
        ];
        for reject in every {
            let wire = serde_json::to_value(&reject).unwrap();
            assert_eq!(
                wire.as_str().unwrap(),
                reject.code(),
                "the code and the wire name disagree for {wire}"
            );
        }
    }
}
