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
pub mod rpc;
pub mod runner;
pub mod serve;
pub mod store;
pub mod supervisor;
pub mod types;
pub mod views;
pub mod web;
pub mod wire;

use serde::{Deserialize, Serialize};

pub use decide::decide;
pub use events::{Command, Event};
pub use objects::comment::{Comment, CommentId, CommentKind, Target};
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
    #[error("not the task's birth attribution")]
    NotBirthAttribution,
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
    #[error("the tip rewinds the recorded checkpoint")]
    CheckpointRewind,
    #[error("the steer is not standing intent")]
    SteerNotStanding,
    #[error("the task runs no incarnation")]
    NoActiveIncarnation,
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
            Reject::InvalidProposalId => "invalid_proposal_id",
            Reject::ProposalAlreadyOpen => "proposal_already_open",
            Reject::InvalidCommentId => "invalid_comment_id",
            Reject::NotBirthAttribution => "not_birth_attribution",
            Reject::HumanOnly => "human_only",
            Reject::NotClaimHolder => "not_claim_holder",
            Reject::InvalidIncarnationId => "invalid_incarnation_id",
            Reject::IncarnationAlreadyActive => "incarnation_already_active",
            Reject::DemandNotOnTask => "demand_not_on_task",
            Reject::WorkspaceAlreadyExists => "workspace_already_exists",
            Reject::WorkspaceMissing => "workspace_missing",
            Reject::WorktreeAlreadyPresent => "worktree_already_present",
            Reject::CheckpointRewind => "checkpoint_rewind",
            Reject::SteerNotStanding => "steer_not_standing",
            Reject::NoActiveIncarnation => "no_active_incarnation",
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
            Reject::InvalidProposalId,
            Reject::ProposalAlreadyOpen,
            Reject::InvalidCommentId,
            Reject::NotBirthAttribution,
            Reject::HumanOnly,
            Reject::NotClaimHolder,
            Reject::InvalidIncarnationId,
            Reject::IncarnationAlreadyActive,
            Reject::DemandNotOnTask,
            Reject::WorkspaceAlreadyExists,
            Reject::WorkspaceMissing,
            Reject::WorktreeAlreadyPresent,
            Reject::CheckpointRewind,
            Reject::SteerNotStanding,
            Reject::NoActiveIncarnation,
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
