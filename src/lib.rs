//! Saccade - issue tracker (idk what makes it special yet other than it's mine)

pub mod api;
pub mod attempts;
pub mod boards;
pub mod client;
pub mod completions;
pub mod db;
pub mod decide;
pub mod events;
pub mod harness;
pub mod herdr;
pub mod host;
pub mod host_contract;
pub mod launch;
pub mod markers;
pub mod objects;
pub mod paths;
pub mod pi_session;
pub mod placement;
pub mod refusals;
pub mod serve;
pub mod skill;
pub mod store;
pub mod types;
pub mod views;
pub use views::Target;
pub mod wire;
pub mod workspace;

use serde::{Deserialize, Serialize};

pub use decide::decide;
pub use events::{Command, Event, RecordRef};
pub use harness::Harness;
pub use launch::{
    Composition, FreshBirth, LaunchArm, LaunchFail, LaunchRequest, ResumeTarget, Submitted,
};
pub use objects::incarnation::{Incarnation, IncarnationContext, IncarnationId};
pub use objects::message::{Message, MessageContext, MessageId, MessageState};
pub use objects::task::{Owner, OwnerEntry, Task, TaskContext, TaskId, TaskState};
pub use objects::workspace::{
    CreatedWorkspace, CreatedWorkspaceContext, RootSpaceBinding, WorkspaceChange, WorkspaceId,
    WorkspaceStage, WorkspaceStageFact,
};
pub use store::{Context, Record, RecordId, Scope, Tier, World};
pub use types::actor::ActorName;
pub use types::pointers::{GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath};
pub use types::prose::Prose;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum Reject {
    // Task
    #[error("no such task")]
    InvalidTaskId,
    #[error("no such incarnation")]
    InvalidIncarnationId,
    #[error("an incarnation already carries this native session")]
    DuplicateNativeSession,
    #[error(
        "this native session id is already recorded with a different saved session; only the exact recorded association resolves"
    )]
    SavedSessionMismatch,
    #[error(
        "this incarnation already carries its saved session; the attach binds a launch-door birth's locator once"
    )]
    LocatorAlreadyBound,
    #[error("no such message for this recipient")]
    InvalidMessageId,
    // Permissions
    #[error("human-only act")]
    HumanOnly,
    // Workspace
    #[error("the task already holds a workspace")]
    WorkspaceAlreadyExists,
    #[error("the task has no workspace")]
    WorkspaceMissing,
    #[error("the tip rewinds the recorded checkpoint")]
    CheckpointRewind,
    #[error("no workspace was born at that record")]
    InvalidWorkspaceId,
    #[error("the workspace's stage does not admit this lifecycle fact")]
    WorkspaceStageMismatch,
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
            Reject::InvalidIncarnationId => "invalid_incarnation_id",
            Reject::DuplicateNativeSession => "duplicate_native_session",
            Reject::SavedSessionMismatch => "saved_session_mismatch",
            Reject::LocatorAlreadyBound => "locator_already_bound",
            Reject::InvalidMessageId => "invalid_message_id",
            Reject::HumanOnly => "human_only",
            Reject::WorkspaceAlreadyExists => "workspace_already_exists",
            Reject::WorkspaceMissing => "workspace_missing",
            Reject::CheckpointRewind => "checkpoint_rewind",
            Reject::InvalidWorkspaceId => "invalid_workspace_id",
            Reject::WorkspaceStageMismatch => "workspace_stage_mismatch",
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
            Reject::InvalidIncarnationId,
            Reject::DuplicateNativeSession,
            Reject::SavedSessionMismatch,
            Reject::LocatorAlreadyBound,
            Reject::InvalidMessageId,
            Reject::HumanOnly,
            Reject::WorkspaceAlreadyExists,
            Reject::WorkspaceMissing,
            Reject::CheckpointRewind,
            Reject::InvalidWorkspaceId,
            Reject::WorkspaceStageMismatch,
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
