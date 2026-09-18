use crate::objects::comment::{Addressee, CommentId, Target};
use crate::objects::incarnation::IncarnationId;
use crate::objects::proposal::{ProposalAction, ProposalId};
use crate::objects::task::TaskId;
use crate::store::RecordId;
use crate::types::actor::ActorName;
use crate::types::failure::FailureEvidence;
use crate::types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
use crate::types::prose::Prose;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    // Task Events
    TaskCreated {
        name: Prose,
        parent_id: Option<TaskId>,
    },
    TaskClaimed {
        id: TaskId,
    },
    TaskDone {
        id: TaskId,
        receipt: Prose,
    },
    TaskDropped {
        id: TaskId,
        note: Prose,
    },
    TaskReleased {
        id: TaskId,
        note: Prose,
    },
    // Proposal Events
    ProposalCreated {
        // No birth event carries its id: a task's id is its push position in
        // the fold, a proposal's id is the record position of its birth
        name: Prose,
        action: ProposalAction,
    },
    ProposalWithdrawn {
        id: ProposalId,
        note: Prose,
    },
    ProposalRejected {
        id: ProposalId,
        note: Prose,
    },
    ProposalAccepted {
        id: ProposalId,
    },
    // Comment Events
    Commented {
        target: Target,
        body: Prose,
        #[serde(skip_serializing_if = "Option::is_none")]
        addressee: Option<Addressee>,
    },
    // Incarnation events: machinery verbs, System-authored by role
    IncarnationBound {
        task_id: TaskId,
        response_target: CommentId,
        trigger: RecordId,
        actor: ActorName,
        session: SessionPointer,
    },
    IncarnationPromptAccepted {
        id: IncarnationId,
    },
    IncarnationPromptRejected {
        id: IncarnationId,
        evidence: FailureEvidence,
    },
    IncarnationSettled {
        id: IncarnationId,
    },
    RecordProducedBy {
        record_id: RecordId,
        incarnation_id: IncarnationId,
    },
    // Workspace events: machinery verbs, System-authored by role
    TaskWorkspaceCreated {
        task_id: TaskId,
        base: GitCommit,
        branch: GitBranch,
    },
    TaskWorktreeCreated {
        task_id: TaskId,
        worktree: WorktreePath,
    },
    TaskWorkspaceCheckpointed {
        task_id: TaskId,
        checkpoint: GitCommit,
    },
}

#[derive(Debug, PartialEq)]
pub enum Command {
    // Task commands
    CreateTask {
        name: Prose,
        parent_id: Option<TaskId>,
    },
    ClaimTask {
        id: TaskId,
    },
    CompleteTask {
        id: TaskId,
        receipt: Prose,
    },
    DropTask {
        id: TaskId,
        note: Prose,
    },
    ReleaseTask {
        id: TaskId,
        note: Prose,
    },
    // Proposal commands
    CreateProposal {
        // See note above about why proposals don't have an id at command time
        name: Prose,
        action: ProposalAction,
    },
    WithdrawProposal {
        id: ProposalId,
        note: Prose,
    },
    RejectProposal {
        id: ProposalId,
        note: Prose,
    },
    AcceptProposal {
        id: ProposalId,
    },
    // Comment commands
    Comment {
        target: Target,
        body: Prose,
        addressee: Option<Addressee>,
    },
    // Machinery verbs: the executor side acts through the same pipeline
    BindIncarnation {
        task_id: TaskId,
        response_target: CommentId,
        trigger: RecordId,
        actor: ActorName,
        session: SessionPointer,
    },
    AcceptPrompt {
        id: IncarnationId,
    },
    RejectPrompt {
        id: IncarnationId,
        evidence: FailureEvidence,
    },
    SettleIncarnation {
        id: IncarnationId,
    },
    MarkRecord {
        record_id: RecordId,
        incarnation_id: IncarnationId,
    },
    // Machinery verbs: workspace records follow provisioning and checkpoints
    CreateWorkspace {
        task_id: TaskId,
        base: GitCommit,
        branch: GitBranch,
    },
    CreateWorktree {
        task_id: TaskId,
        worktree: WorktreePath,
    },
    CheckpointWorkspace {
        task_id: TaskId,
        checkpoint: GitCommit,
    },
}
