use crate::objects::comment::Target;
use crate::objects::proposal::{ProposalAction, ProposalId};
use crate::objects::task::TaskId;
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
    },
}
