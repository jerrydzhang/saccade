use crate::objects::proposal::{ProposalAction, ProposalId};
use crate::objects::task::{Receipt, TaskId};

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    // Task Events
    TaskCreated {
        id: TaskId,
        name: String,
        parent_id: Option<TaskId>,
    },
    TaskClaimed {
        id: TaskId,
    },
    TaskDone {
        id: TaskId,
        receipt: Receipt,
    },
    TaskDropped {
        id: TaskId,
        note: String,
    },
    TaskReleased {
        id: TaskId,
        note: String,
    },
    // Proposal Events
    ProposalCreated {
        // Proposal's are not created with an id at event time since the id is directly the record
        // id of the proposal, this means unlike a taskid the id is directly derivable from the
        // single record where as a task would require counting up all the TaskCreated events to
        // derive the id
        name: String,
        action: ProposalAction,
    },
    ProposalWithdrawn {
        id: ProposalId,
        note: String,
    },
    ProposalRejected {
        id: ProposalId,
        note: String,
    },
    ProposalAccepted {
        id: ProposalId,
    },
}

#[derive(Debug, PartialEq)]
pub enum Command {
    // Task commands
    CreateTask {
        name: String,
        parent_id: Option<TaskId>,
    },
    ClaimTask {
        id: TaskId,
    },
    CompleteTask {
        id: TaskId,
        receipt: Receipt,
    },
    DropTask {
        id: TaskId,
        note: String,
    },
    ReleaseTask {
        id: TaskId,
        note: String,
    },
    // Proposal commands
    CreateProposal {
        // See note above about why proposals don't have an id at command time
        name: String,
        action: ProposalAction,
    },
    WithdrawProposal {
        id: ProposalId,
        note: String,
    },
    RejectProposal {
        id: ProposalId,
        note: String,
    },
    AcceptProposal {
        id: ProposalId,
    },
}
