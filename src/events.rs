use crate::objects::comment::Target;
use crate::objects::proposal::{ProposalAction, ProposalId};
use crate::objects::task::TaskId;
use crate::prose::Prose;

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    // Task Events
    TaskCreated {
        id: TaskId,
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
        // Proposal's are not created with an id at event time since the id is directly the record
        // id of the proposal, this means unlike a taskid the id is directly derivable from the
        // single record where as a task would require counting up all the TaskCreated events to
        // derive the id
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
