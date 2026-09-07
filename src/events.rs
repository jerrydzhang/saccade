use crate::task::{AbandonReason, Receipt, TaskId};

#[derive(Clone, Debug)]
pub enum Event {
    // Task Events
    TaskCreated {
        id: TaskId,
        task_name: String,
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
        reason: AbandonReason,
        note: Option<String>,
    },
}

#[derive(Debug)]
pub enum Command {
    CreateTask {
        task_name: String,
        parent_id: Option<TaskId>,
    },
    ClaimTask {
        id: TaskId,
    },
    CompleteTask {
        id: TaskId,
        receipt: Receipt,
    },
    AbandonTask {
        id: TaskId,
        reason: AbandonReason,
        note: Option<String>,
    },
}
