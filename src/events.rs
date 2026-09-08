use crate::task::{Receipt, TaskId};

#[derive(Clone, Debug, PartialEq)]
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
        note: Option<String>,
    },
    TaskReleased {
        id: TaskId,
        note: Option<String>,
    },
}

#[derive(Debug, PartialEq)]
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
        note: Option<String>,
    },
    ReleaseTask {
        id: TaskId,
        note: Option<String>,
    },
}
