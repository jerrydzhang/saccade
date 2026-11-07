//! Teaching refusals: every id-or-state refusal carries the expected
//! format and, where the world knows it, the likely intended target.
//! The world is consulted on the error path only; the Reject codes
//! stay the wire's vocabulary, the teaching rides as prose.

use crate::objects::task::TaskContext;
use crate::store::World;
use crate::{Command, Reject, TaskId};

/// What the refusal should have said: the id space's grammar and
/// the target the asker probably meant. None when the world has
/// nothing to add to the brief code.
pub fn teach(world: &World, command: &Command, reject: &Reject) -> Option<String> {
    match reject {
        Reject::InvalidTaskId => {
            let id = command_task(command)?;
            Some(match world.tasks.len() {
                0 => format!("no task t-{} exists; this tracker holds no tasks yet", id.0),
                1 => format!("no task t-{} exists; this tracker holds 1 task, t-0", id.0),
                n => format!(
                    "no task t-{} exists; this tracker holds {n} tasks, t-0 through t-{}",
                    id.0,
                    n - 1
                ),
            })
        }
        Reject::InvalidIncarnationId => {
            Some("no incarnation the record birthed carries that reference; a claim names one by its public number i-N, never by task scope alone".into())
        }
        Reject::InvalidStateTransition => match command {
            Command::CancelTask { id, .. } => {
                need(world, *id, "a cancellation stops an open or offered task")
            }
            Command::ReopenTask { id, .. } => {
                need(world, *id, "only a cancelled task reopens")
            }
            _ => None,
        },
        _ => None,
    }
}

/// The task a refusal names, when the command carries one - every
/// command shape that can raise an id-or-state refusal routes here,
/// so no door escapes the teaching by shape.
fn command_task(command: &Command) -> Option<TaskId> {
    match command {
        Command::ClaimTask { id, .. }
        | Command::CancelTask { id, .. }
        | Command::ReopenTask { id, .. }
        | Command::EstablishWorkspace { task_id: id, .. }
        | Command::CheckpointWorkspace { task_id: id, .. } => Some(*id),
        _ => None,
    }
}

/// The task's state as the refusal reads it.
fn need(world: &World, id: TaskId, what: &str) -> Option<String> {
    let ctx = world.tasks.get(id.0)?;
    let state = state_word(ctx);
    Some(format!("t-{} is {state}; {what}", id.0))
}

fn state_word(ctx: &TaskContext) -> &'static str {
    match ctx.task.state {
        crate::TaskState::Open => "open",
        crate::TaskState::Offered => "offered",
        crate::TaskState::Concluded => "concluded",
        crate::TaskState::Cancelled(_) => "cancelled",
    }
}
