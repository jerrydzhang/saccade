//! Teaching refusals: every id-or-state refusal carries the expected
//! format and, where the world knows it, the likely intended target.
//! The world is consulted on the error path only; the Reject codes
//! stay the wire's vocabulary, the teaching rides as prose.

use crate::objects::task::TaskContext;
use crate::store::World;
use crate::{Reject, record::events::Command, record::objects::task::TaskId};

/// The refusal for a tag name no tag bears: the tags that do exist,
/// or the way to make one. Every face that resolves a name speaks it.
pub fn unknown_tag(world: &World, name: &str) -> String {
    if world.tags.is_empty() {
        return "no tags exist; only the human creates one: sac tag create <name> \"<line>\""
            .into();
    }
    format!("no tag '{name}'; the tags are:\n{}", tag_lines(world))
}

/// Each tag with its line, one per row, in birth order.
pub fn tag_lines(world: &World) -> String {
    world
        .tags
        .values()
        .map(|tag| format!("{} - {}", tag.name.as_str(), tag.description.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

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
        Reject::InvalidTagId => {
            if world.tags.is_empty() {
                return Some(
                    "no tags exist; only the human creates one: sac tag create <name> \"<line>\""
                        .into(),
                );
            }
            Some(format!(
                "no tag was born at that reference; the tags are:\n{}",
                tag_lines(world)
            ))
        }
        Reject::DuplicateTag => match command {
            Command::CreateTag { name, .. } => {
                let held = world
                    .tag_named(name.as_str())
                    .and_then(|birth| world.tags.get(&birth))?;
                Some(format!(
                    "the tag '{}' exists already; tag names compare ignoring case",
                    held.name.as_str()
                ))
            }
            _ => None,
        },
        Reject::HumanOnly => match command {
            Command::CreateTag { .. } => Some(
                "only the human creates a tag; ask the human to run sac tag create <name> \"<line>\""
                    .into(),
            ),
            Command::DescribeTag { .. } => Some(
                "only the human rewrites a tag's line; ask the human to run sac tag describe <name> \"<line>\""
                    .into(),
            ),
            _ => None,
        },
        Reject::InvalidIncarnationId => {
            Some("no incarnation the record birthed carries that reference; a claim names one by its public number i-N, never by task scope alone".into())
        }
        Reject::SelfAcceptance => {
            let id = command_task(command)?;
            Some(format!(
                "the declared task scope covers t-{}: a task-bound producer cannot accept its own board - acceptance is an outside consumer's act",
                id.0
            ))
        }
        Reject::NotCurrentParent { expected } => {
            let id = command_task(command)?.0;
            let stands = match expected {
                Some(commit) => format!("its current commit is {commit}"),
                None => "it holds no commit yet".to_string(),
            };
            Some(format!("t-{id} is not at that parent; {stands} - commit onto what stands"))
        }
        Reject::UnrecordedSourceCommit { .. } => Some(
            "an incorporation cites a recorded revision of its source task; commit the source first, or cite a commit its record already carries".into(),
        ),
        Reject::NotCurrentCommit { current } => {
            let id = command_task(command)?.0;
            let stands = match current {
                Some(commit) => format!("its current commit is {commit}"),
                None => "it holds no commit yet".to_string(),
            };
            Some(format!("t-{id} is not at that commit; {stands}"))
        }
        Reject::SourceNotAccepted { .. } => Some(
            "acceptance cites exactly each source's accepted commit; commit and accept the source first, or cite the commit its acceptance carries".into(),
        ),
        Reject::InvalidStateTransition => match command {
            Command::CancelTask { id, .. } => {
                need(world, *id, "a cancellation stops an open or offered task")
            }
            Command::ReopenTask { id, .. } => {
                need(world, *id, "only a cancelled task reopens")
            }
            Command::CommitBoard { task, .. } => {
                need(world, *task, "a commit publishes on an open task")
            }
            Command::OfferBoard { task, .. } => {
                need(world, *task, "an offer needs an open task at its current commit")
            }
            Command::RejectBoard { task, .. } => {
                need(world, *task, "a reject needs a task standing offered")
            }
            Command::AcceptBoard { task, .. } => {
                need(world, *task, "an accept needs a task standing offered")
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
        | Command::CommitBoard { task: id, .. }
        | Command::OfferBoard { task: id, .. }
        | Command::RejectBoard { task: id, .. }
        | Command::TagTask { task: id, .. }
        | Command::UntagTask { task: id, .. }
        | Command::AcceptBoard { task: id } => Some(*id),
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
        crate::record::objects::task::TaskState::Open => "open",
        crate::record::objects::task::TaskState::Offered => "offered",
        crate::record::objects::task::TaskState::Concluded => "concluded",
        crate::record::objects::task::TaskState::Cancelled(_) => "cancelled",
    }
}
