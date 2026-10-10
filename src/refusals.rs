//! Teaching refusals: every id-or-state refusal carries the expected
//! format and, where the world knows it, the likely intended target.
//! The world is consulted on the error path only; the Reject codes
//! stay the wire's vocabulary, the teaching rides as prose.

use crate::objects::comment::CommentKind;
use crate::objects::comment::Target;
use crate::objects::task::TaskContext;
use crate::store::World;
use crate::{Command, Reject, TaskId};

/// What the refusal should have said: the id space's grammar and
/// the target the asker probably meant. None when the world has
/// nothing to add to the brief code.
pub fn teach(world: &World, command: &Command, reject: &Reject) -> Option<String> {
    match reject {
        Reject::InvalidTaskId => {
            let id = command_task(world, command)?;
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
        Reject::InvalidParentTaskId => {
            let Command::CreateTask { parent_id, .. } = command else {
                return None;
            };
            let id = (*parent_id)?;
            Some(format!(
                "no task t-{} to parent under; this tracker holds {}",
                id.0,
                task_count(world.tasks.len())
            ))
        }
        Reject::InvalidCommentId => {
            let seq = command_comment(command)?;
            if let Some(i) = world.tasks.iter().position(|ctx| ctx.birth == seq.0) {
                return Some(format!(
                    "c-{seq} is the birth record of task t-{i}; address its thread as t-{i}",
                    seq = seq.0.0
                ));
            }
            Some(format!(
                "c-{seq} names no comment; a reply addresses a comment record as c-<seq>, a task's thread as t-<n>",
                seq = seq.0.0
            ))
        }
        Reject::NotAnOffer => {
            let offer = command_offer(command)?;
            let ctx = world.comments.get(&offer)?;
            Some(format!(
                "c-{} is a {}, not an offer; accept, reject, and use address an offer",
                offer.0 .0,
                crate::views::kind_of(&ctx.comment.kind)
            ))
        }
        Reject::NotCurrentOffer => {
            let offer = command_offer(command)?;
            let ctx = world.comments.get(&offer)?;
            let task = &world.tasks[ctx.comment.root.0];
            let what = if matches!(command, Command::Comment { kind: CommentKind::Use { .. }, .. }) {
                "a use names the source task's standing Offered or Concluded account"
            } else {
                "stale decisions cannot move the task"
            };
            match task.current_offer {
                Some(current) => Some(format!(
                    "c-{} is no longer current on t-{}; the standing offer is c-{} — {what}",
                    offer.0 .0,
                    ctx.comment.root.0,
                    current.0 .0
                )),
                None => Some(format!(
                    "c-{} was superseded: t-{} holds no standing offer — to replace an account, reject it while it stands, then offer anew",
                    offer.0 .0,
                    ctx.comment.root.0
                )),
            }
        }
        Reject::SelfReliance => Some(
            "a task cannot rely on its own conclusion: use names another internal task's account".into(),
        ),
        Reject::SelfAcceptance => {
            let offer = command_offer(command)?;
            let ctx = world.comments.get(&offer)?;
            Some(format!(
                "the declared task scope covers t-{}: a task-bound producer cannot accept its own conclusion — acceptance is an outside consumer's act",
                ctx.comment.root.0
            ))
        }
        Reject::InvalidIncarnationId => {
            Some("no incarnation the record birthed carries that reference; a claim names one by its birth record, never by task scope alone".into())
        }
        Reject::InvalidStateTransition => match command {
            Command::Comment { kind, target } => match kind {
                crate::CommentKind::Offer { .. } => {
                    let Target::Task(id) = target else {
                        return None;
                    };
                    need(world, *id, "an offer needs an open task and addresses the whole task")
                }
                crate::CommentKind::Accept { .. } | crate::CommentKind::Reject { .. } => {
                    let offer = command_offer(command)?;
                    let ctx = world.comments.get(&offer)?;
                    need(
                        world,
                        ctx.comment.root,
                        match kind {
                            crate::CommentKind::Accept { .. } => {
                                "an accept needs a task standing offered on that offer"
                            }
                            _ => "a reject needs a task standing offered on that offer",
                        },
                    )
                }
                _ => None,
            },
            Command::CancelTask { id, .. } => {
                need(world, *id, "a cancellation stops an open or offered task")
            }
            Command::ReopenTask { id, .. } => {
                need(world, *id, "only a cancelled task reopens")
            }
            Command::WithdrawComment { id, .. } => {
                let ctx = world.comments.get(id)?;
                if ctx.withdrawn.is_some() {
                    Some(format!(
                        "c-{} is withdrawn; the door is terminal — a reached conclusion is a new deposit",
                        id.0 .0
                    ))
                } else {
                    Some(format!(
                        "c-{} is a {}; withdrawal repairs a note deposit — the other variants keep their verdicts",
                        id.0 .0,
                        crate::views::kind_of(&ctx.comment.kind)
                    ))
                }
            }
            Command::ReviseComment { id, .. } => {
                let ctx = world.comments.get(id)?;
                if ctx.withdrawn.is_some() {
                    return Some(format!(
                        "c-{} is withdrawn; the door is terminal — a reached conclusion is a new deposit",
                        id.0 .0
                    ));
                }
                (ctx.comment.kind.prose().is_none()).then(|| {
                    format!(
                        "c-{} is an accept: bodyless by law, no prose to repair",
                        id.0 .0
                    )
                })
            }
            _ => None,
        },
        _ => None,
    }
}

/// The task a refusal names, when the command carries one — every
/// command shape that can raise an id-or-state refusal routes here,
/// so no door escapes the teaching by shape.
fn command_task(world: &World, command: &Command) -> Option<TaskId> {
    match command {
        Command::ClaimTask { id, .. }
        | Command::CancelTask { id, .. }
        | Command::ReopenTask { id, .. }
        | Command::EstablishWorkspace { task_id: id, .. }
        | Command::CheckpointWorkspace { task_id: id, .. } => Some(*id),
        Command::Comment {
            target: Target::Task(id),
            ..
        } => Some(*id),
        // a decision's task lives where its offer does
        Command::Comment { kind, .. } => {
            let offer = kind.offer()?;
            world.comments.get(&offer).map(|c| c.comment.root)
        }
        _ => None,
    }
}

/// The offer a command names, when it names one.
fn command_offer(command: &Command) -> Option<crate::CommentId> {
    match command {
        Command::Comment { kind, .. } => kind.offer(),
        _ => None,
    }
}

/// The comment a command addresses, when the refusal is about one.
fn command_comment(command: &Command) -> Option<crate::CommentId> {
    match command {
        Command::Comment {
            target: Target::Comment(cid),
            ..
        } => Some(*cid),
        Command::ReviseComment { id, .. } | Command::WithdrawComment { id, .. } => Some(*id),
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

/// The count as prose: "1 task" or "N tasks", never "1 tasks".
fn task_count(n: usize) -> String {
    match n {
        0 => "no tasks yet".into(),
        1 => "1 task".into(),
        n => format!("{n} tasks"),
    }
}
