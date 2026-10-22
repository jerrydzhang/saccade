//! Teaching refusals: every id-or-state refusal carries the expected
//! format and, where the world knows it, the likely intended target.
//! The world is consulted on the error path only; the Reject codes
//! stay the wire's vocabulary, the teaching rides as prose.

use crate::objects::comment::CommentKind;
use crate::objects::task::TaskContext;
use crate::store::World;
use crate::{Command, CommentId, Reject, TaskId};

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
        Reject::InvalidCommentId => {
            let seq = command_comment(command)?;
            if let Some(i) = world.tasks.iter().position(|ctx| ctx.birth == seq.0) {
                return Some(format!(
                    "that record is the birth of task t-{i}; address its thread as t-{i}"
                ));
            }
            Some(format!(
                "that record names no comment; this fold holds {} comments, c-0 through c-{}",
                world.comments.len(),
                world.comments.len().saturating_sub(1)
            ))
        }
        Reject::NotAnOffer => {
            let offer = command_offer(command)?;
            let ctx = world.comments.get(&offer)?;
            Some(format!(
                "c-{} is a {}, not an offer; accept, reject, and a binding address an offer",
                num_of(world, offer),
                crate::views::kind_of(&ctx.comment.kind)
            ))
        }
        Reject::NotCurrentOffer => {
            let offer = command_offer(command)?;
            let ctx = world.comments.get(&offer)?;
            let task = &world.tasks[ctx.comment.home.0];
            let what = match command {
                Command::BindUse { .. } => {
                    "a binding names the source task's standing Offered or Concluded account"
                }
                Command::Comment {
                    kind: CommentKind::Use { .. },
                    ..
                } => "a use names its source task; its binding names that task's account",
                _ => "stale decisions cannot move the task",
            };
            match task.current_offer {
                Some(current) => Some(format!(
                    "c-{} is no longer current on t-{}; the standing offer is c-{} - {what}",
                    num_of(world, offer),
                    ctx.comment.home.0,
                    num_of(world, current)
                )),
                None => Some(format!(
                    "c-{} was superseded: t-{} holds no standing offer - to replace an account, reject it while it stands, then offer anew",
                    num_of(world, offer),
                    ctx.comment.home.0
                )),
            }
        }
        Reject::SelfReliance => Some(
            "a task cannot rely on its own conclusion: use names another internal task".into(),
        ),
        Reject::SelfAcceptance => {
            let offer = command_offer(command)?;
            let ctx = world.comments.get(&offer)?;
            Some(format!(
                "the declared task scope covers t-{}: a task-bound producer cannot accept its own conclusion - acceptance is an outside consumer's act",
                ctx.comment.home.0
            ))
        }
        Reject::InvalidIncarnationId => {
            Some("no incarnation the record birthed carries that reference; a claim names one by its public number i-N, never by task scope alone".into())
        }
        Reject::InvalidReference => {
            // the marker the mapping cannot resolve names itself; the
            // target the log cannot hold names the law
            let unresolvable = match command {
                Command::Comment {
                    kind, references, ..
                } => kind.prose().and_then(|prose| {
                    crate::markers::first_unresolved(prose.as_str(), references.len())
                }),
                Command::ReviseComment { id, body } => {
                    let mapped = world
                        .comments
                        .get(id)
                        .map(|ctx| ctx.references.len())
                        .unwrap_or(0);
                    crate::markers::first_unresolved(body.as_str(), mapped)
                }
                _ => None,
            };
            let revising = matches!(command, Command::ReviseComment { .. });
            match unresolvable {
                Some(marker) if revising => Some(format!(
                    "marker [^{marker}] has no reference to resolve; a revision keeps the mapping it was born with - a retargeted citation is a new comment"
                )),
                Some(marker) => Some(format!(
                    "marker [^{marker}] has no reference to resolve; cite with @c-<n> or --ref so the mapping carries it"
                )),
                None => Some(
                    "a citation names an earlier record of this log; code spans and math are literal, never citations".into(),
                ),
            }
        }
        Reject::InvalidStateTransition => match command {
            // the born-bound shape is migration's alone
            Command::Comment {
                kind:
                    CommentKind::Use {
                        bound: Some(_), ..
                    },
                ..
            } => Some(
                "a live use is born prospective; the born-bound shape belongs to migrated history - bind after the birth".into(),
            ),
            Command::Comment { kind, home, .. } => match kind {
                crate::CommentKind::Offer { .. } => {
                    need(world, *home, "an offer needs an open task and homes on it")
                }
                crate::CommentKind::Accept { .. } | crate::CommentKind::Reject { .. } => {
                    let offer = command_offer(command)?;
                    let ctx = world.comments.get(&offer)?;
                    match ctx.comment.home == *home {
                        false => Some(format!(
                            "the decision homes on t-{}, but its offer stands on t-{}; a decision lives on the task of the offer it names",
                            home.0,
                            ctx.comment.home.0
                        )),
                        true => need(
                            world,
                            ctx.comment.home,
                            match kind {
                                crate::CommentKind::Accept { .. } => {
                                    "an accept needs a task standing offered on that offer"
                                }
                                _ => "a reject needs a task standing offered on that offer",
                            },
                        ),
                    }
                }
                crate::CommentKind::Use { source, .. } => match source == home {
                    true => None,
                    false => need(world, *home, "a use homes on its consuming task"),
                },
                _ => None,
            },
            Command::BindUse { id, offer, .. } => {
                let use_ctx = world.comments.get(id)?;
                let offer_ctx = world.comments.get(offer)?;
                match use_ctx.comment.kind.source() {
                    Some(source) if offer_ctx.comment.home != source => Some(format!(
                        "c-{} stands on t-{}, but the use declared t-{} as its source; a binding selects from the source task alone",
                        num_of(world, *offer),
                        offer_ctx.comment.home.0,
                        source.0
                    )),
                    _ => None,
                }
            }
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
                        "c-{} is withdrawn; the door is terminal - a reached conclusion is a new deposit",
                        num_of(world, *id)
                    ))
                } else {
                    Some(format!(
                        "c-{} is a {}; withdrawal repairs a note deposit - the other variants keep their verdicts",
                        num_of(world, *id),
                        crate::views::kind_of(&ctx.comment.kind)
                    ))
                }
            }
            Command::ReviseComment { id, .. } => {
                let ctx = world.comments.get(id)?;
                if ctx.withdrawn.is_some() {
                    return Some(format!(
                        "c-{} is withdrawn; the door is terminal - a reached conclusion is a new deposit",
                        num_of(world, *id)
                    ));
                }
                (ctx.comment.kind.prose().is_none()).then(|| {
                    format!(
                        "c-{} is an accept: bodyless by law, no prose to repair",
                        num_of(world, *id)
                    )
                })
            }
            _ => None,
        },
        _ => None,
    }
}

/// The comment's public number for teaching text; a birth the fold
/// does not hold renders the raw position it named.
fn num_of(world: &World, id: CommentId) -> String {
    world
        .comment_num(id)
        .map(|n| n.to_string())
        .unwrap_or_else(|| format!("? (record {})", id.0.0))
}

/// The task a refusal names, when the command carries one - every
/// command shape that can raise an id-or-state refusal routes here,
/// so no door escapes the teaching by shape.
fn command_task(world: &World, command: &Command) -> Option<TaskId> {
    match command {
        Command::ClaimTask { id, .. }
        | Command::CancelTask { id, .. }
        | Command::ReopenTask { id, .. }
        | Command::EstablishWorkspace { task_id: id, .. }
        | Command::CheckpointWorkspace { task_id: id, .. } => Some(*id),
        Command::Comment { home, .. } => Some(*home),
        // a binding's task lives where the use homes
        Command::BindUse { id, .. } => world.comments.get(id).map(|c| c.comment.home),
        _ => None,
    }
}

/// The offer a command names, when it names one.
fn command_offer(command: &Command) -> Option<crate::CommentId> {
    match command {
        Command::Comment { kind, .. } => kind.offer(),
        Command::BindUse { offer, .. } => Some(*offer),
        _ => None,
    }
}

/// The comment a command addresses, when the refusal is about one.
fn command_comment(command: &Command) -> Option<crate::CommentId> {
    match command {
        Command::Comment { kind, .. } => kind.offer(),
        Command::ReviseComment { id, .. }
        | Command::WithdrawComment { id, .. }
        | Command::BindUse { id, .. } => Some(*id),
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
