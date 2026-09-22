//! Teaching refusals: every id-or-state refusal carries the expected
//! format and, where the world knows it, the likely intended target.
//! The world is consulted on the error path only; the Reject codes
//! stay the wire's vocabulary, the teaching rides as prose.

use crate::objects::comment::Target;
use crate::objects::proposal::{ProposalAction, ProposalState};
use crate::objects::task::TaskContext;
use crate::store::World;
use crate::{Command, ProposalId, RecordId, Reject, TaskId};

/// What the refusal should have said: the id space's grammar and the
/// target the asker probably meant. None when the world has nothing
/// to add to the brief code.
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
        Reject::InvalidProposalId => {
            let id = match command {
                Command::AcceptProposal { id }
                | Command::RejectProposal { id, .. }
                | Command::WithdrawProposal { id, .. } => *id,
                _ => return None,
            };
            let open: Vec<usize> = world
                .proposals
                .iter()
                .filter(|(_, p)| matches!(p.proposal.state, ProposalState::Open))
                .map(|(id, _)| id.0.0)
                .collect();
            Some(if open.is_empty() {
                format!(
                    "no proposal was born at {}; the ruling queue is empty (ids are bare log positions of proposal_created records)",
                    id.0.0
                )
            } else {
                format!(
                    "no proposal was born at {}; the open ones are {}",
                    id.0.0,
                    open.iter()
                        .map(|seq| seq.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
        }
        Reject::InvalidCommentId => {
            let seq = command_comment(command)?;
            if let Some(i) = world.tasks.iter().position(|ctx| ctx.birth.0 == seq) {
                return Some(format!(
                    "#{seq} is the birth record of task t-{i}; address its thread as t-{i}"
                ));
            }
            if world.proposals.contains_key(&ProposalId(RecordId(seq))) {
                return Some(format!(
                    "#{seq} is the birth record of a proposal; rule it by its bare log position (e.g. accept {seq})"
                ));
            }
            Some(format!(
                "#{seq} names no comment; a reply addresses a comment record as #<seq>, a task's thread as t-<n>"
            ))
        }
        Reject::InvalidStateTransition => match command {
            Command::ClaimTask { id } => need(world, *id, "a claim needs an open task"),
            Command::CompleteTask { id, .. } => need(world, *id, "done needs the holder's claim"),
            Command::DropTask { id, .. } => need(world, *id, "a drop needs an open or done task"),
            Command::ReleaseTask { id, .. } => need(world, *id, "a release needs a claimed task"),
            Command::CreateProposal { action, .. } => {
                let (id, what) = match action {
                    ProposalAction::Drop { task_id } => {
                        (*task_id, "a drop proposal needs an open or done task")
                    }
                    ProposalAction::Release { task_id } => {
                        (*task_id, "a release proposal needs a claimed task")
                    }
                };
                need(world, id, what)
            }
            Command::AcceptProposal { id } => {
                // only an open proposal's refusal is about the task; a
                // withdrawn or rejected one refuses for itself
                let ctx = world.proposals.get(id)?;
                if !matches!(ctx.proposal.state, ProposalState::Open) {
                    return None;
                }
                let action = &ctx.proposal.action;
                let (task_id, what) = match action {
                    ProposalAction::Drop { task_id } => {
                        (*task_id, "the proposal's drop needs an open or done task")
                    }
                    ProposalAction::Release { task_id } => {
                        (*task_id, "the proposal's release needs a claimed task")
                    }
                };
                need(world, task_id, what)
            }
            _ => None,
        },
        _ => None,
    }
}

/// The task a refusal names, when the command carries one — every
/// command shape that can raise an id-or-state refusal routes here,
/// so no door escapes the teaching by shape.
fn command_task(command: &Command) -> Option<TaskId> {
    match command {
        Command::ClaimTask { id }
        | Command::CompleteTask { id, .. }
        | Command::DropTask { id, .. }
        | Command::ReleaseTask { id, .. }
        | Command::BindIncarnation { task_id: id, .. }
        | Command::CreateWorkspace { task_id: id, .. }
        | Command::CreateWorktree { task_id: id, .. }
        | Command::CheckpointWorkspace { task_id: id, .. } => Some(*id),
        Command::Comment {
            target: Target::Task(id),
            ..
        } => Some(*id),
        Command::CreateProposal { action, .. } => Some(match action {
            ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => *task_id,
        }),
        _ => None,
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

/// The record a comment refusal names, when the command carries one.
fn command_comment(command: &Command) -> Option<usize> {
    match command {
        Command::Comment {
            target: Target::Comment(id),
            ..
        }
        | Command::BindIncarnation {
            response_target: id,
            ..
        } => Some(id.0.0),
        Command::Comment {
            target: Target::Task(_),
            ..
        } => None,
        _ => None,
    }
}

/// The state is named: what the task is, and what the act needed.
fn need(world: &World, id: TaskId, what: &str) -> Option<String> {
    let ctx = world.tasks.get(id.0)?;
    Some(format!("t-{} is {}; {what}", id.0, state_of(ctx)))
}

fn state_of(ctx: &TaskContext) -> String {
    match &ctx.task.state {
        crate::TaskState::Open => "open".into(),
        crate::TaskState::Claimed => format!(
            "claimed (held by {})",
            ctx.holder
                .as_ref()
                .map(|h| h.as_str().to_string())
                .unwrap_or_else(|| "someone".into())
        ),
        crate::TaskState::Delivered(_) => "delivered (awaiting accept)".into(),
        crate::TaskState::Done(_) => "done".into(),
        crate::TaskState::Dropped => "dropped".into(),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::events::Event;
    use crate::objects::comment::CommentId;
    use crate::store::{Context, Record, Tier};
    use crate::types::actor::ActorName;
    use crate::{CommentKind, Prose, Target};

    fn human() -> Context {
        Context {
            actor: ActorName::new("jerry".into()).unwrap(),
            tier: Tier::Human,
        }
    }

    fn agent() -> Context {
        Context {
            actor: ActorName::new("saccade bot".into()).unwrap(),
            tier: Tier::Agent,
        }
    }

    fn record(seq: usize, at: u64, ctx: &Context, event: Event) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: at,
            context: ctx.clone(),
            event,
        }
    }

    fn task(seq: usize, name: &str) -> Record {
        record(
            seq,
            seq as u64,
            &human(),
            Event::TaskCreated {
                name: Prose::new(name.into()).unwrap(),
                parent_id: None,
            },
        )
    }

    /// t-0 born at #0 and claimed by the agent; t-1 born at #3.
    fn claimed_world() -> World {
        World::replay(vec![
            task(0, "migrate floop"),
            record(1, 1, &agent(), Event::TaskClaimed { id: TaskId(0) }),
            task(3, "other work"),
        ])
        .unwrap()
    }

    #[test]
    fn a_birth_record_used_as_a_comment_target_names_its_task() {
        let world = claimed_world();
        let command = Command::Comment {
            target: Target::Comment(CommentId(RecordId(0))),
            body: Prose::new("replying to a birth".into()).unwrap(),
            kind: CommentKind::Note,
        };
        let taught = teach(&world, &command, &Reject::InvalidCommentId).unwrap();
        assert!(
            taught.contains("#0 is the birth record of task t-0"),
            "{taught}"
        );
        assert!(taught.contains("address its thread as t-0"), "{taught}");

        // a seq no comment, task, or proposal answers keeps the grammar
        let command = Command::Comment {
            target: Target::Comment(CommentId(RecordId(99))),
            body: Prose::new("replying to nothing".into()).unwrap(),
            kind: CommentKind::Note,
        };
        let taught = teach(&world, &command, &Reject::InvalidCommentId).unwrap();
        assert!(taught.contains("#99 names no comment"), "{taught}");
        assert!(taught.contains("#<seq>"), "{taught}");
        assert!(taught.contains("t-<n>"), "{taught}");
    }

    #[test]
    fn a_proposal_birth_used_as_a_comment_target_names_its_door() {
        let world = World::replay(vec![
            task(0, "migrate floop"),
            record(
                1,
                1,
                &agent(),
                Event::ProposalCreated {
                    name: Prose::new("drop floop instead".into()).unwrap(),
                    action: ProposalAction::Drop { task_id: TaskId(0) },
                },
            ),
        ])
        .unwrap();
        let command = Command::Comment {
            target: Target::Comment(CommentId(RecordId(1))),
            body: Prose::new("replying to a proposal".into()).unwrap(),
            kind: CommentKind::Note,
        };
        let taught = teach(&world, &command, &Reject::InvalidCommentId).unwrap();
        assert!(
            taught.contains("#1 is the birth record of a proposal"),
            "{taught}"
        );
        assert!(taught.contains("bare log position"), "{taught}");
    }

    #[test]
    fn a_judgment_refused_by_a_state_names_the_state() {
        let world = claimed_world();
        let command = Command::CreateProposal {
            name: Prose::new("drop floop".into()).unwrap(),
            action: ProposalAction::Drop { task_id: TaskId(0) },
        };
        let taught = teach(&world, &command, &Reject::InvalidStateTransition).unwrap();
        assert!(
            taught.contains("t-0 is claimed (held by saccade bot)"),
            "{taught}"
        );
        assert!(
            taught.contains("a drop proposal needs an open or done task"),
            "{taught}"
        );

        // a task id out of range learns the tracker's extent, by routing:
        // every command shape that names a task flows through the same
        // cell — the claim door, the comment door, the propose door
        for command in [
            Command::ClaimTask { id: TaskId(9) },
            Command::Comment {
                target: Target::Task(TaskId(9)),
                body: Prose::new("a body".into()).unwrap(),
                kind: CommentKind::Note,
            },
            Command::CreateProposal {
                name: Prose::new("drop floop".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(9) },
            },
        ] {
            let taught = teach(&world, &command, &Reject::InvalidTaskId)
                .unwrap_or_else(|| panic!("the door teaches: {command:?}"));
            assert!(
                taught.contains("no task t-9 exists; this tracker holds 2 tasks, t-0 through t-1"),
                "{taught}"
            );
        }

        // a one-task tracker counts honestly
        let single = World::replay(vec![task(0, "only work")]).unwrap();
        let command = Command::ClaimTask { id: TaskId(9) };
        let taught = teach(&single, &command, &Reject::InvalidTaskId).unwrap();
        assert!(
            taught.contains("this tracker holds 1 task, t-0"),
            "{taught}"
        );

        // refusals outside the id-or-state family stay with the brief code
        let command = Command::ClaimTask { id: TaskId(0) };
        assert_eq!(teach(&world, &command, &Reject::HumanOnly), None);
    }
}
