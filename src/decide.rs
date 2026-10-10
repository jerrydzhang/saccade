use crate::Reject;
use crate::events::{Command, Event};
use crate::store::{Context, Tier};

/// Authority requirements per event kind
#[derive(Clone, Debug)]
pub(crate) enum Authority {
    AnyTier,
    Require(Tier),
}

fn required_tier(event: &Event) -> Authority {
    match event {
        Event::TaskCreated { .. }
        | Event::TaskClaimed { .. }
        | Event::Commented { .. }
        | Event::ArtifactAdded { .. }
        | Event::TaskWorkspaceCheckpointed { .. } => Authority::AnyTier,
        // the human-authored judgment acts: cancellation and reopening
        Event::TaskCancelled { .. } | Event::TaskReopened { .. } => Authority::Require(Tier::Human),
        // a presentation repair may come from any judgment tier
        Event::CommentRevised { .. } | Event::CommentWithdrawn { .. } => Authority::AnyTier,
        // the machinery's workspace records follow provisioning by role
        Event::TaskWorkspaceCreated { .. } | Event::TaskWorkspaceMaterialized { .. } => {
            Authority::Require(Tier::System)
        }
    }
}

/// Authority gate over a command's events: the table is law, tier never identity
pub(crate) fn enforce_tier(context: &Context, events: &[Event]) -> Result<(), Reject> {
    for event in events {
        match required_tier(event) {
            Authority::AnyTier => {}
            Authority::Require(tier) if context.tier == tier => {}
            Authority::Require(_) => return Err(Reject::HumanOnly),
        }
    }
    Ok(())
}

/// Stateless command syntax: a command is its events, nothing else
pub fn decide(command: Command) -> Vec<Event> {
    match command {
        Command::CreateTask { name, parent_id } => vec![Event::TaskCreated { name, parent_id }],
        Command::ClaimTask { id, incarnation } => vec![Event::TaskClaimed { id, incarnation }],
        Command::CancelTask { id, explanation } => vec![Event::TaskCancelled { id, explanation }],
        Command::ReopenTask { id, explanation } => vec![Event::TaskReopened { id, explanation }],
        Command::Comment { target, kind } => vec![Event::Commented { target, kind }],
        Command::ReviseComment { id, body } => vec![Event::CommentRevised { id, body }],
        Command::WithdrawComment { id, note } => vec![Event::CommentWithdrawn { id, note }],
        Command::Artifact { root, artifact } => vec![Event::ArtifactAdded { root, artifact }],
        Command::CreateWorkspace {
            task_id,
            base,
            branch,
        } => vec![Event::TaskWorkspaceCreated {
            task_id,
            base,
            branch,
        }],
        Command::MaterializeWorkspace { task_id, worktree } => {
            vec![Event::TaskWorkspaceMaterialized { task_id, worktree }]
        }
        Command::CheckpointWorkspace {
            task_id,
            checkpoint,
        } => vec![Event::TaskWorkspaceCheckpointed {
            task_id,
            checkpoint,
        }],
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::comment::{CommentId, CommentKind, Target};
    use crate::objects::incarnation::IncarnationId;
    use crate::types::actor::ActorName;
    use crate::types::artifact::Artifact;
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};
    use crate::{ContentHash, Prose, RecordId, TaskId};

    fn agent() -> Context {
        Context {
            actor: ActorName::new("saccade bot".into()).unwrap(),
            tier: Tier::Agent,
            scope: None,
        }
    }

    fn human() -> Context {
        Context {
            actor: ActorName::new("human person".into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        }
    }

    fn prose() -> Prose {
        Prose::new("filler".into()).unwrap()
    }

    #[test]
    fn authority_table_gates_exactly_the_gated_events() {
        let events = [
            Event::TaskCreated {
                name: prose(),
                parent_id: None,
            },
            Event::TaskClaimed {
                id: TaskId(0),
                incarnation: None,
            },
            Event::TaskCancelled {
                id: TaskId(0),
                explanation: prose(),
            },
            Event::TaskReopened {
                id: TaskId(0),
                explanation: prose(),
            },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                kind: CommentKind::Note { body: prose() },
            },
            Event::Commented {
                target: Target::Comment(CommentId(RecordId(0))),
                kind: CommentKind::Accept {
                    offer: CommentId(RecordId(0)),
                },
            },
            Event::CommentRevised {
                id: CommentId(RecordId(0)),
                body: prose(),
            },
            Event::CommentWithdrawn {
                id: CommentId(RecordId(0)),
                note: prose(),
            },
            Event::ArtifactAdded {
                root: TaskId(0),
                artifact: Artifact {
                    name: prose(),
                    hash: ContentHash::of(b"bytes"),
                },
            },
            Event::TaskWorkspaceCreated {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            },
            Event::TaskWorkspaceMaterialized {
                task_id: TaskId(0),
                worktree: WorktreePath::new("/wt".into()).unwrap(),
            },
            Event::TaskWorkspaceCheckpointed {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("abc123".into()).unwrap(),
            },
        ];

        for event in &events {
            let expectation = |ctx: &Context| match required_tier(event) {
                Authority::AnyTier => true,
                Authority::Require(tier) => ctx.tier == tier,
            };
            for (name, ctx) in [("agent", agent()), ("human", human())] {
                let passed = enforce_tier(&ctx, std::slice::from_ref(event)).is_ok();
                assert_eq!(
                    passed,
                    expectation(&ctx),
                    "{name} wrong at the authority table: {event:?}"
                );
            }
            let ctx = Context::system();
            let passed = enforce_tier(&ctx, std::slice::from_ref(event)).is_ok();
            assert_eq!(
                passed,
                expectation(&ctx),
                "system wrong at the authority table: {event:?}"
            );
        }
    }

    /// The command translation is the identity: every command is its
    /// events, one record each, no expansion left.
    #[test]
    fn every_command_expands_to_its_single_event() {
        let commands = [
            Command::CreateTask {
                name: prose(),
                parent_id: None,
            },
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: Some(IncarnationId(RecordId(9))),
            },
            Command::CancelTask {
                id: TaskId(0),
                explanation: prose(),
            },
            Command::ReopenTask {
                id: TaskId(0),
                explanation: prose(),
            },
            Command::Comment {
                target: Target::Task(TaskId(0)),
                kind: CommentKind::Offer { body: prose() },
            },
            Command::ReviseComment {
                id: CommentId(RecordId(0)),
                body: prose(),
            },
            Command::WithdrawComment {
                id: CommentId(RecordId(0)),
                note: prose(),
            },
            Command::Artifact {
                root: TaskId(0),
                artifact: Artifact {
                    name: prose(),
                    hash: ContentHash::of(b"bytes"),
                },
            },
            Command::CreateWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            },
            Command::MaterializeWorkspace {
                task_id: TaskId(0),
                worktree: WorktreePath::new("/wt".into()).unwrap(),
            },
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("abc123".into()).unwrap(),
            },
        ];
        for command in commands {
            assert_eq!(decide(command.clone()).len(), 1);
        }
    }
}
