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
        | Event::UseBound { .. }
        | Event::ArtifactAdded { .. }
        | Event::TaskWorkspaceCheckpointed { .. }
        | Event::MessageSent { .. } => Authority::AnyTier,
        // the human-authored judgment acts: cancellation and reopening
        Event::TaskCancelled { .. } | Event::TaskReopened { .. } => Authority::Require(Tier::Human),
        // a presentation repair may come from any judgment tier
        Event::CommentRevised { .. } | Event::CommentWithdrawn { .. } => Authority::AnyTier,
        // provisioning by role: the machinery's establishment, the
        // native registration, and the recipient's delivery record
        Event::WorkspaceEstablished { .. }
        | Event::IncarnationRegistered { .. }
        | Event::MessageSubmitted { .. } => Authority::Require(Tier::System),
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
        Command::CreateTask { name } => vec![Event::TaskCreated { name }],
        Command::ClaimTask { id, incarnation } => vec![Event::TaskClaimed { id, incarnation }],
        Command::CancelTask { id, explanation } => vec![Event::TaskCancelled { id, explanation }],
        Command::ReopenTask { id, explanation } => vec![Event::TaskReopened { id, explanation }],
        Command::Comment {
            home,
            kind,
            references,
        } => vec![Event::Commented {
            home,
            kind,
            references,
        }],
        Command::ReviseComment { id, body } => vec![Event::CommentRevised { id, body }],
        Command::WithdrawComment { id, note } => vec![Event::CommentWithdrawn { id, note }],
        Command::BindUse { id, offer, grounds } => vec![Event::UseBound { id, offer, grounds }],
        Command::Artifact { root, artifact } => vec![Event::ArtifactAdded { root, artifact }],
        Command::EstablishWorkspace {
            task_id,
            base,
            branch,
            worktree,
        } => vec![Event::WorkspaceEstablished {
            task_id,
            base,
            branch,
            worktree,
        }],
        Command::CheckpointWorkspace {
            task_id,
            checkpoint,
        } => vec![Event::TaskWorkspaceCheckpointed {
            task_id,
            checkpoint,
        }],
        Command::RegisterIncarnation {
            native_id,
            saved_session,
        } => vec![Event::IncarnationRegistered {
            native_id,
            saved_session,
        }],
        Command::SendMessage {
            from,
            to,
            reply_to,
            task,
            body,
        } => vec![Event::MessageSent {
            from,
            to,
            reply_to,
            task,
            body,
        }],
        Command::SubmitMessage { id, by } => vec![Event::MessageSubmitted { id, by }],
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::comment::{CommentId, CommentKind};
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::message::MessageId;
    use crate::types::actor::ActorName;
    use crate::types::artifact::Artifact;
    use crate::types::pointers::{
        GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath,
    };
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
        // Expected cells stated independently of required_tier.
        use Authority::*;
        let native_id = || NativeSessionId::new("a1b2c3d4".into()).unwrap();
        let locator = || SessionLocator::new("/sessions/a1b2c3d4.jsonl".into()).unwrap();
        let events: [(Event, Authority); 18] = [
            (Event::TaskCreated { name: prose() }, AnyTier),
            (
                Event::TaskClaimed {
                    id: TaskId(0),
                    incarnation: None,
                },
                AnyTier,
            ),
            (
                Event::TaskCancelled {
                    id: TaskId(0),
                    explanation: prose(),
                },
                Require(Tier::Human),
            ),
            (
                Event::TaskReopened {
                    id: TaskId(0),
                    explanation: prose(),
                },
                Require(Tier::Human),
            ),
            (
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Note { body: prose() },
                    references: Vec::new(),
                },
                AnyTier,
            ),
            (
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Offer { body: prose() },
                    references: Vec::new(),
                },
                AnyTier,
            ),
            (
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Accept {
                        offer: CommentId(RecordId(0)),
                    },
                    references: Vec::new(),
                },
                AnyTier,
            ),
            (
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Reject {
                        offer: CommentId(RecordId(0)),
                        grounds: prose(),
                    },
                    references: Vec::new(),
                },
                AnyTier,
            ),
            (
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Use {
                        source: TaskId(1),
                        reason: prose(),
                        bound: None,
                    },
                    references: Vec::new(),
                },
                AnyTier,
            ),
            (
                Event::UseBound {
                    id: CommentId(RecordId(0)),
                    offer: CommentId(RecordId(1)),
                    grounds: prose(),
                },
                AnyTier,
            ),
            (
                Event::CommentRevised {
                    id: CommentId(RecordId(0)),
                    body: prose(),
                },
                AnyTier,
            ),
            (
                Event::CommentWithdrawn {
                    id: CommentId(RecordId(0)),
                    note: prose(),
                },
                AnyTier,
            ),
            (
                Event::ArtifactAdded {
                    root: TaskId(0),
                    artifact: Artifact {
                        name: prose(),
                        hash: ContentHash::of(b"bytes"),
                    },
                },
                AnyTier,
            ),
            // Provisioning stays machine-authored; checkpoint is the exception.
            (
                Event::WorkspaceEstablished {
                    task_id: TaskId(0),
                    base: GitCommit::new("abc123".into()).unwrap(),
                    branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                    worktree: WorktreePath::new("/wt".into()).unwrap(),
                },
                Require(Tier::System),
            ),
            (
                Event::TaskWorkspaceCheckpointed {
                    task_id: TaskId(0),
                    checkpoint: GitCommit::new("abc123".into()).unwrap(),
                },
                AnyTier,
            ),
            // The native registration and the delivery record are the
            // machinery's; a message is its sending session's act.
            (
                Event::IncarnationRegistered {
                    native_id: native_id(),
                    saved_session: locator(),
                },
                Require(Tier::System),
            ),
            (
                Event::MessageSent {
                    from: IncarnationId(RecordId(0)),
                    to: IncarnationId(RecordId(1)),
                    reply_to: None,
                    task: None,
                    body: prose(),
                },
                AnyTier,
            ),
            (
                Event::MessageSubmitted {
                    id: MessageId(RecordId(1)),
                    by: IncarnationId(RecordId(1)),
                },
                Require(Tier::System),
            ),
        ];

        for (event, expected) in &events {
            for (name, ctx) in [
                ("agent", agent()),
                ("human", human()),
                ("system", Context::system()),
            ] {
                let passed = enforce_tier(&ctx, std::slice::from_ref(event)).is_ok();
                let wanted = match expected {
                    AnyTier => true,
                    Require(tier) => ctx.tier == *tier,
                };
                assert_eq!(
                    passed, wanted,
                    "{name} wrong at the authority table: {event:?}"
                );
            }
        }
    }

    /// Every command is its single event.
    #[test]
    fn every_command_expands_to_its_single_event() {
        let commands = [
            Command::CreateTask { name: prose() },
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
                home: TaskId(0),
                kind: CommentKind::Offer { body: prose() },
                references: vec![RecordId(0)],
            },
            Command::ReviseComment {
                id: CommentId(RecordId(0)),
                body: prose(),
            },
            Command::WithdrawComment {
                id: CommentId(RecordId(0)),
                note: prose(),
            },
            Command::BindUse {
                id: CommentId(RecordId(0)),
                offer: CommentId(RecordId(1)),
                grounds: prose(),
            },
            Command::Artifact {
                root: TaskId(0),
                artifact: Artifact {
                    name: prose(),
                    hash: ContentHash::of(b"bytes"),
                },
            },
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/wt".into()).unwrap(),
            },
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("abc123".into()).unwrap(),
            },
            Command::RegisterIncarnation {
                native_id: NativeSessionId::new("a1b2c3d4".into()).unwrap(),
                saved_session: SessionLocator::new("/sessions/a1b2c3d4.jsonl".into()).unwrap(),
            },
            Command::SendMessage {
                from: IncarnationId(RecordId(0)),
                to: IncarnationId(RecordId(1)),
                reply_to: Some(MessageId(RecordId(2))),
                task: Some(TaskId(0)),
                body: prose(),
            },
            Command::SubmitMessage {
                id: MessageId(RecordId(2)),
                by: IncarnationId(RecordId(1)),
            },
        ];
        for command in commands {
            assert_eq!(decide(command.clone()).len(), 1);
        }
    }
}
