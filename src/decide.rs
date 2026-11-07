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
        | Event::TaskCommitted { .. }
        | Event::TaskOffered { .. }
        | Event::TaskRejected { .. }
        | Event::TaskAccepted { .. }
        | Event::TaskWorkspaceCheckpointed { .. }
        | Event::MessageSent { .. } => Authority::AnyTier,
        // the human-authored judgment acts: cancellation and reopening
        Event::TaskCancelled { .. } | Event::TaskReopened { .. } => Authority::Require(Tier::Human),
        // the human's chosen root destination: an explicit act about
        // host state semantics, never an agent's inference
        Event::RootSpaceBound { .. } | Event::RootSpaceCleared {} => {
            Authority::Require(Tier::Human)
        }
        // provisioning by role: the machinery's establishment and
        // creation, the native registration, and the recipient's
        // delivery record. The confirmed change is machinery's act too:
        // it lands at the probe's pass, never from a caller.
        // Cleanup and restoration land only after their effects verified:
        // the same effect-only authority, never caller-authored facts.
        Event::WorkspaceEstablished { .. }
        | Event::WorkspaceCreated { .. }
        | Event::WorkspaceChanged { .. }
        | Event::WorkspaceCleaned { .. }
        | Event::WorkspaceRestored { .. }
        | Event::IncarnationRegistered { .. }
        | Event::SavedSessionBound { .. }
        | Event::HostAddressAssigned { .. }
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
        Command::CommitBoard {
            task,
            commit,
            parent,
            incorporations,
        } => vec![Event::TaskCommitted {
            task,
            commit,
            parent,
            incorporations,
        }],
        Command::OfferBoard { task, commit } => vec![Event::TaskOffered { task, commit }],
        Command::RejectBoard { task, grounds } => vec![Event::TaskRejected { task, grounds }],
        Command::AcceptBoard { task } => vec![Event::TaskAccepted { task }],
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
        Command::CreateWorkspace {
            base,
            branch,
            worktree,
        } => vec![Event::WorkspaceCreated {
            base,
            branch,
            worktree,
        }],
        Command::ChangeWorkspace {
            incarnation,
            worktree,
        } => vec![Event::WorkspaceChanged {
            incarnation,
            worktree,
        }],
        Command::CleanWorkspace { workspace } => {
            vec![Event::WorkspaceCleaned { workspace }]
        }
        Command::RestoreWorkspace { workspace } => {
            vec![Event::WorkspaceRestored { workspace }]
        }
        Command::RegisterIncarnation {
            native_id,
            saved_session,
            address,
            adopted,
            harness,
        } => vec![Event::IncarnationRegistered {
            native_id,
            saved_session,
            address,
            adopted,
            harness,
        }],
        Command::BindSavedSession {
            incarnation,
            saved_session,
        } => vec![Event::SavedSessionBound {
            incarnation,
            saved_session,
        }],
        Command::AssignHostAddress {
            incarnation,
            address,
            adopted,
        } => vec![Event::HostAddressAssigned {
            incarnation,
            address,
            adopted,
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
        Command::BindRootSpace { space, tab } => vec![Event::RootSpaceBound { space, tab }],
        Command::ClearRootSpace {} => vec![Event::RootSpaceCleared {}],
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::harness::Harness;
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::message::MessageId;
    use crate::types::actor::ActorName;
    use crate::types::pointers::{
        GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath,
    };
    use crate::{Prose, RecordId, TaskId};

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
        let events: [(Event, Authority); 21] = [
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
                Event::RootSpaceBound {
                    space: "spaRoot".into(),
                    tab: "tbRoot".into(),
                },
                Require(Tier::Human),
            ),
            (Event::RootSpaceCleared {}, Require(Tier::Human)),
            (
                Event::TaskCommitted {
                    task: TaskId(0),
                    commit: GitCommit::new("a".repeat(40)).unwrap(),
                    parent: None,
                    incorporations: Vec::new(),
                },
                AnyTier,
            ),
            (
                Event::TaskOffered {
                    task: TaskId(0),
                    commit: GitCommit::new("a".repeat(40)).unwrap(),
                },
                AnyTier,
            ),
            (
                Event::TaskRejected {
                    task: TaskId(0),
                    grounds: prose(),
                },
                AnyTier,
            ),
            (Event::TaskAccepted { task: TaskId(0) }, AnyTier),
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
            // Creation and the confirmed change are the machinery's own
            // acts, like the establishment and the registration.
            (
                Event::WorkspaceCreated {
                    base: GitCommit::new("abc123".into()).unwrap(),
                    branch: GitBranch::new("feature-abc123ab".into()).unwrap(),
                    worktree: WorktreePath::new("/wt".into()).unwrap(),
                },
                Require(Tier::System),
            ),
            (
                Event::WorkspaceChanged {
                    incarnation: IncarnationId(RecordId(0)),
                    worktree: WorktreePath::new("/wt".into()).unwrap(),
                },
                Require(Tier::System),
            ),
            // Cleanup and restoration land only after their effects
            // verified: the machinery's own completed facts.
            (
                Event::WorkspaceCleaned {
                    workspace: RecordId(0),
                },
                Require(Tier::System),
            ),
            (
                Event::WorkspaceRestored {
                    workspace: RecordId(0),
                },
                Require(Tier::System),
            ),
            // The native registration and the delivery record are the
            // machinery's; a message is its sending session's act.
            (
                Event::IncarnationRegistered {
                    native_id: native_id(),
                    saved_session: Some(locator()),
                    address: Some("/tty5".into()),
                    adopted: false,
                    harness: Harness::Pi,
                },
                Require(Tier::System),
            ),
            // The attach's locator binding and the relaunch's address
            // move are the machinery's own association facts.
            (
                Event::SavedSessionBound {
                    incarnation: IncarnationId(RecordId(0)),
                    saved_session: locator(),
                },
                Require(Tier::System),
            ),
            (
                Event::HostAddressAssigned {
                    incarnation: IncarnationId(RecordId(0)),
                    address: Some("/tty7".into()),
                    adopted: false,
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
            Command::CommitBoard {
                task: TaskId(0),
                commit: GitCommit::new("a".repeat(40)).unwrap(),
                parent: None,
                incorporations: Vec::new(),
            },
            Command::OfferBoard {
                task: TaskId(0),
                commit: GitCommit::new("a".repeat(40)).unwrap(),
            },
            Command::RejectBoard {
                task: TaskId(0),
                grounds: prose(),
            },
            Command::AcceptBoard { task: TaskId(0) },
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
            Command::CreateWorkspace {
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("workspace-deadbeef".into()).unwrap(),
                worktree: WorktreePath::new("/wt".into()).unwrap(),
            },
            Command::ChangeWorkspace {
                incarnation: IncarnationId(RecordId(0)),
                worktree: WorktreePath::new("/wt".into()).unwrap(),
            },
            Command::RegisterIncarnation {
                native_id: NativeSessionId::new("a1b2c3d4".into()).unwrap(),
                saved_session: Some(
                    SessionLocator::new("/sessions/a1b2c3d4.jsonl".into()).unwrap(),
                ),
                address: None,
                adopted: false,
                harness: Harness::Pi,
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
            Command::BindRootSpace {
                space: "spaRoot".into(),
                tab: "tbRoot".into(),
            },
            Command::ClearRootSpace {},
        ];
        for command in commands {
            assert_eq!(decide(command.clone()).len(), 1);
        }
    }
}
