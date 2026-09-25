use crate::Reject;
use crate::events::{Command, Event};
use crate::store::{Context, Tier, World};

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
        | Event::TaskDone { .. }
        | Event::TaskDelivered { .. }
        | Event::TaskReleased { .. }
        | Event::ProposalCreated { .. }
        | Event::ProposalWithdrawn { .. }
        | Event::Commented { .. }
        | Event::ArtifactAdded { .. }
        | Event::IncarnationCancelled { .. }
        | Event::TaskWorkspaceCheckpointed { .. } => Authority::AnyTier,
        // accept's door reads the world's birth attribution, so its
        // authority lives in the fold, not in a tier cell; revise's
        // author-or-human door reads the fold the same way
        Event::TaskAccepted { .. } | Event::CommentRevised { .. } => Authority::AnyTier,
        Event::TaskDropped { .. }
        | Event::ProposalRejected { .. }
        | Event::ProposalAccepted { .. } => Authority::Require(Tier::Human),
        Event::DemandRefused { .. }
        | Event::SteerForwarded { .. }
        | Event::IncarnationBound { .. }
        | Event::IncarnationPromptAccepted { .. }
        | Event::IncarnationPromptRejected { .. }
        | Event::IncarnationSettled { .. }
        | Event::RecordProducedBy { .. }
        | Event::TaskWorkspaceCreated { .. }
        | Event::TaskWorktreeCreated { .. } => Authority::Require(Tier::System),
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

/// Stateful expansion between decide and the fold, some events need to be expanded into multiple
/// events based on the current world state.
pub(crate) fn expand(world: &World, events: &[Event]) -> Result<Vec<Event>, Reject> {
    let mut expanded = Vec::with_capacity(events.len() + 1);
    for event in events {
        expanded.push(event.clone());
        // NOTE: switch this to a match statement if we add more events that need expansion
        if let Event::ProposalAccepted { id } = event {
            let proposal_ctx = world.proposals.get(id).ok_or(Reject::InvalidProposalId)?;
            expanded.push(
                proposal_ctx
                    .proposal
                    .action
                    .target_event(&proposal_ctx.proposal.name),
            );
        }
    }
    Ok(expanded)
}

/// Stateless command syntax: a command is its events, nothing else
pub fn decide(command: Command) -> Vec<Event> {
    match command {
        // Task commands
        Command::CreateTask { name, parent_id } => vec![Event::TaskCreated { name, parent_id }],
        Command::ClaimTask { id } => vec![Event::TaskClaimed { id }],
        Command::CompleteTask { id, receipt } => vec![Event::TaskDelivered { id, receipt }],
        Command::AcceptTask { id } => vec![Event::TaskAccepted { id }],
        Command::DropTask { id, note } => vec![Event::TaskDropped { id, note }],
        Command::ReleaseTask { id, note } => vec![Event::TaskReleased { id, note }],
        // Proposal commands
        Command::CreateProposal { name, action } => vec![Event::ProposalCreated { name, action }],
        Command::WithdrawProposal { id, note } => vec![Event::ProposalWithdrawn { id, note }],
        Command::RejectProposal { id, note } => vec![Event::ProposalRejected { id, note }],
        Command::AcceptProposal { id } => vec![Event::ProposalAccepted { id }],
        // Comment commands
        Command::Comment { target, body, kind } => vec![Event::Commented { target, body, kind }],
        Command::ReviseComment { id, body } => vec![Event::CommentRevised { id, body }],
        Command::RefuseDemand { demand, reason } => vec![Event::DemandRefused { demand, reason }],
        Command::Artifact { root, artifact } => vec![Event::ArtifactAdded { root, artifact }],
        Command::ForwardSteer { steer } => vec![Event::SteerForwarded { steer }],
        // Machinery verbs: System authorship comes from the role, never input
        Command::BindIncarnation {
            task_id,
            response_target,
            trigger,
            actor,
            session,
        } => vec![Event::IncarnationBound {
            task_id,
            response_target,
            trigger,
            actor,
            session,
        }],
        Command::AcceptPrompt { id } => vec![Event::IncarnationPromptAccepted { id }],
        Command::RejectPrompt { id, evidence } => {
            vec![Event::IncarnationPromptRejected { id, evidence }]
        }
        Command::SettleIncarnation { id } => vec![Event::IncarnationSettled { id }],
        Command::CancelIncarnation { id } => vec![Event::IncarnationCancelled { id }],
        Command::MarkRecord {
            incarnation_id,
            record_id,
        } => vec![Event::RecordProducedBy {
            record_id,
            incarnation_id,
        }],
        Command::CreateWorkspace {
            task_id,
            base,
            branch,
        } => vec![Event::TaskWorkspaceCreated {
            task_id,
            base,
            branch,
        }],
        Command::CreateWorktree { task_id, worktree } => {
            vec![Event::TaskWorktreeCreated { task_id, worktree }]
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
    use crate::objects::comment::{CommentId, CommentKind};
    use crate::objects::incarnation::IncarnationId;
    use crate::types::actor::ActorName;
    use crate::types::artifact::Artifact;
    use crate::types::failure::{FailureCode, FailureEvidence};
    use crate::types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
    use crate::{ContentHash, ProposalAction, ProposalId, Prose, RecordId, Target, TaskId};

    fn agent() -> Context {
        Context {
            actor: ActorName::new("saccade bot".into()).unwrap(),
            tier: Tier::Agent,
        }
    }

    fn human() -> Context {
        Context {
            actor: ActorName::new("human person".into()).unwrap(),
            tier: Tier::Human,
        }
    }
    #[test]
    fn authority_table_gates_exactly_the_gated_events() {
        let pid = ProposalId(RecordId(0));
        let events = [
            Event::TaskCreated {
                name: Prose::new("filler".into()).unwrap(),
                parent_id: None,
            },
            Event::TaskClaimed { id: TaskId(0) },
            Event::TaskDone {
                id: TaskId(0),
                receipt: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskDelivered {
                id: TaskId(0),
                receipt: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskAccepted { id: TaskId(0) },
            Event::TaskDropped {
                id: TaskId(0),
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::TaskReleased {
                id: TaskId(0),
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::ProposalCreated {
                name: Prose::new("filler".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            Event::ProposalWithdrawn {
                id: pid,
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::ProposalRejected {
                id: pid,
                note: Prose::new("filler".into()).unwrap(),
            },
            Event::ProposalAccepted { id: pid },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("filler".into()).unwrap(),
                kind: CommentKind::Note,
            },
            Event::CommentRevised {
                id: CommentId(RecordId(0)),
                body: Prose::new("filler".into()).unwrap(),
            },
            Event::DemandRefused {
                demand: CommentId(RecordId(0)),
                reason: Prose::new("filler".into()).unwrap(),
            },
            Event::SteerForwarded {
                steer: CommentId(RecordId(0)),
            },
            Event::ArtifactAdded {
                root: TaskId(0),
                artifact: Artifact {
                    name: Prose::new("sweep figure".into()).unwrap(),
                    hash: ContentHash::of(b"bytes"),
                },
            },
            Event::IncarnationBound {
                task_id: TaskId(0),
                response_target: CommentId(RecordId(0)),
                trigger: RecordId(0),
                actor: ActorName::new("pi".into()).unwrap(),
                session: SessionPointer::new("/tmp/session".into()).unwrap(),
            },
            Event::IncarnationPromptAccepted {
                id: IncarnationId(RecordId(0)),
            },
            Event::IncarnationPromptRejected {
                id: IncarnationId(RecordId(0)),
                evidence: FailureEvidence::new(FailureCode::PromptRejected, None),
            },
            Event::IncarnationSettled {
                id: IncarnationId(RecordId(0)),
            },
            Event::IncarnationCancelled {
                id: IncarnationId(RecordId(0)),
            },
            Event::RecordProducedBy {
                incarnation_id: IncarnationId(RecordId(0)),
                record_id: RecordId(0),
            },
            Event::TaskWorkspaceCreated {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            },
            Event::TaskWorktreeCreated {
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
}
