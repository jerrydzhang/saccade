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
        | Event::ProposalCreated { .. }
        | Event::ProposalWithdrawn { .. }
        | Event::Commented { .. } => Authority::AnyTier,
        Event::TaskDropped { .. }
        | Event::TaskReleased { .. }
        | Event::ProposalRejected { .. }
        | Event::ProposalAccepted { .. } => Authority::Require(Tier::Human),
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
        Command::CompleteTask { id, receipt } => vec![Event::TaskDone { id, receipt }],
        Command::DropTask { id, note } => vec![Event::TaskDropped { id, note }],
        Command::ReleaseTask { id, note } => vec![Event::TaskReleased { id, note }],
        // Proposal commands
        Command::CreateProposal { name, action } => vec![Event::ProposalCreated { name, action }],
        Command::WithdrawProposal { id, note } => vec![Event::ProposalWithdrawn { id, note }],
        Command::RejectProposal { id, note } => vec![Event::ProposalRejected { id, note }],
        Command::AcceptProposal { id } => vec![Event::ProposalAccepted { id }],
        // Comment commands
        Command::Comment { target, body } => vec![Event::Commented { target, body }],
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::actor::ActorName;
    use crate::{ProposalAction, ProposalId, Prose, RecordId, Target, TaskId};

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
    /// This test doesn't really test anything its more just a contract that at the time this test
    /// was written this is the expected behavior that shouldn't regress
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
            },
        ];

        for event in &events {
            let gated = matches!(
                event,
                Event::TaskDropped { .. }
                    | Event::TaskReleased { .. }
                    | Event::ProposalRejected { .. }
                    | Event::ProposalAccepted { .. }
            );

            assert_eq!(
                enforce_tier(&agent(), std::slice::from_ref(event)).is_err(),
                gated,
                "agent rejected at the wrong cells: {event:?}"
            );
            assert!(
                enforce_tier(&human(), std::slice::from_ref(event)).is_ok(),
                "human must pass every event kind: {event:?}"
            );
        }
    }
}
