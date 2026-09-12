use crate::Reject;
use crate::events::{Command, Event};
use crate::objects::comment::Target;
use crate::objects::proposal::{ProposalAction, ProposalState};
use crate::prose::Prose;
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

fn enforce_tier(event: &Event, context: &Context) -> Result<(), Reject> {
    // this will probably need to be changed when the system actor is added
    match required_tier(event) {
        Authority::AnyTier => Ok(()),
        Authority::Require(tier) if context.tier == tier => Ok(()),
        Authority::Require(_) => Err(Reject::HumanOnly),
    }
}

/// This function takes in its arguments and the either returns a vec of validated events or rejects
/// the command
pub fn decide(world: &World, context: &Context, command: Command) -> Result<Vec<Event>, Reject> {
    let events = candidate(world, command);

    for event in &events {
        enforce_tier(event, context)?;
    }

    validate(world, &events)?;

    Ok(events)
}

/// Takes the current world state and proposes a vec of events
fn candidate(world: &World, command: Command) -> Vec<Event> {
    match command {
        // Task commands
        Command::CreateTask { name, parent_id } => vec![Event::TaskCreated {
            id: world.next_task_id(),
            name,
            parent_id,
        }],
        Command::ClaimTask { id } => vec![Event::TaskClaimed { id }],
        Command::CompleteTask { id, receipt } => vec![Event::TaskDone { id, receipt }],
        Command::DropTask { id, note } => vec![Event::TaskDropped { id, note }],
        Command::ReleaseTask { id, note } => vec![Event::TaskReleased { id, note }],
        // Proposal commands
        Command::CreateProposal { name, action } => vec![Event::ProposalCreated { name, action }],
        Command::WithdrawProposal { id, note } => vec![Event::ProposalWithdrawn { id, note }],
        Command::RejectProposal { id, note } => vec![Event::ProposalRejected { id, note }],
        Command::AcceptProposal { id } => match world.proposals.get(&id) {
            Some(proposal_ctx) => {
                vec![
                    Event::ProposalAccepted { id },
                    proposal_ctx
                        .proposal
                        .action
                        .target_event(&proposal_ctx.proposal.name),
                ]
            }
            // This will be rejected in validate
            None => vec![Event::ProposalAccepted { id }],
        },
        // Comment commands
        Command::Comment { target, body } => vec![Event::Commented { target, body }],
    }
}

/// Validates an array of events against the current world state
fn validate(world: &World, events: &[Event]) -> Result<(), Reject> {
    for event in events {
        match event {
            // Task
            Event::TaskCreated { parent_id, .. } => {
                if parent_id.is_some_and(|id| world.tasks.len() <= id.0) {
                    return Err(Reject::InvalidParentTaskId);
                }
            }
            event @ Event::TaskClaimed { id } => {
                let task = &world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?.task;
                task.state.validate(event)?;
            }
            event @ Event::TaskDone { id, .. } => {
                let task = &world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?.task;
                task.state.validate(event)?;
            }
            event @ Event::TaskDropped { id, .. } | event @ Event::TaskReleased { id, .. } => {
                let task = &world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?.task;
                task.state.validate(event)?;
            }
            // Proposal
            Event::ProposalCreated { action, .. } => match action {
                ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => {
                    let task = &world
                        .tasks
                        .get(task_id.0)
                        .ok_or(Reject::InvalidTaskId)?
                        .task;
                    task.state
                        .validate(&action.target_event(&Prose::new("probe".into())?))?;
                    // one judgment at a time per task: validate may walk, it is not a render path
                    let pending = world.proposals.values().any(|ctx| {
                        ctx.proposal.state == ProposalState::Open
                            && match &ctx.proposal.action {
                                ProposalAction::Drop { task_id: t }
                                | ProposalAction::Release { task_id: t } => t == task_id,
                            }
                    });
                    if pending {
                        return Err(Reject::ProposalAlreadyOpen);
                    }
                }
            },
            event @ (Event::ProposalWithdrawn { id, .. } | Event::ProposalRejected { id, .. }) => {
                let proposal = &world
                    .proposals
                    .get(id)
                    .ok_or(Reject::InvalidProposalId)?
                    .proposal;
                proposal.state.validate(event)?;
            }
            event @ Event::ProposalAccepted { id } => {
                let proposal = &world
                    .proposals
                    .get(id)
                    .ok_or(Reject::InvalidProposalId)?
                    .proposal;
                proposal.state.validate(event)?;
            }
            // Comment
            Event::Commented { target, .. } => match target {
                Target::Task(id) => {
                    world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?;
                }
                Target::Comment(id) => {
                    world.comments.get(id).ok_or(Reject::InvalidCommentId)?;
                }
            },
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{ProposalAction, ProposalId, Prose, Record, RecordId, Target, TaskId};

    fn agent() -> Context {
        Context {
            actor: "saccade bot".into(),
            tier: Tier::Agent,
        }
    }

    fn human() -> Context {
        Context {
            actor: "human person".into(),
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
                id: TaskId(0),
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
                enforce_tier(event, &agent()).is_err(),
                gated,
                "agent rejected at the wrong cells: {event:?}"
            );
            assert!(
                enforce_tier(event, &human()).is_ok(),
                "human must pass every event kind: {event:?}"
            );
        }
    }

    /// Authority errors supercede state transition errors. This is logical since it if you get
    /// a state transition error first you might suspect it is an issue with the command
    /// arguments when in reality no matter what arguments you input the command itself is
    /// invalid.
    #[test]
    fn err_ordering_authority_supersedes_existence() {
        let agent_ctx = agent();
        let human_ctx = human();

        let world = World::new();

        let drop = || Command::DropTask {
            id: TaskId(0),
            note: Prose::new("invalid drop".into()).unwrap(),
        };
        let err1 = decide(&world, &agent_ctx, drop());
        let err2 = decide(&world, &human_ctx, drop());

        assert!(matches!(err1, Err(Reject::HumanOnly)));
        assert!(matches!(err2, Err(Reject::InvalidTaskId)));

        let accept = || Command::AcceptProposal {
            id: ProposalId(RecordId(99)),
        };
        let err3 = decide(&world, &agent_ctx, accept());
        let err4 = decide(&world, &human_ctx, accept());

        assert!(matches!(err3, Err(Reject::HumanOnly)));
        assert!(matches!(err4, Err(Reject::InvalidProposalId)));
    }

    fn record(seq: usize, event: Event) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: 0,
            context: human(),
            event,
        }
    }

    /// One judgment at a time per task: a second open proposal on the same
    /// target is refused, and the gate reopens once the first resolves.
    #[test]
    fn a_second_open_proposal_on_one_task_is_refused() {
        let floop = |seq, event| record(seq, event);
        let world = World::replay(vec![
            floop(
                0,
                Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("migrate floop".into()).unwrap(),
                    parent_id: None,
                },
            ),
            floop(
                2,
                Event::ProposalCreated {
                    name: Prose::new("drop floop instead".into()).unwrap(),
                    action: ProposalAction::Drop { task_id: TaskId(0) },
                },
            ),
        ]);

        let second = Command::CreateProposal {
            name: Prose::new("drop floop again".into()).unwrap(),
            action: ProposalAction::Drop { task_id: TaskId(0) },
        };
        assert!(matches!(
            decide(&world, &human(), second),
            Err(Reject::ProposalAlreadyOpen)
        ));

        let resolved = World::replay(vec![
            floop(
                0,
                Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("migrate floop".into()).unwrap(),
                    parent_id: None,
                },
            ),
            floop(
                2,
                Event::ProposalCreated {
                    name: Prose::new("drop floop instead".into()).unwrap(),
                    action: ProposalAction::Drop { task_id: TaskId(0) },
                },
            ),
            floop(
                3,
                Event::ProposalRejected {
                    id: ProposalId(RecordId(2)),
                    note: Prose::new("floop stays".into()).unwrap(),
                },
            ),
        ]);
        let third = Command::CreateProposal {
            name: Prose::new("drop floop for real".into()).unwrap(),
            action: ProposalAction::Drop { task_id: TaskId(0) },
        };
        assert!(decide(&resolved, &human(), third).is_ok());
    }
}
