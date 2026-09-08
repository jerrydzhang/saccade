use crate::events::{Command, Event};
use crate::store::{Context, Tier, World};
use crate::task::Reject;

/// Authority requirements per event kind
#[derive(Clone, Debug)]
pub(crate) enum Authority {
    AnyTier,
    Require(Tier),
}

fn required_tier(event: &Event) -> Authority {
    match event {
        Event::TaskCreated { .. } | Event::TaskClaimed { .. } | Event::TaskDone { .. } => {
            Authority::AnyTier
        }
        Event::TaskDropped { .. } | Event::TaskReleased { .. } => Authority::Require(Tier::Human),
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
        Command::CreateTask {
            task_name,
            parent_id,
        } => vec![Event::TaskCreated {
            id: world.next_task_id(),
            task_name,
            parent_id,
        }],
        Command::ClaimTask { id } => vec![Event::TaskClaimed { id }],
        Command::CompleteTask { id, receipt } => vec![Event::TaskDone { id, receipt }],
        Command::AbandonTask { id, note } => vec![Event::TaskDropped { id, note }],
        Command::ReleaseTask { id, note } => vec![Event::TaskReleased { id, note }],
    }
}

/// Validates an array of events against the current world state
fn validate(world: &World, events: &[Event]) -> Result<(), Reject> {
    for event in events {
        match event {
            Event::TaskCreated { parent_id, .. } => {
                if parent_id.is_some_and(|id| world.tasks.len() <= id.0) {
                    return Err(Reject::InvalidParentTaskId);
                }
            }
            event @ (Event::TaskClaimed { id }
            | Event::TaskDone { id, .. }
            | Event::TaskDropped { id, .. }
            | Event::TaskReleased { id, .. }) => {
                let task = world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?;
                task.state.validate(event)?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Receipt, TaskId};

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
        let events = [
            Event::TaskCreated {
                id: TaskId(0),
                task_name: String::new(),
                parent_id: None,
            },
            Event::TaskClaimed { id: TaskId(0) },
            Event::TaskDone {
                id: TaskId(0),
                receipt: Receipt(String::new()),
            },
            Event::TaskDropped {
                id: TaskId(0),
                note: None,
            },
            Event::TaskReleased {
                id: TaskId(0),
                note: None,
            },
        ];

        for event in &events {
            let gated = matches!(
                event,
                Event::TaskDropped { .. } | Event::TaskReleased { .. }
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
    /// arguments when in reality no matter what arguments you input the command itself is invalid
    #[test]
    fn agent_abandoning_invalid_task_err_ordering() {
        let agent_ctx = agent();
        let human_ctx = human();

        let world = World::new();

        let cmd = || Command::AbandonTask {
            id: TaskId(0),
            note: Some("invalid abandon".into()),
        };
        let err1 = decide(&world, &agent_ctx, cmd());
        let err2 = decide(&world, &human_ctx, cmd());

        assert!(matches!(err1, Err(Reject::HumanOnly)));
        assert!(matches!(err2, Err(Reject::InvalidTaskId)));
    }
}
