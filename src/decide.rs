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
        Event::TaskDropped { .. } => Authority::Require(Tier::Human),
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

pub fn decide(world: &World, command: Command, context: &Context) -> Result<Vec<Event>, Reject> {
    let events = match command {
        Command::CreateTask {
            task_name,
            parent_id,
        } => {
            let event = Event::TaskCreated {
                id: world.next_task_id(),
                task_name,
                parent_id,
            };
            enforce_tier(&event, context)?;

            if parent_id.is_some_and(|id| world.tasks.len() <= id.0) {
                return Err(Reject::InvalidParentTaskId);
            }

            vec![event]
        }
        Command::ClaimTask { id } => {
            let event = Event::TaskClaimed { id };
            enforce_tier(&event, context)?;

            let task = world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?;
            task.state.validate(&event)?;

            vec![event]
        }
        Command::CompleteTask { id, receipt } => {
            let event = Event::TaskDone { id, receipt };
            enforce_tier(&event, context)?;

            let task = world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?;
            task.state.validate(&event)?;

            vec![event]
        }
        Command::AbandonTask { id, reason, note } => {
            let event = Event::TaskDropped { id, reason, note };
            enforce_tier(&event, context)?;

            let task = world.tasks.get(id.0).ok_or(Reject::InvalidTaskId)?;
            task.state.validate(&event)?;

            vec![event]
        }
    };

    Ok(events)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{AbandonReason, Receipt, TaskId};

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
                reason: AbandonReason::Unwanted,
                note: None,
            },
        ];

        for event in &events {
            let gated = matches!(event, Event::TaskDropped { .. });

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
            reason: AbandonReason::Unwanted,
            note: Some("invalid abandon".into()),
        };

        let err1 = decide(&world, cmd(), &agent_ctx);
        let err2 = decide(&world, cmd(), &human_ctx);

        assert!(matches!(err1, Err(Reject::HumanOnly)));
        assert!(matches!(err2, Err(Reject::InvalidTaskId)));
    }
}
