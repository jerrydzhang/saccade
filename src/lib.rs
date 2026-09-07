//! Saccade - issue tracker (idk what makes it special yet other than it's mine)

pub mod decide;
pub mod events;
pub mod store;
pub mod task;

pub use decide::decide;
pub use events::{Command, Event};
pub use store::{Context, Log, Record, RecordId, Tier, World};
pub use task::{Receipt, Reject, Task, TaskId, TaskState};

#[cfg(test)]
mod invariant {
    use super::*;

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

    fn populate_log(log: &mut Log) {
        let agent_ctx = agent();
        let human_ctx = human();

        log.execute(
            agent_ctx.clone(),
            Command::CreateTask {
                task_name: "implement foo".into(),
                parent_id: None,
            },
            1,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(0) }, 2)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                task_name: "fix bar".into(),
                parent_id: None,
            },
            3,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(1) }, 4)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(1),
                receipt: Receipt("bar fixed".into()),
            },
            5,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                task_name: "improve baz".into(),
                parent_id: Some(TaskId(0)),
            },
            6,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Receipt("foo completed successfully".into()),
            },
            7,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                task_name: "migrate floop".into(),
                parent_id: None,
            },
            8,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(3) }, 9)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::AbandonTask {
                id: TaskId(2),
                note: Some("scope covered by fix bar".into()),
            },
            10,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::AbandonTask {
                id: TaskId(1),
                note: None,
            },
            11,
        )
        .unwrap();
    }

    #[test]
    fn normal_task_lifecycle_passes() {
        let mut log = Log::new();
        populate_log(&mut log);

        assert_eq!(log.records().len(), 11);
        assert_eq!(
            log.world().tasks[0].state,
            TaskState::Done(Receipt("foo completed successfully".into())),
        );
        assert_eq!(log.world().tasks[1].state, TaskState::Dropped);
        assert_eq!(log.world().tasks[2].state, TaskState::Dropped);
        assert_eq!(log.world().tasks[3].state, TaskState::Claimed);

        assert_eq!(log.records()[8].id.0, 8);
        assert_eq!(log.records()[8].timestamp, 9);
        assert_eq!(log.records()[8].context.actor, "saccade bot");

        assert_eq!(log.records()[10].id.0, 10);
        assert_eq!(log.records()[10].timestamp, 11);
        assert_eq!(log.records()[10].context.actor, "human person");
    }

    #[test]
    fn block_invalid_taskstate_transitions() {
        let mut log = Log::new();
        let agent_ctx = agent();
        populate_log(&mut log);

        let err1 = log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(2),
                receipt: Receipt("foo completed successfully".into()),
            },
            1,
        );

        assert_eq!(log.records().len(), 11);
        assert!(matches!(err1, Err(Reject::InvalidStateTransition)));

        let err2 = log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(3) }, 2);

        assert_eq!(log.records().len(), 11);
        assert!(matches!(err2, Err(Reject::InvalidStateTransition)));

        let err3 = log.execute(
            agent_ctx.clone(),
            Command::AbandonTask {
                id: TaskId(3),
                note: None,
            },
            3,
        );

        assert_eq!(log.records().len(), 11);
        assert!(matches!(err3, Err(Reject::HumanOnly)));

        let err4 = log.execute(
            human(),
            Command::AbandonTask {
                id: TaskId(3),
                note: None,
            },
            4,
        );

        assert_eq!(log.records().len(), 11);
        assert!(matches!(err4, Err(Reject::InvalidStateTransition)));
    }

    #[test]
    fn block_invalid_task_id() {
        let mut log = Log::new();
        let agent_ctx = agent();
        populate_log(&mut log);

        let err = log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(4),
                receipt: Receipt("blip completed successfully".into()),
            },
            3,
        );

        assert_eq!(log.records().len(), 11);
        assert!(matches!(err, Err(Reject::InvalidTaskId)));
    }

    #[test]
    fn block_invalid_parent_task_id() {
        let mut log = Log::new();
        populate_log(&mut log);
        let agent_ctx = agent();

        let err = log.execute(
            agent_ctx.clone(),
            Command::CreateTask {
                task_name: "implement foo primatives".into(),
                parent_id: Some(TaskId(4)),
            },
            1,
        );

        assert_eq!(log.records().len(), 11);
        assert!(matches!(err, Err(Reject::InvalidParentTaskId)));
    }

    #[test]
    fn replay_produces_identical_world() {
        let mut log = Log::new();
        populate_log(&mut log);

        let recreated_world = World::replay(log.records().to_vec());
        assert_eq!(recreated_world, *log.world());
    }

    #[test]
    fn replay_split_equals_whole() {
        let mut log = Log::new();
        populate_log(&mut log);

        for i in 0..log.records().len() {
            let (head, tail) = log.records().split_at(i);
            let mut staged = World::replay(head.to_vec());
            for record in tail {
                staged.apply(record.clone());
            }

            assert_eq!(staged, *log.world());
        }
    }
}
