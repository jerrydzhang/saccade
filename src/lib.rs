//! Saccade - issue tracker (idk what makes it special yet other than it's mine)

pub mod db;
pub mod decide;
pub mod events;
pub mod objects;
pub mod store;
pub mod wire;

pub use decide::decide;
pub use events::{Command, Event};
pub use objects::proposal::{Proposal, ProposalAction, ProposalId, ProposalState};
pub use objects::task::{Receipt, Task, TaskId, TaskState};
pub use store::{Context, Log, Record, RecordId, Tier, World};

#[derive(Debug)]
pub enum Reject {
    // Task
    InvalidTaskId,
    InvalidParentTaskId,
    // Proposal
    InvalidProposalId,
    // Permissions
    HumanOnly,
    // Misc
    InvalidStateTransition,
}

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

    const RECORD_COUNT: usize = 14;

    fn populate_log(log: &mut Log) {
        let agent_ctx = agent();
        let human_ctx = human();

        log.execute(
            agent_ctx.clone(),
            Command::CreateTask {
                name: "implement foo".into(),
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
                name: "fix bar".into(),
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
                name: "improve baz".into(),
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
                name: "migrate floop".into(),
                parent_id: None,
            },
            8,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(3) }, 9)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::ReleaseTask {
                id: TaskId(3),
                note: Some("run dead, reclaim".into()),
            },
            10,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(3) }, 11)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::DropTask {
                id: TaskId(2),
                note: Some("scope covered by fix bar".into()),
            },
            12,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::DropTask {
                id: TaskId(1),
                note: None,
            },
            13,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: "open work".into(),
                parent_id: None,
            },
            14,
        )
        .unwrap();
    }

    #[test]
    fn normal_task_lifecycle_passes() {
        let mut log = Log::new();
        populate_log(&mut log);

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert_eq!(
            log.world().tasks[0].state,
            TaskState::Done(Receipt("foo completed successfully".into())),
        );
        assert_eq!(log.world().tasks[1].state, TaskState::Dropped);
        assert_eq!(log.world().tasks[2].state, TaskState::Dropped);
        assert_eq!(log.world().tasks[3].state, TaskState::Claimed);
        assert_eq!(log.world().tasks[4].state, TaskState::Open);

        assert_eq!(log.records()[8].id.0, 8);
        assert_eq!(log.records()[8].timestamp, 9);
        assert_eq!(log.records()[8].context.actor, "saccade bot");

        assert_eq!(log.records()[9].id.0, 9);
        assert_eq!(log.records()[9].timestamp, 10);
        assert_eq!(log.records()[9].context.actor, "human person");

        assert_eq!(log.records()[10].id.0, 10);
        assert_eq!(log.records()[10].timestamp, 11);
        assert_eq!(log.records()[10].context.actor, "saccade bot");
    }

    #[test]
    fn block_invalid_taskstate_transitions() {
        let mut log = Log::new();
        let agent_ctx = agent();
        populate_log(&mut log);

        // foo is already done, so completing it again is illegal
        let err1 = log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(2),
                receipt: Receipt("foo completed successfully".into()),
            },
            1,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err1, Err(Reject::InvalidStateTransition)));

        // bar is already dropped, so claiming it is illegal
        let err2 = log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(3) }, 2);

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err2, Err(Reject::InvalidStateTransition)));

        // an agent cannot drop a task, so this is illegal
        let err3 = log.execute(
            agent_ctx.clone(),
            Command::DropTask {
                id: TaskId(3),
                note: None,
            },
            3,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err3, Err(Reject::HumanOnly)));

        // a human can drop a task, but this one is already claimed, so it's illegal
        let err4 = log.execute(
            human(),
            Command::DropTask {
                id: TaskId(3),
                note: None,
            },
            4,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err4, Err(Reject::InvalidStateTransition)));

        // An agent cannot release a task, so this is illegal
        let err5 = log.execute(
            agent_ctx.clone(),
            Command::ReleaseTask {
                id: TaskId(3),
                note: None,
            },
            5,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err5, Err(Reject::HumanOnly)));

        // This task is already done, so releasing it is illegal
        let err6 = log.execute(
            human(),
            Command::ReleaseTask {
                id: TaskId(0),
                note: None,
            },
            6,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err6, Err(Reject::InvalidStateTransition)));

        // This task does not exist so proposing to drop it is illegal
        let err7 = log.execute(
            agent_ctx.clone(),
            Command::CreateProposal {
                name: "probe".into(),
                action: ProposalAction::Drop { task_id: TaskId(9) },
            },
            7,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err7, Err(Reject::InvalidTaskId)));

        // This task has already been dropped, so proposing to drop it again is illegal
        let err8 = log.execute(
            agent_ctx.clone(),
            Command::CreateProposal {
                name: "probe".into(),
                action: ProposalAction::Drop { task_id: TaskId(3) },
            },
            8,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err8, Err(Reject::InvalidStateTransition)));

        // This task is already open, so proposing to release it is illegal
        let err9 = log.execute(
            agent_ctx.clone(),
            Command::CreateProposal {
                name: "probe".into(),
                action: ProposalAction::Release { task_id: TaskId(4) },
            },
            9,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
        assert!(matches!(err9, Err(Reject::InvalidStateTransition)));
    }

    #[test]
    fn block_invalid_task_id() {
        let mut log = Log::new();
        let agent_ctx = agent();
        populate_log(&mut log);

        let err = log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(9),
                receipt: Receipt("blip completed successfully".into()),
            },
            3,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
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
                name: "implement foo primatives".into(),
                parent_id: Some(TaskId(9)),
            },
            1,
        );

        assert_eq!(log.records().len(), RECORD_COUNT);
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

    #[test]
    fn accept_compound_write_two_records_in_order() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: "I am going to do floop again".into(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        let proposed = log
            .execute(
                agent(),
                Command::CreateProposal {
                    name: "this is a duplicated task".into(),
                    action: ProposalAction::Drop { task_id: TaskId(0) },
                },
                2,
            )
            .unwrap();

        // A proposal's identity is the record id of its creation act, so the proposal's id is 1
        assert_eq!(proposed[0].id.0, 1);
        assert!(log.world().proposals.contains_key(&ProposalId(RecordId(1))));

        let accepted = log
            .execute(
                human(),
                Command::AcceptProposal {
                    id: ProposalId(RecordId(1)),
                },
                3,
            )
            .unwrap();

        // accepting produces two records: the proposal acceptance and the act of dropping the task
        assert_eq!(accepted.len(), 2);
        assert!(matches!(
            &accepted[0].event,
            Event::ProposalAccepted { id } if id.0.0 == 1
        ));
        let Event::TaskDropped { id, note } = &accepted[1].event else {
            panic!("the embedded act must ride the accept");
        };
        assert_eq!(id.0, 0);
        // the proposal's name becomes the drop's note
        assert_eq!(note.as_deref(), Some("this is a duplicated task"));
        assert_eq!(log.world().tasks[0].state, TaskState::Dropped);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))].state,
            ProposalState::Accepted
        );
    }

    #[test]
    fn agent_accept_writes_nothing() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: "duplicate corpse".into(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: "duplicate of the sibling".into(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            2,
        )
        .unwrap();

        let refused = log.execute(
            agent(),
            Command::AcceptProposal {
                id: ProposalId(RecordId(1)),
            },
            3,
        );
        assert!(matches!(refused, Err(Reject::HumanOnly)));
        assert_eq!(log.records().len(), 2);
        assert_eq!(log.world().tasks[0].state, TaskState::Open);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))].state,
            ProposalState::Open
        );
    }

    #[test]
    fn open_proposals_are_inert_and_stale_accept_fails_atomically() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: "real work".into(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: "not real work".into(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            2,
        )
        .unwrap();

        // claiming the task makes the proposal stale but it still valid
        log.execute(agent(), Command::ClaimTask { id: TaskId(0) }, 3)
            .unwrap();

        // drop from claimed is illegal therefore the accept fails
        let refused = log.execute(
            human(),
            Command::AcceptProposal {
                id: ProposalId(RecordId(1)),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        assert_eq!(log.records().len(), 3);
        assert_eq!(log.world().tasks[0].state, TaskState::Claimed);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))].state,
            ProposalState::Open
        );
    }

    #[test]
    fn re_propose_after_reject_is_free() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: "real work".into(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: "weak evidence".into(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            2,
        )
        .unwrap();
        log.execute(
            human(),
            Command::RejectProposal {
                id: ProposalId(RecordId(1)),
                note: "ruled real work".into(),
            },
            3,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: "better evidence".into(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            4,
        )
        .unwrap();

        assert_eq!(log.world().proposals.len(), 2);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))].state,
            ProposalState::Rejected("ruled real work".into())
        );
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(3))].state,
            ProposalState::Open
        );
        assert_eq!(log.world().tasks[0].state, TaskState::Open);
    }
}
