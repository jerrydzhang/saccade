//! Saccade - issue tracker (idk what makes it special yet other than it's mine)

pub mod db;
pub mod decide;
pub mod events;
pub mod objects;
pub mod paths;
pub mod runner;
pub mod store;
pub mod types;
pub mod views;
pub mod wire;

pub use decide::decide;
pub use events::{Command, Event};
pub use objects::comment::{Addressee, Comment, CommentId, Target};
pub use objects::proposal::{Proposal, ProposalAction, ProposalId, ProposalState};
pub use objects::task::{Task, TaskId, TaskState};
pub use store::{Context, Log, Record, RecordId, Tier, World};
pub use types::actor::ActorName;
pub use types::failure::{FailureCode, FailureEvidence};
pub use types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
pub use types::prose::Prose;

#[derive(Debug)]
pub enum Reject {
    // Task
    InvalidTaskId,
    InvalidParentTaskId,
    // Proposal
    InvalidProposalId,
    ProposalAlreadyOpen,
    InvalidCommentId,
    // Permissions
    HumanOnly,
    NotClaimHolder,
    InvalidIncarnationId,
    IncarnationAlreadyActive,
    DemandNotOnTask,
    WorkspaceAlreadyExists,
    WorkspaceMissing,
    WorktreeAlreadyPresent,
    // Misc
    InvalidActor,
    InvalidStateTransition,
    ReasonRequired,
}

#[cfg(test)]
mod pipeline {
    use super::*;
    use crate::objects::comment::{
        AgentAttemptState, CommentId, CommentState, ResponseState, Target,
    };
    use crate::objects::incarnation::{IncarnationId, IncarnationState};
    use crate::types::actor::ActorName;
    use crate::types::failure::{FailureCode, FailureEvidence};
    use crate::types::pointers::SessionPointer;
    use crate::types::prose::Prose;
    use crate::views::comment_thread;

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

    const RECORD_COUNT: usize = 14;

    fn populate_log(log: &mut Log) {
        let agent_ctx = agent();
        let human_ctx = human();

        log.execute(
            agent_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("implement foo".into()).unwrap(),
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
                name: Prose::new("fix bar".into()).unwrap(),
                parent_id: None,
            },
            3,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(1) }, 4)
            .unwrap();

        log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(1),
                receipt: Prose::new("bar fixed".into()).unwrap(),
            },
            5,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("improve baz".into()).unwrap(),
                parent_id: Some(TaskId(0)),
            },
            6,
        )
        .unwrap();

        log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Prose::new("foo completed successfully".into()).unwrap(),
            },
            7,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("migrate floop".into()).unwrap(),
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
                note: Prose::new("run dead, reclaim".into()).unwrap(),
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
                note: Prose::new("scope covered by fix bar".into()).unwrap(),
            },
            12,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::DropTask {
                id: TaskId(1),
                note: Prose::new("covered by fix bar; kept only as context".into()).unwrap(),
            },
            13,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("open work".into()).unwrap(),
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
            log.world().tasks[0].task.state,
            TaskState::Done(Prose::new("foo completed successfully".into()).unwrap()),
        );
        assert_eq!(log.world().tasks[1].task.state, TaskState::Dropped);
        assert_eq!(log.world().tasks[2].task.state, TaskState::Dropped);
        assert_eq!(log.world().tasks[3].task.state, TaskState::Claimed);
        assert_eq!(log.world().tasks[4].task.state, TaskState::Open);

        assert_eq!(log.records()[8].id.0, 8);
        assert_eq!(log.records()[8].timestamp, 9);
        assert_eq!(log.records()[8].context.actor.as_str(), "saccade bot");

        assert_eq!(log.records()[9].id.0, 9);
        assert_eq!(log.records()[9].timestamp, 10);
        assert_eq!(log.records()[9].context.actor.as_str(), "human person");

        assert_eq!(log.records()[10].id.0, 10);
        assert_eq!(log.records()[10].timestamp, 11);
        assert_eq!(log.records()[10].context.actor.as_str(), "saccade bot");
    }

    #[test]
    fn rejected_command_appends_nothing() {
        let mut log = Log::new();
        populate_log(&mut log);

        // t-3 is dropped: the fold refuses the claim and nothing lands
        let refused = log.execute(agent(), Command::ClaimTask { id: TaskId(3) }, 99);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        assert_eq!(log.records().len(), RECORD_COUNT);
    }

    #[test]
    fn replay_split_equals_whole() {
        let mut log = Log::new();
        populate_log(&mut log);

        for i in 0..log.records().len() {
            let (head, tail) = log.records().split_at(i);
            let mut staged = World::replay(head.to_vec()).unwrap();
            for record in tail {
                staged = staged
                    .apply(record.clone())
                    .expect("decide emitted an unfoldable event");
            }

            assert_eq!(staged, *log.world());
        }
    }

    #[test]
    fn done_and_release_are_holder_gated() {
        let mut log = Log::new();
        populate_log(&mut log);

        let other_agent = Context {
            actor: ActorName::new("other agent".into()).unwrap(),
            tier: Tier::Agent,
        };

        // t-3 is claimed by the agent: only the holder completes
        for (ctx, command) in [
            (
                other_agent.clone(),
                Command::CompleteTask {
                    id: TaskId(3),
                    receipt: Prose::new("not my claim".into()).unwrap(),
                },
            ),
            (
                human(),
                Command::CompleteTask {
                    id: TaskId(3),
                    receipt: Prose::new("not my claim".into()).unwrap(),
                },
            ),
            (
                other_agent.clone(),
                Command::ReleaseTask {
                    id: TaskId(3),
                    note: Prose::new("not my claim".into()).unwrap(),
                },
            ),
        ] {
            let refused = log.execute(ctx, command, 99);
            assert!(matches!(refused, Err(Reject::NotClaimHolder)));
        }
        assert_eq!(log.records().len(), RECORD_COUNT);

        // a human may release any claim
        log.execute(
            human(),
            Command::ReleaseTask {
                id: TaskId(3),
                note: Prose::new("reclaiming for the other agent".into()).unwrap(),
            },
            20,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);
        assert_eq!(log.world().tasks[3].holder, None);

        // an agent may release only its own claim
        log.execute(
            other_agent.clone(),
            Command::ClaimTask { id: TaskId(3) },
            21,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].holder, Some(other_agent.actor.clone()));
        log.execute(
            other_agent.clone(),
            Command::ReleaseTask {
                id: TaskId(3),
                note: Prose::new("handing back".into()).unwrap(),
            },
            22,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);
    }

    #[test]
    fn authority_supersedes_existence_in_rejections() {
        let mut log = Log::new();
        let drop = || Command::DropTask {
            id: TaskId(0),
            note: Prose::new("invalid drop".into()).unwrap(),
        };
        let accept = || Command::AcceptProposal {
            id: ProposalId(RecordId(99)),
        };
        assert!(matches!(
            log.execute(agent(), drop(), 1),
            Err(Reject::HumanOnly)
        ));
        assert!(matches!(
            log.execute(human(), drop(), 1),
            Err(Reject::InvalidTaskId)
        ));
        assert!(matches!(
            log.execute(agent(), accept(), 1),
            Err(Reject::HumanOnly)
        ));
        assert!(matches!(
            log.execute(human(), accept(), 1),
            Err(Reject::InvalidProposalId)
        ));
    }

    #[test]
    fn second_open_proposal_on_one_task_is_refused() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("migrate floop".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            human(),
            Command::CreateProposal {
                name: Prose::new("drop floop instead".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            2,
        )
        .unwrap();

        let second = Command::CreateProposal {
            name: Prose::new("drop floop again".into()).unwrap(),
            action: ProposalAction::Drop { task_id: TaskId(0) },
        };
        assert!(matches!(
            log.execute(human(), second, 3),
            Err(Reject::ProposalAlreadyOpen)
        ));

        log.execute(
            human(),
            Command::RejectProposal {
                id: ProposalId(RecordId(1)),
                note: Prose::new("floop stays".into()).unwrap(),
            },
            4,
        )
        .unwrap();
        let third = Command::CreateProposal {
            name: Prose::new("drop floop for real".into()).unwrap(),
            action: ProposalAction::Drop { task_id: TaskId(0) },
        };
        assert!(log.execute(human(), third, 5).is_ok());
    }

    #[test]
    fn accept_compound_write_two_records_in_order() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("I am going to do floop again".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        let proposed = log
            .execute(
                agent(),
                Command::CreateProposal {
                    name: Prose::new("this is a duplicated task".into()).unwrap(),
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
        assert_eq!(note.as_str(), "this is a duplicated task");
        assert_eq!(log.world().tasks[0].task.state, TaskState::Dropped);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))]
                .proposal
                .state,
            ProposalState::Accepted
        );
    }

    #[test]
    fn agent_accept_writes_nothing() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("duplicate corpse".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: Prose::new("duplicate of the sibling".into()).unwrap(),
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
        assert_eq!(log.world().tasks[0].task.state, TaskState::Open);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))]
                .proposal
                .state,
            ProposalState::Open
        );
    }

    #[test]
    fn open_proposals_are_inert_and_stale_accept_fails_atomically() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("real work".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: Prose::new("not real work".into()).unwrap(),
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
        assert_eq!(log.world().tasks[0].task.state, TaskState::Claimed);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))]
                .proposal
                .state,
            ProposalState::Open
        );
    }

    /// The thread is a walk: targets are stored, depth and membership are
    /// derived, and each task owns exactly its own thread.
    #[test]
    fn incarnation_lifecycle_runs_through_the_machinery_verbs() {
        let mut log = Log::new();
        populate_log(&mut log);
        let session = SessionPointer::new("/tmp/pi-session.jsonl".into()).unwrap();

        // the demand
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("run the suite".into()).unwrap(),
                addressee: Some(Addressee::Agent),
            },
            20,
        )
        .unwrap();
        let demand = CommentId(RecordId(14));
        let trigger = RecordId(14);

        // machinery verbs reject judgment tiers: the role is the only door
        let refused = log.execute(
            agent(),
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger,
                actor: ActorName::new("pi".into()).unwrap(),
                session: session.clone(),
            },
            21,
        );
        assert!(matches!(refused, Err(Reject::HumanOnly)));

        log.execute_system(
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger,
                actor: ActorName::new("pi".into()).unwrap(),
                session: session.clone(),
            },
            21,
        )
        .unwrap();
        let run = IncarnationId(RecordId(15));
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::Bound
        );
        assert_eq!(log.world().tasks[0].active_incarnation, Some(run));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::InFlight { incarnation: run }
            }
        );

        // a second live run on one task is refused
        let refused = log.execute_system(
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger,
                actor: ActorName::new("pi".into()).unwrap(),
                session: session.clone(),
            },
            22,
        );
        assert!(matches!(refused, Err(Reject::IncarnationAlreadyActive)));

        // settling before an accepted prompt is refused
        let refused = log.execute_system(Command::SettleIncarnation { id: run }, 23);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        log.execute_system(Command::AcceptPrompt { id: run }, 24)
            .unwrap();
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::PromptAccepted
        );

        // the session answers; the reply responds but the run holds the slot
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("55 green, nothing flaky".into()).unwrap(),
                addressee: None,
            },
            25,
        )
        .unwrap();
        let reply = CommentId(RecordId(17));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::InFlight { incarnation: run }
            }
        );

        // the run produced its reply, then settled
        log.execute_system(
            Command::MarkRecord {
                incarnation_id: run,
                record_id: RecordId(17),
            },
            26,
        )
        .unwrap();
        log.execute_system(Command::SettleIncarnation { id: run }, 27)
            .unwrap();
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::Settled
        );
        assert_eq!(log.world().tasks[0].active_incarnation, None);
        assert_eq!(log.world().incarnations[&run].produced, vec![RecordId(17)]);
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::Spent
            }
        );
    }

    #[test]
    fn rejected_prompt_terminalizes_and_spends() {
        let mut log = Log::new();
        populate_log(&mut log);
        let session = SessionPointer::new("/tmp/pi-session.jsonl".into()).unwrap();
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("run the flaky one".into()).unwrap(),
                addressee: Some(Addressee::Agent),
            },
            20,
        )
        .unwrap();
        let demand = CommentId(RecordId(14));
        log.execute_system(
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger: RecordId(14),
                actor: ActorName::new("pi".into()).unwrap(),
                session,
            },
            21,
        )
        .unwrap();
        log.execute_system(
            Command::RejectPrompt {
                id: IncarnationId(RecordId(15)),
                evidence: FailureEvidence::new(
                    FailureCode::PromptRejected,
                    Some("session refused the pointer prompt".into()),
                ),
            },
            22,
        )
        .unwrap();
        let run = IncarnationId(RecordId(15));
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::Interrupted
        );
        assert_eq!(log.world().tasks[0].active_incarnation, None);
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Spent
            }
        );

        // a second prompt outcome never lands on the same run
        let refused = log.execute_system(Command::AcceptPrompt { id: run }, 23);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
    }

    #[test]
    fn agent_demands_fold_and_answer_by_exact_tier() {
        let mut log = Log::new();
        populate_log(&mut log);
        let before = log.records().len();

        // the demand is born authorized on its own birth record
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("what is the fold count?".into()).unwrap(),
                addressee: Some(Addressee::Agent),
            },
            20,
        )
        .unwrap();
        let demand = CommentId(RecordId(before));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Authorized {
                    trigger: RecordId(before)
                }
            }
        );

        // a wrong-tier reply lands but does not answer
        log.execute(
            human(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("asking the agent, not you".into()).unwrap(),
                addressee: None,
            },
            21,
        )
        .unwrap();
        assert!(matches!(
            &log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                ..
            }
        ));

        // a deeper descendant never satisfies the ancestor
        let mid = CommentId(RecordId(before + 1));
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(mid),
                body: Prose::new("still gathering".into()).unwrap(),
                addressee: None,
            },
            22,
        )
        .unwrap();
        assert!(matches!(
            &log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                ..
            }
        ));

        // the first exact-tier direct reply answers and spends the attempt
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("fourteen, fixtures unchanged".into()).unwrap(),
                addressee: None,
            },
            23,
        )
        .unwrap();
        let reply = CommentId(RecordId(before + 3));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::Spent,
            }
        );

        // a second exact-tier reply changes nothing
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("also fourteen".into()).unwrap(),
                addressee: None,
            },
            24,
        )
        .unwrap();
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::Spent,
            }
        );
    }

    #[test]
    fn comment_thread_is_derived_from_addresses() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("real work".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("other work".into()).unwrap(),
                parent_id: None,
            },
            2,
        )
        .unwrap();

        log.execute(
            agent(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("triage: how is sections, undecided".into()).unwrap(),
                addressee: None,
            },
            3,
        )
        .unwrap();
        log.execute(
            human(),
            Command::Comment {
                target: Target::Comment(CommentId(RecordId(2))),
                body: Prose::new("no - pure tree, canvas verdict pending".into()).unwrap(),
                addressee: None,
            },
            4,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(CommentId(RecordId(3))),
                body: Prose::new("noted, parked with owner".into()).unwrap(),
                addressee: None,
            },
            5,
        )
        .unwrap();
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(1)),
                body: Prose::new("belongs to the other thread".into()).unwrap(),
                addressee: None,
            },
            6,
        )
        .unwrap();

        let world = log.world();
        assert_eq!(world.comments.len(), 4);
        assert_eq!(
            world.comments[&CommentId(RecordId(2))].comment.target,
            Target::Task(TaskId(0))
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(3))].comment.target,
            Target::Comment(CommentId(RecordId(2)))
        );

        let thread = comment_thread(&world.comments, &world.tasks[0]);
        assert_eq!(
            thread
                .iter()
                .map(|l| (l.seq, l.depth, l.actor.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (2, 1, "saccade bot"),
                (3, 2, "human person"),
                (4, 3, "saccade bot")
            ]
        );
        assert_eq!(comment_thread(&world.comments, &world.tasks[1]).len(), 1);
    }

    /// Comments have no state gate: terminal tasks take them, and only an
    /// unknown comment id refuses.
    #[test]
    fn comments_have_no_state_gate() {
        let mut log = Log::new();
        for name in ["open work", "done work", "dropped work"] {
            log.execute(
                human(),
                Command::CreateTask {
                    name: Prose::new(name.into()).unwrap(),
                    parent_id: None,
                },
                1,
            )
            .unwrap();
        }
        log.execute(human(), Command::ClaimTask { id: TaskId(1) }, 2)
            .unwrap();
        log.execute(
            human(),
            Command::CompleteTask {
                id: TaskId(1),
                receipt: Prose::new("shipped".into()).unwrap(),
            },
            3,
        )
        .unwrap();
        log.execute(
            human(),
            Command::DropTask {
                id: TaskId(2),
                note: Prose::new("run dead".into()).unwrap(),
            },
            4,
        )
        .unwrap();

        let before = log.records().len();
        for (id, state) in [(0, "open"), (1, "done"), (2, "dropped")] {
            log.execute(
                agent(),
                Command::Comment {
                    target: Target::Task(TaskId(id)),
                    body: Prose::new(format!("for the record, on the {state} task")).unwrap(),
                    addressee: None,
                },
                9,
            )
            .unwrap_or_else(|e| panic!("comment on {state} task refused: {e:?}"));
        }
        assert_eq!(log.records().len(), before + 3);

        // only an unknown comment id refuses, writing nothing
        let refused = log.execute(
            human(),
            Command::Comment {
                target: Target::Comment(CommentId(RecordId(99))),
                body: Prose::new("addresses nothing".into()).unwrap(),
                addressee: None,
            },
            10,
        );
        assert!(matches!(refused, Err(Reject::InvalidCommentId)));
        assert_eq!(log.records().len(), before + 3);
    }
}

#[cfg(test)]
mod workspace_pipeline {
    use super::*;
    use crate::objects::task::TaskId;
    use crate::objects::workspace::WorktreeState;
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

    #[test]
    fn workspace_records_run_through_the_machinery_verbs() {
        let mut log = Log::new();
        log.execute(
            Context {
                actor: ActorName::new("human person".into()).unwrap(),
                tier: Tier::Human,
            },
            Command::CreateTask {
                name: Prose::new("run managed demand in a worktree".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();

        // a task with no workspace refuses worktree and checkpoint records
        let base = GitCommit::new("abc123".into()).unwrap();
        let branch = GitBranch::new("saccade/t-0".into()).unwrap();
        let worktree = WorktreePath::new("/repo/wt/t-0".into()).unwrap();
        let refused = log.execute_system(
            Command::CreateWorktree {
                task_id: TaskId(0),
                worktree: worktree.clone(),
            },
            2,
        );
        assert!(matches!(refused, Err(Reject::WorkspaceMissing)));

        // provisioning births the lineage: checkpoint starts at the base
        log.execute_system(
            Command::CreateWorkspace {
                task_id: TaskId(0),
                base: base.clone(),
                branch: branch.clone(),
            },
            2,
        )
        .unwrap();
        let ctx = &log.world().tasks[0];
        assert_eq!(ctx.workspace.as_ref().unwrap().checkpoint, base);
        assert!(matches!(
            ctx.workspace.as_ref().unwrap().worktree,
            WorktreeState::Absent
        ));

        // one lineage per task
        let refused = log.execute_system(
            Command::CreateWorkspace {
                task_id: TaskId(0),
                base: base.clone(),
                branch: branch.clone(),
            },
            3,
        );
        assert!(matches!(refused, Err(Reject::WorkspaceAlreadyExists)));

        // the physical creation is recorded, once
        log.execute_system(
            Command::CreateWorktree {
                task_id: TaskId(0),
                worktree: worktree.clone(),
            },
            4,
        )
        .unwrap();
        let refused = log.execute_system(
            Command::CreateWorktree {
                task_id: TaskId(0),
                worktree: worktree.clone(),
            },
            5,
        );
        assert!(matches!(refused, Err(Reject::WorktreeAlreadyPresent)));
        assert!(matches!(
            log.world().tasks[0].workspace.as_ref().unwrap().worktree,
            WorktreeState::Present(_)
        ));

        // a checkpoint advances; it may advance again
        let head = GitCommit::new("def456".into()).unwrap();
        log.execute_system(
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: head.clone(),
            },
            6,
        )
        .unwrap();
        assert_eq!(
            log.world().tasks[0].workspace.as_ref().unwrap().checkpoint,
            head
        );
    }
}
