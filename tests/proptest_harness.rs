//! The proptest harness: two properties over random legal sequences.
//! no valid-grammar command sequence panics the fold, and the world the
//! write path returns is the one a full reload produces, asserted across
//! three roads: the world `db::record` returns, `World::replay` over the
//! stored records, and the db reloaded through the wire layer. Rejections
//! are part of the contract and always allowed; only panics and divergence
//! fail.

use proptest::prelude::*;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::store::Record;
use saccade::types::actor::ActorName;
use saccade::wire;
use saccade::{
    Command, CommentKind, Context, ProposalAction, ProposalId, Prose, RecordId, Target, TaskId,
    Tier, World,
};

#[derive(Clone, Debug)]
enum Action {
    Create {
        parent: u8,
        human: bool,
    },
    Claim {
        task: u8,
        human: bool,
    },
    Done {
        task: u8,
        human: bool,
    },
    AcceptTask {
        task: u8,
        human: bool,
    },
    Drop {
        task: u8,
        human: bool,
    },
    Release {
        task: u8,
        human: bool,
    },
    Comment {
        task: u8,
        reply: u8,
        variant: u8,
        human: bool,
    },
    ProposeDrop {
        task: u8,
        human: bool,
    },
    ProposeRelease {
        task: u8,
        human: bool,
    },
    Accept {
        proposal: u8,
        human: bool,
    },
    Reject {
        proposal: u8,
        human: bool,
    },
    Withdraw {
        proposal: u8,
        human: bool,
    },
    Cancel {
        incarnation: u8,
        human: bool,
    },
}

impl Action {
    fn human(&self) -> bool {
        match self {
            Action::Create { human, .. }
            | Action::Claim { human, .. }
            | Action::Done { human, .. }
            | Action::AcceptTask { human, .. }
            | Action::Drop { human, .. }
            | Action::Release { human, .. }
            | Action::ProposeDrop { human, .. }
            | Action::ProposeRelease { human, .. }
            | Action::Accept { human, .. }
            | Action::Reject { human, .. }
            | Action::Withdraw { human, .. } => *human,
            Action::Cancel { human, .. } => *human,
            Action::Comment { human, .. } => *human,
        }
    }

    fn number(&self, role: &str) -> u8 {
        match self {
            Action::Create { parent, .. } if role == "parent" => *parent,
            Action::Claim { task, .. }
            | Action::Done { task, .. }
            | Action::AcceptTask { task, .. }
            | Action::Drop { task, .. }
            | Action::Release { task, .. }
            | Action::ProposeDrop { task, .. }
            | Action::ProposeRelease { task, .. }
                if role == "task" =>
            {
                *task
            }
            Action::Comment { task, .. } if role == "task" => *task,
            Action::Accept { proposal, .. }
            | Action::Reject { proposal, .. }
            | Action::Withdraw { proposal, .. }
                if role == "proposal" =>
            {
                *proposal
            }
            Action::Cancel { incarnation, .. } if role == "incarnation" => *incarnation,
            _ => 0,
        }
    }
}

/// Resolves an abstract number against the live world; the `% len + 1`
/// deliberately lands one slot out of range so near-miss ids (a tenth of
/// draws on small worlds) exercise the reject paths.
fn task_at(world: &World, n: u8) -> TaskId {
    TaskId(n as usize % (world.tasks.len() + 1))
}

/// Proposal and comment ids are log positions of their birth records, so
/// resolution draws from the world's actual key set; the sentinel slot
/// (`== ids.len()` after modulo) is one-past-the-end, a guaranteed-absent id.
fn proposal_at(world: &World, n: u8) -> ProposalId {
    let ids: Vec<&ProposalId> = world.proposals.keys().collect();
    let slot = n as usize % (ids.len() + 1);
    if slot == ids.len() {
        ProposalId(RecordId(usize::MAX))
    } else {
        *ids[slot]
    }
}

fn incarnation_at(world: &World, n: u8) -> saccade::objects::incarnation::IncarnationId {
    let ids: Vec<&saccade::objects::incarnation::IncarnationId> =
        world.incarnations.keys().collect();
    let slot = n as usize % (ids.len() + 1);
    if slot == ids.len() {
        saccade::objects::incarnation::IncarnationId(RecordId(usize::MAX))
    } else {
        *ids[slot]
    }
}

/// Task-accept resolution draws from the world's delivered tasks — the
/// only state the door opens from — with the same one-past-the-end
/// sentinel the other id resolutions carry.
fn delivered_task_at(world: &World, n: u8) -> TaskId {
    let ids: Vec<TaskId> = world
        .tasks
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            saccade::views::task_view(world, TaskId(*i)).is_some_and(|v| v.state == "delivered")
        })
        .map(|(i, _)| TaskId(i))
        .collect();
    let slot = n as usize % (ids.len() + 1);
    if slot == ids.len() {
        TaskId(usize::MAX)
    } else {
        ids[slot]
    }
}

fn comment_at(world: &World, n: u8) -> saccade::CommentId {
    let ids: Vec<&saccade::CommentId> = world.comments.keys().collect();
    let slot = n as usize % (ids.len() + 1);
    if slot == ids.len() {
        saccade::CommentId(RecordId(usize::MAX))
    } else {
        *ids[slot]
    }
}

fn command_of(action: &Action, world: &World) -> Command {
    let task = task_at(world, action.number("task"));
    let proposal = proposal_at(world, action.number("proposal"));
    match action {
        Action::Create { parent, .. } => Command::CreateTask {
            name: Prose::new("generated name".into()).unwrap(),
            // the first task of a world is always a root; after that a quarter
            // of creates stay roots and the rest resolve against the live world
            parent_id: if world.tasks.is_empty() {
                None
            } else {
                (parent % 4 == 0).then(|| task_at(world, *parent))
            },
        },
        Action::Claim { .. } => Command::ClaimTask { id: task },
        Action::Done { .. } => Command::CompleteTask {
            id: task,
            receipt: Prose::new("generated receipt".into()).unwrap(),
        },
        Action::AcceptTask { .. } => Command::AcceptTask {
            id: delivered_task_at(world, action.number("task")),
        },
        Action::Drop { .. } => Command::DropTask {
            id: task,
            note: Prose::new("generated note".into()).unwrap(),
        },
        Action::Release { .. } => Command::ReleaseTask {
            id: task,
            note: Prose::new("generated note".into()).unwrap(),
        },
        Action::Comment { reply, variant, .. } => Command::Comment {
            target: if reply % 2 == 0 {
                Target::Task(task)
            } else {
                Target::Comment(comment_at(world, *reply))
            },
            body: Prose::new("generated comment".into()).unwrap(),
            kind: match variant % 4 {
                0 => CommentKind::Note,
                1 => CommentKind::Demand,
                2 => CommentKind::Steer,
                _ => CommentKind::Ask,
            },
        },
        Action::ProposeDrop { .. } => Command::CreateProposal {
            name: Prose::new("generated proposal".into()).unwrap(),
            action: ProposalAction::Drop { task_id: task },
        },
        Action::ProposeRelease { .. } => Command::CreateProposal {
            name: Prose::new("generated proposal".into()).unwrap(),
            action: ProposalAction::Release { task_id: task },
        },
        Action::Accept { .. } => Command::AcceptProposal { id: proposal },
        Action::Reject { .. } => Command::RejectProposal {
            id: proposal,
            note: Prose::new("generated ruling".into()).unwrap(),
        },
        Action::Withdraw { .. } => Command::WithdrawProposal {
            id: proposal,
            note: Prose::new("generated note".into()).unwrap(),
        },
        Action::Cancel { .. } => Command::CancelIncarnation {
            id: incarnation_at(world, action.number("incarnation")),
        },
    }
}

fn context(human: bool) -> Context {
    Context {
        actor: if human {
            ActorName::new("jerry".into()).unwrap()
        } else {
            ActorName::new("saccade bot".into()).unwrap()
        },
        tier: if human { Tier::Human } else { Tier::Agent },
    }
}

/// A 60/40 human/agent coin: gated verbs (drop, release, accept, reject)
/// usually execute, while the 40% agent flips keep HumanOnly exercised.
fn human_coin() -> impl Strategy<Value = bool> {
    any::<u8>().prop_map(|n| n % 5 < 3)
}

fn create_strategy() -> BoxedStrategy<Action> {
    (any::<u8>(), human_coin())
        .prop_map(|(parent, human)| Action::Create { parent, human })
        .boxed()
}

fn action_strategy() -> BoxedStrategy<Action> {
    let id = |f: fn(u8, bool) -> Action| {
        (any::<u8>(), human_coin()).prop_map(move |(n, human)| f(n, human))
    };
    prop_oneof![
        3 => create_strategy(),
        2 => id(|task, human| Action::Claim { task, human }),
        2 => id(|task, human| Action::Done { task, human }),
        2 => id(|task, human| Action::AcceptTask { task, human }),
        2 => id(|task, human| Action::Drop { task, human }),
        2 => id(|task, human| Action::Release { task, human }),
        3 => (any::<u8>(), any::<u8>(), any::<u8>(), human_coin())
            .prop_map(|(task, reply, variant, human)| { Action::Comment { task, reply, variant, human } }),
        3 => id(|task, human| Action::ProposeDrop { task, human }),
        2 => id(|task, human| Action::ProposeRelease { task, human }),
        3 => id(|proposal, human| Action::Accept { proposal, human }),
        3 => id(|proposal, human| Action::Reject { proposal, human }),
        2 => id(|proposal, human| Action::Withdraw { proposal, human }),
        1 => id(|incarnation, human| Action::Cancel { incarnation, human }),
    ]
    .boxed()
}

/// A create prologue so every sequence starts on a populated world; rejection
/// traffic alone cannot starve the state space back to empty.
fn sequence_strategy() -> BoxedStrategy<Vec<Action>> {
    (
        proptest::collection::vec(create_strategy(), 1..4),
        proptest::collection::vec(action_strategy(), 0..48),
    )
        .prop_map(|(mut prologue, rest)| {
            prologue.extend(rest);
            prologue
        })
        .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn random_sequences_never_panic_and_replay_equals_live(actions in sequence_strategy()) {
        let mut conn = db::open(std::path::Path::new(":memory:")).expect(":memory: opens");
        let mut live = World::new();

        for action in &actions {
            let ctx = context(action.human());
            let cmd = command_of(action, &live);
            // rejections are gatekept by the pipeline and simply ignored; the property
            // is that no valid-grammar command sequence panics the fold
            if let Ok((_, world)) = db::record(&mut conn, &ctx, cmd, 1) {
                live = world;
            }
        }

        // road two: the pure fold over the stored records
        let loadout = db::load(&conn).expect("load after generated sequence");
        let rows = loadout.rows.clone();
        let records = stored_to_records(&rows);
        let replayed = World::replay(records).unwrap();
        prop_assert_eq!(&replayed, &live);

        // road three: the persisted log reloaded through the wire layer
        match loadout.state {
            LoadState::Full(world) => prop_assert_eq!(&world, &live),
            LoadState::Degraded(reason) => panic!("wire round-trip degraded the log: {reason}"),
        }
    }
}

/// The guard against silent shallowness: a green property test over empty
/// worlds proves nothing. A fixed deterministic run must reach every deep
/// milestone in a healthy fraction of cases, or the generator has starved.
#[test]
fn generator_reaches_deep_states() {
    use proptest::strategy::ValueTree;
    use std::collections::BTreeMap;

    let cases = 256;
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let mut milestones = BTreeMap::<&str, usize>::new();
    let mut reject_kinds = BTreeMap::<&str, usize>::new();

    for _ in 0..cases {
        let actions = sequence_strategy()
            .new_tree(&mut runner)
            .expect("strategy infallible")
            .current();
        let mut conn = db::open(std::path::Path::new(":memory:")).unwrap();
        let mut live = World::new();
        let mut depth: BTreeMap<saccade::CommentId, usize> = BTreeMap::new();
        let mut max_depth = 0;

        for action in &actions {
            let ctx = context(action.human());
            let cmd = command_of(action, &live);
            match db::record(&mut conn, &ctx, cmd, 1) {
                Ok((stored, world)) => {
                    live = world;
                    let last = stored.last().unwrap();
                    match action {
                        Action::Accept { .. } => *milestones.entry("accept").or_default() += 1,
                        Action::Done { .. } => {
                            if last.kind == "task_delivered" {
                                *milestones.entry("deliver").or_default() += 1;
                            }
                        }
                        Action::AcceptTask { .. } => {
                            *milestones.entry("task accept").or_default() += 1;
                        }
                        Action::Reject { human: true, .. } => {
                            *milestones.entry("human reject").or_default() += 1;
                        }
                        Action::Withdraw { .. } => *milestones.entry("withdraw").or_default() += 1,
                        Action::Comment { variant, .. } => {
                            // the variant grammar must actually land
                            let name = match variant % 4 {
                                0 => "note comment",
                                1 => "demand comment",
                                2 => "steer comment",
                                _ => "ask comment",
                            };
                            *milestones.entry(name).or_default() += 1;
                        }
                        _ => {}
                    }
                    if last.kind == "commented" {
                        let record = stored_to_records(std::slice::from_ref(last)).remove(0);
                        let d = match record.event {
                            saccade::Event::Commented { target, .. } => match target {
                                saccade::Target::Task(_) => 1,
                                saccade::Target::Comment(parent) => {
                                    depth.get(&parent).copied().unwrap_or(1) + 1
                                }
                            },
                            _ => unreachable!("kind checked before decoding"),
                        };
                        depth.insert(saccade::CommentId(record.id), d);
                        max_depth = max_depth.max(d);
                    }
                }
                Err(ref fail) => {
                    let r = reject_of(fail);
                    let kind = match r {
                        saccade::Reject::HumanOnly => "HumanOnly",
                        saccade::Reject::NotClaimHolder => "NotClaimHolder",
                        saccade::Reject::InvalidIncarnationId => "InvalidIncarnationId",
                        saccade::Reject::IncarnationAlreadyActive => "IncarnationAlreadyActive",
                        saccade::Reject::DemandNotOnTask => "DemandNotOnTask",
                        saccade::Reject::WorkspaceAlreadyExists => "WorkspaceAlreadyExists",
                        saccade::Reject::WorkspaceMissing => "WorkspaceMissing",
                        saccade::Reject::WorktreeAlreadyPresent => "WorktreeAlreadyPresent",
                        saccade::Reject::CheckpointRewind => "CheckpointRewind",
                        saccade::Reject::InvalidTaskId => "InvalidTaskId",
                        saccade::Reject::InvalidParentTaskId => "InvalidParentTaskId",
                        saccade::Reject::InvalidProposalId => "InvalidProposalId",
                        saccade::Reject::ProposalAlreadyOpen => "ProposalAlreadyOpen",
                        saccade::Reject::InvalidCommentId => "InvalidCommentId",
                        saccade::Reject::NotBirthAttribution => "NotBirthAttribution",
                        saccade::Reject::InvalidStateTransition => "InvalidStateTransition",
                        saccade::Reject::ReasonRequired => "ReasonRequired",
                        saccade::Reject::InvalidActor => "InvalidActor",
                        saccade::Reject::SteerNotStanding => "SteerNotStanding",
                        saccade::Reject::NoActiveIncarnation => "NoActiveIncarnation",
                    };
                    *reject_kinds.entry(kind).or_default() += 1;
                }
            }
        }

        let world = &live;
        let states: Vec<&str> = (0..world.tasks.len())
            .map(|i| saccade::views::task_view(world, TaskId(i)).unwrap().state)
            .collect();
        if states.contains(&"delivered") {
            *milestones.entry("delivered task").or_default() += 1;
        }
        if states.contains(&"dropped") {
            *milestones.entry("dropped task").or_default() += 1;
        }
        if max_depth >= 2 {
            *milestones.entry("depth 2 thread").or_default() += 1;
        }
    }

    // floors sit at half the measured rates so normal RNG wobble cannot trip them
    for (name, floor) in [
        ("accept", 0.10),
        ("human reject", 0.10),
        ("withdraw", 0.10),
        ("deliver", 0.04),
        ("task accept", 0.008),
        ("delivered task", 0.03),
        ("dropped task", 0.30),
        ("depth 2 thread", 0.04),
        ("demand comment", 0.10),
        ("steer comment", 0.05),
        ("ask comment", 0.05),
    ] {
        let hit = milestones.get(name).copied().unwrap_or(0);
        assert!(
            hit as f64 >= floor * cases as f64,
            "generator starved {name}: {hit}/{cases} cases reached it"
        );
    }
    // every reject kind the grammar can produce must appear at all
    for kind in [
        "HumanOnly",
        "InvalidTaskId",
        "InvalidParentTaskId",
        "InvalidProposalId",
        "InvalidCommentId",
        "InvalidStateTransition",
        "NotBirthAttribution",
    ] {
        assert!(
            reject_kinds.get(kind).copied().unwrap_or(0) > 0,
            "generator never produced {kind}"
        );
    }
}

fn reject_of(fail: &ExecuteFail) -> saccade::Reject {
    match fail {
        ExecuteFail::Reject(r) => r.clone(),
        other => panic!("the generated sequence never hits storage failures: {other}"),
    }
}

fn stored_to_records(rows: &[StoredRecord]) -> Vec<Record> {
    rows.iter()
        .map(|row| Record {
            id: saccade::RecordId(row.seq),
            timestamp: row.event_time,
            context: saccade::Context {
                actor: ActorName::new(row.actor.clone()).unwrap(),
                tier: wire::tier_from(&row.tier).unwrap(),
            },
            event: wire::assemble(&row.kind, &row.payload).unwrap(),
        })
        .collect()
}
