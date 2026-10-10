//! The proptest harness: two properties over random legal sequences.
//! no valid-grammar command sequence panics the fold, and the world the
//! write path returns is the one a full reload produces, asserted across
//! three roads: the world `db::record` returns, `World::replay` over the
//! stored records, and the db reloaded through the wire layer. Rejections
//! are part of the contract and always allowed; only panics and divergence
//! fail.

use proptest::prelude::*;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::store::{Record, Scope};
use saccade::types::actor::ActorName;
use saccade::wire;
use saccade::{
    Command, CommentId, CommentKind, Context, Prose, RecordId, Target, TaskId, Tier, World,
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
    Cancel {
        task: u8,
        human: bool,
    },
    Reopen {
        task: u8,
        human: bool,
    },
    Comment {
        task: u8,
        reply: u8,
        variant: u8,
        human: bool,
        scoped: bool,
    },
    Revise {
        comment: u8,
        human: bool,
    },
    WithdrawComment {
        comment: u8,
        human: bool,
    },
    Artifact {
        task: u8,
        human: bool,
    },
}

impl Action {
    fn human(&self) -> bool {
        match self {
            Action::Create { human, .. }
            | Action::Claim { human, .. }
            | Action::Cancel { human, .. }
            | Action::Reopen { human, .. } => *human,
            Action::Comment { human, .. }
            | Action::Revise { human, .. }
            | Action::WithdrawComment { human, .. }
            | Action::Artifact { human, .. } => *human,
        }
    }

    fn number(&self, role: &str) -> u8 {
        match self {
            Action::Create { parent, .. } if role == "parent" => *parent,
            Action::Claim { task, .. }
            | Action::Cancel { task, .. }
            | Action::Reopen { task, .. }
            | Action::Artifact { task, .. }
                if role == "task" =>
            {
                *task
            }
            Action::Comment { task, .. } if role == "task" => *task,
            Action::Comment { reply, .. } if role == "offer" => *reply,
            Action::Revise { comment, .. } if role == "comment" => *comment,
            Action::WithdrawComment { comment, .. } if role == "comment" => *comment,
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

/// Comment ids are log positions of their birth records, so resolution
/// draws from the world's actual key set; the sentinel slot is
/// one-past-the-end, a guaranteed-absent id.
fn comment_at(world: &World, n: u8) -> CommentId {
    let ids: Vec<&CommentId> = world.comments.keys().collect();
    let slot = n as usize % (ids.len() + 1);
    if slot == ids.len() {
        CommentId(RecordId(usize::MAX))
    } else {
        *ids[slot]
    }
}

/// The standing offers of offered tasks — the only offers a decision
/// can move, the delivered tasks of the old law.
fn standing_offer_at(world: &World, n: u8) -> Option<CommentId> {
    let ids: Vec<CommentId> = world
        .tasks
        .iter()
        .filter(|ctx| ctx.current_offer.is_some())
        .filter_map(|ctx| ctx.current_offer)
        .collect();
    let slot = n as usize % (ids.len() + 1);
    ids.get(slot).copied()
}

/// The standing Offered or Concluded accounts, with a one-past-the-end sentinel.
fn reliable_offer_at(world: &World, n: u8) -> Option<CommentId> {
    let ids: Vec<CommentId> = world
        .tasks
        .iter()
        .filter(|ctx| {
            matches!(
                ctx.task.state,
                saccade::TaskState::Offered | saccade::TaskState::Concluded
            )
        })
        .filter_map(|ctx| ctx.current_offer)
        .collect();
    let slot = n as usize % (ids.len() + 1);
    ids.get(slot).copied()
}

fn command_of(action: &Action, world: &World) -> Command {
    let task = task_at(world, action.number("task"));
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
        Action::Claim { .. } => Command::ClaimTask {
            id: task,
            incarnation: None,
        },
        Action::Cancel { .. } => Command::CancelTask {
            id: task,
            explanation: Prose::new("generated explanation".into()).unwrap(),
        },
        Action::Reopen { .. } => Command::ReopenTask {
            id: task,
            explanation: Prose::new("generated explanation".into()).unwrap(),
        },
        Action::Comment { reply, variant, .. } => {
            // a third of decisions name a plain comment (NotAnOffer stays exercised)
            let decision_ref = if reply % 3 == 0 {
                comment_at(world, *reply)
            } else {
                standing_offer_at(world, *reply).unwrap_or(CommentId(RecordId(usize::MAX)))
            };
            Command::Comment {
                target: match variant % 6 {
                    // the broadcast addresses its whole task
                    2 => Target::Task(task),
                    // the decisions address the offer they name
                    3 | 4 => Target::Comment(decision_ref),
                    // the reliance lives on its consuming task
                    5 => Target::Task(task),
                    _ => {
                        if reply % 2 == 0 {
                            Target::Task(task)
                        } else {
                            Target::Comment(comment_at(world, *reply))
                        }
                    }
                },
                kind: match variant % 6 {
                    0 => CommentKind::Note {
                        body: Prose::new("generated note".into()).unwrap(),
                    },
                    1 => CommentKind::Request {
                        body: Prose::new("generated request".into()).unwrap(),
                    },
                    2 => CommentKind::Offer {
                        body: Prose::new("generated account".into()).unwrap(),
                    },
                    3 => CommentKind::Accept {
                        offer: decision_ref,
                    },
                    4 => CommentKind::Reject {
                        offer: decision_ref,
                        grounds: Prose::new("generated grounds".into()).unwrap(),
                    },
                    _ => CommentKind::Use {
                        offer: reliable_offer_at(world, *reply)
                            .unwrap_or(CommentId(RecordId(usize::MAX))),
                        reason: Prose::new("generated reliance".into()).unwrap(),
                    },
                },
            }
        }
        Action::Revise { .. } => Command::ReviseComment {
            id: comment_at(world, action.number("comment")),
            body: Prose::new("generated revision".into()).unwrap(),
        },
        Action::WithdrawComment { .. } => Command::WithdrawComment {
            id: comment_at(world, action.number("comment")),
            note: Prose::new("generated note".into()).unwrap(),
        },
        Action::Artifact { .. } => Command::Artifact {
            root: task,
            artifact: saccade::types::artifact::Artifact {
                name: Prose::new("generated figure".into()).unwrap(),
                hash: saccade::ContentHash::of(b"generated bytes"),
            },
        },
    }
}

fn context(action: &Action, world: &World) -> Context {
    let human = action.human();
    let base = Context {
        actor: if human {
            ActorName::new("jerry".into()).unwrap()
        } else {
            ActorName::new("saccade bot".into()).unwrap()
        },
        tier: if human { Tier::Human } else { Tier::Agent },
        scope: None,
    };
    // a scoped agent declares a task binding: the self-acceptance gate
    // reads it, ownership never does
    if let Action::Comment { scoped: true, .. } = action {
        Context {
            scope: Some(Scope {
                task: task_at(world, action.number("task")),
                incarnation: None,
                workspace: None,
            }),
            ..base
        }
    } else {
        base
    }
}

/// A 60/40 human/agent coin: gated verbs (cancel, reopen) usually
/// execute, while the 40% agent flips keep HumanOnly exercised.
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
        3 => id(|task, human| Action::Claim { task, human }),
        2 => id(|task, human| Action::Cancel { task, human }),
        2 => id(|task, human| Action::Reopen { task, human }),
        8 => (any::<u8>(), any::<u8>(), any::<u8>(), human_coin(), any::<bool>())
            .prop_map(|(task, reply, variant, human, scoped)| {
                Action::Comment { task, reply, variant, human, scoped }
            }),
        2 => id(|comment, human| Action::Revise { comment, human }),
        2 => id(|comment, human| Action::WithdrawComment { comment, human }),
        2 => id(|task, human| Action::Artifact { task, human }),
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
            let ctx = context(action, &live);
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
            let ctx = context(action, &live);
            let cmd = command_of(action, &live);
            match db::record(&mut conn, &ctx, cmd, 1) {
                Ok((stored, world)) => {
                    live = world;
                    let last = stored.last().unwrap();
                    match action {
                        Action::Cancel { human: true, .. } => {
                            if last.kind == "task_cancelled" {
                                *milestones.entry("cancel").or_default() += 1;
                            }
                        }
                        Action::Reopen { human: true, .. } => {
                            if last.kind == "task_reopened" {
                                *milestones.entry("reopen").or_default() += 1;
                            }
                        }
                        Action::Claim { .. } => {
                            *milestones.entry("claim").or_default() += 1;
                        }
                        Action::Comment { variant, .. } => {
                            // the variant grammar must actually land
                            let name = match variant % 6 {
                                0 => "note comment",
                                1 => "request comment",
                                2 => "offer comment",
                                3 => "accept comment",
                                4 => "reject comment",
                                _ => "use comment",
                            };
                            *milestones.entry(name).or_default() += 1;
                        }
                        Action::Artifact { .. } => {
                            *milestones.entry("artifact").or_default() += 1;
                        }
                        Action::Revise { .. } => {
                            *milestones.entry("revise").or_default() += 1;
                        }
                        Action::WithdrawComment { .. } => {
                            *milestones.entry("comment withdraw").or_default() += 1;
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
                        saccade::Reject::InvalidTaskId => "InvalidTaskId",
                        saccade::Reject::InvalidParentTaskId => "InvalidParentTaskId",
                        saccade::Reject::InvalidCommentId => "InvalidCommentId",
                        saccade::Reject::NotAnOffer => "NotAnOffer",
                        saccade::Reject::NotCurrentOffer => "NotCurrentOffer",
                        saccade::Reject::SelfReliance => "SelfReliance",
                        saccade::Reject::SelfAcceptance => "SelfAcceptance",
                        saccade::Reject::InvalidIncarnationId => "InvalidIncarnationId",
                        saccade::Reject::InvalidStateTransition => "InvalidStateTransition",
                        saccade::Reject::ReasonRequired => "ReasonRequired",
                        saccade::Reject::InvalidActor => "InvalidActor",
                        other => panic!("an unreachable reject appeared: {other:?}"),
                    };
                    *reject_kinds.entry(kind).or_default() += 1;
                }
            }
        }

        let world = &live;
        let states: Vec<&str> = (0..world.tasks.len())
            .map(|i| saccade::views::task_view(world, TaskId(i)).unwrap().state)
            .collect();
        if states.contains(&"offered") {
            *milestones.entry("offered task").or_default() += 1;
        }
        if states.contains(&"concluded") {
            *milestones.entry("concluded task").or_default() += 1;
        }
        if states.contains(&"cancelled") {
            *milestones.entry("cancelled task").or_default() += 1;
        }
        if max_depth >= 2 {
            *milestones.entry("depth 2 thread").or_default() += 1;
        }
    }

    // floors sit at half the measured rates so normal RNG wobble cannot trip them
    for (name, floor) in [
        ("cancel", 0.05),
        ("reopen", 0.005),
        ("claim", 0.10),
        ("concluded task", 0.05),
        ("cancelled task", 0.15),
        ("offered task", 0.10),
        ("depth 2 thread", 0.02),
        ("offer comment", 0.10),
        ("accept comment", 0.03),
        ("reject comment", 0.03),
        ("use comment", 0.05),
        ("request comment", 0.10),
        ("artifact", 0.05),
        ("revise", 0.05),
        ("comment withdraw", 0.015),
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
        "InvalidCommentId",
        "NotAnOffer",
        "NotCurrentOffer",
        "SelfReliance",
        "InvalidStateTransition",
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
            context: Context {
                actor: ActorName::new(row.actor.clone()).unwrap(),
                tier: wire::tier_from(&row.tier).unwrap(),
                scope: row
                    .scope
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .unwrap(),
            },
            event: wire::assemble(&row.kind, &row.payload).unwrap(),
        })
        .collect()
}
