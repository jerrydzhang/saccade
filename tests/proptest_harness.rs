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
use saccade::{Command, CommentId, CommentKind, Context, Prose, RecordId, TaskId, Tier, World};

#[derive(Clone, Debug)]
enum Action {
    Create {
        human: bool,
    },
    Claim {
        task: u8,
        human: bool,
        /// Draws that name the task's live incarnation when one is born
        incarnation: u8,
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
        /// Draws that name citation targets, marker order
        cites: Vec<u8>,
    },
    BindUse {
        use_draw: u8,
        offer: u8,
        human: bool,
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
    Establish {
        task: u8,
    },
    Register {
        draw: u8,
    },
    Send {
        from: u8,
        to: u8,
        reply: u8,
    },
    Submit {
        message: u8,
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
            | Action::BindUse { human, .. }
            | Action::Revise { human, .. }
            | Action::WithdrawComment { human, .. }
            | Action::Artifact { human, .. } => *human,
            Action::Establish { .. }
            | Action::Register { .. }
            | Action::Send { .. }
            | Action::Submit { .. } => false,
        }
    }

    fn number(&self, role: &str) -> u8 {
        match self {
            Action::Claim { task, .. }
            | Action::Cancel { task, .. }
            | Action::Reopen { task, .. }
            | Action::Artifact { task, .. }
            | Action::Establish { task, .. }
                if role == "task" =>
            {
                *task
            }
            Action::Comment { task, .. } if role == "task" => *task,
            Action::Comment { reply, .. } if role == "offer" => *reply,
            Action::BindUse { use_draw, .. } if role == "comment" => *use_draw,
            Action::BindUse { offer, .. } if role == "offer" => *offer,
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

/// The uses the fold holds, with a one-past-the-end sentinel: a
/// binding names a use, and the guard fires on every other variant.
fn use_comment_at(world: &World, n: u8) -> CommentId {
    let ids: Vec<CommentId> = world
        .comments
        .iter()
        .filter(|(_, ctx)| matches!(ctx.comment.kind, CommentKind::Use { .. }))
        .map(|(id, _)| *id)
        .collect();
    let slot = n as usize % (ids.len() + 1);
    if slot == ids.len() {
        CommentId(RecordId(usize::MAX))
    } else {
        ids[slot]
    }
}

/// Any record position the log already holds — births and record
/// facts alike — with a one-past-the-end sentinel that exercises the
/// citation guard. `records` is the log's own length.
fn record_ref_at(records: usize, n: u8) -> RecordId {
    let slot = n as usize % (records + 1);
    if slot == records {
        RecordId(usize::MAX)
    } else {
        RecordId(slot)
    }
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

/// The historical incarnation born with a task's establishment; the
/// sentinel slot draws a guaranteed-absent reference.
fn incarnation_at(world: &World, task: TaskId, n: u8) -> Option<saccade::IncarnationId> {
    let born: Vec<saccade::IncarnationId> = world
        .incarnations
        .iter()
        .filter(|(_, ctx)| {
            matches!(
                ctx.incarnation,
                saccade::Incarnation::Established { task: born, .. } if born == task
            )
        })
        .map(|(id, _)| *id)
        .collect();
    let slot = n as usize % (born.len() + 1);
    if slot == born.len() {
        Some(saccade::IncarnationId(RecordId(usize::MAX)))
    } else {
        Some(born[slot])
    }
}

/// The registered native identities, in birth order.
fn native_at(world: &World, n: u8) -> Option<saccade::IncarnationId> {
    let born: Vec<saccade::IncarnationId> = world
        .incarnations
        .iter()
        .filter(|(_, ctx)| matches!(ctx.incarnation, saccade::Incarnation::Native { .. }))
        .map(|(id, _)| *id)
        .collect();
    let slot = n as usize % (born.len() + 1);
    born.get(slot).copied()
}

/// The messages the fold holds, with a one-past-the-end sentinel.
fn message_at(world: &World, n: u8) -> Option<saccade::MessageId> {
    let ids: Vec<saccade::MessageId> = world.messages.keys().copied().collect();
    let slot = n as usize % (ids.len() + 1);
    ids.get(slot).copied()
}

fn command_of(action: &Action, world: &World, records: usize) -> Command {
    let task = task_at(world, action.number("task"));
    match action {
        Action::Create { .. } => Command::CreateTask {
            name: Prose::new("generated name".into()).unwrap(),
        },
        Action::Claim { incarnation, .. } => Command::ClaimTask {
            id: task,
            incarnation: (*incarnation % 3 == 0)
                .then(|| incarnation_at(world, task, *incarnation))
                .flatten(),
        },
        Action::Cancel { .. } => Command::CancelTask {
            id: task,
            explanation: Prose::new("generated explanation".into()).unwrap(),
        },
        Action::Reopen { .. } => Command::ReopenTask {
            id: task,
            explanation: Prose::new("generated explanation".into()).unwrap(),
        },
        Action::Comment {
            reply,
            variant,
            cites,
            ..
        } => {
            // a third of decisions name a plain comment (NotAnOffer stays exercised)
            let decision_ref = if reply % 3 == 0 {
                comment_at(world, *reply)
            } else {
                standing_offer_at(world, *reply).unwrap_or(CommentId(RecordId(usize::MAX)))
            };
            // notes and requests cite earlier records through their
            // mappings: births and record facts alike
            let references: Vec<RecordId> = match variant % 5 {
                0 => cites.iter().map(|n| record_ref_at(records, *n)).collect(),
                _ => Vec::new(),
            };
            // a decision homes on the task of the offer it names; the
            // near-miss draw keeps the guard's refusal exercised
            let decision_home = world
                .comments
                .get(&decision_ref)
                .map(|c| c.comment.home)
                .unwrap_or(task);
            let home = match variant % 5 {
                2 | 3 => decision_home,
                _ => task,
            };
            Command::Comment {
                home,
                kind: match variant % 5 {
                    0 => CommentKind::Note {
                        body: Prose::new("generated note".into()).unwrap(),
                    },
                    1 => CommentKind::Offer {
                        body: Prose::new("generated account".into()).unwrap(),
                    },
                    2 => CommentKind::Accept {
                        offer: decision_ref,
                    },
                    3 => CommentKind::Reject {
                        offer: decision_ref,
                        grounds: Prose::new("generated grounds".into()).unwrap(),
                    },
                    // the reliance declares its source task, prospective
                    _ => CommentKind::Use {
                        source: task_at(world, *reply),
                        reason: Prose::new("generated reliance".into()).unwrap(),
                        bound: None,
                    },
                },
                references,
            }
        }
        Action::BindUse {
            use_draw, offer, ..
        } => {
            // the selection names the use's declared source task's
            // standing account: the draw targets the real seam, and a
            // zero offer byte keeps the sentinel's refusal traffic
            let id = use_comment_at(world, *use_draw);
            let source = world
                .comments
                .get(&id)
                .and_then(|c| c.comment.kind.source());
            let selected = match offer {
                0 => None,
                _ => source
                    .and_then(|src| world.tasks.get(src.0))
                    .and_then(|t| t.current_offer),
            };
            Command::BindUse {
                id,
                offer: selected.unwrap_or(CommentId(RecordId(usize::MAX))),
                grounds: Prose::new("generated grounds".into()).unwrap(),
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
        Action::Establish { .. } => Command::EstablishWorkspace {
            task_id: task,
            base: saccade::GitCommit::new(format!("abc{:03}", task.0)).unwrap(),
            branch: saccade::GitBranch::new(format!("saccade/t-{}", task.0)).unwrap(),
            worktree: saccade::WorktreePath::new(format!("/generated/wt/t-{}", task.0).into())
                .unwrap(),
        },
        Action::Register { draw } => Command::RegisterIncarnation {
            native_id: saccade::NativeSessionId::new(format!("native-{draw:02}")).unwrap(),
            saved_session: saccade::SessionLocator::new(
                format!("/generated/sessions/native-{draw:02}.jsonl").into(),
            )
            .unwrap(),
        },
        Action::Send { from, to, reply } => Command::SendMessage {
            from: native_at(world, *from).unwrap_or(saccade::IncarnationId(RecordId(usize::MAX))),
            to: native_at(world, *to).unwrap_or(saccade::IncarnationId(RecordId(usize::MAX))),
            reply_to: message_at(world, *reply),
            task: None,
            body: saccade::Prose::new("generated message".into()).unwrap(),
        },
        Action::Submit { message } => {
            let id = message_at(world, *message);
            let by = id
                .and_then(|mid| world.messages.get(&mid))
                .map(|m| m.message.to);
            Command::SubmitMessage {
                id: id.unwrap_or(saccade::MessageId(RecordId(usize::MAX))),
                by: by.unwrap_or(saccade::IncarnationId(RecordId(usize::MAX))),
            }
        }
    }
}

fn context(action: &Action, world: &World) -> Context {
    // provisioning and delivery bookkeeping are the machinery's own acts
    if let Action::Establish { .. } | Action::Register { .. } | Action::Submit { .. } = action {
        return Context::system();
    }
    // a send rides its sender's registered native identity
    if let Action::Send { from, .. } = action {
        return match native_at(world, *from) {
            Some(id) => Context {
                actor: ActorName::new("saccade bot".into()).unwrap(),
                tier: Tier::Agent,
                scope: Some(Scope {
                    task: None,
                    incarnation: Some(id),
                    workspace: None,
                }),
            },
            // no registered sender: the draw lands as a refused system write
            None => Context::system(),
        };
    }
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
                task: Some(task_at(world, action.number("task"))),
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
    human_coin()
        .prop_map(|human| Action::Create { human })
        .boxed()
}

fn action_strategy() -> BoxedStrategy<Action> {
    let id = |f: fn(u8, bool) -> Action| {
        (any::<u8>(), human_coin()).prop_map(move |(n, human)| f(n, human))
    };
    prop_oneof![
        3 => create_strategy(),
        3 => (any::<u8>(), human_coin(), any::<u8>()).prop_map(|(task, human, incarnation)| {
            Action::Claim { task, human, incarnation }
        }),
        2 => id(|task, human| Action::Cancel { task, human }),
        2 => id(|task, human| Action::Reopen { task, human }),
        8 => (
            any::<u8>(),
            any::<u8>(),
            any::<u8>(),
            human_coin(),
            any::<bool>(),
            proptest::collection::vec(any::<u8>(), 0..2),
        )
            .prop_map(|(task, reply, variant, human, scoped, cites)| {
                Action::Comment {
                    task,
                    reply,
                    variant,
                    human,
                    scoped,
                    cites,
                }
            }),
        4 => (any::<u8>(), any::<u8>(), human_coin()).prop_map(|(use_draw, offer, human)| {
            Action::BindUse {
                use_draw,
                offer,
                human,
            }
        }),
        2 => id(|comment, human| Action::Revise { comment, human }),
        2 => id(|comment, human| Action::WithdrawComment { comment, human }),
        2 => id(|task, human| Action::Artifact { task, human }),
        2 => any::<u8>().prop_map(|task| Action::Establish { task }),
        2 => any::<u8>().prop_map(|draw| Action::Register { draw }),
        3 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(from, to, reply)| Action::Send {
            from,
            to,
            reply,
        }),
        2 => any::<u8>().prop_map(|message| Action::Submit { message }),
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
        let mut records = 0usize;

        for action in &actions {
            let ctx = context(action, &live);
            let cmd = command_of(action, &live, records);
            // rejections are gatekept by the pipeline and simply ignored; the property
            // is that no valid-grammar command sequence panics the fold
            if let Ok((stored, world)) = db::record(&mut conn, &ctx, cmd, 1) {
                records += stored.len();
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

        // the task wait matches every record the fold homes on a thread
        let index = saccade::views::thread_index(&live);
        for row in &rows {
            let event = wire::assemble(&row.kind, &row.payload).expect("the loaded log assembles");
            if let Some(home) = index.get(&row.seq) {
                prop_assert!(
                    saccade::views::wait_matches(
                        &live,
                        saccade::RecordId(row.seq),
                        &event,
                        &saccade::Target::Task(*home),
                        saccade::views::WaitUntil::Activity
                    ),
                    "seq {} folds into t-{} but the task wait does not match it",
                    row.seq,
                    home.0
                );
            }
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
        let mut records = 0usize;

        for action in &actions {
            let ctx = context(action, &live);
            let cmd = command_of(action, &live, records);
            match db::record(&mut conn, &ctx, cmd, 1) {
                Ok((stored, world)) => {
                    records += stored.len();
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
                        Action::Establish { .. } => {
                            if last.kind == "workspace_established" {
                                *milestones.entry("workspace established").or_default() += 1;
                            }
                        }
                        Action::Comment { variant, .. } => {
                            // the variant grammar must actually land
                            let name = match variant % 5 {
                                0 => "note comment",
                                1 => "offer comment",
                                2 => "accept comment",
                                3 => "reject comment",
                                _ => "use comment",
                            };
                            *milestones.entry(name).or_default() += 1;
                            if last.payload.contains("\"references\":[") {
                                *milestones.entry("citing comment").or_default() += 1;
                            }
                        }
                        Action::BindUse { .. } => {
                            if last.kind == "use_bound" {
                                *milestones.entry("use bind").or_default() += 1;
                            }
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
                        Action::Register { .. } => {
                            if last.kind == "incarnation_registered" {
                                *milestones.entry("native registered").or_default() += 1;
                            }
                        }
                        Action::Send { .. } => {
                            if last.kind == "message_sent" {
                                *milestones.entry("message sent").or_default() += 1;
                            }
                        }
                        Action::Submit { .. } => {
                            if last.kind == "message_submitted" {
                                *milestones.entry("message submitted").or_default() += 1;
                            }
                        }
                        _ => {}
                    }
                }
                Err(ref fail) => {
                    let r = reject_of(fail);
                    let kind = match r {
                        saccade::Reject::HumanOnly => "HumanOnly",
                        saccade::Reject::InvalidTaskId => "InvalidTaskId",
                        saccade::Reject::InvalidCommentId => "InvalidCommentId",
                        saccade::Reject::NotAnOffer => "NotAnOffer",
                        saccade::Reject::NotCurrentOffer => "NotCurrentOffer",
                        saccade::Reject::SelfReliance => "SelfReliance",
                        saccade::Reject::SelfAcceptance => "SelfAcceptance",
                        saccade::Reject::InvalidStateTransition => "InvalidStateTransition",
                        saccade::Reject::InvalidIncarnationId => "InvalidIncarnationId",
                        saccade::Reject::DuplicateNativeSession => "DuplicateNativeSession",
                        saccade::Reject::InvalidMessageId => "InvalidMessageId",
                        saccade::Reject::InvalidActor => "InvalidActor",
                        saccade::Reject::ReasonRequired => "ReasonRequired",
                        saccade::Reject::WorkspaceAlreadyExists => "WorkspaceAlreadyExists",
                        saccade::Reject::WorkspaceMissing => "WorkspaceMissing",
                        saccade::Reject::CheckpointRewind => "CheckpointRewind",
                        saccade::Reject::InvalidReference => "InvalidReference",
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
    }

    // floors sit at half the measured rates so normal RNG wobble cannot trip them
    for (name, floor) in [
        ("cancel", 0.05),
        ("reopen", 0.005),
        ("claim", 0.10),
        ("concluded task", 0.05),
        ("cancelled task", 0.15),
        ("offered task", 0.10),
        ("citing comment", 0.02),
        ("offer comment", 0.10),
        ("accept comment", 0.03),
        ("reject comment", 0.03),
        ("use comment", 0.05),
        ("native registered", 0.15),
        ("message sent", 0.01),
        ("message submitted", 0.005),
        ("artifact", 0.05),
        ("revise", 0.05),
        ("comment withdraw", 0.015),
        ("workspace established", 0.10),
        ("use bind", 0.01),
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
        "InvalidCommentId",
        "NotAnOffer",
        "NotCurrentOffer",
        "SelfReliance",
        "InvalidStateTransition",
        "InvalidIncarnationId",
        "WorkspaceAlreadyExists",
        "InvalidReference",
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
