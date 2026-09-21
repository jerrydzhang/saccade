//! Worked examples of the beads→saccade mapping. Each test
//! carries a real beads record, trimmed to the fields the mapping consumes,
//! and pins the event sequence that encodes it plus the world that sequence
//! folds into. The in-process half drives `db::record` — the real write
//! path — where the fold is the point; the binary half covers argv hostility
//! and `--at`, where the CLI is the point.
//!
//! The in-process half targets `db::record`, the layer beneath every consumer
//! (clap today, the serve/webui edge later). When a second consumer lands and
//! the shared logic is extracted, re-point these at that surface.
//!
//! Conventions fixed here for the first time (review these):
//! - alias rides the name: `"{title} ({beads id})"`
//! - receipt = `"{close_reason} [imported from beads {id}]"`, standing alone
//!   when beads recorded no close_reason
//! - imports never act at human tier: tier is the acting authority (the
//!   importing agent), actor is the attribution (`created_by`, or
//!   "beads import" where beads recorded no one / for claim+done, which
//!   beads never attributed)

use std::path::PathBuf;

use saccade::Reject;
use saccade::db::{self, LoadState};
use saccade::objects::task::TaskId;
use saccade::store::{Context, Tier, World};
use saccade::types::actor::ActorName;
use saccade::types::pointers::{GitBranch, GitCommit};
use saccade::views;
use saccade::{Command, ProposalAction, ProposalId, Prose, RecordId};

fn importer() -> Context {
    Context {
        actor: ActorName::new("beads import".into()).unwrap(),
        tier: Tier::Agent,
    }
}

/// Historical actor attribution; the authority is still the importing agent.
fn beads_actor(name: &str) -> Context {
    Context {
        actor: ActorName::new(name.into()).unwrap(),
        tier: Tier::Agent,
    }
}

fn alias(title: &str, id: &str) -> Prose {
    Prose::new(format!("{title} ({id})")).unwrap()
}

fn receipt(reason: &str, id: &str) -> Prose {
    Prose::new(if reason.is_empty() {
        format!("[imported from beads {id}]")
    } else {
        format!("{reason} [imported from beads {id}]")
    })
    .unwrap()
}

fn db_path(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("saccade-mapping-{name}-{}.db", std::process::id()));
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
    path
}

fn world_of(conn: &rusqlite::Connection) -> World {
    let loadout = db::load(conn).unwrap();
    let LoadState::Full(world) = loadout.state else {
        panic!("expected a full load");
    };
    world
}

fn done(conn: &mut rusqlite::Connection, id: TaskId, receipt: &str, at: u64) {
    db::record(conn, &importer(), Command::ClaimTask { id }, at).unwrap();
    db::record(
        conn,
        &importer(),
        Command::CompleteTask {
            id,
            receipt: saccade::Prose::new(receipt.into()).unwrap(),
        },
        at,
    )
    .unwrap();
}

/// jernerics-0a0 — the bulk case: closed with an evidence-bearing reason.
/// {"id":"jernerics-0a0","status":"closed","created_by":"assistant",
///  "created_at":1787401555,"closed_at":1787410018,
///  "title":"Enforce streaming limits on ingest and artifact uploads",
///  "close_reason":"Completed: _IngestBodyLimit meters actual ASGI receive bytes …"}
/// create@created_at under created_by; claim+done@closed_at under the
/// importer; receipt = close_reason + provenance.
#[test]
fn closed_with_receipt_folds_to_done() {
    let path = db_path("closed");
    let mut conn = db::open(&path).unwrap();

    db::record(
        &mut conn,
        &beads_actor("assistant"),
        Command::CreateTask {
            name: alias(
                "Enforce streaming limits on ingest and artifact uploads",
                "jernerics-0a0",
            ),
            parent_id: None,
        },
        1787401555,
    )
    .unwrap();
    done(
        &mut conn,
        TaskId(0),
        receipt(
            "Completed: _IngestBodyLimit meters actual ASGI receive bytes (buffer+replay) so chunked bodies cannot bypass 8 MiB",
            "jernerics-0a0",
        )
        .as_str(),
        1787410018,
    );

    let world = world_of(&conn);
    assert_eq!(world.tasks.len(), 1);
    let view = views::task_view(&world, TaskId(0)).unwrap();
    assert_eq!(view.state, "done");
    assert_eq!(
        view.name,
        alias(
            "Enforce streaming limits on ingest and artifact uploads",
            "jernerics-0a0",
        )
        .as_str()
    );

    // bi-temporal split: event times are beads', logged times are ours
    let rows = db::load(&conn).unwrap().rows;
    assert_eq!(rows.len(), 3);
    assert_eq!(
        (rows[0].event_time, rows[0].actor.as_str()),
        (1787401555, "assistant")
    );
    assert_eq!(
        (rows[1].event_time, rows[1].actor.as_str()),
        (1787410018, "beads import")
    );
    assert!(
        rows.iter()
            .all(|r| r.tier == "agent" && r.logged_time >= r.event_time)
    );
    assert!(rows[2].payload.contains("cannot bypass 8 MiB"));
    assert!(
        rows[2]
            .payload
            .contains("[imported from beads jernerics-0a0]")
    );
}

/// jernerics-035 — judgment-shaped closure: a duplicate.
/// close_reason: "Duplicate of jernerics-gvs (accidental bare create)"
/// import as an open task with its alias and stop; a human performs
/// the drop. The machine holds the gate, not the runbook's good behavior.
#[test]
fn duplicate_stops_at_the_gate() {
    let path = db_path("duplicate");
    let mut conn = db::open(&path).unwrap();

    db::record(
        &mut conn,
        &beads_actor("assistant"),
        Command::CreateTask {
            name: alias(
                "Slurm checker chain uses misspelled --kill-on-dep-invalid flag",
                "jernerics-035",
            ),
            parent_id: None,
        },
        1787428204,
    )
    .unwrap();

    let world = world_of(&conn);
    assert_eq!(world.tasks.len(), 1);
    assert_eq!(views::task_view(&world, TaskId(0)).unwrap().state, "open");
    assert_eq!(db::load(&conn).unwrap().rows.len(), 1);

    // the void is human-only: the encoding could not have gone further
    let refused = db::record(
        &mut conn,
        &importer(),
        Command::DropTask {
            id: TaskId(0),
            note: Prose::new("duplicate of jernerics-gvs".into()).unwrap(),
        },
        1787428213,
    );
    assert!(matches!(
        refused,
        Err(db::ExecuteFail::Reject(Reject::HumanOnly))
    ));
}

/// jernerics-jyl.13 — dotted manual child of jernerics-jyl, with no
/// recorded creator and an empty close_reason.
/// dotted ids become parent edges; missing created_by falls back to
/// the importer; an empty reason leaves the provenance line standing alone.
#[test]
fn dotted_child_becomes_a_parent_edge() {
    let path = db_path("dotted");
    let mut conn = db::open(&path).unwrap();

    db::record(
        &mut conn,
        &beads_actor("jerry"),
        Command::CreateTask {
            name: alias("Tracking data model and analysis redesign", "jernerics-jyl"),
            parent_id: None,
        },
        1784860330,
    )
    .unwrap();
    db::record(
        &mut conn,
        &importer(),
        Command::CreateTask {
            name: alias(
                "Implement jernerics trace command (raw step/value series, --json)",
                "jernerics-jyl.13",
            ),
            parent_id: Some(TaskId(0)),
        },
        1785527640,
    )
    .unwrap();
    done(
        &mut conn,
        TaskId(1),
        receipt("", "jernerics-jyl.13").as_str(),
        1785528203,
    );
    done(
        &mut conn,
        TaskId(0),
        receipt("", "jernerics-jyl").as_str(),
        1785534527,
    );

    let world = world_of(&conn);
    let view = views::task_view(&world, TaskId(1)).unwrap();
    assert_eq!(view.parent, Some("t-0".to_string()));
    assert_eq!(view.state, "done");

    let rows = db::load(&conn).unwrap().rows;
    // empty close_reason: the provenance line stands alone
    assert!(
        rows[3]
            .payload
            .contains("[imported from beads jernerics-jyl.13]")
    );
    // creatorless create is attributed to the importer, not fabricated
    assert_eq!(rows[1].actor, "beads import");
}

/// jernerics-jtvv + jtvv.7 — epic and child, both closed with receipts.
/// close_reason (epic): "Epic complete: all children jtvv.1-jtvv.8 verified,
/// merged to main, and closed …"
/// Root convention: the epic is a task with children; the wrap receipt
/// is a done deposit like any other.
#[test]
fn epic_and_child_import_with_wrap_receipts() {
    let path = db_path("epic");
    let mut conn = db::open(&path).unwrap();

    db::record(
        &mut conn,
        &beads_actor("assistant"),
        Command::CreateTask {
            name: alias("Pueue backend parity with Slurm", "jernerics-jtvv"),
            parent_id: None,
        },
        1788531487,
    )
    .unwrap();
    db::record(
        &mut conn,
        &beads_actor("assistant"),
        Command::CreateTask {
            name: alias(
                "Resolve pueue checker worker-slot occupancy",
                "jernerics-jtvv.7",
            ),
            parent_id: Some(TaskId(0)),
        },
        1788531528,
    )
    .unwrap();
    done(
        &mut conn,
        TaskId(1),
        receipt(
            "Verified and merged: 955725b fast-forwarded to main + ty fix 0fe0abc",
            "jernerics-jtvv.7",
        )
        .as_str(),
        1788541949,
    );
    done(
        &mut conn,
        TaskId(0),
        receipt(
            "Epic complete: all children jtvv.1-jtvv.8 verified, merged to main, and closed",
            "jernerics-jtvv",
        )
        .as_str(),
        1788543052,
    );

    let world = world_of(&conn);
    assert_eq!(
        views::task_view(&world, TaskId(1)).unwrap().parent,
        Some("t-0".to_string())
    );
    assert_eq!(views::task_view(&world, TaskId(0)).unwrap().state, "done");
    assert_eq!(views::task_view(&world, TaskId(1)).unwrap().state, "done");
}

/// symlab-gwb — in_progress, no closed_at, updated this week.
/// Live-context door: "in progress" was beads' opinion that work was
/// occurring; at cutover the work is interrupted, and open is the honest
/// state. Re-capture (claim + re-articulated context) happens as work
/// resumes, not at import.
#[test]
fn in_progress_lands_open_for_recapture() {
    let path = db_path("live");
    let mut conn = db::open(&path).unwrap();

    db::record(
        &mut conn,
        &beads_actor("assistant"),
        Command::CreateTask {
            name: alias(
                "Beam-search decode eval on confirm checkpoints (k x length-norm grid, eval-only)",
                "symlab-gwb",
            ),
            parent_id: None,
        },
        1788397367,
    )
    .unwrap();

    let world = world_of(&conn);
    assert_eq!(world.tasks.len(), 1);
    assert_eq!(views::task_view(&world, TaskId(0)).unwrap().state, "open");
    assert_eq!(db::load(&conn).unwrap().rows.len(), 1);
}

/// jernerics-1jvx — the adversarial case: a unicode title (em dash) whose
/// beads life began as a shell-heredoc casualty ("Garbled create —
/// recreating"). The CLI must carry the title verbatim through argv, and
/// `--at` must reach the log as event time while logged time stays ours.
#[test]
fn argv_carries_adversarial_titles_and_backdating() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = db_path("argv");
    let title = alias(
        "Project Overview page still renders the old Browse-scope chrome — cut over to the approved prototype layout",
        "jernerics-1jvx",
    );

    let out = std::process::Command::new(bin)
        .arg("--db")
        .arg(&path)
        .arg("--offline")
        .env("SACCADE_ACTOR", "assistant")
        .arg("--at")
        .arg("1788452437")
        .arg("create")
        .arg("task")
        .arg(title.as_str())
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = std::process::Command::new(bin)
        .arg("--db")
        .arg(&path)
        .arg("log")
        .arg("--json")
        .output()
        .expect("spawn saccade log");
    assert!(out.status.success());
    let rows: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let row = rows.as_array().unwrap().first().unwrap();

    // argv round-trip: byte-identical title inside the payload
    let payload = serde_json::to_string(&row["payload"]).unwrap();
    assert!(payload.contains(&serde_json::to_string(&title).unwrap()));

    // backdating: event time is beads', logged time is ours
    assert_eq!(row["event_time"], 1788452437);
    let logged: u64 = row["logged_time"].as_u64().unwrap();
    assert!(logged > 1788452437);
}

/// The gate-queue deposit, executable: a scan finds corpses, the agent proposes the
/// drops (evidence enters the world, not the chat), the human rules per act,
/// and the log proves the gate held — no agent-tier drop exists anywhere.
#[test]
fn gate_queue_deposit_scenario() {
    let ruler = Context {
        actor: ActorName::new("jerry".into()).unwrap(),
        tier: Tier::Human,
    };
    let path = db_path("gate-queue");
    let mut conn = db::open(&path).unwrap();

    let scan = [
        "corpse: jernerics run dead",
        "corpse: gate camouflage",
        "corpse: superseded by newer sweep",
        "corpse: wrong repo entirely",
        "corpse: hypothesis retired",
        "disputed corpse",
        "real work",
    ];
    for (i, name) in scan.iter().enumerate() {
        db::record(
            &mut conn,
            &importer(),
            Command::CreateTask {
                name: Prose::new((*name).into()).unwrap(),
                parent_id: None,
            },
            100 + i as u64,
        )
        .unwrap();
    }

    // the agent proposes a drop for every corpse, evidence as the name
    for id in 0..6u64 {
        db::record(
            &mut conn,
            &importer(),
            Command::CreateProposal {
                name: Prose::new(format!("evidence for corpse {id}")).unwrap(),
                action: ProposalAction::Drop {
                    task_id: TaskId(id as usize),
                },
            },
            200 + id,
        )
        .unwrap();
    }

    // the human rules per act: accept the five obvious (proposals born at
    // seq 7..=11), reject the disputed one (seq 12)
    for seq in 7..12usize {
        db::record(
            &mut conn,
            &ruler,
            Command::AcceptProposal {
                id: ProposalId(RecordId(seq)),
            },
            300 + seq as u64,
        )
        .unwrap();
    }
    db::record(
        &mut conn,
        &ruler,
        Command::RejectProposal {
            id: ProposalId(RecordId(12)),
            note: Prose::new("ruled real work; re-propose only with new evidence".into()).unwrap(),
        },
        400,
    )
    .unwrap();

    let world = world_of(&conn);
    for id in 0..5 {
        assert_eq!(
            views::task_view(&world, TaskId(id)).unwrap().state,
            "dropped"
        );
    }
    // the disputed corpse and the real work are untouched
    assert_eq!(views::task_view(&world, TaskId(5)).unwrap().state, "open");
    assert_eq!(views::task_view(&world, TaskId(6)).unwrap().state, "open");
    assert_eq!(world.proposals.len(), 6);
    let states: Vec<&str> = world
        .proposals
        .keys()
        .map(|id| views::proposal_view(&world, *id).unwrap().state)
        .collect();
    assert_eq!(states.iter().filter(|s| **s == "accepted").count(), 5);
    assert_eq!(states.iter().filter(|s| **s == "rejected").count(), 1);

    let loadout = db::load(&conn).unwrap();
    // seven creates + six proposes + five compound accepts (two each) + one reject
    assert_eq!(loadout.rows.len(), 7 + 6 + 10 + 1);
    // evidence flow: each drop note is its proposal's name
    let drops: Vec<&str> = loadout
        .rows
        .iter()
        .filter(|r| r.kind == "task_dropped")
        .map(|r| r.payload.as_str())
        .collect();
    assert_eq!(drops.len(), 5);
    for (i, payload) in drops.iter().enumerate() {
        assert!(
            payload.contains(&format!("evidence for corpse {i}")),
            "{payload}"
        );
    }
    // the gate held: every drop in the log is human-tier
    assert!(
        loadout
            .rows
            .iter()
            .all(|r| r.kind != "task_dropped" || r.tier == "human")
    );
}

/// The law's own story, end to end through the write path: five tasks born,
/// claimed, completed, released, and dropped, with ids, times, and actors
/// landing as stored.
#[test]
fn a_task_lifecycle_folds_through_the_write_path() {
    let path = db_path("lifecycle");
    let mut conn = db::open(&path).unwrap();
    let agent = Context {
        actor: ActorName::new("saccade bot".into()).unwrap(),
        tier: Tier::Agent,
    };
    let human = Context {
        actor: ActorName::new("human person".into()).unwrap(),
        tier: Tier::Human,
    };

    let create = |name: &str| Command::CreateTask {
        name: Prose::new(name.into()).unwrap(),
        parent_id: None,
    };

    db::record(&mut conn, &agent, create("implement foo"), 1).unwrap();
    db::record(&mut conn, &agent, Command::ClaimTask { id: TaskId(0) }, 2).unwrap();
    db::record(&mut conn, &human, create("fix bar"), 3).unwrap();
    db::record(&mut conn, &agent, Command::ClaimTask { id: TaskId(1) }, 4).unwrap();
    db::record(
        &mut conn,
        &agent,
        Command::CompleteTask {
            id: TaskId(1),
            receipt: Prose::new("bar fixed".into()).unwrap(),
        },
        5,
    )
    .unwrap();
    db::record(
        &mut conn,
        &human,
        Command::CreateTask {
            name: Prose::new("improve baz".into()).unwrap(),
            parent_id: Some(TaskId(0)),
        },
        6,
    )
    .unwrap();
    db::record(
        &mut conn,
        &agent,
        Command::CompleteTask {
            id: TaskId(0),
            receipt: Prose::new("foo completed successfully".into()).unwrap(),
        },
        7,
    )
    .unwrap();
    db::record(&mut conn, &human, create("migrate floop"), 8).unwrap();
    db::record(&mut conn, &agent, Command::ClaimTask { id: TaskId(3) }, 9).unwrap();
    db::record(
        &mut conn,
        &human,
        Command::ReleaseTask {
            id: TaskId(3),
            note: Prose::new("run dead, reclaim".into()).unwrap(),
        },
        10,
    )
    .unwrap();
    db::record(&mut conn, &agent, Command::ClaimTask { id: TaskId(3) }, 11).unwrap();
    db::record(
        &mut conn,
        &human,
        Command::DropTask {
            id: TaskId(2),
            note: Prose::new("scope covered by fix bar".into()).unwrap(),
        },
        12,
    )
    .unwrap();
    db::record(
        &mut conn,
        &human,
        Command::DropTask {
            id: TaskId(1),
            note: Prose::new("covered by fix bar; kept only as context".into()).unwrap(),
        },
        13,
    )
    .unwrap();
    db::record(&mut conn, &human, create("open work"), 14).unwrap();

    let loadout = db::load(&conn).unwrap();
    assert_eq!(loadout.rows.len(), 14);
    let world = world_of(&conn);
    let state_of = |n: usize| views::TaskView::of(TaskId(n), &world.tasks[n], None).state;
    assert_eq!(state_of(0), "done");
    assert_eq!(state_of(1), "dropped");
    assert_eq!(state_of(2), "dropped");
    assert_eq!(state_of(3), "claimed");
    assert_eq!(state_of(4), "open");

    // identity, time, and authorship land as stored
    assert_eq!(loadout.rows[8].seq, 8);
    assert_eq!(loadout.rows[8].event_time, 9);
    assert_eq!(loadout.rows[8].actor, "saccade bot");
    assert_eq!(loadout.rows[9].seq, 9);
    assert_eq!(loadout.rows[9].event_time, 10);
    assert_eq!(loadout.rows[9].actor, "human person");
    assert_eq!(loadout.rows[10].seq, 10);
    assert_eq!(loadout.rows[10].event_time, 11);
    assert_eq!(loadout.rows[10].actor, "saccade bot");
}

/// The merge-then-checkpoint law, through the write path: a run closes on
/// its receipt commit, the merged head enters through the verb's door at
/// the invoking actor's tier, and the record never returns to a head it
/// left — rewinds refuse, the restated head lands as a no-op.
#[test]
fn a_merged_head_records_through_the_verb_and_never_rewinds() {
    let path = db_path("checkpoint");
    let mut conn = db::open(&path).unwrap();
    let ruler = Context {
        actor: ActorName::new("jerry".into()).unwrap(),
        tier: Tier::Human,
    };
    let agent = Context {
        actor: ActorName::new("pi".into()).unwrap(),
        tier: Tier::Agent,
    };
    let system = Context::system();

    db::record(
        &mut conn,
        &ruler,
        Command::CreateTask {
            name: Prose::new("carry the merge in the worktree".into()).unwrap(),
            parent_id: None,
        },
        1,
    )
    .unwrap();
    db::record(
        &mut conn,
        &system,
        Command::CreateWorkspace {
            task_id: TaskId(0),
            base: GitCommit::new("abc123".into()).unwrap(),
            branch: GitBranch::new("saccade/t-0".into()).unwrap(),
        },
        2,
    )
    .unwrap();
    // the run closes on its receipt commit
    db::record(
        &mut conn,
        &system,
        Command::CheckpointWorkspace {
            task_id: TaskId(0),
            checkpoint: GitCommit::new("def456".into()).unwrap(),
        },
        3,
    )
    .unwrap();

    // the merge advances the branch; the verb records the merged head
    db::record(
        &mut conn,
        &agent,
        Command::CheckpointWorkspace {
            task_id: TaskId(0),
            checkpoint: GitCommit::new("789abc".into()).unwrap(),
        },
        4,
    )
    .unwrap();

    // the verb never rewinds: the base and the closed head are heads it left
    for rewind in ["abc123", "def456"] {
        let refused = db::record(
            &mut conn,
            &agent,
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: GitCommit::new(rewind.into()).unwrap(),
            },
            5,
        );
        assert!(matches!(
            refused,
            Err(db::ExecuteFail::Reject(Reject::CheckpointRewind))
        ));
    }

    // restating the recorded head succeeds as a no-op
    db::record(
        &mut conn,
        &agent,
        Command::CheckpointWorkspace {
            task_id: TaskId(0),
            checkpoint: GitCommit::new("789abc".into()).unwrap(),
        },
        6,
    )
    .unwrap();

    let world = world_of(&conn);
    assert_eq!(
        world.tasks[0]
            .workspace
            .as_ref()
            .unwrap()
            .checkpoint
            .as_str(),
        "789abc"
    );
    let rows = db::load(&conn).unwrap().rows;
    // create, workspace, close, merge, no-op: the two refusals wrote nothing
    assert_eq!(rows.len(), 5);
    // the verb's records carry agent tier under the invoking actor
    let verb_rows: Vec<_> = rows
        .iter()
        .filter(|r| r.kind == "task_workspace_checkpointed" && r.payload.contains("789abc"))
        .collect();
    assert_eq!(verb_rows.len(), 2);
    assert!(
        verb_rows
            .iter()
            .all(|r| r.actor == "pi" && r.tier == "agent")
    );
}

/// Possession fixes the tier: SACCADE_ACTOR present records agent, its
/// absence records human under the account name — the CLI has no tier
/// argument for either side to claim with.
#[test]
fn tier_derives_from_possession_not_argument() {
    let bin = env!("CARGO_BIN_EXE_sac");

    let agent_db = db_path("tier-possession-agent");
    std::process::Command::new(bin)
        .arg("--db")
        .arg(&agent_db)
        .arg("--offline")
        .env("SACCADE_ACTOR", "assistant")
        .arg("create")
        .arg("task")
        .arg("possession agent")
        .output()
        .expect("spawn sac");
    let human_db = db_path("tier-possession-human");
    std::process::Command::new(bin)
        .arg("--db")
        .arg(&human_db)
        .arg("--offline")
        .env_remove("SACCADE_ACTOR")
        .arg("create")
        .arg("task")
        .arg("possession human")
        .output()
        .expect("spawn sac");

    for (path, want) in [(agent_db, "agent"), (human_db, "human")] {
        let conn = saccade::db::open_read(&path).unwrap();
        let rows = saccade::db::load(&conn).unwrap().rows;
        assert_eq!(rows[0].tier, want, "the tier is possessed, not asserted");
    }
}

/// The id space teaches: the known confusion cells at the CLI door —
/// a hashed task id, a birth record used as a comment target, a
/// judgment refused by the task's state — each refusal names the
/// expected format and the likely intended target, and a refused
/// demand shows its refusal on the thread.
#[test]
fn refusals_teach_the_id_space_and_the_state() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = db_path("teaching");
    let mut conn = db::open(&path).unwrap();

    // t-0 born at #0, claimed by the agent; a demand the machinery refuses
    let agent = Context {
        actor: ActorName::new("saccade bot".into()).unwrap(),
        tier: Tier::Agent,
    };
    db::record(
        &mut conn,
        &agent,
        Command::CreateTask {
            name: Prose::new("migrate floop".into()).unwrap(),
            parent_id: None,
        },
        100,
    )
    .unwrap();
    db::record(&mut conn, &agent, Command::ClaimTask { id: TaskId(0) }, 101).unwrap();
    db::record(
        &mut conn,
        &importer(),
        Command::Comment {
            target: saccade::Target::Task(TaskId(0)),
            body: Prose::new("run the migration once more".into()).unwrap(),
            addressee: Some(saccade::Addressee::Agent),
        },
        102,
    )
    .unwrap();
    db::record(
        &mut conn,
        &Context::system(),
        Command::RefuseDemand {
            demand: saccade::CommentId(RecordId(2)),
            reason: Prose::new(
                "t-0 worktree is disk-only leftover; reconcile it through the human".into(),
            )
            .unwrap(),
        },
        103,
    )
    .unwrap();
    drop(conn);

    let sac = |args: &[&str]| {
        std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .arg("--offline")
            .env("SACCADE_ACTOR", "assistant")
            .args(args)
            .output()
            .expect("spawn sac")
    };

    // a birth record used as a comment target names its task
    let out = sac(&["comment", "#0", "replying to a birth"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("rejected: invalid_comment_id"), "{stderr}");
    assert!(
        stderr.contains("#0 is the birth record of task t-0"),
        "{stderr}"
    );
    assert!(stderr.contains("address its thread as t-0"), "{stderr}");

    // a hashed task id learns the bare thread door
    let out = sac(&["comment", "#t-0", "hashing the task"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("drop the '#': the thread is addressed as t-0"),
        "{stderr}"
    );

    // a hashed c-N learns the bare record door
    let out = sac(&["comment", "#c-0", "hashing the render"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("drop the 'c-': the comment is addressed as #0"),
        "{stderr}"
    );

    // a judgment refused by the state names the state and its holder
    let out = sac(&["propose", "drop", "t-0", "--name", "floop is a corpse"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("rejected: invalid_state_transition"),
        "{stderr}"
    );
    assert!(
        stderr.contains("t-0 is claimed (held by saccade bot)"),
        "{stderr}"
    );
    assert!(
        stderr.contains("a drop proposal needs an open or done task"),
        "{stderr}"
    );

    // the refused demand shows its refusal on the thread: reason and time
    let out = sac(&["show", "t-0"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("(to agent, refused)"), "{stdout}");
    assert!(stdout.contains("refused "), "{stdout}");
    assert!(stdout.contains("disk-only leftover"), "{stdout}");
}
