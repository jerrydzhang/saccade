//! Worked examples of the beads→saccade mapping. Each test
//! carries a real beads record, trimmed to the fields the mapping consumes,
//! and pins the event sequence that encodes it plus the world that sequence
//! folds into. The in-process half drives `db::execute` — the real write
//! path — where the fold is the point; the binary half covers argv hostility
//! and `--at`, where the CLI is the point.
//!
//! The in-process half targets `db::execute`, the layer beneath every consumer
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
    db::execute(conn, &importer(), Command::ClaimTask { id }, at).unwrap();
    db::execute(
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

    db::execute(
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

    db::execute(
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
    let refused = db::execute(
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

    db::execute(
        &mut conn,
        &beads_actor("jerry"),
        Command::CreateTask {
            name: alias("Tracking data model and analysis redesign", "jernerics-jyl"),
            parent_id: None,
        },
        1784860330,
    )
    .unwrap();
    db::execute(
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

    db::execute(
        &mut conn,
        &beads_actor("assistant"),
        Command::CreateTask {
            name: alias("Pueue backend parity with Slurm", "jernerics-jtvv"),
            parent_id: None,
        },
        1788531487,
    )
    .unwrap();
    db::execute(
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

    db::execute(
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
    let bin = env!("CARGO_BIN_EXE_saccade");
    let path = db_path("argv");
    let title = alias(
        "Project Overview page still renders the old Browse-scope chrome — cut over to the approved prototype layout",
        "jernerics-1jvx",
    );

    let out = std::process::Command::new(bin)
        .arg("--db")
        .arg(&path)
        .arg("--actor")
        .arg("assistant")
        .arg("--tier")
        .arg("agent")
        .arg("--at")
        .arg("1788452437")
        .arg("create")
        .arg("task")
        .arg(title.as_str())
        .output()
        .expect("spawn saccade");
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
        db::execute(
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
        db::execute(
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
        db::execute(
            &mut conn,
            &ruler,
            Command::AcceptProposal {
                id: ProposalId(RecordId(seq)),
            },
            300 + seq as u64,
        )
        .unwrap();
    }
    db::execute(
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
