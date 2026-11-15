// ---- the workspace operations: real git, the server's own effects ----

use std::path::Path;

use serde_json::{Value, json};

use saccade::store::{Context, Tier};
use saccade::types::actor::ActorName;
use saccade::{Command, TaskId};

use crate::common::{
    attempts_lines, envelope, get_json, json_of, post_command, scratch_db, scrubbed, spawn_serve,
    spawn_server, suite_state,
};

/// The records face keeps degraded rows and refuses corruption.
#[tokio::test(flavor = "multi_thread")]
async fn the_records_face_keeps_degraded_rows_and_refuses_corruption() {
    // a corrupt payload of a known kind: the log will not even load
    let db = scratch_db("records-corrupt");
    let base_url = spawn_server(&db).await;
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "before the damage"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (1, 0, 0, 'pi', 'agent', 'task_created', '{}', NULL)",
            [],
        )
        .unwrap();
    let mut r = ureq::get(&format!("{base_url}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 500);
    let body = r.body_mut().read_to_string().unwrap();
    assert_eq!(json_of(&body)["error"]["code"], "database_error");
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("corrupt record at seq 1")),
        "{body}"
    );
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();

    // an unknown kind: the world will not fold, the rows still travel
    let db = scratch_db("records-degraded");
    let base_url = spawn_server(&db).await;
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "before the damage"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (1, 0, 0, 'pi', 'agent', 'time_travelled', '{}', NULL)",
            [],
        )
        .unwrap();
    let mut r = ureq::get(&format!("{base_url}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let body = r.body_mut().read_to_string().unwrap();
    assert_eq!(json_of(&body)["records"].as_array().unwrap().len(), 2);
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "blocked while degraded"}}),
        ),
    );
    assert_eq!(status, 503);
    assert_eq!(json_of(&body)["error"]["code"], "degraded", "{body}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A supported local write shows on the server's records face: the
/// next server write folds the same file, never a stale world.
#[tokio::test(flavor = "multi_thread")]
async fn a_local_write_shows_on_the_server_face() {
    let db = scratch_db("local-visible");
    let base_url = spawn_server(&db).await;
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "server-born task"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");

    let local = Context {
        actor: ActorName::new("pi".into()).unwrap(),
        tier: Tier::Agent,
        scope: None,
    };
    let mut conn = saccade::db::open(&db).unwrap();
    saccade::db::record(
        &mut conn,
        &local,
        Command::ClaimTask {
            id: TaskId(0),
            incarnation: None,
        },
        saccade::db::now_epoch(),
    )
    .unwrap();

    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(json_of(&body)["records"].as_array().unwrap().len(), 2);

    // the next server write folds the same file, never a stale world
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "jerry",
            "human",
            json!({"create_task": {"name": "after the local write"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let records = json_of(&body);
    let kinds: Vec<String> = records["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(kinds, vec!["task_created", "task_claimed", "task_created"]);

    // a refusal names the log's fresh cursor, external writes included
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "jerry",
            "human",
            json!({"claim_task": {"id": 99, "incarnation": null}}),
        ),
    );
    assert_eq!(status, 400);
    let lines = attempts_lines(&db);
    let refused = lines
        .iter()
        .rev()
        .find(|l| l["outcome"] == "refused")
        .expect("the refusal took a line");
    assert_eq!(refused["cursor"], 2);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[test]
fn board_doors_run_the_full_loop_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("board-loop");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    // a real repo: the doors' telemetry resolves its state dir the way
    // production resolves the tracker it talks to
    let git = |args: &[&str]| {
        saccade::boards::git_command()
            .args(args)
            .current_dir(&project)
            .output()
            .expect("git init");
    };
    git(&["init", "--quiet", project.to_str().unwrap()]);
    git(&[
        "-C",
        project.to_str().unwrap(),
        "config",
        "user.email",
        "smoke@saccade.invalid",
    ]);
    git(&[
        "-C",
        project.to_str().unwrap(),
        "config",
        "user.name",
        "smoke",
    ]);
    let (mut serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", project.to_str().unwrap()],
        &[],
    );
    let sac = |args: &[&str]| {
        let mut cmd = scrubbed(std::process::Command::new(bin));
        cmd.arg("--repo")
            .arg(&project)
            .arg("--server")
            .arg(&url)
            .env("SACCADE_ACTOR", "pi")
            .args(args);
        let out = cmd.output().expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let sac_json = |args: &[&str]| {
        let mut full = vec!["--json"];
        full.extend_from_slice(args);
        let (ok, out, err) = sac(&full);
        assert!(ok, "{}: {err}", args.join(" "));
        let value: Value = serde_json::from_str(&out).expect("the json face parses");
        value
    };

    // two tasks: the source account and its consumer
    let source = sac_json(&["create", "task", "prove the loop"])["id"]
        .as_str()
        .unwrap()
        .to_string();
    let consumer = sac_json(&["create", "task", "use the proof"])["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(source, "t-0");
    assert_eq!(consumer, "t-1");

    // the source's first board: edit writes the fresh template, the
    // author replaces it, check passes, commit publishes r1
    let src = project.join("src-dir");
    let (ok, out, _) = sac(&["board", "edit", &source, src.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(src.join("board.yaml").exists(), "the template landed");
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input\n",
    )
    .unwrap();
    let (ok, out, _) = sac(&["board", "check", src.to_str().unwrap()]);
    assert!(ok, "{out}");
    let reply = sac_json(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "first account",
    ]);
    assert_eq!(reply["revision"], json!(1));
    // a board matching its parent refuses, naming the parent revision, and records nothing
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "the edit never reached the board",
    ]);
    assert!(!ok, "the unchanged board refuses");
    assert!(
        err.contains("boards_unchanged") && err.contains("t-0@r1"),
        "{err}"
    );

    // the consumer incorporates the source's r1 by number; an unknown
    // number refuses with the teaching line
    let con = project.join("con-dir");
    let (ok, out, _) = sac(&["board", "edit", &consumer, con.to_str().unwrap()]);
    assert!(ok, "{out}");
    std::fs::write(
        con.join("board.yaml"),
        "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 9\n        select: [\"/content/0\"]\n      content:\n        - text: relying on the proof\n",
    )
    .unwrap();
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        &consumer,
        con.to_str().unwrap(),
        "-m",
        "cites a revision the source never published",
    ]);
    assert!(!ok, "the unknown revision refuses");
    assert!(
        err.contains("boards_unknown_source_revision") && err.contains("r9"),
        "{err}"
    );
    std::fs::write(
        con.join("board.yaml"),
        "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 1\n        select: [\"/content/0\"]\n      content:\n        - text: relying on the proof\n",
    )
    .unwrap();
    let reply = sac_json(&[
        "board",
        "commit",
        &consumer,
        con.to_str().unwrap(),
        "-m",
        "consumer cites the proof",
    ]);
    assert_eq!(reply["revision"], json!(1));

    // the acceptance refuses while the source stands unaccepted
    let (ok, _, _) = sac(&["offer", &consumer]);
    assert!(ok);
    let (ok, _, err) = sac(&["accept", &consumer]);
    assert!(!ok, "the unaccepted source must refuse the acceptance");
    assert!(err.contains("source_not_accepted"), "{err}");
    let (ok, out, _) = sac(&[
        "reject",
        &consumer,
        "--grounds",
        "wait for the source to conclude",
    ]);
    assert!(ok, "{out}");

    // the source's review loop: offer, reject, annotate, recommit.
    let (ok, out, _) = sac(&["offer", &source]);
    assert!(ok, "{out}");
    // committing a changed board refuses while offered
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input, early\n",
    )
    .unwrap();
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "too early",
    ]);
    assert!(!ok, "the offered stage refuses a commit");
    assert!(err.contains("invalid_state_transition"), "{err}");
    let (ok, out, _) = sac(&[
        "reject",
        &source,
        "--grounds",
        "the boundary case is missing",
    ]);
    assert!(ok, "{out}");
    assert!(
        out.contains("task open; annotate the board at t-0@r1"),
        "the reject teaches the annotation step: {out}"
    );
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input\n  - annotation: the boundary case is covered in the appendix\n",
    )
    .unwrap();
    let reply = sac_json(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "annotate the boundary",
    ]);
    assert_eq!(reply["revision"], json!(2));
    let (ok, _, _) = sac(&["offer", &source]);
    assert!(ok);
    let (ok, _, _) = sac(&["accept", &source]);
    assert!(ok, "the source accepts once its incorporations hold");

    // the consumer reconciles onto the accepted revision and concludes
    std::fs::write(
        con.join("board.yaml"),
        "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 2\n        select: [\"/content/0\"]\n      content:\n        - text: relying on the proof\n",
    )
    .unwrap();
    let reply = sac_json(&[
        "board",
        "commit",
        &consumer,
        con.to_str().unwrap(),
        "-m",
        "rebind to the accepted revision",
    ]);
    assert_eq!(reply["revision"], json!(2));
    let (ok, _, _) = sac(&["offer", &consumer]);
    assert!(ok);

    let (ok, out, _) = sac(&["accept", &consumer]);
    assert!(ok, "{out}");
    assert!(out.contains("t-1 concluded"), "{out}");

    // the task history interleaves commits with the decisions
    let history = sac_json(&["board", "log", &source]);
    let entries = history["history"].as_array().unwrap();
    let names: Vec<&str> = entries
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    assert!(
        names.contains(&"task_created")
            && names.contains(&"task_committed")
            && names.contains(&"task_offered")
            && names.contains(&"task_rejected")
            && names.contains(&"task_accepted"),
        "{names:?}"
    );
    let committed = entries
        .iter()
        .find(|e| e["event"] == "task_committed")
        .unwrap();
    assert_eq!(committed["message"], json!("first account"));
    assert_eq!(committed["revision"], json!(1));
    let rejected = entries
        .iter()
        .find(|e| e["event"] == "task_rejected")
        .unwrap();
    assert_eq!(rejected["grounds"], json!("the boundary case is missing"));

    // the diff names the annotation; get reads any revision; the
    // component form prints one component; the file read rides the
    // flag; the materialized path exists on the server's disk
    let (ok, diff, _) = sac(&["board", "diff", "t-0@r1..r2"]);
    assert!(ok, "{diff}");
    assert!(diff.contains("annotation"), "{diff}");
    // the component summary leads (t-50): the added annotation, by
    // place, before the line diff
    assert!(
        diff.starts_with("components r1..r2\n  added    annotation  /content/1\n"),
        "{diff}"
    );
    let (ok, json_diff, _) = sac(&["--json", "board", "diff", "t-0@r1..r2"]);
    assert!(ok, "{json_diff}");
    let json_diff: serde_json::Value = serde_json::from_str(&json_diff).unwrap();
    assert_eq!(
        json_diff["components"],
        json!([{"change": "added", "label": "annotation", "to": "/content/1"}])
    );
    assert!(json_diff["diff"].as_str().unwrap().contains("annotation"));
    let (ok, component, _) = sac(&["board", "get", "t-0@r2:/content/0"]);
    assert!(ok, "{component}");
    assert!(
        component.contains("the loop holds") && !component.contains("annotation"),
        "{component}"
    );
    let (ok, note, _) = sac(&["board", "get", "t-0@r2:/content/1"]);
    assert!(ok, "{note}");
    assert!(note.contains("annotation"), "{note}");
    let (ok, _, err) = sac(&["board", "get", "t-0@r2:/content/0", "--file", "board.yaml"]);
    assert!(!ok, "the component and file forms refuse each other: {err}");
    assert!(err.contains("not both"), "{err}");
    let (ok, old, _) = sac(&["board", "get", "t-0@r1", "--file", "board.yaml"]);
    assert!(ok, "{old}");
    assert!(
        old.contains("the loop holds") && !old.contains("annotation"),
        "{old}"
    );
    let tree_out = project.join("exported");
    let (ok, out, _) = sac(&["board", "get", "t-0", "--out", tree_out.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(tree_out.join("board.yaml").exists(), "{out}");
    // the boards/get wire the viewer reads: the file list nests under
    // files.files, each entry a path and base64 content
    let tree: Value = ureq::get(&format!("{url}/api/v1/boards/get?task=0"))
        .call()
        .unwrap()
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    let listed = tree["files"]["files"]
        .as_array()
        .expect("the viewer reads files.files as a list");
    assert!(
        listed
            .iter()
            .any(|f| f["path"] == json!("board.yaml") && f["content"].is_string()),
        "{tree}"
    );
    assert_eq!(tree["revision"], json!(2));
    let (ok, path, _) = sac(&["board", "path", &source]);
    assert!(ok, "{path}");
    assert!(Path::new(path.trim()).join("board.yaml").exists(), "{path}");

    // search: current boards one line per task, history changes,
    // references are incorporations and messages
    let (ok, out, _) = sac(&["search", "loop"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-0@r2"), "{out}");
    assert!(
        !out.contains("t-1"),
        "the consumer board names no loop: {out}"
    );
    let (ok, out, _) = sac(&["search", "relying"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-1@r2"), "{out}");
    let (ok, out, _) = sac(&["search", "--history", "boundary"]);
    assert!(ok, "{out}");
    assert!(out.contains("appeared"), "{out}");
    assert!(out.contains("annotate the boundary"), "{out}");
    let (ok, out, _) = sac(&["search", "t-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("incorporates t-0@r"), "{out}");

    // telemetry rides in the repo's state dir: verb, flags, outcome, tier
    let telemetry = std::fs::read_to_string(saccade::telemetry::at(&saccade::paths::db_at_in(
        &suite_state(),
        &project,
    )))
    .unwrap();
    assert!(
        telemetry.contains("\"verb\":\"board.commit\""),
        "{telemetry}"
    );
    assert!(telemetry.contains("\"tier\":\"agent\""), "{telemetry}");
    assert!(
        telemetry.contains("\"outcome\":\"source_not_accepted\""),
        "{telemetry}"
    );

    serve.kill().unwrap();
    let _ = serve.wait();
}

#[test]
fn board_size_refusals_name_the_total_and_the_request_limit() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("board-size");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    saccade::boards::git_command()
        .args(["init", "--quiet", project.to_str().unwrap()])
        .output()
        .expect("git init");
    let (mut serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", project.to_str().unwrap()],
        &[],
    );
    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
            .arg("--repo")
            .arg(&project)
            .arg("--server")
            .arg(&url)
            .env("SACCADE_ACTOR", "pi")
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let figure = |seed: u8, len: usize| -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    };
    // a working directory naming each figure; the answer is the files' total
    let stage = |dir: &Path, figures: &[(&str, Vec<u8>)]| -> usize {
        let mut board = String::from("grammar: 1\ntask: t-0\ncontent:\n  - text: the survey\n");
        let mut total = 0;
        for (path, bytes) in figures {
            std::fs::create_dir_all(dir.join(path).parent().unwrap()).unwrap();
            std::fs::write(dir.join(path), bytes).unwrap();
            total += bytes.len();
            board.push_str(&format!(
                "  - figure:\n      input: {path}\n      caption: {path}\n"
            ));
        }
        std::fs::write(dir.join("board.yaml"), &board).unwrap();
        total + board.len()
    };
    let (ok, _, err) = sac(&["create", "task", "show the survey"]);
    assert!(ok, "{err}");
    let dir = project.join("survey");
    let (ok, _, err) = sac(&["board", "edit", "t-0", dir.to_str().unwrap()]);
    assert!(ok, "{err}");

    // the reported board (a 1,987,615-byte file) beside one at the per-file bound publishes
    let figures = [
        ("figures/ux-study.png", figure(1, 1_987_615)),
        ("figures/survey-map.png", figure(2, 8 * 1024 * 1024)),
    ];
    stage(&dir, &figures);
    let (ok, out, err) = sac(&[
        "board",
        "commit",
        "t-0",
        dir.to_str().unwrap(),
        "-m",
        "the survey",
    ]);
    assert!(ok, "the board within one request publishes: {err}");
    assert!(out.contains("published t-0@r1"), "{out}");
    // and reads back whole
    let back = project.join("survey-back");
    let (ok, _, err) = sac(&["board", "edit", "t-0", back.to_str().unwrap()]);
    assert!(ok, "{err}");
    for (path, bytes) in &figures {
        assert!(
            std::fs::read(back.join(path)).unwrap() == *bytes,
            "{path} round-trips"
        );
    }

    // a board past one request: check warns with the total, commit refuses naming it and the limit
    let big = [
        ("figures/a.png", figure(3, 7_000_000)),
        ("figures/b.png", figure(4, 7_000_000)),
        ("figures/c.png", figure(5, 7_000_000)),
        ("figures/d.png", figure(6, 7_000_000)),
    ];
    let total = stage(&dir, &big);
    let (ok, out, err) = sac(&["board", "check", dir.to_str().unwrap()]);
    assert!(ok, "the warning does not fail the check: {err}");
    assert!(
        out.contains("warning") && out.contains(&total.to_string()),
        "{out}"
    );
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        "t-0",
        dir.to_str().unwrap(),
        "-m",
        "too much",
    ]);
    assert!(!ok, "the oversized request refuses");
    assert!(
        err.contains(&total.to_string()) && err.contains("per request"),
        "{err}"
    );
    assert!(!err.contains("malformed_response"), "{err}");

    // a body just past the limit refuses in the error envelope
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .content_type("application/json")
        .send(" ".repeat(32 * 1024 * 1024 + 1).as_str())
        .expect("the server answers");
    let status = response.status().as_u16();
    let body = response.body_mut().read_to_string().unwrap();
    assert_eq!(status, 413, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        json!("request_too_large"),
        "{body}"
    );

    // a non-JSON refusal surfaces with its status and text
    let error =
        saccade::serve::client::board_get_component(&url, 0, None, "/content/0&revision=zz")
            .err()
            .expect("the bad query refuses");
    let shown = format!("{error:?}");
    assert!(!shown.contains("malformed_response"), "{shown}");
    assert!(
        shown.contains("400") && shown.contains("revision"),
        "{shown}"
    );

    serve.kill().unwrap();
    let _ = serve.wait();
}

#[test]
fn omitted_prose_refuses_without_a_terminal_and_names_the_move() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("prose-door");
    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
            .arg("--db")
            .arg(&path)
            .env("SACCADE_ACTOR", "pi")
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let (ok, _, _) = sac(&["create", "task", "named prose"]);
    assert!(ok);

    // no terminal, no prose: the door refuses and teaches both moves
    let (ok, _, err) = sac(&["create", "task"]);
    assert!(!ok);
    assert!(err.contains("task name is required"), "{err}");
    let (ok, _, err) = sac(&["cancel", "t-0"]);
    assert!(!ok);
    assert!(
        err.contains("cancellation explanation is required"),
        "{err}"
    );
    let (ok, _, err) = sac(&["reject", "t-0"]);
    assert!(!ok);
    assert!(err.contains("rejection grounds is required"), "{err}");

    // explicit prose passes at the doors agents use; cancel and reopen
    // stay the human's acts
}

#[test]
fn board_wire_routes_hold_their_contracts() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("board-wire");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let (mut serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", project.to_str().unwrap()],
        &[],
    );

    // the version route names the supported grammar range
    let body: Value = ureq::get(&format!("{url}/api/v1/version"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap()
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["board_grammar"]["min"], json!(1));
    assert_eq!(body["board_grammar"]["max"], json!(1));
    // the tracker's name for tracker-named locations: the repo's
    // directory basename; the fixture repo directory is "project"
    assert_eq!(body["tracker"], json!("project"));

    // a doubled slash lands on no route: 404, never a collapsed-path
    // answer, so a client that joins one slash too many sees a refusal
    for path in ["//api/v1/version", "//api/v1/boards/get?task=0"] {
        let mut response = ureq::get(&format!("{url}{path}"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap();
        assert_eq!(response.status().as_u16(), 404, "{path}");
        let body = response.body_mut().read_to_string().unwrap();
        assert!(
            !body.contains("board_grammar") && !body.contains("files"),
            "{path}: {body}"
        );
    }

    // the path route refuses a task with no board, and a bad task id
    let created = ureq::post(&format!("{url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(json!({
            "context": {"actor": "pi", "tier": "agent", "scope": null},
            "command": {"create_task": {"name": "wire route task"}},
            "at": null,
            "client": "test"
        }))
        .unwrap();
    assert_eq!(created.status().as_u16(), 200);
    let status = ureq::get(&format!("{url}/api/v1/boards/path/0"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap()
        .status();
    assert_eq!(status.as_u16(), 404, "no board yet: {status}");
    let status = ureq::get(&format!("{url}/api/v1/boards/path/99"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap()
        .status();
    assert_eq!(status.as_u16(), 400);

    // a commit with an unparseable board refuses without recording
    let request = json!({
        "context": {"actor": "pi", "tier": "agent", "scope": null},
        "task": 0,
        "files": {"files": [{"path": "board.yaml", "content": "bm90IGEgYm9hcmQK"}]},
        "message": "broken",
        "client": "test"
    });
    // task 0 exists and the board does not parse: the validation refuses
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(request.clone())
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("boards_board_invalid"));
    // an unknown task id still refuses at the id
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(json!({
            "context": {"actor": "pi", "tier": "agent", "scope": null},
            "task": 99,
            "files": {"files": []},
            "message": "nowhere",
            "client": "test"
        }))
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("invalid_task_id"));

    // malformed json refuses as malformed, never panics
    // malformed JSON still refuses as malformed: the bytes name
    // themselves JSON even when they do not parse
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json")
        .send(&b"{not json"[..])
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("malformed_request"));

    // the origin policy: a browser page always names its origin, so a
    // request from a non-allowlisted origin refuses on reads and writes
    // alike; a shell origin reads with its origin echoed, never a
    // wildcard; a POST names its body JSON, which a cross-site form
    // cannot
    let header = |name: (&str, &str)| {
        ureq::get(&format!("{url}/api/v1/version"))
            .config()
            .http_status_as_error(false)
            .build()
            .header(name.0, name.1)
            .call()
            .unwrap()
    };
    let mut page = header(("origin", "https://evil.example"));
    assert_eq!(page.status().as_u16(), 403, "a stranger origin refuses");
    let body: Value = page.body_mut().read_to_string().unwrap().parse().unwrap();
    assert_eq!(body["error"]["code"], json!("origin_forbidden"));

    let shell = header(("origin", "tauri://localhost"));
    assert_eq!(shell.status().as_u16(), 200, "the shell origin reads");
    assert_eq!(
        shell
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost"),
        "the origin is echoed, never a wildcard",
    );
    assert!(
        shell.headers().get("vary").is_some(),
        "the answer varies by origin"
    );

    // a browser preflights every cross-origin write (Content-Type:
    // application/json is not a simple request): an allowed origin's
    // preflight answers 204 with its origin echoed and the verbs the
    // doors carry; a stranger's refuses; a preflight naming no origin
    // is no browser's
    let preflight = |origin: Option<&str>| {
        let request = ureq::options(&format!("{url}/api/v1/windows/states"))
            .config()
            .http_status_as_error(false)
            .build();
        let request = match origin {
            Some(value) => request
                .header("origin", value)
                .header("access-control-request-method", "POST")
                .header("access-control-request-headers", "content-type"),
            None => request,
        };
        request.call().unwrap()
    };
    let shell_preflight = preflight(Some("tauri://localhost"));
    assert_eq!(
        shell_preflight.status().as_u16(),
        204,
        "the shell's preflight answers"
    );
    assert_eq!(
        shell_preflight
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost")
    );
    assert_eq!(
        shell_preflight
            .headers()
            .get("access-control-allow-methods")
            .and_then(|v| v.to_str().ok()),
        Some("GET, POST, PATCH, PUT, DELETE, OPTIONS")
    );
    let stranger_preflight = preflight(Some("https://evil.example"));
    assert_eq!(stranger_preflight.status().as_u16(), 403);
    let bare_preflight = preflight(None);
    assert_eq!(bare_preflight.status().as_u16(), 400);

    // the echo rides writes: a write from an allowed origin carries its
    // origin on the answer, so the client reads the sequence its echo
    // filter and its next settled write need
    let write = ureq::post(&format!("{url}/api/v1/windows/states"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("origin", "tauri://localhost")
        .send_json(json!({"id": "w-cors", "device": "dev-1"}))
        .unwrap();
    assert_eq!(write.status().as_u16(), 200);
    assert_eq!(
        write
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost")
    );

    // the rebinding guard: a domain name that is not this server
    // refuses before routing, and an allowlisted page can read why; an
    // IP literal never comes from a rebinding and answers
    let rebound = |host: &str| {
        ureq::get(&format!("{url}/api/v1/version"))
            .config()
            .http_status_as_error(false)
            .build()
            .header("host", host)
            .header("origin", "tauri://localhost")
            .call()
            .unwrap()
    };
    let mut stranger = rebound("evil.example:8811");
    assert_eq!(stranger.status().as_u16(), 403, "a stranger name refuses");
    assert_eq!(
        stranger
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost"),
        "the shell reads the refusal"
    );
    let body: Value = stranger
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("host_forbidden"));
    assert_eq!(
        rebound("192.168.0.51:8811").status().as_u16(),
        200,
        "a LAN IP literal answers"
    );
    assert_eq!(
        rebound("[fd00::5]:8811").status().as_u16(),
        200,
        "an IPv6 literal answers"
    );

    let anonymous = ureq::get(&format!("{url}/api/v1/version"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(
        anonymous.status().as_u16(),
        200,
        "no origin header, no page"
    );
    assert!(
        anonymous
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );

    // the forged-write hole: a cross-site form or no-cors POST arrives
    // with text/plain or a form content type and refuses however it came
    let mut form = ureq::post(&format!("{url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("origin", "tauri://localhost")
        .header("content-type", "application/x-www-form-urlencoded")
        .send("context=tier%3Dhuman")
        .unwrap();
    assert_eq!(form.status().as_u16(), 415, "a form body refuses");
    let body: Value = form.body_mut().read_to_string().unwrap().parse().unwrap();
    assert_eq!(body["error"]["code"], json!("content_type_required"));

    let stranger_post = ureq::post(&format!("{url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("origin", "https://evil.example")
        .header("content-type", "application/json")
        .send("{}")
        .unwrap();
    assert_eq!(
        stranger_post.status().as_u16(),
        403,
        "the stranger origin refuses writes too"
    );

    // the search route answers empty on an empty tracker
    let mut response = ureq::post(&format!("{url}/api/v1/boards/search"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(json!({"terms": ["anything"], "history": false, "within": null}))
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["lines"], json!([]));

    serve.kill().unwrap();
    let _ = serve.wait();
}

#[test]
fn show_and_log_json_read_byte_exact_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("json-read-door");
    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
            .arg("--db")
            .arg(&path)
            .env("SACCADE_ACTOR", "pi")
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // the name no honest wrapper survives: quotes and backslashes in
    // one line
    let name = "first: \"quoted\" and back\\slashed";
    assert!(sac(&["create", "task", name]).0);
    assert!(sac(&["claim", "t-0"]).0);
    assert!(sac(&["create", "task", "second task"]).0);

    // the task's json face: the authored name byte-exact
    let (ok, out, _) = sac(&["show", "--json", "t-0"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["cursor"], json!(2));
    let block = &json["blocks"][0];
    assert_eq!(block["id"], json!("t-0"));
    assert_eq!(block["name"], json!(name));

    // the bare position on the json face: the raw log row, no fold
    let (ok, out, _) = sac(&["show", "--json", "0"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["blocks"][0]["seq"], json!(0));
    assert_eq!(json["blocks"][0]["payload"]["name"], json!(name));

    // the log's json face rides the rows with the cursor beside them
    let (ok, out, _) = sac(&["log", "--json"]);
    assert!(ok, "{out}");
    let log: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(log["cursor"], json!(2));
    let rows: Vec<Value> = log["records"]
        .as_array()
        .expect("the records array")
        .clone();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1]["kind"], json!("task_claimed"));

    // text reads end with the cursor line too: show, log, list. The
    // search face left for the boards: its answers are board lines,
    // not record rows, and carry no cursor
    for args in [
        vec!["show", "t-0"],
        vec!["show", "1"],
        vec!["log"],
        vec!["list"],
    ] {
        let (ok, out, _) = sac(&args);
        assert!(ok, "{}: {out}", args.join(" "));
        assert_eq!(
            out.lines().last().unwrap(),
            "cursor 2",
            "{}: {out}",
            args.join(" ")
        );
    }

    // the pipe's json face: every line's outcome, skips included
    let pipe = |input: &str, args: &[&str]| {
        use std::io::Write as _;
        use std::process::Stdio;
        let mut child = scrubbed(std::process::Command::new(bin))
            .arg("--db")
            .arg(&path)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sac");
        child
            .stdin
            .as_mut()
            .expect("stdin pipes")
            .write_all(input.as_bytes())
            .expect("feed stdin");
        let out = child.wait_with_output().expect("wait sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let (ok, out, _) = pipe("bogus\nt-0\n", &["show", "--stdin", "--json"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    let rows = json["blocks"].as_array().expect("the blocks array");
    assert_eq!(rows.len(), 2);
    assert_eq!(json["cursor"], json!(2));
    assert_eq!(rows[0]["id"], json!("bogus"));
    assert!(rows[0]["error"].is_string());
    assert_eq!(rows[1]["name"], json!(name));

    // the stdin text face ends with the cursor line too
    let (ok, out, _) = pipe("t-0\n", &["show", "--stdin"]);
    assert!(ok, "{out}");
    assert_eq!(out.lines().last().unwrap(), "cursor 2", "{out}");

    // a number past the id space refuses at the grammar door with the
    // ordinary exit, never a panic: digits are grammar, the id space is not
    let (ok, _, err) = sac(&["show", "184467440737095516160"]);
    assert!(!ok);
    assert!(err.contains("is not an id"), "{err}");
    let (ok, _, err) = sac(&["show", "#184467440737095516160"]);
    assert!(!ok);
    assert!(
        err.contains("the raw record at a position is the bare <seq>"),
        "{err}"
    );
    let (ok, _, err) = sac(&["show", "bogus"]);
    assert!(!ok);
    assert!(
        err.contains("'bogus' is not an id (expected t-<n>, m-<n>, or a bare <seq>)"),
        "{err}"
    );
    // the raw door opens any record: an ownership event renders as
    // the log renders it, header and payload
    let (ok, out, _) = sac(&["show", "1"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("seq 1  task_claimed  pi\n"),
        "the raw event rides whole: {out}"
    );

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn create_reply_names_the_born_task() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("create-reply");

    let out = scrubbed(std::process::Command::new(bin))
        .arg("--db")
        .arg(&path)
        .env("SACCADE_ACTOR", "assistant")
        .args(["create", "task", "migrate floop"])
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("t-0"));
    assert!(
        lines.next().unwrap().contains("task_created"),
        "the birth record rides under the reference: {stdout}"
    );

    let out = scrubbed(std::process::Command::new(bin))
        .arg("--db")
        .arg(&path)
        .arg("--json")
        .env("SACCADE_ACTOR", "assistant")
        .args(["claim", "t-0"])
        .output()
        .expect("spawn sac claim");
    let claim: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        claim["records"].is_array(),
        "claim answers the wire's shape: its records under the reply object"
    );

    let out = scrubbed(std::process::Command::new(bin))
        .arg("--db")
        .arg(&path)
        .arg("--json")
        .env("SACCADE_ACTOR", "assistant")
        .args(["create", "task", "second scratch task"])
        .output()
        .expect("spawn sac create --json");
    assert!(out.status.success());
    let created: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(created["id"], "t-1");
    assert_eq!(created["records"][0]["kind"], "task_created");
}

/// The skill ships with the binary: install deploys the embedded
/// copy - the on-disk source byte for byte, SKILL.md stamped with the
/// binary's version inside its frontmatter - and check verifies the
/// deployed copy through both of its doors.
#[test]
fn the_binary_deploys_and_verifies_its_embedded_skill() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-skill-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
            .current_dir(&dir)
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // absent until deployed, and check says so honestly
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(!ok);
    assert!(out.contains("skill absent"), "{out}");

    // install writes the embedded copy, stamped with the binary version
    let (ok, out, _) = sac(&["skill", "install"]);
    assert!(ok, "{out}");
    assert!(out.contains("skill installed"), "{out}");
    let deployed = std::fs::read_to_string(dir.join(".agents/skills/saccade/SKILL.md"))
        .expect("the skill deployed");
    assert!(deployed.contains(&format!("x-saccade-version: {}", env!("CARGO_PKG_VERSION"))));

    // check now passes: the copy matches the embed beneath the stamp
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(ok, "{out}");
    assert!(out.contains("skill in-sync"), "{out}");

    // drift refuses: a differing copy survives until deleted
    let skill = dir.join(".agents/skills/saccade/SKILL.md");
    let mut drifted = deployed.clone();
    drifted.push_str("\ndrift\n");
    std::fs::write(&skill, drifted).unwrap();
    let (ok, _, err) = sac(&["skill", "install"]);
    assert!(!ok);
    assert_eq!(
        err.trim_end(),
        "error: .agents/skills/saccade differs from this binary's skill - delete the directory and run sac skill install to deploy this version"
    );
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(!ok, "{out}");
    let version = env!("CARGO_PKG_VERSION");
    assert_eq!(
        out.trim_end(),
        format!("skill drifted: .agents/skills/saccade carries {version}, binary is {version}")
    );

    // deletion is the force flag, and a redeploy restores sync
    std::fs::remove_dir_all(dir.join(".agents")).unwrap();
    let (ok, _, _) = sac(&["skill", "install"]);
    assert!(ok);
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(ok);
    assert!(out.contains("skill in-sync"), "{out}");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// A --db write answers with the wire's own reply shape.
#[tokio::test(flavor = "multi_thread")]
async fn local_replies_match_the_wire() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let local_db = scratch_db("local-shape");
    let served_db = scratch_db("served-shape");

    let local = |db: &std::path::Path, args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
            .arg("--db")
            .arg(db)
            .env("SACCADE_ACTOR", "pi")
            .args(args)
            .output()
            .expect("spawn sac");
        assert!(
            out.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice::<Value>(&out.stdout).expect("json face")
    };

    // the create replies carry the born task's token, both paths
    let created_local = local(&local_db, &["--json", "create", "task", "offline task"]);
    let served = spawn_server(&served_db).await;
    let (_, body) = post_command(
        &served,
        &json!({"context": {"actor": "pi", "tier": "agent"},
                 "command": {"create_task": {"name": "offline task"}}}),
    );
    let created_served = json_of(&body);
    assert_eq!(created_local["id"], created_served["id"]);
    assert_eq!(
        created_local["records"][0]["kind"],
        created_served["records"][0]["kind"]
    );

    // a claim answers with its records, both paths, identical shapes
    let claimed = local(&local_db, &["--json", "claim", "t-0"]);
    assert_eq!(claimed["records"][0]["kind"], "task_claimed");
    let (_, body) = post_command(
        &served,
        &json!({"context": {"actor": "pi", "tier": "agent"},
                 "command": {"claim_task": {"id": 0, "incarnation": null}}}),
    );
    let served_claimed = json_of(&body);
    assert_eq!(
        served_claimed["records"][0]["kind"],
        claimed["records"][0]["kind"]
    );

    std::fs::remove_dir_all(local_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(served_db.parent().unwrap()).unwrap();
}
