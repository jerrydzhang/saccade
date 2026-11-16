// ---- the workspace doors through the CLI face: a fake host, real exec ----

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use saccade::api::AppState;
use saccade::store::RecordId;

use crate::common::{
    bind_next_ask, bounded_sac, fake_hc, fake_hc_spawn, fake_host, get_json, hc_argv, json_of,
    post_json, pty_stdio, scratch_db, scratch_repo, scrubbed, spawn_serve,
    spawn_serving_subprocess, stub_pi, suite_state,
};

async fn spawn_workspace_server(db_path: &std::path::Path, repo: &std::path::Path) -> String {
    let state = AppState::open(db_path)
        .expect("the scratch tracker opens")
        .with_repo_root(repo.to_path_buf());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = saccade::serve::router(state);
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// Whether a guard file is held by a live process: the tests' own
/// kernel probe, mirroring the extension's flock take. Only the
/// documented would-block conflict reads as held; every other failure
/// is unknown.
fn guard_held(guard: &Path) -> Option<bool> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(guard)
        .ok()?;
    let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if taken == 0 {
        return Some(false);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::EWOULDBLOCK) => Some(true),
        _ => None,
    }
}

/// A fixture harness, not actual Pi: it takes the named real kernel guard, records what it saw, and stays alive until killed.
fn guard_pi(dir: &std::path::Path) -> std::path::PathBuf {
    let pi = dir.join("pi guard-holder");
    std::fs::write(
        &pi,
        r#"#!/bin/sh
{
  printf 'argv: %s\n' "$*"
  printf 'pwd: %s\n' "$PWD"
  env | grep '^SACCADE_' | sort
} > "$PI_DUMP"
printf '%s\n' "$$" > "$PI_PIDFILE"
exec 9>>"$SACCADE_TEST_GUARD"
if ! flock -n 9; then
  printf 'guard-busy\n' >> "$FIXTURE_LOG"
  exit 3
fi
printf 'guard-taken\n' >> "$FIXTURE_LOG"
# exec: this pid becomes the sleep and the sole lock holder, so one kill frees the guard
exec sleep 180
"#,
    )
    .unwrap();
    std::fs::set_permissions(&pi, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    pi
}

/// The identity door through the CLI face: the environment's
/// declaration read back, the retired task-bound words refused at the
/// parser, and the discovered server arming a launch it did not name.
#[tokio::test(flavor = "multi_thread")]
async fn identity_reads_the_scope_and_the_retired_words_refuse_at_the_parser() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("ws-cli");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let host = fake_host(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    let host_state = db.parent().unwrap().join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();

    // every sac run rides the fake host's PATH, scrubbed of inherited scope
    let run = |args: &[String], envs: &[(&str, &str)]| {
        let mut cmd = scrubbed(std::process::Command::new(bin));
        cmd.env(
            "PATH",
            format!(
                "{}:{}",
                host.parent().unwrap().display(),
                std::env::var("PATH").unwrap()
            ),
        )
        .env("FAKE_ORCA_LOG", host_log.display().to_string())
        .env("FAKE_ORCA_STATE", host_state.display().to_string())
        .env("SACCADE_PI", &pi)
        .env_remove("SACCADE_ACTOR")
        .env_remove("SACCADE_REPO")
        .env_remove("SACCADE_DB")
        .env_remove("SACCADE_SERVER")
        .env_remove("SACCADE_TASK")
        .env_remove("SACCADE_INCARNATION_SEQ")
        .env_remove("SACCADE_WORKSPACE")
        .current_dir(&root);
        cmd.args(args);
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let routed = |sub: &[&str]| -> Vec<String> {
        let mut full = vec!["--server".to_string(), base_url.clone()];
        full.extend(sub.iter().map(|s| s.to_string()));
        full
    };
    let at_repo = |mut args: Vec<String>| -> Vec<String> {
        args.push("--repo".into());
        args.push(root.display().to_string());
        args
    };

    // identity: the environment's declaration, read-only
    let worktree = db.parent().unwrap().join("declared-ws");
    std::fs::create_dir_all(&worktree).unwrap();
    let (ok, out, err) = run(
        &at_repo(routed(&["identity", "--json"])),
        &[
            ("SACCADE_ACTOR", "pi"),
            ("SACCADE_TASK", "t-0"),
            ("SACCADE_INCARNATION_SEQ", "1"),
            ("SACCADE_WORKSPACE", worktree.to_str().unwrap()),
        ],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    let identity: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(identity["actor"], "pi");
    assert_eq!(identity["tier"], "agent");
    assert_eq!(identity["task"], "t-0");
    // the machine channel discloses the birth position as a number;
    // the public i-N is the fold's dense numbering, never this face
    assert_eq!(identity["incarnation_seq"], 1);
    assert!(identity.get("incarnation").is_none());
    assert_eq!(identity["workspace"], worktree.display().to_string());

    // the retired task-bound words are CLI unknown: the parser itself
    // refuses them, no alias or redirect keeps them callable
    for word in ["start", "open", "exec", "checkpoint", "clean", "integrate"] {
        let (ok, out, err) = run(&routed(&[word]), &[("SACCADE_ACTOR", "pi")]);
        assert!(!ok, "{word} must refuse: {out}{err}");
        assert!(err.contains("unrecognized subcommand"), "{word}: {err}");
    }

    // discovered routing: no --server is named; the launch door resolves
    // the repo's published server and arms the native session with it.
    // The tracker lives at the repo's state root, so serve and clients
    // share one XDG_STATE_HOME
    let db2 = scratch_db("ws-cli-discovered");
    let (root2, _first2, _head2) = scratch_repo(db2.parent().unwrap());
    let state_home2 = db2.parent().unwrap().join("state-home");
    let root2c = std::fs::canonicalize(&root2).unwrap();
    let tracker2 = state_home2
        .join("saccade")
        .join(format!(
            "{}-{:06x}",
            saccade::paths::repo_basename(&root2c),
            saccade::paths::fnv1a(root2c.as_os_str().as_encoded_bytes())
        ))
        .join("saccade.db");
    let (mut serve2, discovered_url) = spawn_serve(
        bin,
        &tracker2,
        &["--repo", root2.to_str().unwrap()],
        &[("XDG_STATE_HOME", state_home2.to_str().unwrap())],
    );
    let discovered_envs = [
        ("SACCADE_ACTOR", "pi"),
        ("XDG_STATE_HOME", state_home2.to_str().unwrap()),
    ];
    let _ = &discovered_envs;
    let discovered_args = |sub: &[&str]| {
        let mut full = vec!["--repo".to_string(), root2.display().to_string()];
        full.extend(sub.iter().map(|s| s.to_string()));
        full
    };
    std::fs::write(&host_log, b"").unwrap();
    let chosen = db2.parent().unwrap().join("chosen-ws");
    std::fs::create_dir_all(&chosen).unwrap();
    // the launch rides the adapter: the fake hc answers and logs
    let hc2 = fake_hc(db2.parent().unwrap());
    let (ok, out, err) = run(
        &discovered_args(&["launch", "--workspace", chosen.to_str().unwrap()]),
        &[
            ("SACCADE_ACTOR", "pi"),
            ("XDG_STATE_HOME", state_home2.to_str().unwrap()),
            ("SACCADE_HERDR", hc2.to_str().unwrap()),
        ],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    let typed2 = hc_argv(&db2.parent().unwrap().join("hc-bin"), "run").join(" ");
    assert!(
        !typed2.contains("SACCADE_SERVER=--server"),
        "no explicit flag: the discovery named the destination: {typed2}"
    );
    assert!(
        typed2.contains(&format!("SACCADE_SERVER={}", discovered_url)),
        "the discovered server arms the session: {typed2}"
    );
    assert!(
        typed2.contains(&format!("SACCADE_REPO={}", root2.display())),
        "{typed2}"
    );

    let _ = serve2.kill();
    let _ = serve2.wait();
    std::fs::remove_dir_all(db2.parent().unwrap()).unwrap();

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn launch_with_direnv_stripped_from_the_path() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("launch-no-direnv");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let host = fake_host(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    std::fs::write(&host_log, b"").unwrap();

    let workspace = db.parent().unwrap().join("bare-workspace");
    std::fs::create_dir_all(&workspace).unwrap();

    // this one spawns raw on purpose: the contract under test is the
    // ambient resolution itself (PATH bait, no override), so the
    // helper's fail-safes would change the answer. Only the state
    // root and the cwd still point at scratch.
    let out = std::process::Command::new(bin)
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "launch",
            "--workspace",
            workspace.to_str().unwrap(),
            "--",
            "register and wait",
        ])
        // the launch composes the caller's location, so this child
        // runs from the scratch workspace
        .current_dir(&workspace)
        .env("XDG_STATE_HOME", suite_state())
        // the fake host sits on PATH as ambient bait; direnv deliberately is not
        .env("PATH", host.parent().unwrap().display().to_string())
        .env("FAKE_ORCA_LOG", host_log.display().to_string())
        .env("SACCADE_PI", &pi)
        .output()
        .expect("spawn sac");
    // the pin is a compile fact, not a runtime choice: a packaged build
    // carries direnv no matter what PATH says and must not pretend to
    // forget it, while a cargo build has no pin and refuses before the
    // host is asked. Each build flavor asserts its own honest contract.
    if option_env!("SACCADE_DIRENV").is_some() {
        assert!(
            out.status.success(),
            "the baked pin governs over a stripped PATH: stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        assert!(
            !out.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(err.contains("direnv"), "{err}");
        assert!(err.contains("never starts unequipped"), "{err}");
        // the refusal precedes any host contact: the ambient bait saw nothing
        assert_eq!(std::fs::read_to_string(&host_log).unwrap(), "");
    }

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_launch_scrubs_stale_inheritance() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("launch-scrub");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());

    let workspace = db.parent().unwrap().join("chosen-workspace");
    std::fs::create_dir_all(&workspace).unwrap();

    let out = scrubbed(std::process::Command::new(bin))
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "launch",
            "--workspace",
            workspace.to_str().unwrap(),
            "--",
            "register and wait",
        ])
        .env("SACCADE_HERDR", &hc)
        .env("SACCADE_PI", &pi)
        // everything stale the launching shell happens to carry
        .env("SACCADE_INCARNATION_SEQ", "999")
        .env("SACCADE_TASK", "t-0")
        .env("SACCADE_WORKSPACE", "/nowhere")
        .env("SACCADE_DB", "/stale/routing.db")
        .env("SACCADE_CLI", "/stale/sac")
        .env("SACCADE_TEST_CONTEXT", "1")
        .output()
        .expect("spawn sac");
    let ok = out.status.success();
    let (out, err) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");

    let typed = hc_argv(&db.parent().unwrap().join("hc-bin"), "run").join(" ");
    let expected_scrub = "-u SACCADE_INCARNATION -u SACCADE_INCARNATION_SEQ \
                         -u SACCADE_TASK -u SACCADE_WORKSPACE -u SACCADE_DB -u SACCADE_CLI \
                         -u SACCADE_TEST_CONTEXT";
    assert!(
        typed.contains(expected_scrub),
        "the launch scrubs stale inheritance: {typed}"
    );
    assert!(
        typed.contains(&format!(
            "SACCADE_ACTOR={}",
            command_label(&pi.display().to_string())
        )),
        "the launch names the harness as the acting agent: {typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_SERVER={base_url}")),
        "{typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_REPO={}", root.display())),
        "{typed}"
    );
    // the workspace command routes through direnv exec so the committed
    // .envrc governs the session, targeting the launch workspace; the
    // session stays the terminal foreground (direnv exec, no shell)
    let exec_at = typed
        .find(" exec ")
        .expect("the launch routes through direnv");
    assert!(
        typed[exec_at..].starts_with(&format!(
            " exec {} env -u SACCADE_INCARNATION",
            workspace.display()
        )),
        "direnv exec targets the launch workspace: {typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_CLI={}", env!("CARGO_BIN_EXE_sac"))),
        "{typed}"
    );
    assert!(
        !typed.contains("/stale/sac"),
        "the stale CLI path never survives the scrub: {typed}"
    );
    assert!(
        typed.ends_with("--session-dir") || typed.ends_with("register and wait"),
        "the composed prompt rides the argv tail: {typed}"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A sac run with stdin on a pty slave: the here door's attach
/// inherits a real terminal, and the door's own faces come back
/// through collected pipes.
fn run_here_in_pty(bin: &str, args: &[String], envs: &[(&str, String)]) -> std::process::Output {
    let mut masters = Vec::new();
    let mut cmd = scrubbed(std::process::Command::new(bin));
    cmd.args(args.iter().map(String::as_str))
        .stdin(pty_stdio(&mut masters));
    for key in [
        "SACCADE_REPO",
        "SACCADE_DB",
        "SACCADE_SERVER",
        "SACCADE_ACTOR",
        "SACCADE_TASK",
        "SACCADE_INCARNATION_SEQ",
        "SACCADE_WORKSPACE",
        "HC_TERMINAL_ID",
    ] {
        cmd.env_remove(key);
    }
    // no pty door ever resolves the ambient adapter: the default host
    // path denies loudly (no such executable, every ask fails) unless
    // the case names its own fake
    cmd.env(
        "SACCADE_HERDR",
        std::env::temp_dir().join("saccade-pty-deny-hc"),
    );
    cmd.env("SACCADE_TEST_CONTEXT", "1");
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("spawn sac");
    for master in masters {
        unsafe { libc::close(master) };
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn the_here_door_launches_through_the_host_then_attaches_this_terminal() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("here-attach");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let workspace = db.parent().unwrap().join("here-workspace");
    std::fs::create_dir_all(&workspace).unwrap();

    let args = |extra: &[&str]| {
        let mut args = vec![
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--workspace".to_string(),
            workspace.to_str().unwrap().to_string(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_string()));
        args
    };
    let envs = [
        ("SACCADE_HERDR", hc.display().to_string()),
        ("SACCADE_PI", pi.display().to_string()),
    ];

    // the attached view: a real stdin terminal launches through the
    // host, births the incarnation, then attaches; the attach's own
    // exit is the command's and the json face names the harness
    let out = run_here_in_pty(bin, &args(&["--here", "--json"]), &envs);
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // the attached view rides the inherited stdio beside the reply:
    // the fake's own screen bytes land in the same stream
    let text = String::from_utf8_lossy(&out.stdout);
    let json_at = text.find('{').expect("the json face answers");
    let launched: Value = serde_json::from_str(&text[json_at..]).unwrap();
    assert_eq!(launched["terminal"], json!("t1"), "{launched}");
    assert_eq!(launched["harness"], json!("pi"), "{launched}");
    assert_eq!(launched["resumed"], Value::Null, "{launched}");
    assert_eq!(launched["exit"], json!(0), "{launched}");
    assert!(launched["native_id"].is_string(), "{launched}");
    // the attach went to the launched address, bare
    assert_eq!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "attach"),
        vec!["t1".to_string()]
    );
    // exactly one host run carried the launch, and no label was asked
    // the fake logs one word per line; every ask starts with its verb
    let asks = |verb: &str| {
        std::fs::read_to_string(
            db.parent()
                .unwrap()
                .join("hc-bin")
                .join(format!("argv-{verb}.log")),
        )
        .unwrap_or_default()
        .lines()
        .filter(|line| *line == verb)
        .count()
    };
    assert_eq!(asks("run"), 1, "exactly one host run");
    assert_eq!(asks("label"), 0);
    assert_eq!(asks("ls"), 0);
    assert_eq!(asks("get"), 0);
    // the json face carries no space, no tab, no refusal
    assert_eq!(launched.get("space"), None);
    assert_eq!(launched.get("tab"), None);
    assert_eq!(launched.get("place_refusal"), None);
    // the birth is a launched kind carrying its harness - never this
    // pane, never an adoption
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let fold = json_of(&body);
    let births: Vec<&Value> = fold["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "incarnation_registered")
        .collect();
    assert_eq!(births.len(), 1, "{fold}");
    assert_eq!(births[0]["payload"]["address"], json!("t1"), "{fold}");
    assert_eq!(births[0]["payload"]["adopted"], json!(false), "{fold}");
    assert_eq!(births[0]["payload"]["harness"], json!("pi"), "{fold}");

    // the text face names the detach law: the attach ended, the
    // session keeps running, and the address stands for a reattach
    let out = run_here_in_pty(bin, &args(&["--here"]), &envs);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("the session keeps running"), "{text}");
    assert!(text.contains("hc attach t1"), "{text}");

    // the resumed here face: one get is the liveness ask, no label asks
    let here_saved = db.parent().unwrap().join("hereres1.jsonl");
    std::fs::write(
        &here_saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"hereres001\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n{{\"type\":\"message\",\"id\":\"aa\",\"parentId\":null}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "hereres001", "session_file": here_saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    let here_num = json_of(&body)["num"].as_u64().unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/address",
        &json!({"incarnation": here_num, "address": "t1", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    std::fs::write(
        db.parent().unwrap().join("hc-bin").join("state"),
        b"finished",
    )
    .unwrap();
    let gets_before = asks("get");
    let runs_before = asks("run");
    let resumed_args = vec![
        "--server".to_string(),
        base_url.clone(),
        "--repo".to_string(),
        root.to_str().unwrap().to_string(),
        "launch".to_string(),
        "--here".to_string(),
        "--resume".to_string(),
        format!("i-{here_num}"),
        "--json".to_string(),
    ];
    let out = run_here_in_pty(bin, &resumed_args, &envs);
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let json_at = text.find('{').expect("the json face answers");
    let resumed_here: Value = serde_json::from_str(&text[json_at..]).unwrap();
    assert_eq!(resumed_here["resumed"], json!(here_num), "{resumed_here}");
    assert_eq!(resumed_here["terminal"], json!("t1"), "{resumed_here}");
    assert_eq!(resumed_here.get("space"), None);
    assert_eq!(resumed_here.get("tab"), None);
    assert_eq!(asks("run"), runs_before + 1, "exactly one more run");
    let gets = asks("get");
    assert_eq!(gets, gets_before + 1, "one get: the liveness ask");
    assert_eq!(asks("label"), 0);
    assert_eq!(asks("ls"), 0);
    assert_eq!(asks("attach"), 3, "one attach per here launch");

    // a refused attach discloses the standing launch: the address and
    // the by-hand command; the birth stands either way
    let mut failed_envs = envs.to_vec();
    failed_envs.push(("SACCADE_FAKE_HC_ATTACH_FAILS", "1".to_string()));
    let out = run_here_in_pty(bin, &args(&["--here"]), &failed_envs);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("the launch stands at address t1"), "{err}");
    assert!(err.contains("hc attach t1"), "{err}");
    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == "incarnation_registered")
            .count(),
        4,
        "{body}"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_here_door_refuses_a_pipe_and_a_nested_pane_before_any_launch() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("here-refusals");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let workspace = db.parent().unwrap().join("here-refused");
    std::fs::create_dir_all(&workspace).unwrap();

    // an incarnation already carries a pane: the nested-guard target
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "nestor001", "address": "tp", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");

    let mut args = vec![
        "--server".to_string(),
        base_url.clone(),
        "--repo".to_string(),
        root.to_str().unwrap().to_string(),
        "launch".to_string(),
        "--workspace".to_string(),
        workspace.to_str().unwrap().to_string(),
        "--here".to_string(),
    ];
    let envs = [
        ("SACCADE_HERDR", hc.display().to_string()),
        ("SACCADE_PI", pi.display().to_string()),
    ];

    // no real stdin terminal: the attach would inherit a pipe, and the
    // door refuses before any host contact
    let (ok, _out, err, timed_out) = bounded_sac(bin, &args, &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("stdin must be a real terminal"), "{err}");
    assert!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "run").is_empty(),
        "the refusal preceded any host contact"
    );

    // the invoking pane is a recorded incarnation's address: a
    // foreground attach would route that incarnation's input into the
    // child, so the door refuses before anything runs
    args.push("--json".to_string());
    let mut nested = envs.to_vec();
    nested.push(("HC_TERMINAL_ID", "tp".to_string()));
    let (ok, _out, err, timed_out) = bounded_sac(bin, &args, &nested);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("already the address of i-0"), "{err}");
    assert!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "run").is_empty(),
        "the nested refusal preceded any host contact"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_resume_runs_its_recorded_harness_and_an_explicit_mismatch_refuses() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("resume-harness");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let workspace = db.parent().unwrap().join("pi-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let saved = db.parent().unwrap().join("resumee01.jsonl");
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"resumee01\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n{{\"type\":\"message\",\"id\":\"aa\",\"parentId\":null}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");

    let resume_args = |extra: &[&str]| {
        let mut args = vec![
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--resume".to_string(),
            "i-0".to_string(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_string()));
        args
    };
    let envs = [
        ("SACCADE_HERDR", hc.display().to_string()),
        ("SACCADE_PI", pi.display().to_string()),
    ];

    // the recorded kind is the authority: claiming another kind is a
    // wrong claim about the session, refused before any host contact
    let (ok, _out, err, timed_out) = bounded_sac(
        bin,
        &resume_args(&["--harness", "claude-code", "--json"]),
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("runs pi as recorded"), "{err}");
    assert!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "run").is_empty(),
        "the mismatch refused before any host contact"
    );

    // the explicit matching claim rides the recorded harness: the
    // launch proceeds and the reply names pi
    let (ok, out, err, timed_out) =
        bounded_sac(bin, &resume_args(&["--harness", "pi", "--json"]), &envs);
    assert!(!timed_out, "{err}");
    assert!(ok, "{out}{err}");
    let launched = json_of(&out);
    assert_eq!(launched["harness"], json!("pi"), "{out}");
    assert_eq!(launched["native_id"], Value::Null, "{out}");
    let run_log = hc_argv(&db.parent().unwrap().join("hc-bin"), "run").join(" ");
    assert!(run_log.contains("--session resumee01"), "{run_log}");

    // the reassignment moved to the attach face: the submitted resume
    // records no address move at all
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let fold = json_of(&body);
    let rows = fold["records"].as_array().unwrap();
    assert!(
        rows.iter().all(|r| r["kind"] != "host_address_assigned"),
        "a submitted resume moves no address: {fold}"
    );

    // the presented pair moves the association at attach evidence
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t1", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records.len(), 1, "{body}");
    assert_eq!(records[0]["kind"], "host_address_assigned", "{body}");
    assert_eq!(records[0]["payload"]["address"], "t1", "{body}");
    assert_eq!(records[0]["payload"]["adopted"], false, "{body}");

    // the equal pair is silent: reload storms re-attach, not re-move
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t1", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["records"].as_array().unwrap().is_empty(),
        "{body}"
    );

    // the mode rides the pair: an adopted pane is the recorded kind
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t5", "adopted": true}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records[0]["payload"]["address"], "t5", "{body}");
    assert_eq!(records[0]["payload"]["adopted"], true, "{body}");
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    let resolved = world
        .incarnations
        .values()
        .find(|ctx| ctx.incarnation.native_id().map(|n| n.as_str()) == Some("resumee01"))
        .unwrap();
    assert_eq!(resolved.incarnation.address(), Some("t5"), "{resolved:?}");
    assert!(resolved.incarnation.adopted());

    // a fresh birth keeps its launch-time address: the same-session
    // attach presenting the equal pair binds the locator, moves nothing
    let (ok, out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--workspace".to_string(),
            workspace.display().to_string(),
            "--json".to_string(),
        ],
        &envs,
    );
    assert!(ok, "{out}{err}");
    assert!(!timed_out, "{err}");
    let fresh_id = json_of(&out)["native_id"].as_str().unwrap().to_string();
    let fresh_saved = format!("/sessions/{fresh_id}.jsonl");
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": fresh_id, "session_file": fresh_saved,
                "terminal": {"address": "t1", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records.len(), 1, "{body}");
    assert_eq!(records[0]["kind"], "saved_session_bound", "{body}");

    // the claude-code face presents the same pair through its own
    // harness: the wire carries it, the birth lands addressed
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "hookcc001", "session_file": "/sessions/hookcc001.jsonl",
                "harness": "claude-code",
                "terminal": {"address": "t8", "adopted": true}}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], true, "{body}");
    let records = reply["records"].as_array().unwrap();
    assert_eq!(records[0]["kind"], "incarnation_registered", "{body}");
    assert_eq!(records[0]["payload"]["address"], "t8", "{body}");
    assert_eq!(records[0]["payload"]["adopted"], true, "{body}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[test]
fn the_sole_writer_submits_only_through_the_adapter_and_launches_nothing() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("wire-adapter-submit");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let hc = fake_hc(db.parent().unwrap());
    let bin_dir = db.parent().unwrap().join("hc-bin");
    let cold_state = db.parent().unwrap().join("xdg-state");
    let (url, pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        &bin_dir,
        Some(&cold_state),
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // a registered sender and one addressed recipient
    for (native_id, address) in [("sender0001", "t4"), ("peer00001", "t5")] {
        let (status, body) = post_json(
            &url,
            "/api/v1/incarnations/register",
            &json!({"native_id": native_id, "address": address, "adopted": false}),
        );
        assert_eq!(status, 200, "{body}");
    }

    // the send records, submits through the adapter, and records the
    // acceptance: one ask at the recorded address, nothing else
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": 1,
            "body": "wake the addressed peer",
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["submission"]["state"], "submitted", "{body}");
    // the submission carries the wire envelope: one origin line, a
    // blank line, the body untouched - no reply, no task here
    let asks = hc_argv(&bin_dir, "send");
    assert_eq!(asks.len(), 4, "{asks:?}");
    assert_eq!(asks[0], "t5");
    assert_eq!(asks[1], "[saccade message m-0 | from i-0]");
    assert_eq!(asks[2], "");
    assert_eq!(asks[3], "wake the addressed peer");

    // no launch ever reached the host: the send path asks send alone
    let runs = std::fs::read_to_string(bin_dir.join("argv-run.log")).unwrap_or_default();
    assert!(runs.trim().is_empty(), "a send launched a session: {runs}");

    let _ = std::process::Command::new("kill")
        .arg(pid.id().to_string())
        .status();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// One owned real-door fixture: the fake host, the guard-holding harness, their logs, and the cold state home.
struct DoorFixture {
    host: PathBuf,
    hcs: PathBuf,
    pi: PathBuf,
    host_log: PathBuf,
    child_log: PathBuf,
    dump: PathBuf,
    pidfile: PathBuf,
    cold_state: PathBuf,
}

fn door_fixture(home: &Path) -> DoorFixture {
    let host = fake_host(home);
    let hcs = fake_hc_spawn(home);
    let pi = guard_pi(home);
    let host_log = home.join("host.log");
    let child_log = home.join("child.log");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&child_log, b"").unwrap();
    DoorFixture {
        host,
        hcs,
        pi,
        host_log,
        child_log,
        dump: home.join("stub.dump"),
        pidfile: home.join("stub.pid"),
        cold_state: home.join("xdg-state"),
    }
}

/// The environment one real-door subprocess runs under: fake host first on PATH, the fixture harness as pinned Pi, and the cold state home the fixture server named its guard under.
fn door_envs(fx: &DoorFixture, guard: Option<&Path>) -> Vec<(&'static str, String)> {
    let mut envs = vec![
        (
            "PATH",
            format!(
                "{}:{}",
                fx.host.parent().unwrap().display(),
                std::env::var("PATH").unwrap()
            ),
        ),
        ("SACCADE_HERDR", fx.hcs.display().to_string()),
        ("XDG_STATE_HOME", fx.cold_state.display().to_string()),
        ("SACCADE_PI", fx.pi.display().to_string()),
        ("PI_DUMP", fx.dump.display().to_string()),
        ("PI_PIDFILE", fx.pidfile.display().to_string()),
        ("FIXTURE_LOG", fx.child_log.display().to_string()),
    ];
    if let Some(guard) = guard {
        envs.push(("SACCADE_TEST_GUARD", guard.display().to_string()));
    }
    envs
}

/// Kill the fixture holder by its pidfile and wait for its guard to actually free.
fn stop_holder(pidfile: &Path, guard: &Path) {
    if let Ok(pid) = std::fs::read_to_string(pidfile) {
        let _ = std::process::Command::new("kill")
            .arg("-9")
            .arg(pid.trim())
            .status();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while guard_held(guard) != Some(false) {
        assert!(
            std::time::Instant::now() < deadline,
            "the fixture holder's guard never freed"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[test]
fn the_real_resume_door_submits_without_a_guard_consult() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("resume-door");
    let home = db.parent().unwrap();
    let (root, _first, _head) = scratch_repo(home);
    let fx = door_fixture(home);
    let (url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        fx.host.parent().unwrap(),
        Some(&fx.cold_state),
        &[],
    );

    // the resumable session: a valid saved header, existing workspace,
    // no live process; the guard is the sidecar beside the saved file
    let workspace = home.join("resumed-ws");
    std::fs::create_dir_all(&workspace).unwrap();
    let saved = home.join("resumee-session.jsonl");
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"resumee01\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["num"], 0);
    let guard = PathBuf::from(format!("{}.guard", saved.display()));

    // a tampered association: recorded, then the file's identity rewritten under it
    let tampered = home.join("tampered-session.jsonl");
    std::fs::write(
        &tampered,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"other002\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "other002", "session_file": tampered.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    std::fs::write(
        &tampered,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"rewritten01\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    // a missing saved file refuses the same way
    let absent = home.join("absent-session.jsonl");
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "missing003", "session_file": absent.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");

    let envs = door_envs(&fx, Some(&guard));
    let door_args = |resume: &str| {
        vec![
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--resume".to_string(),
            resume.to_string(),
            "--json".to_string(),
        ]
    };
    let (ok, out, err, timed_out) = bounded_sac(bin, &door_args("i-1"), &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{out}");
    assert!(err.contains("carries id rewritten01"), "{err}");
    let (ok, _out, err, timed_out) = bounded_sac(bin, &door_args("i-2"), &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("cannot resume"), "{err}");
    assert!(
        std::fs::read_to_string(&fx.host_log).unwrap().is_empty(),
        "a refused resume must contact no host"
    );

    // an addressless resume consults no guard: a preexisting holder
    // (a live pre-B session) no longer refuses the door - the twin
    // catch moved to the resumed session's own attach, and the pre-B
    // migration story is exit + resume. The same launch is the
    // corrected door: the adapter's synchronous ask returns the
    // terminal, the association waits for the attach face, nothing
    // waits on a guard handoff - hc run is synchronous, attach is
    // the harness's
    let held = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&guard)
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    assert_eq!(guard_held(&guard), Some(true));
    let (ok, out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &envs);
    assert!(!timed_out, "the door outwaited its launch: {err}");
    assert!(
        ok,
        "the door consults no guard for an addressless resume: {err}"
    );
    // the door consults no guard for an addressless resume: it succeeded
    // while the guard was held above, and it took nothing itself. After
    // the drop, the resumed session's own attach may already hold the
    // guard - that holder is the session's, never the door's, so the
    // free-state read here would race the child and pin nothing
    drop(held);
    let launched = json_of(&out);
    assert_eq!(
        launched["workspace"],
        json!(workspace.display().to_string()),
        "{out}"
    );
    assert_eq!(launched["terminal"], "t9", "{out}");
    // the launch consults no label ask
    assert_eq!(launched["space"], Value::Null, "{out}");
    assert_eq!(launched["tab"], Value::Null, "{out}");
    assert_eq!(launched["place_refusal"], Value::Null, "{out}");
    let label: Vec<String> = hc_argv(fx.hcs.parent().unwrap(), "label");
    assert!(label.is_empty(), "no label ask rides a resume: {label:?}");

    // the composition rode the host argv: the resume session, its
    // locator, the settled flags
    let submitted_lines = hc_argv(fx.hcs.parent().unwrap(), "run");
    let submitted = submitted_lines.join(" ");
    let log_body = std::fs::read_to_string(home.join("hcs-bin").join("argv-run.log"))
        .unwrap_or_else(|e| format!("unreadable: {e}"));

    assert!(
        submitted.contains("--session resumee01"),
        "submitted=[{submitted}] lines=[{submitted_lines:?}] body=[{log_body}]"
    );
    assert!(submitted.contains("--session-dir"), "{submitted}");
    assert!(submitted.contains("--no-parent"), "{submitted}");
    assert!(submitted.contains("--no-notify"), "{submitted}");

    // the harness ran as the terminal foreground: equip, scrub, argv,
    // and cwd all arrived, and the guard was taken by the harness
    let dump = std::fs::read_to_string(&fx.dump).unwrap();
    assert!(dump.contains("--session resumee01"), "{dump}");
    assert!(dump.contains("--session-dir"), "{dump}");
    assert!(
        dump.contains(&format!("pwd: {}", workspace.display())),
        "{dump}"
    );
    assert!(dump.contains(&format!("SACCADE_SERVER={url}")), "{dump}");
    assert!(
        dump.contains(&format!("SACCADE_REPO={}", root.display())),
        "{dump}"
    );
    assert!(dump.contains("SACCADE_ACTOR="), "{dump}");
    assert!(
        !dump.contains("SACCADE_TASK="),
        "stale scope must not ride the launch: {dump}"
    );
    assert!(!dump.contains("SACCADE_INCARNATION_SEQ="), "{dump}");

    // the address move waits for the attach face: a submitted resume
    // records no move, and the presented pair lands it at evidence
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    assert_eq!(
        world.incarnations[&saccade::record::objects::incarnation::IncarnationId(RecordId(0))]
            .incarnation
            .address(),
        None,
        "a submitted resume moves no address"
    );
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t9", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records.len(), 1, "{body}");
    assert_eq!(records[0]["kind"], "host_address_assigned", "{body}");
    assert_eq!(records[0]["payload"]["address"], "t9", "{body}");
    stop_holder(&fx.pidfile, &guard);

    // a live recorded address refuses the resume
    let run_log = fx.hcs.parent().unwrap().join("argv-run.log");
    let runs_before = std::fs::read_to_string(&run_log).unwrap();
    let mut live_envs = envs.clone();
    live_envs.push(("SACCADE_FAKE_HC_STATE", "live".to_string()));
    let (ok, _out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &live_envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "a live address must refuse the resume");
    assert!(
        err.contains("is live at its recorded host address"),
        "{err}"
    );
    let mut unknown_envs = envs.clone();
    unknown_envs.push(("SACCADE_FAKE_HC_STATE", "mangled".to_string()));
    let (ok, _out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &unknown_envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "an unprobeable address must refuse the resume");
    assert!(err.contains("liveness cannot be read"), "{err}");
    assert!(err.contains("mangled"), "{err}");
    // the refusals probed the address and submitted no launch
    let runs_after = std::fs::read_to_string(&run_log).unwrap();
    assert_eq!(runs_after, runs_before, "the refusals submitted no launch");

    let _ = std::process::Command::new("kill")
        .arg(server_pid.id().to_string())
        .status();
    std::fs::remove_dir_all(home).unwrap();
}

#[test]
fn a_send_to_a_finished_address_discloses_through_the_real_door() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("send-real-door");
    let home = db.parent().unwrap();
    let (root, _first, _head) = scratch_repo(home);
    let fx = door_fixture(home);
    // the server submits through its own pinned fake host; its send arm
    // refuses the t9 address with the host's finished evidence
    let (url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        fx.host.parent().unwrap(),
        Some(&fx.cold_state),
        &[("SACCADE_HERDR", fx.hcs.to_str().unwrap())],
    );

    // a live sender and the finished-address peer
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "sender001", "address": "t1", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let sender_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "peer00001", "address": "t9", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let peer_birth = json_of(&body)["incarnation"].as_u64().unwrap();

    // the send records, then its attempt refuses at the host: the
    // address is finished, and no launch is ever submitted
    let host_before = std::fs::read_to_string(&fx.host_log)
        .unwrap_or_default()
        .len();
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": sender_birth, "workspace": null}},
            "to": 1,
            "body": "wake a finished address",
        }),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "recipient_not_live");
    let detail = json_of(&body)["error"]["detail"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(detail.contains("not live"), "{detail}");
    assert_eq!(json_of(&body)["message"]["num"], 0);
    // no launch reached the host: the log grew by nothing
    let host_after = std::fs::read_to_string(&fx.host_log)
        .unwrap_or_default()
        .len();
    assert_eq!(host_after, host_before, "a send launched a session");

    // delivery facts stay honest: the record stands pending, evidence
    // of the attempt, never a promise
    let (_, body) = get_json(&url, &format!("/api/v1/messages?incarnation={peer_birth}"));
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert!(
        mail.iter()
            .any(|m| m["from"]["num"] == 0 && m["state"] == "pending"),
        "{body}"
    );

    // the CLI send door names the recorded message beside the refusal
    let (ok, _out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "send".to_string(),
            "--to".to_string(),
            "i-1".to_string(),
            "status check".to_string(),
        ],
        &[
            ("SACCADE_ACTOR", "pi".to_string()),
            ("SACCADE_INCARNATION_SEQ", sender_birth.to_string()),
        ],
    );
    assert!(!timed_out, "{err}");
    assert!(!ok, "a refused submission exited success");
    assert!(err.contains("m-1 was recorded"), "{err}");
    assert!(err.contains("recipient_not_live"), "{err}");
    assert!(err.contains("stays pending"), "{err}");

    let _ = std::process::Command::new("kill")
        .arg(server_pid.id().to_string())
        .status();
    std::fs::remove_dir_all(home).unwrap();
}

#[test]
fn the_ask_door_introduces_and_the_send_door_discloses() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("activation-cli");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    // the serving subprocess submits through the fake host: sends to
    // its run address t1 succeed, the finished t9 refuses
    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let (base_url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        hc.parent().unwrap(),
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // the creator is a registered incarnation at its own address; the
    // finished-address peer gives the refusal its evidence
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "creator0001", "address": "t4", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let creator_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "gonepeer1", "address": "t9", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");

    // The initial question waits for the new session's startup binding.
    let workspace = db.parent().unwrap().join("peer-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let binding = bind_next_ask(&base_url);
    let out = scrubbed(std::process::Command::new(bin))
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "launch",
            "--workspace",
            workspace.to_str().unwrap(),
            "--ask",
            "what holds at the failing edge case?",
        ])
        .env("SACCADE_HERDR", &hc)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", creator_birth.to_string())
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    binding.join().unwrap();
    let out_text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "stdout: {out_text}\nstderr: {err}");
    // the newborn is i-2: creator i-0, the finished-address peer i-1
    assert!(out_text.contains("i-2 registered"), "{out_text}");
    assert!(
        out_text.contains("your question went as m-0 from i-0"),
        "{out_text}"
    );

    // the question is durable addressed communication from the
    // creator's own scope, submitted into the newborn's terminal: the
    // record reads submitted, and the host ask named the newborn's
    // address with the body verbatim
    // the mailbox read addresses the birth record position: the
    // newborn was born at seq 2
    let (_, body) = get_json(&base_url, "/api/v1/messages?incarnation=2");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert_eq!(mail.len(), 1, "{body}");
    assert_eq!(mail[0]["from"]["num"], 0);
    assert_eq!(mail[0]["to"]["num"], 2);
    assert_eq!(mail[0]["state"], "submitted");
    assert_eq!(mail[0]["body"], "what holds at the failing edge case?");
    let sends = hc_argv(hc.parent().unwrap(), "send");
    assert!(sends.contains(&"t1".to_string()), "{sends:?}");
    // the question rides the wire envelope: origin line, blank line,
    // the body untouched
    assert!(
        sends
            .iter()
            .any(|s| s == "[saccade message m-0 | from i-0]"),
        "{sends:?}"
    );
    assert!(
        sends
            .iter()
            .any(|s| s.starts_with("what holds at the failing edge case")),
        "{sends:?}"
    );

    // a plain send to the newborn rides the same submission
    let out = scrubbed(std::process::Command::new(bin))
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "send",
            "--to",
            "i-2",
            "status check",
        ])
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", creator_birth.to_string())
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(text.contains("sent m-1 to i-2"), "{text}");
    assert!(
        text.contains("accepted into their session's input queue"),
        "{text}"
    );

    // a send to the finished address refuses, naming the recorded
    // message beside the host's evidence
    let out = scrubbed(std::process::Command::new(bin))
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "send",
            "--to",
            "i-1",
            "wake the finished one",
        ])
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", creator_birth.to_string())
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(!out.status.success(), "a refused submission exited success");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("m-2 was recorded"), "{err}");
    assert!(err.contains("stays pending"), "{err}");
    assert!(err.contains("recipient_not_live"), "{err}");

    // without a declared incarnation the ask door refuses, taught
    let out = scrubbed(std::process::Command::new(bin))
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "launch",
            "--workspace",
            workspace.to_str().unwrap(),
            "--ask",
            "nobody sent this",
        ])
        .env("SACCADE_HERDR", &hc)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", "")
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("SACCADE_INCARNATION_SEQ"), "{err}");

    // An unbound launch remains recorded, but no initial question is recorded or sent.
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/address",
        &json!({"incarnation": 2, "address": "t2", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let send_log = hc.parent().unwrap().join("argv-send.log");
    let sends_before = std::fs::read_to_string(&send_log).unwrap();
    let started = std::time::Instant::now();
    let (ok, out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.display().to_string(),
            "launch".to_string(),
            "--workspace".to_string(),
            workspace.display().to_string(),
            "--ask".to_string(),
            "this question waits for startup".to_string(),
        ],
        &[
            ("SACCADE_HERDR", hc.display().to_string()),
            ("SACCADE_PI", pi.display().to_string()),
            ("SACCADE_ACTOR", "creator".to_string()),
            ("SACCADE_INCARNATION_SEQ", creator_birth.to_string()),
            // the test harness alone shortens the binding bound: the
            // timeout path costs seconds, never the production 30
            ("SACCADE_ASK_BIND_BOUND_SECS", "2".to_string()),
        ],
    );
    assert!(!timed_out, "{out}{err}");
    assert!(!ok, "an unbound startup was reported successful: {out}");
    assert!(err.contains("i-3 was launched at address t1"), "{err}");
    assert!(
        err.contains("initial question was not recorded or sent"),
        "{err}"
    );
    assert!(err.contains("within 2 seconds"), "{err}");
    assert!(started.elapsed() >= std::time::Duration::from_secs(2));
    assert!(
        started.elapsed() < std::time::Duration::from_secs(30),
        "the test knob bounded the wait; the production 30 s never ran"
    );
    assert_eq!(std::fs::read_to_string(send_log).unwrap(), sends_before);
    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["kind"] == "message_sent")
            .count(),
        3,
        "{body}"
    );

    let _ = std::process::Command::new("kill")
        .arg(server_pid.id().to_string())
        .status();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

fn command_label(program: &str) -> String {
    std::path::Path::new(program)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap()
}
