// ---- the workspace lifecycle: ordered cleanup, guards ----

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::common::{
    ServerProc, bounded_sac, envelope, fake_hc_spawn, get_json, json_of, post_command, post_json,
    scratch_db, scratch_repo, scrubbed, sh, spawn_serve, spawn_serving_subprocess, stub_pi,
    suite_state,
};

/// The host's real state parent, as the unhelped environment resolves
/// it: read-only evidence for the helper's pin.
fn real_state_parent() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_STATE_HOME").map(PathBuf::from) {
        return dir;
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
        .join(".local")
        .join("state")
}

/// A lifecycle rig: a scratch repo, a REAL `sac serve` subprocess with
/// the fake host on its PATH - the lifecycle effects run exactly where
/// they run in production, against the fake host and real git - and
/// the CLI face pointed at that server.
struct LifecycleRig {
    serve: ServerProc,
    url: String,
    dir: std::path::PathBuf,
    root: std::path::PathBuf,
    bin: &'static str,
    host_state: std::path::PathBuf,
    close_fail: std::path::PathBuf,
    host_env: Vec<(String, String)>,
}

fn lifecycle_rig(tag: &str) -> LifecycleRig {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-lifecycle-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (root, _first, _head) = scratch_repo(&dir);
    let db = dir.join("saccade.db");
    let pi = stub_pi(&dir);
    let host_state = dir.join("host-terminals");
    let close_fail = dir.join("close-fail");
    std::fs::write(&host_state, b"").unwrap();
    let hcs = fake_hc_spawn(&dir);
    let host_env = vec![
        ("SACCADE_HERDR".to_string(), hcs.display().to_string()),
        (
            "FAKE_HC_TERMINALS".to_string(),
            host_state.display().to_string(),
        ),
        (
            "FAKE_HC_CLOSE_FAIL".to_string(),
            close_fail.display().to_string(),
        ),
        // the rig is hermetic: the serve subprocess and every CLI run
        // name the stub harness, never an inherited or compiled pin
        ("SACCADE_PI".to_string(), pi.display().to_string()),
    ];
    let serve_envs: Vec<(&str, &str)> = host_env
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let (serve, url) = spawn_serve(
        bin,
        &db,
        &["--repo", &root.display().to_string()],
        &serve_envs,
    );
    LifecycleRig {
        serve,
        url,
        dir,
        root,
        bin,
        host_state,
        close_fail,
        host_env,
    }
}

impl Drop for LifecycleRig {
    fn drop(&mut self) {
        let _ = self.serve.kill();
        let _ = self.serve.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Provision one workspace through the create door; the reply's birth
/// seq and recorded facts come back.
fn rig_provision(rig: &LifecycleRig, label: &str) -> (usize, String, String) {
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/workspace/create",
        &json!({"label": label, "client": "lifecycle-test"}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    let seq = reply["workspace"]["seq"].as_u64().unwrap() as usize;
    let branch = reply["workspace"]["branch"].as_str().unwrap().to_string();
    let path = reply["workspace"]["path"].as_str().unwrap().to_string();
    (seq, branch, path)
}

/// An idle terminal the host names for the workspace.
fn rig_open_terminal(rig: &LifecycleRig, address: &str, path: &str) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&rig.host_state)
        .unwrap();
    writeln!(
        file,
        r#"{{"address":"{address}","cwd":"{path}","state":"live"}}"#
    )
    .unwrap();
}

/// The CLI face pointed at the rig's server, the fake host on PATH.
fn rig_sac(rig: &LifecycleRig, args: &[&str]) -> (bool, String, String) {
    let mut envs: Vec<(String, String)> =
        vec![("SACCADE_ACTOR".to_string(), "lifecycle-bot".to_string())];
    envs.extend(rig.host_env.clone());
    let env_refs: Vec<(&str, String)> = envs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let mut full = vec![
        "--server".to_string(),
        rig.url.clone(),
        "--repo".to_string(),
        rig.root.display().to_string(),
    ];
    full.extend(args.iter().map(|s| s.to_string()));
    let (ok, out, err, timed_out) = bounded_sac(rig.bin, &full, &env_refs);
    assert!(!timed_out, "sac {args:?} ran past its bound: {out}{err}");
    (ok, out, err)
}

/// The workspace is gone from git's own registrations.
fn rig_unregistered(rig: &LifecycleRig, path: &str) -> bool {
    let out = sh(&rig.root, &["worktree", "list", "--porcelain"]);
    !out.lines().any(|line| line == format!("worktree {path}"))
}

/// The recorded branch's local tip, empty when git holds none.
fn rig_tip(rig: &LifecycleRig, branch: &str) -> String {
    sh(
        &rig.root,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("refs/heads/{branch}"),
        ],
    )
    .trim()
    .to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_runs_the_ordered_contract_and_the_branch_stands() {
    let rig = lifecycle_rig("clean");
    let (seq, branch, ws) = rig_provision(&rig, "sweep");
    rig_open_terminal(&rig, "term_idle_1", &ws);
    let worktree_rows = |rig: &LifecycleRig| sh(&rig.root, &["worktree", "list", "--porcelain"]);

    // the clean runs the whole ordered contract over a real checkout
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "clean refuses: {out}{err}");
    // the Cleaned fact is the story's second record: the birth is seq 0
    assert!(out.contains("cleaned - record 1"), "{out}");
    let _ = seq;
    assert!(out.contains("closed 1 terminal(s)"), "{out}");
    assert!(out.contains("the host named each finished"), "{out}");
    assert!(out.contains(&format!("branch {branch} retained")), "{out}");
    assert!(
        out.contains("verified: the host names no terminals for the workspace"),
        "{out}"
    );
    assert!(!std::path::Path::new(&ws).exists(), "the checkout is gone");
    assert!(
        rig_unregistered(&rig, &ws),
        "git registers no worktree after the removal"
    );
    // the branch and its tip stand: retained, never deleted
    let tip = rig_tip(&rig, &branch);
    assert_eq!(tip, sh(&rig.root, &["rev-parse", "HEAD"]).trim());
    // the close ran at the terminal's own address, one kill per row
    let kills = std::fs::read_to_string(rig.dir.join("hcs-bin/argv-kill.log")).unwrap_or_default();
    assert!(kills.contains("term_idle_1"), "{kills}");
    let (status, body) = get_json(&rig.url, "/api/v1/records");
    assert_eq!(status, 200, "{body}");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        kinds.contains(&"workspace_cleaned".to_string()),
        "{kinds:?}"
    );

    // the idempotent re-run: recorded Cleaned plus verified absence,
    // no duplicate transition
    let records_before = body.clone();
    let (ok, out, _err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the idempotent re-run refuses: {out}");
    assert!(out.contains("already cleaned"), "{out}");
    let (_, body_after) = get_json(&rig.url, "/api/v1/records");
    assert_eq!(
        json_of(&records_before)["records"],
        json_of(&body_after)["records"],
        "the idempotent re-run landed no record"
    );

    // an empty tail: nothing was left behind
    let _ = worktree_rows(&rig);
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_refuses_dirt_missing_branch_and_unexpected_bytes() {
    let rig = lifecycle_rig("refuse");
    let (_, _branch, ws) = rig_provision(&rig, "guarded");

    // staged, unstaged, and untracked work refuses before any host or
    // git act; the checkout and the record both stand untouched
    std::fs::write(std::path::Path::new(&ws).join("untracked.txt"), "kept\n").unwrap();
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(!ok, "the dirty checkout cleaned anyway: {out}");
    assert!(err.contains("workspace_work_present"), "{err}");
    assert!(std::path::Path::new(&ws).join("untracked.txt").is_file());
    std::fs::remove_file(std::path::Path::new(&ws).join("untracked.txt")).unwrap();

    // unrecognized directories and the primary checkout refuse
    let stranger = rig.dir.join("stranger");
    std::fs::create_dir_all(&stranger).unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", stranger.to_str().unwrap()]);
    assert!(!ok, "an unrecognized directory cleaned: {err}");
    assert!(err.contains("workspace_not_recorded"), "{err}");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", rig.root.to_str().unwrap()]);
    assert!(!ok, "the primary checkout cleaned: {err}");
    assert!(err.contains("workspace_not_recorded"), "{err}");

    // a missing recorded branch refuses the destructive act: rename
    // the recorded branch away and the checkout is uncleanable
    let (seq, branch, ws2) = rig_provision(&rig, "branchless");
    let renamed = format!("{branch}-moved");
    sh(&rig.root, &["branch", "-m", &branch, &renamed]);
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws2]);
    assert!(!ok, "the branchless checkout cleaned: {err}");
    assert!(err.contains("workspace_branch_missing"), "{err}");
    assert!(std::path::Path::new(&ws2).exists());
    sh(&rig.root, &["branch", "-m", &renamed, &branch]);
    let _ = seq;

    // bytes at a recorded path without a git worktree registration
    // are unexpected work: the mismatch refuses, nothing is deleted
    let (_, _, ws3) = rig_provision(&rig, "impostor");
    sh(&rig.root, &["worktree", "remove", &ws3]);
    std::fs::create_dir_all(&ws3).unwrap();
    std::fs::write(std::path::Path::new(&ws3).join("precious"), "not mine\n").unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws3]);
    assert!(!ok, "unexpected bytes were deleted: {err}");
    assert!(err.contains("workspace_checkout_mismatch"), "{err}");
    assert!(std::path::Path::new(&ws3).join("precious").is_file());
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_stops_on_open_terminals_then_retry_completes() {
    let rig = lifecycle_rig("terminals");
    let (_, branch, ws) = rig_provision(&rig, "occupied");
    rig_open_terminal(&rig, "term_live_9", &ws);
    // exactly one close ask fails, then the fake host behaves
    std::fs::write(&rig.close_fail, b"refuse once\n").unwrap();

    // the first clean: the close names no finished state, and the door
    // stops before any checkout removal
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(!ok, "the occupied workspace cleaned anyway: {err}");
    assert!(err.contains("workspace_host_unverified"), "{err}");
    assert!(
        err.contains("the host refused the close of terminal term_live_9"),
        "{err}"
    );
    assert!(
        std::path::Path::new(&ws).exists(),
        "the checkout was removed"
    );
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        !kinds.contains(&"workspace_cleaned".to_string()),
        "{kinds:?}"
    );

    // the retry is the recovery path: the close lands, the list
    // verifies empty, and the completed fact records once
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the retry clean refuses: {out}{err}");
    assert!(!std::path::Path::new(&ws).exists());
    assert_eq!(
        rig_tip(&rig, &branch),
        sh(&rig.root, &["rev-parse", "HEAD"]).trim()
    );

    // the recorded-Cleaned idempotence verifies the host too: a
    // terminal the host still names refuses the no-op success
    rig_open_terminal(&rig, "term_after_2", &ws);
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(
        !ok,
        "the occupied cleaned workspace answered idempotent: {err}"
    );
    assert!(err.contains("workspace_terminals_open"), "{err}");
    // the terminal leaves, and the same door verifies its way to the
    // no-op success again
    std::fs::write(&rig.host_state, b"").unwrap();
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the verified idempotent re-run refuses: {out}{err}");
    assert!(out.contains("already cleaned"), "{out}{err}");
    assert!(out.contains("the host names no terminals"), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_refuses_an_unreadable_terminal_table() {
    let rig = lifecycle_rig("clean-garbage");
    let (_, _, ws) = rig_provision(&rig, "unanswered");
    rig_open_terminal(&rig, "term_garbage_5", &ws);
    // a row the adapter cannot fully read fails the whole answer: the
    // door refuses with the unverified shape and lands no fact
    use std::io::Write as _;
    let mut rows = std::fs::OpenOptions::new()
        .append(true)
        .open(&rig.host_state)
        .unwrap();
    writeln!(rows, r#"{{"address":"term_garbage_5"}}"#).unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(!ok, "the unreadable host cleaned anyway: {err}");
    assert!(err.contains("workspace_host_list_failed"), "{err}");
    assert!(
        std::path::Path::new(&ws).exists(),
        "the unreadable table stopped the door before any removal"
    );
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        !kinds.contains(&"workspace_cleaned".to_string()),
        "{kinds:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cleaned_stage_guards_launch() {
    let rig = lifecycle_rig("guards");
    let (_, _branch, ws) = rig_provision(&rig, "guarded-lifecycle");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the setup clean refuses: {err}");

    // a checkout that stands again at the recorded path (re-created
    // out-of-band) stays recorded Cleaned: a fresh launch into it
    // refuses
    let guard_branch = {
        let (_, body) = get_json(&rig.url, "/api/v1/records");
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "workspace_created" && r["payload"]["worktree"] == json!(ws))
            .map(|r| r["payload"]["branch"].as_str().unwrap().to_string())
            .expect("the guarded birth names its branch")
    };
    sh(&rig.root, &["worktree", "add", &ws, &guard_branch]);
    let (ok, _out, err) = rig_sac(&rig, &["launch", "--workspace", &ws]);
    assert!(!ok, "a cleaned workspace took a launch: {err}");
    assert!(err.contains("recorded Cleaned"), "{err}");
    // the same-terminal arm rides the same stage guard
    let (ok, _out, err) = rig_sac(&rig, &["launch", "--workspace", &ws, "--here"]);
    assert!(!ok, "a cleaned workspace took a here launch: {err}");
    assert!(err.contains("recorded Cleaned"), "{err}");

    // the resume path pins the same guard: a saved session whose
    // header cwd sits in the cleaned workspace refuses - the guard
    // fires before any host submission
    let saved_resume = rig.dir.join("native-cafe0001.jsonl");
    std::fs::write(
        &saved_resume,
        format!(
            "{{\"type\":\"session\",\"id\":\"cafe0001\",\"cwd\":\"{ws}\"}}\n{{\"kind\":\"filler\"}}\n"
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/native/attach",
        &json!({"native_id": "cafe0001", "session_file": saved_resume.display().to_string()}),
    );
    assert_eq!(status, 200, "{body}");
    let resume_num = json_of(&body)["num"].as_u64().unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["launch", "--resume", &format!("i-{resume_num}")]);
    assert!(!ok, "a cleaned workspace took a resume: {err}");
    assert!(err.contains("recorded Cleaned"), "{err}");
}

/// A recorded path addresses its own birth: a birth
/// whose stored string carries a symlink segment resolves through it -
/// stored identity preserved, and two stored paths aliasing one
/// directory refuse as ambiguous rather than picking.
#[tokio::test(flavor = "multi_thread")]
async fn the_recorded_path_itself_addresses_the_birth_through_an_alias() {
    let rig = lifecycle_rig("alias");
    // a birth whose recorded path carries a symlink segment:
    // hand-written, exactly what a log may hold
    let link = rig.dir.join("link");
    let real_home = rig.dir.join("real-wt");
    std::fs::create_dir_all(&real_home).unwrap();
    std::os::unix::fs::symlink(&real_home, &link).unwrap();
    let real = real_home.join("hist-line");
    let recorded = link.join("hist-line");
    sh(
        &rig.root,
        &[
            "worktree",
            "add",
            "-q",
            "-b",
            "saccade/t-9",
            &real.display().to_string(),
            "HEAD",
        ],
    );
    let birth = |at: u64, branch: &str, worktree: &str| {
        let mut conn = saccade::db::open(&rig.dir.join("saccade.db")).unwrap();
        saccade::db::record(
            &mut conn,
            &saccade::store::Context::system(),
            saccade::Command::CreateWorkspace {
                base: saccade::GitCommit::new("abc123".into()).unwrap(),
                branch: saccade::GitBranch::new(branch.into()).unwrap(),
                worktree: saccade::WorktreePath::new(worktree.into()).unwrap(),
            },
            at,
        )
        .unwrap();
    };
    birth(1, "saccade/t-9", recorded.display().to_string().as_str());

    // the recorded string itself addresses the birth: the ordered clean
    // contract runs over the real checkout it names
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", recorded.to_str().unwrap()]);
    assert!(
        ok,
        "the recorded string does not address its birth: {out}{err}"
    );
    assert!(!real.exists(), "the real checkout is gone");
    assert_eq!(
        rig_tip(&rig, "saccade/t-9"),
        sh(&rig.root, &["rev-parse", "HEAD"]).trim()
    );

    // the Cleaned stage blocks ordinary use at BOTH spellings once the
    // checkout stands again: the guard resolves the alias, it never
    // lets a known Cleaned workspace become usable through its twin
    sh(
        &rig.root,
        &[
            "worktree",
            "add",
            &real.display().to_string(),
            "saccade/t-9",
        ],
    );
    for shape in [recorded.to_str().unwrap(), real.to_str().unwrap()] {
        let (ok, _out, err) = rig_sac(&rig, &["launch", "--workspace", shape]);
        assert!(
            !ok,
            "the aliased Cleaned workspace took a launch at {shape}: {err}"
        );
        assert!(err.contains("recorded Cleaned"), "{err}");
    }
    // a second stored path over the same directory makes the alias
    // unsafe: both shapes refuse as ambiguous, neither is picked
    birth(3, "saccade/t-10", real.display().to_string().as_str());
    for shape in [recorded.to_str().unwrap(), real.to_str().unwrap()] {
        let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", shape]);
        assert!(!ok, "the aliased directory resolved to one birth: {err}");
        assert!(err.contains("workspace_ambiguous_path"), "{err}");
    }
}

#[test]
fn completions_emit_scripts_and_the_candidate_doors_answer_the_fold() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("completions-door");
    let system = saccade::Context::system();
    let mut conn = saccade::db::open(&path).unwrap();
    let now = 1;
    // the story: a task, two workspaces (one cleaned), one registered
    // native session
    let commands = [
        saccade::Command::CreateTask {
            name: saccade::Prose::new("implement floop".into()).unwrap(),
        },
        saccade::Command::CreateWorkspace {
            base: saccade::GitCommit::new("aaaa1111".into()).unwrap(),
            branch: saccade::GitBranch::new("workspace-floop".into()).unwrap(),
            worktree: saccade::WorktreePath::new("/tmp/sac-api-floop".into()).unwrap(),
        },
        saccade::Command::CreateWorkspace {
            base: saccade::GitCommit::new("bbbb2222".into()).unwrap(),
            branch: saccade::GitBranch::new("workspace-gloom".into()).unwrap(),
            worktree: saccade::WorktreePath::new("/tmp/sac-api-gloom".into()).unwrap(),
        },
        saccade::Command::CleanWorkspace {
            workspace: saccade::RecordId(2),
        },
        saccade::Command::RegisterIncarnation {
            native_id: saccade::NativeSessionId::new("sess-7777".into()).unwrap(),
            saved_session: Some(
                saccade::SessionLocator::new("/home/j/.pi/sessions/7777.json".into()).unwrap(),
            ),
            address: None,
            adopted: false,
            harness: saccade::Harness::Pi,
        },
    ];
    for command in commands {
        saccade::db::record(&mut conn, &system, command, now).expect("the fixture story records");
    }
    drop(conn);

    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
            .arg("--db")
            .arg(&path)
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // the script door: emitted from the binary, no destination resolves
    let (ok, script, err) = sac(&["completions", "fish"]);
    assert!(ok, "completions fish: {err}");
    assert!(script.contains("complete -c sac"), "the fish skeleton");
    assert!(
        script.contains("sac complete $argv"),
        "the PATH-name door call"
    );
    // the harness flag names both kinds in the value list
    assert!(
        script.contains("-l harness"),
        "the launch harness flag completes: {script}"
    );
    assert!(script.is_ascii(), "authored text is ASCII");
    assert!(!sac(&["completions", "bash"]).0, "unshipped shells refuse");

    // the candidate doors read the fold: tasks with title and state
    let (ok, out, err) = sac(&["complete", "tasks"]);
    assert!(ok, "complete tasks: {err}");
    assert_eq!(out.trim(), "t-0\timplement floop (open)");

    // workspaces: recorded paths with branch and stage, clean leading
    // Available
    let (ok, out, err) = sac(&["complete", "workspaces-clean"]);
    assert!(ok, "complete workspaces-clean: {err}");
    assert_eq!(
        out.trim(),
        "/tmp/sac-api-floop\tworkspace-floop, available\n\
         /tmp/sac-api-gloom\tworkspace-gloom, cleaned"
    );

    // incarnations: the birth actor beside what the record holds
    let (ok, out, err) = sac(&["complete", "incarnations"]);
    assert!(ok, "complete incarnations: {err}");
    assert_eq!(
        out.trim(),
        "i-0\tsaccade native session sess-7777, saved /home/j/.pi/sessions/7777.json"
    );

    // completion never records: the log is unchanged after every read
    let (ok, before, _) = sac(&["log", "--json"]);
    assert!(ok);
    let (ok, _, err) = sac(&["complete", "tasks"]);
    assert!(ok, "{err}");
    let (ok, after, _) = sac(&["log", "--json"]);
    assert!(ok);
    assert_eq!(before, after, "the candidate door records nothing");

    // silent failure: outside a repo and without a reachable
    // destination the door answers empty, exit 0, no stderr
    let outside = std::env::temp_dir().join(format!("sac-api-no-repo-{}", std::process::id()));
    std::fs::create_dir_all(&outside).unwrap();
    let out = scrubbed(std::process::Command::new(bin))
        .arg("complete")
        .arg("tasks")
        .current_dir(&outside)
        .env_remove("SACCADE_REPO")
        .env("HOME", &outside)
        .output()
        .expect("spawn sac outside a repo");
    assert!(out.status.success());
    assert_eq!(out.stdout, b"", "no candidates outside a repo");
    assert_eq!(out.stderr, b"", "the failure stays silent");

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(outside).unwrap();
}

#[test]
fn the_launch_door_births_the_incarnation_with_its_host_address() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-launch-b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // the launch doors resolve the repo's own database: the serve
    // subprocess must publish exactly that path's address hint
    let db = saccade::paths::state_dir_in(&suite_state(), &dir).join("saccade.db");
    let workspace = dir.join("ws");
    std::fs::create_dir_all(&workspace).unwrap();
    // the contract-shaped fake: hc run accepts the composed launch and
    // answers the pinned shape; its argv lands beside it for the door
    // composition's assertions
    let fake_dir = dir.join("fake");
    std::fs::create_dir_all(&fake_dir).unwrap();
    let fake = fake_dir.join("hc");
    std::fs::write(
        &fake,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-$1.log\"\nprintf '{\"address\":\"t1\",\"cwd\":\"x\",\"state\":\"live\",\"turn\":0}\\n'\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (mut serve, url) = spawn_serve(bin, &db, &[], &[]);
    let pi = dir.join("pi");
    std::fs::write(&pi, b"#!/bin/sh\n").unwrap();

    let out = scrubbed(std::process::Command::new(bin))
        .args(["launch", "--json", "--workspace"])
        .arg(&workspace)
        .env("SACCADE_SERVER", &url)
        .env("SACCADE_HERDR", &fake)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_REPO", &dir)
        .output()
        .expect("spawn sac");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "launch failed: {text}{err} (code {:?})",
        out.status.code()
    );
    let reply: Value = serde_json::from_str(&text).expect("the launch face is json");
    assert_eq!(reply["terminal"], "t1", "{text}");
    // no label ask rides the launch
    assert_eq!(reply["space"], Value::Null, "{text}");
    assert_eq!(reply["tab"], Value::Null, "{text}");
    assert_eq!(reply["place_refusal"], Value::Null, "{text}");
    assert!(
        !fake_dir.join("argv-label.log").exists(),
        "no label ask rides the launch"
    );

    // the composed argv carries the settled parentage law, the
    // notes-silence flag, the json ask, and the pre-allocated session
    let argv_log = std::fs::read_to_string(fake_dir.join("argv-run.log")).unwrap();
    let argv: Vec<&str> = argv_log.lines().skip(1).collect();
    assert_eq!(
        argv[..4],
        ["--bg", "--no-parent", "--no-notify", "--cwd"],
        "the composition lost its shape: {argv:?}"
    );
    let composition = argv.join(" ");
    assert!(composition.contains("--json -- "), "the ask rides --json");
    assert!(
        composition.contains("--session-id ") && composition.contains("--session-dir "),
        "the door pre-allocates the native session: {composition}"
    );
    assert!(
        composition.contains("direnv exec "),
        "the equip stays direnv"
    );

    // the fold carries the door-birthed incarnation: identity and
    // address exist at launch time, the locator waits for pi's save
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    let (id, ctx) = world
        .incarnations
        .iter()
        .find(|(_, ctx)| ctx.incarnation.address() == Some("t1"))
        .expect("the door birth carries the host address");
    let saccade::Incarnation::Native {
        native_id,
        saved_session,
        ..
    } = &ctx.incarnation;
    // the pre-allocated id is the one pi will create: it rode argv
    assert!(
        composition.contains(native_id.as_str()),
        "the argv pre-allocates the birth's native id"
    );
    assert!(saved_session.is_none(), "the locator binds at attach");

    // the door-owned session dir exists under the suite's state root,
    // where the helper pointed the child
    let sessions = saccade::paths::state_dir_in(&suite_state(), &dir).join("native-sessions");
    assert!(sessions.is_dir(), "the native-sessions dir stands");

    // the identity is a real registration: the mailbox read resolves it
    let num = world
        .incarnation_num(*id)
        .expect("the fold numbers the birth");
    let mailbox = scrubbed(std::process::Command::new(bin))
        .args(["messages", "--server", url.as_str(), "--repo"])
        .arg(dir.display().to_string())
        .arg(format!("i-{num}"))
        .output()
        .expect("spawn sac");
    assert!(
        mailbox.status.success(),
        "the door-birthed incarnation resolves: {}",
        String::from_utf8_lossy(&mailbox.stderr)
    );

    // a relaunch moves the association's address through the assign
    // door: the same incarnation, another terminal
    let mut assign_response = ureq::post(&format!("{url}/api/v1/incarnations/address"))
        .send_json(serde_json::json!({
            "incarnation": id.0.0,
            "address": "t9",
            "client": "test",
        }))
        .unwrap();
    let assign: Value = assign_response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .expect("the assign face is json");
    assert_eq!(assign["assigned"], true);
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    assert_eq!(
        world.incarnations[id]
            .incarnation
            .address()
            .map(str::to_string),
        Some("t9".to_string()),
        "the address moved with the relaunch"
    );

    std::fs::remove_dir_all(&dir).unwrap();
    let _ = serve.kill();
}

#[test]
fn the_naming_doors_record_every_name_and_label_the_terminal() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-naming-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = saccade::paths::state_dir_in(&suite_state(), &dir).join("saccade.db");
    let workspace = dir.join("ws");
    std::fs::create_dir_all(&workspace).unwrap();
    // the fake host: every verb's argv lands beside it; the first two
    // launches answer t1, a later one t2; a label refuses on demand
    let fake_dir = dir.join("fake");
    std::fs::create_dir_all(&fake_dir).unwrap();
    let fake = fake_dir.join("hc");
    std::fs::write(
        &fake,
        r#"#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s\n' "$@" >> "$here/argv-$1.log"
if [ "$1" = label ] && [ -n "${FAKE_LABEL_REFUSES:-}" ]; then
  printf '{"error":{"kind":"refused","message":"labels are off"}}\n' >&2
  exit 5
fi
if [ "$1" = run ]; then
  n=$(($(cat "$here/.runs" 2>/dev/null || echo 0) + 1))
  echo "$n" > "$here/.runs"
  addr=t1; [ "$n" -gt 2 ] && addr=t2
  printf '{"address":"%s","cwd":"x","state":"live","turn":0}\n' "$addr"
  exit 0
fi
printf '{}\n'
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (mut serve, url) = spawn_serve(bin, &db, &[], &[]);
    let pi = dir.join("pi");
    std::fs::write(&pi, b"#!/bin/sh\n").unwrap();
    let sac = |args: &[&str], envs: &[(&str, &str)]| {
        let out = scrubbed(std::process::Command::new(bin))
            .args(args)
            .env("SACCADE_SERVER", &url)
            .env("SACCADE_HERDR", &fake)
            .env("SACCADE_PI", &pi)
            .env("SACCADE_REPO", &dir)
            .envs(envs.iter().copied())
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let world = || {
        let conn = saccade::db::open_read(&db).unwrap();
        match saccade::db::load(&conn).unwrap().state {
            saccade::db::LoadState::Full(world) => world,
            _ => panic!("the log will not fold"),
        }
    };
    let name_at = |world: &saccade::World, address: &str| {
        world
            .incarnations
            .iter()
            .filter(|(_, ctx)| ctx.incarnation.address() == Some(address))
            .map(|(id, ctx)| (*id, ctx.name.as_ref().map(|n| n.as_str().to_string())))
            .next_back()
            .expect("a birth carries the address")
    };
    let labels = || hc_argv_all(&fake_dir, "label");
    let ws = workspace.to_str().unwrap();

    // a bad name refuses before the host is ever asked
    let (ok, _, err) = sac(&["launch", "--workspace", ws, "--name", " padded"], &[]);
    assert!(!ok, "a padded name refuses");
    assert!(err.contains("invalid_incarnation_name"), "{err}");
    assert!(
        !fake_dir.join("argv-run.log").exists(),
        "no launch was asked"
    );

    // a named fresh launch records the name with its birth and labels
    // the launched terminal
    let (ok, out, err) = sac(
        &["launch", "--json", "--workspace", ws, "--name", "theory"],
        &[],
    );
    assert!(ok, "{out}{err}");
    let reply: Value = serde_json::from_str(&out).expect("the launch face is json");
    assert_eq!(reply["name"], "theory", "{out}");
    assert_eq!(reply["name_refusal"], Value::Null, "{out}");
    assert_eq!(labels(), ["label", "t1", "--json", "name=theory"]);
    let (first, name) = name_at(&world(), "t1");
    assert_eq!(name.as_deref(), Some("theory"));
    let first_seq = first.0.0.to_string();
    let as_first = [
        ("SACCADE_ACTOR", "pi"),
        ("SACCADE_INCARNATION_SEQ", first_seq.as_str()),
    ];

    // the session reads its own name, then renames itself
    let (ok, out, err) = sac(&["identity", "--json"], &as_first);
    assert!(ok, "{out}{err}");
    assert_eq!(json_of(&out)["name"], "theory", "{out}");
    let (ok, out, err) = sac(&["name", "empirical"], &as_first);
    assert!(ok, "{out}{err}");
    assert!(out.contains("is named empirical"), "{out}");
    assert!(out.contains("label is set on t1"), "{out}");
    assert_eq!(&labels()[4..], ["label", "t1", "--json", "name=empirical"]);
    assert_eq!(name_at(&world(), "t1").1.as_deref(), Some("empirical"));

    // the agents page carries the name fact beside the number
    let (_, body) = get_json(&url, "/view/v1/agents");
    let page = json_of(&body);
    let group = page["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["kind"] == "group")
        .expect("the named session's group");
    assert!(
        group["content"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["label"] == "name" && f["value"] == "empirical"),
        "{body}"
    );

    // a refused label never fails the naming: the evidence rides the
    // face and the name is recorded
    let (ok, out, err) = sac(
        &["name", "--json", "theory"],
        &[as_first[0], as_first[1], ("FAKE_LABEL_REFUSES", "1")],
    );
    assert!(ok, "{out}{err}");
    let reply = json_of(&out);
    assert_eq!(reply["name"], "theory");
    assert_eq!(reply["label"]["address"], "t1");
    assert!(
        reply["label"]["refusal"]
            .as_str()
            .is_some_and(|r| r.contains("labels are off")),
        "{out}"
    );
    assert_eq!(name_at(&world(), "t1").1.as_deref(), Some("theory"));

    // a second session at the same terminal supersedes the first: names
    // need not be unique, and the superseded holder's old address is
    // never labeled again
    let (ok, out, err) = sac(&["launch", "--workspace", ws, "--name", "theory"], &[]);
    assert!(ok, "{out}{err}");
    assert!(out.contains("named theory"), "{out}");
    let (second, name) = name_at(&world(), "t1");
    assert_ne!(second, first);
    assert_eq!(name.as_deref(), Some("theory"));
    let before = labels().len();
    let (ok, out, err) = sac(&["name", "--clear"], &as_first);
    assert!(ok, "{out}{err}");
    assert!(out.contains("no live terminal address"), "{out}");
    assert_eq!(
        labels().len(),
        before,
        "the superseded address is not labeled"
    );
    assert_eq!(world().incarnations[&first].name, None);

    // an agent names only its own incarnation
    let second_num = format!("i-{}", world().incarnation_num(second).unwrap());
    let (ok, _, err) = sac(
        &["name", "--incarnation", &second_num, "taken over"],
        &as_first,
    );
    assert!(!ok, "an agent cannot name another session");
    assert!(err.contains("not_own_incarnation"), "{err}");
    // the human names any session, the clear included, through the wire
    let (status, body) = post_command(
        &url,
        &envelope(
            "jerry",
            "human",
            json!({"name_incarnation": {"incarnation": second.0.0, "name": null}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(world().incarnations[&second].name, None);
    let namings = world()
        .incarnations
        .values()
        .filter(|ctx| ctx.name.is_some())
        .count();
    assert_eq!(namings, 0);

    std::fs::remove_dir_all(&dir).unwrap();
    let _ = serve.kill();
}

/// Every argv line the fake hc logged for one verb, the op line kept.
fn hc_argv_all(bin_dir: &std::path::Path, verb: &str) -> Vec<String> {
    std::fs::read_to_string(bin_dir.join(format!("argv-{verb}.log")))
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn serve_startup_repairs_boards_and_refuses_a_missing_logged_commit() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-startup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let repo = dir.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    let db = saccade::paths::state_dir_in(&suite_state(), &repo).join("saccade.db");

    // a log whose fold carries a board commit the repository must hold
    let boards =
        saccade::boards::Boards::open(&saccade::paths::state_dir_in(&suite_state(), &repo))
            .expect("the boards repository opens");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(
        work.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent: []\n",
    )
    .unwrap();
    let landed = boards
        .commit("0", &work, None, "the recorded account")
        .expect("the objects land");
    let commit = saccade::GitCommit::new(landed.commit.clone()).unwrap();

    let mut conn = saccade::db::open(&db).unwrap();
    saccade::db::record(
        &mut conn,
        &saccade::store::Context {
            actor: saccade::types::actor::ActorName::new("pi".into()).unwrap(),
            tier: saccade::store::Tier::Agent,
            scope: None,
        },
        saccade::Command::CreateTask {
            name: saccade::Prose::new("implement foo".into()).unwrap(),
        },
        1,
    )
    .unwrap();
    saccade::db::record(
        &mut conn,
        &saccade::store::Context {
            actor: saccade::types::actor::ActorName::new("pi".into()).unwrap(),
            tier: saccade::store::Tier::Agent,
            scope: None,
        },
        saccade::Command::CommitBoard {
            task: saccade::TaskId(0),
            commit: commit.clone(),
            parent: None,
            incorporations: Vec::new(),
        },
        2,
    )
    .unwrap();
    drop(conn);
    let _ = boards.repair(&[("0".to_string(), commit.as_str().to_string())]);
    assert_eq!(
        landed.commit,
        commit.as_str(),
        "the test seeds the exact commit the log carries"
    );
    let _ = boards.repair(&[("0".to_string(), commit.as_str().to_string())]);

    let log = std::fs::File::create(dir.join("serve.log")).unwrap();
    let mut cmd = scrubbed(std::process::Command::new(bin));
    cmd.args([
        "--db",
        db.to_str().unwrap(),
        "--repo",
        repo.to_str().unwrap(),
        "serve",
        "--bind",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .stdout(log.try_clone().unwrap())
    .stderr(log);
    let mut serve = cmd.spawn().expect("serve spawns");
    std::thread::sleep(std::time::Duration::from_millis(1500));
    assert!(
        serve.try_wait().unwrap().is_none(),
        "serve stays up when the repository holds every logged commit"
    );
    let _ = serve.kill();
    let _ = serve.wait();

    // a logged commit missing from the repository refuses to serve
    std::fs::remove_dir_all(saccade::paths::state_dir_in(&suite_state(), &repo).join("boards.git"))
        .unwrap();
    let log2 = std::fs::File::create(dir.join("serve2.log")).unwrap();
    let mut cmd2 = scrubbed(std::process::Command::new(bin));
    cmd2.args([
        "--db",
        db.to_str().unwrap(),
        "--repo",
        repo.to_str().unwrap(),
        "serve",
        "--bind",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .stdout(log2.try_clone().unwrap())
    .stderr(log2);
    let mut serve2 = cmd2.spawn().expect("serve spawns again");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match serve2.try_wait().expect("poll serve") {
            Some(status) => {
                assert!(!status.success(), "the refusal is a failure exit");
                break;
            }
            None if std::time::Instant::now() >= deadline => {
                let _ = serve2.kill();
                panic!("serve stayed up past a missing logged commit");
            }
            None => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
    let stderr = std::fs::read_to_string(dir.join("serve2.log")).unwrap();
    assert!(
        stderr.contains("boards repair refused to serve"),
        "{stderr}"
    );
    let _ = serve2.wait();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_helper_run_writes_nothing_outside_its_scratch() {
    let bin = env!("CARGO_BIN_EXE_sac");
    // the host's real state root, read as evidence only: the listing
    // before and after the runs must not move
    let root = real_state_parent().join("saccade");
    let listing = || -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    };
    let before = listing();

    // a bare run from the helper's cwd passes the identity guard and
    // finds no tracker; the refusal lands nowhere
    let out = scrubbed(std::process::Command::new(bin))
        .env("SACCADE_ACTOR", "pin bot")
        .args(["create", "task", "nowhere to land"])
        .output()
        .expect("spawn sac");
    assert!(!out.status.success(), "a bare run finds no tracker");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("pass --db"), "the refusal teaches: {err}");

    // an explicit run through the helper writes into the suite state
    // root and nowhere else
    let scratch = scratch_db("helper-pin");
    let out = scrubbed(std::process::Command::new(bin))
        .env("SACCADE_ACTOR", "pin bot")
        .arg("--db")
        .arg(&scratch)
        .args(["create", "task", "scratch only"])
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // the run's own evidence sits inside the scratch: the db carries
    // the record and its telemetry rides beside it
    let out = scrubbed(std::process::Command::new(bin))
        .env("SACCADE_ACTOR", "pin bot")
        .arg("--db")
        .arg(&scratch)
        .args(["--json", "list"])
        .output()
        .expect("spawn sac");
    let out_text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out_text.contains("scratch only"), "{out_text}");
    assert!(
        scratch.with_file_name("telemetry.jsonl").exists(),
        "the run's telemetry rides beside the scratch db"
    );

    let after = listing();
    assert_eq!(before, after, "the real state root gained nothing");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_git_dir_inherited_by_the_server_never_reaches_the_board_store() {
    // the hook-leak shape: the server runs with a caller's GIT_DIR and
    // friends set. The decoy is a real bare repo - if the store's git
    // calls resolved the environment, the commit would be visible
    // there
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("decoy-gitdir");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let decoy = scratch.dir.join("decoy.git");
    saccade::boards::git_command()
        .args(["init", "--bare", "--quiet", decoy.to_str().unwrap()])
        .output()
        .expect("the decoy repo initializes");
    let config_before = std::fs::read_to_string(decoy.join("config")).unwrap();
    let objects_before = std::fs::read_dir(decoy.join("objects")).unwrap().count();
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[
            ("GIT_DIR", decoy.to_str().unwrap()),
            ("GIT_COMMON_DIR", decoy.to_str().unwrap()),
            (
                "GIT_OBJECT_DIRECTORY",
                decoy.join("objects").to_str().unwrap(),
            ),
            ("GIT_INDEX_FILE", decoy.join("index").to_str().unwrap()),
            ("GIT_WORK_TREE", project.to_str().unwrap()),
        ],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/command",
        &json!({
            "context": {"actor": "lane", "tier": "agent", "scope": null},
            "command": {"create_task": {"name": "the decoy probe"}},
            "at": null,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = post_json(
        &url,
        "/api/v1/boards/commit",
        &json!({
            "context": {"actor": "lane", "tier": "agent", "scope": null},
            "task": 0,
            "files": {"files": [{"path": "board.yaml",
                "content": "Z3JhbW1hcjogMQp0YXNrOiB0LTAKY29udGVudDoKICAtIHRleHQ6IHRoZSBwcm9iZSBib2FyZCB0aGUgc3RvcmUgbXVzdCBob2xkIGl0c2VsZgo="}]},
            "message": "the decoy probe commit",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["revision"], json!(1));

    // the decoy never heard about any of it
    let config_after = std::fs::read_to_string(decoy.join("config")).unwrap();
    assert_eq!(config_before, config_after, "the decoy's config changed");
    assert_eq!(
        objects_before,
        std::fs::read_dir(decoy.join("objects")).unwrap().count(),
        "objects landed in the decoy"
    );
    let refs = saccade::boards::git_command()
        .arg("--git-dir")
        .arg(&decoy)
        .args(["for-each-ref"])
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&refs.stdout).trim().is_empty(),
        "refs landed in the decoy"
    );

    // and the store itself holds the board on its own repo
    let (_, got) = get_json(&url, "/api/v1/boards/get?task=0");
    let got = json_of(&got);
    let files = got["files"]["files"].as_array().unwrap();
    assert!(!files.is_empty(), "{got}");

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_send_to_a_superseded_incarnation_refuses_and_discloses_the_supersession() {
    // t-10, the ratified law: an address belongs to one live
    // incarnation. A send to an incarnation whose slot a newer session
    // took refuses at the fold - the host would answer live and the
    // delivery would land in the successor's terminal - and discloses
    // the supersession's evidence: who took the address, at which
    // record
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("recycled-address");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    // the fake host: get answers live for every address; send accepts
    // and logs - a wrong delivery would show in this log
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // the sender: a registered incarnation declaring its seq in the
    // scope, the way a session's own send door does
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the gone incarnation: registered at the slot; its session is
    // gone and nothing records that
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "gone00000001",
            "address": "t65",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let gone_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the successor: a fresh session attaches at the same slot, the
    // way a new launch reuses the host's address numbering. The fold
    // derives the supersession from this assignment's record
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "newborn000001",
            "session_file": "/sessions/newborn000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let successor_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // a send addressed to the gone incarnation, by its own identity
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": gone_seq,
            "body": "never lands in the successor's slot",
            "client": "test",
        }),
    );
    // THE LAW: the reply refuses - the send faces' established 409
    // refusal shape - and discloses the supersession: who took the
    // address, at which record, and the message's own reference. The
    // host's log proves nothing was delivered into the slot
    assert_eq!(status, 409, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["error"]["code"], json!("recipient_gone"), "{body}");
    let detail = reply["error"]["detail"].as_str().unwrap();
    assert!(
        detail.contains(&format!("i-{successor_seq}")),
        "the refusal names the superseding incarnation: {body}"
    );
    assert!(
        detail.contains("t65"),
        "the refusal names the contested address: {body}"
    );
    assert!(
        detail.contains("record seq"),
        "the refusal discloses the supersession's record: {body}"
    );
    assert_eq!(
        reply["message"]["seq"],
        json!(3),
        "the message record stands as evidence though nothing delivered"
    );
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log"));
    assert!(
        log.is_err() || !log.unwrap().contains("never lands in the successor's slot"),
        "the host must never receive the message"
    );

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fresh_assignment_of_its_own_un_supersedes_the_incarnation() {
    // the way back: an incarnation superseded at its old slot, then
    // reassigned to a fresh terminal of its own, is live again -
    // delivery reaches it at the new address and never at the old
    // slot's holder
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("un-supersede");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // A: resident at t65
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "a0000000001",
            "session_file": "/sessions/a0000000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let a_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // B takes t65: A is superseded
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "b0000000001",
            "session_file": "/sessions/b0000000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");

    // the hole as it stood: A's delivery refuses
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": a_seq,
            "body": "refused while superseded",
            "client": "test",
        }),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        json!("recipient_gone"),
        "{body}"
    );

    // A relaunches onto a fresh terminal of its own: t70. The
    // reassignment is A's way back
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "a0000000001",
            "session_file": "/sessions/a0000000001.jsonl",
            "terminal": {"address": "t70", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(
        reply["records"].as_array().map(Vec::len),
        Some(1),
        "the reassignment lands: {body}"
    );

    // A's delivery submits, at t70 - and never at B's t65
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": a_seq,
            "body": "live again at the new address",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["submission"]["state"], json!("submitted"), "{body}");
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log")).unwrap();
    let args: Vec<&str> = log.lines().collect();
    assert!(
        args.contains(&"t70") && !args.contains(&"t65"),
        "the delivery reached t70 and never the old slot: {log}"
    );
    assert!(
        log.contains("live again at the new address"),
        "A's body rode the envelope: {log}"
    );

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_relaunch_that_takes_another_incarnations_address_supersedes_that_holder() {
    // the law on the reassignment path: an existing incarnation whose
    // relaunch lands on an address another incarnation holds
    // supersedes that holder - the older holder's delivery refuses,
    // the mover's own delivery keeps working
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("relaunch-takes-slot");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the mover: resident at t80
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "mover00000001",
            "session_file": "/sessions/mover00000001.jsonl",
            "terminal": {"address": "t80", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let mover_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the neighbor: resident at t65
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "neighbor000001",
            "session_file": "/sessions/neighbor000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let neighbor_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the mover relaunches and the host hands it t65 - the neighbor's
    // slot. The reassignment carries the law with it
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "mover00000001",
            "session_file": "/sessions/mover00000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");

    // the neighbor's delivery refuses, naming the mover
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": neighbor_seq,
            "body": "the neighbor is gone",
            "client": "test",
        }),
    );
    assert_eq!(status, 409, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["error"]["code"], json!("recipient_gone"), "{body}");
    let detail = reply["error"]["detail"].as_str().unwrap();
    assert!(
        detail.contains(&format!("i-{mover_seq}")) && detail.contains("t65"),
        "the refusal names the mover and the address: {body}"
    );

    // the mover's own delivery keeps working at the slot
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": mover_seq,
            "body": "the mover holds the slot now",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["submission"]["state"], json!("submitted"), "{body}");
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log")).unwrap();
    assert!(
        log.lines().any(|l| l == "t65"),
        "the mover's delivery submitted at t65: {log}"
    );

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_resume_at_its_own_address_never_supersedes_the_incarnation() {
    // the ratified law's exemption: the holder of an address who
    // re-presents the same pair is the same live incarnation - no
    // supersession, and delivery keeps working
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("resume-own-address");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the resident: registered and attached at its slot
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "resident000001",
            "session_file": "/sessions/resident000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let resident_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the resume: the same native session re-attaches at its own
    // address, the ordinary compaction-and-resume path
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "resident000001",
            "session_file": "/sessions/resident000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], json!(false), "{body}");
    assert_eq!(
        reply["incarnation"],
        json!(resident_seq),
        "the resume resolves to the same incarnation"
    );
    assert_eq!(
        reply["records"].as_array().map(Vec::len),
        Some(0),
        "an equal pair records no assignment at all: {body}"
    );

    // delivery still works, and the fold never learned a supersession
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": resident_seq,
            "body": "still the same live holder",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["submission"]["state"], json!("submitted"), "{body}");
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log")).unwrap();
    let args: Vec<&str> = log.lines().collect();
    assert!(
        args.contains(&"send") && args.contains(&"t65"),
        "the delivery submitted at the resident's own address: {log}"
    );
    assert!(
        log.contains("still the same live holder"),
        "the body rode the envelope: {log}"
    );

    let _ = server.kill();
    let _ = server.wait();
}
