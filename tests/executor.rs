//! The executor contract: the fake-Pi stub speaks the RPC subset
//! (prompt, steer, abort; response, agent_start, turn events,
//! agent_settled), and the real driver runs it end to end — fire,
//! ask, steer, abort. The real-pi smoke at the bottom validates the
//! pin against the same subset; the stub is the contract, pi is an
//! implementation of it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use saccade::api::AppState;
use saccade::db::{self, LoadState};
use saccade::objects::comment::{CommentId, CommentState, SteerDelivery};
use saccade::runner::{self, Executor};
use saccade::supervisor::{self, RunnerConfig, SessionDriver};
use saccade::types::actor::ActorName;
use saccade::{Command, CommentKind, Context, Prose, Target, TaskId, Tier, World};

fn sh(dir: &Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A judged request with no client binary to name and no restatable
/// body: these tests drive the seam directly, not a door.
fn bare_request() -> saccade::attempts::AsReceived {
    saccade::attempts::AsReceived {
        client: None,
        raw: String::new(),
    }
}

fn human() -> Context {
    Context {
        actor: ActorName::new("human person".into()).unwrap(),
        tier: Tier::Human,
    }
}

/// A fresh repo with one commit, a tracker with one task and one
/// demand on it.
fn scaffold(tag: &str) -> (PathBuf, PathBuf, CommentId) {
    let dir = std::env::temp_dir().join(format!("sac-exec-{tag}-{}", std::process::id()));
    let repo = dir.join("repo");
    let state = dir.join("state");
    std::fs::create_dir_all(&repo).unwrap();
    sh(&repo, &["init", "--initial-branch=main"]);
    std::fs::write(repo.join("readme"), "the project\n").unwrap();
    sh(&repo, &["add", "."]);
    sh(&repo, &["commit", "-m", "base"]);

    let db_path = state.join("saccade.db");
    std::fs::create_dir_all(&state).unwrap();
    let mut conn = db::open(&db_path).unwrap();
    db::record(
        &mut conn,
        &human(),
        Command::CreateTask {
            name: Prose::new("answer the demand over rpc".into()).unwrap(),
            parent_id: None,
        },
        1,
    )
    .unwrap();
    db::record(
        &mut conn,
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(0)),
            body: Prose::new("write the receipt".into()).unwrap(),
            kind: CommentKind::Demand,
        },
        2,
    )
    .unwrap();
    (repo, db_path, CommentId(saccade::RecordId(1)))
}

fn world_of(db_path: &Path) -> World {
    let conn = db::open_read(db_path).unwrap();
    let loadout = db::load(&conn).unwrap();
    let LoadState::Full(world) = loadout.state else {
        panic!("expected a full load")
    };
    world
}

/// Poll until the predicate holds over a fresh world, or panic naming
/// what never came.
fn await_world(db_path: &Path, what: &str, secs: u64, p: impl Fn(&World) -> bool) -> World {
    for _ in 0..(secs * 10) {
        let world = world_of(db_path);
        if p(&world) {
            return world;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("timed out waiting for {what}");
}

/// The fake-Pi stub: the protocol contract in one python3 file. It
/// reads command frames from stdin, emits the event subset on stdout,
/// and does the session's work the way a real executor would — through
/// the sac CLI's own doors, under the actor the runner set. STUB_MODE
/// picks the story: reply, ask, steer, or sleep.
const STUB: &str = r##"#!__PYTHON3__
"""fake-pi: the RPC subset as a contract. Commands in on stdin (JSONL,
LF-framed), events out on stdout; unknown lines are ignored. The mode
and log path are baked in per instance."""
import json, os, subprocess, sys, time, urllib.request

SAC = os.environ.get("SACCADE_SAC", "sac")
SERVER = os.environ.get("SACCADE_SERVER", "http://127.0.0.1:8811")
ACTOR = os.environ.get("SACCADE_ACTOR", "pi-stub")
MODE = "__MODE__"
LOG = "__LOG__"
DB = "__DB__"
if DB:
    os.environ["SACCADE_DB"] = DB


def emit(evt):
    sys.stdout.write(json.dumps(evt) + "\n")
    sys.stdout.flush()


def note(text):
    if LOG:
        with open(LOG, "a") as f:
            f.write(text + "\n")


# the spawn contract: refuse any spawn that leaks the operator's pi
_leaks = [
    f
    for f in ("--no-extensions", "--no-skills", "--no-prompt-templates")
    if f not in sys.argv
]
if not os.environ.get("PI_CODING_AGENT_DIR"):
    _leaks.append("PI_CODING_AGENT_DIR")
if _leaks:
    emit(
        {
            "type": "response",
            "command": "prompt",
            "success": False,
            "error": f"ambient config leak: missing {_leaks}",
        }
    )
    sys.exit(1)

def run(*args, **kw):
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        note(f"sac-failed {args}: rc={r.returncode} err={r.stderr.strip()[:200]}")
    return r


def demand_of(prompt):
    # the pointer prompt names the demand it serves
    import re
    m = re.search(r"demand c-(\d+)", prompt)
    return int(m.group(1)) if m else None


def post_comment(body, kind):
    cmd = {"comment": {"target": {"task": 0}, "body": body, "kind": kind}}
    req = urllib.request.Request(
        f"{SERVER}/api/v1/command",
        data=json.dumps({
            "context": {"actor": ACTOR, "tier": "agent"},
            "at": None,
            "command": cmd,
        }).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        reply = json.load(r)
    return reply["records"][-1]["seq"]


def reply_to_demand(demand, body):
    run(SAC, "comment", f"#{demand}", body)


def handle(line):
    try:
        cmd = json.loads(line)
    except ValueError:
        return
    note(f"command {line.strip()}")
    kind = cmd.get("type")
    if kind == "prompt":
        emit({"type": "response", "command": "prompt", "success": True})
        emit({"type": "agent_start"})
        emit({"type": "turn_start"})
        prompt = cmd.get("message", "")
        if MODE == "reply":
            demand = demand_of(prompt)
            reply_to_demand(demand, "the stub session answered")
            emit({"type": "turn_end"})
            emit({"type": "agent_settled"})
        elif MODE == "ask":
            demand = demand_of(prompt)
            seq = post_comment("which receipt do you want, plain or itemized?", "ask")
            note(f"ask c-{seq}")
            got = run(SAC, "wait", f"c-{seq}")
            note(f"wait: {got.stdout.strip()}")
            reply_to_demand(demand, "asked and answered: " + got.stdout.strip())
            emit({"type": "turn_end"})
            emit({"type": "agent_settled"})
        elif MODE == "steer":
            # hold the turn open: the steer command finishes it
            pass
        elif MODE == "sleep":
            # the work runs in the background; the read loop stays live
            # so the abort reaches this session while it works
            import threading
            threading.Thread(target=lambda: time.sleep(120), daemon=True).start()
    elif kind == "steer":
        emit({"type": "response", "command": "steer", "success": True})
        if MODE == "steer":
            reply_to_demand(demand_of_cache, "steered: " + cmd.get("message", ""))
            emit({"type": "turn_end"})
            emit({"type": "agent_settled"})
    elif kind == "abort":
        emit({"type": "response", "command": "abort", "success": True})
        note("aborted")
        emit({"type": "agent_settled"})
        sys.exit(0)


demand_of_cache = None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if demand_of_cache is None:
        try:
            demand_of_cache = demand_of(json.loads(line).get("message", ""))
        except ValueError:
            pass
    handle(line)
"##;

/// The task's thread lines: bodies without reaching into the fold.
fn thread_bodies(world: &World) -> Vec<String> {
    saccade::views::comment_thread(&world.comments, &world.tasks[0])
        .into_iter()
        .map(|l| l.body)
        .collect()
}

/// The interpreter for the stub's shebang: PATH-resolved, because the
/// build sandbox chroot carries no /usr/bin/env.
fn python3() -> PathBuf {
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("python3");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from("/usr/bin/env python3")
}

/// The stub on disk, executable, with its story and log baked in.
fn write_stub(dir: &Path, mode: &str, db_path: &Path) -> (PathBuf, PathBuf) {
    let log = dir.join(format!("stub-{mode}.log"));
    let stub = dir.join(format!("fake-pi-{mode}"));
    std::fs::write(
        &stub,
        STUB.replace("__PYTHON3__", &python3().to_string_lossy())
            .replace("__MODE__", mode)
            .replace("__LOG__", &log.to_string_lossy())
            .replace("__DB__", &db_path.to_string_lossy()),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    (stub, log)
}

/// Bind the console router over a runner-backed state; the stub's HTTP
/// writes need the real command door.
async fn spawn_serving(state: AppState) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = saccade::serve::router(state);
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// The serving state with the real RPC driver pointed at the stub, and
/// the command door bound over it: the stub's own writes (CLI and HTTP)
/// go through the real server.
async fn stub_app(repo: &Path, db_path: &Path, mode: &str) -> AppState {
    let (stub, _log) = write_stub(&repo.parent().unwrap().join("state"), mode, db_path);
    let server = Arc::new(std::sync::Mutex::new("http://127.0.0.1:1".to_string()));
    let cell = server.clone();
    let driver: SessionDriver = Arc::new(move |run, prompt, runs| {
        let executor = Executor {
            pi: stub.clone(),
            server: cell.lock().unwrap().clone(),
            sac: PathBuf::from(env!("CARGO_BIN_EXE_sac")),
        };
        runner::execute_session(run, prompt, runs, &executor)
    });
    let state = AppState::with_runner(
        db_path,
        RunnerConfig {
            repo_root: repo.to_path_buf(),
            actor: ActorName::new("pi".into()).unwrap(),
            driver,
        },
    )
    .unwrap();
    let base = spawn_serving(state.clone()).await;
    *server.lock().unwrap() = base;
    state
}

/// One full course through the real driver and the stub: the demand
/// fires, the stub speaks the subset, replies through the CLI door,
/// settles, and the run closes with the checkpoint.
#[tokio::test(flavor = "multi_thread")]
async fn the_stub_contract_answers_a_demand_over_rpc() {
    let (repo, db_path, demand) = scaffold("reply");
    let log = repo.parent().unwrap().join("state").join("stub-reply.log");
    let state = stub_app(&repo, &db_path, "reply").await;

    supervisor::sweep(&state);

    let seen = saccade::runner::wait(&db_path, demand, Some(30)).unwrap();
    assert!(seen.contains("settled"), "{seen}");
    let world = world_of(&db_path);
    assert_eq!(world.tasks[0].active_incarnation, None);
    // the stub's reply carries the run's derived attribution
    match &world.comments[&demand].state {
        CommentState::Demand { response, .. } => {
            let reply = match response {
                saccade::objects::comment::ResponseState::Responded { reply } => *reply,
                _ => panic!("the stub's reply landed"),
            };
            assert_eq!(world.comments[&reply].actor.as_str(), "pi/t-0-1");
            assert!(thread_bodies(&world).contains(&"the stub session answered".to_string()));
        }
        other => panic!("demand answered: {other:?}"),
    }
    // the stub really spoke the subset: the prompt went in as a command
    let said = std::fs::read_to_string(&log).unwrap();
    assert!(said.contains("\"type\":\"prompt\""), "{said}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// The ask door through the stub: the run posts an ask through the
/// comment door, blocks on sac wait, and the answer resumes it — the
/// reply carries the answer the human wrote.
#[tokio::test(flavor = "multi_thread")]
async fn an_ask_round_trips_through_the_stub() {
    let (repo, db_path, demand) = scaffold("ask");
    let log = repo.parent().unwrap().join("state").join("stub-ask.log");
    let state = stub_app(&repo, &db_path, "ask").await;

    supervisor::sweep(&state);

    // the ask lands on the thread, authored by the run's actor
    let world = await_world(&db_path, "the ask", 30, |w| {
        w.comments
            .values()
            .any(|c| matches!(c.state, CommentState::Ask { .. }))
    });
    let (ask, asker) = world
        .comments
        .iter()
        .find(|(_, c)| matches!(c.state, CommentState::Ask { .. }))
        .map(|(id, c)| (*id, c.actor.clone()))
        .unwrap();
    assert_eq!(asker.as_str(), "pi/t-0-1");
    assert!(
        thread_bodies(&world)
            .contains(&"which receipt do you want, plain or itemized?".to_string())
    );

    // the human answers; the run resumes on it
    state
        .execute(
            &human(),
            Command::Comment {
                target: Target::Comment(ask),
                body: Prose::new("itemized, with the counts".into()).unwrap(),
                kind: CommentKind::Note,
            },
            None,
            bare_request(),
        )
        .unwrap();

    let seen = saccade::runner::wait(&db_path, demand, Some(60)).unwrap();
    assert!(seen.contains("settled"), "{seen}");
    let world = await_world(&db_path, "the resumed reply", 30, |w| {
        thread_bodies(w)
            .iter()
            .any(|b| b.starts_with("asked and answered"))
    });
    // the answer rode the wait release into the stub's reply
    let reply_body = thread_bodies(&world)
        .into_iter()
        .find(|b| b.starts_with("asked and answered"))
        .unwrap();
    assert!(
        reply_body.contains("itemized, with the counts"),
        "{reply_body}"
    );
    assert!(reply_body.contains("c-"), "{reply_body}");
    let said = std::fs::read_to_string(&log).unwrap();
    assert!(said.contains("wait: c-"), "{said}");
    // the ask holds its answer on the thread
    assert!(matches!(
        world.comments[&ask].state,
        CommentState::Ask {
            response: saccade::objects::comment::ResponseState::Responded { .. },
        }
    ));
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// A steer standing on the thread reaches the run the moment it lives,
/// is consumed exactly once, and its run answers with it.
#[tokio::test(flavor = "multi_thread")]
async fn a_standing_steer_reaches_the_live_session_once() {
    let (repo, db_path, demand) = scaffold("steer");
    let log = repo.parent().unwrap().join("state").join("stub-steer.log");

    // the steer lands while no run lives: it stands as intent
    let mut conn = db::open(&db_path).unwrap();
    db::record(
        &mut conn,
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(0)),
            body: Prose::new("also cover the offline path".into()).unwrap(),
            kind: CommentKind::Steer,
        },
        3,
    )
    .unwrap();
    let steer = CommentId(saccade::RecordId(2));
    assert_eq!(
        world_of(&db_path).comments[&steer].state,
        CommentState::Steer {
            delivery: SteerDelivery::Standing
        }
    );

    let state = stub_app(&repo, &db_path, "steer").await;

    // the run fires; the watcher delivers the standing steer at spawn
    supervisor::sweep(&state);

    await_world(&db_path, "the steer's consumption", 30, |w| {
        matches!(
            w.comments.get(&steer).map(|c| &c.state),
            Some(CommentState::Steer {
                delivery: SteerDelivery::Forwarded
            })
        )
    });

    let seen = saccade::runner::wait(&db_path, demand, Some(30)).unwrap();
    assert!(seen.contains("settled"), "{seen}");
    let world = world_of(&db_path);
    // the run's reply names the steer it consumed
    let reply = thread_bodies(&world)
        .into_iter()
        .find(|b| b.starts_with("steered: "))
        .unwrap();
    assert!(reply.contains("also cover the offline path"), "{reply}");

    // consumed once, never re-fired: exactly one steer command reached
    // the session across every watcher tick
    let said = std::fs::read_to_string(&log).unwrap();
    let deliveries = said
        .lines()
        .filter(|l| l.contains("\"type\":\"steer\""))
        .count();
    assert_eq!(deliveries, 1, "{said}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// A mid-run steer lands through the same door while the turn is open.
#[tokio::test(flavor = "multi_thread")]
async fn a_midrun_steer_reaches_the_open_turn() {
    let (repo, db_path, demand) = scaffold("steer-mid");
    let log = repo.parent().unwrap().join("state").join("stub-steer.log");
    let state = stub_app(&repo, &db_path, "steer").await;

    // fire the run first; its turn holds open for the steer
    supervisor::sweep(&state);
    await_world(&db_path, "the run to bind", 30, |w| {
        w.tasks[0].active_incarnation.is_some()
    });

    // the steer lands while the run lives
    state
        .execute(
            &human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("stop early and report".into()).unwrap(),
                kind: CommentKind::Steer,
            },
            None,
            bare_request(),
        )
        .unwrap();

    let seen = saccade::runner::wait(&db_path, demand, Some(30)).unwrap();
    assert!(seen.contains("settled"), "{seen}");
    let world = world_of(&db_path);
    let consumed = world.comments.values().any(|c| {
        matches!(c.state, CommentState::Steer { .. })
            && thread_bodies(&world).contains(&"stop early and report".to_string())
    });
    assert!(consumed, "the steer stands consumed");
    let said = std::fs::read_to_string(&log).unwrap();
    assert!(said.contains("stop early and report"), "{said}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// Cancel is the protocol act: the abort command reaches the session,
/// which settles and ends; the kill stays in reserve.
#[tokio::test(flavor = "multi_thread")]
async fn a_cancel_aborts_the_session_through_the_protocol() {
    let (repo, db_path, _demand) = scaffold("abort");
    let log = repo.parent().unwrap().join("state").join("stub-sleep.log");
    let state = stub_app(&repo, &db_path, "sleep").await;

    supervisor::sweep(&state);
    let world = await_world(&db_path, "the run to bind", 30, |w| {
        w.tasks[0].active_incarnation.is_some()
    });
    let incarnation = world.tasks[0].active_incarnation.unwrap();

    // the cancel is an ordinary write; the sweep it triggers aborts
    // through the connection
    state
        .execute(
            &human(),
            Command::CancelIncarnation { id: incarnation },
            None,
            bare_request(),
        )
        .unwrap();
    supervisor::sweep(&state);

    let world = await_world(&db_path, "the cancel to end the run", 30, |w| {
        w.tasks[0].active_incarnation.is_none()
    });
    assert_eq!(
        world.incarnations[&incarnation].state,
        saccade::objects::incarnation::IncarnationState::Cancelled
    );
    // the abort reached the session as a protocol act, not a signal:
    // the stub's note arrives asynchronously, so the read polls
    let mut said = String::new();
    for _ in 0..100 {
        if let Ok(text) = std::fs::read_to_string(&log)
            && text.contains("aborted")
        {
            said = text;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(
        said.contains("aborted"),
        "the stub never saw the abort command: {said}"
    );
    assert!(said.contains("\"type\":\"abort\""), "{said}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// The real-pi smoke: the pinned binary through the subset the stub
/// defines — prompt to agent_start..agent_settled, steer, abort — with
/// the ask extension loaded. Runs keyless: a failed model call still
/// completes the event arc.
#[test]
fn the_real_pi_smoke_validates_the_pin() {
    let Ok(pi) = runner::resolve_pi() else {
        // a bare clone has no pin: red must mean contract breakage,
        // never environment. Wrapped environments (the flake's baked
        // SACCADE_PI_PATH, the devShell's SACCADE_PI) run the full smoke
        eprintln!(
            "smoke: skipped, no pinned executor (SACCADE_PI_PATH not baked \
             and SACCADE_PI unset); build through the flake or set SACCADE_PI \
             to validate the pin"
        );
        return;
    };
    // the pin is validated under the same composed config home the
    // runner spawns in, auth symlink included; the ask door loads
    // from where compose materialized it
    let agent_dir = std::env::temp_dir().join(format!("sac-smoke-agent-{}", std::process::id()));
    runner::compose_agent_dir(&agent_dir)
        .unwrap_or_else(|e| panic!("composing the smoke agent dir: {e:?}"));
    let extension = runner::ask_extension_at(&agent_dir);
    let mut child = std::process::Command::new(&pi)
        .args([
            "--mode",
            "rpc",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--offline",
            "-e",
        ])
        .arg(&extension)
        .env("PI_CODING_AGENT_DIR", &agent_dir)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawning the pinned pi ({}): {e}", pi.display()));

    use std::io::Write;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    // a reader thread turns the stream into a channel, so a stalled arc
    // (a network-less sandbox retries the model call forever) can be
    // forced to settle through the abort door — the subset stays
    // validated, bounded everywhere
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel::<std::io::Result<String>>();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut send = |json: &str| {
        // the session may already have departed: a failed write is the
        // EOF story, not a panic
        stdin.write_all(json.as_bytes()).is_ok()
            && stdin.write_all(b"\n").is_ok()
            && stdin.flush().is_ok()
    };
    send("{\"type\":\"prompt\",\"message\":\"Reply with exactly: ready\"}");

    let mut saw_start = false;
    let mut steer_sent = false;
    let mut prompt_refused: Option<String> = None;
    let mut steer_ok = false;
    let mut abort_ok = false;
    let mut aborted = false;
    let mut settled = false;
    let mut extension_error = false;
    let mut deadline = Instant::now() + Duration::from_secs(60);
    while !settled {
        let line = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Ok(line)) => line,
            Ok(Err(_)) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("the pinned pi closed its stream before settling")
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if aborted {
                    panic!("the pinned pi did not settle even aborted");
                }
                // the model call will not complete here: settle by abort
                send("{\"type\":\"abort\"}");
                aborted = true;
                deadline = Instant::now() + Duration::from_secs(30);
                continue;
            }
        };
        match saccade::rpc::ServerEvent::parse(line.trim_end()) {
            saccade::rpc::ServerEvent::AgentStart => {
                saw_start = true;
                if !steer_sent {
                    // a mid-run steer queues behind the turn: its
                    // response is part of the validated subset
                    steer_sent = send("{\"type\":\"steer\",\"message\":\"smoke\"}");
                }
            }
            saccade::rpc::ServerEvent::AgentSettled => settled = true,
            saccade::rpc::ServerEvent::Response {
                command,
                success,
                error,
            } => {
                if command == "prompt" && !success {
                    // a keyless sandbox refuses the prompt before the
                    // arc starts: the command half of the subset still
                    // validated — the framing carried the refusal
                    prompt_refused = error;
                    break;
                }
                if command == "steer" {
                    steer_ok = success;
                }
                if command == "abort" {
                    abort_ok = success;
                }
            }
            saccade::rpc::ServerEvent::Other(t) if t == "extension_error" => {
                extension_error = true;
            }
            _ => {}
        }
    }
    assert!(!extension_error, "the ask extension failed to load");
    if let Some(reason) = prompt_refused {
        // no provider answered: the event arc needs one; the pin's
        // command half (framing, refusal, parse) held
        eprintln!("smoke: prompt refused keyless, arc skipped: {reason}");
        drop(stdin);
        let status = child.wait().unwrap();
        assert!(status.success(), "the pinned pi exited {status}");
        return;
    }
    assert!(saw_start, "agent_start preceded the settle");
    assert!(steer_ok, "the steer response never came");

    if !aborted {
        // a naturally settled run answers the abort door too; its
        // response trails the settle, so drain until it departs
        send("{\"type\":\"abort\"}");
        while let Ok(Ok(line)) = rx.recv_timeout(Duration::from_secs(5))
            && let saccade::rpc::ServerEvent::Response {
                ref command,
                success,
                ..
            } = saccade::rpc::ServerEvent::parse(line.trim_end())
            && command == "abort"
        {
            abort_ok = success;
        }
        assert!(abort_ok, "the abort response never came");
    }

    // EOF ends the session
    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success(), "the pinned pi exited {status}");
}
