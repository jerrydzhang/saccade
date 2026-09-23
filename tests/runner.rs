//! The runner end to end: a real git repository, a real tracker db, the
//! machinery verbs driven through their orchestration. The session driver
//! itself is exercised by the live dogfood run, not here — these tests
//! sit at the seam on either side of it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use saccade::api::AppState;
use saccade::db::{self, LoadState};
use saccade::objects::comment::AgentAttemptState;
use saccade::objects::comment::CommentState;
use saccade::objects::incarnation::IncarnationState;
use saccade::runner::{
    PreparedRun, RunnerFail, close, compose_agent_dir, pointer_prompt, prepare, wait,
};
use saccade::supervisor::{self, LiveRuns, RunHandle, RunnerConfig, SessionDriver};
use saccade::types::actor::ActorName;
use saccade::types::pointers::SessionPointer;
use saccade::{Command, CommentId, CommentKind, Context, Prose, Target, TaskId, Tier, World};

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

fn agent() -> Context {
    Context {
        actor: ActorName::new("pi".into()).unwrap(),
        tier: Tier::Agent,
    }
}

/// A fresh repo with one commit, a tracker with one task and one
/// agent-addressed demand on it.
fn scaffold(tag: &str) -> (PathBuf, PathBuf, CommentId) {
    let dir = std::env::temp_dir().join(format!("sac-runner-{tag}-{}", std::process::id()));
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
            name: Prose::new("answer the demand in a worktree".into()).unwrap(),
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

#[test]
fn a_demand_runs_its_course_through_worktree_and_checkpoint() {
    let (repo, db_path, demand) = scaffold("course");

    let prepared = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();

    // the git side really landed: branch and worktree exist
    let worktree = saccade::paths::worktree_at(&repo, 0);
    assert!(worktree.join(".git").exists());
    let branches = sh(&repo, &["branch", "--list", "saccade/t-0"]);
    assert!(branches.contains("saccade/t-0"));
    // the run's name is derived: executor/task-incarnation
    assert_eq!(prepared.actor.as_str(), "pi/t-0-1");

    // the run is accepted and holding the demand's attempt
    let world = world_of(&db_path);
    assert_eq!(
        world.tasks[0].active_incarnation,
        Some(prepared.incarnation)
    );
    assert_eq!(
        world.incarnations[&prepared.incarnation].state,
        IncarnationState::PromptAccepted
    );

    // the session's reply: the agent answers from the worktree
    let mut conn = db::open(&db_path).unwrap();
    db::record(
        &mut conn,
        &agent(),
        Command::Comment {
            target: Target::Comment(demand),
            body: Prose::new("receipt written, tests green".into()).unwrap(),
            kind: CommentKind::Note,
        },
        3,
    )
    .unwrap();

    // the reply alone releases nothing while the run still works
    assert!(matches!(
        wait(&db_path, demand, Some(0)),
        Err(RunnerFail::Usage(_))
    ));

    // a commit in the worktree becomes the checkpoint
    std::fs::write(worktree.join("receipt"), "done\n").unwrap();
    sh(&worktree, &["add", "."]);
    sh(&worktree, &["commit", "-m", "receipt"]);

    let note = close(&mut db::open(&db_path).unwrap(), TaskId(0)).unwrap();
    assert!(note.contains("settled i-"), "{note}");

    let world = world_of(&db_path);
    assert_eq!(world.tasks[0].active_incarnation, None);
    assert_eq!(
        world.incarnations[&prepared.incarnation].state,
        IncarnationState::Settled
    );
    // the produced pointer names exactly the demand's reply
    let reply = match &world.comments[&demand].state {
        CommentState::Demand { response, .. } => match response {
            saccade::objects::comment::ResponseState::Responded { reply } => *reply,
            _ => panic!("the reply landed"),
        },
        other => panic!("demand answered: {other:?}"),
    };
    assert_eq!(
        world.incarnations[&prepared.incarnation].produced,
        vec![reply.0]
    );
    match &world.comments[&demand].state {
        CommentState::Demand { attempt, .. } => {
            assert!(matches!(
                attempt,
                saccade::objects::comment::AgentAttemptState::Spent
            ));
        }
        other => panic!("demand spent: {other:?}"),
    }
    let workspace = world.tasks[0].workspace.as_ref().unwrap();
    let head = sh(&worktree, &["rev-parse", "HEAD"]);
    assert_eq!(workspace.checkpoint.as_str(), head);

    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn prepare_refuses_what_the_fold_would_refuse() {
    let (repo, db_path, _demand) = scaffold("refuse");

    // a human-addressed comment is not a demand
    let mut conn = db::open(&db_path).unwrap();
    db::record(
        &mut conn,
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(0)),
            body: Prose::new("just talking".into()).unwrap(),
            kind: CommentKind::Note,
        },
        3,
    )
    .unwrap();
    let refused = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        CommentId(saccade::RecordId(2)),
        ActorName::new("pi".into()).unwrap(),
    );
    assert!(matches!(refused, Err(RunnerFail::Usage(_))));
    // and no worktree was orphaned by the refusal
    assert!(!saccade::paths::worktree_at(&repo, 0).exists());

    // a second run on the same task has no free slot
    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        CommentId(saccade::RecordId(1)),
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    let refused = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        CommentId(saccade::RecordId(1)),
        ActorName::new("pi".into()).unwrap(),
    );
    assert!(matches!(refused, Err(RunnerFail::Usage(_))));

    // closing a task with no run refuses
    let mut conn = db::open(&db_path).unwrap();
    db::record(
        &mut conn,
        &human(),
        Command::CreateTask {
            name: Prose::new("never ran".into()).unwrap(),
            parent_id: None,
        },
        4,
    )
    .unwrap();
    assert!(matches!(
        close(&mut db::open(&db_path).unwrap(), TaskId(1)),
        Err(RunnerFail::Usage(_))
    ));

    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn the_pointer_prompt_names_the_work_and_the_reply_door() {
    let (repo, db_path, demand) = scaffold("prompt");
    let prepared = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    let prompt = pointer_prompt(&prepared, "/bin/sac");
    assert!(prompt.contains("c-1"), "{prompt}");
    assert!(prompt.contains("t-0"), "{prompt}");
    assert!(prompt.contains("/bin/sac comment '#1'"), "{prompt}");
    // the reply door names the run's derived attribution
    assert!(prompt.contains("'pi/t-0-1'"), "{prompt}");
    std::fs::remove_dir_all(db_path.parent().unwrap()).unwrap();
}

#[test]
fn wait_reports_an_unanswered_demand_at_its_deadline() {
    let (_repo, db_path, demand) = scaffold("wait");
    let refused = wait(&db_path, demand, Some(0));
    assert!(matches!(refused, Err(RunnerFail::Usage(_))));
    std::fs::remove_dir_all(db_path.parent().unwrap()).unwrap();
}

#[test]
fn wait_releases_on_settlement_naming_the_receipt() {
    let (repo, db_path, demand) = scaffold("wait-settle");
    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    let worker = Context {
        actor: ActorName::new("pi/t-0-1".into()).unwrap(),
        tier: Tier::Agent,
    };
    let mut conn = db::open(&db_path).unwrap();
    db::record(&mut conn, &worker, Command::ClaimTask { id: TaskId(0) }, 3).unwrap();
    db::record(
        &mut conn,
        &worker,
        Command::CompleteTask {
            id: TaskId(0),
            receipt: Prose::new("suite green, 91 tests".into()).unwrap(),
        },
        4,
    )
    .unwrap();
    db::record(
        &mut conn,
        &worker,
        Command::Comment {
            target: Target::Comment(demand),
            body: Prose::new("the deposit stands".into()).unwrap(),
            kind: CommentKind::Note,
        },
        5,
    )
    .unwrap();
    let worktree = saccade::paths::worktree_at(&repo, 0);
    std::fs::write(worktree.join("receipt"), "done\n").unwrap();
    sh(&worktree, &["add", "."]);
    sh(&worktree, &["commit", "-m", "receipt"]);
    close(&mut db::open(&db_path).unwrap(), TaskId(0)).unwrap();

    let seen = wait(&db_path, demand, Some(5)).unwrap();
    assert!(seen.contains("settled"), "{seen}");
    assert!(seen.contains("receipt: suite green, 91 tests"), "{seen}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn wait_releases_on_cancellation() {
    let (repo, db_path, demand) = scaffold("wait-cancel");
    let prepared = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    db::record(
        &mut db::open(&db_path).unwrap(),
        &agent(),
        Command::CancelIncarnation {
            id: prepared.incarnation,
        },
        3,
    )
    .unwrap();

    let seen = wait(&db_path, demand, Some(5)).unwrap();
    assert!(seen.contains("cancelled"), "{seen}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn wait_releases_on_refusal_carrying_the_reason() {
    let (repo, db_path, demand) = scaffold("wait-refuse");
    db::record(
        &mut db::open(&db_path).unwrap(),
        &Context::system(),
        Command::RefuseDemand {
            demand,
            reason: Prose::new("t-0 is dropped; dropped tasks never run".into()).unwrap(),
        },
        3,
    )
    .unwrap();

    let seen = wait(&db_path, demand, Some(5)).unwrap();
    assert!(seen.contains("refused"), "{seen}");
    assert!(seen.contains("dropped tasks never run"), "{seen}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn wait_releases_on_an_answer_with_no_run_behind_it() {
    let (repo, db_path, demand) = scaffold("wait-answer");
    // the human's word ends the demand before any run fires
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::Comment {
            target: Target::Comment(demand),
            body: Prose::new("never mind, handled it myself".into()).unwrap(),
            kind: CommentKind::Note,
        },
        3,
    )
    .unwrap();

    let seen = wait(&db_path, demand, Some(5)).unwrap();
    assert!(seen.contains("no work is coming"), "{seen}");
    assert!(seen.contains("answered by c-2 (human person)"), "{seen}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// The path that has never fired in anger: a run bound whose prompt was
/// neither accepted nor rejected, constructed through the prompt
/// machinery's own verbs.
#[test]
fn wait_releases_on_a_prompt_awaiting_its_answer() {
    let (repo, db_path, demand) = scaffold("wait-prompt");
    db::record(
        &mut db::open(&db_path).unwrap(),
        &Context::system(),
        Command::BindIncarnation {
            task_id: TaskId(0),
            response_target: demand,
            trigger: demand.0,
            actor: ActorName::new("pi/t-0-1".into()).unwrap(),
            session: SessionPointer::new("/tmp/pi-session.jsonl".into()).unwrap(),
        },
        3,
    )
    .unwrap();

    let seen = wait(&db_path, demand, Some(5)).unwrap();
    assert!(
        seen.contains("i-2 raised a prompt that awaits an answer"),
        "{seen}"
    );
    assert!(seen.contains("re-arm sac wait c-1"), "{seen}");
    // the release carries the prompt itself
    assert!(seen.contains("Serve tracked demand c-1"), "{seen}");
    assert!(seen.contains("Reply when done, as 'pi/t-0-1'"), "{seen}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// The session body the supervision tests use: a fixed reply at agent
/// tier under the run's derived attribution, as a real executor would
/// leave through the CLI. Each test's driver owns its db, so parallel
/// tests never share a body.
fn fake_session_for(db: PathBuf) -> SessionDriver {
    Arc::new(move |run: &PreparedRun, _prompt: &str, _runs: &LiveRuns| {
        let mut conn = db::open(&db).unwrap();
        let world = world_of(&db);
        let demand = world
            .comments
            .iter()
            .find(|(_, c)| {
                matches!(
                    &c.state,
                    CommentState::Demand {
                        attempt: AgentAttemptState::InFlight { .. },
                        ..
                    }
                )
            })
            .map(|(id, _)| *id)
            .expect("the bound demand is in flight");
        db::record(
            &mut conn,
            &Context {
                actor: run.actor.clone(),
                tier: Tier::Agent,
            },
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("the fake session answered".into()).unwrap(),
                kind: CommentKind::Note,
            },
            9,
        )
        .unwrap();
        Ok(true)
    })
}

#[test]
fn a_write_that_lands_a_demand_fires_a_run_that_answers_it() {
    let (repo, db_path, demand) = scaffold("supervise");
    let runner = RunnerConfig {
        repo_root: repo.clone(),
        actor: ActorName::new("pi".into()).unwrap(),
        driver: fake_session_for(db_path.clone()),
    };
    let app = AppState::with_runner(&db_path, runner).unwrap();

    // the write that lands the demand is the trigger
    supervisor::sweep(&app);

    let seen = wait(&db_path, demand, Some(10)).unwrap();
    assert!(seen.contains("settled"), "{seen}");
    // the session's write carries the derived attribution
    let world = world_of(&db_path);
    match &world.comments[&demand].state {
        CommentState::Demand { response, .. } => match response {
            saccade::objects::comment::ResponseState::Responded { reply } => {
                assert_eq!(world.comments[reply].actor.as_str(), "pi/t-0-1");
            }
            _ => panic!("the reply landed"),
        },
        other => panic!("demand answered: {other:?}"),
    }
    assert_eq!(world.tasks[0].active_incarnation, None);
    match &world.comments[&demand].state {
        CommentState::Demand { attempt, .. } => {
            assert!(matches!(attempt, AgentAttemptState::Spent));
        }
        other => panic!("demand spent: {other:?}"),
    }
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_demand_queued_behind_an_incarnation_fires_when_the_task_frees() {
    let (repo, db_path, first) = scaffold("queue");
    // a second demand on the same task, arriving behind the first
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(0)),
            body: Prose::new("and then this one".into()).unwrap(),
            kind: CommentKind::Demand,
        },
        3,
    )
    .unwrap();
    let second = CommentId(saccade::RecordId(2));

    let runner = RunnerConfig {
        repo_root: repo.clone(),
        actor: ActorName::new("pi".into()).unwrap(),
        driver: fake_session_for(db_path.clone()),
    };
    let app = AppState::with_runner(&db_path, runner).unwrap();

    supervisor::sweep(&app);
    let seen = wait(&db_path, second, Some(10)).unwrap();
    assert!(seen.contains("settled"), "{seen}");
    // both demands spent, in order
    let world = world_of(&db_path);
    for demand in [first, second] {
        match &world.comments[&demand].state {
            CommentState::Demand { attempt, .. } => {
                assert!(
                    matches!(attempt, AgentAttemptState::Spent),
                    "c-{} spent",
                    demand.0.0
                );
            }
            other => panic!("demand spent: {other:?}"),
        }
    }
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// Delivered is sweep-open: a finding on a delivered task fires its
/// in-thread round without reopening the task or touching the deposit.
#[test]
fn a_demand_on_a_delivered_task_fires_its_round() {
    let (repo, db_path, _first) = scaffold("delivered-sweep");
    let worker = ActorName::new("pi/t-0-1".into()).unwrap();
    db::record(
        &mut db::open(&db_path).unwrap(),
        &Context {
            actor: worker.clone(),
            tier: Tier::Agent,
        },
        Command::ClaimTask { id: TaskId(0) },
        2,
    )
    .unwrap();
    db::record(
        &mut db::open(&db_path).unwrap(),
        &Context {
            actor: worker,
            tier: Tier::Agent,
        },
        Command::CompleteTask {
            id: TaskId(0),
            receipt: Prose::new("delivered; the asker has not accepted".into()).unwrap(),
        },
        3,
    )
    .unwrap();

    // the finding arrives on the delivered task
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(0)),
            body: Prose::new("one more finding on the delivered work".into()).unwrap(),
            kind: CommentKind::Demand,
        },
        4,
    )
    .unwrap();
    let finding = latest_demand(&db_path);

    let runner = RunnerConfig {
        repo_root: repo.clone(),
        actor: ActorName::new("pi".into()).unwrap(),
        driver: fake_session_for(db_path.clone()),
    };
    let app = AppState::with_runner(&db_path, runner).unwrap();

    // the sweep fires the round; the session answers under its derived name
    supervisor::sweep(&app);
    let seen = wait(&db_path, finding, Some(10)).unwrap();
    assert!(seen.contains("settled"), "{seen}");

    let world = world_of(&db_path);
    // delivered survived the round: no reopen, the deposit stands
    assert_eq!(
        saccade::views::task_view(&world, TaskId(0)).unwrap().state,
        "delivered"
    );
    assert_eq!(world.tasks[0].active_incarnation, None);
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_demand_on_a_dropped_task_fires_nothing() {
    let (repo, db_path, demand) = scaffold("dropped");
    // the human drops the task while the demand is still pending
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::DropTask {
            id: TaskId(0),
            note: Prose::new("superseded elsewhere".into()).unwrap(),
        },
        3,
    )
    .unwrap();

    let runner = RunnerConfig {
        repo_root: repo.clone(),
        actor: ActorName::new("pi".into()).unwrap(),
        driver: fake_session_for(db_path.clone()),
    };
    let app = AppState::with_runner(&db_path, runner).unwrap();

    // the sweep sees the live demand but fires nothing on the dropped task
    supervisor::sweep(&app);
    let world = world_of(&db_path);
    assert_eq!(world.tasks[0].active_incarnation, None);
    assert!(supervisor::runnable_demands(&world).is_empty());
    match &world.comments[&demand].state {
        CommentState::Demand { attempt, .. } => {
            assert!(matches!(attempt, AgentAttemptState::Authorized { .. }));
        }
        other => panic!("demand still awaiting: {other:?}"),
    }

    // prepare refuses too: the sweep's snapshot can race a landing drop,
    // and the refusal lands where the asker reads
    let refusal_text = match prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    ) {
        Err(RunnerFail::Refused { reason }) => reason,
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(_) => panic!("the dropped task refused prepare"),
    };
    assert!(refusal_text.contains("dropped"), "{refusal_text}");
    let world = world_of(&db_path);
    match &world.comments[&demand].state {
        CommentState::Demand { attempt, .. } => {
            assert!(matches!(attempt, AgentAttemptState::Spent));
        }
        other => panic!("demand spent: {other:?}"),
    }
    let refusal = world.comments[&demand]
        .refusal
        .as_ref()
        .expect("the refusal fact landed");
    assert!(refusal.reason.as_str().contains("dropped tasks never run"));
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// A terminal task's session artifacts outlive their agent dir: the
/// sweep moves what the executor wrote at runtime to the retention
/// root, and the composed surface — credential links, mirrored
/// settings — dies with the dir, never retained.
#[test]
fn a_terminal_tasks_session_artifacts_survive_the_sweep_at_retention() {
    let (repo, db_path, _demand) = scaffold("retention");
    let agent_dir = saccade::paths::agent_dir_at(&repo, 0);
    compose_agent_dir(&agent_dir).unwrap();
    // the executor's runtime output: what pi writes into its agent dir
    std::fs::write(
        agent_dir.join("run-history.jsonl"),
        "{\"session\":\"t-0\"}\n",
    )
    .unwrap();
    std::fs::create_dir_all(agent_dir.join("sessions").join("a-cwd")).unwrap();
    std::fs::write(
        agent_dir
            .join("sessions")
            .join("a-cwd")
            .join("rollup.jsonl"),
        "{}\n",
    )
    .unwrap();

    // terminal: the human drops the task, and the sweep runs
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::DropTask {
            id: TaskId(0),
            note: Prose::new("superseded elsewhere".into()).unwrap(),
        },
        3,
    )
    .unwrap();
    let runner = RunnerConfig {
        repo_root: repo.clone(),
        actor: ActorName::new("pi".into()).unwrap(),
        driver: Arc::new(|_, _, _| Ok(true)),
    };
    let app = AppState::with_runner(&db_path, runner).unwrap();
    supervisor::sweep(&app);

    // the artifacts survive at the retention root, layout intact
    let retained = saccade::paths::retention_at(&repo, 0);
    assert_eq!(
        std::fs::read_to_string(retained.join("run-history.jsonl")).unwrap(),
        "{\"session\":\"t-0\"}\n"
    );
    assert_eq!(
        std::fs::read_to_string(retained.join("sessions").join("a-cwd").join("rollup.jsonl"))
            .unwrap(),
        "{}\n"
    );
    // the agent dir is gone, and the composed surface never reached
    // retention: credentials are links, and links never move
    assert!(!agent_dir.exists());
    for composed in [
        "auth.json",
        "models.json",
        "models-store.json",
        "settings.json",
    ] {
        assert!(!retained.join(composed).exists(), "{composed} was retained");
    }
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_server_without_a_runner_writes_but_never_fires() {
    let (repo, db_path, demand) = scaffold("quiet");
    let app = AppState::open(&db_path).unwrap();
    supervisor::sweep(&app);
    let world = world_of(&db_path);
    assert_eq!(world.tasks[0].active_incarnation, None);
    let _ = demand;
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// The session body that really sleeps: a child process this server
/// owns, so cancel has something to kill.
fn sleeping_session() -> SessionDriver {
    Arc::new(move |run: &PreparedRun, _prompt: &str, runs: &LiveRuns| {
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg("sleep 30")
            .current_dir(&run.worktree)
            .spawn()
            .expect("the sleeper spawns");
        runs.register(run.incarnation, RunHandle::process_only(child.id()));
        let clean = child.wait().map(|s| s.success()).unwrap_or(false);
        runs.unregister(run.incarnation);
        Ok(clean)
    })
}

#[test]
fn a_cancel_kills_the_run_and_frees_the_task() {
    let (repo, db_path, demand) = scaffold("cancel");
    let runner = RunnerConfig {
        repo_root: repo.clone(),
        actor: ActorName::new("pi".into()).unwrap(),
        driver: sleeping_session(),
    };
    let app = AppState::with_runner(&db_path, runner).unwrap();

    supervisor::sweep(&app);
    // wait for the run to bind and its child to register
    let mut incarnation = None;
    for _ in 0..100 {
        if let Some(id) = world_of(&db_path).tasks[0].active_incarnation
            && !app.runs().ids().is_empty()
        {
            incarnation = Some(id);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let incarnation = incarnation.expect("the run bound and registered");

    // the cancel is an ordinary write; the sweep it triggers does the kill
    app.execute(
        &agent(),
        Command::CancelIncarnation { id: incarnation },
        None,
        bare_request(),
    )
    .unwrap();
    supervisor::sweep(&app);

    // the fold is terminal and the child died
    let mut cancelled = false;
    for _ in 0..100 {
        let world = world_of(&db_path);
        if world.tasks[0].active_incarnation.is_none() {
            cancelled = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(cancelled, "the cancel terminalized the run");
    let world = world_of(&db_path);
    assert_eq!(
        world.incarnations[&incarnation].state,
        IncarnationState::Cancelled
    );
    match &world.comments[&demand].state {
        CommentState::Demand { attempt, .. } => {
            assert!(matches!(attempt, AgentAttemptState::Spent));
        }
        other => panic!("demand spent: {other:?}"),
    }
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn boot_recovery_settles_an_orphaned_run() {
    let (repo, db_path, demand) = scaffold("recover");
    // an orphan: prepare bound and accepted, then the server died
    let app = AppState::open(&db_path).unwrap();
    let actor = ActorName::new("pi".into()).unwrap();
    app.with_conn(|conn| prepare(conn, &repo, demand, actor))
        .unwrap()
        .unwrap();
    assert!(world_of(&db_path).tasks[0].active_incarnation.is_some());

    supervisor::recover(&app);

    let world = world_of(&db_path);
    assert_eq!(world.tasks[0].active_incarnation, None);
    match &world.comments[&demand].state {
        CommentState::Demand { attempt, .. } => {
            assert!(matches!(attempt, AgentAttemptState::Spent));
        }
        other => panic!("demand spent: {other:?}"),
    }
    // the settled orphan leaves its workspace for the next run's reuse
    assert!(world.tasks[0].workspace.is_some());
    // the spent unanswered demand is dead: no live authorization, the
    // sweep must not see it
    assert!(supervisor::runnable_demands(&world).is_empty());
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// The last comment landed is the demand just created: ids are dense
/// log positions.
fn latest_demand(db_path: &Path) -> CommentId {
    let world = world_of(db_path);
    let latest = world.comments.keys().map(|id| id.0).max().unwrap();
    CommentId(latest)
}

#[test]
fn a_severed_branch_rebuilds_at_the_recorded_checkpoint() {
    let (repo, db_path, demand) = scaffold("severed");

    // one full course leaves a workspace, a branch, and a checkpoint
    let checkpoint = run_one_course(&repo, &db_path, demand);

    // review hygiene deletes the worktree and the branch while main moves on
    let worktree = saccade::paths::worktree_at(&repo, 0);
    sh(&repo, &["worktree", "remove", worktree.to_str().unwrap()]);
    sh(&repo, &["branch", "-D", "saccade/t-0"]);
    std::fs::write(repo.join("readme"), "main moved on\n").unwrap();
    sh(&repo, &["add", "."]);
    sh(&repo, &["commit", "-m", "advance main"]);

    // the next demand's prepare succeeds: the canvas is rebuilt at the
    // recorded checkpoint, not at the moved main
    let second = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        follow_up_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    // the second run on the task carries the next ordinal
    assert_eq!(second.actor.as_str(), "pi/t-0-2");

    assert!(sh(&repo, &["branch", "--list", "saccade/t-0"]).contains("saccade/t-0"));
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), checkpoint);
    // the rebuilt canvas sits at the recorded work, receipt included
    assert_eq!(sh(&worktree, &["rev-parse", "HEAD"]), checkpoint);
    assert!(worktree.join("receipt").exists());
    let world = world_of(&db_path);
    // the record never moved: the checkpoint still names the receipt head
    assert_eq!(
        world.tasks[0]
            .workspace
            .as_ref()
            .unwrap()
            .checkpoint
            .as_str(),
        checkpoint
    );
    assert_eq!(
        world.incarnations[&world.tasks[0].active_incarnation.unwrap()].state,
        IncarnationState::PromptAccepted
    );
    assert_eq!(second.worktree, worktree);
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_severed_child_branch_rebuilds_at_its_own_checkpoint() {
    let (repo, db_path, demand) = scaffold("severed-child");
    let actor = ActorName::new("pi".into()).unwrap();

    // the parent task runs one course, leaving its branch alive
    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        actor.clone(),
    )
    .unwrap();
    let parent_worktree = saccade::paths::worktree_at(&repo, 0);
    db::record(
        &mut db::open(&db_path).unwrap(),
        &agent(),
        Command::Comment {
            target: Target::Comment(demand),
            body: Prose::new("parent receipt".into()).unwrap(),
            kind: CommentKind::Note,
        },
        3,
    )
    .unwrap();
    close(&mut db::open(&db_path).unwrap(), TaskId(0)).unwrap();

    // the parent's branch advances past its recorded checkpoint while idle
    std::fs::write(parent_worktree.join("notes"), "parent keeps working\n").unwrap();
    sh(&parent_worktree, &["add", "."]);
    sh(&parent_worktree, &["commit", "-m", "parent advances"]);
    let parent_tip = sh(&repo, &["rev-parse", "saccade/t-0"]);

    // a child task births from main and runs one course
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::CreateTask {
            name: Prose::new("help the parent task".into()).unwrap(),
            parent_id: Some(TaskId(0)),
        },
        4,
    )
    .unwrap();
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(1)),
            body: Prose::new("write the child receipt".into()).unwrap(),
            kind: CommentKind::Demand,
        },
        5,
    )
    .unwrap();
    let child_demand = latest_demand(&db_path);
    let child_worktree = saccade::paths::worktree_at(&repo, 1);
    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        child_demand,
        actor.clone(),
    )
    .unwrap();
    db::record(
        &mut db::open(&db_path).unwrap(),
        &agent(),
        Command::Comment {
            target: Target::Comment(child_demand),
            body: Prose::new("child receipt".into()).unwrap(),
            kind: CommentKind::Note,
        },
        6,
    )
    .unwrap();
    std::fs::write(child_worktree.join("receipt"), "done\n").unwrap();
    sh(&child_worktree, &["add", "."]);
    sh(&child_worktree, &["commit", "-m", "child receipt"]);
    close(&mut db::open(&db_path).unwrap(), TaskId(1)).unwrap();
    let child_checkpoint = sh(&repo, &["rev-parse", "saccade/t-1"]);

    // review hygiene severs the child branch and worktree
    sh(
        &repo,
        &["worktree", "remove", child_worktree.to_str().unwrap()],
    );
    sh(&repo, &["branch", "-D", "saccade/t-1"]);

    // the child's next demand rebuilds at its own recorded checkpoint,
    // not at the parent's advanced tip
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(1)),
            body: Prose::new("one more round".into()).unwrap(),
            kind: CommentKind::Demand,
        },
        7,
    )
    .unwrap();
    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        latest_demand(&db_path),
        actor,
    )
    .unwrap();

    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-1"]), child_checkpoint);
    let world = world_of(&db_path);
    let child_workspace = world.tasks[1].workspace.as_ref().unwrap();
    assert_eq!(child_workspace.checkpoint.as_str(), child_checkpoint);
    assert_ne!(child_workspace.checkpoint.as_str(), parent_tip);
    assert_eq!(
        world.incarnations[&world.tasks[1].active_incarnation.unwrap()].state,
        IncarnationState::PromptAccepted
    );
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_deleted_canvas_rebuilds_calmly_under_its_live_branch() {
    let (repo, db_path, demand) = scaffold("rebuild");

    // one full course checkpoints the receipt head
    let checkpoint = run_one_course(&repo, &db_path, demand);
    let worktree = saccade::paths::worktree_at(&repo, 0);

    // manual hygiene removes the canvas by bare rm, branch and record intact
    std::fs::remove_dir_all(&worktree).unwrap();

    let second = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        follow_up_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    assert_eq!(second.actor.as_str(), "pi/t-0-2");

    // the canvas is back at the recorded checkpoint, recorded work included
    assert_eq!(sh(&worktree, &["rev-parse", "HEAD"]), checkpoint);
    assert!(worktree.join("receipt").exists());
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), checkpoint);
    assert!(world_of(&db_path).tasks[0].active_incarnation.is_some());
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_never_run_tasks_canvas_rebuilds_at_base() {
    let (repo, db_path, demand) = scaffold("at-base");
    let base = sh(&repo, &["rev-parse", "main"]);

    // the first run provisions the workspace but settles by cancellation:
    // no checkpoint was ever recorded, so the checkpoint is born at base
    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    let run = world_of(&db_path).tasks[0].active_incarnation.unwrap();
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::CancelIncarnation { id: run },
        3,
    )
    .unwrap();

    // manual hygiene removes the canvas by bare rm
    let worktree = saccade::paths::worktree_at(&repo, 0);
    std::fs::remove_dir_all(&worktree).unwrap();

    let second = prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        follow_up_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    assert_eq!(second.actor.as_str(), "pi/t-0-2");

    // the canvas is back at the task's base, and no checkpoint was invented
    assert_eq!(sh(&worktree, &["rev-parse", "HEAD"]), base);
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), base);
    let world = world_of(&db_path);
    let workspace = world.tasks[0].workspace.as_ref().unwrap();
    assert_eq!(workspace.checkpoint.as_str(), base);
    assert!(world.tasks[0].active_incarnation.is_some());
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn an_unreachable_checkpoint_refuses_naming_lost_work() {
    let (repo, db_path, demand) = scaffold("unreachable");
    let checkpoint = run_one_course(&repo, &db_path, demand);
    let worktree = saccade::paths::worktree_at(&repo, 0);

    // repo surgery takes the canvas, the branch, and the commit itself
    sh(&repo, &["worktree", "remove", worktree.to_str().unwrap()]);
    sh(&repo, &["branch", "-D", "saccade/t-0"]);
    sh(&repo, &["reflog", "expire", "--expire=now", "--all"]);
    sh(&repo, &["gc", "--prune=now"]);
    let survives = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["cat-file", "-e", &checkpoint])
        .output()
        .expect("git runs");
    assert!(!survives.status.success(), "the fixture lost the commit");

    // the next demand refuses: the recorded work is lost, and the refusal
    // names the task, the branch, and the lost commit
    let refusal = match prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        follow_up_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    ) {
        Err(RunnerFail::Refused { reason }) => reason,
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(_) => panic!("the lost checkpoint refused prepare"),
    };
    assert!(refusal.contains("unreachable"), "{refusal}");
    assert!(refusal.contains(&checkpoint), "{refusal}");
    assert!(refusal.contains("recorded work is lost"), "{refusal}");
    assert!(refusal.contains("sac checkpoint t-0"), "{refusal}");

    // the refusal moved nothing and bound no run
    assert!(sh(&repo, &["branch", "--list", "saccade/t-0"]).is_empty());
    assert!(!worktree.exists());
    let world = world_of(&db_path);
    assert!(world.tasks[0].active_incarnation.is_none());
    let refused = world
        .comments
        .values()
        .find(|c| c.refusal.is_some())
        .expect("the refusal fact landed");
    assert!(
        refused
            .refusal
            .as_ref()
            .unwrap()
            .reason
            .as_str()
            .contains("unreachable")
    );
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_dropped_tasks_demand_arriving_to_no_canvas_refuses_calmly() {
    let (repo, db_path, demand) = scaffold("dropped-calm");
    run_one_course(&repo, &db_path, demand);
    let worktree = saccade::paths::worktree_at(&repo, 0);

    // the human drops the task and collects the residue: no canvas, no branch
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::DropTask {
            id: TaskId(0),
            note: Prose::new("superseded".into()).unwrap(),
        },
        4,
    )
    .unwrap();
    sh(&repo, &["worktree", "remove", worktree.to_str().unwrap()]);
    sh(&repo, &["branch", "-D", "saccade/t-0"]);

    // the stale demand refuses through the dropped door: no git error from
    // the missing canvas, no rebuild behind the refusal
    let refusal = match prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        follow_up_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    ) {
        Err(RunnerFail::Refused { reason }) => reason,
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(_) => panic!("the dropped task refused prepare"),
    };
    assert!(refusal.contains("dropped tasks never run"), "{refusal}");
    assert!(sh(&repo, &["branch", "--list", "saccade/t-0"]).is_empty());
    assert!(!worktree.exists());
    let world = world_of(&db_path);
    assert!(world.tasks[0].active_incarnation.is_none());
    let refused = world
        .comments
        .values()
        .find(|c| c.refusal.is_some())
        .expect("the refusal fact landed");
    assert!(
        refused
            .refusal
            .as_ref()
            .unwrap()
            .reason
            .as_str()
            .contains("dropped")
    );
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn prepare_births_the_task_branch_from_main_even_when_head_elsewhere() {
    let (repo, db_path, demand) = scaffold("birth");

    // the repo's HEAD leaves main before the first run
    sh(&repo, &["checkout", "--detach"]);
    std::fs::write(repo.join("readme"), "work off main\n").unwrap();
    sh(&repo, &["add", "."]);
    sh(&repo, &["commit", "-m", "detached work"]);
    let main_head = sh(&repo, &["rev-parse", "main"]);

    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();

    // the task branch was cut from main, not from the detached HEAD
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), main_head);
    let world = world_of(&db_path);
    assert_eq!(
        world.tasks[0].workspace.as_ref().unwrap().base.as_str(),
        main_head
    );
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

/// One full course: prepare, reply, a commit in the worktree, close. The
/// checkpoint lands on the receipt commit and the task's slot frees.
fn run_one_course(repo: &Path, db_path: &Path, demand: CommentId) -> String {
    prepare(
        &mut db::open(db_path).unwrap(),
        repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    db::record(
        &mut db::open(db_path).unwrap(),
        &agent(),
        Command::Comment {
            target: Target::Comment(demand),
            body: Prose::new("receipt written, tests green".into()).unwrap(),
            kind: CommentKind::Note,
        },
        3,
    )
    .unwrap();
    let worktree = saccade::paths::worktree_at(repo, 0);
    std::fs::write(worktree.join("receipt"), "done\n").unwrap();
    sh(&worktree, &["add", "."]);
    sh(&worktree, &["commit", "-m", "receipt"]);
    close(&mut db::open(db_path).unwrap(), TaskId(0)).unwrap();
    sh(repo, &["rev-parse", "saccade/t-0"])
}

/// A follow-up demand on the settled task, returning its id.
fn follow_up_demand(db_path: &Path) -> CommentId {
    db::record(
        &mut db::open(db_path).unwrap(),
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(0)),
            body: Prose::new("one more round".into()).unwrap(),
            kind: CommentKind::Demand,
        },
        4,
    )
    .unwrap();
    latest_demand(db_path)
}

#[test]
fn a_merged_branch_refuses_until_the_verb_records_the_new_head() {
    let (repo, db_path, demand) = scaffold("merge-door");
    let checkpoint = run_one_course(&repo, &db_path, demand);
    let worktree = saccade::paths::worktree_at(&repo, 0);

    // main advances and merges in: the branch tip descends past the checkpoint
    std::fs::write(repo.join("readme"), "main moved on\n").unwrap();
    sh(&repo, &["add", "."]);
    sh(&repo, &["commit", "-m", "advance main"]);
    sh(&worktree, &["merge", "main"]);
    let merged = sh(&repo, &["rev-parse", "saccade/t-0"]);
    assert_ne!(merged, checkpoint);

    // the next demand refuses: the tip is unrecorded advancement
    let second = follow_up_demand(&db_path);
    let refusal = match prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        second,
        ActorName::new("pi".into()).unwrap(),
    ) {
        Err(RunnerFail::Refused { reason }) => reason,
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(_) => panic!("the advanced tip refused prepare"),
    };
    assert!(
        refusal.contains("branch tip has advanced past the recorded checkpoint"),
        "{refusal}"
    );
    assert!(refusal.contains("run: sac checkpoint t-0"), "{refusal}");
    assert!(refusal.contains("saccade/t-0"), "{refusal}");
    assert!(world_of(&db_path).tasks[0].active_incarnation.is_none());
    // the refusal moved nothing
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), merged);

    // the verb records the merged head with the invoking actor
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_sac"))
        .arg("--db")
        .arg(&db_path)
        .arg("--repo")
        .arg(&repo)
        .arg("--offline")
        .env("SACCADE_ACTOR", "pi")
        .arg("checkpoint")
        .arg("t-0")
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(&merged),
        "stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    // the refusal spent the ask: re-asking is a new comment, and once the
    // verb records the merged head the re-ask runs from it
    db::record(
        &mut db::open(&db_path).unwrap(),
        &human(),
        Command::Comment {
            target: Target::Task(TaskId(0)),
            body: Prose::new("re-ask: the head is recorded now".into()).unwrap(),
            kind: CommentKind::Demand,
        },
        5,
    )
    .unwrap();
    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        latest_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    assert!(world_of(&db_path).tasks[0].active_incarnation.is_some());
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), merged);
    // the refused ask keeps its fact on the thread
    let world = world_of(&db_path);
    assert!(world.comments[&second].refusal.is_some());

    // the verb's records carry agent tier under the invoking actor
    let rows = db::load(&db::open_read(&db_path).unwrap()).unwrap().rows;
    let verb_rows: Vec<_> = rows
        .iter()
        .filter(|r| r.kind == "task_workspace_checkpointed" && r.payload.contains(&merged))
        .collect();
    assert_eq!(verb_rows.len(), 1);
    assert_eq!(verb_rows[0].actor, "pi");
    assert_eq!(verb_rows[0].tier, "agent");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_rewound_branch_restores_its_recorded_work() {
    let (repo, db_path, demand) = scaffold("rewind");
    let checkpoint = run_one_course(&repo, &db_path, demand);
    let worktree = saccade::paths::worktree_at(&repo, 0);
    let base = sh(&repo, &["rev-parse", "main"]);

    // review hygiene rewinds the branch and leaves junk behind
    sh(&worktree, &["reset", "--hard", &base]);
    std::fs::write(worktree.join("scratch"), "junk\n").unwrap();

    prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        follow_up_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();

    // the recorded head is restored and the junk is gone
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), checkpoint);
    assert!(!worktree.join("scratch").exists());
    assert!(world_of(&db_path).tasks[0].active_incarnation.is_some());
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn a_diverged_branch_refuses_naming_both_doors() {
    let (repo, db_path, demand) = scaffold("diverge");
    let _checkpoint = run_one_course(&repo, &db_path, demand);
    let worktree = saccade::paths::worktree_at(&repo, 0);
    let base = sh(&repo, &["rev-parse", "main"]);

    // a rewritten lineage: the branch leaves its recorded history
    sh(&worktree, &["reset", "--hard", &base]);
    std::fs::write(worktree.join("other"), "a different lineage\n").unwrap();
    sh(&worktree, &["add", "."]);
    sh(&worktree, &["commit", "-m", "different lineage"]);
    let diverged = sh(&repo, &["rev-parse", "saccade/t-0"]);

    let refusal = match prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        follow_up_demand(&db_path),
        ActorName::new("pi".into()).unwrap(),
    ) {
        Err(RunnerFail::Refused { reason }) => reason,
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(_) => panic!("the diverged tip refused prepare"),
    };
    assert!(refusal.contains("diverged"), "{refusal}");
    assert!(refusal.contains("sac checkpoint t-0"), "{refusal}");
    assert!(refusal.contains("reset the branch back"), "{refusal}");
    assert!(world_of(&db_path).tasks[0].active_incarnation.is_none());
    // the refusal moved nothing
    assert_eq!(sh(&repo, &["rev-parse", "saccade/t-0"]), diverged);
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn prepare_refuses_a_disk_only_leftover_without_prescribing_git() {
    let (repo, db_path, _demand) = scaffold("leftover");

    // the worktree path exists with no workspace behind it
    let worktree = saccade::paths::worktree_at(&repo, 0);
    std::fs::create_dir_all(&worktree).unwrap();

    let message = match prepare(
        &mut db::open(&db_path).unwrap(),
        &repo,
        CommentId(saccade::RecordId(1)),
        ActorName::new("pi".into()).unwrap(),
    ) {
        Err(RunnerFail::Refused { reason }) => reason,
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(_) => panic!("the leftover worktree refused prepare"),
    };
    assert!(!message.contains("git"), "{message}");
    assert!(message.contains("human"), "{message}");
    // the refusal created nothing behind the record's back
    assert!(sh(&repo, &["branch", "--list", "saccade/t-0"]).is_empty());

    // the refusal is a readable fact: the demand's next reader finds it
    // on the thread, and the sweep will not re-fire the spent ask
    let world = world_of(&db_path);
    let demand = CommentId(saccade::RecordId(1));
    match &world.comments[&demand].state {
        CommentState::Demand { attempt, .. } => {
            assert!(matches!(attempt, AgentAttemptState::Spent));
        }
        other => panic!("the refused ask spent its authorization: {other:?}"),
    }
    let refusal = world.comments[&demand]
        .refusal
        .as_ref()
        .expect("the refusal fact landed");
    assert!(refusal.reason.as_str().contains("disk-only leftover"));
    assert!(supervisor::runnable_demands(&world).is_empty());
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}
