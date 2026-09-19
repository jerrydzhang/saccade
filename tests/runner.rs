//! The runner end to end: a real git repository, a real tracker db, the
//! machinery verbs driven through their orchestration. The session driver
//! itself is exercised by the live dogfood run, not here — these tests
//! sit at the seam on either side of it.

use std::path::{Path, PathBuf};

use saccade::db::{self, LoadState};
use saccade::objects::comment::CommentState;
use saccade::objects::incarnation::IncarnationState;
use saccade::runner::{RunnerFail, close, pointer_prompt, prepare, wait};
use saccade::types::actor::ActorName;
use saccade::{Addressee, Command, CommentId, Context, Prose, Target, TaskId, Tier, World};

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
            addressee: Some(Addressee::Agent),
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
        &db_path,
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
            addressee: None,
        },
        3,
    )
    .unwrap();

    // wait sees the answer as soon as it lands
    let seen = wait(&db_path, demand, Some(5)).unwrap();
    assert!(seen.contains("receipt written, tests green"), "{seen}");

    // a commit in the worktree becomes the checkpoint
    std::fs::write(worktree.join("receipt"), "done\n").unwrap();
    sh(&worktree, &["add", "."]);
    sh(&worktree, &["commit", "-m", "receipt"]);

    let note = close(&db_path, TaskId(0)).unwrap();
    assert!(note.contains("settled i-"), "{note}");

    let world = world_of(&db_path);
    assert_eq!(world.tasks[0].active_incarnation, None);
    assert_eq!(
        world.incarnations[&prepared.incarnation].state,
        IncarnationState::Settled
    );
    // the produced pointer names exactly the demand's reply
    let reply = match &world.comments[&demand].state {
        CommentState::AddressedToAgent { response, .. } => match response {
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
        CommentState::AddressedToAgent { attempt, .. } => {
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
            addressee: Some(Addressee::Human),
        },
        3,
    )
    .unwrap();
    let refused = prepare(
        &db_path,
        &repo,
        CommentId(saccade::RecordId(2)),
        ActorName::new("pi".into()).unwrap(),
    );
    assert!(matches!(refused, Err(RunnerFail::Usage(_))));
    // and no worktree was orphaned by the refusal
    assert!(!saccade::paths::worktree_at(&repo, 0).exists());

    // a second run on the same task has no free slot
    prepare(
        &db_path,
        &repo,
        CommentId(saccade::RecordId(1)),
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    let refused = prepare(
        &db_path,
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
        close(&db_path, TaskId(1)),
        Err(RunnerFail::Usage(_))
    ));

    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn the_pointer_prompt_names_the_work_and_the_reply_door() {
    let (repo, db_path, demand) = scaffold("prompt");
    let prepared = prepare(
        &db_path,
        &repo,
        demand,
        ActorName::new("pi".into()).unwrap(),
    )
    .unwrap();
    let prompt = pointer_prompt(&prepared, "/bin/sac");
    assert!(prompt.contains("c-1"), "{prompt}");
    assert!(prompt.contains("t-0"), "{prompt}");
    assert!(prompt.contains("/bin/sac comment '#1'"), "{prompt}");
    assert!(prompt.contains("--tier agent"), "{prompt}");
    assert!(prompt.contains("'pi'"), "{prompt}");
    std::fs::remove_dir_all(db_path.parent().unwrap()).unwrap();
}

#[test]
fn wait_reports_an_unanswered_demand_at_its_deadline() {
    let (_repo, db_path, demand) = scaffold("wait");
    let refused = wait(&db_path, demand, Some(0));
    assert!(matches!(refused, Err(RunnerFail::Usage(_))));
    std::fs::remove_dir_all(db_path.parent().unwrap()).unwrap();
}
