//! Workspace effects and run orchestration: provision a task's workspace,
//! bind a run serving one of its demands, drive the session, and settle
//! with a real checkpoint. `prepare` and `close` are the server's
//! supervisor calls; `execute_session` is the driver the server replaces
//! with its own.

use std::path::{Path, PathBuf};

use crate::db::{self, ExecuteFail};
use crate::objects::comment::{CommentId, CommentState, ResponseState};
use crate::objects::incarnation::IncarnationId;
use crate::objects::task::{TaskContext, TaskId};
use crate::objects::workspace::WorktreeState;
use crate::paths;
use crate::types::actor::ActorName;
use crate::types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
use crate::{Command, Context, RecordId, World};

#[derive(Debug)]
pub enum RunnerFail {
    Usage(String),
    Git(String),
    Db(ExecuteFail),
}

impl From<ExecuteFail> for RunnerFail {
    fn from(e: ExecuteFail) -> Self {
        RunnerFail::Db(e)
    }
}

fn git(cwd: &Path, args: &[&str]) -> Result<String, RunnerFail> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .map_err(|e| RunnerFail::Git(format!("git {}: {e}", args.first().unwrap_or(&""))))?;
    if !out.status.success() {
        return Err(RunnerFail::Git(format!(
            "git {} failed: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The world as the runner needs it: loaded fully or not at all.
fn load_world(conn: &rusqlite::Connection) -> Result<World, RunnerFail> {
    let loadout = db::load(conn).map_err(ExecuteFail::Db)?;
    match loadout.state {
        db::LoadState::Full(world) => Ok(world),
        db::LoadState::Degraded(reason) => Err(RunnerFail::Usage(format!(
            "world projection unavailable: {reason}"
        ))),
    }
}

fn task_ctx(world: &World, task: TaskId) -> Result<&TaskContext, RunnerFail> {
    world
        .tasks
        .get(task.0)
        .ok_or_else(|| RunnerFail::Usage(format!("no task t-{} in this tracker", task.0)))
}

fn commit(hash: String) -> Result<GitCommit, RunnerFail> {
    GitCommit::new(hash).map_err(|e| RunnerFail::Usage(format!("git gave no commit: {e:?}")))
}

/// A run opened for a demand: everything the session and the close need.
pub struct PreparedRun {
    pub task: TaskId,
    pub demand: CommentId,
    pub incarnation: IncarnationId,
    pub worktree: PathBuf,
    pub session: PathBuf,
    pub actor: ActorName,
}

/// Provision a workspace and bind a run serving the demand: validated
/// against the world first, git effects before their records, a refused
/// record leaves a worktree to remove by hand.
pub fn prepare(
    db_path: &Path,
    repo_root: &Path,
    demand: CommentId,
    actor: ActorName,
) -> Result<PreparedRun, RunnerFail> {
    let mut conn = db::open(db_path).map_err(ExecuteFail::Db)?;

    let world = load_world(&conn)?;
    let demand_ctx = world
        .comments
        .get(&demand)
        .ok_or_else(|| RunnerFail::Usage(format!("no comment c-{} in this tracker", demand.0.0)))?;
    match &demand_ctx.state {
        CommentState::AddressedToAgent { .. } => {}
        _ => {
            return Err(RunnerFail::Usage(format!(
                "c-{} is not an agent-addressed demand",
                demand.0.0
            )));
        }
    }
    // the demand's own root names the task; a mismatch is unrepresentable
    let task = demand_ctx.comment.root;
    let ctx = task_ctx(&world, task)?;
    if ctx.active_incarnation.is_some() {
        return Err(RunnerFail::Usage(format!(
            "t-{} already runs an incarnation",
            task.0
        )));
    }
    if let Some(workspace) = &ctx.workspace
        && !matches!(workspace.worktree, WorktreeState::Absent)
    {
        return Err(RunnerFail::Usage(format!(
            "t-{} already has a worktree recorded",
            task.0
        )));
    }

    let worktree = paths::worktree_at(repo_root, task.0);
    let session = paths::session_at(repo_root, task.0);
    let branch = format!("saccade/t-{}", task.0);
    let base = git(repo_root, &["rev-parse", "HEAD"])?;
    if worktree.exists() {
        return Err(RunnerFail::Usage(format!(
            "{} already exists; remove it before provisioning",
            worktree.display()
        )));
    }
    git(
        repo_root,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            &worktree.to_string_lossy(),
            &base,
        ],
    )?;

    let system = Context::system();
    let now = db::now_epoch();
    db::execute(
        &mut conn,
        &system,
        Command::CreateWorkspace {
            task_id: task,
            base: commit(base)?,
            branch: GitBranch::new(branch)
                .map_err(|e| RunnerFail::Usage(format!("bad branch: {e:?}")))?,
        },
        now,
    )?;
    db::execute(
        &mut conn,
        &system,
        Command::CreateWorktree {
            task_id: task,
            worktree: WorktreePath::new(worktree.clone())
                .map_err(|e| RunnerFail::Usage(format!("worktree path: {e:?}")))?,
        },
        now,
    )?;
    let bound = db::execute(
        &mut conn,
        &system,
        Command::BindIncarnation {
            task_id: task,
            response_target: demand,
            trigger: demand.0,
            actor: actor.clone(),
            session: SessionPointer::new(session.clone())
                .map_err(|e| RunnerFail::Usage(format!("session path: {e:?}")))?,
        },
        now,
    )?;
    let incarnation = IncarnationId(RecordId(bound[0].seq));
    db::execute(
        &mut conn,
        &system,
        Command::AcceptPrompt { id: incarnation },
        now,
    )?;

    Ok(PreparedRun {
        task,
        demand,
        incarnation,
        worktree,
        session,
        actor,
    })
}

/// The minimal pointer prompt: where the work lives and which door the
/// answer goes through. The spawned session's own skill carries the rest.
pub fn pointer_prompt(run: &PreparedRun, sac: &str) -> String {
    let actor = run.actor.as_str();
    format!(
        "Serve tracked demand c-{demand} on task t-{task} of this repository; you are working in its prepared worktree. \
Read it: {sac} show t-{task}. Do the work in this directory. \
Reply when done, at agent tier as '{actor}': \
{sac} comment '#{demand}' '<your answer>' --actor {actor} --tier agent. \
The .agents/skills/saccade skill in this repo documents the tracker.",
        demand = run.demand.0.0,
        task = run.task.0,
    )
}

/// Drive the session: spawn the executor on the prompt in the worktree,
/// named session file, project files trusted, and block until it exits.
/// Returns whether the executor exited cleanly; a clean exit is not a
/// success claim — the reply's presence is.
pub fn execute_session(run: &PreparedRun, prompt: &str) -> Result<bool, RunnerFail> {
    let status = std::process::Command::new("pi")
        .arg("-p")
        .arg("--session")
        .arg(&run.session)
        .arg("-a")
        .arg("--")
        .arg(prompt)
        .current_dir(&run.worktree)
        .status()
        .map_err(|e| RunnerFail::Git(format!("spawning the executor failed: {e}")))?;
    Ok(status.success())
}

/// Close the task's active run: checkpoint the worktree head, mark the
/// demand's reply as the run's production, settle.
pub fn close(db_path: &Path, task: TaskId) -> Result<String, RunnerFail> {
    let mut conn = db::open(db_path).map_err(ExecuteFail::Db)?;

    let world = load_world(&conn)?;
    let ctx = task_ctx(&world, task)?;
    let incarnation = ctx
        .active_incarnation
        .ok_or_else(|| RunnerFail::Usage(format!("t-{} has no incarnation to settle", task.0)))?;
    let run = &world.incarnations[&incarnation];
    let demand = run.response_target;
    let reply = match &world.comments[&demand].state {
        CommentState::AddressedToAgent { response, .. } => match response {
            ResponseState::Responded { reply } => Some(*reply),
            ResponseState::Awaiting => None,
        },
        _ => None,
    };
    let WorktreeState::Present(worktree) = &ctx
        .workspace
        .as_ref()
        .expect("a bound run has a workspace")
        .worktree
    else {
        return Err(RunnerFail::Usage(format!(
            "t-{} has no worktree to checkpoint",
            task.0
        )));
    };

    let checkpoint = git(worktree.as_path(), &["rev-parse", "HEAD"])?;

    let system = Context::system();
    let now = db::now_epoch();
    db::execute(
        &mut conn,
        &system,
        Command::CheckpointWorkspace {
            task_id: task,
            checkpoint: commit(checkpoint.clone())?,
        },
        now,
    )?;
    if let Some(reply) = reply {
        db::execute(
            &mut conn,
            &system,
            Command::MarkRecord {
                record_id: reply.0,
                incarnation_id: incarnation,
            },
            now,
        )?;
    }
    db::execute(
        &mut conn,
        &system,
        Command::SettleIncarnation { id: incarnation },
        now,
    )?;

    Ok(match reply {
        Some(reply) => format!(
            "settled i-{} at {checkpoint}; produced c-{}",
            incarnation.0.0, reply.0.0
        ),
        None => format!(
            "settled i-{} at {checkpoint}; the demand went unanswered",
            incarnation.0.0
        ),
    })
}

/// The whole lifecycle in one call: prepare, drive the session to
/// completion, close. This is the slice the server's supervisor takes
/// over, replacing only the session driver.
pub fn run(
    db_path: &Path,
    repo_root: &Path,
    demand: CommentId,
    actor: ActorName,
) -> Result<String, RunnerFail> {
    let prepared = prepare(db_path, repo_root, demand, actor)?;
    let bound = format!(
        "run i-{} bound, serving c-{}\nworktree: {}\nsession: {}",
        prepared.incarnation.0.0,
        prepared.demand.0.0,
        prepared.worktree.display(),
        prepared.session.display(),
    );
    let sac = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "sac".into());
    let clean = execute_session(&prepared, &pointer_prompt(&prepared, &sac))?;
    let settled = close(db_path, prepared.task)?;
    let exit = if clean {
        String::new()
    } else {
        "the executor exited uncleanly; the reply's presence is the outcome\n".to_string()
    };
    Ok(format!("{bound}\n{exit}{settled}"))
}

/// Block until the demand's reply lands; `deadline` bounds the wait in
/// seconds (None waits forever). Read-only: abandonment costs nothing.
pub fn wait(
    db_path: &Path,
    comment: CommentId,
    deadline: Option<u64>,
) -> Result<String, RunnerFail> {
    let start = std::time::Instant::now();
    loop {
        let conn = db::open_read(db_path).map_err(ExecuteFail::Db)?;
        let world = load_world(&conn)?;
        let ctx = world.comments.get(&comment).ok_or_else(|| {
            RunnerFail::Usage(format!("no comment c-{} in this tracker", comment.0.0))
        })?;
        let reply = match &ctx.state {
            CommentState::Unaddressed => {
                return Err(RunnerFail::Usage(format!(
                    "c-{} addresses nobody; it will never respond",
                    comment.0.0
                )));
            }
            CommentState::AddressedToHuman { response } => match response {
                ResponseState::Responded { reply } => Some(*reply),
                ResponseState::Awaiting => None,
            },
            CommentState::AddressedToAgent { response, .. } => match response {
                ResponseState::Responded { reply } => Some(*reply),
                ResponseState::Awaiting => None,
            },
        };
        if let Some(reply) = reply {
            let reply_ctx = &world.comments[&reply];
            return Ok(format!(
                "c-{} answered by c-{} ({}):\n{}",
                comment.0.0,
                reply.0.0,
                reply_ctx.actor.as_str(),
                reply_ctx.comment.body.as_str(),
            ));
        }
        if let Some(seconds) = deadline
            && start.elapsed().as_secs() >= seconds
        {
            return Err(RunnerFail::Usage(format!(
                "c-{} still awaiting after {seconds}s",
                comment.0.0
            )));
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
