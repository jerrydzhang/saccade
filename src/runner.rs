//! Workspace effects and run orchestration. The lifecycle splits into
//! prepare / execute_session / close so a supervisor can compose the
//! stages with its own session driver.

use std::path::{Path, PathBuf};

use crate::db::{self, ExecuteFail};
use crate::objects::comment::{CommentId, CommentState, ResponseState};
use crate::objects::incarnation::{IncarnationId, IncarnationState};
use crate::objects::task::{TaskContext, TaskId, TaskState};
use crate::objects::workspace::WorktreeState;
use crate::paths;
use crate::types::actor::ActorName;
use crate::types::failure::{FailureCode, FailureEvidence};
use crate::types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
use crate::{Command, Context, Prose, RecordId, World};
use rusqlite::Connection;
use tracing::warn;

#[derive(Debug, thiserror::Error)]
pub enum RunnerFail {
    #[error("{0}")]
    Usage(String),
    #[error("git: {0}")]
    Git(String),
    #[error("{reason}")]
    Refused { reason: String },
    #[error(transparent)]
    Db(#[from] ExecuteFail),
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

/// The named ref's tip, or None when the ref does not resolve.
fn branch_tip(cwd: &Path, refs: &str) -> Option<String> {
    std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["rev-parse", "--verify", refs])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// True when `ancestor` is an ancestor of `descendant`, equal included.
fn is_ancestor(cwd: &Path, ancestor: &str, descendant: &str) -> bool {
    std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["merge-base", "--is-ancestor", ancestor, descendant])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

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

/// The base a healed branch cuts from: the parent task's branch tip
/// while that branch is alive, else main.
fn heal_base(world: &World, repo_root: &Path, task: TaskId) -> Result<String, RunnerFail> {
    let parent_branch = task_ctx(world, task)?
        .task
        .parent_id
        .and_then(|parent| world.tasks.get(parent.0))
        .and_then(|parent| parent.workspace.as_ref())
        .map(|workspace| workspace.branch.clone());
    if let Some(branch) = parent_branch
        && let Ok(tip) = git(
            repo_root,
            &[
                "rev-parse",
                "--verify",
                &format!("refs/heads/{}", String::from(branch)),
            ],
        )
    {
        return Ok(tip);
    }
    git(repo_root, &["rev-parse", "main"])
}

/// A refusal the asker must read: the fact lands on the demand's
/// thread before the error rides out. The operator warn stays as the
/// second home; a fold refusal here means a racing sweep already
/// wrote it.
fn refuse(conn: &mut Connection, demand: CommentId, reason: String) -> RunnerFail {
    let reason = Prose::new(reason).expect("refusal reasons are non-empty");
    let said = db::record(
        conn,
        &Context::system(),
        Command::RefuseDemand {
            demand,
            reason: reason.clone(),
        },
        db::now_epoch(),
    );
    if let Err(e) = said {
        warn!(demand = demand.0.0, "the refusal fact did not land: {e}");
    }
    RunnerFail::Refused {
        reason: reason.into(),
    }
}

pub struct PreparedRun {
    pub task: TaskId,
    pub demand: CommentId,
    pub incarnation: IncarnationId,
    pub worktree: PathBuf,
    pub session: PathBuf,
    pub actor: ActorName,
}

/// Validation completes before any git effect, so a refused run never
/// orphans a worktree.
pub fn prepare(
    conn: &mut Connection,
    repo_root: &Path,
    demand: CommentId,
    actor: ActorName,
) -> Result<PreparedRun, RunnerFail> {
    let world = load_world(conn)?;
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
    // dropped stays terminal: the trigger's snapshot can race a drop
    if matches!(ctx.task.state, TaskState::Dropped) {
        return Err(refuse(
            conn,
            demand,
            format!("t-{} is dropped; dropped tasks never run", task.0),
        ));
    }
    let worktree = paths::worktree_at(repo_root, task.0);
    let session = paths::session_at(repo_root, task.0);
    let branch = format!("saccade/t-{}", task.0);
    let system = Context::system();
    let now = db::now_epoch();

    match &ctx.workspace {
        // a task's workspace persists across runs: the branch carries
        // advancement between checkpoints, so the second run continues
        // from the last one instead of refusing.
        Some(workspace) => {
            let checkpoint = workspace.checkpoint.as_str().to_string();
            let branch: String = workspace.branch.clone().into();
            let refs = format!("refs/heads/{branch}");
            if let Some(tip) = branch_tip(repo_root, &refs) {
                // the branch tip and the recorded checkpoint reconcile
                // before any git effect, so a refused run moves nothing
                let target = if tip == checkpoint {
                    checkpoint.clone()
                } else if is_ancestor(repo_root, &checkpoint, &tip) {
                    // the refusal text names task, branch, and the door; it
                    // rides the error into the sweep's own log line
                    return Err(refuse(
                        conn,
                        demand,
                        format!(
                            "t-{} branch {}: branch tip has advanced past the recorded checkpoint; run: sac checkpoint t-{} to record the current head",
                            task.0, branch, task.0
                        ),
                    ));
                } else if is_ancestor(repo_root, &tip, &checkpoint) {
                    warn!(
                        task = task.0,
                        branch = %branch,
                        restored_to = %checkpoint,
                        found_behind = %tip,
                        "the branch sat behind the recorded checkpoint; restoring the recorded work"
                    );
                    checkpoint.clone()
                } else {
                    return Err(refuse(
                        conn,
                        demand,
                        format!(
                            "t-{} branch {} diverged from the recorded checkpoint; record the new lineage with sac checkpoint t-{}, or reset the branch back",
                            task.0, branch, task.0
                        ),
                    ));
                };
                if worktree.exists() {
                    git(&worktree, &["reset", "--hard", &target])?;
                    git(&worktree, &["clean", "-fd"])?;
                } else {
                    git(
                        repo_root,
                        &["worktree", "add", &worktree.to_string_lossy(), &branch],
                    )?;
                    if target != tip {
                        git(&worktree, &["reset", "--hard", &target])?;
                    }
                }
            } else {
                let base = heal_base(&world, repo_root, task)?;
                warn!(
                    task = task.0,
                    branch = %branch,
                    checkpoint = %checkpoint,
                    rebuilt_from = %base,
                    "the recorded branch is gone; its checkpoint went unreachable with it, rebuilding the worktree"
                );
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
                db::record(
                    conn,
                    &system,
                    Command::CheckpointWorkspace {
                        task_id: task,
                        checkpoint: commit(base)?,
                    },
                    now,
                )?;
            }
        }
        None => {
            let base = git(repo_root, &["rev-parse", "main"])?;
            if worktree.exists() {
                return Err(refuse(
                    conn,
                    demand,
                    format!(
                        "{} already exists; the record has no workspace for this task, so it is disk-only leftover. Reconcile it through the human against the record.",
                        worktree.display()
                    ),
                ));
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
            db::record(
                conn,
                &system,
                Command::CreateWorkspace {
                    task_id: task,
                    base: commit(base)?,
                    branch: GitBranch::new(branch)
                        .map_err(|e| RunnerFail::Usage(format!("bad branch: {e:?}")))?,
                },
                now,
            )?;
            db::record(
                conn,
                &system,
                Command::CreateWorktree {
                    task_id: task,
                    worktree: WorktreePath::new(worktree.clone())
                        .map_err(|e| RunnerFail::Usage(format!("worktree path: {e:?}")))?,
                },
                now,
            )?;
        }
    };
    let (bound, _) = db::record(
        conn,
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
    db::record(
        conn,
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

/// Pointers only: the worktree is the cwd, the demand and reply door are
/// named; the repo's own skill teaches the verbs.
pub fn pointer_prompt(run: &PreparedRun, sac: &str) -> String {
    let actor = run.actor.as_str();
    format!(
        "Serve tracked demand c-{demand} on task t-{task} of this repository; you are working in its prepared worktree. \
Read it: {sac} show t-{task}. Do the work in this directory. \
Reply when done, as '{actor}': \
{sac} comment '#{demand}' '<your answer>'. \
Let other tasks' runs settle on their own; cancel only what you started \
({sac} cancel t-<task> stops a runaway). \
The .agents/skills/saccade skill in this repo documents the tracker.",
        demand = run.demand.0.0,
        task = run.task.0,
    )
}

/// Spawn the executor on the prompt and block until it exits. A clean
/// exit is not a success claim; the reply's presence is. The pid is
/// registered for the run's life so cancel and shutdown can reach it.
pub fn execute_session(
    run: &PreparedRun,
    prompt: &str,
    runs: &crate::supervisor::LiveRuns,
) -> Result<bool, RunnerFail> {
    let mut child = std::process::Command::new("pi")
        // the run's actor is machine-established: the parent's env does
        // not pass through, only the actor the bind named
        .env_remove("SACCADE_ACTOR")
        .env("SACCADE_ACTOR", run.actor.as_str())
        .env_remove("SACCADE_TIER")
        .env_remove("SACCADE_SERVER")
        .arg("-p")
        .arg("--session")
        .arg(&run.session)
        .arg("-a")
        .arg("--")
        .arg(prompt)
        .current_dir(&run.worktree)
        .spawn()
        .map_err(|e| RunnerFail::Git(format!("spawning the executor failed: {e}")))?;
    runs.register(run.incarnation, child.id());
    let clean = child.wait().map(|s| s.success());
    runs.unregister(run.incarnation);
    clean.map_err(|e| RunnerFail::Git(format!("waiting on the executor failed: {e}")))
}

pub fn close(conn: &mut Connection, task: TaskId) -> Result<String, RunnerFail> {
    let world = load_world(conn)?;
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
    db::record(
        conn,
        &system,
        Command::CheckpointWorkspace {
            task_id: task,
            checkpoint: commit(checkpoint.clone())?,
        },
        now,
    )?;
    if let Some(reply) = reply {
        db::record(
            conn,
            &system,
            Command::MarkRecord {
                record_id: reply.0,
                incarnation_id: incarnation,
            },
            now,
        )?;
    }
    db::record(
        conn,
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

/// Close a run the server no longer owns, against disk truth: an
/// accepted run settles through close's honest path; a bound run is
/// interrupted, because the lifecycle table forbids settling a run
/// that never accepted.
pub fn close_as_found(conn: &mut Connection, task: TaskId) -> Result<String, RunnerFail> {
    let world = load_world(conn)?;
    let Some(incarnation) = task_ctx(&world, task)?.active_incarnation else {
        return Err(RunnerFail::Usage(format!(
            "t-{} has no incarnation to recover",
            task.0
        )));
    };
    match world.incarnations[&incarnation].state.clone() {
        IncarnationState::PromptAccepted => close(conn, task),
        IncarnationState::Bound => {
            let system = Context::system();
            db::record(
                conn,
                &system,
                Command::RejectPrompt {
                    id: incarnation,
                    evidence: FailureEvidence::new(
                        FailureCode::PromptRejected,
                        Some("recovered at boot: the run never accepted".into()),
                    ),
                },
                db::now_epoch(),
            )?;
            Ok(format!("recovered i-{} as interrupted", incarnation.0.0))
        }
        terminal => Err(RunnerFail::Usage(format!(
            "i-{} is already {terminal:?}; nothing to recover",
            incarnation.0.0
        ))),
    }
}
