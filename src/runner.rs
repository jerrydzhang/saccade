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
use crate::rpc::{ClientCommand, ServerEvent, frames};
use crate::supervisor::RunHandle;
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

/// The commit an object database still holds: `Some` when the sha
/// resolves as a commit, `None` when nothing retains it.
fn commit_exists(repo_root: &Path, sha: &str) -> Option<String> {
    branch_tip(repo_root, &format!("{sha}^{{commit}}"))
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

/// The ask door's source, embedded from the in-tree `executor/ask.ts`:
/// the file is the source, this is its pinned copy inside the binary,
/// so every build is spawn-capable with no repo lookup anywhere.
const ASK_EXTENSION: &str = include_str!("../executor/ask.ts");

/// The ask door's materialized home: the agent dir compose provisions.
/// One definition, so the writer and the spawner never disagree.
pub fn ask_extension_at(agent_dir: &Path) -> PathBuf {
    agent_dir.join("ask.ts")
}

/// Compose the agent dir an incarnation's pi runs under. Links, not
/// copies: a copied credential would freeze OAuth refresh. The ask
/// door materializes here; a failure to write it is the failure that
/// surfaces — no run spawns on a door it cannot load.
pub fn compose_agent_dir(dir: &Path) -> Result<(), RunnerFail> {
    std::fs::create_dir_all(dir)
        .map_err(|e| RunnerFail::Git(format!("agent dir {}: {e}", dir.display())))?;
    let operator = home_agent_dir();
    for name in ["auth.json", "models.json", "models-store.json"] {
        let source = operator.join(name);
        if source.exists()
            && let Err(e) = symlink_fresh(&source, &dir.join(name))
        {
            return Err(RunnerFail::Git(format!("agent dir link {name}: {e}")));
        }
    }
    let mirrored = operator
        .join("settings.json")
        .exists()
        .then(|| mirror_model_settings(&operator))
        .flatten();
    let settings = match mirrored {
        Some(json) => serde_json::to_string(&json),
        None => Ok("{}".to_string()),
    }
    .map_err(|e| RunnerFail::Git(format!("agent dir settings: {e}")))?;
    std::fs::write(dir.join("settings.json"), settings)
        .map_err(|e| RunnerFail::Git(format!("agent dir settings: {e}")))?;
    std::fs::write(ask_extension_at(dir), ASK_EXTENSION)
        .map_err(|e| RunnerFail::Git(format!("agent dir ask extension: {e}")))?;
    Ok(())
}

/// The operator's own agent dir, pi's conventional `~/.pi/agent`.
fn home_agent_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".pi").join("agent"))
        .unwrap_or_else(|| PathBuf::from("/nonexistent"))
}

fn mirror_model_settings(operator: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(operator.join("settings.json")).ok()?;
    let full: serde_json::Value = serde_json::from_str(&text).ok()?;
    let mut mirrored = serde_json::Map::new();
    for key in ["defaultModel", "defaultProvider", "defaultThinkingLevel"] {
        if let Some(value) = full.get(key) {
            mirrored.insert(key.to_string(), value.clone());
        }
    }
    Some(serde_json::Value::Object(mirrored))
}

/// A symlink that replaces whatever stands at `at`, so recomposition
/// is idempotent.
fn symlink_fresh(source: &Path, at: &Path) -> std::io::Result<()> {
    if at.symlink_metadata().is_ok() {
        std::fs::remove_file(at)?;
    }
    std::os::unix::fs::symlink(source, at)
}

pub struct PreparedRun {
    pub task: TaskId,
    pub demand: CommentId,
    pub incarnation: IncarnationId,
    pub worktree: PathBuf,
    pub session: PathBuf,
    pub agent_dir: PathBuf,
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
        CommentState::Demand { .. } => {}
        _ => {
            return Err(RunnerFail::Usage(format!(
                "c-{} is not a demand",
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
    let agent_dir = paths::agent_dir_at(repo_root, task.0);
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
                    // the canvas is a cache: a manual deletion may have
                    // been a clean remove or a bare rm, and a bare rm
                    // leaves a registration that prune clears
                    git(repo_root, &["worktree", "prune"])?;
                    git(
                        repo_root,
                        &["worktree", "add", &worktree.to_string_lossy(), &branch],
                    )?;
                    if target != tip {
                        git(&worktree, &["reset", "--hard", &target])?;
                    }
                }
            } else if commit_exists(repo_root, &checkpoint).is_some() {
                // the branch is gone but the recorded head still lives:
                // the canvas is rebuilt at the checkpoint the record names
                warn!(
                    task = task.0,
                    branch = %branch,
                    rebuilt_at = %checkpoint,
                    "the recorded branch is gone; rebuilding the canvas at the recorded checkpoint"
                );
                git(repo_root, &["worktree", "prune"])?;
                git(
                    repo_root,
                    &[
                        "worktree",
                        "add",
                        "-b",
                        &branch,
                        &worktree.to_string_lossy(),
                        &checkpoint,
                    ],
                )?;
            } else {
                // nothing retains the recorded head: the work is lost, and
                // the refusal names it rather than silently re-cutting a
                // lineage the record never knew
                return Err(refuse(
                    conn,
                    demand,
                    format!(
                        "t-{} branch {}: the recorded checkpoint {} is unreachable; nothing retains the commit, the recorded work is lost. Re-cut a lineage by hand and record it with sac checkpoint t-{}, or drop the task",
                        task.0, branch, checkpoint, task.0
                    ),
                ));
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
    // the run's derived name, executor/task-incarnation: the worker is
    // always a distinct recorded attribution, so self-approval is dead
    let ordinal = world
        .incarnations
        .values()
        .filter(|run| run.task_id == task)
        .count()
        + 1;
    let session_actor = ActorName::new(format!("{}/t-{}-{}", actor.as_str(), task.0, ordinal))
        .map_err(|e| RunnerFail::Usage(format!("session attribution: {e:?}")))?;
    let (bound, _) = db::record(
        conn,
        &system,
        Command::BindIncarnation {
            task_id: task,
            response_target: demand,
            trigger: demand.0,
            actor: session_actor.clone(),
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

    compose_agent_dir(&agent_dir)?;

    Ok(PreparedRun {
        task,
        demand,
        incarnation,
        worktree,
        session,
        agent_dir,
        actor: session_actor,
    })
}

/// Pointers only: the worktree is the cwd, the demand and reply door are
/// named; the repo's own skill teaches the verbs.
pub fn pointer_prompt(run: &PreparedRun, sac: &str) -> String {
    prompt_text(run.task, run.demand, &run.actor, sac)
}

/// The one prompt text: the executor is spawned on it, and a wait
/// released on an awaiting prompt carries it to the waiter.
fn prompt_text(task: TaskId, demand: CommentId, actor: &ActorName, sac: &str) -> String {
    format!(
        "Serve tracked demand c-{demand} on task t-{task} of this repository; you are working in its prepared worktree. \
Read it: {sac} show t-{task}. Do the work in this directory. \
Reply when done, as '{actor}': \
{sac} comment '#{demand}' '<your answer>'. \
Let other tasks' runs settle on their own; cancel only what you started \
({sac} cancel t-<task> stops a runaway). \
The .agents/skills/saccade skill in this repo documents the tracker.",
        demand = demand.0.0,
        task = task.0,
        actor = actor.as_str(),
    )
}

/// The executor the runner speaks: the pinned pi, the server its
/// sessions write to, and this binary's path for the session's own CLI
/// calls. The flake bakes the pin; SACCADE_PI overrides for development
/// only.
pub struct Executor {
    pub pi: std::path::PathBuf,
    pub server: String,
    pub sac: std::path::PathBuf,
}

/// The pinned executor's path, or why there is none. The ambient
/// binary is never used.
pub fn resolve_pi() -> Result<std::path::PathBuf, String> {
    if let Some(path) = std::env::var_os("SACCADE_PI") {
        return Ok(std::path::PathBuf::from(path));
    }
    option_env!("SACCADE_PI_PATH")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| {
            "no pinned executor: SACCADE_PI_PATH was not baked at build; \
             set SACCADE_PI for development, or build through the flake"
                .into()
        })
}

/// Spawn the executor on the prompt and block until it settles: a
/// JSONL RPC client over the session's stdio. The prompt goes through
/// the protocol, liveness is the event stream (agent_start, turn
/// events, agent_settled), and the reply stays the run's own sac
/// comment — the exit status is never a success claim. The connection
/// registers for the run's life so steer and cancel reach the session
/// as protocol acts; the pid behind it stays the kill of last resort.
pub fn execute_session(
    run: &PreparedRun,
    prompt: &str,
    runs: &crate::supervisor::LiveRuns,
    executor: &Executor,
) -> Result<bool, RunnerFail> {
    // the materialized ask door compose provisions; its presence is
    // compose's guarantee, so no refusal guards the spawn
    let extension = ask_extension_at(&run.agent_dir);
    let mut child = std::process::Command::new(&executor.pi)
        // not pass through, only what the bind named and the doors the
        // session needs — this server, this binary, this task
        .env_remove("SACCADE_TIER")
        .env_remove("SACCADE_PI")
        .env("SACCADE_ACTOR", run.actor.as_str())
        .env("SACCADE_SERVER", &executor.server)
        .env("SACCADE_SAC", &executor.sac)
        .env("SACCADE_TASK", run.task.0.to_string())
        .env("PI_CODING_AGENT_DIR", &run.agent_dir)
        .arg("--mode")
        .arg("rpc")
        // ambient pi config is not a lever; the ask extension below
        // is the whole surface
        .arg("--no-extensions")
        .arg("--no-skills")
        .arg("--no-prompt-templates")
        .arg("--session")
        .arg(&run.session)
        .arg("-a")
        .arg("--extension")
        .arg(&extension)
        .current_dir(&run.worktree)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| RunnerFail::Git(format!("spawning the executor failed: {e}")))?;
    let stdin = child.stdin.take().expect("stdin was piped");
    let handle = RunHandle::new(child.id(), stdin);
    runs.register(run.incarnation, handle.clone());
    handle.send(&ClientCommand::Prompt {
        message: prompt.to_string(),
    });

    // the read loop: events until the session settles or its stream
    // ends. A refused prompt is the one event the runner answers.
    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut prompt_error: Option<String> = None;
    use std::io::Read;
    loop {
        let n = stdout
            .read(&mut chunk)
            .map_err(|e| RunnerFail::Git(format!("reading the executor stream failed: {e}")))?;
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..n]);
        for line in frames(&mut buffer) {
            match ServerEvent::parse(&line) {
                ServerEvent::Response {
                    command,
                    success: false,
                    error,
                } if command == "prompt" => {
                    prompt_error = Some(error.unwrap_or_else(|| "prompt refused".into()));
                }
                ServerEvent::AgentSettled => {
                    // settled work: the session ends at EOF, which the
                    // closed stdin delivers
                    handle.close();
                }
                _ => {}
            }
        }
        if prompt_error.is_some() {
            break;
        }
    }
    if let Some(reason) = prompt_error {
        runs.unregister(run.incarnation);
        handle.close();
        let _ = child.wait();
        return Err(RunnerFail::Usage(format!(
            "the executor refused the prompt: {reason}"
        )));
    }
    let clean = child
        .wait()
        .map(|s| s.success())
        .map_err(|e| RunnerFail::Git(format!("waiting on the executor failed: {e}")))?;
    runs.unregister(run.incarnation);
    Ok(clean)
}

/// Where a settle finds the run's work. The worktree's HEAD is the
/// truth while the worktree lives; a worktree deleted under the
/// deletion law settles at the recorded checkpoint, and nothing
/// surviving even that is a loss the settle names. Absence never
/// refuses.
enum FoundWork {
    Head(GitCommit),
    Checkpoint(GitCommit),
    Lost(GitCommit),
}

fn found_work(repo_root: &Path, ctx: &TaskContext) -> Result<FoundWork, RunnerFail> {
    let workspace = ctx.workspace.as_ref().expect("a bound run has a workspace");
    match &workspace.worktree {
        WorktreeState::Present(worktree) if worktree.as_path().exists() => {
            let head = git(worktree.as_path(), &["rev-parse", "HEAD"])?;
            Ok(FoundWork::Head(commit(head)?))
        }
        _ => {
            let checkpoint = workspace.checkpoint.clone();
            if commit_exists(repo_root, checkpoint.as_str()).is_some() {
                Ok(FoundWork::Checkpoint(checkpoint))
            } else {
                Ok(FoundWork::Lost(checkpoint))
            }
        }
    }
}

pub fn close(conn: &mut Connection, repo_root: &Path, task: TaskId) -> Result<String, RunnerFail> {
    let world = load_world(conn)?;
    let ctx = task_ctx(&world, task)?;
    let incarnation = ctx
        .active_incarnation
        .ok_or_else(|| RunnerFail::Usage(format!("t-{} has no incarnation to settle", task.0)))?;
    let run = &world.incarnations[&incarnation];
    let demand = run.response_target;
    let reply = match &world.comments[&demand].state {
        CommentState::Demand { response, .. } => match response {
            ResponseState::Responded { reply } => Some(*reply),
            ResponseState::Awaiting => None,
        },
        _ => None,
    };
    let found = found_work(repo_root, ctx)?;

    let system = Context::system();
    let now = db::now_epoch();
    // a gone worktree's finding is the checkpoint the record already
    // names: there is no new head to record
    if let FoundWork::Head(checkpoint) = &found {
        db::record(
            conn,
            &system,
            Command::CheckpointWorkspace {
                task_id: task,
                checkpoint: checkpoint.clone(),
            },
            now,
        )?;
    }
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

    let said = match &found {
        FoundWork::Head(checkpoint) => {
            format!("settled i-{} at {}", incarnation.0.0, checkpoint.as_str())
        }
        FoundWork::Checkpoint(checkpoint) => format!(
            "settled i-{} at {}; the worktree is gone, the recorded checkpoint is the finding",
            incarnation.0.0,
            checkpoint.as_str()
        ),
        FoundWork::Lost(checkpoint) => format!(
            "settled i-{}; the worktree is gone and nothing retains the recorded checkpoint {}: no recorded work survives",
            incarnation.0.0,
            checkpoint.as_str()
        ),
    };
    Ok(match reply {
        Some(reply) => format!("{said}; produced c-{}", reply.0.0),
        None => format!("{said}; the demand went unanswered"),
    })
}

/// Block until the demand's run asks something of the waiter, never on
/// replies; `deadline` bounds the wait in seconds (None waits forever).
/// Read-only: abandonment costs nothing.
pub fn wait(
    db_path: &Path,
    comment: CommentId,
    deadline: Option<u64>,
) -> Result<String, RunnerFail> {
    let start = std::time::Instant::now();
    loop {
        let conn = db::open_read(db_path).map_err(ExecuteFail::Db)?;
        let world = load_world(&conn)?;
        if let Some(release) = release_of(&world, comment)? {
            return Ok(release);
        }
        if let Some(seconds) = deadline
            && start.elapsed().as_secs() >= seconds
        {
            return Err(RunnerFail::Usage(format!(
                "c-{} saw no release after {seconds}s",
                comment.0.0
            )));
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

/// The demand's release, when a fold fact fires one: the run settled,
/// cancelled, or asked something of the waiter; the demand refused;
/// or the demand was answered with no run behind the answer. A reply
/// alone never releases — a run still working holds the wait. An ask
/// holds its own wait: it releases on its answer, carrying it.
fn release_of(world: &World, comment: CommentId) -> Result<Option<String>, RunnerFail> {
    let ctx = world.comments.get(&comment).ok_or_else(|| {
        RunnerFail::Usage(format!("no comment c-{} in this tracker", comment.0.0))
    })?;
    match &ctx.state {
        CommentState::Note => {
            return Err(RunnerFail::Usage(format!(
                "c-{} is a note; it addresses nobody and will never respond",
                comment.0.0
            )));
        }
        CommentState::Steer { delivery } => {
            return Err(RunnerFail::Usage(format!(
                "c-{} is a steer ({delivery:?}); a steer holds no wait — wait on the task's demand",
                comment.0.0
            )));
        }
        CommentState::Ask { response } => {
            return match response {
                ResponseState::Awaiting => Ok(None),
                ResponseState::Responded { reply } => {
                    let answer = &world.comments[reply];
                    Ok(Some(format!(
                        "c-{}: answered by c-{} ({})\n{}",
                        comment.0.0,
                        reply.0.0,
                        answer.actor.as_str(),
                        answer.comment.body.as_str(),
                    )))
                }
            };
        }
        CommentState::Demand { .. } => {}
    }
    if let Some(refusal) = &ctx.refusal {
        return Ok(Some(format!(
            "c-{}: refused; {}",
            comment.0.0,
            refusal.reason.as_str()
        )));
    }
    // binding consumes the authorization, so a demand names at most one
    // run; the last bound is the run in question
    let bound = world
        .incarnations
        .iter()
        .rfind(|(_, run)| run.response_target == comment);
    if let Some((id, run)) = bound {
        match run.state {
            IncarnationState::Settled => {
                let receipt = match &world.tasks[run.task_id.0].task.state {
                    TaskState::Delivered(receipt) | TaskState::Done(receipt) => {
                        format!("; receipt: {}", receipt.as_str())
                    }
                    _ => format!("; t-{} holds no receipt", run.task_id.0),
                };
                return Ok(Some(format!(
                    "c-{}: i-{} settled{receipt}",
                    comment.0.0, id.0.0
                )));
            }
            IncarnationState::Cancelled => {
                return Ok(Some(format!("c-{}: i-{} cancelled", comment.0.0, id.0.0)));
            }
            IncarnationState::Bound => {
                let sac = std::env::current_exe()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "sac".into());
                return Ok(Some(format!(
                    "c-{}: i-{} raised a prompt that awaits an answer; act, then re-arm sac wait c-{}\n\n{}",
                    comment.0.0,
                    id.0.0,
                    comment.0.0,
                    prompt_text(run.task_id, run.response_target, &run.actor, &sac)
                )));
            }
            // accepted work has not ended; an interrupted run never
            // accepted it: neither asks anything of the waiter yet
            IncarnationState::PromptAccepted => {
                // the run suspends on a question it authored: every
                // waiter on the task releases to answer it
                if let Some((id, ask)) = world.comments.iter().find(|(_, c)| {
                    c.comment.root == run.task_id
                        && c.actor == run.actor
                        && matches!(
                            c.state,
                            CommentState::Ask {
                                response: ResponseState::Awaiting,
                            }
                        )
                }) {
                    let sac = std::env::current_exe()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| "sac".into());
                    return Ok(Some(format!(
                        "c-{}: i-{} asks (c-{}):\n{}\nanswer: {sac} comment '#{}' '<your answer>', then re-arm sac wait c-{}",
                        comment.0.0,
                        id.0.0,
                        id.0.0,
                        ask.comment.body.as_str(),
                        id.0.0,
                        comment.0.0,
                    )));
                }
                return Ok(None);
            }
            IncarnationState::Interrupted => {
                return Ok(None);
            }
        }
    }
    let response = match &ctx.state {
        CommentState::Demand { response, .. } => response,
        CommentState::Note | CommentState::Steer { .. } | CommentState::Ask { .. } => {
            unreachable!("the variant doors returned above")
        }
    };
    match response {
        ResponseState::Responded { reply } => {
            let reply_ctx = &world.comments[reply];
            Ok(Some(format!(
                "c-{}: answered by c-{} ({}); no work is coming",
                comment.0.0,
                reply.0.0,
                reply_ctx.actor.as_str()
            )))
        }
        ResponseState::Awaiting => Ok(None),
    }
}

/// Close a run the server no longer owns, against disk truth: an
/// accepted run settles through close's honest path; a bound run is
/// interrupted, because the lifecycle table forbids settling a run
/// that never accepted.
pub fn close_as_found(
    conn: &mut Connection,
    repo_root: &Path,
    task: TaskId,
) -> Result<String, RunnerFail> {
    let world = load_world(conn)?;
    let Some(incarnation) = task_ctx(&world, task)?.active_incarnation else {
        return Err(RunnerFail::Usage(format!(
            "t-{} has no incarnation to recover",
            task.0
        )));
    };
    match world.incarnations[&incarnation].state.clone() {
        IncarnationState::PromptAccepted => close(conn, repo_root, task),
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
