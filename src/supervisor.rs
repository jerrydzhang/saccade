//! The server's side of a run's whole life: when it starts, what it
//! is, what is alive, how it ends — settle, cancel, or crash.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use tracing::{info, warn};

use crate::api::AppState;
use crate::db;
use crate::objects::comment::{
    AgentAttemptState, CommentId, CommentState, ResponseState, SteerDelivery,
};
use crate::objects::incarnation::{IncarnationId, IncarnationState};
use crate::objects::task::{TaskId, TaskState};
use crate::paths;
use crate::rpc::ClientCommand;
use crate::runner::{self, PreparedRun, RunnerFail};
use crate::store::World;
use crate::types::actor::ActorName;
use crate::types::failure::{FailureCode, FailureEvidence};
use crate::{Command, Context};

/// What the session body is: run to completion, clean exit or not. The
/// reply's presence is the outcome, not the exit status. The last
/// parameter is the recording door — one system command into the log
/// at the moment the session's own truth makes it land, so acceptance
/// records at the executor's word, never ahead of it.
pub type SessionDriver = Arc<
    dyn Fn(
            &PreparedRun,
            &str,
            &LiveRuns,
            &dyn Fn(Command) -> Result<(), String>,
        ) -> Result<bool, RunnerFail>
        + Send
        + Sync,
>;

/// One live session's connection: the protocol handle for commands,
/// the pid for the last resort. The stdin behind it is the session's
/// only command door — dropping it closes the session (EOF).
#[derive(Clone)]
pub struct RunHandle {
    pid: u32,
    stdin: Arc<Mutex<Option<std::process::ChildStdin>>>,
    abort_sent: Arc<AtomicBool>,
}

impl RunHandle {
    /// A session the runner cannot command: only its pid is known.
    pub fn process_only(pid: u32) -> Self {
        RunHandle {
            pid,
            stdin: Arc::new(Mutex::new(None)),
            abort_sent: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn new(pid: u32, stdin: std::process::ChildStdin) -> Self {
        RunHandle {
            pid,
            stdin: Arc::new(Mutex::new(Some(stdin))),
            abort_sent: Arc::new(AtomicBool::new(false)),
        }
    }

    fn has_protocol(&self) -> bool {
        self.stdin
            .lock()
            .expect("the run registry is not poisoned")
            .is_some()
    }

    /// Write one command frame. False when the session's stdin is
    /// already gone — the session ended.
    pub(crate) fn send(&self, command: &ClientCommand) -> bool {
        use std::io::Write;
        let mut guard = self.stdin.lock().expect("the run registry is not poisoned");
        match guard.as_mut() {
            Some(stdin) => stdin
                .write_all(command.frame().as_bytes())
                .and_then(|_| stdin.flush())
                .is_ok(),
            None => false,
        }
    }

    /// Close the session's command door: EOF ends the process.
    pub(crate) fn close(&self) {
        self.stdin
            .lock()
            .expect("the run registry is not poisoned")
            .take();
    }

    /// The protocol act of cancel: one abort, once. False when there is
    /// no protocol to abort through or the abort already went out —
    /// the caller escalates to the kill.
    fn abort(&self) -> bool {
        if !self.has_protocol() || self.abort_sent.swap(true, Ordering::Relaxed) {
            return false;
        }
        self.send(&ClientCommand::Abort)
    }
}

/// The sessions this server spawned, by the incarnation they serve —
/// connection state, never record: outcomes live in the log, processes
/// live here.
#[derive(Clone, Default)]
pub struct LiveRuns(Arc<Mutex<HashMap<IncarnationId, RunHandle>>>);

impl LiveRuns {
    pub fn register(&self, id: IncarnationId, handle: RunHandle) {
        self.0
            .lock()
            .expect("the run registry is not poisoned")
            .insert(id, handle);
    }

    pub fn unregister(&self, id: IncarnationId) {
        self.0
            .lock()
            .expect("the run registry is not poisoned")
            .remove(&id);
    }

    pub fn is_empty(&self) -> bool {
        self.0
            .lock()
            .expect("the run registry is not poisoned")
            .is_empty()
    }

    pub fn ids(&self) -> Vec<IncarnationId> {
        self.0
            .lock()
            .expect("the run registry is not poisoned")
            .keys()
            .copied()
            .collect()
    }

    fn handle(&self, id: IncarnationId) -> Option<RunHandle> {
        self.0
            .lock()
            .expect("the run registry is not poisoned")
            .get(&id)
            .cloned()
    }

    /// Deliver a steer to the live session. False when no session owns
    /// the incarnation or its door already closed.
    pub fn steer(&self, id: IncarnationId, message: &str) -> bool {
        self.handle(id).is_some_and(|h| {
            h.send(&ClientCommand::Steer {
                message: message.into(),
            })
        })
    }

    /// The protocol half of a cancel: abort through the connection.
    /// None means no live session owns the incarnation — the run is an
    /// orphan the fold already terminalized.
    pub fn abort(&self, id: IncarnationId) -> Option<bool> {
        self.handle(id).map(|h| h.abort())
    }

    /// TERM the session's process. The last resort: the protocol took
    /// its abort, or there is no protocol to take one. std has no
    /// kill(2) without libc, so kill(1) carries the signal.
    pub fn kill(&self, id: IncarnationId) -> Option<bool> {
        let pid = self.handle(id)?.pid;
        Some(
            std::process::Command::new("kill")
                .arg(pid.to_string())
                .status()
                .map(|s| s.success())
                .unwrap_or(false),
        )
    }

    pub fn kill_all(&self) {
        for id in self.ids() {
            self.kill(id);
        }
    }
}

/// Where the repo is, who sessions run as, and what a session is. Absent
/// means this server writes but never runs.
#[derive(Clone)]
pub struct RunnerConfig {
    pub repo_root: PathBuf,
    pub actor: ActorName,
    pub driver: SessionDriver,
}

impl RunnerConfig {
    /// A server that runs what it is asked: demands fire RPC sessions
    /// against the pinned executor. None when the pin resolves to
    /// nothing — this server writes but never runs.
    pub fn serving(repo_root: PathBuf, actor: ActorName, server_url: &str) -> Option<Self> {
        let pi = runner::resolve_pi().ok()?;
        let sac = std::env::current_exe().unwrap_or_else(|_| "sac".into());
        let executor = runner::Executor {
            pi,
            server: server_url.to_string(),
            sac,
        };
        Some(RunnerConfig {
            repo_root,
            actor,
            driver: Arc::new(move |run, prompt, runs, record| {
                runner::execute_session(run, prompt, runs, &executor, record)
            }),
        })
    }
}

/// The oldest pending agent demand on each incarnation-free task.
pub fn runnable_demands(world: &World) -> Vec<CommentId> {
    let mut demands = Vec::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        if ctx.active_incarnation.is_some() {
            continue;
        }
        // dropped stays terminal: a demand on a dropped task never fires
        if matches!(ctx.task.state, TaskState::Dropped) {
            continue;
        }
        let task = TaskId(i);
        let oldest = world
            .comments
            .iter()
            .find(|(_, c)| {
                c.comment.root == task
                    && matches!(
                        &c.state,
                        CommentState::Demand {
                            response: ResponseState::Awaiting,
                            // only live authorization fires: a spent
                            // attempt on an unanswered demand is dead —
                            // re-asking is a new comment
                            attempt: AgentAttemptState::Authorized { .. },
                        }
                    )
            })
            .map(|(id, _)| *id);
        if let Some(demand) = oldest {
            demands.push(demand);
        }
    }
    demands
}

/// Close every run the world says is active but this server does not
/// own: the boot reconciliation. Settle-as-found — the worktree's HEAD
/// is the truth while the worktree lives; a worktree deleted under the
/// deletion law settles at the recorded checkpoint, and a checkpoint
/// nothing retains settles naming the loss.
pub fn recover(app: &AppState, repo_root: &std::path::Path) {
    let orphans = match app.snapshot() {
        Ok(s) => s
            .world
            .tasks
            .iter()
            .enumerate()
            .filter_map(|(i, ctx)| ctx.active_incarnation.map(|_| TaskId(i)))
            .collect::<Vec<_>>(),
        Err(degraded) => {
            warn!("no recovery: {}", degraded.reason);
            return;
        }
    };
    for task in orphans {
        match app.with_conn(|conn| runner::close_as_found(conn, repo_root, task)) {
            Ok(Ok(note)) => info!("{note}"),
            Ok(Err(e)) => warn!(task = task.0, "recovery refused, the run stays active: {e}"),
            Err(e) => warn!(task = task.0, "recovery lost the writer: {e}"),
        }
    }
}

/// The sweep reconciles reality with the record: what the record ended
/// dies, what the record demands fires. Called after each landed write,
/// at boot, and when a run settles. The runs are fire-and-forget
/// threads; a write never waits on a session.
pub fn sweep(app: &AppState) {
    let Some(config) = app.runner_config() else {
        return;
    };
    let world = match app.snapshot() {
        Ok(s) => s.world,
        Err(degraded) => {
            warn!("the trigger is quiet: {}", degraded.reason);
            return;
        }
    };
    for id in app.runs().ids() {
        if world
            .incarnations
            .get(&id)
            .is_some_and(|run| run.is_terminal())
        {
            // the record already ended this run: the protocol abort is
            // the first act, the kill the last resort — a session with
            // no protocol, or one that already took its abort, dies now
            match app.runs().abort(id) {
                Some(true) => info!(
                    incarnation = id.0.0,
                    "the record already ended this run; aborted through the protocol"
                ),
                _ => match app.runs().kill(id) {
                    Some(true) => info!(
                        incarnation = id.0.0,
                        "the record already ended this run; killed"
                    ),
                    Some(false) => warn!(
                        incarnation = id.0.0,
                        "the record ended this run but the kill failed; the fold is terminal, the process is not"
                    ),
                    None => {}
                },
            }
        }
    }
    for (i, ctx) in world.tasks.iter().enumerate() {
        // a terminal task never runs again: what its executor wrote at
        // runtime moves to the retention root, then the composed agent
        // dir has no consumer left
        if matches!(ctx.task.state, TaskState::Done(_) | TaskState::Dropped)
            && ctx.active_incarnation.is_none()
        {
            if let Err(e) = retain_artifacts(&config.repo_root, i) {
                warn!(task = i, "session artifact retention failed: {e}");
            }
            if let Err(e) = std::fs::remove_dir_all(paths::agent_dir_at(&config.repo_root, i))
                && e.kind() != std::io::ErrorKind::NotFound
            {
                warn!(task = i, "agent dir sweep failed: {e}");
            }
        }
    }
    for demand in runnable_demands(&world) {
        spawn_run(app.clone(), config.clone(), demand);
    }
}

/// The composed surface of an agent dir: links into the operator's
/// credentials, the mirrored settings, and the materialized ask door.
/// Everything else in the dir is the executor's own runtime output —
/// session artifacts that outlive the run through retention.
const COMPOSED_SURFACE: [&str; 5] = [
    "auth.json",
    "models.json",
    "models-store.json",
    "settings.json",
    "ask.ts",
];

/// Move a terminal task's session artifacts from its agent dir to the
/// retention root before the dir dies: post-hoc joins over them stay
/// possible after completion. Links never move — retention is for
/// content, not pointers into the operator's machine. A reopened
/// task's second terminal pass replaces the first: the newest life's
/// artifacts win.
fn retain_artifacts(repo_root: &std::path::Path, task: usize) -> std::io::Result<()> {
    let agent_dir = paths::agent_dir_at(repo_root, task);
    let entries = match std::fs::read_dir(&agent_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_symlink() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if COMPOSED_SURFACE.contains(&name.as_str()) {
            continue;
        }
        let to = paths::retention_at(repo_root, task).join(&name);
        std::fs::create_dir_all(to.parent().expect("retention paths name a file"))?;
        if let Ok(meta) = to.symlink_metadata() {
            if meta.is_dir() {
                std::fs::remove_dir_all(&to)?;
            } else {
                std::fs::remove_file(&to)?;
            }
        }
        std::fs::rename(entry.path(), &to)?;
    }
    Ok(())
}

fn spawn_run(app: AppState, config: RunnerConfig, demand: CommentId) {
    thread::spawn(move || {
        let prepared = match app.with_conn(|conn| {
            runner::prepare(conn, &config.repo_root, demand, config.actor.clone())
        }) {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => {
                warn!(demand = demand.0.0, "run refused at prepare: {e}");
                return;
            }
            Err(_) => return,
        };

        // standing steers reach the session once it lives: the watcher
        // polls read-only until the run's thread ends it
        let watch = SteerWatch::start(&app, prepared.task, prepared.incarnation);

        let sac = env::current_exe()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "sac".into());
        let prompt = runner::pointer_prompt(&prepared, &sac);
        // the recording door: the driver writes the machinery's own
        // facts through the sole writer, at the moment they land
        let door = |command: Command| -> Result<(), String> {
            app.with_conn(|conn| db::record(conn, &Context::system(), command, db::now_epoch()))
                .and_then(|said| said.map(|_| ()).map_err(|e| e.to_string()))
        };
        let clean = match (config.driver)(&prepared, &prompt, &app.runs(), &door) {
            Ok(clean) => clean,
            Err(e) => {
                if reject_unaccepted(&app, prepared.incarnation, &e) {
                    watch.stop();
                    // the rejection freed the task; whatever queued
                    // behind it fires now
                    sweep(&app);
                    return;
                }
                warn!(
                    incarnation = prepared.incarnation.0.0,
                    "session driver failed: {e}"
                );
                false
            }
        };
        watch.stop();
        if !clean {
            warn!(
                incarnation = prepared.incarnation.0.0,
                "executor exited uncleanly; the reply's presence is the outcome"
            );
        }

        // a cancel that landed while the session ran already closed the
        // record; the run thread's close would be refused noise
        let still_active = app
            .snapshot()
            .map(|s| {
                s.world.tasks[prepared.task.0].active_incarnation == Some(prepared.incarnation)
            })
            .unwrap_or(true);
        if !still_active {
            return;
        }

        let settled =
            match app.with_conn(|conn| runner::close(conn, &config.repo_root, prepared.task)) {
                Ok(Ok(_)) => true,
                Ok(Err(e)) => {
                    warn!(
                        incarnation = prepared.incarnation.0.0,
                        "close failed; the run stays active for recovery: {e}"
                    );
                    false
                }
                Err(_) => false,
            };

        // the settle freed the task; whatever queued behind it fires now
        if settled {
            sweep(&app);
        }
    });
}

/// A driver failure while the run never accepted its prompt is a
/// stillbirth: it terminalizes as a recorded rejection carrying the
/// cause, so the supervising wait releases on a fold fact instead of
/// holding on a run that will never answer. The record lands before
/// any warn names it — the refuse() ordering. A failure after
/// acceptance returns false: that run keeps the settle shape, work
/// may have happened, and the reply's presence is the outcome.
fn reject_unaccepted(app: &AppState, incarnation: IncarnationId, failure: &RunnerFail) -> bool {
    let bound = app
        .snapshot()
        .map(|s| {
            s.world
                .incarnations
                .get(&incarnation)
                .is_some_and(|run| run.state == IncarnationState::Bound)
        })
        .unwrap_or(false);
    if !bound {
        return false;
    }
    let said = app.with_conn(|conn| {
        db::record(
            conn,
            &Context::system(),
            Command::RejectPrompt {
                id: incarnation,
                evidence: FailureEvidence::new(
                    FailureCode::PromptRejected,
                    Some(failure.to_string()),
                ),
            },
            db::now_epoch(),
        )
    });
    match said {
        Ok(Ok(_)) => true,
        Ok(Err(e)) => {
            warn!(
                incarnation = incarnation.0.0,
                "the rejection fact did not land: {e}"
            );
            false
        }
        Err(e) => {
            warn!(
                incarnation = incarnation.0.0,
                "the rejection fact did not land: {e}"
            );
            false
        }
    }
}

/// The steer watcher: standing intent on the run's task reaches the
/// live session. Wait-shaped — a read-only poll, abandonment costs
/// nothing — because no write trigger need fire between a run's bind
/// and its first comment.
struct SteerWatch {
    stop: Arc<AtomicBool>,
}

impl SteerWatch {
    fn start(app: &AppState, task: TaskId, incarnation: IncarnationId) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let app = app.clone();
        thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                deliver_standing_steers(&app, task, incarnation);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
        SteerWatch { stop }
    }

    fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// One pass: every standing steer on the task, delivered to the
/// session and recorded as its consumption. The send precedes the
/// record — a crash between them may duplicate a delivery, never
/// lose the intent.
fn deliver_standing_steers(app: &AppState, task: TaskId, incarnation: IncarnationId) {
    let Ok(snapshot) = app.snapshot() else {
        return;
    };
    for (id, c) in &snapshot.world.comments {
        let body = match (&c.state, c.comment.root) {
            (
                CommentState::Steer {
                    delivery: SteerDelivery::Standing,
                },
                root,
            ) if root == task => c.comment.body.as_str().to_string(),
            _ => continue,
        };
        // no live session behind the handle: the steer stands for the
        // next run, unconsumed
        if !app.runs().steer(incarnation, &body) {
            continue;
        }
        let recorded = app.with_conn(|conn| {
            db::record(
                conn,
                &Context::system(),
                Command::ForwardSteer { steer: *id },
                db::now_epoch(),
            )
        });
        if let Err(e) = recorded {
            warn!(steer = id.0.0, "the forward fact did not land: {e}");
        }
    }
}
