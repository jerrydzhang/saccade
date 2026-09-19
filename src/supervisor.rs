//! The server's runner: demands fire runs, runs answer demands. The
//! trigger is a pure function of the world a write produced — after any
//! write (and at boot), every pending agent demand on a task with no
//! active incarnation gets a session.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use tracing::{info, warn};

use crate::api::AppState;
use crate::db::StoredRecord;
use crate::events::Event;
use crate::objects::comment::{AgentAttemptState, CommentId, CommentState, ResponseState};
use crate::objects::incarnation::IncarnationId;
use crate::objects::task::TaskId;
use crate::runner::{self, PreparedRun, RunnerFail};
use crate::store::World;
use crate::types::actor::ActorName;

/// What the session body is: run to completion, clean exit or not. The
/// reply's presence is the outcome, not the exit status. Production
/// spawns pi and registers its pid; tests bring their own body and
/// usually ignore the registry.
pub type SessionDriver =
    Arc<dyn Fn(&PreparedRun, &str, &LiveRuns) -> Result<bool, RunnerFail> + Send + Sync>;

/// The pids of sessions this server spawned, by the incarnation they
/// serve. Operational state, never record: outcomes live in the log,
/// processes live here.
#[derive(Clone, Default)]
pub struct LiveRuns(Arc<Mutex<HashMap<IncarnationId, u32>>>);

impl LiveRuns {
    pub fn register(&self, id: IncarnationId, pid: u32) {
        self.0
            .lock()
            .expect("the run registry is not poisoned")
            .insert(id, pid);
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

    /// TERM the session's process. The run thread's wait observes the
    /// exit and closes; std has no kill(2) without libc, so kill(1)
    /// carries the signal. None means no live session owns the
    /// incarnation — the run is an orphan the fold already terminalized.
    pub fn kill(&self, id: IncarnationId) -> Option<bool> {
        let pid = self
            .0
            .lock()
            .expect("the run registry is not poisoned")
            .get(&id)
            .copied();
        let pid = pid?;
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
    pub fn serving(repo_root: PathBuf, actor: ActorName) -> Self {
        RunnerConfig {
            repo_root,
            actor,
            driver: Arc::new(runner::execute_session),
        }
    }
}

/// The oldest pending agent demand on each incarnation-free task.
pub fn runnable_demands(world: &World) -> Vec<CommentId> {
    let mut demands = Vec::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        if ctx.active_incarnation.is_some() {
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
                        CommentState::AddressedToAgent {
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
/// own: the boot reconciliation. Settle-as-found — the worktree's
/// actual HEAD is the truth, never a guess about a dead process.
/// Called once at serve start, before the boot scan, so freed tasks
/// can fire their queued demands.
pub fn recover(app: &AppState) {
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
        match app.with_conn(|conn| runner::close_as_found(conn, task)) {
            Ok(Ok(note)) => info!("{note}"),
            Ok(Err(e)) => warn!(task = task.0, "recovery refused, the run stays active: {e}"),
            Err(e) => warn!(task = task.0, "recovery lost the writer: {e}"),
        }
    }
}

/// The server's reaction to records it just wrote: cancelled runs die,
/// then the sweep fires what the new world allows.
pub fn react(app: &AppState, records: &[StoredRecord]) {
    for stored in records {
        if stored.kind != "incarnation_cancelled" {
            continue;
        }
        if let Ok(Event::IncarnationCancelled { id }) =
            crate::wire::assemble(&stored.kind, &stored.payload)
        {
            match app.runs().kill(id) {
                Some(true) => info!(incarnation = id.0.0, "cancelled run killed"),
                Some(false) => warn!(
                    incarnation = id.0.0,
                    "cancel landed but the kill failed; the fold is terminal, the process is not"
                ),
                None => {}
            }
        }
    }
    sweep(app);
}

/// Start a run for every demand the world says should be running —
/// called after each landed write, at boot, and when a run settles. The
/// runs are fire-and-forget threads; a write never waits on a session.
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
    for demand in runnable_demands(&world) {
        spawn_run(app.clone(), config.clone(), demand);
    }
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

        let sac = env::current_exe()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "sac".into());
        let prompt = runner::pointer_prompt(&prepared, &sac);
        let clean = match (config.driver)(&prepared, &prompt, &app.runs()) {
            Ok(clean) => clean,
            Err(e) => {
                warn!(
                    incarnation = prepared.incarnation.0.0,
                    "session driver failed: {e}"
                );
                false
            }
        };
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

        let settled = match app.with_conn(|conn| runner::close(conn, prepared.task)) {
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
