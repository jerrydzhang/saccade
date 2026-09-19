//! The server's runner: demands fire runs, runs answer demands. The
//! trigger is a pure function of the world a write produced — after any
//! write (and at boot), every pending agent demand on a task with no
//! active incarnation gets a session.

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use tracing::warn;

use crate::api::AppState;
use crate::objects::comment::{CommentId, CommentState, ResponseState};
use crate::objects::task::TaskId;
use crate::runner::{self, PreparedRun, RunnerFail};
use crate::store::World;
use crate::types::actor::ActorName;

/// What the session body is: run to completion, clean exit or not. The
/// reply's presence is the outcome, not the exit status. Production
/// spawns pi; tests bring their own body.
pub type SessionDriver = Arc<dyn Fn(&PreparedRun, &str) -> Result<bool, RunnerFail> + Send + Sync>;

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
                            ..
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
        let clean = match (config.driver)(&prepared, &prompt) {
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
