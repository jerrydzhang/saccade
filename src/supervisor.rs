//! The server's runner: demands fire runs, runs answer demands. The
//! trigger is a pure function of the world a write produced — after any
//! write (and at boot), every pending agent demand on a task with no
//! active incarnation gets a session.

use std::path::PathBuf;

use tracing::{info, warn};

use crate::api::AppState;
use crate::objects::comment::{CommentId, CommentState, ResponseState};
use crate::objects::task::TaskId;
use crate::runner::{self, PreparedRun, RunnerFail};
use crate::store::World;
use crate::types::actor::ActorName;

/// What the session body is: run to completion, clean exit or not. The
/// reply's presence is the outcome, not the exit status. Production
/// spawns pi; tests bring their own body.
pub type SessionDriver =
    std::sync::Arc<dyn Fn(&PreparedRun, &str) -> Result<bool, RunnerFail> + Send + Sync>;

/// Construction-time identity of the server's runner: where the repo is,
/// who sessions run as, and what a session is. Absent means the server
/// writes but never runs — the tests' shape, and the brake nobody needed.
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
            driver: std::sync::Arc::new(runner::execute_session),
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

/// Fire every runnable demand found in the server's current world. Called
/// after each landed write and once at boot; each spawned run sweeps
/// again when it settles, so a demand queued behind an incarnation is
/// picked up the moment the task frees.
pub fn after_write(app: &AppState) {
    let Some(config) = app.runner_config() else {
        return;
    };
    let world = match app.snapshot() {
        Ok(s) => s.world,
        Err(_) => return,
    };
    for demand in runnable_demands(&world) {
        spawn_run(app.clone(), config.clone(), demand);
    }
}

fn spawn_run(app: AppState, config: RunnerConfig, demand: CommentId) {
    std::thread::spawn(move || {
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
        info!(
            incarnation = prepared.incarnation.0.0,
            demand = prepared.demand.0.0,
            "run bound"
        );

        let sac = std::env::current_exe()
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
        let _ = clean;

        let settled = match app.with_conn(|conn| runner::close(conn, prepared.task)) {
            Ok(Ok(note)) => {
                info!(
                    incarnation = prepared.incarnation.0.0,
                    "run settled: {note}"
                );
                true
            }
            Ok(Err(e)) => {
                warn!(
                    incarnation = prepared.incarnation.0.0,
                    "close failed; the run stays active for recovery: {e}"
                );
                false
            }
            Err(_) => false,
        };

        if settled {
            after_write(&app);
        }
    });
}
