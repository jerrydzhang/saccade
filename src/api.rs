//! The versioned command surface. System authorship is unrepresentable
//! here: the wire tier admits only human and agent, so no request can
//! present the machinery's author.

use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::error;

use crate::ActorName;
use crate::db::{self, ExecuteFail, StoredRecord};
use crate::store::{Context, Tier, World};
use crate::supervisor;
use crate::{Command, Reject};

enum ServerState {
    Ready(Inner),
    Degraded(String, rusqlite::Connection),
}

struct Inner {
    conn: rusqlite::Connection,
    world: World,
    rows: Vec<StoredRecord>,
}

/// The sole writer's state: writes refold rather than apply incrementally,
/// so one definition of truth serves readers and writers alike. The runner
/// config rides outside the lock: immutable, and absent when this server
/// only writes.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<ServerState>>,
    runner: Option<supervisor::RunnerConfig>,
    runs: supervisor::LiveRuns,
}

pub struct Snapshot {
    pub world: World,
    pub rows: Vec<StoredRecord>,
}

/// Degraded still serves the raw rows: the stream is the log's face when
/// the world will not fold.
pub struct Degraded {
    pub reason: String,
    pub rows: Vec<StoredRecord>,
}

impl AppState {
    pub fn open(db_path: &Path) -> Result<Self, String> {
        let conn = db::open(db_path).map_err(|e| format!("database: {e}"))?;
        let loadout = db::load(&conn).map_err(|e| format!("database: {e}"))?;
        let state = match loadout.state {
            db::LoadState::Full(world) => ServerState::Ready(Inner {
                conn,
                world,
                rows: loadout.rows,
            }),
            db::LoadState::Degraded(reason) => ServerState::Degraded(reason, conn),
        };
        Ok(AppState {
            inner: Arc::new(Mutex::new(state)),
            runner: None,
            runs: supervisor::LiveRuns::default(),
        })
    }

    /// A server that runs what it is asked: demands fire sessions.
    pub fn with_runner(db_path: &Path, runner: supervisor::RunnerConfig) -> Result<Self, String> {
        let app = Self::open(db_path)?;
        Ok(AppState {
            runner: Some(runner),
            ..app
        })
    }

    pub fn runner_config(&self) -> Option<&supervisor::RunnerConfig> {
        self.runner.as_ref()
    }

    /// The live sessions this server owns; empty for a server that
    /// never runs.
    pub fn runs(&self) -> supervisor::LiveRuns {
        self.runs.clone()
    }

    /// Run against the sole writer's connection under the lock; blocking
    /// work inside is the caller's discipline. Whatever the closure wrote
    /// becomes the cache: readers never see a world the log contradicts.
    pub fn with_conn<T>(
        &self,
        f: impl FnOnce(&mut rusqlite::Connection) -> T,
    ) -> Result<T, String> {
        let mut guard = self.inner.lock().expect("the writer lock is not poisoned");
        match &mut *guard {
            ServerState::Ready(inner) => {
                let out = f(&mut inner.conn);
                let loadout = db::load(&inner.conn).map_err(|e| e.to_string())?;
                match loadout.state {
                    db::LoadState::Full(world) => {
                        inner.world = world;
                        inner.rows = loadout.rows;
                    }
                    db::LoadState::Degraded(reason) => return Err(reason),
                }
                Ok(out)
            }
            ServerState::Degraded(reason, _) => Err(reason.clone()),
        }
    }

    pub fn execute(
        &self,
        context: &Context,
        command: Command,
        at: Option<u64>,
    ) -> Result<Vec<StoredRecord>, ExecuteFail> {
        let mut guard = self.inner.lock().expect("the writer lock is not poisoned");
        match &mut *guard {
            ServerState::Degraded(reason, _) => Err(ExecuteFail::Degraded(reason.clone())),
            ServerState::Ready(inner) => {
                let (stored, world) = db::record(
                    &mut inner.conn,
                    context,
                    command,
                    at.unwrap_or_else(db::now_epoch),
                )?;
                inner.world = world;
                inner.rows.extend(stored.iter().cloned());
                Ok(stored)
            }
        }
    }

    pub fn snapshot(&self) -> Result<Snapshot, Degraded> {
        let guard = self.inner.lock().expect("the writer lock is not poisoned");
        match &*guard {
            ServerState::Ready(inner) => Ok(Snapshot {
                world: inner.world.clone(),
                rows: inner.rows.clone(),
            }),
            ServerState::Degraded(reason, conn) => Err(Degraded {
                reason: reason.clone(),
                rows: db::load(conn).map(|l| l.rows).unwrap_or_default(),
            }),
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireTier {
    Human,
    Agent,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct WireContext {
    pub actor: ActorName,
    pub tier: WireTier,
}

#[derive(Serialize, Deserialize)]
pub struct Envelope {
    pub(crate) context: WireContext,
    pub(crate) command: Command,
    pub(crate) at: Option<u64>,
}

enum ApiFail {
    Reject(Reject),
    Degraded(String),
    Db(String),
    Malformed(String),
}

impl From<ExecuteFail> for ApiFail {
    fn from(e: ExecuteFail) -> Self {
        match e {
            ExecuteFail::Reject(r) => ApiFail::Reject(r),
            ExecuteFail::Degraded(reason) => ApiFail::Degraded(reason),
            ExecuteFail::Db(e) => ApiFail::Db(e.to_string()),
        }
    }
}

impl ApiFail {
    fn parts(self) -> (StatusCode, String, Value) {
        match self {
            ApiFail::Reject(reject) => {
                let detail = serde_json::to_value(&reject).unwrap_or(Value::Null);
                let code = detail.as_str().unwrap_or("rejected").to_string();
                (StatusCode::BAD_REQUEST, code, detail)
            }
            ApiFail::Degraded(reason) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "degraded".into(),
                Value::String(reason),
            ),
            ApiFail::Db(detail) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage".into(),
                Value::String(detail),
            ),
            ApiFail::Malformed(detail) => (
                StatusCode::BAD_REQUEST,
                "malformed_request".into(),
                Value::String(detail),
            ),
        }
    }
}

pub async fn command(State(app): State<AppState>, body: Bytes) -> Response {
    let envelope: Envelope = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => {
            let (status, code, detail) = ApiFail::Malformed(e.to_string()).parts();
            return (
                status,
                Json(json!({"error": {"code": code, "detail": detail}})),
            )
                .into_response();
        }
    };
    let context = Context {
        actor: envelope.context.actor,
        tier: match envelope.context.tier {
            WireTier::Human => Tier::Human,
            WireTier::Agent => Tier::Agent,
        },
    };
    match app.execute(&context, envelope.command, envelope.at) {
        Ok(stored) => {
            let fired = app.clone();
            let landed = stored.clone();
            tokio::task::spawn_blocking(move || supervisor::react(&fired, &landed));
            (StatusCode::OK, Json(json!({"records": &stored}))).into_response()
        }
        Err(e) => {
            let (status, code, detail) = ApiFail::from(e).parts();
            error!(%code, "command refused");
            (
                status,
                Json(json!({"error": {"code": code, "detail": detail}})),
            )
                .into_response()
        }
    }
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/api/v1/command", post(command))
}
