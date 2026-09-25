//! The versioned command surface. System authorship is unrepresentable
//! here: the wire tier admits only human and agent, so no request can
//! present the machinery's author.

use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ActorName;
use crate::attempts::{AsReceived, Attempts, Outcome};
use crate::db::{self, ExecuteFail, StoredRecord};
use crate::store::{Context, RecordId, Tier, World};
use crate::supervisor;
use crate::web;
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
    attempts: Attempts,
    /// The artifact store the bytes door serves from; a server that
    /// was not told a repo root has none.
    artifacts: Option<std::path::PathBuf>,
    /// The serving repo's name, when boot resolved a root; the
    /// console's chrome names its instance by it.
    repo_name: Option<web::RepoName>,
}

pub struct Snapshot {
    pub world: World,
    pub rows: Vec<StoredRecord>,
}

/// Degraded still serves the raw rows: the stream is the log's face when
/// the world will not fold.
#[derive(Debug)]
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
            attempts: Attempts::beside(db_path),
            artifacts: None,
            repo_name: None,
        })
    }

    /// A server that runs what it is asked: demands fire sessions.
    pub fn with_runner(db_path: &Path, runner: supervisor::RunnerConfig) -> Result<Self, String> {
        let artifacts = Some(crate::paths::artifacts_at(&runner.repo_root));
        let app = Self::open(db_path)?;
        Ok(AppState {
            runner: Some(runner),
            artifacts,
            ..app
        })
    }

    /// Name the artifact store this server serves bytes from.
    pub fn with_artifacts(mut self, dir: std::path::PathBuf) -> Self {
        self.artifacts = Some(dir);
        self
    }

    /// Name the serving repo; the console's chrome names its
    /// instance by it.
    pub fn with_repo_name(mut self, name: web::RepoName) -> Self {
        self.repo_name = Some(name);
        self
    }

    /// The serving repo's name, when a serve resolved one.
    pub fn repo_name(&self) -> Option<&web::RepoName> {
        self.repo_name.as_ref()
    }

    /// The artifact store's home, when this server knows one.
    pub fn artifacts_dir(&self) -> Option<&Path> {
        self.artifacts.as_deref()
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

    /// The judging seam: one write attempt, one line in the attempts
    /// log, appended while the writer lock is held so the file's order
    /// is the judgment order. Duration includes the lock wait — the
    /// queueing a request met is part of what the diagnoser wants.
    pub fn execute(
        &self,
        context: &Context,
        command: Command,
        at: Option<u64>,
        received: AsReceived,
    ) -> Result<Vec<StoredRecord>, ExecuteFail> {
        let started = std::time::Instant::now();
        let mut guard = self.inner.lock().expect("the writer lock is not poisoned");
        let cursor = match &*guard {
            ServerState::Ready(inner) => inner.rows.last().map(|r| r.seq),
            ServerState::Degraded(..) => None,
        };
        let outcome = match &mut *guard {
            ServerState::Degraded(reason, _) => Err(ExecuteFail::Degraded(reason.clone())),
            ServerState::Ready(inner) => db::record(
                &mut inner.conn,
                context,
                command,
                at.unwrap_or_else(db::now_epoch),
            )
            .map(|(stored, world)| {
                inner.world = world;
                inner.rows.extend(stored.iter().cloned());
                stored
            }),
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        let code = match &outcome {
            Ok(_) => None,
            Err(ExecuteFail::Reject(reject)) => Some(reject.code()),
            Err(ExecuteFail::Degraded(_)) => Some("degraded"),
            Err(ExecuteFail::Db(_)) => Some("storage"),
        };
        match &outcome {
            Ok(stored) => self.attempts.append(
                Some(context.actor.as_str()),
                received.client.as_deref(),
                duration_ms,
                Outcome::Landed {
                    seqs: &stored.iter().map(|s| s.seq).collect::<Vec<_>>(),
                },
            ),
            Err(_) => self.attempts.append(
                Some(context.actor.as_str()),
                received.client.as_deref(),
                duration_ms,
                Outcome::Refused {
                    code: code.expect("a refusal names its code"),
                    cursor,
                    request: &received.raw,
                },
            ),
        }
        outcome
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
    /// The client binary that built the request, when it names itself;
    /// the attempts log carries it beside the server's own version.
    #[serde(default)]
    pub(crate) client: Option<String>,
}

enum ApiFail {
    Reject(Reject, Option<String>),
    Degraded(String),
    Db(String),
    Malformed(String),
}

impl From<ExecuteFail> for ApiFail {
    fn from(e: ExecuteFail) -> Self {
        match e {
            ExecuteFail::Reject(r) => ApiFail::Reject(r, None),
            ExecuteFail::Degraded(reason) => ApiFail::Degraded(reason),
            ExecuteFail::Db(e) => ApiFail::Db(e.to_string()),
        }
    }
}

impl ApiFail {
    fn parts(self) -> (StatusCode, String, Value) {
        match self {
            ApiFail::Reject(reject, taught) => {
                let code = reject.code().to_string();
                let detail = serde_json::to_value(reject).unwrap_or(Value::Null);
                match taught {
                    Some(text) => (StatusCode::BAD_REQUEST, code, Value::String(text)),
                    None => (StatusCode::BAD_REQUEST, code, detail),
                }
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
            // a refusal that enters no journal: the line is the request's
            // only surviving shape, with no actor to name
            let cursor = app
                .snapshot()
                .ok()
                .and_then(|s| s.rows.last().map(|r| r.seq));
            app.attempts.append(
                None,
                None,
                0,
                Outcome::Refused {
                    code: "malformed_request",
                    cursor,
                    request: &String::from_utf8_lossy(&body),
                },
            );
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
    let command = envelope.command;
    let received = AsReceived {
        client: envelope.client,
        raw: String::from_utf8_lossy(&body).into_owned(),
    };
    match app.execute(&context, command.clone(), envelope.at, received) {
        Ok(stored) => {
            let fired = app.clone();
            tokio::task::spawn_blocking(move || supervisor::sweep(&fired));
            let mut reply = json!({"records": &stored});
            // a create reply names what was born: the id resolved from
            // the post-write fold, matched by the birth record it landed
            if matches!(command, Command::CreateTask { .. }) {
                let birth = stored
                    .iter()
                    .find(|r| r.kind == "task_created")
                    .expect("a create lands its birth record");
                let born = app
                    .snapshot()
                    .expect("a landed write leaves a foldable world")
                    .world
                    .task_born_at(RecordId(birth.seq))
                    .expect("the birth record folds into its task");
                reply["id"] = json!(format!("t-{}", born.0));
            }
            (StatusCode::OK, Json(reply)).into_response()
        }
        Err(e) => {
            let (status, code, detail) = match e {
                // the refusal teaches: the expected format and, where the
                // world knows it, the likely intended target
                ExecuteFail::Reject(reject) => {
                    let taught = app
                        .snapshot()
                        .ok()
                        .and_then(|s| crate::refusals::teach(&s.world, &command, &reject));
                    ApiFail::Reject(reject, taught).parts()
                }
                other => ApiFail::from(other).parts(),
            };
            (
                status,
                Json(json!({"error": {"code": code, "detail": detail}})),
            )
                .into_response()
        }
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/command", post(command))
        .route("/api/v1/version", get(version))
}

/// The handshake surface: which build is serving, nothing more
async fn version() -> Json<serde_json::Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION")}))
}
