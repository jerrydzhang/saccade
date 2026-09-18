//! The versioned command surface: one route, one envelope. The wire body
//! carries who is asking and what they are asking for; the server answers
//! with the stored records or a typed error envelope. System authorship is
//! unrepresentable here — the API tier admits only human and agent, and
//! machinery verbs fail validation under both, exactly as the law requires.

use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{error, info};

use crate::db::{self, ExecuteFail, StoredRecord};
use crate::store::{Context, Tier, World};
use crate::wire;
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

/// The sole writer's state: one connection, one world cache refreshed by
/// refolding after every accepted write.
#[derive(Clone)]
pub struct AppState(Arc<Mutex<ServerState>>);

/// A read-only copy of everything the render paths need.
pub struct Snapshot {
    pub world: World,
    pub rows: Vec<StoredRecord>,
}

/// Why the world will not fold, with the raw rows the stream still serves.
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
        Ok(AppState(Arc::new(Mutex::new(state))))
    }

    /// Execute a command through the sole writer; on success the world
    /// cache refolds so every reader sees the write.
    pub fn execute(
        &self,
        context: &Context,
        command: Command,
    ) -> Result<Vec<StoredRecord>, ExecuteFail> {
        let mut guard = self.0.lock().expect("the writer lock is not poisoned");
        match &mut *guard {
            ServerState::Degraded(reason, _) => Err(ExecuteFail::Degraded(reason.clone())),
            ServerState::Ready(inner) => {
                let stored = db::execute(&mut inner.conn, context, command, db::now_epoch())?;
                let loadout = db::load(&inner.conn)?;
                let db::LoadState::Full(world) = loadout.state else {
                    // a write we just accepted cannot fold back: refuse further writes
                    *guard = ServerState::Degraded(
                        "the world stopped folding after a write".into(),
                        std::mem::replace(
                            &mut inner.conn,
                            rusqlite::Connection::open_in_memory().expect("scratch connection"),
                        ),
                    );
                    return Ok(stored);
                };
                inner.world = world;
                inner.rows = loadout.rows;
                Ok(stored)
            }
        }
    }

    /// The current world and raw rows for read paths.
    pub fn snapshot(&self) -> Result<Snapshot, Degraded> {
        let guard = self.0.lock().expect("the writer lock is not poisoned");
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

#[derive(Deserialize)]
struct WireContext {
    actor: crate::ActorName,
    tier: WireTier,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireTier {
    Human,
    Agent,
}

/// The request envelope: one route, context plus command.
#[derive(Deserialize)]
pub struct Envelope {
    context: WireContext,
    command: Command,
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

fn record_json(row: &StoredRecord) -> Value {
    json!({
        "seq": row.seq,
        "et": row.event_time,
        "lt": row.logged_time,
        "actor": row.actor,
        "tier": row.tier,
        "kind": row.kind,
        "payload": serde_json::from_str::<Value>(&row.payload)
            .unwrap_or(Value::String(row.payload.clone())),
    })
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
    match app.execute(&context, envelope.command) {
        Ok(stored) => {
            info!(
                actor = %context.actor.as_str(),
                tier = wire::tier_of(&context.tier),
                last = stored.last().map(|r| r.seq).unwrap_or(0),
                "command accepted"
            );
            (
                StatusCode::OK,
                Json(json!({"records": stored.iter().map(record_json).collect::<Vec<_>>()})),
            )
                .into_response()
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

/// The /api/v1 routes, mountable behind the shared writer state.
pub fn routes() -> Router<AppState> {
    Router::new().route("/api/v1/command", post(command))
}
