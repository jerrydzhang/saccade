//! The versioned command surface. System authorship is unrepresentable
//! here: the wire tier admits only human and agent, so no request can
//! present the machinery's author.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ActorName;
use crate::attempts::{AsReceived, Attempts, Outcome};
use crate::db::{self, ExecuteFail, StoredRecord};
use crate::events::Event;
use crate::host::HostFail;
use crate::store::{Context, RecordId, Scope, Tier, World};
use crate::types::pointers::WorktreePath;
use crate::workspace::{self, WorkspaceFail};
use crate::{Command, Reject};

enum ServerState {
    Ready(Inner),
    Degraded(String, rusqlite::Connection),
}

struct Inner {
    conn: rusqlite::Connection,
    /// The tracker's own db: the worktree home derives from it
    db_path: std::path::PathBuf,
}

/// One launch registration: the claiming incarnation's birth seq and
/// public number, held until the token expires.
type LaunchEntry = (usize, usize, std::time::Instant);

/// One runtime association observation: live delivery connections now,
/// and when the association was last seen - the bridge's parked read
/// cycles are the heartbeat. Operational only.
#[derive(Clone, Copy)]
pub struct Association {
    pub connections: usize,
    pub last_seen: Instant,
}

/// The native bridge's operational cache: which incarnations hold a
/// live delivery connection right now, the launch tokens a compose
/// door is waiting on, and the delivery tasks of send attempts now in
/// flight. Runtime facts only - transport state and correlation, never
/// record identity, authority, or obligation; lost on restart and
/// rebuilt by the sessions themselves.
#[derive(Clone, Default)]
pub struct BridgeCache {
    /// wakes parked delivery polls when a record lands or a task is
    /// staged
    notify: Arc<tokio::sync::Notify>,
    /// incarnation birth seq -> runtime association
    live: Arc<std::sync::Mutex<HashMap<usize, Association>>>,
    /// launch token -> (incarnation seq, public number), expiring
    launches: Arc<Mutex<HashMap<String, LaunchEntry>>>,
    /// the send attempts' staged delivery tasks, handed exactly once
    stages: Arc<Mutex<Vec<StagedDelivery>>>,
}

impl BridgeCache {
    /// A record landed: every parked mailbox poll re-checks its
    /// predicate against the new fold.
    pub fn wake(&self) {
        self.notify.notify_waiters();
    }

    /// Remember which incarnation claimed a compose launch token; the
    /// launching door polls this to learn the birth it caused. The
    /// mapping binds once: the same association may refresh, a
    /// different association never retargets - a forked or existing
    /// session reattaching under a stale token changes nothing.
    pub fn remember_launch(&self, token: String, incarnation: usize, num: usize) {
        let mut launches = self.launches.lock().expect("the launch cache holds");
        launches.retain(|_, (_, _, at)| at.elapsed() < std::time::Duration::from_secs(900));
        match launches.get(&token) {
            Some((existing, _, _)) if *existing == incarnation => {
                launches.insert(token, (incarnation, num, Instant::now()));
            }
            Some(_) => {}
            None => {
                launches.insert(token, (incarnation, num, Instant::now()));
            }
        }
    }

    /// The incarnation that registered under a launch token, if the
    /// token is still fresh.
    pub fn launch(&self, token: &str) -> Option<(usize, usize)> {
        let mut launches = self.launches.lock().expect("the launch cache holds");
        launches.retain(|_, (_, _, at)| at.elapsed() < std::time::Duration::from_secs(900));
        launches.get(token).map(|(i, n, _)| (*i, *n))
    }

    /// Stage one delivery task for the recipient's integration: the
    /// send attempt's transport half, woken immediately.
    fn stage_delivery(&self, incarnation: usize, message: usize, token: String) {
        let mut stages = self.stages.lock().expect("the delivery stage holds");
        stages.retain(|task| task.staged_at.elapsed().as_secs() < STAGE_TTL_SECS);
        stages.push(StagedDelivery {
            incarnation,
            message,
            token,
            staged_at: Instant::now(),
        });
    }

    /// Hand every staged task for this incarnation to the transport
    /// read: each attempt's task is taken exactly once, expired stages
    /// pruned on the way.
    fn take_staged(&self, incarnation: usize) -> Vec<(String, usize)> {
        let mut stages = self.stages.lock().expect("the delivery stage holds");
        stages.retain(|task| task.staged_at.elapsed().as_secs() < STAGE_TTL_SECS);
        let mut taken = Vec::new();
        stages.retain(|task| {
            if task.incarnation == incarnation {
                taken.push((task.token.clone(), task.message));
                false
            } else {
                true
            }
        });
        taken
    }

    /// Retire a stage at its attempt's end: true when it was still
    /// here, meaning no transport read ever handed it.
    fn drop_stage(&self, token: &str) -> bool {
        let mut stages = self.stages.lock().expect("the delivery stage holds");
        let before = stages.len();
        stages.retain(|task| task.token != *token);
        before != stages.len()
    }

    /// The incarnations holding at least one delivery connection: the
    /// activation decision's operational input, never a liveness proof.
    pub fn connected(&self) -> Vec<(usize, usize)> {
        let live = self.live.lock().expect("the live cache holds");
        let mut entries: Vec<(usize, usize)> = live
            .iter()
            .filter(|(_, a)| a.connections > 0)
            .map(|(i, a)| (*i, a.connections))
            .collect();
        entries.sort();
        entries
    }

    /// Whether this incarnation's runtime association was observed
    /// within the given window: the bridge's parked read cycles are
    /// the heartbeat, and a transient socket loss stays associated
    /// until the window passes. Not proof of liveness past the window -
    /// past it, activation falls to host evidence.
    pub fn recently_associated(&self, incarnation: usize, window: std::time::Duration) -> bool {
        let live = self.live.lock().expect("the live cache holds");
        live.get(&incarnation)
            .map(|a| a.last_seen.elapsed() <= window)
            .unwrap_or(false)
    }
}

/// Counts one delivery connection for its scope: parked poll or
/// immediate read, the bridge announces itself for the call's duration.
struct BridgeLive {
    live: Arc<std::sync::Mutex<HashMap<usize, Association>>>,
    incarnation: usize,
}

impl BridgeLive {
    fn new(cache: &BridgeCache, incarnation: usize) -> Self {
        let mut live = cache.live.lock().expect("the live cache holds");
        let entry = live.entry(incarnation).or_insert(Association {
            connections: 0,
            last_seen: Instant::now(),
        });
        entry.connections += 1;
        entry.last_seen = Instant::now();
        Self {
            live: Arc::clone(&cache.live),
            incarnation,
        }
    }
}

impl Drop for BridgeLive {
    fn drop(&mut self) {
        let mut live = self.live.lock().expect("the live cache holds");
        if let Some(entry) = live.get_mut(&self.incarnation) {
            entry.connections -= 1;
            // the association outlives the connection: the window a
            // transient socket loss spends inside is held, not silent
            entry.last_seen = Instant::now();
            if entry.connections == 0 {
                // keep last_seen; prune only when the window passed
                let now = Instant::now();
                live.retain(|_, a| {
                    a.connections > 0
                        || now.duration_since(a.last_seen).as_secs() < BRIDGE_ASSOCIATION_SECS
                });
            }
        }
    }
}

/// The sole writer's state: every face loads the log fresh under the
/// one lock, so no read can serve a world the file contradicts.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<ServerState>>,
    db_path: std::path::PathBuf,
    attempts: Attempts,
    /// The native bridge's operational cache: delivery-connection
    /// liveness, launch tokens, and the wake for parked mailbox reads.
    pub bridge: BridgeCache,
    /// The artifact store the bytes door serves from; a server that
    /// was not told a repo root has none.
    artifacts: Option<std::path::PathBuf>,
    /// The serving repo's root, when boot resolved one.
    repo_root: Option<std::path::PathBuf>,
    /// This server's own URL, published at bind: the routing a launched
    /// session needs to reach the tracker it was started from.
    server_url: Option<String>,
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

/// A read that cannot present the world: the log will not fold (its
/// raw rows still travel), or will not even load - corruption and
/// storage failures are database errors, never an empty successful log.
#[derive(Debug)]
pub enum ReadFail {
    Degraded(Degraded),
    Db(String),
}

impl ReadFail {
    pub fn reason(&self) -> &str {
        match self {
            ReadFail::Degraded(d) => &d.reason,
            ReadFail::Db(reason) => reason,
        }
    }
}

impl AppState {
    pub fn open(db_path: &Path) -> Result<Self, String> {
        // the worktree home derives from the db's own directory, so the
        // path is absolute from the start - a fresh tracker resolves
        // through its parent, an existing one canonically
        let canonical = if db_path.exists() {
            std::fs::canonicalize(db_path)
                .map_err(|e| format!("database {}: {e}", db_path.display()))?
        } else {
            let parent = db_path.parent().unwrap_or_else(|| Path::new("."));
            let parent = if parent.as_os_str().is_empty() {
                Path::new(".").to_path_buf()
            } else {
                parent.to_path_buf()
            };
            let canonical_parent = std::fs::canonicalize(&parent)
                .map_err(|e| format!("database {}: {e}", db_path.display()))?;
            canonical_parent.join(db_path.file_name().unwrap_or_default())
        };
        let conn = db::open(&canonical).map_err(|e| format!("database: {e}"))?;
        let state = match db::load(&conn).map_err(|e| format!("database: {e}"))?.state {
            db::LoadState::Full(_) => ServerState::Ready(Inner {
                conn,
                db_path: canonical.clone(),
            }),
            db::LoadState::Degraded(reason) => ServerState::Degraded(reason, conn),
        };
        Ok(AppState {
            inner: Arc::new(Mutex::new(state)),
            db_path: canonical,
            attempts: Attempts::beside(db_path),
            bridge: BridgeCache::default(),
            artifacts: None,
            repo_root: None,
            server_url: None,
        })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Name the artifact store this server serves bytes from.
    pub fn with_artifacts(mut self, dir: std::path::PathBuf) -> Self {
        self.artifacts = Some(dir);
        self
    }

    /// Name the serving repo's root.
    pub fn with_repo_root(mut self, root: std::path::PathBuf) -> Self {
        self.repo_root = Some(root);
        self
    }

    /// Name this server's own published URL.
    pub fn with_server_url(mut self, url: String) -> Self {
        self.server_url = Some(url);
        self
    }

    /// This server's own published URL, when bound.
    pub fn server_url(&self) -> Option<&str> {
        self.server_url.as_deref()
    }

    /// The serving repo's root, when boot resolved one.
    pub fn repo_root(&self) -> Option<&Path> {
        self.repo_root.as_deref()
    }

    /// The artifact store's home, when this server knows one.
    pub fn artifacts_dir(&self) -> Option<&Path> {
        self.artifacts.as_deref()
    }

    /// The judging seam: one write attempt, one line in the attempts
    /// log, appended while the writer lock is held so the file's order
    /// is the judgment order. Duration includes the lock wait - the
    /// queueing a request met is part of what the diagnoser wants.
    pub fn execute(
        &self,
        context: &Context,
        command: Command,
        at: Option<u64>,
        received: AsReceived,
    ) -> Result<Vec<StoredRecord>, ExecuteFail> {
        let started = Instant::now();
        let mut guard = self.inner.lock().expect("the writer lock is not poisoned");
        // the cursor names the log as it stands at judgment, external
        // writes included
        let cursor = match &*guard {
            ServerState::Ready(inner) => Some(db::last_seq(&inner.conn).map_err(ExecuteFail::Db)?),
            ServerState::Degraded(..) => None,
        }
        .flatten();
        let outcome = match &mut *guard {
            ServerState::Degraded(reason, _) => Err(ExecuteFail::Degraded(reason.clone())),
            ServerState::Ready(inner) => db::record(
                &mut inner.conn,
                context,
                command,
                at.unwrap_or_else(db::now_epoch),
            )
            .map(|(stored, _)| stored),
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        let code = match &outcome {
            Ok(_) => None,
            Err(ExecuteFail::Reject(reject)) => Some(reject.code()),
            Err(ExecuteFail::Degraded(_)) => Some("degraded"),
            Err(ExecuteFail::Db(_)) => Some("storage"),
        };
        match &outcome {
            Ok(stored) => {
                // a landed write may be mail a parked delivery poll is
                // waiting on: the landing is the wake, never a timer
                self.bridge.notify.notify_waiters();
                self.attempts.append(
                    Some(context.actor.as_str()),
                    received.client.as_deref(),
                    duration_ms,
                    Outcome::Landed {
                        seqs: &stored.iter().map(|s| s.seq).collect::<Vec<_>>(),
                    },
                );
            }
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

    pub fn snapshot(&self) -> Result<Snapshot, ReadFail> {
        let guard = self.inner.lock().expect("the writer lock is not poisoned");
        let (boot_reason, conn) = match &*guard {
            ServerState::Ready(inner) => (None, &inner.conn),
            ServerState::Degraded(reason, conn) => (Some(reason.clone()), conn),
        };
        let loadout = db::load(conn).map_err(|e| ReadFail::Db(e.to_string()))?;
        match loadout.state {
            db::LoadState::Full(world) => match boot_reason {
                None => Ok(Snapshot {
                    world,
                    rows: loadout.rows,
                }),
                Some(reason) => Err(ReadFail::Degraded(Degraded {
                    reason,
                    rows: loadout.rows,
                })),
            },
            db::LoadState::Degraded(fresh) => Err(ReadFail::Degraded(Degraded {
                reason: boot_reason.unwrap_or(fresh),
                rows: loadout.rows,
            })),
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
    /// The environment-declared scope, when the sending session carries one
    #[serde(default)]
    pub scope: Option<Scope>,
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
    /// An effect-layer refusal: the tracker names its cause
    Workspace(WorkspaceFail),
    /// The serving edge has no repo to run effects against
    NoRepo,
    /// A launch construction or host-submission refusal: the shared
    /// host mechanics name the stage, the door names its code
    Host(HostFail),
    /// A controller refusal with its own status and code: quiescence,
    /// association, or a partial outcome the caller must see
    Refused(StatusCode, &'static str, String),
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
            ApiFail::Workspace(fail) => {
                // infrastructure failures are the server's, never the
                // caller's bad input: unavailable entropy and a
                // directory the OS refused are the effect layer's break
                let status = match fail {
                    WorkspaceFail::Git(_)
                    | WorkspaceFail::Entropy(_)
                    | WorkspaceFail::DirCreate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
                    _ => StatusCode::BAD_REQUEST,
                };
                (status, fail.code().into(), Value::String(fail.detail()))
            }
            ApiFail::NoRepo => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_no_repo".into(),
                Value::String(
                    "the server resolved no repo root at boot; it cannot establish workspaces"
                        .into(),
                ),
            ),
            ApiFail::Host(fail) => {
                // the workspace named is the caller's selection; every
                // other stage is the host environment's, never the
                // caller's bad input
                let status = match fail {
                    HostFail::Workspace(_) => StatusCode::BAD_REQUEST,
                    HostFail::NoPi
                    | HostFail::NoDirenv
                    | HostFail::WorktreeShow(_)
                    | HostFail::CreateSpawn(_)
                    | HostFail::NoHandle => StatusCode::INTERNAL_SERVER_ERROR,
                    HostFail::Undiscovered(_) => StatusCode::GATEWAY_TIMEOUT,
                    HostFail::CreateRefused(_) => StatusCode::BAD_GATEWAY,
                    // lifecycle host answers the server could not read:
                    // the environment's break, never the caller's input
                    HostFail::TerminalList(_)
                    | HostFail::TerminalClose(_)
                    | HostFail::WorkspaceShow(_) => StatusCode::BAD_GATEWAY,
                };
                (
                    status,
                    fail.wire_code().into(),
                    Value::String(fail.to_string()),
                )
            }
            ApiFail::Refused(status, code, detail) => {
                (status, code.to_string(), Value::String(detail))
            }
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
        scope: envelope.context.scope,
    };
    let command = envelope.command;
    let received = AsReceived {
        client: envelope.client,
        raw: String::from_utf8_lossy(&body).into_owned(),
    };
    match app.execute(&context, command.clone(), envelope.at, received) {
        Ok(stored) => {
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
            if let Command::SendMessage { to: to_id, .. } = &command {
                let numbers = app
                    .snapshot()
                    .expect("a landed write leaves a foldable world")
                    .world;
                let sent = stored.first().expect("a message lands its one record").seq;
                let mctx = numbers
                    .messages
                    .get(&crate::MessageId(RecordId(sent)))
                    .expect("the landed message folds in");
                let num = numbers.message_num(crate::MessageId(RecordId(sent)));
                let label = match num {
                    Some(n) => format!("m-{n}"),
                    None => format!("seq {sent}"),
                };
                reply["message"] = json!({
                    "seq": sent,
                    "num": num,
                });
                reply["from"] = json!({
                    "seq": mctx.message.from.0 .0,
                    "num": numbers.incarnation_num(mctx.message.from),
                });
                // the same record-then-attempt as the send face: one
                // mechanism, no entrance left out; the command landed,
                // so the attempt's truth rides the reply as data
                let to_num = numbers.incarnation_num(*to_id).unwrap_or(0);
                reply["submission"] =
                    match attempt_submission(&app, *to_id, to_num, sent, &label).await {
                        Submission::Submitted => json!({"state": "submitted"}),
                        Submission::Refused { code, detail } => {
                            json!({"state": "refused", "code": code, "detail": detail})
                        }
                        Submission::Unacknowledged { detail } => {
                            json!({"state": "unacknowledged", "detail": detail})
                        }
                    };
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
        .route("/api/v1/records", get(records))
        .route("/api/v1/workspace/create", post(workspace_create))
        .route("/api/v1/workspace/change", post(workspace_change))
        .route("/api/v1/workspace/clean", post(workspace_clean))
        .route("/api/v1/workspace/restore", post(workspace_restore))
        .route("/api/v1/workspace/reconcile", post(workspace_reconcile))
        .route("/api/v1/native/attach", post(native_attach))
        .route("/api/v1/native/bridge", get(native_bridge))
        .route("/api/v1/native/launch", get(native_launch))
        .route("/api/v1/messages/send", post(message_send))
        .route("/api/v1/messages/submit", post(message_submit))
        .route("/api/v1/messages", get(messages_history))
        .route("/api/v1/messages/pending", get(messages_pending_retired))
        .route("/api/v1/messages/deliver", get(messages_deliver))
        .route("/api/v1/version", get(version))
        .route("/api/v1/workbench", get(workbench))
        .route("/api/v1/workbench/preview", post(workbench_preview))
        .route("/api/v1/workbench/identity", get(workbench_identity))
        .route("/api/v1/workbench/command", post(workbench_command))
        .route("/api/v1/workbench/launch", post(workbench_launch))
}

// -- the workbench read and act faces --------------------------------------

fn actor_cookie(name: &ActorName) -> header::HeaderValue {
    header::HeaderValue::from_str(&format!(
        "actor={}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax",
        name.as_str()
    ))
    .expect("a valid actor name is a valid cookie value")
}

/// The claimed actor a browser's actor cookie names, when one stands.
fn cookie_actor(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())?
        .split(';')
        .find_map(|pair| {
            pair.trim()
                .strip_prefix("actor=")
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        })
}

/// The workbench's identity read: the cookie-claimed actor when one
/// stands, plus the db-bound legacy storage instance. The prototype
/// keyed drafts by sha256(db path)[:16]; a plain-LAN-HTTP browser
/// cannot recompute that (no crypto.subtle off secure contexts), so
/// the db names its own legacy key here - an exact lookup, never a
/// scan.
async fn workbench_identity(State(app): State<AppState>, headers: HeaderMap) -> Response {
    use sha2::Digest;
    let actor = cookie_actor(&headers).unwrap_or_default();
    let legacy: String = sha2::Sha256::digest(app.db_path.to_string_lossy().as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    (
        StatusCode::OK,
        Json(json!({
            "actor": if actor.is_empty() { Value::Null } else { json!(actor) },
            "legacy_instance": &legacy[..16],
        })),
    )
        .into_response()
}

#[derive(Deserialize)]
struct WorkbenchQuery {
    /// The log position the reader last saw: while the log stands at
    /// it, a waiting read parks instead of returning the same fold
    cursor: Option<usize>,
    /// Bounded park in milliseconds, capped like the mailbox read
    wait_ms: Option<u64>,
}

/// The folded workbench projection: canonical rendered bodies, citation
/// edges, and the cursor, read straight off the store's own fold. A
/// degraded fold refuses honestly; nothing writes. With cursor and
/// wait_ms the read parks while the log stands at the named cursor and
/// returns once the cursor moves past it. The landing wake is the fast
/// path for writes through this server's own door; the park is sliced
/// at one second so the cheap cursor check also surfaces records other
/// processes wrote to the shared database (the CLI face), which fire
/// no wake here. No parameters is the immediate read it always was.
async fn workbench(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<WorkbenchQuery>,
) -> Response {
    let wait_ms = query.wait_ms.unwrap_or(0).min(30_000);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(wait_ms);
    loop {
        // armed before the read so a landing between the read and the
        // park cannot be missed
        let wake = app.bridge.notify.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        let stands_at_cursor = {
            let guard = app.inner.lock().expect("the writer lock is not poisoned");
            match &*guard {
                ServerState::Ready(inner) => {
                    db::last_seq(&inner.conn).ok().flatten() == query.cursor
                }
                ServerState::Degraded(..) => false,
            }
        };
        if !stands_at_cursor || wait_ms == 0 || tokio::time::Instant::now() >= deadline {
            let snapshot = match app.snapshot() {
                Ok(s) => s,
                Err(ReadFail::Degraded(degraded)) => {
                    // the raw rows still travel through the records face; the
                    // workbench projection itself refuses honestly
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(json!({"error": {"code": "degraded", "detail": degraded.reason}})),
                    )
                        .into_response();
                }
                Err(ReadFail::Db(reason)) => return api_failure(ApiFail::Db(reason)),
            };
            break workbench_projection(&app, snapshot);
        }
        // one bounded slice: woken early by a landing, otherwise the
        // cursor is re-checked when the slice ends
        let slice_end = std::cmp::min(
            deadline,
            tokio::time::Instant::now() + std::time::Duration::from_secs(1),
        );
        let _ = tokio::time::timeout_at(slice_end, wake).await;
    }
}

fn workbench_projection(app: &AppState, snapshot: Snapshot) -> Response {
    let store = match app.artifacts_dir() {
        Some(dir) => crate::web::ArtifactStore::at(dir.to_path_buf()),
        None => crate::web::ArtifactStore::default(),
    };
    match crate::workbench::snapshot(&snapshot.world, &snapshot.rows, &store) {
        Ok(value) => (StatusCode::OK, Json(value)).into_response(),
        Err(fail) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"code": "degraded", "detail": fail.reason()}})),
        )
            .into_response(),
    }
}

/// The rows a snapshot holds, for the preview's reference existence.
fn loadout_of(snapshot: &Snapshot) -> crate::db::Loadout {
    crate::db::Loadout {
        rows: snapshot.rows.clone(),
        state: crate::db::LoadState::Full(snapshot.world.clone()),
    }
}

/// The stateless draft preview: one body against fixed references,
/// rendered by the canonical renderer. It never writes and never
/// remembers; an unknown reference refuses.
async fn workbench_preview(State(app): State<AppState>, body: Bytes) -> Response {
    let request: crate::workbench::PreviewRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let snapshot = match app.snapshot() {
        Ok(s) => s,
        Err(fail) => return api_failure(ApiFail::Db(fail.reason().to_string())),
    };
    let store = match app.artifacts_dir() {
        Some(dir) => crate::web::ArtifactStore::at(dir.to_path_buf()),
        None => crate::web::ArtifactStore::default(),
    };
    match crate::workbench::preview(&loadout_of(&snapshot), &snapshot.world, &request, &store) {
        Ok(value) => (StatusCode::OK, Json(value)).into_response(),
        Err(detail) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"code": "invalid_reference", "detail": detail}})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct WorkbenchCommand {
    /// The name the review surface showed; the door records exactly it
    actor: ActorName,
    command: Command,
}

/// Whether a request declares itself cross-site, the CSRF guard's
/// whole reading: same-origin and none (same-origin tools) pass.
fn cross_site(headers: &HeaderMap) -> bool {
    matches!(
        headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()),
        Some(v) if v != "same-origin" && v != "none"
    )
}

/// The workbench's one write door: the console identity rule with the
/// attribution bound to the submission. A cookie that changed under a
/// confirmed review refuses loudly, never silently re-attributes; the
/// tier is pinned human and the scope stays empty.
async fn workbench_command(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if cross_site(&headers) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": {"code": "cross_site_refused", "detail": "cross-site POST refused"}})),
        )
            .into_response();
    }
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: WorkbenchCommand = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
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
                    request: &raw,
                },
            );
            return api_failure(ApiFail::Malformed(e.to_string()));
        }
    };
    if let Some(claimed) = cookie_actor(&headers)
        && claimed != request.actor.as_str()
    {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": {"code": "actor_changed", "detail": format!(
                "this browser claims actor {claimed} while the review confirmed {}; re-review under one name",
                request.actor.as_str()
            )}})),
        )
            .into_response();
    }
    let context = Context {
        actor: request.actor.clone(),
        tier: Tier::Human,
        scope: None,
    };
    let command = request.command;
    let received = AsReceived {
        client: Some("workbench".into()),
        raw,
    };
    let first_claim = cookie_actor(&headers).is_none();
    match app.execute(&context, command.clone(), None, received) {
        Ok(stored) => {
            let mut reply = json!({"records": &stored});
            // a create reply names what was born, exactly as the command
            // door names it, so the workbench can open the born thread
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
            let mut response = (StatusCode::OK, Json(reply)).into_response();
            if first_claim {
                response
                    .headers_mut()
                    .insert(header::SET_COOKIE, actor_cookie(&request.actor));
            }
            response
        }
        Err(e) => {
            let (status, code, detail) = match e {
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

// -- the workbench host launch door ---------------------------------------

#[derive(Deserialize)]
struct WorkbenchLaunchRequest {
    /// The explicit existing workspace directory the session starts in;
    /// the door selects, it never creates
    workspace: WorkbenchLaunchWorkspace,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum WorkbenchLaunchWorkspace {
    /// An existing directory, exactly as the CLI door demands
    Existing(String),
}

/// What a submission did: the host's terminal acceptance and the
/// workspace it names - never a claim that a session runs.
#[derive(Serialize)]
pub struct LaunchReply {
    pub submitted: bool,
    pub workspace: String,
    pub terminal: String,
}

/// The workbench's host launch: the same shared construction the CLI
/// doors use, submitted through this server's own published URL so the
/// session reaches the tracker it was started from. No record lands and
/// no task, claim, or binding rides it; the cross-site guard is the
/// door's whole human-facing law, because a launch's only authority is
/// the host itself.
async fn workbench_launch(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if cross_site(&headers) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": {"code": "cross_site_refused", "detail": "cross-site POST refused"}})),
        )
            .into_response();
    }
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: WorkbenchLaunchRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
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
                    request: &raw,
                },
            );
            return api_failure(ApiFail::Malformed(e.to_string()));
        }
    };
    let received = AsReceived {
        client: Some("workbench-launch".into()),
        raw,
    };
    let started = Instant::now();
    // the host ask can wait on its bounded discovery: it runs off the
    // runtime, and the attempts line waits for the whole answer
    let blocking_app = app.clone();
    let outcome =
        tokio::task::spawn_blocking(move || launch_operation(&blocking_app, &request.workspace))
            .await
            .expect("the launch operation does not panic");
    let guard = app.inner.lock().expect("the writer lock is not poisoned");
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

/// One workbench launch: the server's own published URL for the
/// routing, the shared pinned-pi and direnv construction, the host
/// submission. The stages and their order are the CLI door's.
fn launch_operation(
    app: &AppState,
    workspace: &WorkbenchLaunchWorkspace,
) -> Result<LaunchReply, ApiFail> {
    let WorkbenchLaunchWorkspace::Existing(path) = workspace;
    let Some(url) = app.server_url().map(str::to_string) else {
        return Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "launch_no_server_url",
            "the serving edge cannot name its own address; a launch cannot route the session it starts"
                .into(),
        ));
    };
    let pi = crate::host::pinned_pi().map_err(ApiFail::Host)?;
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    // a recorded Cleaned workspace takes no new sessions until an
    // explicit restore; unrelated directories and primary stay valid
    {
        let conn = db::open_read(app.db_path()).map_err(|e| ApiFail::Db(e.to_string()))?;
        let world = match db::load(&conn)
            .map_err(|e| ApiFail::Db(e.to_string()))?
            .state
        {
            db::LoadState::Full(world) => world,
            db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
        };
        let canonical =
            crate::workspace::canonical_dir(Path::new(path)).map_err(ApiFail::Workspace)?;
        match crate::workspace::resolve_workspace_address(&world, Path::new(path))
            .map_err(ApiFail::Workspace)?
        {
            crate::workspace::LifecycleAddress::Single(birth) => {
                if world.workspace_stage(birth) == Some(crate::WorkspaceStage::Cleaned) {
                    return Err(ApiFail::Refused(
                        StatusCode::CONFLICT,
                        "workspace_cleaned",
                        format!(
                            "{} is recorded Cleaned (record {}); a cleaned workspace takes no new sessions until an explicit restore: sac workspace restore {}",
                            canonical.display(),
                            birth.0,
                            canonical.display()
                        ),
                    ));
                }
            }
            // several stored paths over one directory: the ordinary-use
            // guard refuses rather than picking an identity
            crate::workspace::LifecycleAddress::Ambiguous(births) => {
                return Err(ApiFail::Refused(
                    StatusCode::CONFLICT,
                    "workspace_ambiguous_path",
                    format!(
                        "{} is the recorded path of {} workspace births; resolve their history by hand before launching there",
                        canonical.display(),
                        births.len()
                    ),
                ));
            }
            crate::workspace::LifecycleAddress::Unrecorded => {}
        }
    }
    let label = crate::host::command_label(&pi.display().to_string());
    let mut routing = format!(
        "SACCADE_ACTOR={} SACCADE_SERVER={} SACCADE_REPO={}",
        crate::host::shell_quote(&label),
        crate::host::shell_quote(&url),
        crate::host::shell_quote(&repo_root.display().to_string()),
    );
    if let Ok(exe) = std::env::current_exe() {
        routing.push_str(&format!(
            " SACCADE_CLI={}",
            crate::host::shell_quote(&exe.display().to_string())
        ));
    }
    let (dir_str, command) =
        crate::host::launch_command(Path::new(path), &routing, &[pi.display().to_string()])
            .map_err(ApiFail::Host)?;
    let terminal =
        crate::host::submit_terminal(&dir_str, "sac launch", &command).map_err(ApiFail::Host)?;
    Ok(LaunchReply {
        submitted: true,
        workspace: dir_str,
        terminal,
    })
}

// -- the task-free workspace controller ----------------------------------

/// What a creation provisioned: actual Git facts and the birth record
/// that carries them - the workspace's identity, never a numbering.
#[derive(Serialize)]
pub struct WorkspaceBirth {
    /// The birth record's position: the workspace's stable identity
    pub seq: usize,
    pub base: String,
    pub branch: String,
    pub path: String,
}

#[derive(Serialize)]
pub struct CreateReply {
    pub created: bool,
    pub workspace: WorkspaceBirth,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct CreateRequest {
    /// The descriptive label of the c-366 name: optional, falling
    /// back to workspace-<token>
    #[serde(default)]
    label: Option<String>,
    /// The canonical base: a ref, defaulting to the primary HEAD
    #[serde(default)]
    base: Option<String>,
    #[serde(default)]
    client: Option<String>,
}

/// Provision through the real Git provisioner under the writer lock,
/// then land the birth with the facts the effect actually produced.
fn create_operation(
    app: &AppState,
    guard: &mut ServerState,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<CreateReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    let home = workspace::worktree_home(&inner.db_path);
    let provisioned =
        workspace::provision(&repo_root, &home, label, base).map_err(ApiFail::Workspace)?;
    let drafts = vec![db::RecordDraft {
        context: Context::system(),
        event: Event::WorkspaceCreated {
            base: provisioned.base.clone(),
            branch: provisioned.branch.clone(),
            worktree: provisioned.worktree.clone(),
        },
    }];
    match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
        Ok(records) => {
            let birth = records[0].seq;
            Ok(CreateReply {
                created: true,
                workspace: WorkspaceBirth {
                    seq: birth,
                    base: provisioned.base.as_str().to_string(),
                    branch: provisioned.branch.as_str().to_string(),
                    path: provisioned.worktree.as_path().display().to_string(),
                },
                records,
            })
        }
        // the effect stands: a failed birth is disclosed, never rolled back
        Err(e) => Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage",
            format!(
                "provisioned branch {} at {} but the birth record could not land: {e}; \
                 the branch and directory stand - inspect them before creating another",
                provisioned.branch.as_str(),
                provisioned.worktree.as_path().display()
            ),
        )),
    }
}

async fn workspace_create(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: CreateRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = create_operation(
        &app,
        &mut guard,
        request.label.as_deref(),
        request.base.as_deref(),
    );
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

/// What a stopped-session change did: the incarnation, the directory
/// it carried before, and the directory it carries now.
#[derive(Serialize)]
pub struct ChangeReply {
    pub changed: bool,
    /// The birth record position: the machine identity channel
    pub incarnation: usize,
    /// The incarnation's dense public number, the human-facing i-N
    pub num: usize,
    pub from: String,
    pub to: String,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct ChangeRequest {
    /// The incarnation's public number, the human-facing i-N
    incarnation: usize,
    /// The selected directory: an explicit existing one, never created here
    to: String,
    #[serde(default)]
    client: Option<String>,
}

/// One stopped-session change: resolve the recorded incarnation,
/// prove quiescence through its kernel guard, retarget the saved
/// header atomically, then land the System change fact. Busy and
/// unknown refuse before any mutation; the session's activation lock
/// serializes the whole window against automatic mail activation.
fn change_operation(
    app: &AppState,
    guard: &mut ServerState,
    num: usize,
    to: &str,
) -> Result<ChangeReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    // the fold resolves the incarnation before anything moves: a
    // missing or historical identity is no change target
    let world = match db::load(&inner.conn)
        .map_err(|e| ApiFail::Db(e.to_string()))?
        .state
    {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    let id = world
        .incarnation_at_num(num)
        .ok_or(ApiFail::Reject(Reject::InvalidIncarnationId, None))?;
    let (native_id, saved_session) = match &world.incarnations[&id].incarnation {
        crate::Incarnation::Native {
            native_id,
            saved_session,
        } => (native_id.clone(), saved_session.clone()),
        crate::Incarnation::Established { .. } => {
            return Err(ApiFail::Refused(
                StatusCode::BAD_REQUEST,
                "invalid_incarnation_id",
                format!(
                    "i-{num} is a historical attribution identity: it carries no saved native session to move"
                ),
            ));
        }
    };
    // the target is validated before any lock is taken: an explicit
    // existing directory, never one this route creates
    let target = std::path::PathBuf::from(to);
    let target = WorktreePath::new(target.clone()).map_err(|_| {
        ApiFail::Refused(
            StatusCode::BAD_REQUEST,
            "workspace_path_invalid",
            format!("{} is not an absolute path", target.display()),
        )
    })?;
    if !target.as_path().is_dir() {
        return Err(ApiFail::Refused(
            StatusCode::BAD_REQUEST,
            "workspace_path_missing",
            format!(
                "{} does not exist; a change selects an existing directory and creates nothing",
                target.as_path().display()
            ),
        ));
    }
    // a recorded Cleaned workspace blocks ordinary use as a switch
    // target until an explicit restore; unrelated directories pass
    let target_canonical =
        crate::workspace::canonical_dir(target.as_path()).map_err(ApiFail::Workspace)?;
    match crate::workspace::resolve_workspace_address(&world, target.as_path())
        .map_err(ApiFail::Workspace)?
    {
        crate::workspace::LifecycleAddress::Single(birth) => {
            if world.workspace_stage(birth) == Some(crate::WorkspaceStage::Cleaned) {
                return Err(ApiFail::Refused(
                    StatusCode::CONFLICT,
                    "workspace_cleaned",
                    format!(
                        "{} is recorded Cleaned (record {}); a cleaned workspace takes no new sessions until an explicit restore: sac workspace restore {}",
                        target_canonical.display(),
                        birth.0,
                        target_canonical.display()
                    ),
                ));
            }
        }
        crate::workspace::LifecycleAddress::Ambiguous(births) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_ambiguous_path",
                format!(
                    "{} is the recorded path of {} workspace births; resolve their history by hand before switching there",
                    target_canonical.display(),
                    births.len()
                ),
            ));
        }
        crate::workspace::LifecycleAddress::Unrecorded => {}
    }

    // the session's activation lock: automatic mail activation and
    // this change serialize on the very same file, so neither can
    // interleave with the other's decision
    let _session = match crate::activate::lock_session(&repo_root, native_id.as_str()) {
        Ok(lock) => lock,
        Err(crate::activate::LockFail::InFlight) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "session_activation_in_flight",
                "another activation holds this session's lock; try again when it settles".into(),
            ));
        }
        Err(crate::activate::LockFail::Unavailable(e)) => {
            return Err(ApiFail::Refused(
                StatusCode::INTERNAL_SERVER_ERROR,
                "session_lock_unavailable",
                format!("the session's activation lock could not be opened: {e}"),
            ));
        }
    };
    // quiescence is kernel evidence, never a guess: a held guard is a
    // live process, an unprobeable one is unknown - both refuse first
    let guard_path = crate::activate::guard_path(&repo_root, native_id.as_str());
    match crate::activate::guard_held(&guard_path) {
        Some(false) => {}
        Some(true) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "session_live",
                format!(
                    "i-{num}'s launch guard is held by a live process; exit that session first - this route moves stopped sessions only"
                ),
            ));
        }
        None => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "session_quiescence_unknown",
                format!(
                    "i-{num}'s launch guard at {} cannot be probed; liveness is unknown and nothing was moved",
                    guard_path.display()
                ),
            ));
        }
    }
    // the physical retarget: the adapter validates the immutable saved
    // association (id match) before any byte moves, then replaces the
    // header atomically
    let from =
        crate::pi_session::retarget_cwd(saved_session.as_path(), native_id.as_str(), &target)
            .map_err(|e| {
                let (status, code) = match &e {
                    crate::pi_session::HandoffFail::Missing(_) => {
                        (StatusCode::BAD_REQUEST, "saved_session_missing")
                    }
                    crate::pi_session::HandoffFail::IdMismatch { .. } => {
                        (StatusCode::BAD_REQUEST, "saved_session_mismatch")
                    }
                    _ => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "saved_session_unreadable",
                    ),
                };
                ApiFail::Refused(status, code, e.to_string())
            })?;
    // the record lands only after the effect: a failed publication is
    // disclosed with its pointers, never rolled back into a pending state
    let command = Command::ChangeWorkspace {
        incarnation: id,
        worktree: target.clone(),
    };
    match db::record(
        &mut inner.conn,
        &Context::system(),
        command,
        db::now_epoch(),
    ) {
        Ok((records, _)) => Ok(ChangeReply {
            changed: true,
            incarnation: id.0.0,
            num,
            from,
            to: target.as_path().display().to_string(),
            records,
        }),
        Err(e) => Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage",
            format!(
                "the saved session at {} was retargeted to {} but the change record could not land: {e}; \
                 the header change stands and the record is missing - resume still works from the new directory",
                saved_session.as_path().display(),
                target.as_path().display()
            ),
        )),
    }
}

async fn workspace_change(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: ChangeRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = change_operation(&app, &mut guard, request.incarnation, &request.to);
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

// -- the workspace lifecycle controller ---------------------------------
// Clean and restore: the ratified reversible lifecycle. The ordered
// clean contract runs can-clean checks, terminal close, terminal
// verification, git worktree remove (branch retained), checkout and
// worktree absence verification, and the host's visible absence,
// and only then lands the Cleaned fact. Restore establishes or
// validates the matching checkout and lands Restored; it launches no
// session and creates no terminal. Reconciliation is the one-time
// cutover verification: absent resources record Cleaned now, present
// ones stand preserved, conflicts surface - never a classification.

/// The recorded facts of the lifecycle's target: the birth identity
/// and what the birth carried.
#[derive(Serialize)]
pub struct LifecycleWorkspace {
    seq: usize,
    base: String,
    branch: String,
    path: String,
    /// The establishment's home task, when the birth was task-bound
    home: Option<usize>,
}

impl LifecycleWorkspace {
    fn of(facts: crate::store::WorkspaceFacts) -> Self {
        LifecycleWorkspace {
            seq: facts.birth.0,
            base: facts.base.as_str().to_string(),
            branch: facts.branch.as_str().to_string(),
            path: facts.worktree.as_path().display().to_string(),
            home: facts.home.map(|t| t.0),
        }
    }
}

#[derive(Serialize)]
pub struct CleanReply {
    /// A Cleaned fact landed with this call
    pub cleaned: bool,
    /// Recorded cleaned and every verified resource already absent:
    /// idempotent success, no duplicate transition
    pub already_cleaned: bool,
    pub workspace: LifecycleWorkspace,
    /// The honest step list: what the door asked, what it verified,
    /// and any host ask that failed before the list confirmed closure
    pub verified: Vec<String>,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct CleanRequest {
    /// The exact recorded workspace path; never inferred from a task
    /// or incarnation
    path: String,
    #[serde(default)]
    client: Option<String>,
}

#[derive(Serialize)]
pub struct RestoreReply {
    pub restored: bool,
    /// Recorded Available with the matching registered checkout
    /// already standing: no-op success, no duplicate restoration fact
    pub already_restored: bool,
    /// The matching registered worktree stood and was reused as-is,
    /// local changes preserved; nothing was checked out
    pub reused: bool,
    /// The branch's current local tip after restoration, when git
    /// named one; never a reset target, always a report
    pub tip: Option<String>,
    pub workspace: LifecycleWorkspace,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct RestoreRequest {
    path: String,
    #[serde(default)]
    client: Option<String>,
}

#[derive(Serialize)]
pub struct ReconcileEntry {
    pub seq: usize,
    pub path: String,
    pub reason: String,
}

#[derive(Serialize)]
pub struct ReconcileReply {
    /// Whether the verified Cleaned facts were recorded (or a dry run)
    pub applied: bool,
    pub cleaned: Vec<ReconcileEntry>,
    pub preserved: Vec<ReconcileEntry>,
    pub conflicts: Vec<ReconcileEntry>,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct ReconcileRequest {
    /// Record the verified Cleaned facts; without this the door only
    /// verifies and reports
    #[serde(default)]
    apply: bool,
    #[serde(default)]
    client: Option<String>,
}

/// The world fold plus the canonical target resolution every lifecycle
/// door shares: exact recorded workspace births at this path, primary
/// and unrecognized directories refused.
fn lifecycle_target(
    inner: &Inner,
    path_str: &str,
) -> Result<(crate::store::World, crate::store::WorkspaceFacts), ApiFail> {
    let world = match db::load(&inner.conn)
        .map_err(|e| ApiFail::Db(e.to_string()))?
        .state
    {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    // the canonical form names the directory in refusal messages; the
    // resolver owns the addressing decision
    let canonical = workspace::canonical_dir(Path::new(path_str)).map_err(ApiFail::Workspace)?;
    let facts = match workspace::resolve_workspace_address(&world, Path::new(path_str))
        .map_err(ApiFail::Workspace)?
    {
        workspace::LifecycleAddress::Single(birth) => world
            .workspace_facts(birth)
            .expect("the resolver named a recorded birth"),
        workspace::LifecycleAddress::Ambiguous(births) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_ambiguous_path",
                format!(
                    "{} is the recorded path of {} workspace births ({}); resolve their history by hand before any lifecycle act",
                    canonical.display(),
                    births.len(),
                    births
                        .iter()
                        .map(|b| b.0.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
        workspace::LifecycleAddress::Unrecorded => {
            return Err(ApiFail::Refused(
                StatusCode::BAD_REQUEST,
                "workspace_not_recorded",
                format!(
                    "{} names no recorded workspace birth; clean and restore select an exact recorded workspace, never a primary checkout or an unrecognized directory",
                    canonical.display()
                ),
            ));
        }
    };
    Ok((world, facts))
}

/// How long the visible-absence check re-asks a Listed answer before
/// the door gives its at-ceiling answer: the host's own budget for
/// discovering external changes is 30 seconds, and this ceiling holds
/// the same margin over it the launch discovery wait holds. A Listed
/// answer inside the window is the host's stale catalog, never
/// evidence of anything.
const HOST_ABSENCE_WAIT_SECS: u64 = 90;

/// The visible-absence poll cadence: the host's outer snapshot cache
/// turns over each second, so faster asks buy nothing.
const HOST_ABSENCE_POLL_MS: u64 = 2000;

/// The visible-absence bound's one test seam: a scratch tracker holds
/// the window short so the honest classification at expiry is observable
/// without waiting the production bound. Production never sets it.
fn host_absence_wait_secs() -> u64 {
    std::env::var("SACCADE_HOST_ABSENCE_WAIT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(HOST_ABSENCE_WAIT_SECS)
}

/// The visible-absence cadence's test seam, beside the ceiling's: the
/// stale-then-absent model must turn over inside a shrunk ceiling.
/// Production never sets it.
fn host_absence_poll_ms() -> u64 {
    std::env::var("SACCADE_HOST_ABSENCE_POLL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|ms| *ms > 0)
        .unwrap_or(HOST_ABSENCE_POLL_MS)
}

/// The visible-absence poll's at-ceiling outcome: the host still names
/// the workspace, or no verified answer ever arrived.
enum AbsenceCeiling {
    StillListed,
    Unverified(String),
}

/// The bounded wait for the host's visible absence. A Listed answer is
/// the host's documented propagation window after an external removal,
/// never evidence: the ask repeats until the host names nothing or the
/// ceiling passes. selector_not_found alone proves absence; host
/// errors retry inside the same ceiling, and uncertainty at the bound
/// stays uncertainty - arbitrary host errors are never read as absence.
fn visible_absence_within(dir: &str) -> Result<(), AbsenceCeiling> {
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(host_absence_wait_secs());
    let cadence = std::time::Duration::from_millis(host_absence_poll_ms());
    loop {
        match crate::host::workspace_visible_presence(dir) {
            Ok(crate::host::HostPresence::VisiblyAbsent) => return Ok(()),
            Ok(crate::host::HostPresence::Listed) => {
                if std::time::Instant::now() >= deadline {
                    return Err(AbsenceCeiling::StillListed);
                }
            }
            Err(e) => {
                if std::time::Instant::now() >= deadline {
                    return Err(AbsenceCeiling::Unverified(e.to_string()));
                }
            }
        }
        std::thread::sleep(cadence);
    }
}

/// One completed cleanup: the ordered contract, every step honest.
#[allow(clippy::too_many_lines)]
fn clean_operation(
    app: &AppState,
    guard: &mut ServerState,
    path_str: &str,
) -> Result<CleanReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    let (world, facts) = lifecycle_target(inner, path_str)?;
    let birth = facts.birth;
    let canonical =
        workspace::canonical_dir(facts.worktree.as_path()).map_err(ApiFail::Workspace)?;
    let dir_str = canonical.display().to_string();
    if canonical == repo_root {
        return Err(ApiFail::Refused(
            StatusCode::BAD_REQUEST,
            "workspace_primary_refused",
            format!(
                "{} is the primary checkout; cleanup never touches it",
                canonical.display()
            ),
        ));
    }
    let stage = world
        .workspace_stage(birth)
        .expect("a recorded birth holds a stage");
    let mut registrations =
        workspace::registered_worktrees(&repo_root).map_err(ApiFail::Workspace)?;
    let registered = registrations.get(&canonical).cloned();
    let present = canonical.is_dir();

    let mut verified: Vec<String> = Vec::new();
    // recorded cleaned and every resource absent is idempotent success:
    // no duplicate transition, verified through the same doors the
    // cleanup itself ran - terminal list and visible absence alike
    if stage == crate::WorkspaceStage::Cleaned {
        if present || registered.is_some() {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_cleaned_mismatch",
                format!(
                    "{} is recorded Cleaned but {} stands; a recorded cleanup never authorizes deleting unexpected work - inspect it by hand",
                    dir_str,
                    if present {
                        "a directory"
                    } else {
                        "a git worktree registration"
                    }
                ),
            ));
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            match crate::host::workspace_terminals(&dir_str) {
                Ok(terminals) if terminals.is_empty() => {
                    verified.push("verified: the host names no terminals for the workspace".into());
                    break;
                }
                Ok(terminals) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(ApiFail::Refused(
                            StatusCode::CONFLICT,
                            "workspace_terminals_open",
                            format!(
                                "the host names {} live terminal(s) at a workspace recorded Cleaned; the recorded closure does not hold - inspect and retry",
                                terminals.len()
                            ),
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                Err(e) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(ApiFail::Refused(
                            StatusCode::BAD_GATEWAY,
                            "workspace_host_unverified",
                            format!(
                                "terminal state unverified at a workspace recorded Cleaned: {e}; the idempotence answer names only what verified - retry when the host answers"
                            ),
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }
        match visible_absence_within(&dir_str) {
            Ok(()) => {
                verified.push(format!(
                    "verified: the host names no workspace at {dir_str}"
                ));
            }
            Err(AbsenceCeiling::StillListed) => {
                return Err(ApiFail::Refused(
                    StatusCode::CONFLICT,
                    "workspace_still_listed",
                    format!(
                        "{} is recorded Cleaned but the host still lists a workspace there; the recorded closure does not hold - inspect and retry",
                        dir_str
                    ),
                ));
            }
            Err(AbsenceCeiling::Unverified(_)) => {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_GATEWAY,
                    "workspace_host_unverified",
                    format!(
                        "the host gave no verified answer for visible absence of {dir_str} within its bound; nothing further was done"
                    ),
                ));
            }
        }
        return Ok(CleanReply {
            cleaned: false,
            already_cleaned: true,
            workspace: LifecycleWorkspace::of(facts),
            verified,
            records: Vec::new(),
        });
    }

    let destructive = registered.is_some();
    // a stale git registration at a recorded path is named, never
    // pruned by a door: git's own repair stays a human act
    if registered.is_some() && !canonical.is_dir() {
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_checkout_mismatch",
            format!(
                "git still registers a worktree at {} whose directory is gone; cleanup refuses the stale registration and never prunes it - inspect it and run git worktree prune by hand",
                dir_str
            ),
        ));
    }
    if destructive {
        // the retained branch is the restore contract's foundation: a
        // missing branch refuses the destructive act before anything runs
        if workspace::branch_tip(&repo_root, &facts.branch)
            .map_err(ApiFail::Workspace)?
            .is_none()
        {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_branch_missing",
                format!(
                    "the recorded branch {} holds no local tip; a cleanup that would remove its checkout refuses, and restore would refuse too",
                    facts.branch.as_str()
                ),
            ));
        }
        // work preservation: staged, unstaged, and untracked changes
        // refuse the whole door before any host or git act
        if let Some(detail) = workspace::preserved_work(&canonical).map_err(ApiFail::Workspace)? {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_work_present",
                format!(
                    "{} holds staged, unstaged, or untracked work that cleanup refuses to remove:\n{}",
                    dir_str, detail
                ),
            ));
        }
        verified.push(
            "can-clean: recorded Available, branch present, no staged, unstaged, or untracked work"
                .into(),
        );
    } else if present {
        // bytes at the recorded path are not the checkout: a probe
        // finding them neither restores the identity nor deletes them
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_checkout_mismatch",
            format!(
                "{} holds a directory git does not register as this repo's worktree; cleanup deletes no unexpected work - inspect it by hand",
                dir_str
            ),
        ));
    } else {
        verified.push("can-clean: recorded Available, checkout already absent".into());
    }

    // the host stage: ask for closure, then verify the list - accepted
    // host input is never treated as confirmed closure
    if let Err(e) = crate::host::close_workspace_terminals(&dir_str) {
        verified.push(format!(
            "terminal close ask failed ({}); the verification list decides",
            e
        ));
    } else {
        verified.push("asked the host to close the workspace's terminals".into());
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match crate::host::workspace_terminals(&dir_str) {
            Ok(terminals) if terminals.is_empty() => {
                verified.push("verified: the host names no terminals for the workspace".into());
                break;
            }
            Ok(terminals) => {
                if std::time::Instant::now() >= deadline {
                    return Err(ApiFail::Refused(
                        StatusCode::CONFLICT,
                        "workspace_terminals_open",
                        format!(
                            "the host still names {} live terminal(s) for {}; no checkout removal ran",
                            terminals.len(),
                            dir_str
                        ),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(e) => {
                if std::time::Instant::now() >= deadline {
                    return Err(ApiFail::Refused(
                        StatusCode::BAD_GATEWAY,
                        "workspace_host_unverified",
                        format!(
                            "terminal closure is unverified: {e}; no checkout removal ran - retry clean when the host answers"
                        ),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }

    // the checkout removal: git's normal protections, the branch and
    // every commit on it retained; an already-absent checkout satisfies
    if destructive {
        workspace::remove_checkout(&repo_root, &canonical).map_err(ApiFail::Workspace)?;
        verified.push(format!(
            "removed the checkout with git worktree remove; branch {} retained",
            facts.branch.as_str()
        ));
    }

    // absence: the path is gone and git no longer registers it
    if canonical.exists() {
        return Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_absence_unverified",
            format!(
                "{} still exists after the removal attempt; the record lands only on verified absence - inspect and retry",
                dir_str
            ),
        ));
    }
    registrations = workspace::registered_worktrees(&repo_root).map_err(ApiFail::Workspace)?;
    if registrations.contains_key(&canonical) {
        return Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_absence_unverified",
            format!(
                "git still registers a worktree at {} after the removal attempt; the record lands only on verified absence - inspect and retry",
                dir_str
            ),
        ));
    }
    verified.push("verified: the checkout is gone and git registers no worktree there".into());

    // the host's visible absence, after the git removal: the host's
    // own read contract, never assumed from an earlier step
    match visible_absence_within(&dir_str) {
        Ok(()) => {
            verified.push("verified: the host names no workspace there".into());
        }
        Err(AbsenceCeiling::StillListed) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_still_listed",
                format!(
                    "the host still lists a workspace at {} after the checkout removal; the record lands only on visible absence - inspect and retry",
                    dir_str
                ),
            ));
        }
        Err(AbsenceCeiling::Unverified(_)) => {
            return Err(ApiFail::Refused(
                StatusCode::BAD_GATEWAY,
                "workspace_host_unverified",
                format!(
                    "the host gave no verified answer for visible absence of {dir_str} within its bound; nothing further was done"
                ),
            ));
        }
    }

    // every step verified: the completed fact lands, machinery-authored
    let drafts = vec![db::RecordDraft {
        context: Context::system(),
        event: Event::WorkspaceCleaned { workspace: birth },
    }];
    match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
        Ok(records) => Ok(CleanReply {
            cleaned: true,
            already_cleaned: false,
            workspace: LifecycleWorkspace::of(facts),
            verified,
            records,
        }),
        Err(ExecuteFail::Reject(crate::Reject::WorkspaceStageMismatch)) => {
            // a concurrent door recorded the same completion: the
            // resources are verified absent either way - idempotent
            Ok(CleanReply {
                cleaned: false,
                already_cleaned: true,
                workspace: LifecycleWorkspace::of(facts),
                verified,
                records: Vec::new(),
            })
        }
        Err(e) => Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage",
            format!(
                "every cleanup step verified but the Cleaned fact could not land: {e}; the resources stand absent - retry clean to record"
            ),
        )),
    }
}

/// One completed restoration: establish or validate the matching
/// checkout, then record. No terminal, no session, no incarnation.
fn restore_operation(
    app: &AppState,
    guard: &mut ServerState,
    path_str: &str,
) -> Result<RestoreReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    let (world, facts) = lifecycle_target(inner, path_str)?;
    let birth = facts.birth;
    let canonical =
        workspace::canonical_dir(facts.worktree.as_path()).map_err(ApiFail::Workspace)?;
    let dir_str = canonical.display().to_string();
    let registrations = workspace::registered_worktrees(&repo_root).map_err(ApiFail::Workspace)?;
    let registered = registrations.get(&canonical).cloned();
    // a git registration whose directory is gone restores and reuses
    // nothing, at either stage: a stale admin entry is named as the
    // mismatch it is, and git's own repair stays a human act - nothing
    // counts as a matching checkout without the directory standing
    if registered.is_some() && !canonical.is_dir() {
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_checkout_mismatch",
            format!(
                "git still registers a worktree at {} whose directory is gone; a stale registration neither restores nor reuses - inspect it and run git worktree prune by hand",
                dir_str
            ),
        ));
    }

    // the matching registered checkout standing on an Available
    // workspace is no-op success: no duplicate restoration fact lands,
    // and the local changes it holds stand untouched. The absent case
    // is not special: restoration is idempotent by postcondition, so
    // an Available workspace with no checkout recreates it below under
    // the same safety checks and records the actual fact. The safety
    // checks hold at both stages - the other shapes are mismatches.
    if world.workspace_stage(birth) == Some(crate::WorkspaceStage::Available)
        && matches!(&registered, Some(Some(b)) if b.as_str() == facts.branch.as_str())
        && canonical.is_dir()
    {
        let tip = workspace::branch_tip(&repo_root, &facts.branch)
            .map_err(ApiFail::Workspace)?
            .map(|c| c.as_str().to_string());
        return Ok(RestoreReply {
            restored: false,
            already_restored: true,
            reused: true,
            tip,
            workspace: LifecycleWorkspace::of(facts),
            records: Vec::new(),
        });
    }
    if workspace::branch_tip(&repo_root, &facts.branch)
        .map_err(ApiFail::Workspace)?
        .is_none()
    {
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_branch_missing",
            format!(
                "the recorded branch {} holds no local tip; restoration refuses - the branch is never reconstructed from a base or a remote",
                facts.branch.as_str()
            ),
        ));
    }
    let (reused, tip) = match registered {
        Some(Some(checked_out)) if checked_out == facts.branch => {
            // the matching worktree stands: reuse as-is, local changes
            // preserved, no checkout, no reset - the tip is git's own
            let tip = workspace::branch_tip(&repo_root, &facts.branch)
                .map_err(ApiFail::Workspace)?
                .map(|c| c.as_str().to_string());
            (true, tip)
        }
        Some(Some(checked_out)) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_checkout_mismatch",
                format!(
                    "{} is registered on branch {}, not the recorded {}; no forced checkout, reset, or overwrite runs",
                    dir_str,
                    checked_out.as_str(),
                    facts.branch.as_str()
                ),
            ));
        }
        Some(None) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_checkout_mismatch",
                format!(
                    "{} is registered without a branch (bare or detached); restoration matches a named branch and runs nothing",
                    dir_str
                ),
            ));
        }
        None if canonical.exists() => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_checkout_mismatch",
                format!(
                    "{} holds a directory git does not register as this repo's worktree; restoration overwrites nothing - inspect it by hand",
                    dir_str
                ),
            ));
        }
        None => {
            // absent path: recreate at the branch's current local tip
            // under git's normal protections - a branch already checked
            // out elsewhere refuses, as does a missing-but-registered
            // worktree at the path
            workspace::add_checkout(&repo_root, &canonical, &facts.branch)
                .map_err(ApiFail::Workspace)?;
            let tip = workspace::branch_tip(&repo_root, &facts.branch)
                .map_err(ApiFail::Workspace)?
                .map(|c| c.as_str().to_string());
            (false, tip)
        }
    };
    let drafts = vec![db::RecordDraft {
        context: Context::system(),
        event: Event::WorkspaceRestored { workspace: birth },
    }];
    match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
        Ok(records) => Ok(RestoreReply {
            restored: true,
            already_restored: false,
            reused,
            tip,
            workspace: LifecycleWorkspace::of(facts),
            records,
        }),
        Err(e) => Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage",
            format!(
                "the checkout at {} stands restored but the Restored fact could not land: {e}; the checkout is real - retry restore to record",
                dir_str
            ),
        )),
    }
}

/// The one-time cutover verification over every recorded workspace:
/// absent resources record Cleaned now, matching present ones stand
/// preserved, everything uncertain or conflicting surfaces. Never a
/// deletion, never a terminal close, never an age classification.
fn reconcile_operation(
    app: &AppState,
    guard: &mut ServerState,
    apply: bool,
) -> Result<ReconcileReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    let world = match db::load(&inner.conn)
        .map_err(|e| ApiFail::Db(e.to_string()))?
        .state
    {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    let registrations = workspace::registered_worktrees(&repo_root).map_err(ApiFail::Workspace)?;
    // every workspace birth in birth order, created and established
    let births = world.workspace_births();
    // paths shared by several births are a conflict before any act:
    // sharing keys on the canonical form, so twin spellings of one
    // directory count as the same unsafe address
    let mut shared: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut entries: Vec<(RecordId, crate::store::WorkspaceFacts)> = Vec::new();
    for birth in births {
        if let Some(facts) = world.workspace_facts(birth) {
            if let Ok(canonical) = workspace::canonical_dir(facts.worktree.as_path()) {
                *shared.entry(canonical.display().to_string()).or_default() += 1;
            }
            entries.push((birth, facts));
        }
    }
    let mut cleaned = Vec::new();
    let mut preserved = Vec::new();
    let mut conflicts = Vec::new();
    let mut to_record: Vec<Event> = Vec::new();
    for (birth, facts) in entries {
        let path_string = facts.worktree.as_path().display().to_string();
        let entry = |reason: String| ReconcileEntry {
            seq: birth.0,
            path: path_string.clone(),
            reason,
        };
        let shares = workspace::canonical_dir(facts.worktree.as_path())
            .map(|canonical| shared.get(&canonical.display().to_string()).copied())
            .unwrap_or(None)
            .unwrap_or(0);
        if shares > 1 {
            conflicts.push(entry("several workspace births share this path".into()));
            continue;
        }
        let canonical = match workspace::canonical_dir(facts.worktree.as_path()) {
            Ok(canonical) => canonical,
            Err(e) => {
                conflicts.push(entry(format!(
                    "the recorded path cannot canonicalize: {}",
                    e.detail()
                )));
                continue;
            }
        };
        let registered = registrations.get(&canonical).cloned();
        let present = canonical.is_dir();
        let stage = world.workspace_stage(birth);
        match (stage, registered, present) {
            (Some(crate::WorkspaceStage::Cleaned), None, false) => {
                // recorded Cleaned is verified through the same doors the
                // cleanup itself ran: orphan or live host resources
                // surface as this birth's conflict, never as a
                // reconciled state
                let dir_str = canonical.display().to_string();
                match crate::host::workspace_terminals(&dir_str) {
                    Ok(t) if !t.is_empty() => {
                        conflicts.push(entry(format!(
                            "the host names {} live terminal(s) at a workspace recorded Cleaned",
                            t.len()
                        )));
                        continue;
                    }
                    Err(e) => {
                        conflicts.push(entry(format!("terminal state unreadable: {e}")));
                        continue;
                    }
                    _ => {}
                }
                match visible_absence_within(&dir_str) {
                    Ok(()) => preserved.push(entry(
                        "verified: recorded Cleaned, checkout absent, host names no workspace"
                            .into(),
                    )),
                    Err(AbsenceCeiling::StillListed) => conflicts.push(entry(
                        "the host still lists a workspace recorded Cleaned".into(),
                    )),
                    Err(AbsenceCeiling::Unverified(e)) => {
                        conflicts.push(entry(format!("host presence unreadable: {e}")))
                    }
                }
            }
            (Some(crate::WorkspaceStage::Cleaned), _, true) => {
                conflicts.push(entry(
                    "recorded Cleaned but a directory stands there".into(),
                ));
            }
            (Some(crate::WorkspaceStage::Cleaned), Some(_), false) => {
                conflicts.push(entry(
                    "recorded Cleaned but git still registers a worktree there".into(),
                ));
            }
            (Some(crate::WorkspaceStage::Available), Some(Some(branch)), true)
                if branch.as_str() == facts.branch.as_str() =>
            {
                match workspace::preserved_work(&canonical) {
                    Ok(None) => preserved.push(entry(
                        "matching registered worktree, no staged, unstaged, or untracked work"
                            .into(),
                    )),
                    Ok(Some(_)) => conflicts.push(entry(
                        "matching worktree holds staged, unstaged, or untracked work".into(),
                    )),
                    Err(e) => {
                        conflicts.push(entry(format!("worktree state unreadable: {}", e.detail())))
                    }
                }
            }
            (Some(crate::WorkspaceStage::Available), Some(_), _) => {
                conflicts.push(entry(
                    "git registers the path on another branch, without a branch, or over gone bytes".into(),
                ));
            }
            (Some(crate::WorkspaceStage::Available), None, true) => {
                conflicts.push(entry(
                    "a directory stands at the recorded path but git registers no worktree".into(),
                ));
            }
            (Some(crate::WorkspaceStage::Available), None, false) => {
                // absent checkout: the host's visible and terminal
                // answers decide; uncertainty and live terminals surface
                let dir_str = canonical.display().to_string();
                let terminals = crate::host::workspace_terminals(&dir_str);
                match terminals {
                    Ok(t) if !t.is_empty() => {
                        conflicts.push(entry(format!(
                            "the host still names {} live terminal(s) there",
                            t.len()
                        )));
                        continue;
                    }
                    Err(e) => {
                        conflicts.push(entry(format!("terminal state unreadable: {e}")));
                        continue;
                    }
                    _ => {}
                }
                match visible_absence_within(&dir_str) {
                    Ok(()) => {
                        if apply {
                            to_record.push(Event::WorkspaceCleaned { workspace: birth });
                        }
                        cleaned.push(entry(format!(
                            "checkout absent and the host names no workspace{}",
                            if apply {
                                "; Cleaned recorded now"
                            } else {
                                "; dry run"
                            }
                        )));
                    }
                    Err(AbsenceCeiling::StillListed) => {
                        conflicts.push(entry("the host still lists a workspace there".into()));
                    }
                    Err(AbsenceCeiling::Unverified(e)) => {
                        conflicts.push(entry(format!("host presence unreadable: {e}")))
                    }
                }
            }
            (None, _, _) => {
                conflicts.push(entry("the fold holds no stage for this birth".into()));
            }
        }
    }
    let mut records = Vec::new();
    if apply && !to_record.is_empty() {
        let drafts = to_record
            .into_iter()
            .map(|event| db::RecordDraft {
                context: Context::system(),
                event,
            })
            .collect();
        match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
            Ok(landed) => records = landed,
            Err(e) => {
                return Err(ApiFail::Refused(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "storage",
                    format!(
                        "the verification ran but the Cleaned facts could not land: {e}; nothing is recorded - retry reconcile"
                    ),
                ));
            }
        }
    }
    Ok(ReconcileReply {
        applied: apply,
        cleaned,
        preserved,
        conflicts,
        records,
    })
}

async fn workspace_clean(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: CleanRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = clean_operation(&app, &mut guard, &request.path);
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

async fn workspace_restore(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: RestoreRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = restore_operation(&app, &mut guard, &request.path);
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

async fn workspace_reconcile(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: ReconcileRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = reconcile_operation(&app, &mut guard, request.apply);
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

// -- the native integration's doors -------------------------------------

/// One native integration attempt's outcome, logged like every
/// workspace operation: the attempts log is the diagnoser's memory.
fn integration_outcome<T>(
    app: &AppState,
    guard: &ServerState,
    actor: Option<String>,
    received: &AsReceived,
    started: std::time::Instant,
    outcome: &Result<T, ApiFail>,
) {
    let duration_ms = started.elapsed().as_millis() as u64;
    match outcome {
        Ok(_) => {
            let cursor = match guard {
                ServerState::Ready(inner) => db::last_seq(&inner.conn).ok().flatten(),
                ServerState::Degraded(..) => None,
            };
            app.attempts.append(
                actor.as_deref(),
                received.client.as_deref(),
                duration_ms,
                Outcome::Landed {
                    seqs: &cursor.map(|c| vec![c]).unwrap_or_default(),
                },
            );
        }
        Err(fail) => {
            let code = match fail {
                ApiFail::Reject(reject, _) => reject.code(),
                ApiFail::Degraded(_) => "degraded",
                ApiFail::Db(_) => "storage",
                ApiFail::Malformed(_) => "malformed_request",
                ApiFail::Workspace(effect) => effect.code(),
                ApiFail::NoRepo => "workspace_no_repo",
                ApiFail::Host(host) => host.wire_code(),
                ApiFail::Refused(_, code, _) => code,
            };
            let cursor = match guard {
                ServerState::Ready(inner) => db::last_seq(&inner.conn).ok().flatten(),
                ServerState::Degraded(..) => None,
            };
            app.attempts.append(
                actor.as_deref(),
                received.client.as_deref(),
                duration_ms,
                Outcome::Refused {
                    code,
                    cursor,
                    request: &received.raw,
                },
            );
        }
    }
}

/// What the integration learns when its session attaches.
#[derive(Serialize)]
pub struct AttachReply {
    /// Whether this call birthed the registration; an existing exact
    /// native session resolves, never a second incarnation
    pub registered: bool,
    /// The birth record position: the machine identity channel
    pub incarnation: usize,
    /// The incarnation's dense public number, the human-facing i-N
    pub num: usize,
    /// This session's launch-guard file: the integration holds an
    /// exclusive kernel flock on it from association until death - the
    /// per-native-session runtime evidence activation reads. Server
    /// authoritative; the extension never derives paths.
    pub guard: Option<String>,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct AttachRequest {
    native_id: crate::NativeSessionId,
    session_file: crate::SessionLocator,
    /// A compose launch's correlation token: the launching door learns
    /// which incarnation registered under its launch. Runtime plumbing
    /// only - nothing lands in the log with it.
    #[serde(default)]
    launch_token: Option<String>,
    #[serde(default)]
    client: Option<String>,
}

fn attach_operation(
    guard: &mut ServerState,
    repo_root: Option<&Path>,
    native_id: &crate::NativeSessionId,
    saved_session: &crate::SessionLocator,
) -> Result<AttachReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
    let loadout = db::load(&inner.conn).map_err(|e| ApiFail::Db(e.to_string()))?;
    let world = match loadout.state {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    // an existing exact native session is the same incarnation: the
    // recorded association is native id AND saved locator; a reused
    // id with a different file binds other history, it never resolves
    let existing = world
        .incarnations
        .iter()
        .find(|(_, ctx)| ctx.incarnation.native_id() == Some(native_id));
    if let Some((id, ctx)) = existing {
        if ctx.incarnation.saved_session() != Some(saved_session) {
            return Err(ApiFail::Reject(
                crate::Reject::SavedSessionMismatch,
                Some(format!(
                    "native session {} is already registered as i-{} from {}; attach carries {}",
                    native_id.as_str(),
                    world
                        .incarnation_num(*id)
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| id.0.0.to_string()),
                    ctx.incarnation
                        .saved_session()
                        .map(|l| l.as_path().display().to_string())
                        .unwrap_or_default(),
                    saved_session.as_path().display().to_string(),
                )),
            ));
        }
        return Ok(AttachReply {
            registered: false,
            incarnation: id.0.0,
            num: world
                .incarnation_num(*id)
                .expect("a resident incarnation holds a number"),
            guard: repo_root.map(|repo| {
                crate::activate::guard_path(repo, native_id.as_str())
                    .display()
                    .to_string()
            }),
            records: Vec::new(),
        });
    }
    let (stored, _) = db::record(
        &mut inner.conn,
        &Context::system(),
        Command::RegisterIncarnation {
            native_id: native_id.clone(),
            saved_session: saved_session.clone(),
        },
        db::now_epoch(),
    )?;
    let birth = RecordId(stored[0].seq);
    let reloaded = db::load(&inner.conn).map_err(|e| ApiFail::Db(e.to_string()))?;
    let db::LoadState::Full(world) = reloaded.state else {
        return Err(ApiFail::Degraded("the log will not fold".into()));
    };
    Ok(AttachReply {
        registered: true,
        incarnation: birth.0,
        num: world
            .incarnation_num(crate::IncarnationId(birth))
            .expect("the landed registration folds in"),
        guard: repo_root.map(|repo| {
            crate::activate::guard_path(repo, native_id.as_str())
                .display()
                .to_string()
        }),
        records: stored,
    })
}

async fn native_attach(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: AttachRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = attach_operation(
        &mut guard,
        app.repo_root(),
        &request.native_id,
        &request.session_file,
    );
    // the guard's parent must exist before the integration opens it: a
    // cold state root is readiness, and an uncreatable one is a
    // refusal - never a reply naming a path that cannot exist
    if let (Ok(reply), Some(repo)) = (&outcome, app.repo_root()) {
        if let Some(guard) = &reply.guard {
            if let Some(parent) = Path::new(guard).parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return api_failure(ApiFail::Db(format!(
                        "the session guard's directory cannot be created: {e}"
                    )));
                }
            }
            let _ = repo;
        }
    }
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    // only a fresh birth claims the launch mapping: an existing or
    // forked session reattaching under a stale token must never
    // silently overwrite the door's correlation
    if let (Ok(reply), Some(token)) = (&outcome, request.launch_token) {
        if reply.registered {
            app.bridge
                .remember_launch(token, reply.incarnation, reply.num);
        }
    }
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

#[derive(Deserialize)]
struct MessageSendRequest {
    context: WireContext,
    /// The recipient's public number, the human-facing i-N - the same
    /// token every envelope and contact names
    to: usize,
    #[serde(default)]
    reply_to: Option<usize>,
    #[serde(default)]
    task: Option<usize>,
    body: crate::Prose,
    #[serde(default)]
    client: Option<String>,
}

/// How long a bridge association stays deliverable past its last
/// observed connection: a parked read's brief gap or a transient
/// socket loss must not fail a send the next cycle would carry. The
/// bridge cache prunes associations past the same window.
const BRIDGE_ASSOCIATION_SECS: u64 = 300;

/// How long a send waits for the recipient's integration to accept the
/// delivery into the native input queue. The confirmation bar is queue
/// acceptance, never the recipient's turn or work, so the bound covers
/// transport cycles, not conversation.
const SUBMISSION_CONFIRM_WAIT_SECS: u64 = 15;

/// The confirmation bound's one test seam: a scratch tracker holds the
/// window short so the honest classification at expiry is observable
/// without waiting the production bound. Production never sets it.
fn confirm_wait_secs() -> u64 {
    std::env::var("SACCADE_CONFIRM_WAIT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(SUBMISSION_CONFIRM_WAIT_SECS)
}

/// How long a staged delivery task outlives its send handler: a send
/// whose process dies leaves at most one late delivery, never a retry
/// loop.
const STAGE_TTL_SECS: u64 = 120;

/// One staged native-submission task: the runtime half of a send,
/// born after the record landed and gone when its attempt ends. Never
/// a record and never an obligation - a dead handler leaves at most
/// one late delivery.
struct StagedDelivery {
    incarnation: usize,
    message: usize,
    token: String,
    staged_at: Instant,
}

/// A correlation token for one delivery attempt: transport identity
/// only, never a record reference.
fn delivery_token() -> String {
    let mut bytes = [0u8; 16];
    let filled = std::fs::File::open("/dev/urandom")
        .is_ok_and(|mut file| std::io::Read::read_exact(&mut file, &mut bytes).is_ok());
    if !filled {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mix = nanos ^ ((std::process::id() as u128) << 64);
        for (at, piece) in bytes.chunks_mut(8).enumerate() {
            let shift = 8 * at;
            for (i, slot) in piece.iter_mut().enumerate() {
                *slot = (mix >> (shift + 8 * i)) as u8;
            }
        }
    }
    let mut token = String::with_capacity(32);
    for b in bytes {
        token.push_str(&format!("{b:02x}"));
    }
    token
}

/// Whether the recipient's integration can take a delivery right now.
/// The bridge's live connection - or its recent association, a parked
/// read's brief gap - is the only channel into a native session;
/// beyond it the session's own guard distinguishes the refusal
/// reasons. Runtime evidence only: nothing launches, nothing infers a
/// lifecycle, nothing claims an exited session from uncertain
/// evidence.
enum Channel {
    Deliverable,
    /// proven non-submission: nothing reached the native queue
    Severed(&'static str, String),
}

fn delivery_channel(
    app: &AppState,
    world: &World,
    to: crate::IncarnationId,
    to_num: usize,
    label: &str,
) -> Channel {
    if app.bridge.connected().iter().any(|(i, _)| *i == to.0.0)
        || app.bridge.recently_associated(
            to.0.0,
            std::time::Duration::from_secs(BRIDGE_ASSOCIATION_SECS),
        )
    {
        return Channel::Deliverable;
    }
    let ctx = world
        .incarnations
        .get(&to)
        .expect("the landed message names a resident recipient");
    let crate::Incarnation::Native { native_id, .. } = &ctx.incarnation else {
        return Channel::Severed(
            "invalid_incarnation_id",
            format!("{label} names a recipient with no native session boundary"),
        );
    };
    let Some(repo_root) = app.repo_root() else {
        return Channel::Severed(
            "recipient_unknown",
            format!(
                "{label} was recorded, but i-{to_num}'s guard cannot be probed because the server holds no repo state root; their integration holds no delivery connection and liveness is unknown, so nothing reached their session"
            ),
        );
    };
    let guard = crate::activate::guard_path(repo_root, native_id.as_str());
    match crate::activate::guard_held(&guard) {
        Some(true) => Channel::Severed(
            "recipient_disconnected",
            format!(
                "{label} was recorded, but i-{to_num}'s session is running - its guard is held - while its integration holds no delivery connection; nothing reached their session, and a send never launches or restarts one"
            ),
        ),
        Some(false) => Channel::Severed(
            "recipient_stopped",
            format!(
                "{label} was recorded, but i-{to_num}'s session is not running (its guard is free); nothing reached their session. Resume them explicitly with `sac launch --resume i-{to_num}`, then send again"
            ),
        ),
        None => Channel::Severed(
            "recipient_unknown",
            format!(
                "{label} was recorded, but i-{to_num}'s session guard cannot be probed, so liveness is unknown; their integration holds no delivery connection and nothing reached their session"
            ),
        ),
    }
}

/// One send's native-submission attempt: stage a runtime delivery task
/// for the recipient's integration, then wait for its positive
/// confirmation - the record moving to Submitted, which only the
/// integration's queue-acceptance submission writes. The bound never
/// waits on the recipient's work. Outcome families: submitted; a
/// proven refusal naming why nothing reached the queue; or
/// unacknowledged, the attempt's honest uncertainty.
enum Submission {
    Submitted,
    Refused { code: &'static str, detail: String },
    Unacknowledged { detail: String },
}

async fn attempt_submission(
    app: &AppState,
    to: crate::IncarnationId,
    to_num: usize,
    sent: usize,
    label: &str,
) -> Submission {
    let snapshot = match app.snapshot() {
        Ok(snapshot) => snapshot,
        Err(_) => {
            return Submission::Unacknowledged {
                detail: format!(
                    "{label} was recorded, but the fold could not be read to stage the delivery"
                ),
            };
        }
    };
    let Channel::Severed(code, detail) = delivery_channel(app, &snapshot.world, to, to_num, label)
    else {
        let token = delivery_token();
        app.bridge.stage_delivery(to.0.0, sent, token.clone());
        app.bridge.wake();
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(confirm_wait_secs());
        loop {
            // armed before the read so a confirmation landing between
            // the read and the park cannot be missed
            let wake = app.bridge.notify.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            let submitted = app
                .snapshot()
                .ok()
                .and_then(|s| {
                    s.world
                        .messages
                        .get(&crate::MessageId(RecordId(sent)))
                        .map(|mctx| mctx.state == crate::MessageState::Submitted)
                })
                .unwrap_or(false);
            if submitted {
                app.bridge.drop_stage(&token);
                return Submission::Submitted;
            }
            if tokio::time::Instant::now() >= deadline {
                // retire the stage first, then one last fold read: a
                // confirmation landing inside this gap is still the
                // attempt's own, never an unacknowledged outcome
                let still_staged = app.bridge.drop_stage(&token);
                let submitted = app
                    .snapshot()
                    .ok()
                    .and_then(|s| {
                        s.world
                            .messages
                            .get(&crate::MessageId(RecordId(sent)))
                            .map(|mctx| mctx.state == crate::MessageState::Submitted)
                    })
                    .unwrap_or(false);
                if submitted {
                    return Submission::Submitted;
                }
                // still staged means their integration took no delivery:
                // nothing reached the queue, a proven refusal. A taken
                // stage whose acknowledgement never landed stays
                // uncertain, never a proven failure.
                return match still_staged {
                    true => Submission::Refused {
                        code: "submission_untaken",
                        detail: format!(
                            "{label} was recorded, but their integration took no delivery within the attempt window; nothing reached their session's input queue"
                        ),
                    },
                    false => Submission::Unacknowledged {
                        detail: format!(
                            "{label} was recorded and their integration took the delivery, but no queue-acceptance acknowledgement landed within the attempt window"
                        ),
                    },
                };
            }
            let _ = tokio::time::timeout_at(deadline, wake).await;
        }
    };
    Submission::Refused { code, detail }
}

async fn message_send(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: MessageSendRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    // the sender is the sending session's registered identity, read
    // from the declared scope - never caller-authored prose
    let Some(from) = request.context.scope.as_ref().and_then(|s| s.incarnation) else {
        return api_failure(ApiFail::Reject(
            Reject::InvalidIncarnationId,
            Some(
                "messages send as a registered incarnation: the sending session declares \
                 SACCADE_INCARNATION_SEQ"
                    .into(),
            ),
        ));
    };
    let context = Context {
        actor: request.context.actor,
        tier: match request.context.tier {
            WireTier::Human => Tier::Human,
            WireTier::Agent => Tier::Agent,
        },
        scope: request.context.scope,
    };
    // the public numbers resolve against the fold before the write:
    // a send's reference guards teach before anything lands
    let resolved = (|| {
        let snapshot = app.snapshot().map_err(|read_fail| match read_fail {
            ReadFail::Degraded(d) => ApiFail::Degraded(d.reason),
            ReadFail::Db(reason) => ApiFail::Db(reason),
        })?;
        let to = snapshot
            .world
            .incarnation_at_num(request.to)
            .ok_or(ApiFail::Reject(Reject::InvalidIncarnationId, None))?;
        let reply_to = match request.reply_to {
            Some(n) => Some(
                snapshot
                    .world
                    .message_at_num(n)
                    .ok_or(ApiFail::Reject(Reject::InvalidMessageId, None))?,
            ),
            None => None,
        };
        Ok((to, reply_to))
    })();
    let (to_id, reply_to) = match resolved {
        Ok(resolved) => resolved,
        Err(fail) => return api_failure(fail),
    };
    let command = Command::SendMessage {
        from,
        to: to_id,
        reply_to,
        task: request.task.map(crate::TaskId),
        body: request.body,
    };
    match app.execute(&context, command.clone(), None, received) {
        Ok(stored) => {
            let sent = stored.first().expect("a message lands its one record").seq;
            let numbers = app
                .snapshot()
                .expect("a landed write leaves a foldable world")
                .world;
            let mctx = numbers
                .messages
                .get(&crate::MessageId(RecordId(sent)))
                .expect("the landed message folds in");
            let num = numbers.message_num(crate::MessageId(RecordId(sent)));
            let label = match num {
                Some(n) => format!("m-{n}"),
                None => format!("seq {sent}"),
            };
            let message = json!({
                "seq": sent,
                "num": num,
            });
            let from = json!({
                "seq": mctx.message.from.0 .0,
                "num": numbers.incarnation_num(mctx.message.from),
            });
            // the record stands; the attempt reports its own truth:
            // success only on the queue's confirmed acceptance
            match attempt_submission(&app, to_id, request.to, sent, &label).await {
                Submission::Submitted => (
                    StatusCode::OK,
                    Json(json!({
                        "records": stored,
                        "message": message,
                        "from": from,
                        "submission": {"state": "submitted"},
                    })),
                )
                    .into_response(),
                Submission::Refused { code, detail } => (
                    StatusCode::CONFLICT,
                    Json(json!({
                        "error": {"code": code, "detail": detail},
                        "message": message,
                    })),
                )
                    .into_response(),
                Submission::Unacknowledged { detail } => (
                    StatusCode::CONFLICT,
                    Json(json!({
                        "error": {"code": "submission_unacknowledged", "detail": detail},
                        "message": message,
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => {
            let (status, code, detail) = match e {
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

#[derive(Deserialize)]
struct MessageSubmitRequest {
    /// The submitting session's incarnation: the addressee
    incarnation: usize,
    message: usize,
    #[serde(default)]
    client: Option<String>,
}

async fn message_submit(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: MessageSubmitRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    // the delivery record is machinery's own act, like the registration
    let command = Command::SubmitMessage {
        id: crate::MessageId(RecordId(request.message)),
        by: crate::IncarnationId(RecordId(request.incarnation)),
    };
    match app.execute(&Context::system(), command, None, received) {
        Ok(stored) => (StatusCode::OK, Json(json!({"records": stored}))).into_response(),
        Err(e) => {
            let (status, code, detail) = ApiFail::from(e).parts();
            (
                status,
                Json(json!({"error": {"code": code, "detail": detail}})),
            )
                .into_response()
        }
    }
}

#[derive(Deserialize)]
struct MailboxQuery {
    /// The reader's own incarnation: a bare birth-record position, the
    /// machine identity channel the session attaches with
    incarnation: usize,
}

/// The mailbox JSON: the reader's mail in birth order, any state, both
/// directions, with the log's cursor.
fn mailbox_view(snapshot: &Snapshot, id: crate::IncarnationId) -> Value {
    let world = &snapshot.world;
    let messages: Vec<Value> = world
        .messages
        .iter()
        .filter(|(_, mctx)| mctx.message.to == id || mctx.message.from == id)
        .map(|(mid, mctx)| {
            let sender = &mctx.message.from;
            json!({
                "seq": mid.0 .0,
                "num": world.message_num(*mid),
                "from": {
                    "seq": sender.0 .0,
                    "num": world.incarnation_num(*sender),
                    "actor": mctx.actor.as_str(),
                },
                "to": {
                    "seq": mctx.message.to.0 .0,
                    "num": world.incarnation_num(mctx.message.to),
                },
                "reply_to": mctx.message.reply_to.map(|r| world.message_num(r)),
                "task": mctx.message.task.map(|t| t.0),
                "state": match mctx.state {
                    crate::MessageState::Pending => "pending",
                    crate::MessageState::Submitted => "submitted",
                },
                "body": mctx.message.body.as_str(),
                "born_at": mctx.born_at,
            })
        })
        .collect();
    json!({ "messages": messages, "cursor": snapshot.rows.last().map(|r| r.seq) })
}

/// The reader's own mail, in birth order, any state: a descriptive
/// read over the records. `pending` is a recorded message without a
/// confirmed submission - evidence of its attempt, never a promise of
/// later delivery - and `submitted` means the recipient's native input
/// queue accepted it.
async fn messages_pending_retired() -> Response {
    (
        StatusCode::GONE,
        Json(json!({"error": {"code": "retired_messages_pending", "detail": "the pending transport retired: recorded mail is evidence and is never delivered by machinery; history reads use GET /api/v1/messages?incarnation=N, deliveries use GET /api/v1/messages/deliver; upgrade the saccade native integration"}})),
    )
        .into_response()
}

/// The descriptive history read: a reader's own mail, in birth order,
/// any state, read-only by construction. `pending` is a recorded
/// message without a confirmed submission - evidence of its attempt,
/// never a promise of later delivery - and `submitted` means the
/// recipient's native input queue accepted it.
async fn messages_history(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<MailboxQuery>,
) -> Response {
    let snapshot = match app.snapshot() {
        Ok(s) => s,
        Err(read_fail) => {
            let (status, code, detail) = match read_fail {
                ReadFail::Degraded(d) => ApiFail::Degraded(d.reason).parts(),
                ReadFail::Db(reason) => ApiFail::Db(reason).parts(),
            };
            return (
                status,
                Json(json!({"error": {"code": code, "detail": detail}})),
            )
                .into_response();
        }
    };
    let id = crate::IncarnationId(RecordId(query.incarnation));
    if !snapshot.world.incarnations.contains_key(&id) {
        return api_failure(ApiFail::Reject(Reject::InvalidIncarnationId, None));
    }
    (StatusCode::OK, Json(mailbox_view(&snapshot, id))).into_response()
}

#[derive(Deserialize)]
struct DeliveryQuery {
    /// The delivering session's incarnation: the addressee taking its
    /// own staged mail
    incarnation: usize,
    /// Park until a staged delivery task exists or this many
    /// milliseconds pass, capped at 30s: the integration's parked
    /// transport read, woken by a staged task, never a timer over the
    /// log. Absent, the read returns immediately.
    #[serde(default)]
    wait_ms: Option<u64>,
}

/// The recipient integration's transport read: the staged delivery
/// tasks of send attempts now in flight, each handed exactly once.
/// This read is the bridge itself - it announces the association for
/// its duration - and it never drains recorded mail: a recorded
/// message without confirmed submission is evidence, and no machinery
/// carries it later.
async fn messages_deliver(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<DeliveryQuery>,
) -> Response {
    let _live = BridgeLive::new(&app.bridge, query.incarnation);
    let wait_ms = query.wait_ms.unwrap_or(0).min(30_000);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(wait_ms);
    let id = crate::IncarnationId(RecordId(query.incarnation));
    loop {
        // armed before the read so a staged task landing between the
        // read and the park cannot be missed
        let wake = app.bridge.notify.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        let snapshot = match app.snapshot() {
            Ok(s) => s,
            Err(read_fail) => {
                let (status, code, detail) = match read_fail {
                    ReadFail::Degraded(d) => ApiFail::Degraded(d.reason).parts(),
                    ReadFail::Db(reason) => ApiFail::Db(reason).parts(),
                };
                return (
                    status,
                    Json(json!({"error": {"code": code, "detail": detail}})),
                )
                    .into_response();
            }
        };
        if !snapshot.world.incarnations.contains_key(&id) {
            return api_failure(ApiFail::Reject(Reject::InvalidIncarnationId, None));
        }
        let staged = app.bridge.take_staged(query.incarnation);
        let deliveries: Vec<Value> = staged
            .iter()
            .filter_map(|(token, seq)| {
                let mctx = snapshot
                    .world
                    .messages
                    .get(&crate::MessageId(RecordId(*seq)))?;
                // a stage whose record already reads submitted was
                // confirmed out-of-band: nothing left to hand
                if mctx.state != crate::MessageState::Pending {
                    return None;
                }
                let sender = &mctx.message.from;
                Some(json!({
                    "token": token,
                    "message": {
                        "seq": seq,
                        "num": snapshot.world.message_num(crate::MessageId(RecordId(*seq))),
                        "from": {
                            "seq": sender.0 .0,
                            "num": snapshot.world.incarnation_num(*sender),
                            "actor": mctx.actor.as_str(),
                        },
                        "to": {
                            "seq": mctx.message.to.0 .0,
                            "num": snapshot.world.incarnation_num(mctx.message.to),
                        },
                        "reply_to": mctx.message.reply_to.map(|r| snapshot.world.message_num(r)),
                        "task": mctx.message.task.map(|t| t.0),
                        "body": mctx.message.body.as_str(),
                        "born_at": mctx.born_at,
                    },
                }))
            })
            .collect();
        if !deliveries.is_empty() || wait_ms == 0 || tokio::time::Instant::now() >= deadline {
            return (StatusCode::OK, Json(json!({ "deliveries": deliveries }))).into_response();
        }
        let _ = tokio::time::timeout_at(deadline, wake).await;
    }
}

/// The bridge liveness cache as a read: which incarnations hold a
/// delivery connection right now. Operational only - absence is not
/// proof a native process exited.
async fn native_bridge(State(app): State<AppState>) -> Json<Value> {
    let incarnations: Vec<Value> = app
        .bridge
        .connected()
        .into_iter()
        .map(|(incarnation, connections)| {
            json!({"incarnation": incarnation, "connections": connections})
        })
        .collect();
    Json(json!({ "incarnations": incarnations }))
}

#[derive(Deserialize)]
struct LaunchQuery {
    token: String,
}

/// Which incarnation registered under a compose launch token: how the
/// launching door learns the birth it caused. Runtime correlation only.
async fn native_launch(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<LaunchQuery>,
) -> Response {
    match app.bridge.launch(&query.token) {
        Some((incarnation, num)) => (
            StatusCode::OK,
            Json(json!({"incarnation": incarnation, "num": num})),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "code": "launch_unknown",
                    "detail": "no composed session registered under this launch token"
                }
            })),
        )
            .into_response(),
    }
}

fn api_failure(fail: ApiFail) -> Response {
    let (status, code, detail) = fail.parts();
    (
        status,
        Json(json!({"error": {"code": code, "detail": detail}})),
    )
        .into_response()
}

/// The handshake surface: which build is serving, nothing more
async fn version(State(app): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "db": app.db_path().display().to_string(),
    }))
}

async fn records(State(app): State<AppState>) -> Response {
    match app.snapshot() {
        Ok(s) => Json(json!({ "records": s.rows })).into_response(),
        Err(ReadFail::Degraded(d)) => Json(json!({ "records": d.rows })).into_response(),
        Err(ReadFail::Db(reason)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": "database_error", "detail": reason}})),
        )
            .into_response(),
    }
}
