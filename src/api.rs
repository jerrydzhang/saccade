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
use crate::{Command, MessageId, Reject};

enum ServerState {
    Ready(Inner),
    Degraded(String, rusqlite::Connection),
}

struct Inner {
    conn: rusqlite::Connection,
    /// The tracker's own db: the worktree home derives from it
    db_path: std::path::PathBuf,
}

/// The sole writer's submit serialization: one lock per recipient
/// incarnation, so concurrent sends to one recipient submit in record
/// order even when its address moves mid-flight - an address key
/// would let sends on either side of a move take different locks and
/// interleave. Runtime facts only - never record identity, authority,
/// or obligation; lost on restart.
pub struct SubmitLocks {
    locks: std::sync::Mutex<HashMap<usize, Arc<Mutex<()>>>>,
}

impl Default for SubmitLocks {
    fn default() -> Self {
        SubmitLocks {
            locks: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl SubmitLocks {
    /// Acquire the per-recipient lock: concurrent sends to one
    /// recipient serialize here; different recipients never contend.
    /// The lock is synchronous because the adapter's submit is a
    /// blocking subprocess call (the bounded attempt's own bound
    /// contains it).
    pub fn acquire(&self, to: crate::IncarnationId) -> Arc<Mutex<()>> {
        let mut map = self.locks.lock().expect("the submit locks hold");
        map.entry(to.0.0)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

/// The landing wake: a parked read's fast path, fired when this
/// server's own door lands a write. Never a timer, never delivery -
/// parked reads re-check their predicate against the fold.
#[derive(Clone, Default)]
pub struct LandingWake {
    notify: Arc<tokio::sync::Notify>,
}

/// The sole writer's state: every face loads the log fresh under the
/// one lock, so no read can serve a world the file contradicts.
#[derive(Clone)]
pub struct AppState {
    submit_locks: Arc<SubmitLocks>,
    /// the landing wake: parked reads' fast path when this server's
    /// own door lands a write
    pub landing: LandingWake,
    inner: Arc<Mutex<ServerState>>,
    db_path: std::path::PathBuf,
    attempts: Attempts,
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
            submit_locks: Arc::new(SubmitLocks::default()),
            landing: LandingWake::default(),
            inner: Arc::new(Mutex::new(state)),
            db_path: canonical,
            attempts: Attempts::beside(db_path),
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
                // a landed write may be a parked read's predicate: the
                // landing is the wake, never a timer
                self.landing.notify.notify_waiters();
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

impl From<HostFail> for ApiFail {
    fn from(e: HostFail) -> Self {
        ApiFail::Host(e)
    }
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
                    HostFail::NoPi | HostFail::NoDirenv => StatusCode::INTERNAL_SERVER_ERROR,
                    // lifecycle host answers the server could not read:
                    // the environment's break, never the caller's input
                    HostFail::TerminalList(_) => StatusCode::BAD_GATEWAY,
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
    // a send command rides the send operation like the send face: the
    // per-recipient lock must cover its record, and the attempt's truth
    // rides the reply as data
    if let Command::SendMessage { .. } = &command {
        let send_app = app.clone();
        let send_context = context.clone();
        let send_command = command.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            send_operation(&send_app, &send_context, send_command, received)
        })
        .await
        .expect("the send operation does not panic");
        return match outcome {
            Ok((stored, submission)) => {
                let (_, message, from) = message_facts(&app, &stored);
                let mut reply = json!({"records": &stored});
                reply["message"] = message;
                reply["from"] = from;
                reply["submission"] = submission.json();
                (StatusCode::OK, Json(reply)).into_response()
            }
            Err(e) => execute_failure_response(&app, e, &command),
        };
    }
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
            (StatusCode::OK, Json(reply)).into_response()
        }
        Err(e) => execute_failure_response(&app, e, &command),
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
        .route("/api/v1/incarnations/register", post(incarnation_register))
        .route("/api/v1/incarnations/address", post(incarnation_address))
        .route("/api/v1/messages/send", post(message_send))
        .route("/api/v1/messages", get(messages_history))
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
        let wake = app.landing.notify.notified();
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
    /// The harness the fresh session runs; absent folds as pi, the
    /// same decode law the registration and attach carry
    #[serde(default)]
    harness: crate::Harness,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum WorkbenchLaunchWorkspace {
    /// An existing directory, exactly as the CLI door demands
    Existing(String),
}

/// What a submission did: the host's terminal acceptance and the
/// workspace it names - never a claim that a session runs. The
/// placement is what the door declared at birth; the refusal, when it
/// stands, discloses that the declaration did not land. The reply
/// names the harness the launch actually composed under.
#[derive(Serialize)]
pub struct LaunchReply {
    pub submitted: bool,
    pub workspace: String,
    pub terminal: String,
    pub space: String,
    pub tab: String,
    pub place_refusal: Option<String>,
    pub harness: crate::Harness,
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
    let outcome = tokio::task::spawn_blocking(move || {
        launch_operation(&blocking_app, &request.workspace, request.harness)
    })
    .await
    .expect("the launch operation does not panic");
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    // the door-birthed incarnation lands under the writer lock, right
    // after the address returns: identity and address exist at launch.
    // The birth rides the same resolve-or-recover the register
    // endpoint uses - the newborn's own attach can win the unlocked
    // host-effect window and bind the locator first, and the recovery
    // folds that into one incarnation with this launch's address
    if let Ok((_, Some((native_id, address, harness)))) = &outcome {
        let native_id = match crate::NativeSessionId::new(native_id.clone()) {
            Ok(id) => id,
            Err(e) => {
                return api_failure(ApiFail::Db(format!("{:?}", e)));
            }
        };
        if let ServerState::Ready(inner) = &mut *guard
            && let Err(e) =
                incarnation_register_operation(inner, native_id, address.clone(), false, *harness)
        {
            return api_failure(e);
        }
    }
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    drop(guard);
    match outcome {
        Ok((reply, _)) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

/// The serving edge's face for a shared launch refusal: the code is
/// the composition's own, the status is the stage's - a caller-named
/// workspace refuses as the caller's input, every environment stage
/// as the deployment's break, and the host's own answer as a bad
/// gateway.
fn launch_fail(fail: crate::launch::LaunchFail) -> ApiFail {
    match fail {
        crate::launch::LaunchFail::Refused { code, detail } => {
            let status = match code {
                "workspace_path_invalid" => StatusCode::BAD_REQUEST,
                "workspace_cleaned" | "workspace_ambiguous_path" => StatusCode::CONFLICT,
                "launch_host_refused" => StatusCode::BAD_GATEWAY,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            ApiFail::Refused(status, code, detail)
        }
        // the workbench never resumes: a door-shaped refusal has no
        // wire consumer here
        crate::launch::LaunchFail::Usage(detail) => {
            ApiFail::Refused(StatusCode::BAD_REQUEST, "usage", detail)
        }
        crate::launch::LaunchFail::Degraded(reason) => ApiFail::Degraded(reason),
    }
}

/// The birth facts a submission carries for the in-process door: the
/// pre-allocated native id, the returned address, the composed
/// harness.
type LaunchBirth = (String, String, crate::Harness);

/// One workbench launch: the server's own published URL for the
/// routing, the shared composition the CLI doors use, the host
/// submission. The stages and their order are the CLI door's; the
/// fresh birth facts ride the answer for the in-process registration.
fn launch_operation(
    app: &AppState,
    workspace: &WorkbenchLaunchWorkspace,
    harness: crate::Harness,
) -> Result<(LaunchReply, Option<LaunchBirth>), ApiFail> {
    let WorkbenchLaunchWorkspace::Existing(path) = workspace;
    let Some(url) = app.server_url().map(str::to_string) else {
        return Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "launch_no_server_url",
            "the serving edge cannot name its own address; a launch cannot route the session it starts"
                .into(),
        ));
    };
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    // one world read for the guard and the composition
    let conn = db::open_read(app.db_path()).map_err(|e| ApiFail::Db(e.to_string()))?;
    let world = match db::load(&conn)
        .map_err(|e| ApiFail::Db(e.to_string()))?
        .state
    {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    let composed = crate::launch::compose(crate::launch::LaunchRequest {
        world: &world,
        repo: &repo_root,
        url,
        exe: std::env::current_exe().ok(),
        arm: crate::launch::LaunchArm::Fresh {
            workspace: Path::new(path),
            prompt: &[],
            harness,
        },
    })
    .map_err(launch_fail)?;
    let submitted = crate::launch::submit(composed).map_err(launch_fail)?;
    let reply = LaunchReply {
        submitted: true,
        workspace: submitted.workspace.clone(),
        terminal: submitted.terminal.clone(),
        space: submitted.space,
        tab: submitted.tab,
        place_refusal: submitted.place_refusal,
        harness,
    };
    let birth = submitted
        .fresh
        .map(|fresh| (fresh.native_id, submitted.terminal, harness));
    Ok((reply, birth))
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
/// prove quiescence through the host's liveness answer, then land the
/// System change fact - the fold's change cache is the whole effect,
/// and resume reads the workspace from it. Busy and unknown refuse
/// before anything records.
fn change_operation(guard: &mut ServerState, num: usize, to: &str) -> Result<ChangeReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
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
    let (saved_session, carried, adopted) = match &world.incarnations[&id].incarnation {
        crate::Incarnation::Native {
            saved_session,
            address,
            adopted,
            ..
        } => {
            // the change needs a from fact: a launch-door birth whose
            // locator pi has not bound yet has no saved launch cwd and
            // no recorded change, so it refuses rather than guessing
            let Some(saved_session) = saved_session else {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_REQUEST,
                    "saved_session_missing",
                    format!(
                        "i-{num}'s saved session locator is not bound yet: the session has not saved, so there is no launch fact to change"
                    ),
                ));
            };
            (saved_session.clone(), address.clone(), *adopted)
        }
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
    // the target is validated before any probe runs: an explicit
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

    // quiescence is host evidence, never a guess, and only for the
    // launched kind: a launched terminal dies with its session, so a
    // live terminal names a live session. An adopted pane is the
    // caller's own program and outlives the session - terminal
    // liveness is never session liveness for it, and the native
    // duplicate-attach refusal is that kind's backstop. An
    // unprobeable answer is unknown and refuses; an unaddressed
    // recipient is the pre-B shape, whose migration is exit + resume
    let Some(address) = carried.clone().filter(|_| !adopted) else {
        if carried.is_none() {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "recipient_unaddressed",
                format!(
                    "i-{num} carries no host address: a pre-B session's switch needs its migration (exit + resume) first"
                ),
            ));
        }
        // an adopted pane outlives its session: no probe, the native
        // duplicate-attach refusal is this kind's backstop
        return record_change(inner, id, num, &world, saved_session, target);
    };
    let host = crate::herdr::HerdrCore::resolved().map_err(|e| {
        ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "host_unavailable",
            e.to_string(),
        )
    })?;
    use crate::host_contract::HostContract;
    match host.live(&crate::host_contract::HostAddress(address)) {
        crate::host_contract::Liveness::Gone => {}
        crate::host_contract::Liveness::Live => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "session_live",
                format!(
                    "i-{num}'s terminal is live at its recorded host address; exit that session first - this route moves stopped sessions only"
                ),
            ));
        }
        crate::host_contract::Liveness::Unknown(detail) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "session_quiescence_unknown",
                format!(
                    "i-{num}'s liveness cannot be read from the host ({detail}); nothing was moved"
                ),
            ));
        }
    }
    record_change(inner, id, num, &world, saved_session, target)
}

/// The change's whole effect is the record: the fold carries where a
/// session lives, and resume reads it there - no session file is
/// touched, so the record cannot strand a moved file. The from fact
/// is the recorded change when one stands, else the saved header's
/// launch cwd.
fn record_change(
    inner: &mut Inner,
    id: crate::IncarnationId,
    num: usize,
    world: &World,
    saved_session: crate::SessionLocator,
    target: WorktreePath,
) -> Result<ChangeReply, ApiFail> {
    let from = match world.workspace_changes.get(&id) {
        Some(change) => change.worktree.as_path().display().to_string(),
        None => {
            let header = crate::pi_session::read_header(saved_session.as_path()).map_err(|e| {
                let (status, code) = match &e {
                    crate::pi_session::HandoffFail::Missing(_) => {
                        (StatusCode::BAD_REQUEST, "saved_session_missing")
                    }
                    _ => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "saved_session_unreadable",
                    ),
                };
                ApiFail::Refused(status, code, e.to_string())
            })?;
            header.cwd
        }
    };
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
            format!("the change record could not land: {e}; nothing moved - retry the switch"),
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
    let outcome = change_operation(&mut guard, request.incarnation, &request.to);
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
        // one read answers idempotence: the recorded closure holds
        // only while the host names no terminal at the exact path
        let host = crate::herdr::HerdrCore::resolved().map_err(|e| {
            ApiFail::Refused(
                StatusCode::BAD_GATEWAY,
                "workspace_host_unverified",
                format!(
                    "terminal state unverified at a workspace recorded Cleaned: {e}; the idempotence answer names only what verified - retry when the host answers"
                ),
            )
        })?;
        match crate::host::workspace_terminals(&host, &dir_str) {
            Ok(terminals) if terminals.is_empty() => {
                verified.push("verified: the host names no terminals for the workspace".into());
            }
            Ok(terminals) => {
                return Err(ApiFail::Refused(
                    StatusCode::CONFLICT,
                    "workspace_terminals_open",
                    format!(
                        "the host names {} live terminal(s) at a workspace recorded Cleaned; the recorded closure does not hold - inspect and retry",
                        terminals.len()
                    ),
                ));
            }
            Err(e) => {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_GATEWAY,
                    "workspace_host_unverified",
                    format!(
                        "terminal state unverified at a workspace recorded Cleaned: {e}; the idempotence answer names only what verified - retry when the host answers"
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

    // the host stage: enumerate by exact path over one read, close
    // each address, and take the finished fact from the close replies
    // themselves - the host's own read of what it ended, never a
    // re-list; a reply without the fact refuses rather than guessing
    let host = crate::herdr::HerdrCore::resolved().map_err(|e| {
        ApiFail::Refused(
            StatusCode::BAD_GATEWAY,
            "workspace_host_unverified",
            format!(
                "terminal closure is unverified: {e}; no checkout removal ran - retry clean when the host answers"
            ),
        )
    })?;
    use crate::host_contract::{CloseFact, HostContract};
    // the enumeration's own face: an unreadable table is the list
    // read's refusal, never the closure's
    let addresses = crate::host::workspace_terminals(&host, &dir_str)?;
    for address in &addresses {
        match host.close(&crate::host_contract::HostAddress(address.clone())) {
            Ok(CloseFact::Finished) => {}
            Ok(CloseFact::Unconfirmed(evidence)) => {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_GATEWAY,
                    "workspace_host_unverified",
                    format!(
                        "the close of terminal {address} named no finished state ({evidence}); no checkout removal ran - retry clean"
                    ),
                ));
            }
            Err(e) => {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_GATEWAY,
                    "workspace_host_unverified",
                    format!(
                        "the host refused the close of terminal {address}: {e}; no checkout removal ran - retry clean"
                    ),
                ));
            }
        }
    }
    if addresses.is_empty() {
        verified.push("verified: the host names no terminals for the workspace".into());
    } else {
        verified.push(format!(
            "closed {} terminal(s); the host named each finished",
            addresses.len()
        ));
    }
    // absence: one more read, a verification read - a terminal born
    // between the enumeration and the closes surfaces here, and the
    // record never lands over an open terminal
    let again = crate::host::workspace_terminals(&host, &dir_str)?;
    if !again.is_empty() {
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_terminals_open",
            format!(
                "the host still names {} terminal(s) for {}; no checkout removal ran",
                again.len(),
                dir_str
            ),
        ));
    }
    verified.push("verified: the host names no terminals for the workspace".into());

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
                let read = crate::herdr::HerdrCore::resolved()
                    .map_err(|e| e.to_string())
                    .and_then(|h| {
                        crate::host::workspace_terminals(&h, &dir_str).map_err(|e| e.to_string())
                    });
                match read
                {
                    Ok(t) if !t.is_empty() => {
                        conflicts.push(entry(format!(
                            "the host names {} live terminal(s) at a workspace recorded Cleaned",
                            t.len()
                        )));
                    }
                    Ok(_) => preserved.push(entry(
                        "verified: recorded Cleaned, checkout absent, the host names no terminals there"
                            .into(),
                    )),
                    Err(e) => {
                        conflicts.push(entry(format!("terminal state unreadable: {e}")));
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
                let read = crate::herdr::HerdrCore::resolved()
                    .map_err(|e| e.to_string())
                    .and_then(|h| {
                        crate::host::workspace_terminals(&h, &dir_str).map_err(|e| e.to_string())
                    });
                match read {
                    Ok(t) if !t.is_empty() => {
                        conflicts.push(entry(format!(
                            "the host still names {} live terminal(s) there",
                            t.len()
                        )));
                    }
                    Err(e) => {
                        conflicts.push(entry(format!("terminal state unreadable: {e}")));
                    }
                    Ok(_) => {
                        if apply {
                            to_record.push(Event::WorkspaceCleaned { workspace: birth });
                        }
                        cleaned.push(entry(format!(
                            "checkout absent and the host names no terminals there{}",
                            if apply {
                                "; Cleaned recorded now"
                            } else {
                                "; dry run"
                            }
                        )));
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
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct AttachRequest {
    native_id: crate::NativeSessionId,
    session_file: crate::SessionLocator,
    /// The harness the attaching session runs under; the pi
    /// extension sends none and folds as pi.
    #[serde(default)]
    harness: crate::Harness,
    #[serde(default)]
    client: Option<String>,
}

fn attach_operation(
    guard: &mut ServerState,
    native_id: &crate::NativeSessionId,
    saved_session: &crate::SessionLocator,
    harness: crate::Harness,
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
    // recorded association is harness AND native id AND saved locator;
    // a reused id under another harness, or with a different file,
    // binds other history, it never resolves
    let existing = world.incarnations.iter().find(|(_, ctx)| {
        ctx.incarnation.native_id() == Some(native_id) && ctx.incarnation.harness() == Some(harness)
    });
    if let Some((id, ctx)) = existing {
        match ctx.incarnation.saved_session() {
            // a launch-door birth precedes pi's first save: the attach
            // carries the locator and binds it, completing the birth
            None => {
                let command = Command::BindSavedSession {
                    incarnation: *id,
                    saved_session: saved_session.clone(),
                };
                let (bound, _) = db::record(
                    &mut inner.conn,
                    &Context::system(),
                    command,
                    db::now_epoch(),
                )?;
                return Ok(AttachReply {
                    registered: false,
                    incarnation: id.0.0,
                    num: world
                        .incarnation_num(*id)
                        .expect("a resident incarnation holds a number"),
                    records: bound,
                });
            }
            Some(bound) if bound == saved_session => {}
            Some(_) => {
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
                        saved_session.as_path().display(),
                    )),
                ));
            }
        }
        return Ok(AttachReply {
            registered: false,
            incarnation: id.0.0,
            num: world
                .incarnation_num(*id)
                .expect("a resident incarnation holds a number"),
            records: Vec::new(),
        });
    }
    let (stored, _) = db::record(
        &mut inner.conn,
        &Context::system(),
        Command::RegisterIncarnation {
            native_id: native_id.clone(),
            saved_session: Some(saved_session.clone()),
            address: None,
            adopted: false,
            harness,
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
        records: stored,
    })
}

/// The launch door's own birth: the pre-allocated native id and the
/// opaque host address, recorded before pi's first save - the
/// association exists at launch time, no correlation token, no wait.
#[derive(Deserialize)]
struct IncarnationRegisterRequest {
    native_id: crate::NativeSessionId,
    address: String,
    /// The address kind: the here door adopts the caller terminal,
    /// every other birth launches a new terminal.
    #[serde(default)]
    adopted: bool,
    /// The harness the fresh session runs under; absent folds as pi.
    #[serde(default)]
    harness: crate::Harness,
    #[serde(default)]
    client: Option<String>,
}

async fn incarnation_register(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: IncarnationRegisterRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = match &mut *guard {
        ServerState::Ready(inner) => incarnation_register_operation(
            inner,
            request.native_id,
            request.address,
            request.adopted,
            request.harness,
        ),
        ServerState::Degraded(reason, _) => Err(ApiFail::Degraded(reason.clone())),
    };
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    match outcome {
        Ok((records, incarnation, registered)) => (
            StatusCode::OK,
            Json(json!({
                "registered": registered,
                "incarnation": incarnation,
                "records": records,
            })),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

/// Recovery is the attach-born unaddressed case only: an addressed
/// session keeps the fold's registration refusal - address moves
/// belong to the relaunch door.
fn incarnation_register_operation(
    inner: &mut Inner,
    native_id: crate::NativeSessionId,
    address: String,
    adopted: bool,
    harness: crate::Harness,
) -> Result<(Vec<StoredRecord>, usize, bool), ApiFail> {
    let loadout = db::load(&inner.conn).map_err(|e| ApiFail::Db(e.to_string()))?;
    let db::LoadState::Full(world) = loadout.state else {
        return Err(ApiFail::Degraded("the log will not fold".into()));
    };
    let known = world.incarnations.iter().find(|(_, ctx)| {
        ctx.incarnation.native_id() == Some(&native_id)
            && ctx.incarnation.harness() == Some(harness)
    });
    match known {
        Some((id, ctx))
            if ctx.incarnation.saved_session().is_some() && ctx.incarnation.address().is_none() =>
        {
            let command = Command::AssignHostAddress {
                incarnation: *id,
                address: Some(address),
                adopted,
            };
            let (stored, _) = db::record(
                &mut inner.conn,
                &Context::system(),
                command,
                db::now_epoch(),
            )?;
            Ok((stored, id.0.0, false))
        }
        _ => {
            let command = Command::RegisterIncarnation {
                native_id,
                saved_session: None,
                address: Some(address),
                adopted,
                harness,
            };
            let (stored, _) = db::record(
                &mut inner.conn,
                &Context::system(),
                command,
                db::now_epoch(),
            )?;
            let birth = stored[0].seq;
            Ok((stored, birth, true))
        }
    }
}

/// The association's address moves with a relaunch: the same
/// incarnation, another terminal.
#[derive(Deserialize)]
struct IncarnationAddressRequest {
    incarnation: usize,
    address: String,
    #[serde(default)]
    adopted: bool,
    #[serde(default)]
    client: Option<String>,
}

async fn incarnation_address(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: IncarnationAddressRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = match &mut *guard {
        ServerState::Ready(inner) => {
            let command = Command::AssignHostAddress {
                incarnation: crate::IncarnationId(RecordId(request.incarnation)),
                address: Some(request.address),
                adopted: request.adopted,
            };
            db::record(
                &mut inner.conn,
                &Context::system(),
                command,
                db::now_epoch(),
            )
            .map(|(stored, _)| stored)
            .map_err(ApiFail::from)
        }
        ServerState::Degraded(reason, _) => Err(ApiFail::Degraded(reason.clone())),
    };
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    match outcome {
        Ok(records) => (
            StatusCode::OK,
            Json(json!({"assigned": true, "records": records})),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
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
        &request.native_id,
        &request.session_file,
        request.harness,
    );
    integration_outcome(&app, &guard, None, &received, started, &outcome);
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

// -- the send path at the sole writer -------------------------------------

/// One send's submission attempt at the sole writer, after its record
/// landed: the adapter's bounded submit to the recipient's recorded
/// address. Submitted means the host accepted the ordered submission
/// and the outcome record landed; Unrecorded means the host accepted
/// while the outcome record could not land (the message reads pending
/// though the submission is real); Refused means nothing reached the
/// session and the message stays pending - evidence of the attempt,
/// never a promise.
enum SubmitOutcome {
    Submitted,
    Unrecorded,
    Refused { code: &'static str, detail: String },
}

impl SubmitOutcome {
    /// The reply face's summary: the state plus, on refusal, the
    /// evidence.
    fn json(&self) -> Value {
        match self {
            SubmitOutcome::Submitted => json!({"state": "submitted"}),
            SubmitOutcome::Unrecorded => json!({
                "state": "unrecorded",
                "detail": "the host accepted the submission but its outcome record could not land; the message reads pending",
            }),
            SubmitOutcome::Refused { code, detail } => {
                json!({"state": "refused", "code": code, "detail": detail})
            }
        }
    }
}

/// The whole send path every face shares: serialize on the
/// recipient's host address, record the message, submit through the
/// adapter, then record the outcome. The per-recipient lock covers
/// record and submit together, so concurrent sends to one recipient
/// submit in record order; the adapter's own bound contains a hung
/// core, and the lock order is always address-then-writer, never the
/// reverse, so the server cannot deadlock. Runs on a blocking thread
/// (the launch door's pattern): the adapter ask is a subprocess call.
/// The wire envelope a submission carries: machine-supplied origin on
/// one ASCII line - the dense message number, the sender's incarnation
/// number, and the reply and task segments only when they exist -
/// then a blank line and the body untouched. Wire-only: the record's
/// body and every read stay as authored; the envelope exists at the
/// submit door alone, so a recipient knows who is talking and holds
/// the id to reply to without a prior query.
fn wire_envelope(world: &World, id: MessageId) -> Option<String> {
    let mail = &world.messages.get(&id)?.message;
    let num = world.message_num(id)?;
    let from = world.incarnation_num(mail.from)?;
    let mut head = format!("[saccade message m-{num} | from i-{from}");
    if let Some(reply) = mail.reply_to {
        head += &format!(" | replying to m-{}", world.message_num(reply)?);
    }
    if let Some(task) = mail.task {
        head += &format!(" | task t-{}", task.0);
    }
    head.push(']');
    Some(format!("{head}\n\n{}", mail.body.as_str()))
}

fn send_operation(
    app: &AppState,
    context: &Context,
    command: Command,
    received: AsReceived,
) -> Result<(Vec<StoredRecord>, SubmitOutcome), ExecuteFail> {
    let Command::SendMessage { to, .. } = &command else {
        unreachable!("the send operation serves SendMessage alone");
    };
    let to = *to;
    // the recipient's id names the lock; the address is resolved
    // fresh under it, so a move between record and submit submits at
    // the new association without surrendering serialization
    let addressed = |world: &World| {
        world
            .incarnations
            .get(&to)
            .and_then(|ctx| ctx.incarnation.address().map(str::to_string))
    };
    let lock = app.submit_locks.acquire(to);
    // the guard, not the Arc: the serialization is the lock itself,
    // held across the record and the submit
    let _serialized = lock.lock().expect("the submit lock holds");
    let stored = app.execute(context, command, None, received)?;
    let sent = stored.first().expect("a message lands its one record").seq;
    // the freshest address under the lock: an unaddressed recipient is
    // the pre-B shape - the record stands as evidence, nothing submits
    let snapshot = app.snapshot().ok();
    let Some(address) = snapshot.as_ref().and_then(|s| addressed(&s.world)) else {
        return Ok((
            stored,
            SubmitOutcome::Refused {
                code: "recipient_unaddressed",
                detail: "the recipient carries no host address: pre-B sessions lose inbound \
                         delivery; the sanctioned migration is exit + resume"
                    .into(),
            },
        ));
    };
    // the envelope composes from the recorded facts at the submit
    // door; the snapshot that resolved the address holds the
    // just-landed message, so its numbers are fold-guaranteed
    let message_id = MessageId(RecordId(sent));
    let wire_text = snapshot
        .as_ref()
        .and_then(|s| wire_envelope(&s.world, message_id))
        .expect("the just-landed message composes its wire envelope");
    use crate::host_contract::HostContract;
    let submission = match crate::herdr::HerdrCore::resolved() {
        Ok(host) => host.submit(&crate::host_contract::HostAddress(address), &wire_text),
        Err(e) => {
            return Ok((
                stored,
                SubmitOutcome::Refused {
                    code: "host_unavailable",
                    detail: e.to_string(),
                },
            ));
        }
    };
    match submission {
        Ok(()) => {
            // the addressee's machinery act: the outcome record is the
            // recipient's, never the sender's
            let outcome_record = Command::SubmitMessage {
                id: crate::MessageId(RecordId(sent)),
                by: to,
            };
            let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
            let recorded = match &mut *guard {
                ServerState::Ready(inner) => db::record(
                    &mut inner.conn,
                    &Context::system(),
                    outcome_record,
                    db::now_epoch(),
                )
                .is_ok(),
                ServerState::Degraded(..) => false,
            };
            drop(guard);
            Ok((
                stored,
                if recorded {
                    SubmitOutcome::Submitted
                } else {
                    SubmitOutcome::Unrecorded
                },
            ))
        }
        Err(refusal) => Ok((
            stored,
            SubmitOutcome::Refused {
                code: match &refusal {
                    crate::host_contract::SubmitRefusal::AddressNotLive(_) => "recipient_not_live",
                    crate::host_contract::SubmitRefusal::HostRefused(_) => "submission_refused",
                    crate::host_contract::SubmitRefusal::Unknown(_) => "submission_unknown",
                    crate::host_contract::SubmitRefusal::Unresolved => "submission_unresolved",
                },
                detail: refusal.to_string(),
            },
        )),
    }
}

/// The message facts both send faces reply with: the birth position,
/// the dense m-N, and the sender's resolved identity.
fn message_facts(app: &AppState, stored: &[StoredRecord]) -> (usize, Value, Value) {
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
    let message = json!({
        "seq": sent,
        "num": num,
    });
    let from = json!({
        "seq": mctx.message.from.0 .0,
        "num": numbers.incarnation_num(mctx.message.from),
    });
    (sent, message, from)
}

/// The refusal face a refused write shares: the rejection teaches
/// where the world knows the intended target.
fn execute_failure_response(app: &AppState, e: ExecuteFail, command: &Command) -> Response {
    let (status, code, detail) = match e {
        ExecuteFail::Reject(reject) => {
            let taught = app
                .snapshot()
                .ok()
                .and_then(|s| crate::refusals::teach(&s.world, command, &reject));
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
    // the whole send path runs off the async runtime: the adapter ask
    // is a bounded subprocess call, serialized per recipient address
    let send_app = app.clone();
    let send_context = context.clone();
    let send_command = command.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        send_operation(&send_app, &send_context, send_command, received)
    })
    .await
    .expect("the send operation does not panic");
    match outcome {
        Ok((stored, submission)) => {
            let (_, message, from) = message_facts(&app, &stored);
            match submission {
                SubmitOutcome::Submitted | SubmitOutcome::Unrecorded => (
                    StatusCode::OK,
                    Json(json!({
                        "records": stored,
                        "message": message,
                        "from": from,
                        "submission": submission.json(),
                    })),
                )
                    .into_response(),
                SubmitOutcome::Refused { code, detail } => (
                    StatusCode::CONFLICT,
                    Json(json!({
                        "error": {"code": code, "detail": detail},
                        "message": message,
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => execute_failure_response(&app, e, &command),
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
