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
use crate::events::Event;
use crate::objects::task::TaskId;
use crate::store::{Context, RecordId, Scope, Tier, World};
use crate::views::WorkspaceFacts;
use crate::web;
use crate::workspace::{self, Established, WorkspaceFail};
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

/// The sole writer's state: every face loads the log fresh under the
/// one lock, so no read can serve a world the file contradicts.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<ServerState>>,
    db_path: std::path::PathBuf,
    attempts: Attempts,
    /// The artifact store the bytes door serves from; a server that
    /// was not told a repo root has none.
    artifacts: Option<std::path::PathBuf>,
    /// The serving repo's root, when boot resolved one.
    repo_root: Option<std::path::PathBuf>,
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

/// A read that cannot present the world: the log will not fold (its
/// raw rows still travel), or will not even load — corruption and
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
        // path is absolute from the start — a fresh tracker resolves
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
            artifacts: None,
            repo_root: None,
            repo_name: None,
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

    /// The serving repo's root, when boot resolved one.
    pub fn repo_root(&self) -> Option<&Path> {
        self.repo_root.as_deref()
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
                let status = match fail {
                    WorkspaceFail::Git(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
            if matches!(command, Command::SendMessage { .. }) {
                let numbers = app
                    .snapshot()
                    .expect("a landed write leaves a foldable world")
                    .world;
                let sent = stored.first().expect("a message lands its one record").seq;
                let mctx = numbers
                    .messages
                    .get(&crate::MessageId(RecordId(sent)))
                    .expect("the landed message folds in");
                reply["message"] = json!({
                    "seq": sent,
                    "num": numbers.message_num(crate::MessageId(RecordId(sent))),
                });
                reply["from"] = json!({
                    "seq": mctx.message.from.0 .0,
                    "num": numbers.incarnation_num(mctx.message.from),
                });
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
        .route("/api/v1/workspace/start", post(workspace_start))
        .route("/api/v1/workspace/open", post(workspace_open))
        .route("/api/v1/native/attach", post(native_attach))
        .route("/api/v1/messages/send", post(message_send))
        .route("/api/v1/messages/submit", post(message_submit))
        .route("/api/v1/messages/pending", get(messages_pending))
        .route("/api/v1/version", get(version))
}

impl WorkspaceFacts {
    fn of(established: &Established) -> Self {
        WorkspaceFacts {
            task: established.task.0,
            base: established.base.as_str().to_string(),
            branch: established.branch.as_str().to_string(),
            path: established.worktree.as_path().display().to_string(),
        }
    }
}

/// One workspace operation's outcome.
#[derive(Serialize, Deserialize)]
pub struct WorkspaceReply {
    /// Whether this call birthed the establishment
    pub established: bool,
    /// Whether a missing checkout was recreated from its branch tip
    pub recreated: bool,
    pub workspace: WorkspaceFacts,
    /// The establishment record's position: the shared
    /// workspace/incarnation identity
    pub incarnation: usize,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct StartRequest {
    context: WireContext,
    task: usize,
    #[serde(default)]
    base: Option<String>,
    #[serde(default)]
    client: Option<String>,
}

#[derive(Deserialize)]
struct OpenRequest {
    task: usize,
    #[serde(default)]
    client: Option<String>,
}

enum OperationOutcome {
    Done(WorkspaceReply),
    Failed(ApiFail),
}

/// One workspace operation under the held writer lock: establish on
/// first sight or ensure the recorded environment; a standing
/// association lands nothing.
fn workspace_operation(
    app: &AppState,
    guard: &mut ServerState,
    actor: Option<String>,
    received: &AsReceived,
    task: usize,
    base: Option<&str>,
    caller_records: impl FnOnce(RecordId) -> Vec<db::RecordDraft>,
) -> OperationOutcome {
    let started = std::time::Instant::now();
    let outcome = run_operation(app, guard, task, base, caller_records);
    let duration_ms = started.elapsed().as_millis() as u64;
    match &outcome {
        OperationOutcome::Done(reply) => {
            let seqs: Vec<usize> = reply.records.iter().map(|s| s.seq).collect();
            app.attempts.append(
                actor.as_deref(),
                received.client.as_deref(),
                duration_ms,
                Outcome::Landed { seqs: &seqs },
            );
        }
        OperationOutcome::Failed(fail) => {
            let code = match fail {
                ApiFail::Reject(reject, _) => reject.code(),
                ApiFail::Degraded(_) => "degraded",
                ApiFail::Db(_) => "storage",
                ApiFail::Malformed(_) => "malformed_request",
                ApiFail::Workspace(effect) => effect.code(),
                ApiFail::NoRepo => "workspace_no_repo",
            };
            let cursor = match guard {
                ServerState::Ready(inner) => db::last_seq(&inner.conn)
                    .map_err(|e| e.to_string())
                    .ok()
                    .flatten(),
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
    outcome
}

fn run_operation(
    app: &AppState,
    guard: &mut ServerState,
    task: usize,
    base: Option<&str>,
    caller_records: impl FnOnce(RecordId) -> Vec<db::RecordDraft>,
) -> OperationOutcome {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return OperationOutcome::Failed(ApiFail::Degraded(reason.clone()));
    };
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return OperationOutcome::Failed(ApiFail::NoRepo);
    };
    // the world as the file stands: a supported local write must be
    // visible to the next operation
    let loadout = match db::load(&inner.conn) {
        Ok(loadout) => loadout,
        Err(e) => return OperationOutcome::Failed(ApiFail::Db(e.to_string())),
    };
    let world = match loadout.state {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => {
            return OperationOutcome::Failed(ApiFail::Degraded(reason));
        }
    };
    if world.tasks.get(task).is_none() {
        return OperationOutcome::Failed(ApiFail::Reject(Reject::InvalidTaskId, None));
    }
    let task_id = TaskId(task);
    match world.tasks[task].workspace.as_ref() {
        Some(recorded_ctx) => {
            // the association governs: recreate the checkout, never another birth
            let facts = workspace::recorded(task_id, recorded_ctx);
            match workspace::ensure_checkout(&repo_root, &facts) {
                Ok(recreated) => OperationOutcome::Done(WorkspaceReply {
                    established: false,
                    recreated,
                    workspace: WorkspaceFacts::of(&facts),
                    incarnation: recorded_ctx.birth.0,
                    records: Vec::new(),
                }),
                Err(fail) => OperationOutcome::Failed(ApiFail::Workspace(fail)),
            }
        }
        None => {
            let home = workspace::worktree_home(&inner.db_path);
            let established = match workspace::establish(&repo_root, &home, task_id, base) {
                Ok(facts) => facts,
                Err(fail) => return OperationOutcome::Failed(ApiFail::Workspace(fail)),
            };
            let birth = RecordId(loadout.rows.len());
            let mut drafts = vec![db::RecordDraft {
                context: Context::system(),
                event: Event::WorkspaceEstablished {
                    task_id,
                    base: established.base.clone(),
                    branch: established.branch.clone(),
                    worktree: established.worktree.clone(),
                },
            }];
            drafts.extend(caller_records(birth));
            match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
                Ok(records) => OperationOutcome::Done(WorkspaceReply {
                    established: true,
                    recreated: false,
                    workspace: WorkspaceFacts::of(&established),
                    incarnation: birth.0,
                    records,
                }),
                Err(ExecuteFail::Reject(reject)) => {
                    OperationOutcome::Failed(ApiFail::Reject(reject, None))
                }
                Err(ExecuteFail::Degraded(reason)) => {
                    OperationOutcome::Failed(ApiFail::Degraded(reason))
                }
                Err(ExecuteFail::Db(e)) => OperationOutcome::Failed(ApiFail::Db(e.to_string())),
            }
        }
    }
}

async fn workspace_start(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: StartRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let caller = Context {
        actor: request.context.actor,
        tier: match request.context.tier {
            WireTier::Human => Tier::Human,
            WireTier::Agent => Tier::Agent,
        },
        scope: request.context.scope,
    };
    let task = request.task;
    let base = request.base;
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = workspace_operation(
        &app,
        &mut guard,
        Some(caller.actor.as_str().to_string()),
        &received,
        task,
        base.as_deref(),
        |_| Vec::new(),
    );
    respond_operation(guard, outcome)
}

async fn workspace_open(State(app): State<AppState>, body: Bytes) -> Response {
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: OpenRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = workspace_operation(
        &app,
        &mut guard,
        None,
        &received,
        request.task,
        None,
        |_| Vec::new(),
    );
    respond_operation(guard, outcome)
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
    #[serde(default)]
    client: Option<String>,
}

fn attach_operation(
    guard: &mut ServerState,
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
    let started = std::time::Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = attach_operation(&mut guard, &request.native_id, &request.session_file);
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
    /// The recipient's public number, the human-facing i-N — the same
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
    // from the declared scope — never caller-authored prose
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
    // the public numbers resolve against the fold before the write
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
            let reply = json!({
                "records": stored,
                "message": {
                    "seq": sent,
                    "num": numbers.message_num(crate::MessageId(RecordId(sent))),
                },
                "from": {
                    "seq": mctx.message.from.0 .0,
                    "num": numbers.incarnation_num(mctx.message.from),
                },
            });
            (StatusCode::OK, Json(reply)).into_response()
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

/// The reader's own mail, in birth order, any state: a read the
/// integration polls, never a subscription the tracker keeps.
async fn messages_pending(
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
    let world = snapshot.world;
    let id = crate::IncarnationId(RecordId(query.incarnation));
    if !world.incarnations.contains_key(&id) {
        return api_failure(ApiFail::Reject(Reject::InvalidIncarnationId, None));
    }
    let messages: Vec<serde_json::Value> = world
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
    let cursor = snapshot.rows.last().map(|r| r.seq);
    (
        StatusCode::OK,
        Json(json!({"messages": messages, "cursor": cursor})),
    )
        .into_response()
}

fn respond_operation(
    guard: std::sync::MutexGuard<'_, ServerState>,
    outcome: OperationOutcome,
) -> Response {
    drop(guard);
    match outcome {
        OperationOutcome::Done(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        OperationOutcome::Failed(fail) => api_failure(fail),
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
