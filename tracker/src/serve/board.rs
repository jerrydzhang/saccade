use std::time::Instant;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::Reject;
use crate::attempts::AsReceived;
use crate::db::{self, ExecuteFail, StoredRecord};
use serde_json::json;

use super::door::integration_outcome;
use super::fail::{ApiFail, api_failure, boards_status};
use super::state::{AppState, ReadFail, ServerState, boards_of, snapshot_of};
use super::wire::wire_context_of;

/// The handshake surface: which build is serving, nothing more
pub(super) async fn board_path(
    State(app): State<AppState>,
    axum::extract::Path(task): axum::extract::Path<usize>,
) -> Response {
    let snapshot = match app.snapshot() {
        Ok(snapshot) => snapshot,
        Err(fail) => {
            return api_failure(match fail {
                ReadFail::Degraded(degraded) => ApiFail::Degraded(degraded.reason),
                ReadFail::Db(reason) => ApiFail::Db(reason),
            });
        }
    };
    let Some(context) = snapshot.world.tasks.get(task) else {
        return api_failure(ApiFail::Reject(Reject::InvalidTaskId, None));
    };
    if context.commits.is_empty() {
        return api_failure(ApiFail::Refused(
            StatusCode::NOT_FOUND,
            "board_missing",
            format!("task t-{task} has no committed board"),
        ));
    }
    let Some(root) = app.repo_root() else {
        return api_failure(ApiFail::NoRepo);
    };
    let boards = match crate::boards::Boards::open(&crate::paths::state_dir(root)) {
        Ok(boards) => boards,
        Err(fail) => {
            return api_failure(ApiFail::Refused(
                StatusCode::INTERNAL_SERVER_ERROR,
                fail.code(),
                fail.detail(),
            ));
        }
    };
    let path = boards.materialized_path(&context.birth.0.to_string());
    (
        StatusCode::OK,
        Json(json!({"task": format!("t-{task}"), "path": path.display().to_string()})),
    )
        .into_response()
}
// -- the board doors -----------------------------------------------------
//
// Working directories are local to the caller's machine; every Git
// effect runs here, under the writer lock, in the D7 order. Reads
// open the repository per request: the repair at boot keeps its
// caches true, and an open re-reads the refs.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::serve::board_wire::{
    BoardFile, BoardFiles, CommitReply, CommitRequest, DiffQuery, DiffReply, EditQuery, EditReply,
    GetQuery, GetReply, MAX_BOARD_FILE_BYTES, SearchReply, TreeReply, UpdateReply, UpdateRequest,
};
use axum::extract::rejection::BytesRejection;
use base64::Engine as _;

/// The board doors that carry a whole board take one request's limit.
pub(super) fn board_body_limit() -> axum::extract::DefaultBodyLimit {
    axum::extract::DefaultBodyLimit::max(crate::serve::board_wire::BOARD_REQUEST_LIMIT)
}

/// A board door's body, or the refusal its rejection earns: past the
/// limit is the error envelope naming it, never the extractor's text.
fn board_body(body: Result<Bytes, BytesRejection>) -> Result<Bytes, ApiFail> {
    use crate::serve::board_wire::BOARD_REQUEST_LIMIT;
    body.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiFail::Refused(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_too_large",
                format!(
                    "this server accepts about {BOARD_REQUEST_LIMIT} bytes per request; a board's files ride inside it as base64, so about {} bytes of files fit",
                    BOARD_REQUEST_LIMIT / 4 * 3
                ),
            )
        } else {
            ApiFail::Malformed(rejection.body_text())
        }
    })
}

static BOARD_TEMP: AtomicUsize = AtomicUsize::new(0);

fn board_tempdir(label: &str) -> std::path::PathBuf {
    let n = BOARD_TEMP.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("saccade-board-{label}-{}-{n}", std::process::id()))
}
pub(super) fn board_ready(
    guard: &mut ServerState,
) -> Result<&mut crate::serve::api::Inner, ApiFail> {
    match guard {
        ServerState::Ready(inner) => Ok(inner),
        ServerState::Degraded(reason, _) => Err(ApiFail::Degraded(reason.clone())),
    }
}
pub(super) async fn board_commit(
    State(app): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match board_body(body) {
        Ok(body) => body,
        Err(fail) => return api_failure(fail),
    };
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: CommitRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.context_client(),
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = board_commit_operation(&app, &mut guard, &request);
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    if outcome.is_ok() {
        app.wake_landing();
    }
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

/// The publish sequence (D7): the objects land first, every selector
/// resolves, the caller's TaskCommitted records under the store's own
/// guards, and only then do the ref and the checkout move. The writer
/// lock is held throughout, so a concurrent commit refuses on the
/// parent guard instead of interleaving.
fn board_commit_operation(
    app: &AppState,
    guard: &mut ServerState,
    request: &CommitRequest,
) -> Result<CommitReply, ApiFail> {
    let inner = board_ready(guard)?;
    let world = match db::load(&inner.conn)
        .map_err(|e| ApiFail::Db(e.to_string()))?
        .state
    {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    let task_id = crate::TaskId(request.task);
    if world.tasks.get(request.task).is_none() {
        return Err(ApiFail::Reject(crate::Reject::InvalidTaskId, None));
    }
    let boards = boards_of(app)?;
    let workdir = board_tempdir(&format!("commit-{}", request.task));
    request
        .files
        .write_to(&workdir)
        .map_err(ApiFail::Malformed)?;
    let context = wire_context_of(&request.context);
    let conn = &mut inner.conn;
    let mut stored: Option<Vec<StoredRecord>> = None;
    let mut revision: Option<usize> = None;
    let ref_key = world
        .tasks
        .get(request.task)
        .map(|ctx| ctx.birth.0.to_string())
        .expect("the task resolved before the publish");
    let landed = boards.publish_board(
        task_id,
        &ref_key,
        &world,
        &workdir,
        &request.message,
        |command| match db::record(conn, &context, command, db::now_epoch()) {
            Ok((rows, world)) => {
                revision = world.tasks.get(request.task).map(|ctx| ctx.commits.len());
                stored = Some(rows);
                Ok(())
            }
            Err(ExecuteFail::Reject(reject)) => Err(format!(
                "{}: {}",
                reject.code(),
                serde_json::to_string(&reject).unwrap_or_default()
            )),
            Err(ExecuteFail::Degraded(reason)) => Err(reason),
            Err(ExecuteFail::Db(error)) => Err(error.to_string()),
        },
    );
    let _ = std::fs::remove_dir_all(&workdir);
    let commit = landed
        .map_err(|fail| ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail()))?;
    Ok(CommitReply {
        records: stored.unwrap_or_default(),
        commit,
        revision: revision.unwrap_or_default(),
    })
}

pub(super) async fn board_update(
    State(app): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match board_body(body) {
        Ok(body) => body,
        Err(fail) => return api_failure(fail),
    };
    let request: UpdateRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let outcome = board_update_operation(&app, &request);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

/// The three-way merge onto the current commit (D2): the caller's
/// working files ride up, the merged tree rides back, conflicts
/// marked in place. The recorded base moves to the new commit either
/// way, so a later commit names the right parent.
fn board_update_operation(app: &AppState, request: &UpdateRequest) -> Result<UpdateReply, ApiFail> {
    let snapshot = snapshot_of(app)?;
    let Some(ctx) = snapshot.world.tasks.get(request.task) else {
        return Err(ApiFail::Reject(crate::Reject::InvalidTaskId, None));
    };
    let Some(current) = ctx.commits.last() else {
        return Err(ApiFail::Refused(
            StatusCode::NOT_FOUND,
            "board_missing",
            format!("task t-{} has no committed board", request.task),
        ));
    };
    let current = current.commit.as_str().to_string();
    if request.base == current {
        return Ok(UpdateReply {
            files: request.files.clone(),
            board_conflict: false,
            figure_conflicts: Vec::new(),
            base: current,
            up_to_date: true,
        });
    }
    let boards = boards_of(app)?;
    let workdir = board_tempdir(&format!("update-{}", request.task));
    let outcome = (|| {
        request
            .files
            .write_to(&workdir)
            .map_err(ApiFail::Malformed)?;
        crate::boards::record_base(&workdir, &request.base)
            .map_err(|fail| ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail()))?;
        let merged = boards
            .update(&workdir, &current)
            .map_err(|fail| ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail()))?;
        let files = BoardFiles::from_workdir(&workdir).map_err(ApiFail::Malformed)?;
        Ok(UpdateReply {
            files,
            board_conflict: merged.board_conflict,
            figure_conflicts: merged.figure_conflicts,
            base: current.clone(),
            up_to_date: false,
        })
    })();
    let _ = std::fs::remove_dir_all(&workdir);
    outcome
}

pub(super) async fn board_edit(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<EditQuery>,
) -> Response {
    let outcome = (|| {
        let snapshot = snapshot_of(&app)?;
        let Some(ctx) = snapshot.world.tasks.get(query.task) else {
            return Err(ApiFail::Reject(crate::Reject::InvalidTaskId, None));
        };
        let boards = boards_of(&app)?;
        let current = ctx.commits.last().map(|c| c.commit.as_str().to_string());
        let chosen = current.clone();
        let files = match &chosen {
            Some(commit) => {
                let workdir = board_tempdir(&format!("edit-{}", query.task));
                let files = (|| {
                    boards.export_tree(commit, &workdir).map_err(|fail| {
                        ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail())
                    })?;
                    BoardFiles::from_workdir(&workdir).map_err(ApiFail::Malformed)
                })();
                let _ = std::fs::remove_dir_all(&workdir);
                files?
            }
            None => BoardFiles {
                files: vec![encode_board_file(
                    "board.yaml",
                    crate::serve::board_wire::fresh_board_text(query.task).as_bytes(),
                )?],
            },
        };
        Ok(EditReply {
            files,
            base: current,
            commit: chosen,
            task: format!("t-{}", query.task),
        })
    })();
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

fn encode_board_file(path: &str, bytes: &[u8]) -> Result<BoardFile, ApiFail> {
    if bytes.len() > MAX_BOARD_FILE_BYTES {
        return Err(ApiFail::Malformed(format!(
            "{path} exceeds the per-file bound"
        )));
    }
    Ok(BoardFile {
        path: path.to_string(),
        content: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

pub(super) async fn board_get(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<GetQuery>,
) -> Response {
    let outcome = (|| {
        let snapshot = snapshot_of(&app)?;
        let Some(ctx) = snapshot.world.tasks.get(query.task) else {
            return Err(ApiFail::Reject(crate::Reject::InvalidTaskId, None));
        };
        let revision = query.revision.unwrap_or(ctx.commits.len());
        let at = revision.checked_sub(1).ok_or_else(|| {
            ApiFail::Refused(
                StatusCode::NOT_FOUND,
                "revision_missing",
                format!("revision r{revision} does not exist"),
            )
        })?;
        let Some(board_commit) = ctx.commits.get(at) else {
            return Err(ApiFail::Refused(
                StatusCode::NOT_FOUND,
                "revision_missing",
                format!("t-{} has no revision r{revision}", query.task),
            ));
        };
        let commit = board_commit.commit.as_str();
        let boards = boards_of(&app)?;
        if let Some(select) = &query.select {
            // the spec names a component of the revision's board; the
            // reply carries it in the grammar's written form
            let board = boards.read_board_at(commit).map_err(|fail| {
                ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail())
            })?;
            let path = board_grammar::SelectPath::parse(select).map_err(|error| {
                ApiFail::Refused(
                    StatusCode::BAD_REQUEST,
                    "bad_select_path",
                    error.to_string(),
                )
            })?;
            let component = board_grammar::select(&board, &path).map_err(|error| {
                ApiFail::Refused(
                    StatusCode::BAD_REQUEST,
                    "bad_select_path",
                    error.to_string(),
                )
            })?;
            return Ok(serde_json::to_value(&GetReply {
                name: select.to_string(),
                content: base64::engine::general_purpose::STANDARD
                    .encode(board_grammar::render(component)),
            })
            .expect("the reply is plain data"));
        }
        if let Some(path) = &query.path {
            // the spec names the file with the component-path leading
            // slash; the tree stores bare relative paths
            let path = path.trim_start_matches('/');
            let bytes = boards.file_bytes(commit, path).map_err(|fail| {
                ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail())
            })?;
            let Some(bytes) = bytes else {
                return Err(ApiFail::Refused(
                    StatusCode::NOT_FOUND,
                    "board_file_missing",
                    format!("{path} is not in t-{}@r{revision}", query.task),
                ));
            };
            return Ok(serde_json::to_value(&GetReply {
                name: path.to_string(),
                content: base64::engine::general_purpose::STANDARD.encode(bytes),
            })
            .expect("the reply is plain data"));
        }
        let workdir = board_tempdir(&format!("get-{}", query.task));
        let files = (|| {
            boards.export_tree(commit, &workdir).map_err(|fail| {
                ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail())
            })?;
            BoardFiles::from_workdir(&workdir).map_err(ApiFail::Malformed)
        })();
        let _ = std::fs::remove_dir_all(&workdir);
        let files = files?;
        Ok(serde_json::to_value(&TreeReply {
            files,
            commit: commit.to_string(),
            revision,
        })
        .expect("the reply is plain data"))
    })();
    match outcome {
        Ok(reply) => (StatusCode::OK, Json(reply)).into_response(),
        Err(fail) => api_failure(fail),
    }
}

pub(super) async fn board_diff(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<DiffQuery>,
) -> Response {
    let outcome = (|| {
        let snapshot = snapshot_of(&app)?;
        let Some(ctx) = snapshot.world.tasks.get(query.task) else {
            return Err(ApiFail::Reject(crate::Reject::InvalidTaskId, None));
        };
        let commit_at = |revision: usize| {
            ctx.commits
                .get(revision.checked_sub(1)?)
                .map(|c| c.commit.as_str().to_string())
        };
        let from = commit_at(query.from).ok_or_else(|| {
            ApiFail::Refused(
                StatusCode::NOT_FOUND,
                "revision_missing",
                format!("t-{} has no revision r{}", query.task, query.from),
            )
        })?;
        let to = commit_at(query.to).ok_or_else(|| {
            ApiFail::Refused(
                StatusCode::NOT_FOUND,
                "revision_missing",
                format!("t-{} has no revision r{}", query.task, query.to),
            )
        })?;
        let boards = boards_of(&app)?;
        let diff = boards
            .diff(&from, &to)
            .map_err(|fail| ApiFail::Refused(boards_status(&fail), fail.code(), fail.detail()))?;
        Ok(serde_json::to_value(&DiffReply {
            diff,
            from: query.from,
            to: query.to,
        })
        .expect("the reply is plain data"))
    })();
    match outcome {
        Ok(reply) => (StatusCode::OK, Json(reply)).into_response(),
        Err(fail) => api_failure(fail),
    }
}

pub(super) async fn board_log(
    State(app): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<EditQuery>,
) -> Response {
    let outcome = (|| {
        let snapshot = snapshot_of(&app)?;
        if snapshot.world.tasks.get(query.task).is_none() {
            return Err(ApiFail::Reject(crate::Reject::InvalidTaskId, None));
        }
        let boards = boards_of(&app)?;
        let history = crate::serve::board_wire::board_history(
            &snapshot.world,
            &snapshot.rows,
            &boards,
            query.task,
        )
        .map_err(ApiFail::Db)?;
        Ok(serde_json::json!({ "history": history }))
    })();
    match outcome {
        Ok(reply) => (StatusCode::OK, Json(reply)).into_response(),
        Err(fail) => api_failure(fail),
    }
}

pub(super) async fn board_search(State(app): State<AppState>, body: Bytes) -> Response {
    let query: crate::serve::board_wire::SearchQuery = match serde_json::from_slice(&body) {
        Ok(q) => q,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let outcome = (|| {
        let snapshot = snapshot_of(&app)?;
        let boards = boards_of(&app)?;
        let lines = crate::serve::board_wire::board_search(
            &snapshot.world,
            &snapshot.rows,
            &boards,
            &query,
        )
        .map_err(ApiFail::Malformed)?;
        Ok(serde_json::to_value(&SearchReply { lines }).expect("the reply is plain data"))
    })();
    match outcome {
        Ok(reply) => (StatusCode::OK, Json(reply)).into_response(),
        Err(fail) => api_failure(fail),
    }
}
