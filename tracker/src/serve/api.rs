//! The versioned command surface. System authorship is unrepresentable
//! here: the wire tier admits only human and agent, so no request can
//! present the machinery's author.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde_json::json;

use super::board::{
    board_body_limit, board_commit, board_diff, board_edit, board_get, board_log, board_path,
    board_search, board_update,
};
use super::command::command;
use super::incarnation::{incarnation_address, incarnation_register, native_attach};
use super::message::{message_send, messages_history};
use super::workspace::{workspace_clean, workspace_create};
pub use crate::serve::hosts::Hosts;
pub(crate) use crate::serve::hosts::origin_policy;
pub(super) use crate::serve::state::Inner;
pub use crate::serve::state::{AppState, LandingWake, ReadFail, Snapshot};
pub use crate::serve::wire::{Envelope, WireContext, WireTier};

/// The shells the viewer ships in; their origins are fixed by the
/// webview runtimes.
pub const TAURI_ORIGINS: [&str; 3] = [
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
];

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/command", post(command))
        .route("/api/v1/records", get(records))
        .route("/api/v1/workspace/create", post(workspace_create))
        .route("/api/v1/workspace/clean", post(workspace_clean))
        .route("/api/v1/native/attach", post(native_attach))
        .route("/api/v1/incarnations/register", post(incarnation_register))
        .route("/api/v1/incarnations/address", post(incarnation_address))
        .route("/api/v1/messages/send", post(message_send))
        .route("/api/v1/messages", get(messages_history))
        .route("/api/v1/boards/path/{task}", get(board_path))
        .route("/api/v1/boards/edit", get(board_edit))
        .route("/api/v1/boards/get", get(board_get))
        .route("/api/v1/boards/diff", get(board_diff))
        .route("/api/v1/boards/log", get(board_log))
        .route("/api/v1/boards/search", post(board_search))
        .route(
            "/api/v1/boards/commit",
            post(board_commit).layer(board_body_limit()),
        )
        .route(
            "/api/v1/boards/update",
            post(board_update).layer(board_body_limit()),
        )
        .route("/api/v1/version", get(version))
        // the pages: the instruments' read surface, one door per route
        // tail, every answer a view document or raw file bytes
        .route("/view/v1", get(crate::serve::pages::view_root))
        .route("/view/v1/", get(crate::serve::pages::view_root))
        .route("/view/v1/{*location}", get(crate::serve::pages::view_page))
        // the window store: the suite's window-state door, outside the record
        // (t-14 ratified, t-24@r20); same edge, same origin policy, the
        // same conceded write boundary as /api/v1
        .route("/api/v1/windows", get(crate::serve::windows::windows_root))
        .route("/api/v1/windows/", get(crate::serve::windows::windows_root))
        .route(
            "/api/v1/windows/states",
            get(crate::serve::windows::window_states)
                .post(crate::serve::windows::window_state_create),
        )
        .route(
            "/api/v1/windows/states/{id}",
            get(crate::serve::windows::window_state_get)
                .patch(crate::serve::windows::window_state_patch)
                .delete(crate::serve::windows::window_state_discard),
        )
        .route(
            "/api/v1/windows/devices/{device}/attach",
            post(crate::serve::windows::windows_attach),
        )
        .route(
            "/api/v1/windows/devices/{device}/detach",
            post(crate::serve::windows::windows_detach),
        )
        .route(
            "/api/v1/windows/devices/{device}/links/{from}",
            put(crate::serve::windows::window_link_set)
                .delete(crate::serve::windows::window_link_clear),
        )
}

async fn version(State(app): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        // the tracker's name for tracker-named locations: the serving
        // repo's directory basename; a --db server without a repo
        // names nothing
        "tracker": app.repo_root().map(crate::paths::repo_basename),
        "board_grammar": {
            "min": *board_grammar::SUPPORTED_GRAMMAR_VERSIONS.start(),
            "max": *board_grammar::SUPPORTED_GRAMMAR_VERSIONS.end(),
        },
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
