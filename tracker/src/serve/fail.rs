use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::db::ExecuteFail;
use crate::host::HostFail;
use crate::workspace::WorkspaceFail;
use crate::{Command, Reject};

use super::state::AppState;

pub(super) enum ApiFail {
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
    pub(super) fn parts(self) -> (StatusCode, String, Value) {
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
/// The refusal face a refused write shares: the rejection teaches
/// where the world knows the intended target.
pub(super) fn execute_failure_response(
    app: &AppState,
    e: ExecuteFail,
    command: &Command,
) -> Response {
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
pub(super) fn api_failure(fail: ApiFail) -> Response {
    let (status, code, detail) = fail.parts();
    (
        status,
        Json(json!({"error": {"code": code, "detail": detail}})),
    )
        .into_response()
}
/// The wire status a board effect refusal earns: validation is the
/// caller's input, state conflicts are concurrent or sequence errors,
/// and the infrastructure's failures are the server's own.
pub(super) fn boards_status(fail: &crate::boards::BoardsFail) -> StatusCode {
    use crate::boards::BoardsFail;
    match fail {
        BoardsFail::BoardInvalid(_)
        | BoardsFail::MissingFigure { .. }
        | BoardsFail::FigureTooLarge { .. }
        | BoardsFail::MarkdownImage { .. }
        | BoardsFail::MissingBase { .. }
        | BoardsFail::BadBase { .. }
        | BoardsFail::InvalidKey { .. }
        | BoardsFail::UnknownSourceTask { .. }
        | BoardsFail::UnknownSourceRevision { .. }
        | BoardsFail::TaskMismatch { .. }
        | BoardsFail::WorkingBoard { .. } => StatusCode::BAD_REQUEST,
        BoardsFail::MissingCommit { .. } | BoardsFail::Unchanged { .. } | BoardsFail::Record(_) => {
            StatusCode::CONFLICT
        }
        BoardsFail::MissingCommits { .. } | BoardsFail::Git(_) | BoardsFail::Io { .. } => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}
