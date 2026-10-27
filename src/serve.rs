//! The serve edge: one sole-writer process behind the workbench and the
//! /api/v1 surface alike. The workbench page is one embedded asset set;
//! /t/N and /c/N deep-link into it, the artifact bytes door and the API
//! routes are unchanged. Form posts are gone with the console: the
//! workbench writes through its own /api/v1/workbench doors.

use axum::extract::{OriginalUri, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tracing::info;

use crate::api::AppState;
use crate::web;
use crate::workbench::assets;

pub async fn run(
    db_path: &std::path::Path,
    repo_root: Option<std::path::PathBuf>,
    bind: &str,
    port: u16,
) -> Result<std::convert::Infallible, String> {
    let addr = format!("{bind}:{port}");
    let mut state = AppState::open(db_path)?;
    if let Some(root) = &repo_root {
        state = state
            .with_artifacts(crate::paths::artifacts_at(root))
            .with_repo_root(root.clone());
    }
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("cannot bind {addr}: {e}"))?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    let url = published_url(&bound);
    state = state.with_server_url(url.clone());
    publish_address(db_path, &url)?;
    if !matches!(bind, "127.0.0.1" | "localhost" | "::1" | "[::1]") {
        eprintln!(
            "bound {bind}: any device that can reach port {port} can write events (human tier)"
        );
    }
    info!(%addr, "serving http");
    let router = router(state);
    axum::serve(listener, router)
        .await
        .expect("axum serves until killed");
    unreachable!("axum::serve returns only on shutdown")
}

/// Published only after a successful bind; a wildcard bind publishes
/// the local loopback, never a connect target of all zeros.
fn publish_address(db_path: &std::path::Path, url: &str) -> Result<(), String> {
    std::fs::write(crate::paths::address_at(db_path), format!("{url}\n"))
        .map_err(|e| format!("cannot write the address hint: {e}"))
}

/// The connect URL this server publishes: the loopback for a wildcard
/// bind, the bound address otherwise.
fn published_url(bound: &std::net::SocketAddr) -> String {
    let ip = if bound.ip().is_unspecified() {
        std::net::IpAddr::from(std::net::Ipv4Addr::LOCALHOST)
    } else {
        bound.ip()
    };
    let host = match ip {
        std::net::IpAddr::V4(v4) => v4.to_string(),
        v6 => format!("[{v6}]"),
    };
    format!("http://{host}:{}", bound.port())
}

/// The whole HTTP surface: the versioned API plus the workbench page.
pub fn router(state: AppState) -> axum::Router {
    crate::api::routes()
        .fallback(get(web_get))
        .with_state(state)
}

async fn web_get(State(app): State<AppState>, OriginalUri(uri): OriginalUri) -> Response {
    let url = uri.to_string();
    respond_get(&url, &app)
}

fn respond_get(url: &str, app: &AppState) -> Response {
    match parse_route(url) {
        // the workbench owns /, and /t/N with /c/N deep-link into it
        Route::Workbench => asset(200, "text/html; charset=utf-8", assets::INDEX),
        Route::Asset(name) => match name.as_str() {
            "app.js" => asset(200, "text/javascript; charset=utf-8", assets::APP),
            "style.css" => asset(200, "text/css; charset=utf-8", assets::STYLE),
            _ => page(app, 404, "nothing here - try /"),
        },
        Route::Artifact(hash) => artifact_bytes(app, &hash),
        Route::NotFound => page(app, 404, "nothing here - try /"),
    }
}

fn asset(status: u16, content_type: &str, body: &'static str) -> Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::OK),
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
}

/// The artifact bytes door: store bytes by sniffed content type, an
/// honest miss for anything the store does not hold. The hash is a
/// validated token before it ever names a path.
fn artifact_bytes(app: &AppState, hash: &str) -> Response {
    let valid = hash.len() == 64 && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if !valid {
        return page(app, 404, "nothing here - try /");
    }
    let Some(dir) = app.artifacts_dir() else {
        return page(app, 404, "artifact unavailable");
    };
    let path = dir.join(hash);
    let Ok(bytes) = std::fs::read(&path) else {
        return page(app, 404, "artifact unavailable");
    };
    let content_type = web::sniff_image(&bytes).unwrap_or("application/octet-stream");
    let mut response = (StatusCode::OK, bytes).into_response();
    if let Ok(value) = header::HeaderValue::from_str(content_type) {
        response.headers_mut().insert(header::CONTENT_TYPE, value);
    }
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
}

fn page(app: &AppState, status: u16, msg: &str) -> Response {
    let _ = app;
    let title = "saccade";
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        format!(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{title}</title></head>\n<body>\n<div class=\"err\">{} | <a href=\"/\">workbench</a></div>\n</body></html>\n",
            esc(msg),
        ),
    )
        .into_response()
}

#[derive(Debug, PartialEq)]
enum Route {
    Workbench,
    Asset(String),
    Artifact(String),
    NotFound,
}

fn parse_route(url: &str) -> Route {
    let path = url.split('?').next().unwrap_or(url);
    match path {
        "/" => Route::Workbench,
        "/app.js" | "/style.css" => Route::Asset(path.trim_start_matches('/').to_string()),
        _ => {
            if path
                .strip_prefix("/t/")
                .and_then(|rest| rest.parse::<usize>().ok())
                .is_some()
            {
                return Route::Workbench;
            }
            if path
                .strip_prefix("/c/")
                .and_then(|rest| rest.parse::<usize>().ok())
                .is_some()
            {
                return Route::Workbench;
            }
            if let Some(hash) = path.strip_prefix("/a/") {
                return Route::Artifact(hash.to_string());
            }
            Route::NotFound
        }
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_parse() {
        assert!(matches!(parse_route("/"), Route::Workbench));
        assert!(matches!(parse_route("/?x=1"), Route::Workbench));
        assert!(matches!(parse_route("/t/17"), Route::Workbench));
        assert!(matches!(parse_route("/t/17?x=1"), Route::Workbench));
        assert!(matches!(parse_route("/c/3"), Route::Workbench));
        assert!(matches!(parse_route("/nope"), Route::NotFound));
        assert!(matches!(parse_route("/t/x"), Route::NotFound));
        match parse_route("/app.js") {
            Route::Asset(name) => assert_eq!(name, "app.js"),
            other => panic!("the app bundle parses: {other:?}"),
        }
        match parse_route("/style.css") {
            Route::Asset(name) => assert_eq!(name, "style.css"),
            other => panic!("the style parses: {other:?}"),
        }
        match parse_route("/a/3941d4453740985f0c363433c74070c2c4aa649d8a73fe52a573831c0f87aa7e") {
            Route::Artifact(hash) => assert_eq!(hash.len(), 64),
            other => panic!("the bytes door parses: {other:?}"),
        }
    }
}
