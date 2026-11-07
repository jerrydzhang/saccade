//! The serve edge: one sole-writer process behind the /api/v1 surface.

use tracing::info;

use crate::api::AppState;

pub async fn run(
    db_path: &std::path::Path,
    repo_root: Option<std::path::PathBuf>,
    bind: &str,
    port: u16,
) -> Result<std::convert::Infallible, String> {
    let addr = format!("{bind}:{port}");
    let mut state = AppState::open(db_path)?;
    if let Some(root) = &repo_root {
        state = state.with_repo_root(root.clone());
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

/// The whole HTTP surface: the versioned API.
pub fn router(state: AppState) -> axum::Router {
    crate::api::routes().with_state(state)
}
