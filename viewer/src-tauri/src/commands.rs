//! Read-only declaration and asset reads. The guard mirrors the frontend
//! resolve rules exactly: relative segments only, no parent traversal, no
//! backslash separators, no absolute paths, no NUL bytes. Canonicalized
//! prefix checking runs second so a guard bypass still cannot escape.
//!
//! Local locations read through std::fs exactly as before. A location
//! that names a machine reads through the LOCAL herdr-core's read-only
//! face instead (file.read for text; fixed-argv exec for bytes and
//! remote path resolution), with containment proven on the remote side
//! and every machine state surfacing as a structured error. The core
//! socket is the only door here, and it opens read-only calls; the
//! renderer gets no socket, no shell and no machine management.

use board_viewer_core::{CoreClient, CoreFail, SocketTransport, guard_relative};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

static INITIAL_BOARD: OnceLock<Option<String>> = OnceLock::new();

pub fn set_initial_board(path: Option<String>) {
    let _ = INITIAL_BOARD.set(path);
}

/// The explicit board location handed to the foreground launch, if any:
/// `/path/to/board.yaml` or `machine:/path/to/board.yaml`. Never a
/// registry, never a fallback - without it the app starts empty.
#[tauri::command]
pub fn initial_board() -> Option<String> {
    eprintln!(
        "[bridge] initial_board invoked: {:?}",
        INITIAL_BOARD.get().cloned().flatten()
    );
    INITIAL_BOARD.get().cloned().flatten()
}

#[derive(Serialize, Deserialize)]
pub struct DeclaredText {
    pub text: String,
}

#[derive(Serialize, Deserialize)]
pub struct BridgeError {
    pub code: String,
    pub message: String,
}

fn deny(reason: &str) -> BridgeError {
    BridgeError {
        code: "asset-denied".into(),
        message: reason.into(),
    }
}

fn missing(path: &str) -> BridgeError {
    BridgeError {
        code: "asset-missing".into(),
        message: format!("not found or unreadable: {path}"),
    }
}

fn core_failure(fail: CoreFail) -> BridgeError {
    BridgeError {
        code: fail.code().into(),
        message: fail.to_string(),
    }
}

fn core_client() -> Result<CoreClient<SocketTransport>, BridgeError> {
    let path = SocketTransport::default_path().map_err(core_failure)?;
    Ok(CoreClient::new(SocketTransport::new(path)))
}

/// The machine argument of a command, present and non-empty.
fn machine_of(on: &Option<String>) -> Option<&str> {
    on.as_deref().filter(|machine| !machine.is_empty())
}

/// The frontend already applied these rules; repeat them here so the
/// host boundary holds even against a tampered renderer.
fn canonical_prefix_holds(base: &std::path::Path, joined: &std::path::Path) -> bool {
    match (base.canonicalize(), joined.canonicalize()) {
        (Ok(base), Ok(joined)) => joined.starts_with(base),
        _ => false,
    }
}

fn log_bridge(operation: &str, path: &str, outcome: &str) -> std::io::Result<()> {
    // A useful outcome line on the app's own stderr: the human running
    // the foreground app sees what the bridge actually read.
    eprintln!("[bridge] {operation} {path}: {outcome}");
    Ok(())
}

fn read_text_resolved(base: &str, relative: Option<&str>) -> Result<DeclaredText, BridgeError> {
    let base_path = std::path::Path::new(base);
    let target = match relative {
        None => base_path.to_path_buf(),
        Some(rel) => {
            guard_relative(rel).map_err(deny)?;
            base_path.join(rel)
        }
    };
    if !canonical_prefix_holds(base_path, &target) {
        return Err(deny("resolved path left the declaration directory"));
    }
    let bytes = std::fs::read(&target).map_err(|_| {
        let _ = log_bridge("read_declaration", &target.to_string_lossy(), "missing");
        missing(&target.to_string_lossy())
    })?;
    let _ = log_bridge(
        "read_declaration",
        &target.to_string_lossy(),
        &format!("{} bytes", bytes.len()),
    );
    String::from_utf8(bytes)
        .map(|text| DeclaredText { text })
        .map_err(|_| BridgeError {
            code: "asset-denied".into(),
            message: "declaration bytes are not UTF-8 text".into(),
        })
}

fn read_declaration_remote(machine: &str, path: &str) -> Result<DeclaredText, BridgeError> {
    if !path.starts_with('/') {
        return Err(deny("remote declaration path must be absolute"));
    }
    if path.contains('\0') {
        return Err(deny("NUL byte in path"));
    }
    let client = core_client()?;
    let text = client.read_text(machine, path).map_err(core_failure)?;
    let _ = log_bridge(
        "read_declaration",
        &format!("{machine}:{path}"),
        &format!("{} bytes", text.len()),
    );
    Ok(DeclaredText { text })
}

#[tauri::command]
pub fn read_declaration(path: String, on: Option<String>) -> Result<DeclaredText, BridgeError> {
    if path.is_empty() {
        return Err(deny("empty path"));
    }
    match machine_of(&on) {
        Some(machine) => read_declaration_remote(machine, &path),
        None => read_text_resolved(&path, None),
    }
}

fn read_asset_remote(machine: &str, base: &str, relative: &str) -> Result<Vec<u8>, BridgeError> {
    if !base.starts_with('/') {
        return Err(deny("remote base path must be absolute"));
    }
    if base.contains('\0') {
        return Err(deny("NUL byte in path"));
    }
    let client = core_client()?;
    let bytes = client
        .asset_bytes(machine, base, relative)
        .map_err(core_failure)?;
    let _ = log_bridge(
        "read_asset",
        &format!("{machine}:{base}/{relative}"),
        &format!("{} bytes", bytes.len()),
    );
    Ok(bytes)
}

#[tauri::command]
pub fn read_asset(
    base: String,
    relative: String,
    on: Option<String>,
) -> Result<Vec<u8>, BridgeError> {
    guard_relative(&relative).map_err(deny)?;
    match machine_of(&on) {
        Some(machine) => read_asset_remote(machine, &base, &relative),
        None => {
            let base_path = std::path::Path::new(&base);
            let target = base_path.join(&relative);
            if !canonical_prefix_holds(base_path, &target) {
                return Err(deny("resolved path left the declaration directory"));
            }
            let bytes = std::fs::read(&target).map_err(|_| {
                let _ = log_bridge("read_asset", &target.to_string_lossy(), "missing");
                missing(&target.to_string_lossy())
            })?;
            let _ = log_bridge(
                "read_asset",
                &target.to_string_lossy(),
                &format!("{} bytes", bytes.len()),
            );
            Ok(bytes)
        }
    }
}

/// A static identity statement for the machine-readable surface: what this
/// binary is, never a claim about any session or tracker state.
#[tauri::command]
pub fn viewer_identity() -> serde_json::Value {
    serde_json::json!({
        "component": "saccade-board-viewer",
        "version": env!("CARGO_PKG_VERSION"),
        "read_only": true,
    })
}

#[cfg(test)]
mod tests {
    use board_viewer_core::guard_relative;

    #[test]
    fn parent_traversal_is_denied() {
        for attempt in ["../secret", "assets/../../x", "a/../.."] {
            assert!(guard_relative(attempt).is_err(), "{attempt}");
        }
    }

    #[test]
    fn absolute_and_windows_paths_are_denied() {
        for attempt in ["/etc/passwd", "C:/x", "~/.ssh/id_ed25519", "a\\b"] {
            assert!(guard_relative(attempt).is_err(), "{attempt}");
        }
    }

    #[test]
    fn ordinary_relative_paths_pass() {
        assert!(guard_relative("assets/figure.svg").is_ok());
        assert!(guard_relative("./nested/./fig.png").is_ok());
    }

    #[test]
    fn nul_bytes_are_denied() {
        assert!(guard_relative("a\0b").is_err());
    }
}

/// The system clipboard write, through the app's own GTK display
/// connection. The webview's copy event never dispatches under the
/// native WebKitGTK (execCommand copy returns false even for plain
/// prose selections), so the frontend routes its copy payload here on
/// Ctrl+C and suppresses the native command.
#[tauri::command]
pub fn write_clipboard(window: tauri::Window, text: String) -> Result<usize, String> {
    use gtk::traits::WidgetExt;
    let display = window
        .gtk_window()
        .map_err(|_| "no gtk window".to_string())?
        .display();
    let selection = gtk::gdk::Atom::from("CLIPBOARD");
    let clipboard = gtk::Clipboard::for_display(&display, &selection);
    clipboard.set_text(&text);
    eprintln!("[bridge] clipboard set: {} chars", text.chars().count());
    Ok(text.chars().count())
}
