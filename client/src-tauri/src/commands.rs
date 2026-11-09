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

use board_viewer_core::{guard_relative, CoreClient, CoreFail, SocketTransport};
use serde::{Deserialize, Serialize};

use crate::look_config;

/// The lanes' test-open channel (BOARD_VIEWER_TEST_OPEN): the one
/// location the app opens at startup, riding the same model.open the
/// inspection surface uses. Test-only machinery, named as such
/// (t-24@r11); a human run has the variable unset and starts empty -
/// the launch argument and the local-device path form left the human
/// surface with it.
#[tauri::command]
pub fn test_open() -> Option<String> {
    let spec = std::env::var("BOARD_VIEWER_TEST_OPEN")
        .ok()
        .filter(|s| !s.is_empty());
    eprintln!("[bridge] test_open invoked: {:?}", spec);
    spec
}

/// The declared tracker machine (t-24@r12): the Herdr machine that
/// hosts this tracker, read on demand by a bare-path open. None on the
/// hub, where the local core's own machine is read.
#[tauri::command]
pub fn tracker_machine() -> Option<String> {
    look_config::current_tracker_machine()
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

/// The Herdr machine a command rides. `Some("")` is the local core's
/// own machine (t-24@r12: the hub's bare-path reads go through the
/// local core, never std::fs); `Some(name)` is a named machine; None
/// is the host file reader, test-only machinery since t-24@r11.
enum CoreRead {
    Local,
    Named(String),
    HostFile,
}

fn core_read_of(on: &Option<String>) -> CoreRead {
    match on.as_deref() {
        None => CoreRead::HostFile,
        Some("") => CoreRead::Local,
        Some(machine) => CoreRead::Named(machine.to_string()),
    }
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
    let text = match client.read_text(machine, path) {
        Ok(text) => text,
        Err(fail) => {
            let _ = log_bridge(
                "read_declaration",
                &format!("{}{path}", core_machine_label(machine)),
                &format!("failed: {}", fail.code()),
            );
            return Err(core_failure(fail));
        }
    };
    let _ = log_bridge(
        "read_declaration",
        &format!("{}{path}", core_machine_label(machine)),
        &format!("{} bytes", text.len()),
    );
    Ok(DeclaredText { text })
}

/// The log label of a core machine: the local core's own machine has
/// no name, and the log says so plainly.
fn core_machine_label(machine: &str) -> String {
    if machine.is_empty() {
        "(local core)".to_string()
    } else {
        format!("{machine}:")
    }
}

#[tauri::command]
pub fn read_declaration(path: String, on: Option<String>) -> Result<DeclaredText, BridgeError> {
    if path.is_empty() {
        return Err(deny("empty path"));
    }
    match core_read_of(&on) {
        CoreRead::Named(machine) => read_declaration_remote(&machine, &path),
        CoreRead::Local => read_declaration_remote("", &path),
        CoreRead::HostFile => read_text_resolved(&path, None),
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
    let bytes = match client.asset_bytes(machine, base, relative) {
        Ok(bytes) => bytes,
        Err(fail) => {
            let _ = log_bridge(
                "read_asset",
                &format!("{machine}:{base}/{relative}"),
                &format!("failed: {}", fail.code()),
            );
            return Err(core_failure(fail));
        }
    };
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
    match core_read_of(&on) {
        CoreRead::Named(machine) => read_asset_remote(&machine, &base, &relative),
        CoreRead::Local => read_asset_local_core(&base, &relative),
        CoreRead::HostFile => {
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

/// One asset through the local core's own machine (t-24@r12): the same
/// fixed-argv read the named machines use, with containment proven on
/// the core side, never a std::fs shortcut.
fn read_asset_local_core(base: &str, relative: &str) -> Result<Vec<u8>, BridgeError> {
    if !base.starts_with('/') {
        return Err(deny("remote base path must be absolute"));
    }
    if base.contains('\0') {
        return Err(deny("NUL byte in base path"));
    }
    let client = core_client()?;
    let bytes = match client.asset_bytes("", base, relative) {
        Ok(bytes) => bytes,
        Err(fail) => {
            let _ = log_bridge(
                "read_asset",
                &format!("(local core){base}/{relative}"),
                &format!("failed: {}", fail.code()),
            );
            return Err(core_failure(fail));
        }
    };
    let _ = log_bridge(
        "read_asset",
        &format!("(local core){base}/{relative}"),
        &format!("{} bytes", bytes.len()),
    );
    Ok(bytes)
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
    use super::{core_read_of, guard_relative, CoreRead};

    #[test]
    fn the_read_target_follows_the_machine_argument() {
        assert!(matches!(core_read_of(&None), CoreRead::HostFile));
        assert!(matches!(core_read_of(&Some("".into())), CoreRead::Local));
        assert!(
            matches!(core_read_of(&Some("tired".into())), CoreRead::Named(name) if name == "tired")
        );
    }

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
