//! Read-only declaration and asset reads. The guard mirrors the frontend
//! resolve rules exactly: relative segments only, no parent traversal, no
//! backslash separators, no absolute paths, no NUL bytes. Canonicalized
//! prefix checking runs second so a guard bypass still cannot escape.

use serde::{Deserialize, Serialize};

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

/// The frontend already applied these rules; repeat them here so the host
/// boundary holds even against a tampered renderer.
fn guard_relative(relative: &str) -> Result<(), BridgeError> {
    if relative.is_empty() {
        return Err(deny("empty path"));
    }
    if relative.contains('\0') {
        return Err(deny("NUL byte in path"));
    }
    if relative.contains('\\') {
        return Err(deny("backslash is not a separator"));
    }
    if relative.starts_with('/') || relative.starts_with('~') {
        return Err(deny("absolute path"));
    }
    if relative.split(':').count() > 1 && relative.chars().take(2).all(|c| c.is_ascii_alphabetic())
    {
        return Err(deny("windows-style absolute path"));
    }
    for segment in relative.split('/') {
        if segment == ".." {
            return Err(deny("parent traversal"));
        }
    }
    Ok(())
}

fn canonical_prefix_holds(base: &std::path::Path, joined: &std::path::Path) -> bool {
    match (base.canonicalize(), joined.canonicalize()) {
        (Ok(base), Ok(joined)) => joined.starts_with(base),
        _ => false,
    }
}

fn read_text_resolved(base: &str, relative: Option<&str>) -> Result<DeclaredText, BridgeError> {
    let base_path = std::path::Path::new(base);
    let target = match relative {
        None => base_path.to_path_buf(),
        Some(rel) => {
            guard_relative(rel)?;
            base_path.join(rel)
        }
    };
    if !canonical_prefix_holds(base_path, &target) {
        return Err(deny("resolved path left the declaration directory"));
    }
    let bytes = std::fs::read(&target).map_err(|_| missing(&target.to_string_lossy()))?;
    String::from_utf8(bytes)
        .map(|text| DeclaredText { text })
        .map_err(|_| BridgeError {
            code: "asset-denied".into(),
            message: "declaration bytes are not UTF-8 text".into(),
        })
}

#[tauri::command]
pub fn read_declaration(path: String) -> Result<DeclaredText, BridgeError> {
    if path.is_empty() {
        return Err(deny("empty path"));
    }
    read_text_resolved(&path, None)
}

#[tauri::command]
pub fn read_asset(base: String, relative: String) -> Result<Vec<u8>, BridgeError> {
    guard_relative(&relative)?;
    let base_path = std::path::Path::new(&base);
    let target = base_path.join(&relative);
    if !canonical_prefix_holds(base_path, &target) {
        return Err(deny("resolved path left the declaration directory"));
    }
    std::fs::read(&target).map_err(|_| missing(&target.to_string_lossy()))
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
    use super::guard_relative;

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
