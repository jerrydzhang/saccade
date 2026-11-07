//! The frontend resolve rules, repeated at the host boundary so a
//! tampered renderer cannot widen the read surface. Pure string checks;
//! canonical containment is checked separately: locally via
//! `canonicalize()`, and on the remote side inside
//! `client::CoreClient::asset_bytes`, where the host enforces nothing.

/// Relative segments only, no parent traversal, no backslash
/// separators, no absolute paths, no NUL bytes.
pub fn guard_relative(relative: &str) -> Result<(), &'static str> {
    if relative.is_empty() {
        return Err("empty path");
    }
    if relative.contains('\0') {
        return Err("NUL byte in path");
    }
    if relative.contains('\\') {
        return Err("backslash is not a separator");
    }
    if relative.starts_with('/') || relative.starts_with('~') {
        return Err("absolute path");
    }
    let mut chars = relative.chars();
    if let (Some(first), Some(second)) = (chars.next(), chars.next()) {
        if first.is_ascii_alphabetic() && second == ':' {
            return Err("windows-style absolute path");
        }
    }
    for segment in relative.split('/') {
        if segment == ".." {
            return Err("parent traversal");
        }
    }
    Ok(())
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
    fn nul_bytes_and_empty_paths_are_denied() {
        assert!(guard_relative("a\0b").is_err());
        assert!(guard_relative("").is_err());
    }
}
