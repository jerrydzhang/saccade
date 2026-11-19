//! The workbench declaration: `workbench.toml` at the repository's
//! root, the one file the app and the tracker both read, so what they
//! know cannot disagree. It holds the project's name and a `[tracker]`
//! section with its port; it names no machine - the machine is the one
//! the file was read on. The tracker reads its port from here; later
//! steps grow the same file (a root file with members, a runs
//! section).

use std::path::Path;

#[derive(serde::Deserialize)]
struct Declaration {
    #[serde(default)]
    tracker: Option<TrackerSection>,
}

#[derive(serde::Deserialize)]
struct TrackerSection {
    port: u16,
}

/// The port the repository's tracker serves at, read from the
/// declaration's `[tracker]` section - the one source, so a flag
/// cannot disagree with the file the app reads.
pub fn tracker_port(repo_root: &Path) -> Result<u16, String> {
    let path = repo_root.join("workbench.toml");
    let text = std::fs::read_to_string(&path).map_err(|_| {
        format!(
            "serve reads its port from workbench.toml at the repository's root; none is readable at {}",
            path.display()
        )
    })?;
    let declaration: Declaration =
        toml::from_str(&text).map_err(|e| format!("{} is unreadable: {e}", path.display()))?;
    declaration
        .tracker
        .map(|tracker| tracker.port)
        .ok_or_else(|| {
            format!(
                "{} carries no [tracker] section with its port",
                path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A per-test scratch root, unique per process: parallel lib tests
    /// never share a directory, and a killed run leaves nothing claimed.
    fn dir(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("sac-declare-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn the_port_comes_from_the_tracker_section() {
        let root = dir("port");
        fs::write(
            root.join("workbench.toml"),
            "name = \"saccade\"\n\n[tracker]\nport = 8811\n",
        )
        .unwrap();
        assert_eq!(tracker_port(&root).unwrap(), 8811);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_declaration_names_its_path() {
        let root = dir("missing");
        let message = tracker_port(&root).unwrap_err();
        assert!(message.contains("workbench.toml"), "{message}");
        assert!(message.contains(root.to_str().unwrap()), "{message}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_declaration_without_a_tracker_section_says_so() {
        let root = dir("no-tracker");
        fs::write(root.join("workbench.toml"), "name = \"saccade\"\n").unwrap();
        let message = tracker_port(&root).unwrap_err();
        assert!(message.contains("[tracker]"), "{message}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unreadable_declaration_carries_the_parse_fault() {
        let root = dir("broken");
        fs::write(root.join("workbench.toml"), "name = [unterminated\n").unwrap();
        let message = tracker_port(&root).unwrap_err();
        assert!(message.contains("unreadable"), "{message}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_port_outside_the_port_range_refuses() {
        let root = dir("range");
        fs::write(root.join("workbench.toml"), "[tracker]\nport = 70000\n").unwrap();
        assert!(tracker_port(&root).is_err());
        let _ = fs::remove_dir_all(&root);
    }
}
