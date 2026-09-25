//! State homes: the tracker's records and the executor's artifacts live
//! outside any repository, under one per-repo state root. The repo keeps
//! only what git tracks; nothing runtime accrues in the working tree.

use std::path::{Path, PathBuf};

/// The state root parent: `$XDG_STATE_HOME` or `~/.local/state`.
fn state_parent() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
    {
        return dir;
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".local").join("state")
}

/// A stable 32-bit FNV-1a of the path; hashers in std are not stable
/// across versions, and this slug must outlive them.
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// The repo's basename: the state-dir slug and the console's
/// instance name share this one extraction.
pub fn repo_basename(repo_root: &Path) -> String {
    repo_root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "repo".into())
}

/// The favicon's hue for a repo: the same root hash the state-dir
/// slug tails, reduced to degrees — one hash, two faces.
pub fn repo_hue(repo_root: &Path) -> u32 {
    fnv1a(repo_root.to_string_lossy().as_bytes()) % 360
}

/// The per-repo state directory: repo basename plus a short hash of the
/// resolved root, so same-named repos on one machine never collide.
pub fn state_dir(repo_root: &Path) -> PathBuf {
    let slug = format!(
        "{}-{:06x}",
        repo_basename(repo_root),
        fnv1a(repo_root.to_string_lossy().as_bytes())
    );
    state_parent().join("saccade").join(slug)
}

/// The main repo root for the current directory: worktrees resolve to the
/// repository they belong to, so a session inside a worktree finds the
/// same tracker as the main checkout.
pub fn repo_root(cwd: &Path) -> Result<PathBuf, String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .output()
        .map_err(|e| format!("git discovery failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "'{}' is not inside a git repository; pass --db to name the tracker",
            cwd.display()
        ));
    }
    let common = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    let root = common
        .parent()
        .ok_or("git reported a common dir without a parent")?
        .to_path_buf();
    Ok(root)
}

/// The default db location for a repo: its state root, not its working tree.
pub fn db_at(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("saccade.db")
}

/// The write-attempt log's home: beside the tracker it observes, so a
/// `--db`-named tracker gets its attempts named the same way.
pub fn attempts_at(db_path: &Path) -> PathBuf {
    db_path.with_file_name("attempts.jsonl")
}

/// The artifact store: content-addressed bytes under the state root,
/// one home per repo. The writer parks `<sha256>` files here; the read
/// door serves them. Lifetime is store-wide, never per task.
pub fn artifacts_at(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("artifacts")
}

/// Where a terminal task's session artifacts outlive their agent dir.
pub fn retention_at(repo_root: &Path, task: usize) -> PathBuf {
    state_dir(repo_root)
        .join("retention")
        .join(format!("t-{task}"))
}

/// The worktree a task's workspace provisions.
pub fn worktree_at(repo_root: &Path, task: usize) -> PathBuf {
    state_dir(repo_root)
        .join("worktrees")
        .join(format!("t-{task}"))
}

/// The session pointer's conventional backing file.
pub fn session_at(repo_root: &Path, task: usize) -> PathBuf {
    state_dir(repo_root)
        .join("sessions")
        .join(format!("t-{task}.jsonl"))
}

/// The pi agent dir a task's runs are composed under.
pub fn agent_dir_at(repo_root: &Path, task: usize) -> PathBuf {
    state_dir(repo_root).join("agent").join(format!("t-{task}"))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn slug_hashes_the_resolved_root_stably() {
        let a = state_dir(Path::new("/home/jerry/Projects/saccade"));
        let b = state_dir(Path::new("/home/jerry/Projects/saccade"));
        let other = state_dir(Path::new("/srv/saccade"));
        assert_eq!(a, b, "same root, same slug");
        assert_ne!(a, other, "same basename, different roots, different slugs");
        assert!(a.starts_with(state_parent().join("saccade")));
    }

    #[test]
    fn the_hue_is_the_slug_hash_reduced() {
        assert_eq!(repo_hue(Path::new("/srv/hornet")), 324);
        // same basename, different roots: the hues split where the slugs do
        assert_eq!(repo_hue(Path::new("/home/j/hornet")), 153);
    }
}
