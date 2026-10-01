use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq)]
pub enum PointerError {
    /// Session and worktree paths are absolute when first recorded.
    Relative,
    Empty,
    /// The pre-bases demand marker: `unrecorded` is never a commit a
    /// git context would produce, so any leak self-diagnoses.
    Reserved,
}

macro_rules! pointer {
    ($name:ident, $inner:ty, $wire:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(try_from = $wire, into = $wire)]
        pub struct $name($inner);
    };
}

pointer!(
    SessionPointer,
    PathBuf,
    "PathBuf",
    "Absolute path to a live session's backing store."
);
pointer!(
    WorktreePath,
    PathBuf,
    "PathBuf",
    "Absolute path to a task's worktree."
);
pointer!(
    GitCommit,
    String,
    "String",
    "Commit hash; canonicalized by Git before append."
);
pointer!(
    GitBranch,
    String,
    "String",
    "Branch name; canonicalized by Git before append."
);

impl TryFrom<PathBuf> for SessionPointer {
    type Error = String;

    fn try_from(path: PathBuf) -> Result<Self, String> {
        SessionPointer::new(path).map_err(|e| format!("{e:?}"))
    }
}

impl From<SessionPointer> for PathBuf {
    fn from(p: SessionPointer) -> PathBuf {
        p.0
    }
}

impl TryFrom<PathBuf> for WorktreePath {
    type Error = String;

    fn try_from(path: PathBuf) -> Result<Self, String> {
        WorktreePath::new(path).map_err(|e| format!("{e:?}"))
    }
}

impl From<WorktreePath> for PathBuf {
    fn from(p: WorktreePath) -> PathBuf {
        p.0
    }
}

impl TryFrom<String> for GitCommit {
    type Error = String;

    fn try_from(value: String) -> Result<Self, String> {
        GitCommit::new(value).map_err(|e| format!("{e:?}"))
    }
}

impl From<GitCommit> for String {
    fn from(p: GitCommit) -> String {
        p.0
    }
}

impl TryFrom<String> for GitBranch {
    type Error = String;

    fn try_from(value: String) -> Result<Self, String> {
        GitBranch::new(value).map_err(|e| format!("{e:?}"))
    }
}

impl From<GitBranch> for String {
    fn from(p: GitBranch) -> String {
        p.0
    }
}

impl SessionPointer {
    pub fn new(path: PathBuf) -> Result<Self, PointerError> {
        if !path.is_absolute() {
            return Err(PointerError::Relative);
        }
        Ok(Self(path))
    }
}

impl WorktreePath {
    pub fn new(path: PathBuf) -> Result<Self, PointerError> {
        if !path.is_absolute() {
            return Err(PointerError::Relative);
        }
        Ok(Self(path))
    }

    /// The runner executes git here; executors read their session pointer.
    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

impl GitCommit {
    /// The pre-bases marker is a plain word, deliberately not all-zeros:
    /// the null sha has legal git meaning, a plain word is illegal in
    /// every git context, so any leak self-diagnoses.
    pub(crate) const UNRECORDED: &'static str = "unrecorded";

    pub fn new(value: String) -> Result<Self, PointerError> {
        if value.is_empty() {
            return Err(PointerError::Empty);
        }
        if value == Self::UNRECORDED {
            return Err(PointerError::Reserved);
        }
        Ok(Self(value))
    }

    /// The decoder's bare-string map is this constructor's sole
    /// caller; every door constructs through `new`, which rejects
    /// the word.
    pub fn unrecorded() -> Self {
        Self(Self::UNRECORDED.to_string())
    }

    /// Views and tests state the hash a checkpoint recorded.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl GitBranch {
    pub fn new(value: String) -> Result<Self, PointerError> {
        if value.is_empty() {
            return Err(PointerError::Empty);
        }
        Ok(Self(value))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pointers_admit_only_absolute_paths_and_nonempty_git_values() {
        assert!(SessionPointer::new(PathBuf::from("/tmp/s")).is_ok());
        assert!(matches!(
            SessionPointer::new(PathBuf::from("rel/s")),
            Err(PointerError::Relative)
        ));
        assert!(WorktreePath::new(PathBuf::from("/wt")).is_ok());
        assert!(matches!(
            WorktreePath::new(PathBuf::from("wt")),
            Err(PointerError::Relative)
        ));
        assert!(GitCommit::new("abc123".into()).is_ok());
        assert!(matches!(
            GitCommit::new(String::new()),
            Err(PointerError::Empty)
        ));
        // the reserved word never enters through a door: only the
        // decoder's constructor mints it
        assert!(matches!(
            GitCommit::new("unrecorded".into()),
            Err(PointerError::Reserved)
        ));
        assert_eq!(GitCommit::unrecorded().as_str(), "unrecorded");
        assert!(GitBranch::new("refs/heads/main".into()).is_ok());
        assert!(matches!(
            GitBranch::new(String::new()),
            Err(PointerError::Empty)
        ));
    }
}
