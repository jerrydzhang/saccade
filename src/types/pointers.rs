use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq)]
pub enum PointerError {
    /// Session and worktree paths are absolute when first recorded.
    Relative,
    Empty,
}

macro_rules! pointer {
    ($name:ident, $inner:ty, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $name($inner);
    };
}

pointer!(
    SessionPointer,
    PathBuf,
    "Absolute path to a live session's backing store."
);
pointer!(WorktreePath, PathBuf, "Absolute path to a task's worktree.");
pointer!(
    GitCommit,
    String,
    "Commit hash; canonicalized by Git before append."
);
pointer!(
    GitBranch,
    String,
    "Branch name; canonicalized by Git before append."
);

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
    pub fn new(value: String) -> Result<Self, PointerError> {
        if value.is_empty() {
            return Err(PointerError::Empty);
        }
        Ok(Self(value))
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
        assert!(GitBranch::new("refs/heads/main".into()).is_ok());
        assert!(matches!(
            GitBranch::new(String::new()),
            Err(PointerError::Empty)
        ));
    }
}
