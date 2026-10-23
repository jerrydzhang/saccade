//! The Git effects that provision task-free workspaces: the branch
//! mechanics a creation birth is born from. The birth carries what
//! these effects provisioned, never caller-claimed values; the fold
//! never checks the filesystem, and these functions never touch the
//! record.

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

/// An effect-layer refusal: operation codes, never fold refusals.
#[derive(Debug)]
pub enum WorkspaceFail {
    /// The composed branch name is already taken in the repo.
    BranchExists(GitBranch),
    /// The workspace's path is already occupied on disk.
    PathOccupied(PathBuf),
    /// The descriptive label is not a safe single path and ref component.
    InvalidLabel(String),
    /// The OS entropy source could not be read; no unique token exists.
    Entropy(String),
    /// A worktree directory could not be created: the home or the target.
    DirCreate { path: PathBuf, detail: String },
    /// A git command failed; the detail is its stderr.
    Git(String),
}

impl WorkspaceFail {
    pub fn code(&self) -> &'static str {
        match self {
            WorkspaceFail::BranchExists(_) => "workspace_branch_exists",
            WorkspaceFail::PathOccupied(_) => "workspace_path_occupied",
            WorkspaceFail::InvalidLabel(_) => "workspace_label_invalid",
            WorkspaceFail::Entropy(_) => "workspace_entropy_unavailable",
            WorkspaceFail::DirCreate { .. } => "workspace_dir_create_failed",
            WorkspaceFail::Git(_) => "workspace_effect",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            WorkspaceFail::BranchExists(branch) => format!(
                "branch {} already exists; provisioning would take its name",
                branch.as_str()
            ),
            WorkspaceFail::PathOccupied(path) => {
                format!("workspace path {} is already occupied", path.display())
            }
            WorkspaceFail::Git(detail) => detail.clone(),
            WorkspaceFail::InvalidLabel(reason) => {
                format!("invalid descriptive label: {reason}")
            }
            WorkspaceFail::Entropy(detail) => detail.clone(),
            WorkspaceFail::DirCreate { path, detail } => {
                format!(
                    "could not create worktree directory {}: {detail}",
                    path.display()
                )
            }
        }
    }
}

/// The facts a named provisioning established: actual Git pointers the
/// record will carry later. Plumbing for the controller, not a resident
/// object.
#[derive(Clone, Debug, PartialEq)]
pub struct Provisioned {
    pub base: GitCommit,
    pub branch: GitBranch,
    pub worktree: WorktreePath,
}

pub fn git(root: &Path, args: &[&str]) -> Result<String, WorkspaceFail> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| WorkspaceFail::Git(format!("git {}: {e}", args.first().unwrap_or(&""))))?;
    if !out.status.success() {
        return Err(WorkspaceFail::Git(format!(
            "git {} failed: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The canonical commit a name resolves to, or the repo has none.
fn resolve_commit(root: &Path, name: Option<&str>) -> Result<GitCommit, WorkspaceFail> {
    let reference = name.unwrap_or("HEAD");
    let tip = git(root, &["rev-parse", "--verify", reference])?;
    GitCommit::new(tip).map_err(|e| WorkspaceFail::Git(format!("git gave no commit: {e:?}")))
}

fn branch_exists(root: &Path, branch: &GitBranch) -> bool {
    std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("refs/heads/{}", branch.as_str()),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// One worktree cut shared by every provisioning shape: a fresh
/// branch at the canonical base, checked out at the path.
fn cut_worktree(
    root: &Path,
    branch: &GitBranch,
    path: &Path,
    base: &GitCommit,
) -> Result<(), WorkspaceFail> {
    git(
        root,
        &[
            "worktree",
            "add",
            "-q",
            "-b",
            branch.as_str(),
            &path.display().to_string(),
            base.as_str(),
        ],
    )
    .map(|_| ())
}

/// A fresh uniqueness token: real OS entropy, never counters, clocks,
/// or process ids. Exactly four bytes: /dev/urandom never ends, so
/// anything read-to-EOF here would allocate forever.
fn fresh_token() -> Result<String, WorkspaceFail> {
    let mut urandom = std::fs::File::open("/dev/urandom").map_err(|e| {
        WorkspaceFail::Entropy(format!("the OS entropy source is unavailable: {e}"))
    })?;
    token_from_reader(&mut urandom)
}

/// Four bytes from any reader, hex-encoded; the seam the contract test
/// pins entropy refusal through.
fn token_from_reader<R: Read>(reader: &mut R) -> Result<String, WorkspaceFail> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| WorkspaceFail::Entropy(format!("the OS entropy source failed: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// The label is one filename and ref component: the safety rules are
/// Git's ref-component rules plus path-separator refusal. Unicode
/// carries - this is data, never transliterated.
fn validate_label(label: &str) -> Result<(), WorkspaceFail> {
    let refused = |why: &str| {
        Err(WorkspaceFail::InvalidLabel(format!(
            "invalid label {label:?}: {why}"
        )))
    };
    if label.is_empty() {
        return refused("empty");
    }
    if label.trim() != label {
        return refused("leading or trailing whitespace");
    }
    if label.starts_with('-') {
        return refused("leading dash reads as a command-line flag");
    }
    if label.starts_with('.') || label.ends_with('.') {
        return refused("dots at either end");
    }
    if label.ends_with(".lock") {
        return refused("the .lock suffix is a ref lock file");
    }
    if label.contains("..") {
        return refused("consecutive dots");
    }
    if label.contains("@{") {
        return refused("the @{ sequence is reflog syntax");
    }
    if label.chars().any(|c| {
        matches!(
            c,
            '/' | '\\' | ' ' | '~' | '^' | ':' | '?' | '*' | '[' | ']'
        ) || c.is_control()
    }) {
        return refused("characters a path or ref component cannot carry");
    }
    Ok(())
}

/// Provision a NEW task-independent workspace: a fresh branch named
/// from the optional descriptive label plus a generated uniqueness
/// suffix (workspace-<token> without a label), cut at the named base
/// or the primary HEAD, checked out under the home. Actual Git facts
/// come back; nothing existing is ever touched or adopted.
pub fn provision(
    root: &Path,
    worktree_home: &Path,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<Provisioned, WorkspaceFail> {
    let token = fresh_token()?;
    provision_with_token(root, worktree_home, label, base, &token)
}

/// The same provisioning with the token supplied: the test seam keeps
/// entropy out of name assertions.
fn provision_with_token(
    root: &Path,
    worktree_home: &Path,
    label: Option<&str>,
    base: Option<&str>,
    token: &str,
) -> Result<Provisioned, WorkspaceFail> {
    if let Some(label) = label {
        validate_label(label)?;
    }
    let name = match label {
        Some(label) => format!("{label}-{token}"),
        None => format!("workspace-{token}"),
    };
    let branch = GitBranch::new(name.clone())
        .map_err(|e| WorkspaceFail::Git(format!("the composed name is no branch: {e:?}")))?;
    let base = resolve_commit(root, base)?;
    if branch_exists(root, &branch) {
        return Err(WorkspaceFail::BranchExists(branch));
    }
    let path = worktree_home.join(&name);
    // the home is built only after every check above passed
    if let Err(e) = std::fs::create_dir_all(worktree_home) {
        return Err(WorkspaceFail::DirCreate {
            path: worktree_home.to_path_buf(),
            detail: e.to_string(),
        });
    }
    // the exclusive claim: existing, even empty or a dangling symlink,
    // refuses before git fills our freshly owned directory
    if let Err(e) = std::fs::create_dir(&path) {
        return Err(match e.kind() {
            std::io::ErrorKind::AlreadyExists => WorkspaceFail::PathOccupied(path),
            _ => WorkspaceFail::DirCreate {
                path: path.clone(),
                detail: e.to_string(),
            },
        });
    }
    cut_worktree(root, &branch, &path, &base).map_err(|e| match e {
        // no rollback: name the attempt and that git may have left pieces
        WorkspaceFail::Git(detail) => WorkspaceFail::Git(format!(
            "provisioning branch {} at {} failed: {detail}; \
             git may have left the branch or directory behind",
            branch.as_str(),
            path.display()
        )),
        other => other,
    })?;
    let worktree = WorktreePath::new(path).map_err(|e| WorkspaceFail::Git(format!("{e:?}")))?;
    Ok(Provisioned {
        base,
        branch,
        worktree,
    })
}

/// The worktree home: worktrees live beside the tracker's own db.
pub fn worktree_home(db_path: &Path) -> PathBuf {
    db_path
        .parent()
        .map(|dir| dir.join("worktrees"))
        .unwrap_or_else(|| PathBuf::from("worktrees"))
}

#[cfg(test)]
mod test {
    use super::*;

    /// A scratch repo with one commit; returns its root.
    fn scratch_repo(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sac-wsfx-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init", "-q"]).unwrap();
        git(&dir, &["config", "user.email", "t@t"]).unwrap();
        git(&dir, &["config", "user.name", "t"]).unwrap();
        std::fs::write(dir.join("f.txt"), "base\n").unwrap();
        git(&dir, &["add", "."]).unwrap();
        git(&dir, &["commit", "-qm", "base"]).unwrap();
        dir
    }

    #[test]
    fn provision_cuts_from_the_primary_head_or_a_named_base() {
        let root = scratch_repo("provision");
        std::fs::write(root.join("f.txt"), "advanced\n").unwrap();
        git(&root, &["commit", "-qam", "advanced"]).unwrap();

        let home = root.join("worktrees");

        // named provisioning: label plus injected token, cut at HEAD
        let facts = provision_with_token(&root, &home, Some("feature"), None, "abc123ab").unwrap();
        assert_eq!(facts.base, resolve_commit(&root, None).unwrap());
        assert_eq!(facts.branch.as_str(), "feature-abc123ab");
        assert_eq!(facts.worktree.as_path(), home.join("feature-abc123ab"));
        assert!(home.join("feature-abc123ab").join("f.txt").is_file());
        let feature_branch = facts.branch.clone();
        let tip_of = |branch: &GitBranch| -> Option<String> {
            if !branch_exists(&root, branch) {
                return None;
            }
            git(
                &root,
                &[
                    "rev-parse",
                    "--verify",
                    &format!("refs/heads/{}", branch.as_str()),
                ],
            )
            .ok()
        };
        let feature_tip = tip_of(&feature_branch);

        // an omitted label gets the ratified fallback name
        let facts = provision_with_token(&root, &home, None, None, "deadbeef").unwrap();
        assert_eq!(facts.branch.as_str(), "workspace-deadbeef");
        assert_eq!(facts.worktree.as_path(), home.join("workspace-deadbeef"));

        // an explicit base names the canonical commit
        std::fs::write(root.join("f.txt"), "newer\n").unwrap();
        git(&root, &["commit", "-qam", "newer"]).unwrap();
        let named_base = resolve_commit(&root, Some("HEAD~1")).unwrap();
        let facts =
            provision_with_token(&root, &home, Some("pinned"), Some("HEAD~1"), "cafe0000").unwrap();
        assert_eq!(facts.base, named_base);

        // unicode labels are data: preserved exactly, never transliterated
        let facts = provision_with_token(
            &root,
            &home,
            Some("\u{e9}tude-\u{4e2d}\u{6587}"),
            None,
            "f00dcafe",
        )
        .unwrap();
        assert_eq!(
            facts.branch.as_str(),
            "\u{e9}tude-\u{4e2d}\u{6587}-f00dcafe"
        );
        assert!(
            home.join("\u{e9}tude-\u{4e2d}\u{6587}-f00dcafe")
                .join("f.txt")
                .is_file()
        );

        // unsafe labels refuse before any side effect: no branch, no directory
        let bad = [
            "../escape",
            "a/b",
            "a\\b",
            "-flag",
            ".lead",
            "tail.",
            "end.lock",
            "a..b",
            "a@{b",
            "a b",
            "a~b",
            "a^b",
            "a:b",
            "a?b",
            "a*b",
            "a[b]",
            "",
            "  ",
            "pad ",
        ];
        for label in bad {
            let refused = provision_with_token(&root, &home, Some(label), None, "abc123ab");
            assert!(
                matches!(refused, Err(WorkspaceFail::InvalidLabel(_))),
                "label {label:?} must refuse"
            );
        }
        assert_eq!(
            tip_of(&feature_branch),
            feature_tip,
            "the earlier provisioned branch is untouched by the refusals"
        );
        let created: Vec<_> = std::fs::read_dir(&home)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            created.len(),
            4,
            "the four provisions, nothing else: {created:?}"
        );
        for name in [
            "feature-abc123ab",
            "workspace-deadbeef",
            "pinned-cafe0000",
            "\u{e9}tude-\u{4e2d}\u{6587}-f00dcafe",
        ] {
            assert!(home.join(name).is_dir(), "provision {name} exists");
        }

        // a token collision refuses: an existing branch is never reused
        let refused = provision_with_token(&root, &home, Some("feature"), None, "abc123ab");
        assert!(matches!(refused, Err(WorkspaceFail::BranchExists(_))));
        // a path collision refuses: the occupied directory is never adopted
        std::fs::create_dir_all(home.join("workspace-c011dce1")).unwrap();
        let refused = provision_with_token(&root, &home, None, None, "c011dce1");
        assert!(matches!(refused, Err(WorkspaceFail::PathOccupied(_))));
        assert!(
            !root
                .join(".git")
                .join("refs")
                .join("heads")
                .join("workspace-c011dce1")
                .exists()
        );

        // an existing empty directory is occupied all the same
        std::fs::create_dir_all(home.join("workspace-0cc1ded0")).unwrap();
        let refused = provision_with_token(&root, &home, None, None, "0cc1ded0");
        assert!(matches!(refused, Err(WorkspaceFail::PathOccupied(_))));
        assert!(
            home.join("workspace-0cc1ded0")
                .read_dir()
                .unwrap()
                .next()
                .is_none(),
            "the empty directory was never populated"
        );
        assert!(!branch_exists(
            &root,
            &GitBranch::new("workspace-0cc1ded0".into()).unwrap()
        ));

        // a dangling symlink at the target is occupied all the same
        std::os::unix::fs::symlink(home.join("no-such-place"), home.join("workspace-dab9ced0"))
            .unwrap();
        let refused = provision_with_token(&root, &home, None, None, "dab9ced0");
        assert!(matches!(refused, Err(WorkspaceFail::PathOccupied(_))));
        assert!(
            home.join("workspace-dab9ced0")
                .symlink_metadata()
                .unwrap()
                .file_type()
                .is_symlink(),
            "the dangling symlink was never replaced"
        );
        assert!(!branch_exists(
            &root,
            &GitBranch::new("workspace-dab9ced0".into()).unwrap()
        ));

        // fresh tokens come from real entropy: two live calls differ
        let a = provision(&root, &home, Some("live"), None).unwrap();
        let b = provision(&root, &home, Some("live"), None).unwrap();
        assert_ne!(a.branch.as_str(), b.branch.as_str());

        // the entropy seam refuses when the source fails: four bytes or nothing
        struct DeadEntropy;
        impl std::io::Read for DeadEntropy {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("no entropy today"))
            }
        }
        let refused = token_from_reader(&mut DeadEntropy);
        assert!(matches!(refused, Err(WorkspaceFail::Entropy(_))));
        // a good source reads exactly four bytes into eight hex digits
        let mut four: &[u8] = &[0xde, 0xad, 0xbe, 0xef];
        assert_eq!(token_from_reader(&mut four).unwrap(), "deadbeef");

        std::fs::remove_dir_all(&root).unwrap();
    }
}
