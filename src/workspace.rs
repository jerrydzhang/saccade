//! The Git effects that establish workspace facts: the checkout and
//! branch mechanics the record's workspace lineage is born from. The
//! birth carries what these effects established, never caller-claimed
//! values; the fold never checks the filesystem, and these functions
//! never touch the record.

use std::path::{Path, PathBuf};

use crate::objects::task::TaskId;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

/// An effect-layer refusal: operation codes, never fold refusals.
#[derive(Debug)]
pub enum WorkspaceFail {
    /// The task's default branch name is already taken in the repo.
    BranchExists(GitBranch),
    /// The recorded branch is gone; recovery of a missing branch is not built.
    BranchMissing(GitBranch),
    /// The workspace's path is already occupied on disk.
    PathOccupied(PathBuf),
    /// The checkout carries staged, unstaged, or untracked work.
    Dirty { path: PathBuf, files: Vec<String> },
    /// The descriptive label is not a safe single path and ref component.
    InvalidLabel(String),
    /// A git command failed; the detail is its stderr.
    Git(String),
}

impl WorkspaceFail {
    pub fn code(&self) -> &'static str {
        match self {
            WorkspaceFail::BranchExists(_) => "workspace_branch_exists",
            WorkspaceFail::BranchMissing(_) => "workspace_branch_missing",
            WorkspaceFail::PathOccupied(_) => "workspace_path_occupied",
            WorkspaceFail::Dirty { .. } => "workspace_dirty",
            WorkspaceFail::InvalidLabel(_) => "workspace_label_invalid",
            WorkspaceFail::Git(_) => "workspace_effect",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            WorkspaceFail::BranchExists(branch) => format!(
                "branch {} already exists; the establishment would take another task's branch",
                branch.as_str()
            ),
            WorkspaceFail::BranchMissing(branch) => format!(
                "branch {} no longer exists in the repo; recovery of a missing branch is not built",
                branch.as_str()
            ),
            WorkspaceFail::PathOccupied(path) => {
                format!("workspace path {} is already occupied", path.display())
            }
            WorkspaceFail::Dirty { path, files } => {
                let listed = files
                    .iter()
                    .take(8)
                    .map(|f| f.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let rest = files.len().saturating_sub(8);
                let more = if rest > 0 {
                    format!(", and {rest} more")
                } else {
                    String::new()
                };
                format!(
                    "workspace {} carries staged, unstaged, or untracked work: {listed}{more}; \
                     commit or remove it before cleanup",
                    path.display()
                )
            }
            WorkspaceFail::Git(detail) => detail.clone(),
            WorkspaceFail::InvalidLabel(reason) => {
                format!("invalid descriptive label: {reason}")
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

/// The facts an establishment established: exactly what the birth records.
#[derive(Clone, Debug, PartialEq)]
pub struct Established {
    pub task: TaskId,
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

/// One worktree cut shared by every establishment shape: a fresh
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
fn fresh_token() -> String {
    let mut bytes = [0u8; 4];
    let filled = std::fs::File::open("/dev/urandom")
        .is_ok_and(|mut file| std::io::Read::read_exact(&mut file, &mut bytes).is_ok());
    if !filled {
        // no urandom: the caller's collision refusal is the safety net
        return "00000000".to_string();
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
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
    provision_with_token(root, worktree_home, label, base, &fresh_token())
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
    if path.exists() {
        return Err(WorkspaceFail::PathOccupied(path));
    }
    cut_worktree(root, &branch, &path, &base)?;
    let worktree = WorktreePath::new(path).map_err(|e| WorkspaceFail::Git(format!("{e:?}")))?;
    Ok(Provisioned {
        base,
        branch,
        worktree,
    })
}

/// The task's default branch name.
pub fn branch_name(task: TaskId) -> GitBranch {
    GitBranch::new(format!("saccade/t-{}", task.0)).expect("the scheme names a valid branch")
}

/// Establish: a new branch at the named base (the primary checkout's
/// HEAD by default) and its worktree under the tracker's home; nothing
/// existing is ever touched.
pub fn establish(
    root: &Path,
    worktree_home: &Path,
    task: TaskId,
    base: Option<&str>,
) -> Result<Established, WorkspaceFail> {
    let base = resolve_commit(root, base)?;
    let branch = branch_name(task);
    if branch_exists(root, &branch) {
        return Err(WorkspaceFail::BranchExists(branch));
    }
    let path = worktree_home.join(format!("t-{}", task.0));
    if path.exists() {
        return Err(WorkspaceFail::PathOccupied(path));
    }
    cut_worktree(root, &branch, &path, &base)?;
    let worktree = WorktreePath::new(path).map_err(|e| WorkspaceFail::Git(format!("{e:?}")))?;
    Ok(Established {
        task,
        base,
        branch,
        worktree,
    })
}

/// Ensure the recorded checkout exists: a present one is left alone,
/// a missing one is recreated from its branch's current tip (committed
/// advancement preserved). Identity and record untouched.
pub fn ensure_checkout(root: &Path, facts: &Established) -> Result<bool, WorkspaceFail> {
    let path = facts.worktree.as_path();
    if path.is_dir() {
        return Ok(false);
    }
    if !branch_exists(root, &facts.branch) {
        return Err(WorkspaceFail::BranchMissing(facts.branch.clone()));
    }
    let add = |force: bool| {
        let path_arg = path.display().to_string();
        let branch_arg = facts.branch.as_str();
        let mut args = vec!["worktree", "add", "-q"];
        if force {
            args.push("-f");
        }
        args.push(path_arg.as_str());
        args.push(branch_arg);
        git(root, &args)
    };
    match add(false) {
        Ok(_) => Ok(true),
        // the stale registration for a checkout that vanished is the
        // recreate case: clear it for this path only
        Err(WorkspaceFail::Git(detail)) if detail.contains("missing but already registered") => {
            add(true).map(|_| true)
        }
        Err(other) => Err(other),
    }
}

/// The teardown gate: a checkout carrying staged, unstaged, or
/// untracked work refuses cleanup before anything destructive runs;
/// ignored files are not work, and an absent checkout has none. The
/// untracked inventory is forced on, so a display config like
/// status.showUntrackedFiles=no cannot hide work from the gate.
pub fn refuse_dirty(facts: &Established) -> Result<(), WorkspaceFail> {
    let path = facts.worktree.as_path();
    if !path.is_dir() {
        return Ok(());
    }
    let out = git(path, &["status", "--porcelain", "--untracked-files=all"])?;
    let files: Vec<String> = out.lines().map(str::to_string).collect();
    if files.is_empty() {
        return Ok(());
    }
    Err(WorkspaceFail::Dirty {
        path: path.to_path_buf(),
        files,
    })
}

pub fn remove_checkout(root: &Path, facts: &Established) -> Result<bool, WorkspaceFail> {
    let path = facts.worktree.as_path();
    if !path.is_dir() {
        return Ok(false);
    }
    git(root, &["worktree", "remove", &path.display().to_string()])?;
    Ok(true)
}

pub fn branch_tip(root: &Path, branch: &GitBranch) -> Result<Option<GitCommit>, WorkspaceFail> {
    if !branch_exists(root, branch) {
        return Ok(None);
    }
    let tip = git(
        root,
        &[
            "rev-parse",
            "--verify",
            &format!("refs/heads/{}", branch.as_str()),
        ],
    )?;
    GitCommit::new(tip)
        .map(Some)
        .map_err(|e| WorkspaceFail::Git(format!("git gave no commit: {e:?}")))
}

/// A recorded workspace's facts read back for the effects.
pub fn recorded(
    task: TaskId,
    workspace: &crate::objects::workspace::WorkspaceContext,
) -> Established {
    Established {
        task,
        base: workspace.workspace.base.clone(),
        branch: workspace.workspace.branch.clone(),
        worktree: workspace.workspace.worktree.clone(),
    }
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
    fn establish_cuts_from_the_primary_head_or_a_named_base() {
        let root = scratch_repo("establish");
        std::fs::write(root.join("f.txt"), "advanced\n").unwrap();
        git(&root, &["commit", "-qam", "advanced"]).unwrap();
        let head = resolve_commit(&root, None).unwrap();

        // default: the primary checkout's current HEAD, not a ref name
        let home = root.join("worktrees");
        let facts = establish(&root, &home, TaskId(0), None).unwrap();
        assert_eq!(facts.base, head);
        assert_eq!(facts.branch.as_str(), "saccade/t-0");
        assert_eq!(facts.worktree.as_path(), home.join("t-0"));
        assert!(home.join("t-0").join("f.txt").is_file());

        // an explicit base names another commit: HEAD~1 is the prior one
        std::fs::write(root.join("f.txt"), "stacked\n").unwrap();
        git(&root, &["commit", "-qam", "stacked"]).unwrap();
        let facts = establish(&root, &home, TaskId(1), Some("HEAD~1")).unwrap();
        assert_eq!(facts.base, head);
        assert!(home.join("t-1").join("f.txt").is_file());

        // the branch name is taken: refuse, touching nothing
        let refused = establish(&root, &home, TaskId(0), None);
        assert!(matches!(refused, Err(WorkspaceFail::BranchExists(_))));
        // the path is occupied: refuse, touching nothing
        std::fs::create_dir_all(home.join("t-2")).unwrap();
        let refused = establish(&root, &home, TaskId(2), None);
        assert!(matches!(refused, Err(WorkspaceFail::PathOccupied(_))));

        // named provisioning: label plus injected token, cut at HEAD
        let facts = provision_with_token(&root, &home, Some("feature"), None, "abc123ab").unwrap();
        assert_eq!(facts.base, resolve_commit(&root, None).unwrap());
        assert_eq!(facts.branch.as_str(), "feature-abc123ab");
        assert_eq!(facts.worktree.as_path(), home.join("feature-abc123ab"));
        assert!(home.join("feature-abc123ab").join("f.txt").is_file());
        let feature_branch = facts.branch.clone();
        let feature_tip = branch_tip(&root, &feature_branch).unwrap();

        // an omitted label gets the ratified fallback name
        let facts = provision_with_token(&root, &home, None, None, "deadbeef").unwrap();
        assert_eq!(facts.branch.as_str(), "workspace-deadbeef");
        assert_eq!(facts.worktree.as_path(), home.join("workspace-deadbeef"));

        // an explicit base names the canonical commit, as in the task shape
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
            branch_tip(&root, &feature_branch).unwrap(),
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
            7,
            "four provisions plus the three task worktrees: {created:?}"
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

        // fresh tokens come from real entropy: two live calls differ
        let a = provision(&root, &home, Some("live"), None).unwrap();
        let b = provision(&root, &home, Some("live"), None).unwrap();
        assert_ne!(a.branch.as_str(), b.branch.as_str());

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn recreation_continues_from_the_branch_tip_and_leaves_a_present_checkout_alone() {
        let root = scratch_repo("recreate");
        let home = root.join("worktrees");
        let facts = establish(&root, &home, TaskId(0), None).unwrap();

        // committed advancement past the checkpoint, then the checkout dies
        let wt = facts.worktree.as_path();
        std::fs::write(wt.join("f.txt"), "advanced\n").unwrap();
        git(wt, &["commit", "-qam", "advanced"]).unwrap();
        let tip = resolve_commit(wt, None).unwrap();
        // uncommitted dirt stays out of the way: it dies with the checkout
        std::fs::write(wt.join("f.txt"), "uncommitted\n").unwrap();
        std::fs::remove_dir_all(wt).unwrap();

        // recreate: from the branch's current tip, advancement preserved
        assert!(ensure_checkout(&root, &facts).unwrap());
        assert_eq!(resolve_commit(wt, None).unwrap(), tip);
        assert!(wt.join(".git").exists(), "a real worktree, not a copy");

        // a present checkout is never touched, dirt included
        std::fs::write(wt.join("f.txt"), "dirty\n").unwrap();
        assert!(!ensure_checkout(&root, &facts).unwrap());
        assert_eq!(
            std::fs::read_to_string(wt.join("f.txt")).unwrap(),
            "dirty\n"
        );

        // a vanished branch is not recovered
        std::fs::remove_dir_all(wt).unwrap();
        git(&root, &["worktree", "prune"]).unwrap();
        git(&root, &["branch", "-qD", facts.branch.as_str()]).unwrap();
        let refused = ensure_checkout(&root, &facts);
        assert!(matches!(refused, Err(WorkspaceFail::BranchMissing(_))));

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_teardown_gate_refuses_work_but_lets_ignored_files_pass() {
        let root = scratch_repo("gate");
        let home = root.join("worktrees");
        let facts = establish(&root, &home, TaskId(0), None).unwrap();
        let wt = facts.worktree.as_path();

        // a clean checkout passes
        refuse_dirty(&facts).unwrap();

        // unstaged work refuses
        std::fs::write(wt.join("f.txt"), "edited\n").unwrap();
        let refused = refuse_dirty(&facts);
        assert!(matches!(refused, Err(WorkspaceFail::Dirty { .. })));

        // staged work refuses
        git(wt, &["add", "f.txt"]).unwrap();
        let refused = refuse_dirty(&facts);
        assert!(matches!(refused, Err(WorkspaceFail::Dirty { .. })));

        // untracked work refuses
        git(wt, &["reset", "-q"]).unwrap();
        git(wt, &["checkout", "-q", "--", "f.txt"]).unwrap();
        std::fs::write(wt.join("stray.txt"), "untracked\n").unwrap();
        let refused = refuse_dirty(&facts);
        assert!(matches!(refused, Err(WorkspaceFail::Dirty { .. })));

        // untracked work refuses even when the display config hides it
        std::fs::remove_file(wt.join("stray.txt")).unwrap();
        git(&root, &["config", "status.showUntrackedFiles", "no"]).unwrap();
        std::fs::write(wt.join("hidden.txt"), "untracked\n").unwrap();
        let refused = refuse_dirty(&facts);
        assert!(matches!(refused, Err(WorkspaceFail::Dirty { .. })));
        std::fs::remove_file(wt.join("hidden.txt")).unwrap();

        // ignored files are not work
        std::fs::write(wt.join(".gitignore"), "ignored.log\n").unwrap();
        git(wt, &["add", ".gitignore"]).unwrap();
        git(wt, &["commit", "-qm", "ignore pattern"]).unwrap();
        std::fs::write(wt.join("ignored.log"), "noise\n").unwrap();
        refuse_dirty(&facts).unwrap();

        // an absent checkout has nothing to refuse
        std::fs::remove_dir_all(wt).unwrap();
        refuse_dirty(&facts).unwrap();

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn checkout_removal_keeps_the_branch_and_its_unmerged_commit() {
        let root = scratch_repo("remove");
        let home = root.join("worktrees");
        let facts = establish(&root, &home, TaskId(0), None).unwrap();
        let wt = facts.worktree.as_path();
        std::fs::write(wt.join("f.txt"), "unmerged\n").unwrap();
        git(wt, &["commit", "-qam", "unmerged work"]).unwrap();
        let tip = resolve_commit(wt, None).unwrap();

        assert!(remove_checkout(&root, &facts).unwrap());
        assert!(!wt.exists());
        assert_eq!(branch_tip(&root, &facts.branch).unwrap(), Some(tip.clone()));

        // an absent checkout is already removed
        assert!(!remove_checkout(&root, &facts).unwrap());

        // the branch survives: the checkout is recreated from its tip
        assert!(ensure_checkout(&root, &facts).unwrap());
        assert_eq!(resolve_commit(wt, None).unwrap(), tip);

        std::fs::remove_dir_all(&root).unwrap();
    }
}
