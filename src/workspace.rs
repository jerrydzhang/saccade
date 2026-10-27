//! The Git effects that provision task-free workspaces: the branch
//! mechanics a creation birth is born from. The birth carries what
//! these effects provisioned, never caller-claimed values; the fold
//! never checks the filesystem, and these functions never touch the
//! record.

use std::collections::BTreeMap;
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

// -- the one lifecycle address resolver ----------------------------------
// Every door that names a recorded workspace resolves its address
// through this resolver: a birth is addressed by its own stored
// string or by the canonical form it resolves to, stored identities
// are never rewritten, and several stored paths over one directory
// refuse as ambiguous rather than being picked between. Filesystem
// resolution happens here in the effect/adapter boundary - the fold
// never reads the filesystem.

/// The address resolution over the recorded births.
#[derive(Debug, PartialEq)]
pub enum LifecycleAddress {
    /// Exactly one birth stands at this address
    Single(RecordId),
    /// Several stored paths resolve here: unsafe ambiguity, refused
    Ambiguous(Vec<RecordId>),
    /// No recorded birth addresses this path
    Unrecorded,
}

use crate::store::{RecordId, World};

/// Resolve one caller path to its recorded workspace birth, for every
/// door that names one - clean, restore, reconcile, and the
/// ordinary-use guards alike, so no door can drift apart again.
pub fn resolve_workspace_address(
    world: &World,
    path: &Path,
) -> Result<LifecycleAddress, WorkspaceFail> {
    let canonical = canonical_dir(path)?;
    let mut births = world.workspace_births_at(&canonical);
    for birth in world.workspace_births() {
        if births.contains(&birth) {
            continue;
        }
        let stored = world.workspace_facts(birth).map(|f| f.worktree);
        let aliases = stored
            .map(|p| canonical_dir(p.as_path()).map(|c| c == canonical))
            .unwrap_or(Ok(false));
        if aliases.unwrap_or(false) {
            births.push(birth);
        }
    }
    births.sort();
    Ok(match births.as_slice() {
        [] => LifecycleAddress::Unrecorded,
        [one] => LifecycleAddress::Single(*one),
        many => LifecycleAddress::Ambiguous(many.to_vec()),
    })
}

// -- the workspace lifecycle effects -------------------------------------
// Checkout inspection and removal under normal Git protections. Every
// function states what it verified; none touches the record. The
// retained branch is never deleted by any door here: git worktree
// remove leaves refs alone, and no other verb runs.

impl WorkspaceFail {
    pub(crate) fn fs(path: &Path, detail: std::io::Error) -> WorkspaceFail {
        WorkspaceFail::DirCreate {
            path: path.to_path_buf(),
            detail: detail.to_string(),
        }
    }
}

/// The canonical absolute form of a path whose leaf may be absent:
/// canonicalize the deepest existing ancestor, then append the remainder.
/// A recorded workspace's path compares equal only in this form.
pub fn canonical_dir(path: &Path) -> Result<PathBuf, WorkspaceFail> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let cwd = std::env::current_dir().map_err(|e| WorkspaceFail::fs(path, e))?;
        cwd.join(path)
    };
    let mut remainder = Vec::new();
    let mut probe = path.as_path();
    while !probe.exists() {
        match probe.parent() {
            Some(parent) => {
                remainder.push(probe.file_name().ok_or_else(|| {
                    WorkspaceFail::Git(format!("{} has no final component", path.display()))
                })?);
                probe = parent;
            }
            None => break,
        }
    }
    let mut out = std::fs::canonicalize(probe)
        .map_err(|e| WorkspaceFail::Git(format!("cannot canonicalize {}: {e}", path.display())))?;
    for part in remainder.iter().rev() {
        out.push(part);
    }
    Ok(out)
}

/// The worktree registrations git holds for this repo: canonical path
/// to the checked-out branch (None for a bare or detached entry).
/// These are git's own facts, never the record's claims.
pub fn registered_worktrees(
    root: &Path,
) -> Result<BTreeMap<PathBuf, Option<GitBranch>>, WorkspaceFail> {
    let out = git(root, &["worktree", "list", "--porcelain"])?;
    let mut map = BTreeMap::new();
    let mut current: Option<PathBuf> = None;
    for line in out.lines() {
        if let Some(path) = line.strip_prefix("worktree ") {
            current = Some(PathBuf::from(path));
            map.entry(PathBuf::from(path)).or_insert(None);
        } else if let Some(reference) = line.strip_prefix("branch ")
            && let Some(path) = &current
        {
            let name = reference
                .strip_prefix("refs/heads/")
                .unwrap_or(reference)
                .to_string();
            map.insert(path.clone(), GitBranch::new(name).ok());
        }
    }
    Ok(map)
}

/// The local branch's current tip, or None when git holds no such
/// branch. Quiet verification: a missing branch is a fact, not a
/// plumbing failure.
pub fn branch_tip(root: &Path, branch: &GitBranch) -> Result<Option<GitCommit>, WorkspaceFail> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("refs/heads/{}", branch.as_str()),
        ])
        .output()
        .map_err(|e| WorkspaceFail::Git(format!("git rev-parse: {e}")))?;
    if !out.status.success() {
        return Ok(None);
    }
    GitCommit::new(String::from_utf8_lossy(&out.stdout).trim().to_string())
        .map(Some)
        .map_err(|e| WorkspaceFail::Git(format!("git gave no commit: {e:?}")))
}

/// The checkout's preservation check: staged, unstaged, and untracked
/// changes all refuse; ignored files are git's own answer and may
/// disappear. Some detail names what stands in the way.
pub fn preserved_work(path: &Path) -> Result<Option<String>, WorkspaceFail> {
    let out = git(path, &["status", "--porcelain"])?;
    Ok(if out.is_empty() { None } else { Some(out) })
}

/// Remove the checkout with git's normal protections: no force, no
/// branch deletion - the branch and every commit on it stand.
pub fn remove_checkout(root: &Path, path: &Path) -> Result<(), WorkspaceFail> {
    git(root, &["worktree", "remove", &path.display().to_string()])
        .map(|_| ())
        .map_err(|e| match e {
            WorkspaceFail::Git(detail) => WorkspaceFail::Git(format!(
                "git worktree remove {} refused: {detail}",
                path.display()
            )),
            other => other,
        })
}

/// Establish the checkout at the branch's current local tip with git's
/// normal protections: no force, no base, no pull - a branch already
/// checked out elsewhere refuses, as does a missing-but-registered
/// worktree at the path. The branch's tip is git's, never a reset.
pub fn add_checkout(root: &Path, path: &Path, branch: &GitBranch) -> Result<(), WorkspaceFail> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| WorkspaceFail::fs(parent, e))?;
    }
    git(
        root,
        &[
            "worktree",
            "add",
            &path.display().to_string(),
            branch.as_str(),
        ],
    )
    .map(|_| ())
    .map_err(|e| match e {
        WorkspaceFail::Git(detail) => WorkspaceFail::Git(format!(
            "git worktree add {} {} refused: {detail}",
            path.display(),
            branch.as_str()
        )),
        other => other,
    })
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

    /// The lifecycle effects: canonical forms, registration reads,
    /// quiet branch tips, work preservation, and the normal Git
    /// protections both doors keep - the branch survives removal and
    /// nothing forces anywhere.
    #[test]
    fn the_lifecycle_effects_keep_gits_own_protections() {
        let root = scratch_repo("lifecycle");
        std::fs::write(root.join("f.txt"), "advanced\n").unwrap();
        git(&root, &["commit", "-qam", "advanced"]).unwrap();
        let head = git(&root, &["rev-parse", "HEAD"]).unwrap();
        let home = root.join("worktrees");
        std::fs::create_dir_all(&home).unwrap();

        // canonical forms: an existing path resolves through symlinks,
        // an absent leaf keeps its tail on the deepest existing ancestor
        let temp_base = std::env::temp_dir().canonicalize().unwrap();
        let absent = temp_base.join("no-such-dir").join("leaf");
        assert_eq!(
            canonical_dir(&absent).unwrap(),
            temp_base.join("no-such-dir").join("leaf")
        );
        let relative = canonical_dir(Path::new("f.txt"));
        assert!(relative.is_ok());

        // a registered worktree reads back with its branch; an absent
        // branch tip reads None quietly, not as a plumbing failure
        let branch = GitBranch::new("swept-line".into()).unwrap();
        let path = home.join("swept-line");
        git(
            &root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                branch.as_str(),
                &path.display().to_string(),
                &head,
            ],
        )
        .unwrap();
        let registrations = registered_worktrees(&root).unwrap();
        let registered = registrations.get(&path.canonicalize().unwrap());
        assert_eq!(registered.cloned().flatten().as_ref(), Some(&branch));
        assert_eq!(
            branch_tip(&root, &branch).unwrap(),
            Some(GitCommit::new(head.clone()).unwrap())
        );
        let missing = GitBranch::new("never-existed".into()).unwrap();
        assert_eq!(branch_tip(&root, &missing).unwrap(), None);

        // a clean checkout preserves nothing; an untracked file does
        assert_eq!(preserved_work(&path).unwrap(), None);
        std::fs::write(path.join("untracked.txt"), "kept\n").unwrap();
        let refusal = preserved_work(&path).unwrap();
        assert!(refusal.is_some(), "untracked work is preservation");
        std::fs::remove_file(path.join("untracked.txt")).unwrap();

        // removal keeps the branch: the ref and its tip stand after
        remove_checkout(&root, &path).unwrap();
        assert!(!path.exists());
        assert!(!registered_worktrees(&root).unwrap().contains_key(&path));
        assert_eq!(
            branch_tip(&root, &branch).unwrap(),
            Some(GitCommit::new(head).unwrap()),
            "the branch survives the checkout removal"
        );

        // re-adding refuses while the branch is checked out elsewhere:
        // git's own protection, never bypassed
        let elsewhere = home.join("holder");
        git(
            &root,
            &[
                "worktree",
                "add",
                "-q",
                &elsewhere.display().to_string(),
                "-b",
                "holder-line",
                "HEAD",
            ],
        )
        .unwrap();
        let holder_branch = GitBranch::new("holder-line".into()).unwrap();
        let refused = add_checkout(&root, &home.join("swept-again"), &holder_branch);
        assert!(
            matches!(refused, Err(WorkspaceFail::Git(_))),
            "a branch checked out elsewhere refuses: {refused:?}"
        );
        assert!(!home.join("swept-again").exists());

        // add succeeds where git allows, at the branch's current tip
        let readded = home.join("swept-again");
        add_checkout(&root, &readded, &branch).unwrap();
        assert_eq!(
            git(&readded, &["rev-parse", "HEAD"]).unwrap(),
            branch_tip(&root, &branch).unwrap().unwrap().as_str()
        );

        std::fs::remove_dir_all(&root).unwrap();
    }
}
