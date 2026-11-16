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
    let out = crate::boards::git_command()
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
    crate::boards::git_command()
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

use crate::objects::workspace::WorkspaceStage;
use crate::store::{RecordId, World};

/// Resolve one caller path to its recorded workspace birth, for every
/// door that names one - clean and the
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
    let out = crate::boards::git_command()
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

// -- the merged rule ------------------------------------------------------
// One function decides mergedness for the workspace doors. No target
// branch is chosen or recorded: a workspace is merged into every local
// branch that contains its branch tip, except the branch itself and the
// recorded (uncleaned) workspaces' branches. The inputs are data - the
// branch, the recorded branches, the local branches - and ancestry is
// the caller's closure, so the rule itself tests without a repo; live
// callers pass git's own answer through `branch_is_ancestor`.

/// Every local branch the workspace is merged into, in the caller's
/// order: the list names the first, and the sweep and clean doors read
/// the rest. A candidate qualifies when the branch's tip is an
/// ancestor of its tip, it is not the branch itself, and it is not a
/// recorded (uncleaned) workspace's branch - excluding the recorded
/// branches keeps a stacked workspace from reading its parent as
/// merged, and a hand-made branch containing the tip (a backup)
/// qualifies, so the row names it and the human sees it.
pub fn merged_branches(
    branch: &GitBranch,
    recorded: &[GitBranch],
    local: &[GitBranch],
    is_ancestor: impl Fn(&GitBranch, &GitBranch) -> Result<bool, WorkspaceFail>,
) -> Result<Vec<GitBranch>, WorkspaceFail> {
    let mut merged = Vec::new();
    for candidate in local {
        if candidate == branch || recorded.contains(candidate) {
            continue;
        }
        if is_ancestor(branch, candidate)? {
            merged.push(candidate.clone());
        }
    }
    Ok(merged)
}

/// git's own ancestry answer between two branches: `merge-base
/// --is-ancestor` through the scrubbed builder. Exit 0 is ancestor,
/// exit 1 is not; any other exit is a plumbing failure, never a
/// verdict.
pub fn branch_is_ancestor(
    root: &Path,
    ancestor: &GitBranch,
    descendant: &GitBranch,
) -> Result<bool, WorkspaceFail> {
    let out = crate::boards::git_command()
        .arg("-C")
        .arg(root)
        .args([
            "merge-base",
            "--is-ancestor",
            ancestor.as_str(),
            descendant.as_str(),
        ])
        .output()
        .map_err(|e| WorkspaceFail::Git(format!("git merge-base --is-ancestor: {e}")))?;
    match out.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(WorkspaceFail::Git(format!(
            "git merge-base --is-ancestor {} {} failed: {}",
            ancestor.as_str(),
            descendant.as_str(),
            String::from_utf8_lossy(&out.stderr).trim()
        ))),
    }
}

// -- the standing list ----------------------------------------------------
// One read joining the fold's recorded workspaces with git's own
// answers, so the list door reports facts, never guesses. A missing
// worktree or a missing branch is a status of its own: the row still
// answers, the numbers it cannot have stay empty. Any other git
// plumbing failure refuses the listing - a face that lies is worse
// than one that declines.

/// The recorded address a workspace is missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingPiece {
    /// The recorded worktree path holds no directory
    Worktree,
    /// Git holds no local branch under the recorded name
    Branch,
}

/// A row's status half: what is missing, the local branches the
/// workspace branch's tip is merged into (the merged rule's full
/// answer, first by name renders), and whether the checkout holds
/// preserved work. A missing worktree cannot be dirty; a missing
/// branch carries no numbers at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowStatus {
    pub missing: Option<MissingPiece>,
    pub merged_into: Vec<GitBranch>,
    pub dirty: bool,
}

/// One standing workspace's row: the fold's recorded facts joined
/// with git's own answers at the repository root.
#[derive(Clone, Debug, PartialEq)]
pub struct ListRow {
    pub birth: RecordId,
    pub base: GitCommit,
    pub branch: GitBranch,
    pub worktree: WorktreePath,
    pub status: RowStatus,
    /// Files, insertions, deletions between the recorded base and the
    /// branch tip; None only when the branch is gone
    pub change: Option<(u64, u64, u64)>,
    /// Commits the branch tip carries beyond the recorded base
    pub ahead: Option<u64>,
    /// The branch tip's last commit time, epoch seconds
    pub touched: Option<u64>,
}

/// The repository's local branches, by name: the merged rule's
/// candidate order, so the first candidate is the first by name.
pub fn local_branches(root: &Path) -> Result<Vec<GitBranch>, WorkspaceFail> {
    let out = git(
        root,
        &["for-each-ref", "refs/heads", "--format=%(refname:short)"],
    )?;
    out.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            GitBranch::new(line.to_string())
                .map_err(|e| WorkspaceFail::Git(format!("git named no branch {line:?}: {e:?}")))
        })
        .collect()
}

/// The shortstat reduction between two commits: files, insertions,
/// deletions - the row's "16f +33 -256". An empty diff is zeros.
pub fn shortstat(
    root: &Path,
    from: &GitCommit,
    to: &GitCommit,
) -> Result<(u64, u64, u64), WorkspaceFail> {
    let range = format!("{}..{}", from.as_str(), to.as_str());
    let out = git(root, &["diff", "--shortstat", &range])?;
    Ok(parse_shortstat(&out))
}

/// The shortstat line's reduction: each comma section names a count
/// and its noun; sections git omits read zero.
fn parse_shortstat(out: &str) -> (u64, u64, u64) {
    let mut figures = (0u64, 0u64, 0u64);
    for section in out.split(',') {
        let section = section.trim();
        let Some(count) = section
            .split_whitespace()
            .next()
            .and_then(|n| n.parse::<u64>().ok())
        else {
            continue;
        };
        if section.contains("file") {
            figures.0 = count;
        } else if section.contains("insertion") {
            figures.1 = count;
        } else if section.contains("deletion") {
            figures.2 = count;
        }
    }
    figures
}

/// Commits the tip carries beyond the base: the row's ahead count.
pub fn commits_ahead(root: &Path, base: &GitCommit, tip: &GitCommit) -> Result<u64, WorkspaceFail> {
    let range = format!("{}..{}", base.as_str(), tip.as_str());
    let out = git(root, &["rev-list", "--count", &range])?;
    out.parse()
        .map_err(|e| WorkspaceFail::Git(format!("git rev-list gave no count: {e:?}")))
}

/// The tip's last commit time, epoch seconds: the age renders from it.
pub fn last_commit_time(root: &Path, tip: &GitCommit) -> Result<u64, WorkspaceFail> {
    let out = git(root, &["show", "-s", "--format=%ct", tip.as_str()])?;
    out.trim()
        .parse()
        .map_err(|e| WorkspaceFail::Git(format!("git show gave no time: {e:?}")))
}

/// Every recorded workspace that is not cleaned, in birth order, each
/// row the fold's facts joined with git's answers at `root`. The
/// merged rule reads every uncleaned recorded branch as its
/// exclusion, so a stacked workspace never reads its parent as
/// merged; a cleaned workspace's retained branch still counts as a
/// candidate, exactly the rule's contract.
pub fn workspace_list(world: &World, root: &Path) -> Result<Vec<ListRow>, WorkspaceFail> {
    let local = local_branches(root)?;
    let standing: Vec<_> = world
        .workspace_births()
        .into_iter()
        .filter_map(|birth| world.workspace_facts(birth))
        .filter(|facts| facts.stage != WorkspaceStage::Cleaned)
        .collect();
    let recorded: Vec<GitBranch> = standing.iter().map(|f| f.branch.clone()).collect();
    let live = |work: &GitBranch, target: &GitBranch| branch_is_ancestor(root, work, target);
    let mut rows = Vec::new();
    for facts in &standing {
        let (status, change, ahead, touched) = match branch_tip(root, &facts.branch)? {
            // no branch, no tip, no numbers: the row still answers
            None => (
                RowStatus {
                    missing: Some(MissingPiece::Branch),
                    merged_into: Vec::new(),
                    dirty: false,
                },
                None,
                None,
                None,
            ),
            Some(tip) => {
                let merged_into = merged_branches(&facts.branch, &recorded, &local, live)?;
                let missing =
                    (!facts.worktree.as_path().is_dir()).then_some(MissingPiece::Worktree);
                let dirty =
                    missing.is_none() && preserved_work(facts.worktree.as_path())?.is_some();
                (
                    RowStatus {
                        missing,
                        merged_into,
                        dirty,
                    },
                    Some(shortstat(root, &facts.base, &tip)?),
                    Some(commits_ahead(root, &facts.base, &tip)?),
                    Some(last_commit_time(root, &tip)?),
                )
            }
        };
        rows.push(ListRow {
            birth: facts.birth,
            base: facts.base.clone(),
            branch: facts.branch.clone(),
            worktree: facts.worktree.clone(),
            status,
            change,
            ahead,
            touched,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::events::Event;
    use crate::store::{Context, Record, Tier};
    use crate::types::actor::ActorName;

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

    /// A scratch repo with a known mainline branch name at the base
    /// commit; the merged-rule tests branch from there.
    fn scratch_with_main(tag: &str) -> PathBuf {
        let root = scratch_repo(tag);
        git(&root, &["checkout", "-qB", "main"]).unwrap();
        root
    }

    /// One branch name, as the rule's inputs carry it.
    fn branch(name: &str) -> GitBranch {
        GitBranch::new(name.to_string()).unwrap()
    }

    /// Branch names as data: the recorded and local lists.
    fn branches(names: &[&str]) -> Vec<GitBranch> {
        names.iter().map(|n| branch(n)).collect()
    }

    /// Commit a file's new content on the checked-out branch; the
    /// tip comes back as the rule's ancestry reads it.
    fn commit(root: &Path, name: &str, content: &str, message: &str) -> GitCommit {
        std::fs::write(root.join(name), content).unwrap();
        git(root, &["add", name]).unwrap();
        git(root, &["commit", "-qm", message]).unwrap();
        GitCommit::new(git(root, &["rev-parse", "HEAD"]).unwrap()).unwrap()
    }

    /// Plain merge: the workspace branch reaches main by a true merge
    /// commit and reads merged into main, named by main alone.
    #[test]
    fn the_merged_rule_reads_a_plain_merge_merged_into_main() {
        let root = scratch_with_main("merged-plain");
        let feature = branch("feature");
        git(&root, &["checkout", "-qb", feature.as_str()]).unwrap();
        commit(&root, "f.txt", "feature\n", "feature work");
        // the human merges with a real merge commit, never a fast-forward
        git(&root, &["checkout", "-q", "main"]).unwrap();
        git(
            &root,
            &[
                "merge",
                "-q",
                "--no-ff",
                "-m",
                "merge the workspace",
                feature.as_str(),
            ],
        )
        .unwrap();

        // the workspace stands recorded and uncleaned, so its own
        // branch rides the recorded set with the rest
        let recorded = branches(&["feature"]);
        let local = branches(&["main", "feature"]);
        let live = |work: &GitBranch, target: &GitBranch| branch_is_ancestor(&root, work, target);
        assert_eq!(
            merged_branches(&feature, &recorded, &local, live).unwrap(),
            vec![branch("main")]
        );

        // a missing branch is a plumbing failure, never a "not merged"
        // verdict
        assert!(matches!(
            branch_is_ancestor(&root, &branch("never-existed"), &branch("main")),
            Err(WorkspaceFail::Git(_))
        ));

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Fast-forward: main moves onto the workspace tip with no merge
    /// commit; a commit is its own ancestor, so the row still reads
    /// merged.
    #[test]
    fn the_merged_rule_reads_a_fast_forward_merged() {
        let root = scratch_with_main("merged-ff");
        let quickfix = branch("quickfix");
        git(&root, &["checkout", "-qb", quickfix.as_str()]).unwrap();
        let tip = commit(&root, "f.txt", "quickfix\n", "quickfix work");
        // the fast-forward: main's tip becomes the workspace tip itself
        git(&root, &["checkout", "-q", "main"]).unwrap();
        git(&root, &["merge", "-q", "--ff-only", quickfix.as_str()]).unwrap();
        assert_eq!(
            GitCommit::new(git(&root, &["rev-parse", "main"]).unwrap()).unwrap(),
            tip,
            "the fast-forward really landed main on the workspace tip"
        );

        let recorded = branches(&["quickfix"]);
        let local = branches(&["main", "quickfix"]);
        let live = |work: &GitBranch, target: &GitBranch| branch_is_ancestor(&root, work, target);
        assert_eq!(
            merged_branches(&quickfix, &recorded, &local, live).unwrap(),
            vec![branch("main")]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A stack of three: the child workspace sits on the parent
    /// workspace's branch and the parent takes the child's work in, so
    /// git ancestry holds towards the parent's branch - yet the child
    /// must NOT read merged there, because the parent's branch is a
    /// recorded workspace's branch. It reads merged into main only
    /// once the parent's branch itself reaches main.
    #[test]
    fn the_merged_rule_keeps_a_stacked_child_unmerged_outside_its_parent() {
        let root = scratch_with_main("merged-stack");
        // the parent workspace: one commit on main
        git(&root, &["checkout", "-qb", "stack-parent"]).unwrap();
        commit(&root, "parent.txt", "parent\n", "parent work");
        // the child workspace: stacked on the parent, one more commit
        git(&root, &["checkout", "-qb", "stack-child"]).unwrap();
        commit(&root, "child.txt", "child\n", "child work");
        // the parent workspace takes the child's work in: its branch
        // tip now contains the child's tip, so plain ancestry would
        // read merged
        git(&root, &["checkout", "-q", "stack-parent"]).unwrap();
        git(&root, &["merge", "-q", "--ff-only", "stack-child"]).unwrap();

        // both workspaces stand recorded and uncleaned
        let recorded = branches(&["stack-parent", "stack-child"]);
        let local = branches(&["main", "stack-parent", "stack-child"]);
        let child = branch("stack-child");
        let live = |work: &GitBranch, target: &GitBranch| branch_is_ancestor(&root, work, target);
        let merged = merged_branches(&child, &recorded, &local, live).unwrap();
        assert!(
            merged.is_empty(),
            "the child reads unmerged, never merged into its parent's branch: {merged:?}"
        );

        // once the parent's branch reaches main, the child reads
        // merged into main - and still never into the parent's branch
        git(&root, &["checkout", "-q", "main"]).unwrap();
        git(&root, &["merge", "-q", "--ff-only", "stack-parent"]).unwrap();
        assert_eq!(
            merged_branches(&child, &recorded, &local, live).unwrap(),
            vec![branch("main")]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A detached rewrite branch: the repository's work integrates
    /// into a rewrite branch, not main, and a workspace merged there
    /// reads merged into rewrite - no branch is special-cased by name.
    #[test]
    fn the_merged_rule_names_a_detached_rewrite_branch() {
        let root = scratch_with_main("merged-rewrite");
        // the rewrite line moves ahead of main
        git(&root, &["checkout", "-qb", "rewrite"]).unwrap();
        commit(&root, "rewrite.txt", "rewrite\n", "rewrite base");
        // a workspace cut from the rewrite line
        git(&root, &["checkout", "-qb", "rewrite-lexer"]).unwrap();
        commit(&root, "lexer.txt", "lexer\n", "lexer work");
        // the rewrite branch takes the workspace in; main never does
        git(&root, &["checkout", "-q", "rewrite"]).unwrap();
        git(&root, &["merge", "-q", "--ff-only", "rewrite-lexer"]).unwrap();

        let lexer = branch("rewrite-lexer");
        let recorded = branches(&["rewrite-lexer"]);
        let local = branches(&["main", "rewrite", "rewrite-lexer"]);
        let live = |work: &GitBranch, target: &GitBranch| branch_is_ancestor(&root, work, target);
        assert_eq!(
            merged_branches(&lexer, &recorded, &local, live).unwrap(),
            vec![branch("rewrite")]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A squash merge: main gains the same content as one new commit
    /// with no ancestry behind it, so the workspace reads UNMERGED -
    /// the honest answer git itself gives.
    #[test]
    fn the_merged_rule_reads_a_squash_merge_unmerged() {
        let root = scratch_with_main("merged-squash");
        let squashed = branch("squashed");
        git(&root, &["checkout", "-qb", squashed.as_str()]).unwrap();
        commit(&root, "f.txt", "squash\n", "squash work");
        // the human squashes into main: same content, no shared ancestry
        git(&root, &["checkout", "-q", "main"]).unwrap();
        git(&root, &["merge", "--squash", squashed.as_str()]).unwrap();
        git(&root, &["commit", "-qm", "squash the workspace"]).unwrap();

        let recorded = branches(&["squashed"]);
        let local = branches(&["main", "squashed"]);
        let live = |work: &GitBranch, target: &GitBranch| branch_is_ancestor(&root, work, target);
        let merged = merged_branches(&squashed, &recorded, &local, live).unwrap();
        assert!(merged.is_empty(), "no ancestry, no verdict: {merged:?}");

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A hand-made backup branch pointing at the workspace tip
    /// contains the tip, so the workspace reads merged into it and the
    /// row names it for the human to see.
    #[test]
    fn the_merged_rule_names_a_hand_made_backup_branch() {
        let root = scratch_with_main("merged-backup");
        let risky = branch("risky");
        git(&root, &["checkout", "-qb", risky.as_str()]).unwrap();
        commit(&root, "f.txt", "work\n", "risky work");
        // a hand-made backup branch lands exactly on the workspace tip
        git(&root, &["branch", "backup-before-clean", risky.as_str()]).unwrap();

        let recorded = branches(&["risky"]);
        let local = branches(&["main", "risky", "backup-before-clean"]);
        let live = |work: &GitBranch, target: &GitBranch| branch_is_ancestor(&root, work, target);
        assert_eq!(
            merged_branches(&risky, &recorded, &local, live).unwrap(),
            vec![branch("backup-before-clean")]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The rule over an in-memory ancestry relation, no git at all:
    /// every qualifying candidate comes back in the caller's order,
    /// the branch itself and the recorded workspaces' branches drop
    /// out even when ancestry would hold, and a non-ancestor never
    /// qualifies.
    #[test]
    fn the_merged_rule_returns_every_candidate_in_order_over_a_fake() {
        let work = branch("ws");
        let recorded = branches(&["ws", "older-ws"]);
        let local = branches(&["main", "older-ws", "backup", "ws", "later-ws", "unrelated"]);
        // ancestry as data: ws is contained in five of the six
        // candidates, itself and older-ws included
        let contains = [
            ("ws", "main"),
            ("ws", "backup"),
            ("ws", "later-ws"),
            ("ws", "older-ws"),
            ("ws", "ws"),
        ];
        let fake = |work: &GitBranch, target: &GitBranch| {
            Ok(contains
                .iter()
                .any(|(w, t)| *w == work.as_str() && *t == target.as_str()))
        };
        assert_eq!(
            merged_branches(&work, &recorded, &local, fake).unwrap(),
            branches(&["main", "backup", "later-ws"])
        );
    }

    /// The shortstat reduction: each noun's count, sections git omits
    /// reading zero, the empty diff zeros.
    #[test]
    fn the_shortstat_reduction_reads_each_section_git_names() {
        assert_eq!(parse_shortstat(""), (0, 0, 0));
        assert_eq!(
            parse_shortstat(" 1 file changed, 1 insertion(+)"),
            (1, 1, 0)
        );
        assert_eq!(
            parse_shortstat(" 2 files changed, 3 insertions(+)"),
            (2, 3, 0)
        );
        assert_eq!(parse_shortstat(" 1 file changed, 1 deletion(-)"), (1, 0, 1));
        assert_eq!(
            parse_shortstat(" 16 files changed, 33 insertions(+), 256 deletions(-)"),
            (16, 33, 256)
        );
    }

    /// One workspace birth record for the list's world: the facts the
    /// creation door would have recorded.
    fn birth_record(seq: usize, base: &str, branch: &str, worktree: &Path) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: seq as u64,
            context: Context {
                actor: ActorName::new("pi".into()).unwrap(),
                tier: Tier::Agent,
                scope: None,
            },
            event: Event::WorkspaceCreated {
                base: GitCommit::new(base.to_string()).unwrap(),
                branch: GitBranch::new(branch.to_string()).unwrap(),
                worktree: WorktreePath::new(worktree.to_path_buf()).unwrap(),
            },
        }
    }

    /// The list join over a live repo: six recorded workspaces stand
    /// in every state the sketch names - merged (its worktree since
    /// removed), unmerged dirty, stacked on a recorded branch, fresh
    /// with no commits, branch deleted - and one cleaned workspace is
    /// omitted. Missing pieces are statuses, never crashes; the
    /// numbers only die with the branch.
    #[test]
    fn the_list_joins_the_fold_with_gits_answers_and_names_its_missing_pieces() {
        let root = scratch_with_main("list");
        let base = GitCommit::new(git(&root, &["rev-parse", "HEAD"]).unwrap()).unwrap();
        let home = root.join("worktrees");
        std::fs::create_dir_all(&home).unwrap();
        let cut = |branch: &str, name: &str, at: &str| {
            git(
                &root,
                &[
                    "worktree",
                    "add",
                    "-q",
                    "-b",
                    branch,
                    &home.join(name).display().to_string(),
                    at,
                ],
            )
            .unwrap();
            home.join(name)
        };

        // every workspace cuts at the base before any merge moves main:
        // the merged workspace merges into main; its worktree goes
        // away afterwards - the branch keeps every number alive
        let merged_wt = cut("list-merged", "list-merged", "HEAD");
        commit(&merged_wt, "a.txt", "one line\n", "merged work");

        // the dirty workspace: one commit, plus untracked work
        let dirty_wt = cut("list-dirty", "list-dirty", "HEAD");
        commit(&dirty_wt, "b.txt", "one\ntwo\n", "dirty work");
        std::fs::write(dirty_wt.join("untracked.txt"), "kept\n").unwrap();

        // the stacked workspace sits on the dirty workspace's branch
        let stacked_wt = cut("list-stacked", "list-stacked", "list-dirty");
        commit(&stacked_wt, "c.txt", "one\ntwo\nthree\n", "stacked work");

        // fresh: no commits, clean checkout
        cut("list-fresh", "list-fresh", "HEAD");

        // the lost branch: its checkout detaches, then the branch goes
        let lost_wt = cut("list-lost", "list-lost", "HEAD");
        git(&lost_wt, &["checkout", "-q", "--detach"]).unwrap();
        git(&root, &["branch", "-D", "list-lost"]).unwrap();

        // the cleaned workspace's branch stands, the fold moved on
        cut("list-swept", "list-swept", "HEAD");
        git(&root, &["checkout", "-q", "main"]).unwrap();
        git(&root, &["merge", "-q", "--ff-only", "list-merged"]).unwrap();
        std::fs::remove_dir_all(&merged_wt).unwrap();
        let dirty_base = GitCommit::new(git(&root, &["rev-parse", "list-dirty"]).unwrap()).unwrap();
        let mut world = World::new();
        let story = [
            birth_record(1, base.as_str(), "list-merged", &merged_wt),
            birth_record(2, base.as_str(), "list-dirty", &dirty_wt),
            birth_record(3, dirty_base.as_str(), "list-stacked", &stacked_wt),
            birth_record(4, base.as_str(), "list-fresh", &home.join("list-fresh")),
            birth_record(5, base.as_str(), "list-lost", &lost_wt),
            birth_record(6, base.as_str(), "list-swept", &home.join("list-swept")),
        ];
        for record in story {
            world = world.apply(record).unwrap();
        }
        world = world
            .apply(Record {
                id: RecordId(7),
                timestamp: 7,
                context: Context {
                    actor: ActorName::new("pi".into()).unwrap(),
                    tier: Tier::Agent,
                    scope: None,
                },
                event: Event::WorkspaceCleaned {
                    workspace: RecordId(6),
                },
            })
            .unwrap();

        // the candidate order is the branch-name order the doors read
        assert_eq!(
            local_branches(&root).unwrap(),
            branches(&[
                "list-dirty",
                "list-fresh",
                "list-merged",
                "list-stacked",
                "list-swept",
                "main"
            ])
        );

        let rows = workspace_list(&world, &root).unwrap();
        assert_eq!(
            rows.iter().map(|r| r.birth.0).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5],
            "birth order, the cleaned workspace omitted"
        );

        // merged, worktree gone: the status of its own rides beside a
        // full row - the branch still answers every number
        assert_eq!(
            rows[0].status,
            RowStatus {
                missing: Some(MissingPiece::Worktree),
                merged_into: branches(&["main"]),
                dirty: false,
            }
        );
        assert_eq!(rows[0].change, Some((1, 1, 0)));
        assert_eq!(rows[0].ahead, Some(1));
        assert!(rows[0].touched.is_some());

        // unmerged and dirty: the untracked work reads dirty, no
        // candidate contains the tip
        assert_eq!(
            rows[1].status,
            RowStatus {
                missing: None,
                merged_into: Vec::new(),
                dirty: true,
            }
        );
        assert_eq!(rows[1].change, Some((1, 2, 0)));
        assert_eq!(rows[1].ahead, Some(1));

        // the stack stays unmerged: its parent's recorded branch is
        // excluded even though it contains the tip
        assert_eq!(
            rows[2].status,
            RowStatus {
                missing: None,
                merged_into: Vec::new(),
                dirty: false,
            }
        );
        assert_eq!(rows[2].change, Some((1, 3, 0)));
        assert_eq!(rows[2].ahead, Some(1));

        // fresh: the zero diff renders as facts, ahead zero; its tip
        // is contained by main and by the cleaned workspace's retained
        // branch - cleaned branches are not excluded, so both count,
        // and the first by name renders
        assert_eq!(rows[3].change, Some((0, 0, 0)));
        assert_eq!(rows[3].ahead, Some(0));
        assert_eq!(
            rows[3].status,
            RowStatus {
                missing: None,
                merged_into: branches(&["list-swept", "main"]),
                dirty: false,
            }
        );

        // the lost branch: a status of its own, no numbers at all
        assert_eq!(
            rows[4].status,
            RowStatus {
                missing: Some(MissingPiece::Branch),
                merged_into: Vec::new(),
                dirty: false,
            }
        );
        assert_eq!(rows[4].change, None);
        assert_eq!(rows[4].ahead, None);
        assert_eq!(rows[4].touched, None);

        std::fs::remove_dir_all(&root).unwrap();
    }
}
