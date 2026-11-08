//! The board store's Git effects: one bare repository per tracker holds
//! task board revisions, actor checkpoints, and the materialized current
//! checkout. Git objects are written before anything names them (D7): a
//! commit lands as objects first, the event records it, and only then a
//! ref moves, so refs and the checkout are derived caches the log can
//! repair from scratch. These functions never touch the record and never
//! read the fold; the log's view is handed in where repair needs it.
//!
//! A working directory is board.yaml plus its referenced figure files.
//! The committed tree holds exactly those files and nothing else, so the
//! store-kept bookkeeping beside board.yaml never enters Git: the base
//! commit a working directory started from is recorded in
//! .saccade-board.json beside board.yaml ({"base": "<commit>"}), written
//! by `record_base`, read by `recorded_base` and `update`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use board_grammar::{Board, Component, SelectPath, parse};
use board_grammar::{ErrorKind, ParseError, Where as GrammarWhere};

use crate::objects::task::TaskId as FoldTaskId;
use crate::store::World;

/// The largest figure a commit or checkpoint accepts: 8 MiB.
pub const MAX_FIGURE_BYTES: u64 = 8 * 1024 * 1024;

/// The working-directory bookkeeping file beside board.yaml, outside the
/// committed tree.
pub const BASE_FILE: &str = ".saccade-board.json";

pub const COMMIT_REF_PREFIX: &str = "refs/saccade/tasks";
const BOARD_FILE: &str = "board.yaml";
const BOARD_AUTHOR_NAME: &str = "saccade boards";
const BOARD_AUTHOR_EMAIL: &str = "boards@saccade.invalid";

/// An effect-layer refusal: operation codes, never fold refusals.
#[derive(Debug)]
pub enum BoardsFail {
    /// The board.yaml in the working directory does not parse.
    BoardInvalid(board_grammar::ParseError),
    /// A referenced figure file is absent from the working directory.
    MissingFigure { path: String },
    /// A referenced figure file exceeds the size bound.
    FigureTooLarge { path: String, size: u64 },
    /// A commit the effect needed does not exist in the repository.
    MissingCommit { task: String, commit: String },
    /// Repair found logged commits the repository does not hold.
    MissingCommits { missing: Vec<(String, String)> },
    /// The working directory records no base commit.
    MissingBase { workdir: PathBuf },
    /// The working directory's base record does not parse.
    BadBase { workdir: PathBuf, detail: String },
    /// A task key or actor is not a single safe ref path component.
    InvalidKey { key: String },
    /// The working directory's board.yaml is unreadable.
    WorkingBoard { path: PathBuf, detail: String },
    /// An incorporation names a source task the fold does not hold.
    UnknownSourceTask { name: String },
    /// An incorporation names a revision number its source task never
    /// published.
    UnknownSourceRevision { name: String, revision: usize },
    /// The board's task field does not name the committing task.
    TaskMismatch {
        board_task: String,
        expected: String,
    },
    /// The store refused the TaskCommitted the publish attempted.
    Record(String),
    /// An acceptance was attempted while checkpoints stood unsaved.
    UnsavedCheckpoints { actors: Vec<String> },
    /// A git command failed; the detail is its stderr.
    Git(String),
    /// A filesystem effect failed.
    Io { detail: String },
}

impl BoardsFail {
    pub fn code(&self) -> &'static str {
        match self {
            BoardsFail::BoardInvalid(_) => "boards_board_invalid",
            BoardsFail::MissingFigure { .. } => "boards_figure_missing",
            BoardsFail::FigureTooLarge { .. } => "boards_figure_too_large",
            BoardsFail::MissingCommit { .. } => "boards_missing_commit",
            BoardsFail::MissingCommits { .. } => "boards_missing_commit",
            BoardsFail::MissingBase { .. } => "boards_base_unrecorded",
            BoardsFail::BadBase { .. } => "boards_base_unreadable",
            BoardsFail::InvalidKey { .. } => "boards_invalid_key",
            BoardsFail::UnknownSourceTask { .. } => "boards_unknown_source_task",
            BoardsFail::UnknownSourceRevision { .. } => "boards_unknown_source_revision",
            BoardsFail::TaskMismatch { .. } => "boards_task_mismatch",
            BoardsFail::Record(_) => "boards_record_refused",
            BoardsFail::UnsavedCheckpoints { .. } => "boards_unsaved_checkpoints",
            BoardsFail::WorkingBoard { .. } => "boards_working_board_unreadable",
            BoardsFail::Git(_) => "boards_effect",
            BoardsFail::Io { .. } => "boards_io",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            BoardsFail::BoardInvalid(error) => error.to_string(),
            BoardsFail::MissingFigure { path } => format!(
                "the figure {path} is missing from the working directory; place the file there or drop the figure"
            ),
            BoardsFail::FigureTooLarge { path, size } => format!(
                "the figure {path} holds {size} bytes; figures stay at most {MAX_FIGURE_BYTES} bytes - shrink it or link an artifact instead"
            ),
            BoardsFail::MissingCommit { task, commit } => format!(
                "the commit {commit} that task {task} needs is not in the boards repository"
            ),
            BoardsFail::MissingCommits { missing } => {
                let listed = missing
                    .iter()
                    .map(|(task, commit)| format!("{task} -> {commit}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "logged commits the repository does not hold: {listed}; the boards repository refuses to serve until the record and the objects agree"
                )
            }
            BoardsFail::MissingBase { workdir } => format!(
                "the working directory {} records no base commit; it cannot be merged without knowing the commit it started from",
                workdir.display()
            ),
            BoardsFail::BadBase { workdir, detail } => format!(
                "the base record in {} does not read as a base commit: {detail}",
                workdir.display()
            ),
            BoardsFail::InvalidKey { key } => {
                format!(
                    "'{key}' is not a safe ref path component; task keys and actors are single names without slashes"
                )
            }
            BoardsFail::UnknownSourceTask { name } => format!(
                "the incorporation cites '{name}', which no task in this tracker bears; incorporations cite tasks as t-N, and the task must exist"
            ),
            BoardsFail::UnknownSourceRevision { name, revision } => format!(
                "the incorporation cites {name} revision r{revision}, which that task never published; cite one of its published revisions by number"
            ),
            BoardsFail::TaskMismatch {
                board_task,
                expected,
            } => format!(
                "the board names {board_task}, but this account belongs to {expected}; the board's task field must carry the committing task"
            ),
            BoardsFail::Record(detail) => detail.clone(),
            BoardsFail::UnsavedCheckpoints { actors } => format!(
                "checkpoints by {} stand unsaved for this task; discard them or commit first - acceptance certifies the recorded revisions alone",
                actors.join(", ")
            ),
            BoardsFail::WorkingBoard { path, detail } => {
                format!("could not read {}: {detail}", path.display())
            }
            BoardsFail::Git(detail) => detail.clone(),
            BoardsFail::Io { detail } => detail.clone(),
        }
    }
}

/// The boards repository: the bare repo and the materialized checkout root.
pub struct Boards {
    repo: PathBuf,
    checkouts: PathBuf,
}

/// What a validated commit produced: the object's hash and the declared
/// incorporation references, ready for the event that will name the commit.
#[derive(Clone, Debug, PartialEq)]
pub struct Committed {
    pub commit: String,
    pub incorporations: Vec<(String, usize)>,
}

/// What a working-directory update did: whether board.yaml merged with
/// conflict markers left in place, and which figures both sides changed.
#[derive(Clone, Debug, PartialEq)]
pub struct UpdateOutcome {
    pub board_conflict: bool,
    pub figure_conflicts: Vec<String>,
}

impl Boards {
    /// The derived checkout path for a task's current committed board.
    pub fn materialized_path(&self, task: &str) -> PathBuf {
        self.checkouts.join(task)
    }

    /// Open the tracker's boards repository, initializing it if absent.
    /// Auto-gc is disabled: recorded commits are never garbage-collected.
    pub fn open(state_dir: &Path) -> Result<Boards, BoardsFail> {
        let repo = state_dir.join("boards.git");
        if !repo.join("HEAD").exists() {
            std::fs::create_dir_all(&repo).map_err(|error| BoardsFail::Io {
                detail: format!("{}: {error}", repo.display()),
            })?;
            run_ok(&repo, &["init", "--bare", "--quiet", "."], None)?;
            run_ok(&repo, &["config", "gc.auto", "0"], None)?;
            // the object store carries its own author: commits must not
            // depend on the ambient machine identity
            run_ok(&repo, &["config", "user.name", BOARD_AUTHOR_NAME], None)?;
            run_ok(&repo, &["config", "user.email", BOARD_AUTHOR_EMAIL], None)?;
        }
        Ok(Boards {
            repo,
            checkouts: state_dir.join("boards").join("current"),
        })
    }

    /// Validate and commit the working directory's board as objects,
    /// naming nothing: the caller records the event between this and
    /// `advance_commit`. The parent is the commit the caller believes
    /// current; the store refuses a parent the repository does not hold.
    pub fn commit(
        &self,
        task: &str,
        workdir: &Path,
        parent: Option<&str>,
        message: &str,
    ) -> Result<Committed, BoardsFail> {
        single_component(task)?;
        let board = Self::validate_workdir(workdir)?;
        if let Some(parent) = parent
            && !self.commit_exists(parent)
        {
            return Err(BoardsFail::MissingCommit {
                task: task.to_string(),
                commit: parent.to_string(),
            });
        }
        let commit = self.commit_tree(workdir, &board, parent, message)?;
        Ok(Committed {
            commit,
            incorporations: board_grammar::incorporations(&board),
        })
    }

    /// The local validation shared by `board check` and commit.
    pub fn validate_workdir(workdir: &Path) -> Result<Board, BoardsFail> {
        let board = read_working_board(workdir)?;
        for path in board_grammar::figures(&board) {
            let file = workdir.join(&path);
            let meta = std::fs::metadata(&file)
                .map_err(|_| BoardsFail::MissingFigure { path: path.clone() })?;
            let size = meta.len();
            if size > MAX_FIGURE_BYTES {
                return Err(BoardsFail::FigureTooLarge { path, size });
            }
        }
        Ok(board)
    }

    /// Move the task's commit ref onto a commit the repository holds.
    /// The caller does this only after the naming event landed (D7).
    pub fn advance_commit(&self, task: &str, commit: &str) -> Result<(), BoardsFail> {
        single_component(task)?;
        if !self.commit_exists(commit) {
            return Err(BoardsFail::MissingCommit {
                task: task.to_string(),
                commit: commit.to_string(),
            });
        }
        run_ok(
            &self.repo,
            &[
                "update-ref",
                &format!("{COMMIT_REF_PREFIX}/{task}/commit"),
                commit,
            ],
            None,
        )?;
        Ok(())
    }

    /// Commit the working directory onto the actor's checkpoint chain
    /// without validation - a checkpoint is a snapshot mid-edit, never
    /// a published revision, and it never touches the commit ref.
    /// The chain's first commit hangs off `base`, the commit the actor
    /// started from; later checkpoints chain onto the actor's tip.
    pub fn checkpoint(
        &self,
        task: &str,
        actor: &str,
        workdir: &Path,
        base: &str,
        message: &str,
    ) -> Result<String, BoardsFail> {
        single_component(task)?;
        single_component(actor)?;
        if !self.commit_exists(base) {
            return Err(BoardsFail::MissingCommit {
                task: task.to_string(),
                commit: base.to_string(),
            });
        }
        let parent = match self.read_ref(&format!("{COMMIT_REF_PREFIX}/{task}/wip/{actor}"))? {
            Some(tip) => tip,
            None => base.to_string(),
        };
        // a checkpoint validates nothing: the tree carries board.yaml and
        // whatever referenced figures currently exist on disk
        let board_path = workdir.join(BOARD_FILE);
        let board_text =
            std::fs::read_to_string(&board_path).map_err(|error| BoardsFail::WorkingBoard {
                path: board_path,
                detail: error.to_string(),
            })?;
        let board = match parse(&board_text) {
            Ok(board) => board,
            Err(_) => Board {
                task: String::new(),
                content: Vec::new(),
            },
        };
        let mut files = vec![(
            BOARD_FILE.to_string(),
            self.write_blob(board_text.as_bytes())?,
        )];
        for path in board_grammar::figures(&board) {
            if let Ok(bytes) = std::fs::read(workdir.join(&path)) {
                files.push((path, self.write_blob(&bytes)?));
            }
        }
        let tree = self.write_tree(&files)?;
        let args: Vec<&str> = vec!["commit-tree", &tree, "-p", &parent, "-m", message];
        let commit = run_ok(&self.repo, &args, None)?;
        run_ok(
            &self.repo,
            &[
                "update-ref",
                &format!("{COMMIT_REF_PREFIX}/{task}/wip/{actor}"),
                commit.trim(),
            ],
            None,
        )?;
        Ok(commit.trim().to_string())
    }

    /// Delete the actor's checkpoint chain reference; the chain's
    /// objects stay in the repository, retained.
    pub fn discard_checkpoint(&self, task: &str, actor: &str) -> Result<(), BoardsFail> {
        single_component(task)?;
        single_component(actor)?;
        run_ok(
            &self.repo,
            &[
                "update-ref",
                "-d",
                &format!("{COMMIT_REF_PREFIX}/{task}/wip/{actor}"),
            ],
            None,
        )?;
        Ok(())
    }

    /// The actors whose checkpoint chains currently hold unsaved work.
    pub fn unsaved_checkpoints(&self, task: &str) -> Result<Vec<String>, BoardsFail> {
        single_component(task)?;
        let prefix = format!("{COMMIT_REF_PREFIX}/{task}/wip/");
        let refs = run_ok(
            &self.repo,
            &["for-each-ref", "--format=%(refname)", &format!("{prefix}*")],
            None,
        )?;
        Ok(refs
            .lines()
            .filter_map(|line| line.strip_prefix(&prefix))
            .map(str::to_string)
            .collect())
    }

    /// Whether the repository holds the commit object.
    pub fn commit_exists(&self, commit: &str) -> bool {
        matches!(
            run(&self.repo, &["cat-file", "-e", &format!("{commit}^{{commit}}")], None),
            Ok(outcome) if outcome.status == 0
        )
    }

    /// Parse board.yaml out of a stored revision.
    pub fn read_board_at(&self, commit: &str) -> Result<Board, BoardsFail> {
        let bytes = self
            .blob_at(commit, BOARD_FILE)?
            .ok_or_else(|| BoardsFail::MissingCommit {
                task: String::new(),
                commit: commit.to_string(),
            })?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        parse(&text).map_err(BoardsFail::BoardInvalid)
    }

    /// Write a stored revision's files into `dest`, creating it.
    pub fn export_tree(&self, commit: &str, dest: &Path) -> Result<(), BoardsFail> {
        let listing = run_ok(&self.repo, &["ls-tree", "-r", "-z", commit], None)?;
        std::fs::create_dir_all(dest).map_err(|error| BoardsFail::Io {
            detail: format!("{}: {error}", dest.display()),
        })?;
        for entry in listing.split('\0').filter(|entry| !entry.is_empty()) {
            // "<mode> <type> <hash>\t<path>"
            let Some((head, path)) = entry.split_once('\t') else {
                continue;
            };
            let mut parts = head.split(' ');
            let mode = parts.next().unwrap_or("");
            let _type = parts.next().unwrap_or("");
            let Some(hash) = parts.next() else {
                continue;
            };
            let out = run(&self.repo, &["cat-file", "blob", hash], None)?;
            if out.status != 0 {
                return Err(BoardsFail::Git(format!(
                    "git cat-file blob failed: {}",
                    out.stderr.trim()
                )));
            }
            let bytes = out.stdout;
            let file = dest.join(path);
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent).map_err(|error| BoardsFail::Io {
                    detail: format!("{}: {error}", parent.display()),
                })?;
            }
            std::fs::write(&file, &bytes).map_err(|error| BoardsFail::Io {
                detail: format!("{}: {error}", file.display()),
            })?;
            if mode == "100755" {
                set_executable(&file)?;
            }
        }
        Ok(())
    }

    /// Rewrite the task's materialized checkout from a commit: export
    /// into a per-commit directory, then swap a symlink at the task's
    /// path, so the checkout path resolves to a complete tree at every
    /// instant - the old tree survives until the swap has landed.
    pub fn materialize(&self, task: &str, commit: &str) -> Result<(), BoardsFail> {
        single_component(task)?;
        if !self.commit_exists(commit) {
            return Err(BoardsFail::MissingCommit {
                task: task.to_string(),
                commit: commit.to_string(),
            });
        }
        std::fs::create_dir_all(&self.checkouts).map_err(|error| BoardsFail::Io {
            detail: format!("{}: {error}", self.checkouts.display()),
        })?;
        let tree_dir = self.checkouts.join(format!(".{task}-{commit}"));
        let _ = std::fs::remove_dir_all(&tree_dir);
        self.export_tree(commit, &tree_dir)?;
        let final_path = self.checkouts.join(task);
        // a leftover real checkout from before the symlink swap is not
        // replaceable by rename; drop it once so the swap is atomic from
        // here on
        let previous = std::fs::symlink_metadata(&final_path)
            .ok()
            .filter(|meta| !meta.is_symlink());
        if previous.is_some() {
            std::fs::remove_dir_all(&final_path).map_err(|error| BoardsFail::Io {
                detail: format!("{}: {error}", final_path.display()),
            })?;
        }
        let old_tree = std::fs::read_link(&final_path).ok();
        let swap = self
            .checkouts
            .join(format!(".swap-{task}-{}", std::process::id()));
        let _ = std::fs::remove_file(&swap);
        symlink(&format!(".{task}-{commit}"), &swap)?;
        std::fs::rename(&swap, &final_path).map_err(|error| BoardsFail::Io {
            detail: format!("{} to {}: {error}", swap.display(), final_path.display()),
        })?;
        // the tree the symlink pointed at survives until the swap landed
        if let Some(old) = old_tree {
            let _ = std::fs::remove_dir_all(self.checkouts.join(old));
        }
        Ok(())
    }

    /// The D7 publish sequence for a board commit: the objects land
    /// first, every incorporation selector resolves against its cited
    /// source revision, the caller's `record` closure lands the
    /// TaskCommitted (its guards run under the write lock), and only
    /// then do the ref and the checkout move. A refusal at any stage
    /// leaves the caches untouched; orphaned objects are the
    /// crash-window cost the repair law already absorbs.
    ///
    /// The world resolves the incorporation source names to fold tasks
    /// and hands the current commit as the parent; the store re-checks
    /// both under its locks.
    pub fn publish_board(
        &self,
        task: FoldTaskId,
        ref_key: &str,
        world: &World,
        workdir: &Path,
        message: &str,
        mut record: impl FnMut(crate::events::Command) -> Result<(), String>,
    ) -> Result<String, BoardsFail> {
        let board = read_working_board(workdir)?;
        let expected = format!("t-{}", task.0);
        if board.task != expected {
            return Err(BoardsFail::TaskMismatch {
                board_task: board.task.clone(),
                expected,
            });
        }
        let declared = incorporation_sources(&board);
        let parent = world
            .tasks
            .get(task.0)
            .and_then(|ctx| ctx.commits.last().map(|c| c.commit.clone()));
        // citations arrive as (t-N, K): the door resolves K to that
        // task's recorded commit, and an unknown K refuses before the
        // record is attempted
        let mut resolved = Vec::with_capacity(declared.len());
        for (name, revision, selects) in &declared {
            let position = name
                .strip_prefix("t-")
                .and_then(|digits| digits.parse::<usize>().ok())
                .filter(|n| world.tasks.get(*n).is_some())
                .ok_or_else(|| BoardsFail::UnknownSourceTask { name: name.clone() })?;
            let source = FoldTaskId(position);
            let cited = world
                .tasks
                .get(position)
                .and_then(|ctx| ctx.commits.get(revision.checked_sub(1)?))
                .map(|c| c.commit.clone())
                .ok_or_else(|| BoardsFail::UnknownSourceRevision {
                    name: name.clone(),
                    revision: *revision,
                })?;
            resolved.push((source, cited, selects));
        }
        let landed = self.commit(
            ref_key,
            workdir,
            parent.as_ref().map(|c| c.as_str()),
            message,
        )?;
        // selector verification against each cited revision
        for (source, cited, selects) in resolved.iter() {
            let name = format!("t-{}", source.0);
            let cited_str = cited.as_str();
            let source_board = match self.read_board_at(cited_str) {
                Ok(board) => board,
                Err(error) => {
                    return Err(BoardsFail::BoardInvalid(selector_refusal(
                        &name,
                        cited_str,
                        "",
                        &ParseError {
                            kind: ErrorKind::NoReceivingText,
                            at: GrammarWhere::Path(cited_str.to_string()),
                            message: error.detail(),
                        },
                    )));
                }
            };
            for select in selects.iter() {
                if let Err(error) = board_grammar::select(&source_board, select) {
                    return Err(BoardsFail::BoardInvalid(selector_refusal(
                        &name,
                        cited_str,
                        &select.to_string(),
                        &ParseError {
                            kind: ErrorKind::BadSelectPath,
                            at: GrammarWhere::Path(select.to_string()),
                            message: format!("{error}"),
                        },
                    )));
                }
            }
        }
        let mut pairs = Vec::with_capacity(resolved.len());
        for (source, cited, _) in resolved.iter() {
            let cited_str = cited.as_str();
            let cited =
                crate::types::pointers::GitCommit::new(cited_str.to_string()).map_err(|error| {
                    BoardsFail::Git(format!("the cited revision {cited_str}: {error:?}"))
                })?;
            pairs.push((*source, cited));
        }
        let commit = crate::types::pointers::GitCommit::new(landed.commit.clone())
            .map_err(|error| BoardsFail::Git(format!("the landed commit: {error:?}")))?;
        record(crate::events::Command::CommitBoard {
            task,
            commit: commit.clone(),
            parent: parent.clone(),
            incorporations: pairs,
        })
        .map_err(BoardsFail::Record)?;
        self.advance_commit(ref_key, &landed.commit)?;
        self.materialize(ref_key, &landed.commit)?;
        Ok(landed.commit)
    }

    /// The accept door's effect: no actor of the task may hold
    /// unsaved checkpoints when the acceptance records.
    pub fn accept_guard(&self, task: &str) -> Result<(), BoardsFail> {
        let actors = self.unsaved_checkpoints(task)?;
        if actors.is_empty() {
            Ok(())
        } else {
            Err(BoardsFail::UnsavedCheckpoints { actors })
        }
    }

    /// Set every commit ref and materialized checkout to what the log
    /// says is current. Refuses the whole repair, naming each offending
    /// pair, if any logged commit is missing from the repository.
    pub fn repair(&self, current: &[(String, String)]) -> Result<(), BoardsFail> {
        let mut missing = Vec::new();
        for (task, commit) in current {
            if !self.commit_exists(commit) {
                missing.push((task.clone(), commit.clone()));
            }
        }
        if !missing.is_empty() {
            return Err(BoardsFail::MissingCommits { missing });
        }
        for (task, commit) in current {
            self.advance_commit(task, commit)?;
            self.materialize(task, commit)?;
        }
        Ok(())
    }

    /// Three-way merge a working directory onto a new commit (D2): ours
    /// is the working directory, base is its recorded base commit,
    /// theirs is the new commit. board.yaml goes through
    /// `git merge-file`, conflicts left marked in the file. Figures
    /// merge per path: a path both sides changed is reported as a
    /// conflict with the working file left as ours. The recorded base
    /// moves to the new commit either way.
    pub fn update(&self, workdir: &Path, new_commit: &str) -> Result<UpdateOutcome, BoardsFail> {
        if !self.commit_exists(new_commit) {
            return Err(BoardsFail::MissingCommit {
                task: String::new(),
                commit: new_commit.to_string(),
            });
        }
        let base = recorded_base(workdir)?;
        let working_board = workdir.join(BOARD_FILE);
        std::fs::read_to_string(&working_board).map_err(|error| BoardsFail::WorkingBoard {
            path: working_board.clone(),
            detail: error.to_string(),
        })?;
        let base_text = self.text_at(&base, BOARD_FILE)?;
        let theirs_text = self.text_at(new_commit, BOARD_FILE)?;
        let base_figures = figures_in(&base_text);
        let theirs_figures = figures_in(&theirs_text);

        let mut outcome = UpdateOutcome {
            board_conflict: false,
            figure_conflicts: Vec::new(),
        };
        let base_file = workdir.join(".saccade-merge-base");
        let theirs_file = workdir.join(".saccade-merge-theirs");
        std::fs::write(&base_file, base_text.as_bytes()).map_err(|error| BoardsFail::Io {
            detail: format!("{}: {error}", base_file.display()),
        })?;
        std::fs::write(&theirs_file, theirs_text.as_bytes()).map_err(|error| BoardsFail::Io {
            detail: format!("{}: {error}", theirs_file.display()),
        })?;
        let merged = run(
            &self.repo,
            &[
                "merge-file",
                "-L",
                "ours",
                "-L",
                "base",
                "-L",
                "theirs",
                &working_board.to_string_lossy(),
                &base_file.to_string_lossy(),
                &theirs_file.to_string_lossy(),
            ],
            None,
        )?;
        let _ = std::fs::remove_file(&base_file);
        let _ = std::fs::remove_file(&theirs_file);
        match merged.status {
            0 => {}
            255 => {
                return Err(BoardsFail::Git(format!(
                    "git merge-file failed: {}",
                    merged.stderr.trim()
                )));
            }
            _ => outcome.board_conflict = true,
        }

        for path in base_figures.union(&theirs_figures) {
            let base_bytes = self.blob_at(&base, path)?;
            let theirs_bytes = self.blob_at(new_commit, path)?;
            if base_bytes == theirs_bytes {
                continue;
            }
            let ours_bytes = std::fs::read(workdir.join(path)).ok();
            if ours_bytes != base_bytes {
                outcome.figure_conflicts.push(path.clone());
                continue;
            }
            match theirs_bytes {
                Some(bytes) => {
                    let file = workdir.join(path);
                    if let Some(parent) = file.parent() {
                        std::fs::create_dir_all(parent).map_err(|error| BoardsFail::Io {
                            detail: format!("{}: {error}", parent.display()),
                        })?;
                    }
                    std::fs::write(&file, &bytes).map_err(|error| BoardsFail::Io {
                        detail: format!("{}: {error}", file.display()),
                    })?;
                }
                None => {
                    let _ = std::fs::remove_file(workdir.join(path));
                }
            }
        }
        outcome.figure_conflicts.sort();
        record_base(workdir, new_commit)?;
        Ok(outcome)
    }

    fn commit_tree(
        &self,
        workdir: &Path,
        board: &Board,
        parent: Option<&str>,
        message: &str,
    ) -> Result<String, BoardsFail> {
        let board_text = std::fs::read_to_string(workdir.join(BOARD_FILE)).map_err(|error| {
            BoardsFail::WorkingBoard {
                path: workdir.join(BOARD_FILE),
                detail: error.to_string(),
            }
        })?;
        let mut files = vec![(
            BOARD_FILE.to_string(),
            self.write_blob(board_text.as_bytes())?,
        )];
        for path in board_grammar::figures(board) {
            let bytes = std::fs::read(workdir.join(&path)).map_err(|error| BoardsFail::Io {
                detail: format!("{}: {error}", workdir.join(&path).display()),
            })?;
            files.push((path, self.write_blob(&bytes)?));
        }
        let tree = self.write_tree(&files)?;
        let mut args: Vec<&str> = vec!["commit-tree", &tree];
        if let Some(parent) = parent {
            args.push("-p");
            args.push(parent);
        }
        args.push("-m");
        args.push(message);
        let commit = run_ok(&self.repo, &args, None)?;
        Ok(commit.trim().to_string())
    }

    fn write_blob(&self, bytes: &[u8]) -> Result<String, BoardsFail> {
        run_ok(&self.repo, &["hash-object", "-w", "--stdin"], Some(bytes))
    }

    fn write_tree(&self, files: &[(String, String)]) -> Result<String, BoardsFail> {
        let mut by_name: BTreeMap<String, String> = BTreeMap::new();
        for (path, blob) in files {
            by_name.insert(path.clone(), blob.clone());
        }
        self.write_subtree(&by_name)
    }

    fn write_subtree(&self, files: &BTreeMap<String, String>) -> Result<String, BoardsFail> {
        // group one level: leaf files stay, deeper paths recurse under
        // their first segment
        let mut entries: Vec<(String, String, bool)> = Vec::new();
        let mut subtrees: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        for (path, blob) in files {
            match path.split_once('/') {
                None => entries.push((path.clone(), blob.clone(), false)),
                Some((head, tail)) => {
                    subtrees
                        .entry(head.to_string())
                        .or_default()
                        .insert(tail.to_string(), blob.clone());
                }
            }
        }
        for (name, files) in subtrees {
            let hash = self.write_subtree(&files)?;
            entries.push((name, hash, true));
        }
        // git tree entry order: names compare byte-wise, directories as
        // if their name ended in a slash
        entries.sort_by(|(a_name, _, a_dir), (b_name, _, b_dir)| {
            let mut a = a_name.as_bytes().to_vec();
            let mut b = b_name.as_bytes().to_vec();
            if *a_dir {
                a.push(b'/');
            }
            if *b_dir {
                b.push(b'/');
            }
            a.cmp(&b)
        });
        let input = entries
            .iter()
            .map(|(name, hash, is_dir)| {
                if *is_dir {
                    format!("40000 tree {hash}\t{name}")
                } else {
                    format!("100644 blob {hash}\t{name}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        run_ok(&self.repo, &["mktree"], Some(input.as_bytes()))
    }

    fn blob_at(&self, commit: &str, path: &str) -> Result<Option<Vec<u8>>, BoardsFail> {
        let spec = format!("{commit}:{path}");
        let probe = run(&self.repo, &["cat-file", "-e", &spec], None)?;
        if probe.status != 0 {
            return Ok(None);
        }
        let out = run(&self.repo, &["cat-file", "blob", &spec], None)?;
        if out.status != 0 {
            return Err(BoardsFail::Git(format!(
                "git cat-file blob failed: {}",
                out.stderr.trim()
            )));
        }
        Ok(Some(out.stdout))
    }

    /// The actor's checkpoint chain tip, when one stands.
    pub fn checkpoint_tip(&self, task: &str, actor: &str) -> Result<Option<String>, BoardsFail> {
        single_component(task)?;
        single_component(actor)?;
        self.read_ref(&format!("{COMMIT_REF_PREFIX}/{task}/wip/{actor}"))
    }

    /// One file's bytes inside a stored revision.
    pub fn file_bytes(&self, commit: &str, path: &str) -> Result<Option<Vec<u8>>, BoardsFail> {
        self.blob_at(commit, path)
    }

    /// A stored file's text; an absent path reads as empty.
    pub fn text_at(&self, commit: &str, path: &str) -> Result<String, BoardsFail> {
        match self.blob_at(commit, path)? {
            Some(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
            None => Ok(String::new()),
        }
    }

    /// The unified diff between two stored revisions.
    pub fn diff(&self, from: &str, to: &str) -> Result<String, BoardsFail> {
        let out = run(&self.repo, &["diff", from, to], None)?;
        if out.status != 0 {
            return Err(BoardsFail::Git(format!(
                "git diff failed: {}",
                out.stderr.trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// A commit's full message, trimmed.
    pub fn commit_message(&self, commit: &str) -> Result<String, BoardsFail> {
        let out = run(&self.repo, &["log", "-1", "--format=%B", commit], None)?;
        if out.status != 0 {
            return Err(BoardsFail::Git(format!(
                "git log failed for {commit}: {}",
                out.stderr.trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    fn read_ref(&self, refname: &str) -> Result<Option<String>, BoardsFail> {
        let out = run(
            &self.repo,
            &["rev-parse", "--verify", "--quiet", refname],
            None,
        )?;
        if out.status != 0 {
            return Ok(None);
        }
        Ok(Some(
            String::from_utf8_lossy(&out.stdout).trim().to_string(),
        ))
    }
}

/// Record the commit a working directory started from.
pub fn record_base(workdir: &Path, commit: &str) -> Result<(), BoardsFail> {
    let record = format!("{{\"base\": \"{commit}\"}}\n");
    std::fs::write(workdir.join(BASE_FILE), record).map_err(|error| BoardsFail::Io {
        detail: format!("{}: {error}", workdir.join(BASE_FILE).display()),
    })
}

/// The commit a working directory started from.
pub fn recorded_base(workdir: &Path) -> Result<String, BoardsFail> {
    let file = workdir.join(BASE_FILE);
    let text = std::fs::read_to_string(&file).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            BoardsFail::MissingBase {
                workdir: workdir.to_path_buf(),
            }
        } else {
            BoardsFail::Io {
                detail: format!("{}: {error}", file.display()),
            }
        }
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| BoardsFail::BadBase {
            workdir: workdir.to_path_buf(),
            detail: error.to_string(),
        })?;
    let base = value
        .get("base")
        .and_then(|v| v.as_str())
        .ok_or_else(|| BoardsFail::BadBase {
            workdir: workdir.to_path_buf(),
            detail: "no \"base\" string".to_string(),
        })?;
    Ok(base.to_string())
}

fn figures_in(board_text: &str) -> std::collections::BTreeSet<String> {
    match parse(board_text) {
        Ok(board) => board_grammar::figures(&board).into_iter().collect(),
        // checkpoint trees can hold mid-edit boards; their figures are
        // simply unknown to the merge
        Err(_) => std::collections::BTreeSet::new(),
    }
}

/// Every incorporation's citation from a parsed board: the source
/// task's name, the pinned revision number, and the selectors it takes.
fn incorporation_sources(board: &Board) -> Vec<(String, usize, Vec<SelectPath>)> {
    let mut found = Vec::new();
    walk_board(&board.content, &mut |component| {
        if let Component::Incorporation { source, .. } = component {
            found.push((source.task.clone(), source.revision, source.select.clone()));
        }
    });
    found
}

fn walk_board(content: &[Component], visit: &mut impl FnMut(&Component)) {
    for component in content {
        visit(component);
        match component {
            Component::Group { content, .. } | Component::Incorporation { content, .. } => {
                walk_board(content, visit)
            }
            _ => {}
        }
    }
}

/// A selector's refusal, taught with the citation it rode in on.
fn selector_refusal(name: &str, revision: &str, select: &str, cause: &ParseError) -> ParseError {
    ParseError {
        kind: cause.kind.clone(),
        at: cause.at.clone(),
        message: format!(
            "the incorporation citing {name} at {revision} selects {select}: {}",
            cause.message
        ),
    }
}

fn read_working_board(workdir: &Path) -> Result<Board, BoardsFail> {
    let path = workdir.join(BOARD_FILE);
    let text = std::fs::read_to_string(&path).map_err(|error| BoardsFail::WorkingBoard {
        path: path.clone(),
        detail: error.to_string(),
    })?;
    parse(&text).map_err(BoardsFail::BoardInvalid)
}

fn single_component(key: &str) -> Result<(), BoardsFail> {
    if key.is_empty() || key == "." || key == ".." || key.contains('/') {
        return Err(BoardsFail::InvalidKey {
            key: key.to_string(),
        });
    }
    Ok(())
}

fn symlink(target: &str, link: &Path) -> Result<(), BoardsFail> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).map_err(|error| BoardsFail::Io {
            detail: format!("{} to {}: {error}", link.display(), target),
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = (target, link);
    }
    Ok(())
}

fn set_executable(path: &Path) -> Result<(), BoardsFail> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)
            .map_err(|error| BoardsFail::Io {
                detail: format!("{}: {error}", path.display()),
            })?
            .permissions();
        perms.set_mode(perms.mode() | 0o111);
        std::fs::set_permissions(path, perms).map_err(|error| BoardsFail::Io {
            detail: format!("{}: {error}", path.display()),
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

struct Outcome {
    status: i32,
    stdout: Vec<u8>,
    stderr: String,
}

/// The repo-locating variables git reads before anything else, pinned
/// as data: `git rev-parse --local-env-vars`.
const INHERITED_REPO_ENV: &[&str] = &[
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
];

/// A git command whose repository comes only from its own arguments:
/// the inherited repo-locating variables are stripped, so a server or
/// a suite spawned under a hook's GIT_DIR cannot touch that
/// repository.
pub fn git_command() -> Command {
    let mut command = Command::new("git");
    scrub_inherited_repo_env(&mut command);
    command
}

/// The scrub alone, for callers assembling a git command themselves.
pub fn scrub_inherited_repo_env(command: &mut Command) {
    let mut names: Vec<String> = INHERITED_REPO_ENV
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    // the config triplets ride the same env: a surviving count with its
    // key/value pairs would be config lines the caller never meant
    if let Ok(count) = std::env::var("GIT_CONFIG_COUNT")
        && let Ok(count) = count.parse::<u64>()
    {
        for index in 0..count.min(4096) {
            names.push(format!("GIT_CONFIG_KEY_{index}"));
            names.push(format!("GIT_CONFIG_VALUE_{index}"));
        }
    }
    for name in names {
        command.env_remove(name);
    }
}

fn run(repo: &Path, args: &[&str], stdin: Option<&[u8]>) -> Result<Outcome, BoardsFail> {
    let mut command = git_command();
    command.arg("-C").arg(repo).args(args);
    if stdin.is_some() {
        command.stdin(std::process::Stdio::piped());
    } else {
        command.stdin(std::process::Stdio::null());
    }
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    let mut child = command.spawn().map_err(|error| {
        BoardsFail::Git(format!("git {}: {error}", args.first().unwrap_or(&"")))
    })?;
    if let Some(bytes) = stdin {
        child
            .stdin
            .take()
            .expect("stdin is piped")
            .write_all(bytes)
            .map_err(|error| {
                BoardsFail::Git(format!("git {}: {error}", args.first().unwrap_or(&"")))
            })?;
    }
    let output = child.wait_with_output().map_err(|error| {
        BoardsFail::Git(format!("git {}: {error}", args.first().unwrap_or(&"")))
    })?;
    Ok(Outcome {
        status: output.status.code().unwrap_or(255),
        stdout: output.stdout,
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn run_ok(repo: &Path, args: &[&str], stdin: Option<&[u8]>) -> Result<String, BoardsFail> {
    let outcome = run(repo, args, stdin)?;
    if outcome.status != 0 {
        return Err(BoardsFail::Git(format!(
            "git {} failed: {}",
            args.first().unwrap_or(&""),
            outcome.stderr.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&outcome.stdout).trim().to_string())
}

#[cfg(test)]
mod test {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sac-boards-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the scratch root is writable");
        dir
    }

    fn boards(tag: &str) -> Boards {
        let dir = scratch(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the scratch root is writable");
        Boards::open(&dir).expect("the boards repository opens")
    }

    fn workdir(tag: &str, name: &str) -> PathBuf {
        let dir = scratch(tag).join("work").join(name);
        std::fs::create_dir_all(&dir).expect("the working directory is writable");
        dir
    }

    const LEDGER_FINDING: &str = r#"grammar: 1
task: t-0
content:
  - text: |
      The backfill rebuilt the floop index from the recorded ledger and
      stopped at the first gap it cannot explain.
"#;

    const LEDGER_CITED: &str = r#"grammar: 1
task: t-0
content:
  - incorporation:
      source:
        task: t-1
        revision: 1
        select: [/content/1]
      content:
        - text: |
            The recorded factors carry the contraction window the
            backfill leans on.
  - incorporation:
      source:
        task: ledger-notes
        revision: 1
        select: [/content/0]
      content:
        - text: |
            The ledger line explains the gap the backfill stopped at.
"#;

    fn plinth_board(first_line: &str, third_line: &str) -> String {
        let pivot = "The arm rides the damped pivot.";
        format!(
            r#"grammar: 1
task: t-4
content:
  - text: |
      {first_line}
      {pivot}
      {third_line}
  - figure:
      input: figures/plinth-section.png
      caption: Section through the rebuilt plinth.
"#
        )
    }

    const ORIGINAL_DRAWING: &[u8] = b"first section drawing";
    const REVISED_DRAWING: &[u8] = b"revised section drawing";

    fn record(
        seq: usize,
        context: &crate::store::Context,
        event: crate::events::Event,
    ) -> crate::store::Record {
        crate::store::Record {
            id: crate::store::RecordId(seq),
            timestamp: seq as u64,
            context: context.clone(),
            event,
        }
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) {
        let file = dir.join(name);
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).expect("the parent directory is writable");
        }
        std::fs::write(file, bytes).expect("the file is writable");
    }

    // the shared opening move: a valid working directory with its figure,
    // committed and named through the recorded path
    fn first_commit(store: &Boards, tag: &str, name: &str, board: &str) -> String {
        let dir = workdir(tag, name);
        write(&dir, BOARD_FILE, board.as_bytes());
        write(&dir, "figures/plinth-section.png", ORIGINAL_DRAWING);
        let landed = store
            .commit("tone-arm-resonance", &dir, None, "first recorded account")
            .expect("the first commit lands");
        store
            .advance_commit("tone-arm-resonance", &landed.commit)
            .expect("the commit ref advances");
        landed.commit
    }

    const ABSENT: &str = "1111111111111111111111111111111111111111";

    // the merge fixture family: base and theirs are commits, ours is a
    // working directory recording the base it started from
    fn merge_pair(tag: &str) -> (Boards, PathBuf, String) {
        let store = boards(tag);
        let base_board = plinth_board(
            "The sweep found the peak at 7 Hz.",
            "The mat was on during the sweep.",
        );
        let dir = workdir(tag, "base");
        write(&dir, BOARD_FILE, base_board.as_bytes());
        write(&dir, "figures/plinth-section.png", ORIGINAL_DRAWING);
        let landed = store
            .commit("tone-arm-resonance", &dir, None, "the base account")
            .expect("the base commit lands");
        store
            .advance_commit("tone-arm-resonance", &landed.commit)
            .expect("the commit ref advances");
        let theirs = workdir(tag, "theirs");
        write(
            &theirs,
            BOARD_FILE,
            plinth_board(
                "The sweep found the peak at 7 Hz.",
                "The mat was off during the resweep.",
            )
            .as_bytes(),
        );
        write(&theirs, "figures/plinth-section.png", REVISED_DRAWING);
        let new_commit = store
            .commit(
                "tone-arm-resonance",
                &theirs,
                Some(&landed.commit),
                "the resweep account",
            )
            .expect("the new commit lands")
            .commit;
        let ours = workdir(tag, "ours");
        write(
            &ours,
            BOARD_FILE,
            plinth_board(
                "The sweep found the peak at 6.9 Hz.",
                "The mat was on during the sweep.",
            )
            .as_bytes(),
        );
        write(&ours, "figures/plinth-section.png", ORIGINAL_DRAWING);
        record_base(&ours, &landed.commit).expect("the base records");
        (store, ours, new_commit)
    }

    #[test]
    fn open_is_idempotent_and_bare() {
        let tag = "open";
        let state = scratch(tag);
        let first = Boards::open(&state).expect("the first open");
        let second = Boards::open(&state).expect("the second open opens the same repository");
        assert_eq!(first.repo, second.repo);
        assert!(first.repo.join("HEAD").exists());
        let bare =
            run_ok(&first.repo, &["rev-parse", "--is-bare-repository"], None).expect("it answers");
        assert_eq!(bare, "true");
    }

    // a commit writes objects and returns the declared references, but
    // moves nothing: the ref appears only through advance_commit
    #[test]
    fn commit_writes_objects_and_extracts_incorporations() {
        let tag = "commit";
        let store = boards(tag);
        let dir = workdir(tag, "cited");
        write(&dir, BOARD_FILE, LEDGER_CITED.as_bytes());
        let landed = store
            .commit(
                "floop-index-backfill",
                &dir,
                None,
                "the backfill account, first draft",
            )
            .expect("the commit lands");
        assert_eq!(landed.commit.len(), 40);
        assert!(
            landed
                .commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "{}",
            landed.commit
        );
        assert_eq!(
            landed.incorporations,
            vec![("t-1".to_string(), 1), ("ledger-notes".to_string(), 1)]
        );
        assert!(store.commit_exists(&landed.commit));
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/floop-index-backfill/commit"))
                .expect("the ref reads"),
            None,
            "the commit names nothing before the caller advances it"
        );
    }

    // the crash point between objects and ref: the objects exist but
    // nothing names them, and recovery runs through the recorded path
    #[test]
    fn a_crash_after_the_objects_names_nothing() {
        let tag = "crash";
        let store = boards(tag);
        let dir = workdir(tag, "finding");
        write(&dir, BOARD_FILE, LEDGER_FINDING.as_bytes());
        let landed = store
            .commit(
                "floop-index-backfill",
                &dir,
                None,
                "an account lost to a crash",
            )
            .expect("the objects land");
        let refs = run_ok(
            &store.repo,
            &[
                "for-each-ref",
                &format!("{COMMIT_REF_PREFIX}/floop-index-backfill"),
            ],
            None,
        )
        .expect("the refs list");
        assert!(refs.is_empty(), "no ref names the crashed commit: {refs}");
        assert!(store.commit_exists(&landed.commit));
        store
            .advance_commit("floop-index-backfill", &landed.commit)
            .expect("the recovery advances the ref");
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/floop-index-backfill/commit"))
                .expect("the ref reads"),
            Some(landed.commit.clone())
        );
    }

    #[test]
    fn a_parent_the_repository_lacks_refuses() {
        let tag = "parent";
        let store = boards(tag);
        let dir = workdir(tag, "finding");
        write(&dir, BOARD_FILE, LEDGER_FINDING.as_bytes());
        let error = store
            .commit(
                "floop-index-backfill",
                &dir,
                Some(ABSENT),
                "an account on nothing",
            )
            .expect_err("the missing parent refuses");
        assert_eq!(error.code(), "boards_missing_commit");
        assert!(
            error.detail().contains(ABSENT),
            "the refusal names the commit: {}",
            error.detail()
        );
    }

    #[test]
    fn a_figure_must_exist_within_the_bound() {
        let tag = "figures";
        let store = boards(tag);
        let dir = workdir(tag, "plinth");
        write(
            &dir,
            BOARD_FILE,
            plinth_board("The peak sits near 7 Hz.", "The mat was on.").as_bytes(),
        );
        let error = store
            .commit(
                "tone-arm-resonance",
                &dir,
                None,
                "the section is not placed yet",
            )
            .expect_err("the missing figure refuses");
        assert_eq!(error.code(), "boards_figure_missing");
        write(
            &dir,
            "figures/plinth-section.png",
            &vec![7u8; MAX_FIGURE_BYTES as usize],
        );
        store
            .commit(
                "tone-arm-resonance",
                &dir,
                None,
                "the section at the exact bound",
            )
            .expect("a figure at exactly the bound passes");
        write(
            &dir,
            "figures/plinth-section.png",
            &vec![7u8; MAX_FIGURE_BYTES as usize + 1],
        );
        let error = store
            .commit(
                "tone-arm-resonance",
                &dir,
                None,
                "the section grew too large",
            )
            .expect_err("the oversized figure refuses");
        assert_eq!(error.code(), "boards_figure_too_large");
        assert!(
            error.detail().contains("figures/plinth-section.png"),
            "{}",
            error.detail()
        );
    }

    #[test]
    fn advance_and_read_round_trip() {
        let tag = "roundtrip";
        let store = boards(tag);
        let dir = workdir(tag, "finding");
        write(&dir, BOARD_FILE, LEDGER_FINDING.as_bytes());
        let landed = store
            .commit("floop-index-backfill", &dir, None, "the backfill account")
            .expect("the commit lands");
        store
            .advance_commit("floop-index-backfill", &landed.commit)
            .expect("the ref advances");
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/floop-index-backfill/commit"))
                .expect("the ref reads"),
            Some(landed.commit.clone())
        );
        let board = store
            .read_board_at(&landed.commit)
            .expect("the stored board reads");
        assert_eq!(board.task, "t-0");
        let error = store
            .read_board_at(ABSENT)
            .expect_err("a commit the repository lacks refuses");
        assert_eq!(error.code(), "boards_missing_commit");
    }

    // the repair arm of D7: the ref and the checkout are derived caches
    // set to whatever the log says is current, forward or backward
    #[test]
    fn repair_moves_a_lagging_ref_and_its_checkout() {
        let tag = "repair";
        let store = boards(tag);
        let older = first_commit(
            &store,
            tag,
            "older",
            &plinth_board("The older sweep found the peak at 7 Hz.", "The mat was on."),
        );
        let dir = workdir(tag, "newer");
        write(
            &dir,
            BOARD_FILE,
            plinth_board("The newer sweep found 7.2 Hz.", "The mat was off.").as_bytes(),
        );
        write(&dir, "figures/plinth-section.png", REVISED_DRAWING);
        let newer = store
            .commit(
                "tone-arm-resonance",
                &dir,
                Some(&older),
                "the revised account",
            )
            .expect("the newer commit lands")
            .commit;
        store
            .repair(&[("tone-arm-resonance".to_string(), newer.clone())])
            .expect("the repair lands the newer state");
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/tone-arm-resonance/commit"))
                .expect("the ref reads"),
            Some(newer.clone())
        );
        let checkout =
            std::fs::read_to_string(store.checkouts.join("tone-arm-resonance").join(BOARD_FILE))
                .expect("the materialized board reads");
        assert!(checkout.contains("7.2 Hz"), "{}", checkout);
        assert_eq!(
            std::fs::read(
                store
                    .checkouts
                    .join("tone-arm-resonance")
                    .join("figures/plinth-section.png")
            )
            .expect("the materialized figure reads"),
            REVISED_DRAWING
        );
        store
            .repair(&[("tone-arm-resonance".to_string(), older.clone())])
            .expect("the backward repair lands");
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/tone-arm-resonance/commit"))
                .expect("the ref reads"),
            Some(older)
        );
    }

    #[test]
    fn repair_refuses_when_a_logged_commit_is_missing() {
        let tag = "repair-missing";
        let store = boards(tag);
        let landed = first_commit(
            &store,
            tag,
            "finding",
            &plinth_board("The sweep found the peak at 7 Hz.", "The mat was on."),
        );
        let error = store
            .repair(&[("tone-arm-resonance".to_string(), ABSENT.to_string())])
            .expect_err("the missing logged commit refuses the repair");
        assert_eq!(error.code(), "boards_missing_commit");
        assert!(error.detail().contains(ABSENT), "{}", error.detail());
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/tone-arm-resonance/commit"))
                .expect("the ref reads"),
            Some(landed),
            "the repair refused without moving the ref"
        );
    }

    // checkpoints hang off the actor's own chain, seeded by the commit
    // the session started from, and stay invisible at the commit ref
    #[test]
    fn a_checkpoint_never_becomes_the_commit_ref() {
        let tag = "checkpoint";
        let store = boards(tag);
        let base = first_commit(
            &store,
            tag,
            "finding",
            &plinth_board("The sweep found the peak at 7 Hz.", "The mat was on."),
        );
        let dir = workdir(tag, "drafting");
        write(
            &dir,
            BOARD_FILE,
            plinth_board("Drafting: the peak moved to 7.2 Hz.", "The mat was on.").as_bytes(),
        );
        let first = store
            .checkpoint(
                "tone-arm-resonance",
                "pi",
                &dir,
                &base,
                "checkpoint: drafting",
            )
            .expect("the first checkpoint lands");
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/tone-arm-resonance/wip/pi"))
                .expect("the wip ref reads"),
            Some(first.clone())
        );
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/tone-arm-resonance/commit"))
                .expect("the commit ref reads"),
            Some(base.clone()),
            "the checkpoint never shows at the commit ref"
        );
        assert_eq!(
            run_ok(&store.repo, &["rev-parse", &format!("{first}^")], None)
                .expect("the chain parent reads"),
            base,
            "the first checkpoint hangs off the session's base"
        );
        write(
            &dir,
            BOARD_FILE,
            plinth_board("Drafting: the peak at 7.4 Hz.", "The mat was off.").as_bytes(),
        );
        let second = store
            .checkpoint(
                "tone-arm-resonance",
                "pi",
                &dir,
                &base,
                "checkpoint: drafting again",
            )
            .expect("the second checkpoint lands");
        assert_eq!(
            run_ok(&store.repo, &["rev-parse", &format!("{second}^")], None)
                .expect("the chain parent reads"),
            first,
            "later checkpoints chain onto the actor's tip"
        );
        assert_eq!(
            store
                .unsaved_checkpoints("tone-arm-resonance")
                .expect("the actors list"),
            vec!["pi".to_string()]
        );
        store
            .discard_checkpoint("tone-arm-resonance", "pi")
            .expect("the discard lands");
        assert_eq!(
            store
                .unsaved_checkpoints("tone-arm-resonance")
                .expect("the actors list"),
            Vec::<String>::new()
        );
        assert!(
            store.commit_exists(&second),
            "discarding the reference retains the chain's objects"
        );
    }

    // checkpoints skip grammar and figure checks: a snapshot mid-edit
    // lands even while the board does not parse
    #[test]
    fn a_checkpoint_snapshots_without_validation() {
        let tag = "checkpoint-raw";
        let store = boards(tag);
        let base = first_commit(
            &store,
            tag,
            "finding",
            &plinth_board("The sweep found the peak at 7 Hz.", "The mat was on."),
        );
        let dir = workdir(tag, "mid-edit");
        write(&dir, BOARD_FILE, b"grammar: 2\ntask: t-4\ncontent: []\n");
        let saved = store
            .checkpoint(
                "tone-arm-resonance",
                "claude",
                &dir,
                &base,
                "checkpoint: mid-edit",
            )
            .expect("the checkpoint lands without validation");
        assert!(store.commit_exists(&saved));
    }

    // the checkout is a symlink swap: the task's path resolves to a
    // complete tree at every instant, and the previous tree leaves only
    // after the swap has landed
    #[test]
    fn the_checkout_path_always_resolves_to_a_complete_tree() {
        let tag = "materialize";
        let store = boards(tag);
        let older = first_commit(
            &store,
            tag,
            "older",
            &plinth_board("The older sweep found the peak at 7 Hz.", "The mat was on."),
        );
        let dir = workdir(tag, "newer");
        write(
            &dir,
            BOARD_FILE,
            plinth_board("The newer sweep found 7.2 Hz.", "The mat was off.").as_bytes(),
        );
        write(&dir, "figures/plinth-section.png", REVISED_DRAWING);
        let newer = store
            .commit(
                "tone-arm-resonance",
                &dir,
                Some(&older),
                "the revised account",
            )
            .expect("the newer commit lands")
            .commit;
        store
            .materialize("tone-arm-resonance", &newer)
            .expect("the newer checkout materializes");
        let checkout = store.checkouts.join("tone-arm-resonance");
        assert!(checkout.is_symlink(), "the task's path is the swap point");
        assert_eq!(
            std::fs::read_link(&checkout).expect("the link reads"),
            PathBuf::from(format!(".tone-arm-resonance-{newer}"))
        );
        let board = std::fs::read_to_string(checkout.join(BOARD_FILE)).expect("the checkout reads");
        assert!(board.contains("7.2 Hz"), "{}", board);
        assert_eq!(
            std::fs::read(checkout.join("figures/plinth-section.png")).expect("the figure reads"),
            REVISED_DRAWING
        );
        store
            .materialize("tone-arm-resonance", &older)
            .expect("the older checkout replaces it");
        assert!(checkout.is_symlink());
        let board = std::fs::read_to_string(checkout.join(BOARD_FILE)).expect("the checkout reads");
        assert!(board.contains("7 Hz"), "{}", board);
        assert_eq!(
            std::fs::read(checkout.join("figures/plinth-section.png")).expect("the figure reads"),
            ORIGINAL_DRAWING
        );
        assert!(
            !store
                .checkouts
                .join(format!(".tone-arm-resonance-{newer}"))
                .exists(),
            "the replaced tree leaves after the swap"
        );
        let leftovers: Vec<String> = std::fs::read_dir(&store.checkouts)
            .expect("the checkout root lists")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(".swap-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "no swap temporary remains: {leftovers:?}"
        );
    }

    #[test]
    fn a_clean_merge_takes_both_sides() {
        let tag = "merge-clean";
        let (store, ours, new_commit) = merge_pair(tag);
        let outcome = store
            .update(&ours, &new_commit)
            .expect("the clean update lands");
        assert!(!outcome.board_conflict);
        assert!(outcome.figure_conflicts.is_empty());
        let board = std::fs::read_to_string(ours.join(BOARD_FILE)).expect("the merged board reads");
        assert!(board.contains("6.9 Hz"), "ours survives: {}", board);
        assert!(
            board.contains("The mat was off during the resweep."),
            "theirs arrives: {}",
            board
        );
        assert_eq!(
            std::fs::read(ours.join("figures/plinth-section.png")).expect("the figure reads"),
            REVISED_DRAWING,
            "a figure only theirs changed arrives"
        );
        assert_eq!(recorded_base(&ours).expect("the base reads"), new_commit);
    }

    #[test]
    fn a_conflicting_line_is_marked_and_reported() {
        let tag = "merge-conflict";
        let (store, ours, new_commit) = merge_pair(tag);
        write(
            &ours,
            BOARD_FILE,
            plinth_board(
                "The sweep found the peak at 7 Hz.",
                "The mat was off during the sweep.",
            )
            .as_bytes(),
        );
        let outcome = store
            .update(&ours, &new_commit)
            .expect("the update reports its conflict");
        assert!(outcome.board_conflict);
        assert!(outcome.figure_conflicts.is_empty());
        let board = std::fs::read_to_string(ours.join(BOARD_FILE)).expect("the marked board reads");
        assert!(board.contains("<<<<<<< ours"), "{}", board);
        assert!(board.contains(">>>>>>> theirs"), "{}", board);
        assert_eq!(recorded_base(&ours).expect("the base reads"), new_commit);
    }

    #[test]
    fn a_figure_both_sides_changed_conflicts() {
        let tag = "merge-figure";
        let (store, ours, new_commit) = merge_pair(tag);
        write(
            &ours,
            "figures/plinth-section.png",
            b"the author's annotated drawing",
        );
        let outcome = store
            .update(&ours, &new_commit)
            .expect("the update reports its figure conflict");
        assert!(!outcome.board_conflict);
        assert_eq!(
            outcome.figure_conflicts,
            vec!["figures/plinth-section.png".to_string()],
            "a path both sides changed is the conflict"
        );
        assert_eq!(
            std::fs::read(ours.join("figures/plinth-section.png")).expect("the figure reads"),
            b"the author's annotated drawing",
            "the conflict leaves the working file as ours"
        );
    }

    #[test]
    fn update_refuses_a_workdir_without_a_recorded_base() {
        let tag = "merge-nobase";
        let (store, ours, new_commit) = merge_pair(tag);
        let _ = std::fs::remove_file(ours.join(BASE_FILE));
        let error = store
            .update(&ours, &new_commit)
            .expect_err("the unrecorded working directory refuses");
        assert_eq!(error.code(), "boards_base_unrecorded");
    }

    #[test]
    fn the_base_record_round_trips() {
        let tag = "base-record";
        let dir = workdir(tag, "finding");
        write(&dir, BOARD_FILE, LEDGER_FINDING.as_bytes());
        let error = recorded_base(&dir).expect_err("nothing records a base yet");
        assert_eq!(error.code(), "boards_base_unrecorded");
        record_base(&dir, "647de6e6d2b90c507bf2d05dacc32740558e0bfc").expect("the base records");
        assert_eq!(
            recorded_base(&dir).expect("the base reads"),
            "647de6e6d2b90c507bf2d05dacc32740558e0bfc"
        );
        write(&dir, BASE_FILE, b"not json");
        let error = recorded_base(&dir).expect_err("the corrupted record refuses");
        assert_eq!(error.code(), "boards_base_unreadable");
    }

    // the publish wiring runs the D7 order: the record lands before
    // the ref moves, and a store refusal leaves the caches untouched
    #[test]
    fn publish_wires_the_record_between_the_objects_and_the_ref() {
        let tag = "publish";
        let store = boards(tag);
        let dir = workdir(tag, "finding");
        write(&dir, BOARD_FILE, LEDGER_FINDING.as_bytes());
        let mut world = crate::store::World::new();
        world = world
            .apply(record(
                0,
                &crate::store::Context {
                    actor: crate::types::actor::ActorName::new("human person".into()).unwrap(),
                    tier: crate::store::Tier::Human,
                    scope: None,
                },
                crate::events::Event::TaskCreated {
                    name: crate::types::prose::Prose::new("implement foo".into()).unwrap(),
                },
            ))
            .expect("the seed world folds");
        let task = crate::objects::task::TaskId(0);

        // a store refusal (the closure declines) leaves the ref and the
        // checkout untouched, the objects already written
        let refused =
            store.publish_board(task, "0", &world, &dir, "the declined publication", |_| {
                Err("the store refused".to_string())
            });
        assert!(refused.is_err(), "the declined record surfaces");
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/0/commit"))
                .expect("the ref reads"),
            None,
            "the ref never moved past the refused record"
        );
        assert!(!store.checkouts.join("0").exists());

        // the real record lands, and the caches follow it
        let landed = store
            .publish_board(
                task,
                "0",
                &world,
                &dir,
                "the published account",
                |command| {
                    assert!(matches!(
                        command,
                        crate::events::Command::CommitBoard { .. }
                    ));
                    Ok(())
                },
            )
            .expect("the publish lands");
        assert_eq!(
            store
                .read_ref(&format!("{COMMIT_REF_PREFIX}/0/commit"))
                .expect("the ref reads"),
            Some(landed.clone())
        );
        assert!(
            store.checkouts.join("0").join(BOARD_FILE).exists(),
            "the checkout materializes"
        );

        // the refusal pins, folded beside the publish contract: the
        // board's task field must name the committing task, and an
        // incorporation cites an existing t-N
        write(
            &dir,
            BOARD_FILE,
            r#"grammar: 1
task: t-9
content:
  - text: somebody else's account
"#
            .as_bytes(),
        );
        let mismatched = store.publish_board(
            crate::objects::task::TaskId(3),
            "3",
            &world,
            &dir,
            "somebody else's account",
            |_| Ok(()),
        );
        let error = mismatched.expect_err("the mismatch refuses");
        assert_eq!(error.code(), "boards_task_mismatch");
        assert!(
            error.detail().contains("t-9") && error.detail().contains("t-3"),
            "the refusal names both tasks: {}",
            error.detail()
        );
        {
            let source = workdir(tag, "source");
            write(&source, BOARD_FILE, LEDGER_FINDING.as_bytes());
            store
                .commit("0", &source, None, "the source account")
                .expect("the source commit lands");
        }
        write(
            &dir,
            BOARD_FILE,
            r#"grammar: 1
task: t-0
content:
  - incorporation:
      source:
        task: t-9
        revision: 1
        select: [/content/0]
      content:
        - text: |
            The receiving text explains what this board takes.
"#
            .as_bytes(),
        );
        let unknown = store.publish_board(
            crate::objects::task::TaskId(0),
            "0",
            &world,
            &dir,
            "the citing account",
            |_| Ok(()),
        );
        let error = unknown.expect_err("the unknown source refuses");
        assert_eq!(error.code(), "boards_unknown_source_task");
        assert!(
            error.detail().contains("t-9"),
            "the refusal names the citation: {}",
            error.detail()
        );
    }

    // a selector the cited source revision does not hold refuses the
    // publish before the record is attempted
    #[test]
    fn a_selector_outside_its_cited_revision_refuses_the_publish() {
        let tag = "selector";
        let store = boards(tag);
        // seed the source revision: a two-text board at revision X
        let source = workdir(tag, "source");
        write(&source, BOARD_FILE, LEDGER_FINDING.as_bytes());
        store
            .commit("0", &source, None, "the source account")
            .expect("the source commit lands");
        let source_commit = store
            .commit("0", &source, None, "the source account")
            .expect("the source commit lands")
            .commit;

        // the citing board selects /content/9, which the source lacks
        let dir = workdir(tag, "citing");
        write(
            &dir,
            BOARD_FILE,
            format!(
                r#"grammar: 1
task: t-0
content:
  - incorporation:
      source:
        task: t-0
        revision: {source_commit}
        select: [/content/9]
      content:
        - text: |
            The receiving text explains what this board takes.
"#
            )
            .as_bytes(),
        );
        let world = crate::store::World::new();
        let refused = store.publish_board(
            crate::objects::task::TaskId(0),
            "0",
            &world,
            &dir,
            "the citing account",
            |_| Ok(()),
        );
        assert!(refused.is_err(), "the out-of-bounds selector refuses");
    }

    #[test]
    fn a_task_key_or_actor_must_be_a_single_component() {
        let tag = "keys";
        let store = boards(tag);
        for bad in ["", ".", "..", "tone/resonance"] {
            let error = store
                .advance_commit(bad, ABSENT)
                .expect_err("the unsafe key refuses");
            assert_eq!(error.code(), "boards_invalid_key");
        }
        let error = store
            .discard_checkpoint("tone-arm-resonance", "pi/chain")
            .expect_err("the unsafe actor refuses");
        assert_eq!(error.code(), "boards_invalid_key");
    }
}
