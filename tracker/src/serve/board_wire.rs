//! The board wire: request and reply shapes for the board doors, and
//! the read-side helpers that turn a fold plus the boards repository
//! into history, search, and export lines. File bytes ride base64;
//! every file stays within the per-file bound.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Component as PathComponent, Path, PathBuf};

use crate::RecordId;
use crate::api::WireContext;
use crate::objects::task::TaskId;
use crate::store::World;
use crate::wire;

pub const MAX_BOARD_FILE_BYTES: usize = 8 * 1024 * 1024;
pub const BOARD_FILE_NAME: &str = "board.yaml";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoardFile {
    pub path: String,
    pub content: String,
}

/// One working directory over the wire: board.yaml plus its figures.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BoardFiles {
    pub files: Vec<BoardFile>,
}

/// The board content a first edit starts from: the task named as the
/// tracker addresses it, the grammar this build speaks, and no
/// components yet.
pub fn fresh_board_text(task: usize) -> String {
    format!("grammar: 1\ntask: t-{task}\ncontent: []\n")
}

impl BoardFiles {
    /// Collect board.yaml and its referenced figures from a working
    /// directory. A board.yaml that does not parse rides alone: the
    /// server's validator, not the collect step, refuses it.
    pub fn from_workdir(workdir: &Path) -> Result<Self, String> {
        let board = read_file(workdir, &workdir.join(BOARD_FILE_NAME))?;
        let mut files = vec![encode(BOARD_FILE_NAME, &board)?];
        if let Ok(text) = std::str::from_utf8(&board)
            && let Ok(parsed) = board_grammar::parse(text)
        {
            for path in board_grammar::figures(&parsed) {
                let bytes = read_file(workdir, &workdir.join(&path))?;
                files.push(encode(&path, &bytes)?);
            }
        }
        Ok(BoardFiles { files })
    }

    pub fn write_to(&self, destination: &Path) -> Result<(), String> {
        std::fs::create_dir_all(destination).map_err(|error| error.to_string())?;
        for file in &self.files {
            let path = safe_relative(&file.path)?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&file.content)
                .map_err(|error| format!("{} is not base64: {error}", file.path))?;
            if bytes.len() > MAX_BOARD_FILE_BYTES {
                return Err(format!(
                    "{} is {} bytes; board files stay at most {} bytes",
                    file.path,
                    bytes.len(),
                    MAX_BOARD_FILE_BYTES
                ));
            }
            let output = destination.join(path);
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            std::fs::write(output, bytes).map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

fn encode(path: &str, bytes: &[u8]) -> Result<BoardFile, String> {
    if bytes.len() > MAX_BOARD_FILE_BYTES {
        return Err(format!(
            "{path} is {} bytes; board files stay at most {} bytes",
            bytes.len(),
            MAX_BOARD_FILE_BYTES
        ));
    }
    Ok(BoardFile {
        path: path.to_string(),
        content: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

fn read_file(workdir: &Path, path: &Path) -> Result<Vec<u8>, String> {
    let metadata =
        std::fs::symlink_metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{} is not a regular board file", path.display()));
    }
    if !path.starts_with(workdir) {
        return Err(format!(
            "{} escapes the board working directory",
            path.display()
        ));
    }
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() > MAX_BOARD_FILE_BYTES {
        return Err(format!(
            "{} is {} bytes; board files stay at most {} bytes",
            path.display(),
            bytes.len(),
            MAX_BOARD_FILE_BYTES
        ));
    }
    Ok(bytes)
}

fn safe_relative(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if value.is_empty()
        || path
            .components()
            .any(|component| !matches!(component, PathComponent::Normal(_)))
    {
        return Err(format!("{value:?} is not a safe relative board path"));
    }
    Ok(path.to_path_buf())
}

// -- wire requests ---------------------------------------------------

#[derive(Deserialize, Serialize)]
pub struct CheckpointRequest {
    pub context: WireContext,
    pub task: usize,
    pub files: BoardFiles,
    pub base: String,
    pub message: String,
    #[serde(default)]
    pub client: Option<String>,
}

impl CheckpointRequest {
    pub fn context_client(&self) -> Option<String> {
        self.client.clone()
    }
}

#[derive(Serialize, Deserialize)]
pub struct CheckpointReply {
    pub commit: String,
}

#[derive(Deserialize, Serialize)]
pub struct CommitRequest {
    pub context: WireContext,
    pub task: usize,
    pub files: BoardFiles,
    pub message: String,
    #[serde(default)]
    pub client: Option<String>,
}

impl CommitRequest {
    pub fn context_client(&self) -> Option<String> {
        self.client.clone()
    }
}

#[derive(Serialize, Deserialize)]
pub struct CommitReply {
    pub records: Vec<crate::db::StoredRecord>,
    pub commit: String,
    pub revision: usize,
}

#[derive(Deserialize, Serialize)]
pub struct DiscardRequest {
    pub context: WireContext,
    pub task: usize,
    #[serde(default)]
    pub client: Option<String>,
}

impl DiscardRequest {
    pub fn context_client(&self) -> Option<String> {
        self.client.clone()
    }
}

#[derive(Serialize, Deserialize)]
pub struct DiscardReply {
    pub task: String,
    pub actor: String,
}

#[derive(Deserialize, Serialize)]
pub struct UpdateRequest {
    pub task: usize,
    pub base: String,
    pub files: BoardFiles,
}

#[derive(Serialize, Deserialize)]
pub struct UpdateReply {
    pub files: BoardFiles,
    pub board_conflict: bool,
    pub figure_conflicts: Vec<String>,
    pub base: String,
    pub up_to_date: bool,
}

#[derive(Deserialize, Serialize)]
pub struct EditQuery {
    pub task: usize,
    pub actor: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct EditReply {
    pub files: BoardFiles,
    pub base: Option<String>,
    pub commit: Option<String>,
    /// The committing task, named as the board's task field must
    pub task: String,
}

#[derive(Deserialize, Serialize)]
pub struct GetQuery {
    pub task: usize,
    pub revision: Option<usize>,
    /// A file in the board tree; the whole tree when absent
    pub path: Option<String>,
    /// A positional path naming one component of the revision's board
    pub select: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct GetReply {
    pub name: String,
    pub content: String,
}

#[derive(Serialize, Deserialize)]
pub struct TreeReply {
    pub files: BoardFiles,
    pub commit: String,
    pub revision: usize,
}

#[derive(Deserialize, Serialize)]
pub struct DiffQuery {
    pub task: usize,
    pub from: usize,
    pub to: usize,
}

#[derive(Serialize, Deserialize)]
pub struct DiffReply {
    pub diff: String,
    pub from: usize,
    pub to: usize,
}

#[derive(Deserialize, Serialize)]
pub struct SearchQuery {
    pub terms: Vec<String>,
    pub history: bool,
    pub within: Option<String>,
}

/// One search answer line; the tag picks the human face.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SearchLine {
    /// A current board matched: one line per task.
    Board {
        task: usize,
        name: String,
        revision: usize,
        excerpt: String,
    },
    /// A history hit: the revision where a term appeared or
    /// disappeared, or a decision's grounds riding its revision.
    History {
        task: usize,
        revision: Option<usize>,
        event: String,
        term: Option<String>,
        detail: Option<String>,
    },
    /// A reference hit: incorporations citing the task or revision,
    /// and messages carrying the task.
    Reference { line: String },
}

#[derive(Serialize, Deserialize)]
pub struct SearchReply {
    pub lines: Vec<SearchLine>,
}

// -- read-side assembly ----------------------------------------------

/// One history entry: the record facts of the task's story, the commit
/// messages from Git, and the session behind the record - the
/// incarnation its recorded scope names, when the fold resolves it.
/// The CLI face and the pages render the same entries.
#[derive(Serialize)]
pub struct HistoryEntry {
    pub seq: usize,
    pub event: String,
    pub actor: String,
    pub tier: String,
    pub at: u64,
    pub revision: Option<usize>,
    pub commit: Option<String>,
    pub message: Option<String>,
    pub grounds: Option<String>,
    pub explanation: Option<String>,
    pub session: Option<String>,
}

/// The incarnation a record's recorded scope names, as the public
/// i-N; a scope naming no resident incarnation names no session.
pub(crate) fn session_of(world: &World, row: &crate::db::StoredRecord) -> Option<String> {
    let scope: crate::store::Scope = serde_json::from_str(row.scope.as_ref()?).ok()?;
    let num = world.incarnation_num(scope.incarnation?);
    num.map(|num| format!("i-{num}"))
}

/// The task's history: commits interleaved with decisions, grouped by
/// revision, ordered by record. Commit messages come from Git; every
/// other fact comes from the fold.
pub fn board_history(
    world: &World,
    rows: &[crate::db::StoredRecord],
    boards: &crate::boards::Boards,
    task: usize,
) -> Result<Vec<HistoryEntry>, String> {
    let mut entries = Vec::new();
    for row in rows {
        let Ok(event) = wire::assemble(&row.kind, &row.payload) else {
            continue;
        };
        let named: Option<(TaskId, Option<serde_json::Value>)> = match &event {
            crate::Event::TaskCreated { .. } => world
                .task_born_at(RecordId(row.seq))
                .filter(|id| id.0 == task)
                .map(|id| (id, None)),
            crate::Event::TaskClaimed { id, incarnation } => (*id == TaskId(task))
                .then_some((
                    *id,
                    Some(json!({
                        "owner": incarnation.map(|i| format!("i-{}", i.0 .0)),
                    })),
                ))
                .filter(|(_, extra)| extra.is_some()),
            crate::Event::TaskCommitted {
                task: id, commit, ..
            } => (*id == TaskId(task)).then_some((*id, Some(json!({ "commit": commit.as_str() })))),
            crate::Event::TaskOffered { task: id, commit } => {
                (*id == TaskId(task)).then_some((*id, Some(json!({ "commit": commit.as_str() }))))
            }
            crate::Event::TaskRejected { task: id, grounds } => {
                (*id == TaskId(task)).then_some((*id, Some(json!({ "grounds": grounds.as_str() }))))
            }
            crate::Event::TaskAccepted { task: id } => (*id == TaskId(task)).then_some((*id, None)),
            crate::Event::TaskCancelled { id, explanation } => (*id == TaskId(task))
                .then_some((*id, Some(json!({ "explanation": explanation.as_str() }))))
                .filter(|(_, extra)| extra.is_some()),
            crate::Event::TaskReopened { id, explanation } => (*id == TaskId(task))
                .then_some((*id, Some(json!({ "explanation": explanation.as_str() }))))
                .filter(|(_, extra)| extra.is_some()),
            _ => None,
        };
        let Some((_, extra)) = named else {
            continue;
        };
        let extra = extra.unwrap_or(json!({}));
        let revision = world
            .tasks
            .get(task)
            .and_then(|ctx| ctx.revision_of(RecordId(row.seq)));
        let commit = extra["commit"].as_str().map(str::to_string);
        let message = match &commit {
            Some(hash) => boards.commit_message(hash).ok(),
            None => None,
        };
        entries.push(HistoryEntry {
            seq: row.seq,
            event: row.kind.clone(),
            actor: row.actor.clone(),
            tier: row.tier.clone(),
            at: row.event_time,
            revision,
            commit,
            message,
            grounds: extra["grounds"].as_str().map(str::to_string),
            explanation: extra["explanation"].as_str().map(str::to_string),
            session: session_of(world, row),
        });
    }
    Ok(entries)
}

/// The board search over a fold plus the boards repository: current
/// boards by default, history changes, and id terms are references.
pub fn board_search(
    world: &World,
    rows: &[crate::db::StoredRecord],
    boards: &crate::boards::Boards,
    query: &SearchQuery,
) -> Result<Vec<SearchLine>, String> {
    if let Some(reference) = query.terms.iter().find(|t| parse_reference(t).is_some()) {
        if query.terms.len() != 1 {
            return Err(format!(
                "{reference} is a reference term; it searches alone"
            ));
        }
        return reference_search(world, reference);
    }
    if query.history {
        history_search(world, rows, boards, query)
    } else {
        current_search(world, boards, query)
    }
}

/// `t-N`, `t-N@rK`, or `t-N@rA..rB`.
#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    pub task: usize,
    pub from: Option<usize>,
    pub to: Option<usize>,
}

pub fn parse_reference(token: &str) -> Option<Reference> {
    let (task_part, rest) = match token.split_once('@') {
        Some((task_part, rest)) => (task_part, Some(rest)),
        None => (token, None),
    };
    let task = task_part.strip_prefix("t-")?.parse().ok()?;
    let Some(rest) = rest else {
        return Some(Reference {
            task,
            from: None,
            to: None,
        });
    };
    if rest.is_empty() {
        return Some(Reference {
            task,
            from: None,
            to: None,
        });
    }
    if let Some((from, to)) = rest.split_once("..") {
        let from = from.strip_prefix('r')?.parse().ok()?;
        let to = to.strip_prefix('r')?.parse().ok()?;
        return Some(Reference {
            task,
            from: Some(from),
            to: Some(to),
        });
    }
    let revision = rest.strip_prefix('r')?.parse().ok()?;
    Some(Reference {
        task,
        from: Some(revision),
        to: Some(revision),
    })
}

fn within_range(within: &Option<String>, task: usize, revision: Option<usize>) -> bool {
    match within {
        None => true,
        Some(spec) => match parse_reference(spec) {
            Some(Reference { task: t, from, to }) if t == task => match (from, to, revision) {
                (None, None, _) => true,
                (Some(f), Some(t), Some(r)) => r >= f && r <= t,
                (Some(f), None, Some(r)) => r >= f,
                _ => revision.is_some(),
            },
            _ => false,
        },
    }
}

fn commit_of(world: &World, task: usize, revision: Option<usize>) -> Option<String> {
    let ctx = world.tasks.get(task)?;
    match revision {
        Some(r) => ctx
            .commits
            .get(r.checked_sub(1)?)
            .map(|c| c.commit.as_str().to_string()),
        None => ctx.commits.last().map(|c| c.commit.as_str().to_string()),
    }
}

fn current_search(
    world: &World,
    boards: &crate::boards::Boards,
    query: &SearchQuery,
) -> Result<Vec<SearchLine>, String> {
    let mut lines = Vec::new();
    for (position, ctx) in world.tasks.iter().enumerate() {
        let Some(commit) = ctx.commits.last() else {
            continue;
        };
        let revision = ctx.commits.len();
        if !within_range(&query.within, position, Some(revision)) {
            continue;
        }
        let text = boards
            .text_at(commit.commit.as_str(), BOARD_FILE_NAME)
            .map_err(|error| error.detail())?;
        let name = ctx.task.name.as_str().to_string();
        let Some(excerpt) = first_excerpt(&text, &query.terms).or_else(|| {
            query
                .terms
                .iter()
                .all(|t| term_matches(&name, t))
                .then(|| format!("task named {}", name))
        }) else {
            continue;
        };
        lines.push(SearchLine::Board {
            task: position,
            name,
            revision,
            excerpt,
        });
    }
    Ok(lines)
}

fn history_search(
    world: &World,
    rows: &[crate::db::StoredRecord],
    boards: &crate::boards::Boards,
    query: &SearchQuery,
) -> Result<Vec<SearchLine>, String> {
    let mut lines = Vec::new();
    for (position, ctx) in world.tasks.iter().enumerate() {
        // within narrows to exactly one task
        if let Some(spec) = &query.within
            && parse_reference(spec).map(|r| r.task) != Some(position)
        {
            continue;
        }
        let mut previous: Option<String> = None;
        for (index, board_commit) in ctx.commits.iter().enumerate() {
            let revision = index + 1;
            if !within_range(&query.within, position, Some(revision)) {
                continue;
            }
            let text = boards
                .text_at(board_commit.commit.as_str(), BOARD_FILE_NAME)
                .map_err(|error| error.detail())?;
            let message = boards
                .commit_message(board_commit.commit.as_str())
                .unwrap_or_default();
            for term in &query.terms {
                let present = term_matches(&text, term);
                let was = previous
                    .as_deref()
                    .map(|p| term_matches(p, term))
                    .unwrap_or(false);
                if present && !was {
                    lines.push(SearchLine::History {
                        task: position,
                        revision: Some(revision),
                        event: "appeared".into(),
                        term: Some(term.clone()),
                        detail: Some(message.clone()),
                    });
                } else if !present && was {
                    lines.push(SearchLine::History {
                        task: position,
                        revision: Some(revision),
                        event: "disappeared".into(),
                        term: Some(term.clone()),
                        detail: Some(message.clone()),
                    });
                }
            }
            previous = Some(text);
        }
        // the decisions ride their revisions: the offer, the reject's
        // grounds, the accept - the history view's grounds half
        for row in rows {
            let Ok(event) = wire::assemble(&row.kind, &row.payload) else {
                continue;
            };
            let (revision, event_name, detail) = match &event {
                crate::Event::TaskOffered { task: id, .. } if id.0 == position => (
                    world.tasks[position].revision_of(RecordId(row.seq)),
                    "offer",
                    None,
                ),
                crate::Event::TaskRejected { task: id, grounds } if id.0 == position => {
                    (None, "reject", Some(grounds.as_str().to_string()))
                }
                crate::Event::TaskAccepted { task: id } if id.0 == position => {
                    (None, "accept", None)
                }
                _ => continue,
            };
            if !within_range(&query.within, position, revision) {
                continue;
            }
            lines.push(SearchLine::History {
                task: position,
                revision,
                event: event_name.into(),
                term: None,
                detail,
            });
        }
    }
    Ok(lines)
}

fn reference_search(world: &World, term: &str) -> Result<Vec<SearchLine>, String> {
    let Some(reference) = parse_reference(term) else {
        return Ok(Vec::new());
    };
    let mut lines = Vec::new();
    // incorporations: every consumer commit citing this task, or this
    // task at exactly the cited revision
    let source_commit = reference
        .from
        .map(|revision| commit_of(world, reference.task, Some(revision)))
        .unwrap_or(None);
    for (consumer, ctx) in world.tasks.iter().enumerate() {
        for (index, board_commit) in ctx.commits.iter().enumerate() {
            for (source, cited) in &board_commit.incorporations {
                if source.0 != reference.task {
                    continue;
                }
                if let Some(want) = &source_commit
                    && cited.as_str() != want
                {
                    continue;
                }
                let source_revision = world
                    .tasks
                    .get(reference.task)
                    .and_then(|src| {
                        src.commits
                            .iter()
                            .position(|c| c.commit.as_str() == cited.as_str())
                    })
                    .map(|p| p + 1);
                lines.push(SearchLine::Reference {
                    line: format!(
                        "t-{consumer}@r{} incorporates t-{}@r{}",
                        index + 1,
                        reference.task,
                        source_revision
                            .map(|r| r.to_string())
                            .unwrap_or_else(|| "?".into())
                    ),
                });
            }
        }
    }
    // messages carrying the task
    for (id, mctx) in &world.messages {
        if mctx.message.task != Some(TaskId(reference.task)) {
            continue;
        }
        let from = world.incarnation_num(mctx.message.from).unwrap_or(0);
        let to = world.incarnation_num(mctx.message.to).unwrap_or(0);
        let num = world.message_num(*id).unwrap_or(0);
        lines.push(SearchLine::Reference {
            line: format!("m-{num} carries t-{} (i-{from} to i-{to})", reference.task),
        });
    }
    Ok(lines)
}

/// The first line carrying every term, word-bounded and case-folded.
fn first_excerpt(text: &str, terms: &[String]) -> Option<String> {
    text.lines()
        .find(|line| terms.iter().all(|t| term_matches(line, t)))
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.chars().count() > 120 {
                format!("{}...", trimmed.chars().take(117).collect::<String>())
            } else {
                trimmed.to_string()
            }
        })
}

/// Exact term match, whole-word-bounded, case-folded - the search
/// law's shape over board text.
pub fn term_matches(text: &str, term: &str) -> bool {
    let lowered = text.to_lowercase();
    let needle = term.to_lowercase();
    let mut start = 0;
    while let Some(found) = lowered[start..].find(&needle) {
        let at = start + found;
        let end = at + needle.len();
        let before_ok = lowered[..at]
            .chars()
            .next_back()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(true);
        let after_ok = lowered[end..]
            .chars()
            .next()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(true);
        if before_ok && after_ok {
            return true;
        }
        start = at + 1;
    }
    false
}
