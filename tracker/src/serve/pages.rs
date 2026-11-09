//! The /view/v1 pages: the instruments' read surface. A page belongs to
//! a location, not an endpoint; pages are server-shaped hypermedia
//! documents in a small versioned view grammar, and every ref is a
//! location. The pages are pure record: they read the same fold and
//! board repository as every other read face, and nothing here watches
//! the host. The settled shapes are t-24@r2's answers beside t-14
//! "Pages, the instruments' read surface"; the framework items - the
//! envelope, the part kinds, refusals, the wait loop, and the dispatch
//! match - are what later page families extend.

use std::time::{Duration, Instant};

use axum::extract::{Path, RawQuery, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::api::{AppState, ReadFail, Snapshot};
use crate::boards::BoardsFail;
use crate::db::StoredRecord;
use crate::objects::incarnation::Incarnation;
use crate::objects::incarnation::IncarnationId;
use crate::objects::task::{Owner, TaskId, TaskState};
use crate::record::views::{self, Target, WaitUntil};
use crate::serve::board_wire::{HistoryEntry, board_history};
use crate::store::{RecordId, World};
use crate::wire;
use board_grammar::{Component, SelectPath};

/// The view grammar major this surface serves; the path's `v1` names
/// it, and every document repeats it.
pub const VIEW_MAJOR: u64 = 1;

/// The host machine id's home: the root page reads it here, so a
/// client can prove which hub a page came from (t-24@r14).
const MACHINE_ID_PATH: &str = "/etc/machine-id";

/// The host machine id: `/etc/machine-id`, trimmed. Missing, empty or
/// unreadable is `None` - never defaulted, so two machines can never
/// byte-equal on an empty id. The path is injectable: the suite pins
/// the reader without touching the host.
fn machine_id(path: &str) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// The hold a `?after=` request gets without `?timeout=`, and the
/// ceiling any client may ask for.
const DEFAULT_HOLD: Duration = Duration::from_secs(25);
const MAX_HOLD: Duration = Duration::from_secs(55);

// -- the page document --------------------------------------------------

/// The view grammar's major, carried by every page.
#[derive(Serialize)]
struct ViewMajor {
    major: u64,
}

/// The board grammar range the root page names.
#[derive(Serialize)]
struct GrammarRange {
    min: u64,
    max: u64,
}

/// A page document: the view major, the page kind, its location, the
/// fold cursor it was computed at, and the kind's own body. A refusal
/// is a document too; only the file family answers with raw bytes.
#[derive(Serialize)]
struct PageDoc {
    view: ViewMajor,
    kind: &'static str,
    location: String,
    seq: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tracker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    view_majors: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    board_grammar: Option<GrammarRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    machine_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    revision: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Vec<Part>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    entries: Option<Vec<serde_json::Value>>,
}

/// One view-grammar component. v1 serves six kinds (t-24@r2): text,
/// group, fact, time, ref, figure. Two location rules hold across
/// every family: a part's own `location` names what it is in the
/// tracker's addressing grammar (a wrapped board part is
/// `t-N@rK:/content/...`), and a ref's `location` names where to go in
/// the page-location grammar - the route tail after `/view/v1/`.
#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Part {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        location: Option<String>,
    },
    Group {
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        content: Vec<Part>,
        #[serde(skip_serializing_if = "Option::is_none")]
        location: Option<String>,
    },
    Fact {
        label: String,
        value: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        location: Option<String>,
    },
    /// a v1 grammar kind (t-24@r2) no page of step 1a emits: the
    /// pipeline page's rows carry the moments
    #[allow(dead_code)]
    Time {
        at: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        location: Option<String>,
    },
    Ref {
        location: String,
        label: String,
    },
    Figure {
        caption: String,
        file: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        location: Option<String>,
    },
}

fn fact(label: &str, value: impl Into<String>) -> Part {
    Part::Fact {
        label: label.to_string(),
        value: value.into(),
        location: None,
    }
}

fn page_ref(location: impl Into<String>, label: impl Into<String>) -> Part {
    Part::Ref {
        location: location.into(),
        label: label.into(),
    }
}

/// A refusal's body: a fact naming the refusal, a text naming what
/// would make it answerable.
fn refusal_doc(location: &str, code: &str, detail: &str, seq: Option<usize>) -> PageDoc {
    PageDoc {
        view: ViewMajor { major: VIEW_MAJOR },
        kind: "refusal",
        machine_id: None,
        repo_path: None,
        location: location.to_string(),
        seq,
        tracker: None,
        view_majors: None,
        board_grammar: None,
        code: Some(code.to_string()),
        task: None,
        revision: None,
        content: Some(vec![
            fact("refusal", code),
            Part::Text {
                text: detail.to_string(),
                location: None,
            },
        ]),
        entries: None,
    }
}

fn serve_doc(doc: &PageDoc, status: StatusCode) -> Response {
    let body = serde_json::to_string(doc).expect("pages are plain data");
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

fn serve_refusal(location: &str, fail: &PageFail, seq: Option<usize>) -> Response {
    let doc = refusal_doc(location, fail.code, &fail.detail, seq);
    serve_doc(&doc, fail.status)
}

/// A read-side failure of a page: status, machine code, teaching text.
#[derive(Debug)]
struct PageFail {
    status: StatusCode,
    code: &'static str,
    detail: String,
}

impl PageFail {
    fn new(status: StatusCode, code: &'static str, detail: impl Into<String>) -> Self {
        PageFail {
            status,
            code,
            detail: detail.into(),
        }
    }
}

impl From<BoardsFail> for PageFail {
    fn from(fail: BoardsFail) -> Self {
        // a stored board or commit this build cannot read is version
        // skew or the repository's own break - never the caller's
        // input, always the server's
        PageFail::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            fail.code(),
            fail.detail(),
        )
    }
}

// -- the location grammar -----------------------------------------------

/// What a route tail after `/view/v1/` names. The grammar v1 serves:
/// `` (the root), `t-N[@rK]`, `t-N[@rK]/history`, `t-N@rA..rB/diff`,
/// `t-N[@rK]/file/<path>`, `t-N@rK/content/<select-rest>`. A later
/// family adds its arm here.
#[derive(Debug)]
enum PageLocation {
    Root,
    Agents,
    Board {
        task: usize,
        revision: Option<usize>,
    },
    History {
        task: usize,
        revision: Option<usize>,
    },
    Diff {
        task: usize,
        from: usize,
        to: usize,
    },
    File {
        task: usize,
        revision: Option<usize>,
        path: String,
    },
    Component {
        task: usize,
        revision: usize,
        path: SelectPath,
    },
}

fn parse_task_spec(token: &str) -> Result<(usize, Option<usize>), PageFail> {
    let unknown = || {
        PageFail::new(
            StatusCode::BAD_REQUEST,
            "unknown_location",
            format!(
                "'{token}' is not a page location; the v1 grammar serves t-N[@rK], \
                 t-N[@rK]/history, t-N@rA..rB/diff, t-N[@rK]/file/<path> and \
                 t-N@rK/content/<select path>"
            ),
        )
    };
    let (head, revision) = match token.split_once('@') {
        Some((head, revision)) => (head, Some(revision)),
        None => (token, None),
    };
    let task = head
        .strip_prefix("t-")
        .and_then(|digits| digits.parse::<usize>().ok())
        .ok_or_else(unknown)?;
    let revision = match revision {
        None => None,
        Some(spec) => Some(
            spec.strip_prefix('r')
                .and_then(|digits| digits.parse::<usize>().ok())
                .ok_or_else(unknown)?,
        ),
    };
    Ok((task, revision))
}

/// `t-N@rA..rB`, the diff page's address: one task, two revisions, A
/// before B.
fn parse_range(token: &str) -> Result<(usize, usize, usize), PageFail> {
    let bad = |detail: String| {
        PageFail::new(
            StatusCode::BAD_REQUEST,
            "bad_range",
            format!("{detail}; a diff names one task and two revisions, like t-4@r1..r2/diff"),
        )
    };
    let (head, range) = token
        .split_once('@')
        .ok_or_else(|| bad(format!("'{token}' names no revision range")))?;
    let (from, to) = range
        .split_once("..")
        .ok_or_else(|| bad(format!("'{range}' is not a revision range")))?;
    let parse_rev = |value: &str| {
        value
            .strip_prefix('r')
            .and_then(|digits| digits.parse::<usize>().ok())
            .ok_or_else(|| bad(format!("'{value}' is not a revision")))
    };
    let from = parse_rev(from)?;
    let to = parse_rev(to)?;
    if from >= to {
        return Err(bad(format!("r{from} does not precede r{to}")));
    }
    let task = head
        .strip_prefix("t-")
        .and_then(|digits| digits.parse::<usize>().ok())
        .ok_or_else(|| bad(format!("'{head}' is not a task id")))?;
    Ok((task, from, to))
}

fn parse_location(raw: &str) -> Result<PageLocation, PageFail> {
    let location = raw.trim_matches('/');
    if location.is_empty() {
        return Ok(PageLocation::Root);
    }
    if location == "agents" {
        return Ok(PageLocation::Agents);
    }
    // the file family parses first: a tree file may be named
    // `history` or `diff`, and the /file/ marker is the route's own
    if let Some((head, path)) = location.split_once("/file/") {
        let (task, revision) = parse_task_spec(head)?;
        if path.is_empty() || path.ends_with('/') {
            return Err(PageFail::new(
                StatusCode::BAD_REQUEST,
                "bad_file_path",
                format!("'{path}' is not a file in the board tree"),
            ));
        }
        let safe = std::path::Path::new(path)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)));
        if !safe {
            return Err(PageFail::new(
                StatusCode::BAD_REQUEST,
                "bad_file_path",
                format!("'{path}' is not a safe relative board path"),
            ));
        }
        return Ok(PageLocation::File {
            task,
            revision,
            path: path.to_string(),
        });
    }
    if let Some(prefix) = location.strip_suffix("/diff") {
        let (task, from, to) = parse_range(prefix)?;
        return Ok(PageLocation::Diff { task, from, to });
    }
    if let Some(prefix) = location.strip_suffix("/history") {
        let (task, revision) = parse_task_spec(prefix)?;
        return Ok(PageLocation::History { task, revision });
    }
    if let Some((head, rest)) = location.split_once("/content/") {
        let (task, revision) = parse_task_spec(head)?;
        // the select family names a pinned revision: a component
        // citation must be stable, so `t-N/content/...` is not a route
        let Some(revision) = revision else {
            return Err(PageFail::new(
                StatusCode::BAD_REQUEST,
                "unknown_location",
                format!(
                    "'{location}' names no revision; a component page pins one, \
                     like t-{task}@r1/content/0"
                ),
            ));
        };
        let path = SelectPath::parse(&format!("/content/{rest}")).map_err(|error| {
            PageFail::new(
                StatusCode::BAD_REQUEST,
                "bad_select_path",
                error.to_string(),
            )
        })?;
        return Ok(PageLocation::Component {
            task,
            revision,
            path,
        });
    }
    if location.ends_with("/content") {
        return Err(PageFail::new(
            StatusCode::BAD_REQUEST,
            "bad_select_path",
            format!(
                "'{location}' stops before naming a component; a select path carries \
                 an index, like /content/0"
            ),
        ));
    }
    let (task, revision) = parse_task_spec(location)?;
    Ok(PageLocation::Board { task, revision })
}

// -- the wait -----------------------------------------------------------

/// What a parked wait watches: the root wakes on any record, a page of
/// a task wakes on the records the fold attributes to that task (the
/// wait filter's activity set). Later families extend this enum with
/// their own relevance; the park loop only re-checks.
#[derive(Clone, Copy)]
enum Watch {
    Any,
    Task(usize),
    /// The agents page's subject: the registration family and the
    /// claims that move ownership. A landing whose kind is in the set
    /// can change who the sessions are, where their terminals live, or
    /// what each owns - the whole page's content.
    Kinds(&'static [&'static str]),
}

enum Step {
    Serve(Response),
    Park,
}

/// Whether the record a wait watched for landed: assembled and matched
/// against the watch. Root needs no assembly.
fn relevant(world: &World, watch: &Watch, row: &StoredRecord) -> bool {
    match watch {
        Watch::Any => true,
        Watch::Kinds(kinds) => kinds.contains(&row.kind.as_str()),
        Watch::Task(task) => {
            let Ok(event) = wire::assemble(&row.kind, &row.payload) else {
                return false;
            };
            views::wait_matches(
                world,
                RecordId(row.seq),
                &event,
                &Target::Task(TaskId(*task)),
                WaitUntil::Activity,
            )
        }
    }
}

/// A build failure that may still name a watch: a page that refuses
/// today may answer tomorrow, so its wait keeps watching.
struct BuildFail {
    fail: PageFail,
    watch: Option<Watch>,
}

impl From<PageFail> for BuildFail {
    fn from(fail: PageFail) -> Self {
        BuildFail { fail, watch: None }
    }
}

impl From<BoardsFail> for BuildFail {
    fn from(fail: BoardsFail) -> Self {
        BuildFail {
            fail: fail.into(),
            watch: None,
        }
    }
}

/// What a built page serves: a document, or the file family's raw
/// bytes. The watch rides both, so a waiting client parks on the page's
/// subject whatever the answer says today.
enum Built {
    Doc(PageDoc, StatusCode, Watch),
    File(Response, Watch),
}

/// The watch a location carries, from its shape alone: a task page
/// watches its task whatever the fold answers today, the root watches
/// everything.
fn watch_of(location: &PageLocation) -> Watch {
    match location {
        PageLocation::Root => Watch::Any,
        PageLocation::Agents => Watch::Kinds(AGENTS_WATCH),
        PageLocation::Board { task, .. }
        | PageLocation::History { task, .. }
        | PageLocation::Diff { task, .. }
        | PageLocation::File { task, .. }
        | PageLocation::Component { task, .. } => Watch::Task(*task),
    }
}

/// One evaluation: build the page as the fold stands, then either
/// serve it or park on its watch. A `?after=` cursor a relevant record
/// already passed serves at once; anything else parks.
fn evaluate(app: &AppState, location: &str, after: Option<u64>) -> Step {
    let snapshot = match app.snapshot() {
        Ok(snapshot) => snapshot,
        Err(ReadFail::Degraded(degraded)) => {
            let doc = refusal_doc(
                location,
                "degraded",
                &degraded.reason,
                degraded.rows.last().map(|row| row.seq),
            );
            return Step::Serve(serve_doc(&doc, StatusCode::SERVICE_UNAVAILABLE));
        }
        Err(ReadFail::Db(reason)) => {
            let doc = refusal_doc(location, "storage", &reason, None);
            return Step::Serve(serve_doc(&doc, StatusCode::INTERNAL_SERVER_ERROR));
        }
    };
    let seq = snapshot.rows.last().map(|row| row.seq);
    let (watch, step) = match build(app, &snapshot, location) {
        Ok(Built::Doc(doc, status, watch)) => (Some(watch), Step::Serve(serve_doc(&doc, status))),
        Ok(Built::File(response, watch)) => (Some(watch), Step::Serve(response)),
        Err(BuildFail { fail, watch }) => (watch, Step::Serve(serve_refusal(location, &fail, seq))),
    };
    let Some(after) = after else {
        return step;
    };
    let Some(watch) = watch else {
        // a location that names no watchable subject cannot wait
        return step;
    };
    let passed = snapshot
        .rows
        .iter()
        .rev()
        .take_while(|row| row.seq > after as usize)
        .any(|row| relevant(&snapshot.world, &watch, row));
    if passed { step } else { Step::Park }
}

/// Build the page the fold now carries.
fn build(app: &AppState, snapshot: &Snapshot, location: &str) -> Result<Built, BuildFail> {
    let parsed = parse_location(location).map_err(BuildFail::from)?;
    let watch = watch_of(&parsed);
    match parsed {
        PageLocation::Root => {
            let doc = root_page(app, snapshot);
            Ok(Built::Doc(doc, StatusCode::OK, watch))
        }
        PageLocation::Agents => {
            let doc = agents_page(snapshot);
            Ok(Built::Doc(doc, StatusCode::OK, watch))
        }
        PageLocation::Board { task, revision } => {
            let doc = board_page(app, snapshot, task, revision)?;
            Ok(Built::Doc(doc, StatusCode::OK, watch))
        }
        PageLocation::History { task, revision } => {
            let doc = history_page(app, snapshot, task, revision)?;
            Ok(Built::Doc(doc, StatusCode::OK, watch))
        }
        PageLocation::Diff { task, from, to } => {
            let doc = diff_page(app, snapshot, task, from, to)?;
            Ok(Built::Doc(doc, StatusCode::OK, watch))
        }
        PageLocation::File {
            task,
            revision,
            path,
        } => match file_bytes_of(app, snapshot, task, revision, &path) {
            Ok(bytes) => Ok(Built::File(raw_bytes(&path, bytes), watch)),
            Err(fail) => Err(BuildFail {
                fail,
                watch: Some(watch),
            }),
        },
        PageLocation::Component {
            task,
            revision,
            path,
        } => {
            let doc = component_page(app, snapshot, task, revision, &path)?;
            Ok(Built::Doc(doc, StatusCode::OK, watch))
        }
    }
}

// -- the page builders --------------------------------------------------

/// The root page: what an instrument reads first, where it refuses one
/// project rather than failing.
fn root_page(app: &AppState, snapshot: &Snapshot) -> PageDoc {
    PageDoc {
        view: ViewMajor { major: VIEW_MAJOR },
        kind: "root",
        location: "/".to_string(),
        seq: cursor(snapshot),
        tracker: app.repo_root().map(crate::paths::repo_basename),
        view_majors: Some(vec![VIEW_MAJOR]),
        board_grammar: Some(GrammarRange {
            min: *board_grammar::SUPPORTED_GRAMMAR_VERSIONS.start(),
            max: *board_grammar::SUPPORTED_GRAMMAR_VERSIONS.end(),
        }),
        machine_id: machine_id(MACHINE_ID_PATH),
        repo_path: app.repo_root().map(|path| path.display().to_string()),
        code: None,
        task: None,
        revision: None,
        content: Some(families_parts()),
        entries: None,
    }
}

fn cursor(snapshot: &Snapshot) -> Option<usize> {
    snapshot.rows.last().map(|row| row.seq)
}

/// The record kinds whose landing can change the agents page: the
/// registration family (who the sessions are, where their terminals
/// and saved sessions live) and the claims that move ownership. The
/// page's watch is exactly this set, so a parked hold wakes when any
/// of them lands - and stays quiet through unrelated records.
const AGENTS_WATCH: &[&str] = &[
    "incarnation_registered",
    "saved_session_bound",
    "host_address_assigned",
    "task_claimed",
];

/// The tasks the incarnation currently owns, latest claim first: the
/// record's claims are whole-task responsibility, so the latest claim
/// names the task and the count of the rest rides the companion's
/// title, not the page's facts.
fn owned_tasks(world: &World, id: IncarnationId) -> Vec<usize> {
    let mut owned: Vec<(RecordId, usize)> = world
        .tasks
        .iter()
        .enumerate()
        .filter_map(|(index, ctx)| {
            views::current_owner(ctx).and_then(|entry| match entry.owner {
                Owner::Incarnation(owned) if owned == id => Some((entry.record, index)),
                _ => None,
            })
        })
        .collect();
    owned.sort_by_key(|(record, _)| std::cmp::Reverse(*record));
    owned.into_iter().map(|(_, index)| index).collect()
}

/// The agents page (t-24@r4): one group per registered native session,
/// its location the incarnation, facts inside naming the terminal
/// address, the harness, the owned tasks and the workspace. The
/// workspace derives here, server-side, from the saved session file
/// with the tracker's own header readers - the Pi header's cwd, the
/// first cwd-bearing Claude Code transcript entry - never from Claude
/// Code's project directory name. Pure record: no Herdr data, no
/// liveness - a session that never saved, or whose header cannot be
/// read, carries no workspace fact until it does.
fn agents_page(snapshot: &Snapshot) -> PageDoc {
    let world = &snapshot.world;
    let mut content: Vec<Part> = Vec::new();
    for (id, ctx) in &world.incarnations {
        let Incarnation::Native { .. } = &ctx.incarnation else {
            continue;
        };
        // a superseded holder no longer carries the address; the
        // holder the fold names now is the one the page lists
        if world.superseded.contains_key(id) {
            continue;
        }
        let incarnation = format!(
            "i-{}",
            world
                .incarnation_num(*id)
                .expect("an iteration over the fold's own incarnations is resident",)
        );
        let mut facts = vec![fact("incarnation", incarnation.clone())];
        if let Some(address) = ctx.incarnation.address() {
            facts.push(fact("terminal address", address.to_string()));
        }
        if let Some(harness) = ctx.incarnation.harness() {
            facts.push(fact("harness", harness.as_str().to_string()));
        }
        let owned = owned_tasks(world, *id);
        if !owned.is_empty() {
            facts.push(fact(
                "tasks",
                owned
                    .iter()
                    .map(|task| format!("t-{task}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            ));
        }
        if let Some(harness) = ctx.incarnation.harness()
            && let Some(locator) = ctx.incarnation.saved_session()
            && let Ok(header) = harness.read_header(locator.as_path())
        {
            facts.push(fact("workspace", header.cwd));
        }
        content.push(Part::Group {
            title: None,
            content: facts,
            location: Some(incarnation),
        });
    }
    PageDoc {
        view: ViewMajor { major: VIEW_MAJOR },
        kind: "document",
        location: "agents".to_string(),
        seq: cursor(snapshot),
        tracker: None,
        view_majors: None,
        board_grammar: None,
        machine_id: None,
        repo_path: None,
        code: None,
        task: None,
        revision: None,
        content: Some(content),
        entries: None,
    }
}

/// The page families this tracker serves, as the root names them. A
/// later family adds its fact here.
fn families_parts() -> Vec<Part> {
    vec![
        fact("family", "boards: /view/v1/t-N and /view/v1/t-N@rK"),
        fact(
            "family",
            "history: /view/v1/t-N/history and /view/v1/t-N@rK/history",
        ),
        fact("family", "diff: /view/v1/t-N@rA..rB/diff"),
        fact("family", "files: /view/v1/t-N[@rK]/file/<path>"),
        fact(
            "family",
            "components: /view/v1/t-N@rK/content/<select path>",
        ),
        fact("family", "agents: /view/v1/agents"),
        Part::Text {
            text: "every ref in a page names a location in these forms; \
                   waiting is ?after=<seq> on any route, ?timeout= holds \
                   up to 55s, quiet is a normal answer."
                .to_string(),
            location: None,
        },
    ]
}

/// The task context a page needs: exists, holds a board, the revision
/// resolves. The refusal keeps the task's watch so a waiting client
/// parks through a missing board or revision.
fn task_revision(
    snapshot: &Snapshot,
    task: usize,
    revision: Option<usize>,
) -> Result<(usize, usize), BuildFail> {
    let Some(ctx) = snapshot.world.tasks.get(task) else {
        return Err(BuildFail {
            fail: PageFail::new(
                StatusCode::NOT_FOUND,
                "invalid_task_id",
                format!(
                    "no task t-{task}; the fold holds {} tasks",
                    snapshot.world.tasks.len()
                ),
            ),
            watch: Some(Watch::Task(task)),
        });
    };
    let current = ctx.commits.len();
    if current == 0 {
        return Err(BuildFail {
            fail: PageFail::new(
                StatusCode::NOT_FOUND,
                "board_missing",
                format!("task t-{task} has no committed board"),
            ),
            watch: Some(Watch::Task(task)),
        });
    }
    let revision = revision.unwrap_or(current);
    if revision == 0 || revision > current {
        return Err(BuildFail {
            fail: PageFail::new(
                StatusCode::NOT_FOUND,
                "revision_missing",
                format!("t-{task} has no revision r{revision}"),
            ),
            watch: Some(Watch::Task(task)),
        });
    }
    Ok((current, revision))
}

fn boards_of(app: &AppState) -> Result<crate::boards::Boards, PageFail> {
    let root = app.repo_root().ok_or_else(|| {
        PageFail::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "no_repo",
            "the server resolved no repo root at boot; it cannot read boards",
        )
    })?;
    crate::boards::Boards::open(&crate::paths::state_dir(root)).map_err(BoardsFail::into)
}

/// The board page: the task's facts, its authored content wrapped in
/// the view grammar, every authored part carrying its own location.
fn board_page(
    app: &AppState,
    snapshot: &Snapshot,
    task: usize,
    requested: Option<usize>,
) -> Result<PageDoc, BuildFail> {
    let (current, revision) = task_revision(snapshot, task, requested)?;
    let ctx = &snapshot.world.tasks[task];
    let commit = ctx.commits[revision - 1].commit.as_str().to_string();
    let boards = boards_of(app)?;
    let board = boards.read_board_at(&commit)?;
    let pinned = format!("t-{task}@r{revision}");
    let mut content = vec![
        fact("task", format!("t-{task}")),
        fact("name", ctx.task.name.as_str()),
        fact("state", views::state_str(&ctx.task.state)),
    ];
    if let TaskState::Cancelled(why) = &ctx.task.state {
        content.push(fact("explanation", why.as_str()));
    }
    if let Some(owner) = views::current_owner(ctx) {
        content.push(fact("owner", owner_string(&snapshot.world, &owner.owner)));
    }
    content.push(fact("revision", format!("r{revision} of {current}")));
    content.push(fact("commit", &commit));
    content.push(fact(
        "board grammar",
        board_grammar::SUPPORTED_GRAMMAR_VERSIONS
            .start()
            .to_string(),
    ));
    content.push(page_ref(
        if requested.is_some() {
            format!("{pinned}/history")
        } else {
            format!("t-{task}/history")
        },
        "history",
    ));
    content.push(page_ref(format!("{pinned}/file/board.yaml"), "board.yaml"));
    if revision != current {
        content.push(page_ref(
            format!("t-{task}"),
            format!("current revision r{current}"),
        ));
    }
    for (index, component) in board.content.iter().enumerate() {
        content.push(part_of(
            component,
            &format!("/content/{index}"),
            task,
            revision,
        ));
    }
    Ok(PageDoc {
        view: ViewMajor { major: VIEW_MAJOR },
        kind: "document",
        location: pinned,
        seq: cursor(snapshot),
        tracker: None,
        view_majors: None,
        board_grammar: None,
        machine_id: None,
        repo_path: None,
        code: None,
        task: None,
        revision: None,
        content: Some(content),
        entries: None,
    })
}

/// The owner an ownership entry names, as the pages render it: the
/// human or external actor's name, the incarnation's public number.
fn owner_string(world: &World, owner: &Owner) -> String {
    match owner {
        Owner::Human(actor) | Owner::External(actor) => actor.as_str().to_string(),
        Owner::Incarnation(id) => format!(
            "i-{}",
            world
                .incarnation_num(*id)
                .expect("an owner is a resident incarnation")
        ),
    }
}

/// One authored component as a view part, with its own location in
/// the tracker addressing form (`t-N@rK:/content/...`, the path named
/// by the select grammar). An annotation rides a titled group (v1's
/// kinds have no annotation); an incorporation rides its collapsed
/// shape: the source revision, the receiving text, and a ref per
/// cited component.
fn part_of(component: &Component, path: &str, task: usize, revision: usize) -> Part {
    let location = format!("t-{task}@r{revision}:{path}");
    match component {
        Component::Text { text } => Part::Text {
            text: text.clone(),
            location: Some(location),
        },
        Component::Group { title, content } => {
            let children = content
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    part_of(
                        child,
                        &format!("{path}/group/content/{index}"),
                        task,
                        revision,
                    )
                })
                .collect();
            Part::Group {
                title: title.clone(),
                content: children,
                location: Some(location),
            }
        }
        Component::Figure { input, caption } => Part::Figure {
            caption: caption.clone(),
            file: format!("t-{task}@r{revision}/file/{input}"),
            location: Some(location),
        },
        Component::Incorporation { source, content } => {
            let mut children = vec![page_ref(
                format!("{}@r{}", source.task, source.revision),
                format!("source revision {}@r{}", source.task, source.revision),
            )];
            children.extend(content.iter().enumerate().map(|(index, child)| {
                part_of(
                    child,
                    &format!("{path}/incorporation/content/{index}"),
                    task,
                    revision,
                )
            }));
            children.extend(source.select.iter().map(|path| {
                page_ref(
                    format!("{}@r{}{}", source.task, source.revision, path),
                    format!("cites {path}"),
                )
            }));
            Part::Group {
                title: Some(format!("incorporates {}@r{}", source.task, source.revision)),
                content: children,
                location: Some(location),
            }
        }
        Component::Annotation { text } => Part::Group {
            title: Some("annotation".to_string()),
            content: vec![Part::Text {
                text: text.clone(),
                location: Some(location.clone()),
            }],
            location: Some(location),
        },
    }
}

/// The history page: one timeline of commits interleaved with the
/// decisions, in record order. A pinned revision cuts the timeline at
/// the commit that landed it: the page answers how that revision came
/// to be.
fn history_page(
    app: &AppState,
    snapshot: &Snapshot,
    task: usize,
    requested: Option<usize>,
) -> Result<PageDoc, BuildFail> {
    let (_current, revision) = task_revision(snapshot, task, requested)?;
    let ctx = &snapshot.world.tasks[task];
    let boards = boards_of(app)?;
    let entries =
        board_history(&snapshot.world, &snapshot.rows, &boards, task).map_err(|error| {
            BuildFail {
                fail: PageFail::new(StatusCode::INTERNAL_SERVER_ERROR, "storage", error),
                watch: Some(Watch::Task(task)),
            }
        })?;
    let cut = ctx.commits[revision - 1].record.0;
    let location = if requested.is_some() {
        format!("t-{task}@r{revision}/history")
    } else {
        format!("t-{task}/history")
    };
    // only a pinned revision cuts the timeline at its commit; the
    // current history is the whole timeline, the decisions that
    // landed after the last commit included
    let entries: Vec<serde_json::Value> = entries
        .into_iter()
        .filter(|entry| match requested {
            None => true,
            Some(_) => entry.seq <= cut,
        })
        .map(|entry| entry_value(&entry, task))
        .collect();
    Ok(PageDoc {
        view: ViewMajor { major: VIEW_MAJOR },
        kind: "timeline",
        location,
        seq: cursor(snapshot),
        tracker: None,
        view_majors: None,
        board_grammar: None,
        machine_id: None,
        repo_path: None,
        code: None,
        task: Some(format!("t-{task}")),
        revision: Some(revision),
        content: None,
        entries: Some(entries),
    })
}

/// One timeline entry: the history facts, the session behind the
/// record (the incarnation from its recorded scope, when it names one),
/// and the page refs the entry answers for - its revision and the diff
/// from the revision before it.
fn entry_value(entry: &HistoryEntry, task: usize) -> serde_json::Value {
    let mut value = serde_json::to_value(entry).expect("history entries are plain data");
    value["session"] = match &entry.session {
        Some(session) => serde_json::json!({
            "kind": "ref",
            "location": session,
            "label": format!("session {session}"),
        }),
        None => serde_json::Value::Null,
    };
    let mut refs = Vec::new();
    if let Some(revision) = entry.revision {
        refs.push(serde_json::json!({
            "kind": "ref",
            "location": format!("t-{task}@r{revision}"),
            "label": format!("revision r{revision}"),
        }));
        if revision >= 2 {
            refs.push(serde_json::json!({
                "kind": "ref",
                "location": format!("t-{task}@r{}..r{revision}/diff", revision - 1),
                "label": format!("diff r{}..r{revision}", revision - 1),
            }));
        }
    }
    value["refs"] = serde_json::Value::Array(refs);
    value
}

/// The diff page: the stored unified diff between two revisions, one
/// text part. The viewer's structure-aware word diff stays client-side,
/// computed from the two revisions.
fn diff_page(
    app: &AppState,
    snapshot: &Snapshot,
    task: usize,
    from: usize,
    to: usize,
) -> Result<PageDoc, BuildFail> {
    let (_, from) = task_revision(snapshot, task, Some(from))?;
    let (_, to) = task_revision(snapshot, task, Some(to))?;
    let ctx = &snapshot.world.tasks[task];
    let at = |revision: usize| ctx.commits[revision - 1].commit.as_str().to_string();
    let boards = boards_of(app)?;
    let diff = boards.diff(&at(from), &at(to))?;
    Ok(PageDoc {
        view: ViewMajor { major: VIEW_MAJOR },
        kind: "document",
        location: format!("t-{task}@r{from}..r{to}/diff"),
        seq: cursor(snapshot),
        tracker: None,
        view_majors: None,
        board_grammar: None,
        machine_id: None,
        repo_path: None,
        code: None,
        task: None,
        revision: None,
        content: Some(vec![Part::Text {
            text: diff,
            location: None,
        }]),
        entries: None,
    })
}

/// The component page: the one component a select path names, carrying
/// its own location. Revision-pinned, so immutable.
fn component_page(
    app: &AppState,
    snapshot: &Snapshot,
    task: usize,
    revision: usize,
    path: &SelectPath,
) -> Result<PageDoc, BuildFail> {
    let (_, revision) = task_revision(snapshot, task, Some(revision))?;
    let ctx = &snapshot.world.tasks[task];
    let commit = ctx.commits[revision - 1].commit.as_str().to_string();
    let boards = boards_of(app)?;
    let board = boards.read_board_at(&commit)?;
    let component = board_grammar::select(&board, path).map_err(|error| {
        let (status, code) = match error {
            board_grammar::SelectError::OutOfBounds { .. } => {
                (StatusCode::NOT_FOUND, "component_missing")
            }
            board_grammar::SelectError::KindMismatch { .. } => {
                (StatusCode::BAD_REQUEST, "bad_select_path")
            }
        };
        BuildFail {
            fail: PageFail::new(status, code, error.to_string()),
            watch: Some(Watch::Task(task)),
        }
    })?;
    let location = path.to_string();
    Ok(PageDoc {
        view: ViewMajor { major: VIEW_MAJOR },
        kind: "document",
        location: format!("t-{task}@r{revision}{location}"),
        seq: cursor(snapshot),
        tracker: None,
        view_majors: None,
        board_grammar: None,
        machine_id: None,
        repo_path: None,
        code: None,
        task: None,
        revision: None,
        content: Some(vec![part_of(component, &location, task, revision)]),
        entries: None,
    })
}

/// The file family's bytes: one file of a stored revision.
fn file_bytes_of(
    app: &AppState,
    snapshot: &Snapshot,
    task: usize,
    revision: Option<usize>,
    path: &str,
) -> Result<Vec<u8>, PageFail> {
    let (_, revision) = task_revision(snapshot, task, revision).map_err(|fail| fail.fail)?;
    let ctx = &snapshot.world.tasks[task];
    let commit = ctx.commits[revision - 1].commit.as_str().to_string();
    let boards = boards_of(app)?;
    boards
        .file_bytes(&commit, path)
        .map_err(PageFail::from)?
        .ok_or_else(|| {
            PageFail::new(
                StatusCode::NOT_FOUND,
                "file_missing",
                format!("{path} is not in t-{task}@r{revision}"),
            )
        })
}

/// The content type a board file answers as: the kinds the viewer
/// renders, text for the authored sources, octet streams otherwise.
fn content_type_of(path: &str) -> &'static str {
    let extension = path.rsplit('.').next().unwrap_or("");
    match extension {
        "yaml" | "yml" => "text/yaml",
        "md" => "text/markdown",
        "txt" => "text/plain",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

fn raw_bytes(path: &str, bytes: Vec<u8>) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, content_type_of(path))],
        bytes,
    )
        .into_response()
}

// -- the doors ----------------------------------------------------------

/// The query string a page accepts: `?after=<seq>` waits, `?timeout=`
/// bounds the hold. A malformed value refuses as a document.
fn parse_query(raw: Option<&str>) -> Result<(Option<u64>, Duration), PageFail> {
    let Some(raw) = raw else {
        return Ok((None, Duration::ZERO));
    };
    let mut after = None;
    let mut timeout = None;
    for pair in raw.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        match key {
            "after" => match value.parse::<u64>() {
                Ok(parsed) => after = Some(parsed),
                Err(_) => {
                    return Err(PageFail::new(
                        StatusCode::BAD_REQUEST,
                        "bad_query",
                        format!("'{value}' is not a record sequence; after= names a cursor"),
                    ));
                }
            },
            "timeout" => match value.parse::<u64>() {
                Ok(parsed) => timeout = Some(parsed),
                Err(_) => {
                    return Err(PageFail::new(
                        StatusCode::BAD_REQUEST,
                        "bad_query",
                        format!("'{value}' is not a seconds bound; timeout= names seconds"),
                    ));
                }
            },
            _ => {}
        }
    }
    let hold = match after {
        None => Duration::ZERO,
        Some(_) => timeout
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_HOLD)
            .min(MAX_HOLD),
    };
    Ok((after, hold))
}

/// The page door: build, park on the landing wake until a relevant
/// record lands past the cursor or the hold expires, answer the page
/// as it stands. Quiet is a normal answer. The hold is wake-driven:
/// the registration enters the wake before the deciding read runs, so
/// a write landing between the read and the park wakes the hold, and
/// the wait's only timer is the deadline itself.
async fn serve_page(app: &AppState, location: &str, raw_query: Option<&str>) -> Response {
    let (after, hold) = match parse_query(raw_query) {
        Ok(parsed) => parsed,
        Err(fail) => return serve_refusal(location, &fail, None),
    };
    let deadline = Instant::now() + hold;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let served = app
            .landing
            .hold(
                || match evaluate(app, location, after) {
                    Step::Serve(response) => Some(response),
                    Step::Park => None,
                },
                remaining,
            )
            .await;
        if let Some(response) = served {
            return response;
        }
        if Instant::now() >= deadline {
            // the hold expired quiet: answer the page as it stands
            return match evaluate(app, location, None) {
                Step::Serve(response) => response,
                Step::Park => unreachable!("a wait-free evaluation serves"),
            };
        }
    }
}

pub async fn view_page(
    State(app): State<AppState>,
    Path(location): Path<String>,
    RawQuery(query): RawQuery,
) -> Response {
    serve_page(&app, &location, query.as_deref()).await
}

pub async fn view_root(State(app): State<AppState>, RawQuery(query): RawQuery) -> Response {
    serve_page(&app, "", query.as_deref()).await
}

// -- tests --------------------------------------------------------------

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn the_location_grammar_parses_the_v1_families() {
        assert!(matches!(parse_location(""), Ok(PageLocation::Root)));
        assert!(matches!(parse_location("/"), Ok(PageLocation::Root)));
        let board = parse_location("t-14").unwrap();
        assert!(matches!(
            board,
            PageLocation::Board {
                task: 14,
                revision: None
            }
        ));
        let pinned = parse_location("/t-14@r3").unwrap();
        assert!(matches!(
            pinned,
            PageLocation::Board {
                task: 14,
                revision: Some(3)
            }
        ));
        let history = parse_location("t-14@r3/history").unwrap();
        assert!(matches!(
            history,
            PageLocation::History {
                task: 14,
                revision: Some(3)
            }
        ));
        let live = parse_location("t-14/history").unwrap();
        assert!(matches!(
            live,
            PageLocation::History {
                task: 14,
                revision: None
            }
        ));
        let diff = parse_location("t-14@r2..r5/diff").unwrap();
        assert!(matches!(
            diff,
            PageLocation::Diff {
                task: 14,
                from: 2,
                to: 5
            }
        ));
        let file = parse_location("t-14@r3/file/figures/plot.png").unwrap();
        assert!(matches!(
            file,
            PageLocation::File {
                task: 14,
                revision: Some(3),
                ..
            }
        ));
        let current_file = parse_location("t-14/file/board.yaml").unwrap();
        assert!(matches!(
            current_file,
            PageLocation::File {
                task: 14,
                revision: None,
                ..
            }
        ));
        let reserved = parse_location("t-5@r1/file/nested/history").unwrap();
        assert!(matches!(
            reserved,
            PageLocation::File {
                task: 5,
                revision: Some(1),
                ..
            }
        ));
        let reserved_diff = parse_location("t-5@r1/file/nested/diff").unwrap();
        assert!(matches!(
            reserved_diff,
            PageLocation::File {
                task: 5,
                revision: Some(1),
                ..
            }
        ));
        let component = parse_location("t-14@r3/content/2/group/content/0").unwrap();
        assert!(matches!(
            component,
            PageLocation::Component {
                task: 14,
                revision: 3,
                ..
            }
        ));
    }

    #[test]
    fn malformed_locations_refuse_with_teaching_codes() {
        let code = |raw: &str| parse_location(raw).unwrap_err().code;
        assert_eq!(code("nonsense"), "unknown_location");
        assert_eq!(code("t-"), "unknown_location");
        assert_eq!(code("t-14@x"), "unknown_location");
        assert_eq!(code("t-14/rubbish"), "unknown_location");
        assert_eq!(code("t-14@r3..r1/diff"), "bad_range");
        assert_eq!(code("t-14..r2/diff"), "bad_range");
        assert_eq!(code("t-14@r2/history/x"), "unknown_location");
        assert_eq!(code("t-14@r3/file/"), "unknown_location");
        assert_eq!(code("t-14@r3/file/../board.yaml"), "bad_file_path");
        assert_eq!(code("t-14@r3/content/"), "bad_select_path");
        assert_eq!(code("t-14@r3/content/x"), "bad_select_path");
        assert_eq!(code("t-14/content/0"), "unknown_location");
        assert_eq!(code("t-14@r3/content"), "bad_select_path");
    }

    #[test]
    fn a_range_requires_two_ordered_revisions() {
        let error = parse_location("t-4@r2..r2/diff").unwrap_err();
        assert_eq!(error.code, "bad_range");
        assert!(
            error.detail.contains("does not precede"),
            "{}",
            error.detail
        );
    }

    #[test]
    fn query_values_refuse_with_teaching_codes() {
        assert_eq!(
            parse_query(Some("after=abc")).unwrap_err().code,
            "bad_query"
        );
        assert_eq!(
            parse_query(Some("timeout=soon")).unwrap_err().code,
            "bad_query"
        );
        let (after, hold) = parse_query(Some("after=9")).unwrap();
        assert_eq!(after, Some(9));
        assert_eq!(hold, DEFAULT_HOLD);
        let (_, hold) = parse_query(Some("after=9&timeout=70")).unwrap();
        assert_eq!(hold, MAX_HOLD);
        let (after, hold) = parse_query(None).unwrap();
        assert_eq!(after, None);
        assert_eq!(hold, Duration::ZERO);
    }

    #[test]
    fn wrapped_parts_carry_their_own_locations() {
        let board = board_grammar::parse(
            "grammar: 1\ntask: t-0\ncontent:\n  - text: one\n  - group:\n      title: g\n      content:\n        - text: two\n  - annotation: a concern\n",
        )
        .unwrap();
        let parts: Vec<Part> = board
            .content
            .iter()
            .enumerate()
            .map(|(index, component)| part_of(component, &format!("/content/{index}"), 0, 1))
            .collect();
        let rendered = serde_json::to_value(&parts).unwrap();
        assert_eq!(rendered[0]["location"], "t-0@r1:/content/0");
        assert_eq!(rendered[1]["location"], "t-0@r1:/content/1");
        assert_eq!(
            rendered[1]["content"][0]["location"],
            "t-0@r1:/content/1/group/content/0"
        );
        // the annotation rides a titled group, the concern's text keeps
        // the annotation's own location
        assert_eq!(rendered[2]["title"], "annotation");
        assert_eq!(rendered[2]["content"][0]["location"], "t-0@r1:/content/2");
    }

    #[test]
    fn incorporations_collapse_to_source_line_receiving_text_and_citation_refs() {
        let board = board_grammar::parse(
            "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 2\n        select: [\"/content/0\", \"/content/1\"]\n      content:\n        - text: relying on the proof\n",
        )
        .unwrap();
        let part = part_of(&board.content[0], "/content/0", 1, 1);
        let rendered = serde_json::to_value(&part).unwrap();
        assert_eq!(rendered["title"], "incorporates t-0@r2");
        assert_eq!(rendered["location"], "t-1@r1:/content/0");
        assert_eq!(rendered["content"][0]["kind"], "ref");
        assert_eq!(rendered["content"][0]["location"], "t-0@r2");
        assert_eq!(
            rendered["content"][1]["location"],
            "t-1@r1:/content/0/incorporation/content/0"
        );
        let citations: Vec<&serde_json::Value> = rendered["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|part| part["label"].as_str().unwrap_or("").starts_with("cites"))
            .collect();
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0]["location"], "t-0@r2/content/0");
        assert_eq!(citations[1]["location"], "t-0@r2/content/1");
    }

    #[test]
    fn file_content_types_name_the_kinds_the_viewer_renders() {
        assert_eq!(content_type_of("board.yaml"), "text/yaml");
        assert_eq!(content_type_of("figures/plot.png"), "image/png");
        assert_eq!(content_type_of("figures/plot.svg"), "image/svg+xml");
        assert_eq!(content_type_of("paper.pdf"), "application/pdf");
        assert_eq!(content_type_of("data.bin"), "application/octet-stream");
    }

    #[test]
    fn the_machine_id_reader_trims_and_never_defaults() {
        let dir = std::env::temp_dir().join(format!("sac-machine-id-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("machine-id");
        std::fs::write(&path, "b1f7c2e0a9d84e6f8a3c5d7e9f0b1c2d\n").unwrap();
        assert_eq!(
            machine_id(path.to_str().unwrap()).as_deref(),
            Some("b1f7c2e0a9d84e6f8a3c5d7e9f0b1c2d"),
            "read and trimmed"
        );
        std::fs::write(&path, "   \n").unwrap();
        assert_eq!(
            machine_id(path.to_str().unwrap()),
            None,
            "an empty id is None, never defaulted: two machines can never byte-equal on it"
        );
        assert_eq!(
            machine_id(dir.join("missing").to_str().unwrap()),
            None,
            "unreadable is None"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
