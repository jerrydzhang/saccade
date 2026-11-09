//! The read-side projections: what the CLI and the wire render from a
//! folded world. Views state record facts, never presentation
//! judgments; the human-facing numbers derive here, from birth order.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::events::Event;
use crate::objects::incarnation::IncarnationId;
use crate::objects::message::MessageId;
use crate::objects::task::{Owner, OwnerEntry, TaskContext, TaskId, TaskState};
use crate::store::{RecordId, World};

/// One message as the mailbox face renders it: the dense birth
/// number, the addressed endpoints by theirs, and the delivery state
/// - bookkeeping, never a claim of reading or agreement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessageLine {
    pub num: usize,
    pub seq: usize,
    pub from: usize,
    pub to: usize,
    pub reply_to: Option<usize>,
    pub task: Option<usize>,
    pub state: &'static str,
    pub actor: String,
    pub body: String,
    pub born_at: u64,
}

pub fn message_line(world: &World, id: MessageId) -> Option<MessageLine> {
    let mctx = world.messages.get(&id)?;
    Some(MessageLine {
        num: world.message_num(id)?,
        seq: id.0.0,
        from: world.incarnation_num(mctx.message.from)?,
        to: world.incarnation_num(mctx.message.to)?,
        reply_to: mctx.message.reply_to.and_then(|r| world.message_num(r)),
        task: mctx.message.task.map(|t| t.0),
        state: match mctx.state {
            crate::objects::message::MessageState::Pending => "pending",
            crate::objects::message::MessageState::Submitted => "submitted",
        },
        actor: mctx.actor.as_str().to_string(),
        body: mctx.message.body.as_str().to_string(),
        born_at: mctx.born_at,
    })
}

/// The messages an incarnation participates in, in record order:
/// everything addressed to it plus everything it sent.
pub fn mailbox(world: &World, id: IncarnationId) -> Vec<MessageId> {
    world
        .messages
        .keys()
        .copied()
        .filter(|m| {
            let mctx = &world.messages[m];
            mctx.message.from == id || mctx.message.to == id
        })
        .collect()
}

/// What the task wait waits for. The offer arm returns with the task
/// events; until then a wait is activity only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitUntil {
    Activity,
    Offer,
}

/// Whether the record the event lands in wakes a wait on `target`.
/// Task facts wake their task; the machinery's own facts and the
/// message traffic ride no task and wake nothing.
pub fn wait_matches(
    world: &World,
    at: RecordId,
    event: &Event,
    target: &Target,
    until: WaitUntil,
) -> bool {
    if matches!(until, WaitUntil::Offer) {
        return matches!(event, Event::TaskOffered { task, .. } if *target == Target::Task(*task));
    }
    match event {
        Event::TaskCreated { .. } => {
            matches!(*target, Target::Task(t) if world.task_born_at(at) == Some(t))
        }
        Event::TaskClaimed { id, .. }
        | Event::TaskCancelled { id, .. }
        | Event::TaskReopened { id, .. }
        | Event::TaskCommitted { task: id, .. }
        | Event::TaskOffered { task: id, .. }
        | Event::TaskRejected { task: id, .. }
        | Event::TaskAccepted { task: id } => *target == Target::Task(*id),
        Event::WorkspaceEstablished { task_id, .. }
        | Event::TaskWorkspaceCheckpointed { task_id, .. } => *target == Target::Task(*task_id),
        // the root destination acts, the registrations, the relaunch
        // facts, the workspace lifecycle, and the messages name no
        // task the wait could wake on
        _ => false,
    }
}

/// What a wait addresses: a task by its id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Task(TaskId),
}

pub struct TaskView {
    pub id: String,
    pub state: &'static str,
    pub name: String,
    pub owner: Option<String>,
}

pub fn state_str(state: &TaskState) -> &'static str {
    match state {
        TaskState::Open => "open",
        TaskState::Offered => "offered",
        TaskState::Concluded => "concluded",
        TaskState::Cancelled(_) => "cancelled",
    }
}

impl TaskView {
    pub fn of(world: &World, id: TaskId, ctx: &TaskContext) -> TaskView {
        TaskView {
            id: format!("t-{}", id.0),
            state: state_str(&ctx.task.state),
            name: ctx.task.name.as_str().to_string(),
            owner: current_owner(ctx).map(|o| owner_str(world, &o.owner)),
        }
    }
}

/// The ownership ledger's latest entry, the current holder.
pub fn current_owner(ctx: &TaskContext) -> Option<&OwnerEntry> {
    ctx.ownership.last()
}

fn owner_str(world: &World, owner: &Owner) -> String {
    match owner {
        Owner::Human(actor) | Owner::External(actor) => actor.as_str().to_string(),
        // the public number: incarnation birth order, derived by the fold
        Owner::Incarnation(id) => format!(
            "i-{}",
            world
                .incarnation_num(*id)
                .expect("an owner is a resident incarnation")
        ),
    }
}

pub struct ShowView {
    pub id: String,
    pub state: &'static str,
    pub name: String,
    /// The cancellation's recorded explanation, on a cancelled task
    pub explanation: Option<String>,
    pub owner: Option<String>,
    /// The established workspace's branch and address, when one stands
    pub workspace: Option<String>,
}

impl ShowView {
    pub fn of(world: &World, id: TaskId, ctx: &TaskContext) -> ShowView {
        ShowView {
            id: format!("t-{}", id.0),
            state: state_str(&ctx.task.state),
            name: ctx.task.name.as_str().to_string(),
            explanation: match &ctx.task.state {
                TaskState::Cancelled(why) => Some(why.as_str().to_string()),
                _ => None,
            },
            owner: current_owner(ctx).map(|o| owner_str(world, &o.owner)),
            workspace: ctx.workspace.as_ref().map(|w| {
                format!(
                    "{} | {}",
                    w.workspace.branch.as_str(),
                    w.workspace.worktree.as_path().display()
                )
            }),
        }
    }
}

/// Read helpers: follow the world's pointers so emitters need not.
pub fn task_view(world: &World, id: TaskId) -> Option<TaskView> {
    world
        .tasks
        .get(id.0)
        .map(|ctx| TaskView::of(world, id, ctx))
}

pub fn show_view(world: &World, id: TaskId) -> Option<ShowView> {
    world
        .tasks
        .get(id.0)
        .map(|ctx| ShowView::of(world, id, ctx))
}

/// The board: live tasks in identity order, flat.
pub fn board(world: &World) -> Vec<TaskView> {
    world
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, ctx)| matches!(ctx.task.state, TaskState::Open | TaskState::Offered))
        .map(|(i, ctx)| TaskView::of(world, TaskId(i), ctx))
        .collect()
}

/// The rail's archive: concluded and cancelled tasks, newest first -
/// the sort key is last touch, which never sits earlier than the close
/// itself.
pub fn closed_tasks(world: &World) -> Vec<TaskView> {
    let mut rows: Vec<(u64, usize, TaskView)> = world
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, ctx)| {
            matches!(
                ctx.task.state,
                TaskState::Concluded | TaskState::Cancelled(_)
            )
        })
        .map(|(i, ctx)| (ctx.last_record_at, i, TaskView::of(world, TaskId(i), ctx)))
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    rows.into_iter().map(|(_, _, row)| row).collect()
}

// -- the reference resolver's target vocabulary ----------------------

/// What a record's reference resolves to on the reading side: the
/// birth kinds carry their own shapes, every other record fact rides
/// the provenance index to its home task.
#[derive(Clone, Debug, PartialEq)]
pub enum RefTarget {
    Task {
        task: usize,
    },
    Incarnation {
        num: usize,
        /// The birth task a historical establishment carries; a
        /// registered native identity has none
        task: Option<usize>,
    },
    Message {
        num: usize,
    },
    /// Any other record fact - a claim, a checkpoint: its position
    /// and its home task, None when the fold gives it none
    Record {
        seq: usize,
        task: Option<usize>,
    },
}

/// One reference's render target, resolved against the fold.
pub fn ref_target(world: &World, rid: RecordId) -> Option<RefTarget> {
    if let Some(task) = world.task_born_at(rid) {
        return Some(RefTarget::Task { task: task.0 });
    }
    if let Some(ictx) = world.incarnations.get(&crate::IncarnationId(rid)) {
        return Some(RefTarget::Incarnation {
            num: world.incarnation_num(crate::IncarnationId(rid))?,
            task: match &ictx.incarnation {
                crate::Incarnation::Established { task, .. } => Some(task.0),
                crate::Incarnation::Native { .. } => None,
            },
        });
    }
    if world.messages.contains_key(&crate::MessageId(rid)) {
        return Some(RefTarget::Message {
            num: world.message_num(crate::MessageId(rid))?,
        });
    }
    Some(RefTarget::Record {
        seq: rid.0,
        task: world.provenance.get(&rid).copied().flatten().map(|t| t.0),
    })
}

/// Each record's home task, where the fold gives it one: a task by
/// its birth, the task-lifecycle facts by the task they move.
pub fn thread_index(world: &World) -> BTreeMap<usize, TaskId> {
    world
        .provenance
        .iter()
        .filter_map(|(rid, task)| task.map(|t| (rid.0, t)))
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Facet {
    /// Only records living on this task
    In(TaskId),
    /// Only records whose author's name contains this, case-folded
    By(String),
    /// Only records of this kind; the vocabulary is SEARCH_KINDS
    Kind(&'static str),
}

/// The kinds a searched record can be - kind:'s whole vocabulary.
pub const SEARCH_KINDS: [&str; 2] = ["task", "explanation"];

/// A term: plain words match text; task ids are reference searches.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// Matches as a whole, word-bounded token, case-folded
    Plain(String),
    /// t-N: prose namings of the task
    Task(TaskId),
}

impl Term {
    /// The string scanned for in prose.
    fn canonical(&self) -> String {
        match self {
            Term::Plain(s) => s.clone(),
            Term::Task(id) => format!("t-{}", id.0),
        }
    }
}

/// The parsed query: every term must match, every facet must pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchQuery {
    pub terms: Vec<Term>,
    pub facets: Vec<Facet>,
}

/// The grammar is strict so a mistyped facet never becomes a quiet term.
#[derive(Debug, PartialEq)]
pub enum SearchFail {
    Usage(String),
    /// A facet named a task the fold does not hold
    NoSuchTask {
        asked: TaskId,
        holds: usize,
    },
}

impl std::fmt::Display for SearchFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchFail::Usage(m) => write!(f, "{m}"),
            SearchFail::NoSuchTask { asked, holds } => match holds {
                0 => write!(
                    f,
                    "no task t-{} exists; this tracker holds no tasks yet",
                    asked.0
                ),
                1 => write!(
                    f,
                    "no task t-{} exists; this tracker holds 1 task, t-0",
                    asked.0
                ),
                n => write!(
                    f,
                    "no task t-{} exists; this tracker holds {n} tasks, t-0 through t-{}",
                    asked.0,
                    n - 1
                ),
            },
        }
    }
}

fn task_ref(token: &str, facet: &str) -> Result<TaskId, SearchFail> {
    token
        .strip_prefix("t-")
        .and_then(|n| n.parse::<usize>().ok())
        .map(TaskId)
        .ok_or_else(|| SearchFail::Usage(format!("{facet} takes a task id, e.g. {facet}t-4")))
}

fn not_a_reference(token: &str) -> SearchFail {
    SearchFail::Usage(format!("'{token}' is not a reference (expected t-<n>)"))
}

impl SearchQuery {
    pub fn parse(tokens: &[String]) -> Result<SearchQuery, SearchFail> {
        let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
        let mut query = SearchQuery::default();
        for token in tokens {
            if let Some(rest) = token.strip_prefix("in:") {
                query.facets.push(Facet::In(task_ref(rest, "in:")?));
            } else if let Some(rest) = token.strip_prefix("by:") {
                if rest.is_empty() {
                    return Err(SearchFail::Usage(
                        "by: needs an actor name, e.g. by:pi".into(),
                    ));
                }
                query.facets.push(Facet::By(rest.to_lowercase()));
            } else if let Some(rest) = token.strip_prefix("kind:") {
                let Some(kind) = SEARCH_KINDS.iter().find(|k| **k == rest) else {
                    return Err(SearchFail::Usage(format!(
                        "kind: {rest} is not a kind; the kinds are {}",
                        SEARCH_KINDS.join(", ")
                    )));
                };
                query.facets.push(Facet::Kind(kind));
            } else if let Some(n) = token.strip_prefix('#').filter(|n| numeric(n)) {
                return Err(SearchFail::Usage(format!(
                    "drop the '#': the reference search is t-{n}"
                )));
            } else if let Some(n) = token.strip_prefix("t-").filter(|n| numeric(n)) {
                let n = n.parse().map_err(|_| not_a_reference(token))?;
                query.terms.push(Term::Task(TaskId(n)));
            } else if token.starts_with("c-") && numeric(token.strip_prefix("c-").unwrap_or("")) {
                return Err(SearchFail::Usage(
                    "c-N named no resident object; the tracker holds no comments - search tasks with t-<n>".into(),
                ));
            } else if token.is_empty() {
                return Err(SearchFail::Usage("a term is required".into()));
            } else {
                query.terms.push(Term::Plain(token.to_lowercase()));
            }
        }
        if query.terms.is_empty() && query.facets.is_empty() {
            return Err(SearchFail::Usage(
                "give at least one term or a facet (in:, by:, kind:); the moves:\n  sac search telemetry      find where it was decided\n  sac search floop in:t-0   narrow with facets".into(),
            ));
        }
        Ok(query)
    }
}

/// Id grammar is word grammar: letters, digits, '-', '#', '_' are word
/// characters, so "t-49" never matches inside "t-490" or "pi/t-90-1".
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '-' || c == '#' || c == '_'
}

/// A term matches a field when it appears as a whole token, case-folded;
/// never a substring, never ranked.
fn token_in_field(term: &str, field: &str) -> bool {
    let field = field.to_lowercase();
    let mut from = 0;
    while let Some(at) = field[from..].find(term) {
        let start = from + at;
        let end = start + term.len();
        let bounded = field[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_word_char(c))
            && field[end..].chars().next().is_none_or(|c| !is_word_char(c));
        if bounded {
            return true;
        }
        from = end;
    }
    false
}

/// The line a result renders: the first line a term lands on, else
/// the field's first line (a reference-only match still points). A
/// facet-only query has no terms, so its results read as first lines.
pub fn matched_line(terms: &[Term], text: &str) -> String {
    for line in text.lines() {
        if terms.is_empty() || terms.iter().any(|t| token_in_field(&t.canonical(), line)) {
            return line.to_string();
        }
    }
    text.lines().next().unwrap_or_default().to_string()
}

/// One matched record: a pointer and the full text of the field it
/// matched; the rendered face cuts that to the matched line.
#[derive(Debug, PartialEq)]
pub struct SearchRecord {
    pub pointer: String,
    pub kind: &'static str,
    /// The author, when the fold keeps one
    pub actor: Option<String>,
    pub body: String,
    /// The record's position
    order: usize,
}

impl SearchRecord {
    /// The record's searched text.
    pub fn text(&self) -> String {
        self.body.clone()
    }
}

/// A task's results: its records after the facets, and the term
/// matches it held before any facet narrowed them.
#[derive(Debug)]
pub struct SearchGroup {
    pub task: usize,
    pub title: String,
    pub records: Vec<SearchRecord>,
    pub total: usize,
}

/// Search the fold: no index, no ranking, no memory between calls.
/// A query with no terms matches everything the facets then narrow.
pub fn search(world: &World, query: &SearchQuery) -> Result<Vec<SearchGroup>, SearchFail> {
    for facet in &query.facets {
        let asked = match facet {
            Facet::In(id) => Some(*id),
            _ => None,
        };
        if let Some(id) = asked
            && id.0 >= world.tasks.len()
        {
            return Err(SearchFail::NoSuchTask {
                asked: id,
                holds: world.tasks.len(),
            });
        }
    }

    let text_match = |term: &Term, text: &str| token_in_field(&term.canonical(), text);
    let mut matches: Vec<(usize, SearchRecord)> = Vec::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        // a t-N term is a reference search: the task's own records match
        // it whatever the prose says, a plain term reads the text
        let term_hits = |field: &str, term: &Term| match term {
            Term::Task(id) => i == id.0 || text_match(term, field),
            Term::Plain(_) => text_match(term, field),
        };
        let title = ctx.task.name.as_str();
        if query.terms.iter().all(|t| term_hits(title, t)) {
            matches.push((
                i,
                SearchRecord {
                    pointer: format!("t-{i}"),
                    kind: "task",
                    actor: Some(ctx.birth_actor.as_str().to_string()),
                    body: title.to_string(),
                    order: ctx.birth.0,
                },
            ));
        }
        if let TaskState::Cancelled(why) = &ctx.task.state {
            let text = why.as_str();
            if query.terms.iter().all(|t| term_hits(text, t)) {
                matches.push((
                    i,
                    SearchRecord {
                        pointer: format!("t-{i} explanation"),
                        kind: "explanation",
                        actor: None,
                        body: text.to_string(),
                        order: usize::MAX,
                    },
                ));
            }
        }
    }

    // group by task; records by position, the explanation closing its group
    let mut by_task: BTreeMap<usize, Vec<SearchRecord>> = BTreeMap::new();
    for (task, record) in matches {
        by_task.entry(task).or_default().push(record);
    }
    // a facet-only query has no terms, so a task it filters out holds
    // nothing worth a count row - only the tasks it matches render
    let facet_only = query.terms.is_empty();
    let mut groups: Vec<(usize, SearchGroup)> = Vec::new();
    for (task, mut records) in by_task {
        records.sort_by_key(|r| r.order);
        let first = records.first().expect("groups hold a match").order;
        let total = records.len();
        let records: Vec<SearchRecord> = records
            .into_iter()
            .filter(|r| facets_pass(query, task, r))
            .collect();
        if facet_only && records.is_empty() {
            continue;
        }
        groups.push((
            first,
            SearchGroup {
                task,
                title: world.tasks[task].task.name.as_str().to_string(),
                records,
                total,
            },
        ));
    }
    // tasks enter by their first match, never by narrowing side effects
    groups.sort_by_key(|(first, g)| (*first, g.task));
    Ok(groups.into_iter().map(|(_, g)| g).collect())
}

fn facets_pass(query: &SearchQuery, task: usize, record: &SearchRecord) -> bool {
    query.facets.iter().all(|facet| match facet {
        Facet::In(id) => task == id.0,
        Facet::By(name) => record
            .actor
            .as_deref()
            .is_some_and(|a| a.to_lowercase().contains(name)),
        Facet::Kind(kind) => record.kind == *kind,
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::events::Event;
    use crate::objects::message::MessageState;
    use crate::objects::task::Task;
    use crate::store::{Context, Record, Scope, Tier, World};
    use crate::types::actor::ActorName;
    use crate::types::pointers::{NativeSessionId, SessionLocator};
    use crate::types::prose::Prose;

    fn filler() -> Prose {
        Prose::new("implement foo".into()).unwrap()
    }

    fn ctx_of(state: TaskState) -> TaskContext {
        TaskContext {
            task: Task {
                state,
                name: filler(),
            },
            last_updated: RecordId(0),
            birth: RecordId(0),
            last_record_at: 0,
            ownership: Vec::new(),
            commits: Vec::new(),
            birth_actor: ActorName::new("human person".into()).unwrap(),
            workspace: None,
        }
    }

    fn record(seq: usize, context: &Context, event: Event) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: seq as u64,
            context: context.clone(),
            event,
        }
    }

    fn ctx(name: &str) -> Context {
        Context {
            actor: ActorName::new(name.into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        }
    }

    #[test]
    fn state_strings_are_the_identifier_vocabulary() {
        let world = World::new();
        assert_eq!(
            TaskView::of(&world, TaskId(0), &ctx_of(TaskState::Open)).state,
            "open"
        );
        assert_eq!(
            TaskView::of(&world, TaskId(0), &ctx_of(TaskState::Offered)).state,
            "offered"
        );
        assert_eq!(
            TaskView::of(&world, TaskId(0), &ctx_of(TaskState::Concluded)).state,
            "concluded"
        );
        assert_eq!(
            TaskView::of(&world, TaskId(0), &ctx_of(TaskState::Cancelled(filler()))).state,
            "cancelled"
        );
    }

    // the rail's archive holds only the closed states, newest close
    // first; the board holds the live pair
    #[test]
    fn the_rail_archive_holds_concluded_and_cancelled_only() {
        let mut world = World::new();
        let mut rows = vec![
            ctx_of(TaskState::Open),
            ctx_of(TaskState::Offered),
            {
                let mut ctx = ctx_of(TaskState::Concluded);
                ctx.last_record_at = 5;
                ctx
            },
            {
                let mut ctx = ctx_of(TaskState::Cancelled(filler()));
                ctx.last_record_at = 7;
                ctx
            },
        ];
        rows.reverse();
        world.tasks = rows;

        let archive = closed_tasks(&world);
        assert_eq!(
            archive.iter().map(|r| r.state).collect::<Vec<_>>(),
            vec!["cancelled", "concluded"],
            "newest close first"
        );
        let live: Vec<_> = board(&world).into_iter().map(|r| r.state).collect();
        assert_eq!(live, vec!["offered".to_string(), "open".to_string()]);
    }

    // the search surface that remains: task names and cancellation
    // explanations, narrowed by the facets
    #[test]
    fn search_reads_names_and_explanations() {
        let mut world = World::new();
        let mut open = ctx_of(TaskState::Open);
        open.task.name = Prose::new("migrate the floop index".into()).unwrap();
        open.birth_actor = ActorName::new("human person".into()).unwrap();
        let mut cancelled = ctx_of(TaskState::Cancelled(
            Prose::new("the floop migration is no longer wanted".into()).unwrap(),
        ));
        cancelled.task.name = Prose::new("migrate floop".into()).unwrap();
        cancelled.birth = RecordId(3);
        world.tasks = vec![open, cancelled];

        // a plain term reaches both fields
        let groups = search(&world, &SearchQuery::parse(&["floop".to_string()]).unwrap()).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].task, 0);
        assert_eq!(groups[0].records[0].kind, "task");
        assert_eq!(groups[1].records.len(), 2);
        assert_eq!(groups[1].records[0].kind, "task");
        assert_eq!(groups[1].records[1].kind, "explanation");

        // a t-N reference search names its task's records
        let groups = search(&world, &SearchQuery::parse(&["t-1".to_string()]).unwrap()).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].task, 1);
        assert_eq!(groups[0].records.len(), 2);

        // the kind facet narrows to one vocabulary half
        let groups = search(
            &world,
            &SearchQuery::parse(&["kind:explanation".to_string()]).unwrap(),
        )
        .unwrap();
        assert_eq!(groups.len(), 1);
        assert!(groups[0].records.iter().all(|r| r.kind == "explanation"));
    }

    // an unknown in: task names what the fold holds
    #[test]
    fn an_unknown_task_facet_names_the_holds() {
        let world = World::new();
        let error = SearchQuery::parse(&["in:t-4".to_string()])
            .and_then(|query| search(&world, &query))
            .expect_err("the empty fold holds no t-4");
        assert!(error.to_string().contains("holds no tasks yet"));
    }

    // the task wait wakes on exactly its task's lifecycle and
    // workspace records, and nothing else
    #[test]
    fn the_task_wait_matches_exactly_its_own_records() {
        let mut world = World::new();
        let mut events = Vec::new();
        let push = |world: &mut World,
                    events: &mut Vec<(usize, Event)>,
                    seq: usize,
                    context: &Context,
                    event: Event| {
            let folded = world.clone().apply(record(seq, context, event.clone()));
            *world = folded.expect("the story folds");
            events.push((seq, event));
        };
        push(
            &mut world,
            &mut events,
            0,
            &ctx("human person"),
            Event::TaskCreated {
                name: Prose::new("implement foo".into()).unwrap(),
            },
        );
        push(
            &mut world,
            &mut events,
            1,
            &ctx("human person"),
            Event::TaskCreated {
                name: Prose::new("fix bar".into()).unwrap(),
            },
        );
        push(
            &mut world,
            &mut events,
            2,
            &ctx("pi"),
            Event::TaskClaimed {
                id: TaskId(0),
                incarnation: None,
            },
        );
        push(
            &mut world,
            &mut events,
            3,
            &ctx("pi"),
            Event::WorkspaceEstablished {
                task_id: TaskId(0),
                base: crate::GitCommit::new("abc123".into()).unwrap(),
                branch: crate::GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: crate::WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
        );
        push(
            &mut world,
            &mut events,
            4,
            &Context {
                actor: ActorName::new("human person".into()).unwrap(),
                tier: Tier::Human,
                scope: None,
            },
            Event::TaskCancelled {
                id: TaskId(1),
                explanation: Prose::new("no longer wanted".into()).unwrap(),
            },
        );
        push(
            &mut world,
            &mut events,
            5,
            &ctx("pi"),
            Event::IncarnationRegistered {
                native_id: NativeSessionId::new("aaaa1111".into()).unwrap(),
                saved_session: None,
                address: None,
                adopted: false,
                harness: crate::Harness::Pi,
            },
        );

        let matching = |target: &Target| {
            events
                .iter()
                .filter(|(seq, event)| {
                    wait_matches(&world, RecordId(*seq), event, target, WaitUntil::Activity)
                })
                .map(|(seq, _)| *seq)
                .collect::<Vec<_>>()
        };
        assert_eq!(matching(&Target::Task(TaskId(0))), vec![0, 2, 3]);
        assert_eq!(matching(&Target::Task(TaskId(1))), vec![1, 4]);
    }

    mod messages {
        use super::*;

        fn registered(_seq: usize, id: &str) -> Event {
            Event::IncarnationRegistered {
                native_id: NativeSessionId::new(id.into()).unwrap(),
                saved_session: Some(
                    SessionLocator::new(format!("/sessions/{id}.jsonl").into()).unwrap(),
                ),
                address: None,
                adopted: false,
                harness: crate::Harness::Pi,
            }
        }

        fn sent(from: usize, to: usize, reply_to: Option<usize>, task: Option<usize>) -> Event {
            Event::MessageSent {
                from: IncarnationId(RecordId(from)),
                to: IncarnationId(RecordId(to)),
                reply_to: reply_to.map(|r| MessageId(RecordId(r))),
                task: task.map(TaskId),
                body: Prose::new("the failing edge case".into()).unwrap(),
            }
        }

        /// A send rides its sender's declared identity: the store
        /// matches the event's from against the scope.
        fn sender(_seq: usize, incarnation: usize) -> Context {
            Context {
                actor: ActorName::new("pi".into()).unwrap(),
                tier: Tier::Agent,
                scope: Some(Scope {
                    task: None,
                    incarnation: Some(IncarnationId(RecordId(incarnation))),
                    workspace: None,
                }),
            }
        }

        /// The dense message numbers and the mailbox's order: births
        /// count births of their kind, never record positions.
        #[test]
        fn message_numbers_count_births_and_mailboxes_order_them() {
            let mut world = World::new();
            for (seq, event) in [
                (0, registered(0, "aaaa1111")),
                (1, registered(1, "bbbb2222")),
                (2, sent(0, 1, None, None)),
                (3, sent(1, 0, Some(2), None)),
            ] {
                let context = match event {
                    Event::MessageSent { from, .. } => sender(seq, from.0.0),
                    _ => ctx("pi"),
                };
                world = world
                    .apply(record(seq, &context, event))
                    .expect("the story folds");
            }
            assert_eq!(world.message_num(MessageId(RecordId(2))), Some(0));
            assert_eq!(world.message_num(MessageId(RecordId(3))), Some(1));
            assert_eq!(world.message_at_num(1), Some(MessageId(RecordId(3))));
            assert_eq!(world.message_num(MessageId(RecordId(9))), None);

            // the mailbox reads both directions in record order
            let box_of_one = mailbox(&world, IncarnationId(RecordId(1)));
            assert_eq!(
                box_of_one,
                vec![MessageId(RecordId(2)), MessageId(RecordId(3))]
            );

            // the lines carry dense endpoints and the pending state
            let line = message_line(&world, MessageId(RecordId(3))).unwrap();
            assert_eq!((line.num, line.from, line.to), (1, 1, 0));
            assert_eq!(line.reply_to, Some(0));
            assert_eq!(line.state, "pending");
        }

        /// A submitted message reads submitted: bookkeeping, never a
        /// claim of reading or agreement.
        #[test]
        fn the_line_names_the_delivery_state() {
            let mut world = World::new();
            for (seq, event) in [
                (0, registered(0, "aaaa1111")),
                (1, registered(1, "bbbb2222")),
                (2, sent(0, 1, None, None)),
                (
                    3,
                    Event::MessageSubmitted {
                        id: MessageId(RecordId(2)),
                        by: IncarnationId(RecordId(1)),
                    },
                ),
            ] {
                let context = if seq == 2 { sender(seq, 0) } else { ctx("pi") };
                world = world
                    .apply(record(seq, &context, event))
                    .expect("the story folds");
            }
            let line = message_line(&world, MessageId(RecordId(2))).unwrap();
            assert_eq!(line.state, "submitted");
            let mctx = &world.messages[&MessageId(RecordId(2))];
            assert_eq!(mctx.state, MessageState::Submitted);
        }

        /// Messages take no home: a task context rides without
        /// publication, and the provenance index stays None.
        #[test]
        fn messages_take_no_home() {
            let mut world = World::new();
            for (seq, event) in [
                (
                    0,
                    Event::TaskCreated {
                        name: Prose::new("implement foo".into()).unwrap(),
                    },
                ),
                (1, registered(1, "aaaa1111")),
                (2, registered(2, "bbbb2222")),
                (3, sent(1, 2, None, Some(0))),
            ] {
                let context = if seq == 3 { sender(seq, 1) } else { ctx("pi") };
                world = world
                    .apply(record(seq, &context, event))
                    .expect("the story folds");
            }
            assert_eq!(world.provenance[&RecordId(1)], None);
            assert_eq!(world.provenance[&RecordId(3)], None);
            // the wait door never matches message traffic
            let event = wire_event(&world.messages[&MessageId(RecordId(3))]);
            assert!(!wait_matches(
                &world,
                RecordId(3),
                &event,
                &Target::Task(TaskId(0)),
                WaitUntil::Activity
            ));
        }

        fn wire_event(mctx: &crate::objects::message::MessageContext) -> Event {
            Event::MessageSent {
                from: mctx.message.from,
                to: mctx.message.to,
                reply_to: mctx.message.reply_to,
                task: mctx.message.task,
                body: mctx.message.body.clone(),
            }
        }
    }
}
