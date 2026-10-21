//! Statements: the render shapes and their constructors, one per reader.
//! Views read objects and contexts; nothing in objects knows views exist.

use std::collections::BTreeMap;

use crate::events::Event;
use crate::objects::comment::{CommentContext, CommentId, CommentKind, CommentState, Target};
use crate::objects::incarnation::IncarnationId;
use crate::objects::message::{MessageId, MessageState};
use crate::objects::task::{Owner, OwnerEntry, TaskContext, TaskId, TaskState};
use crate::store::{RecordId, World};
use crate::types::artifact::Artifact;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceFacts {
    pub task: usize,
    pub base: String,
    pub branch: String,
    pub path: String,
}

/// One message as the mailbox face renders it: the dense birth
/// number, the addressed endpoints by theirs, and the delivery state
/// — bookkeeping, never a claim of reading or agreement.
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
            MessageState::Pending => "pending",
            MessageState::Submitted => "submitted",
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitUntil {
    Activity,
    Offer,
}

/// A comment's structural edges: its citation mapping and the offer a
/// decision names. Prose-only mentions of an id are never edges.
fn cites(cctx: &CommentContext, watched: CommentId) -> bool {
    cctx.references.contains(&watched.0)
        || cctx
            .comment
            .kind
            .offer()
            .is_some_and(|offer| offer == watched)
}

/// Binding is activity about the use it binds, never about the offer
/// it names: a comment wait on the offer does not wake on a binding.
pub fn wait_matches(
    world: &World,
    at: RecordId,
    event: &Event,
    target: &Target,
    until: WaitUntil,
) -> bool {
    if matches!(until, WaitUntil::Offer) {
        return matches!(
            event,
            Event::Commented {
                kind: CommentKind::Offer { .. },
                home,
                ..
            } if Target::Task(*home) == *target
        );
    }
    match event {
        Event::TaskCreated { .. } => match target {
            Target::Task(task) => world.task_born_at(at) == Some(*task),
            Target::Comment(_) => false,
        },
        Event::TaskClaimed { id, .. }
        | Event::TaskCancelled { id, .. }
        | Event::TaskReopened { id, .. } => *target == Target::Task(*id),
        Event::Commented { home, .. } => match target {
            Target::Task(task) => home == task,
            Target::Comment(watched) => world
                .comments
                .get(&CommentId(at))
                .is_some_and(|cctx| cites(cctx, *watched)),
        },
        Event::CommentRevised { id, .. } | Event::CommentWithdrawn { id, .. } => match target {
            Target::Task(task) => world
                .comments
                .get(id)
                .is_some_and(|c| c.comment.home == *task),
            Target::Comment(watched) => {
                *id == *watched
                    || world
                        .comments
                        .get(id)
                        .is_some_and(|cctx| cites(cctx, *watched))
            }
        },
        Event::UseBound { id, .. } => match target {
            Target::Task(task) => world
                .comments
                .get(id)
                .is_some_and(|c| c.comment.home == *task),
            Target::Comment(watched) => *id == *watched,
        },
        Event::ArtifactAdded { root, .. } => *target == Target::Task(*root),
        Event::WorkspaceEstablished { task_id, .. }
        | Event::TaskWorkspaceCheckpointed { task_id, .. } => *target == Target::Task(*task_id),
        // messages take no thread home; they are never task activity
        Event::IncarnationRegistered { .. }
        | Event::MessageSent { .. }
        | Event::MessageSubmitted { .. } => false,
    }
}

pub struct TaskView {
    pub id: String,
    pub state: &'static str,
    pub name: String,
    pub offer: Option<usize>,
    pub owner: Option<String>,
    /// Size of the attached comment thread; renders as a mere pointer (`#`).
    pub n_comments: usize,
}

/// The task state's identifier vocabulary, shared by every view that names it.
fn state_str(state: &TaskState) -> &'static str {
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
            offer: ctx.current_offer.map(|c| c.0.0),
            owner: current_owner(ctx).map(|o| owner_str(world, &o.owner)),
            n_comments: ctx.thread.len(),
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

/// A row of the task's thread view. The home task is explicit at
/// birth; there is no reply parenting — a reply is a comment citing
/// its parent through the reference mapping.
#[derive(Debug)]
pub struct CommentLine {
    /// The public number, the c-N a human addresses: comment birth order
    pub num: usize,
    /// The birth record's position in the log
    pub seq: usize,
    pub home: TaskId,
    pub actor: String,
    pub tier: String,
    pub kind: &'static str,
    pub body: String,
    /// The offer a decision names, as its public number
    pub offer: Option<usize>,
    /// The source task a use declares
    pub source: Option<TaskId>,
    /// The use's current binding, when one has landed
    pub binding: Option<LineBinding>,
    /// The citation mapping in marker order: marker k is references[k-1]
    pub references: Vec<RefTarget>,
    /// The descriptive mark: the fold presents a revised body
    pub revised: bool,
    /// The reviser's name, when the body was revised by an actor who
    /// is not the birth author — a record fact, never a judgment
    pub reviser: Option<String>,
    /// The standing-error repair: the body left the fold's default
    /// presentation, the tombstone holds its place
    pub withdrawn: bool,
    /// The withdrawer's name, when the withdrawal was by an actor who
    /// is not the birth author — a record fact, never a judgment
    pub withdrawer: Option<String>,
    /// The tombstone's note
    pub withdrawal_note: Option<String>,
    pub born_at: u64,
}

/// A use's current selection as a line renders it.
#[derive(Debug, PartialEq)]
pub struct LineBinding {
    /// The selected offer's public number
    pub offer: usize,
    pub grounds: String,
}

/// An artifact's thread line: its position and its pointer. The fold
/// keeps no author and no time for an artifact — the record names
/// content, not a moment of speech.
#[derive(Debug, PartialEq)]
pub struct ArtifactLine {
    pub seq: usize,
    pub name: String,
    pub hash: String,
}

impl ArtifactLine {
    /// The pointer line's short form of the hash.
    pub fn short_hash(&self) -> String {
        self.hash.chars().take(12).collect()
    }
}

/// The focused task's thread in record order: every comment flat at
/// its home, artifacts clustered within themselves.
#[derive(Debug)]
pub enum ThreadItem {
    Note(CommentLine),
    /// Artifacts clustered within themselves: consecutive artifact
    /// records as one item, never attached to a comment's row —
    /// association with a comment is what citing is for.
    Artifacts(Vec<ArtifactLine>),
}

/// The focused task's thread as a view: the context's pointer index,
/// comments in birth order, artifact clusters between them.
pub struct ThreadView {
    pub items: Vec<ThreadItem>,
}

/// The variant a comment carries — the kind never moves; a withdrawn
/// deposit is still the note it was born.
pub fn kind_of(kind: &CommentKind) -> &'static str {
    match kind {
        CommentKind::Note { .. } => "note",
        CommentKind::Offer { .. } => "offer",
        CommentKind::Accept { .. } => "accept",
        CommentKind::Reject { .. } => "reject",
        CommentKind::Use { .. } => "use",
    }
}

fn line_of(world: &World, cid: CommentId) -> CommentLine {
    let cctx = world
        .comments
        .get(&cid)
        .expect("apply guarantees thread members are resident");
    CommentLine {
        num: world
            .comment_num(cid)
            .expect("a resident comment holds a number"),
        seq: cid.0.0,
        home: cctx.comment.home,
        actor: cctx.actor.as_str().to_string(),
        tier: format!("{:?}", cctx.tier).to_lowercase(),
        kind: if matches!(cctx.comment.state, CommentState::Withdrawn(_)) {
            "note"
        } else {
            kind_of(&cctx.comment.kind)
        },
        body: cctx
            .comment
            .kind
            .prose()
            .map(|p| p.as_str().to_string())
            .unwrap_or_default(),
        offer: cctx.comment.kind.offer().and_then(|o| world.comment_num(o)),
        source: cctx.comment.kind.source(),
        binding: match &cctx.comment.state {
            CommentState::UseBound(payload) => Some(LineBinding {
                offer: world
                    .comment_num(payload.offer)
                    .expect("a bound offer is a resident comment"),
                grounds: payload.grounds.as_str().to_string(),
            }),
            _ => None,
        },
        references: cctx
            .references
            .iter()
            .map(|rid| {
                ref_target(world, *rid).unwrap_or(RefTarget::Record {
                    seq: rid.0,
                    task: world.provenance.get(rid).copied().flatten().map(|t| t.0),
                })
            })
            .collect(),
        revised: cctx.revised.is_some(),
        reviser: cctx
            .revised
            .as_ref()
            .filter(|revision| revision.reviser != cctx.actor)
            .map(|revision| revision.reviser.as_str().to_string()),
        withdrawn: cctx.withdrawn.is_some(),
        withdrawer: cctx
            .withdrawn
            .as_ref()
            .filter(|withdrawal| withdrawal.withdrawer != cctx.actor)
            .map(|withdrawal| withdrawal.withdrawer.as_str().to_string()),
        withdrawal_note: match &cctx.comment.state {
            CommentState::Withdrawn(note) => Some(note.as_str().to_string()),
            _ => None,
        },
        born_at: cctx.born_at,
    }
}

fn artifact_line_of(seq: usize, artifact: &Artifact) -> ArtifactLine {
    ArtifactLine {
        seq,
        name: artifact.name.as_str().to_string(),
        hash: artifact.hash.as_str().to_string(),
    }
}

/// One artifact as the thread view renders it, whatever thread holds
/// it — the pointer format `show` prints for a record id.
pub fn artifact_line(world: &World, id: RecordId) -> Option<ArtifactLine> {
    world.tasks.iter().find_map(|ctx| {
        ctx.artifacts
            .iter()
            .find(|(rid, _)| *rid == id)
            .map(|(_, artifact)| artifact_line_of(id.0, artifact))
    })
}

/// One comment as the thread view renders it, whatever thread it lives
/// on — the body format `show` prints for a pointer.
pub fn comment_line(world: &World, id: CommentId) -> Option<CommentLine> {
    world.comments.contains_key(&id).then(|| line_of(world, id))
}

pub struct ShowView {
    pub id: String,
    pub state: &'static str,
    pub name: String,
    /// The cancellation's recorded explanation, on a cancelled task
    pub explanation: Option<String>,
    pub owner: Option<String>,
    pub offer: Option<usize>,
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
            offer: ctx.current_offer.map(|c| c.0.0),
            workspace: ctx.workspace.as_ref().map(|w| {
                format!(
                    "{} · {}",
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

/// Follow the world's pointers to one task's thread: every comment at
/// its home in birth order, artifacts clustered within themselves.
pub fn thread_view(world: &World, id: TaskId) -> Option<ThreadView> {
    let ctx = world.tasks.get(id.0)?;
    enum Member {
        Comment(CommentId),
        Artifact(RecordId, Artifact),
    }
    // the thread's utterances merged by position; artifacts never join
    // a comment's row. Consecutive artifacts accumulate into one
    // cluster and any comment closes it — the thread-top cluster and
    // the mid-stream one are the same path.
    let mut stream: Vec<(usize, Member)> = ctx
        .thread
        .iter()
        .map(|cid| (cid.0.0, Member::Comment(*cid)))
        .collect();
    stream.extend(
        ctx.artifacts
            .iter()
            .map(|(rid, artifact)| (rid.0, Member::Artifact(*rid, artifact.clone()))),
    );
    stream.sort_by_key(|(seq, _)| *seq);

    let mut items: Vec<ThreadItem> = Vec::new();
    let mut open_cluster: Option<Vec<ArtifactLine>> = None;
    for (_, member) in stream {
        match member {
            Member::Comment(cid) => {
                if let Some(cluster) = open_cluster.take() {
                    items.push(ThreadItem::Artifacts(cluster));
                }
                items.push(ThreadItem::Note(line_of(world, cid)));
            }
            Member::Artifact(rid, artifact) => {
                let line = artifact_line_of(rid.0, &artifact);
                match &mut open_cluster {
                    Some(lines) => lines.push(line),
                    None => open_cluster = Some(vec![line]),
                }
            }
        }
    }
    if let Some(cluster) = open_cluster.take() {
        items.push(ThreadItem::Artifacts(cluster));
    }
    Some(ThreadView { items })
}

/// The board: live tasks in identity order, flat. An offered task
/// still takes findings and its accept.
pub fn board(world: &World) -> Vec<TaskView> {
    world
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, ctx)| matches!(ctx.task.state, TaskState::Open | TaskState::Offered))
        .map(|(i, ctx)| TaskView::of(world, TaskId(i), ctx))
        .collect()
}

/// The rail's archive: concluded and cancelled tasks, newest first —
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

// -- the citation resolver's target vocabulary ----------------------

/// What a citation's target resolves to on the reading side: the home
/// a birth renders at. The birth's identity carries its public number
/// where the reader addresses it.
#[derive(Clone, Debug, PartialEq)]
pub enum RefTarget {
    Comment {
        num: usize,
        task: usize,
    },
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
    Artifact {
        seq: usize,
        task: usize,
        name: String,
        hash: String,
    },
    /// Any other record fact a citation names — a claim, a revision,
    /// a binding, a checkpoint: its position and its home thread,
    /// None when the fold gives it none
    Record {
        seq: usize,
        task: Option<usize>,
    },
}

/// One citation's render target, resolved against the fold: the birth
/// kinds carry their own shapes, every other record fact rides the
/// provenance index to its home thread.
pub fn ref_target(world: &World, rid: RecordId) -> Option<RefTarget> {
    if let Some(cctx) = world.comments.get(&CommentId(rid)) {
        return Some(RefTarget::Comment {
            num: world.comment_num(CommentId(rid))?,
            task: cctx.comment.home.0,
        });
    }
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
    for (i, ctx) in world.tasks.iter().enumerate() {
        if let Some((_, artifact)) = ctx.artifacts.iter().find(|(r, _)| *r == rid) {
            return Some(RefTarget::Artifact {
                seq: rid.0,
                task: i,
                name: artifact.name.as_str().to_string(),
                hash: artifact.hash.as_str().to_string(),
            });
        }
    }
    Some(RefTarget::Record {
        seq: rid.0,
        task: world.provenance.get(&rid).copied().flatten().map(|t| t.0),
    })
}

/// Each record's home thread, where the fold gives it one: a task by
/// its birth, a comment by its home, an artifact by the thread it
/// rides. Records about a member — a revision, a withdrawal, a
/// binding — carry none: they fold into the member, not the thread.
pub fn thread_index(world: &World) -> BTreeMap<usize, TaskId> {
    world
        .provenance
        .iter()
        .filter_map(|(rid, task)| task.map(|t| (rid.0, t)))
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Facet {
    /// Only records living on this thread
    In(TaskId),
    /// Only records whose author's name contains this, case-folded
    By(String),
    /// Only records of this kind; the vocabulary is SEARCH_KINDS
    Kind(&'static str),
}

/// The kinds a searched record can be — kind:'s whole vocabulary.
pub const SEARCH_KINDS: [&str; 8] = [
    "task",
    "note",
    "offer",
    "accept",
    "reject",
    "use",
    "artifact",
    "explanation",
];

/// A term: plain words match text; ids are reference searches.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// Matches as a whole, word-bounded token, case-folded
    Plain(String),
    /// c-N over the fold's public comment numbering: prose mentions
    /// and structural citations of that comment
    Comment(usize),
    /// t-N: prose namings of the task, comments homed on it
    Task(TaskId),
}

impl Term {
    /// The string scanned for in prose.
    fn canonical(&self) -> String {
        match self {
            Term::Plain(s) => s.clone(),
            Term::Comment(n) => format!("c-{n}"),
            Term::Task(id) => format!("t-{}", id.0),
        }
    }
}

/// The parsed query: every term must match, every facet must pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchQuery {
    pub terms: Vec<Term>,
    pub facets: Vec<Facet>,
    /// Withdrawn bodies join the match set — they are omitted by
    /// default, the tombstone never speaks for the body it evicted
    pub withdrawn: bool,
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
                    "drop the '#': the reference search is c-{n}"
                )));
            } else if let Some(n) = token.strip_prefix("c-").filter(|n| numeric(n)) {
                query
                    .terms
                    .push(Term::Comment(n.parse().expect("digits checked")));
            } else if let Some(n) = token.strip_prefix("t-").filter(|n| numeric(n)) {
                query
                    .terms
                    .push(Term::Task(TaskId(n.parse().expect("digits checked"))));
            } else if token.is_empty() {
                return Err(SearchFail::Usage("a term is required".into()));
            } else {
                query.terms.push(Term::Plain(token.to_lowercase()));
            }
        }
        if query.terms.is_empty() && query.facets.is_empty() {
            return Err(SearchFail::Usage(
                "give at least one term or a facet (in:, by:, kind:); the moves:\n  sac search telemetry      find where it was decided\n  sac search c-907         follow a reference\n  sac search floop in:t-0   narrow with facets\n  sac search c-907 -C 3    read the neighborhood".into(),
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

/// A structural reference a record carries beside its text.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ref {
    Comment(CommentId),
    Task(TaskId),
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

/// A thread's results: its records after the facets, and the term
/// matches it held before any facet narrowed them.
#[derive(Debug)]
pub struct SearchGroup {
    pub task: usize,
    pub title: String,
    pub records: Vec<SearchRecord>,
    pub total: usize,
}

/// Search the fold: no index, no ranking, no memory between calls.
/// A term with no terms matches everything the facets then narrow.
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

    // a comment term's public number resolves against this fold before
    // any structural matching; prose matching reads the number as text
    let text_match = |term: &Term, text: &str| token_in_field(&term.canonical(), text);
    let comment_match = |term: &Term, body: &str, refs: &[Ref]| match term {
        Term::Comment(n) => {
            text_match(term, body)
                || world
                    .comment_at_num(*n)
                    .is_some_and(|wanted| refs.contains(&Ref::Comment(wanted)))
        }
        _ => false,
    };

    let mut matches: Vec<(usize, SearchRecord)> = Vec::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        let title = ctx.task.name.as_str();
        if query.terms.iter().all(|t| text_match(t, title)) {
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
            if query.terms.iter().all(|t| text_match(t, text)) {
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
        // artifacts are name-indexed: the pointer's text is the name
        for (rid, artifact) in &ctx.artifacts {
            let name = artifact.name.as_str();
            if query.terms.iter().all(|t| text_match(t, name)) {
                matches.push((
                    i,
                    SearchRecord {
                        pointer: format!("seq {rid}", rid = rid.0),
                        kind: "artifact",
                        actor: None,
                        body: name.to_string(),
                        order: rid.0,
                    },
                ));
            }
        }
    }
    for (id, cctx) in &world.comments {
        // a withdrawn deposit's body left the fold's default
        // presentation: it joins only behind the flag
        if cctx.withdrawn.is_some() && !query.withdrawn {
            continue;
        }
        let body = cctx
            .comment
            .kind
            .prose()
            .map(|p| p.as_str().to_string())
            .unwrap_or_default();
        // the structural edges: the home, the use's source, the offer
        // a decision names, the current binding, the citations
        let mut refs = vec![Ref::Task(cctx.comment.home)];
        if let Some(source) = cctx.comment.kind.source() {
            refs.push(Ref::Task(source));
        }
        if let Some(offer) = cctx.comment.kind.offer() {
            refs.push(Ref::Comment(offer));
        }
        if let CommentState::UseBound(payload) = &cctx.comment.state {
            refs.push(Ref::Comment(payload.offer));
        }
        for rid in &cctx.references {
            if world.comments.contains_key(&CommentId(*rid)) {
                refs.push(Ref::Comment(CommentId(*rid)));
            } else if let Some(task) = world.task_born_at(*rid) {
                refs.push(Ref::Task(task));
            }
        }
        let matched = query.terms.iter().all(|t| match t {
            Term::Comment(_) => comment_match(t, &body, &refs),
            Term::Task(id) => text_match(t, &body) || refs.contains(&Ref::Task(*id)),
            Term::Plain(_) => text_match(t, &body),
        });
        if matched {
            matches.push((
                cctx.comment.home.0,
                SearchRecord {
                    pointer: format!(
                        "c-{num}",
                        num = world
                            .comment_num(*id)
                            .expect("a resident comment holds a number")
                    ),
                    kind: kind_of(&cctx.comment.kind),
                    actor: Some(cctx.actor.as_str().to_string()),
                    body,
                    order: id.0.0,
                },
            ));
        }
    }

    // group by thread; records by position, the explanation closing its group
    let mut by_thread: BTreeMap<usize, Vec<SearchRecord>> = BTreeMap::new();
    for (thread, record) in matches {
        by_thread.entry(thread).or_default().push(record);
    }
    // a facet-only query has no terms, so a thread it filters out holds
    // nothing worth a count row — only the threads it matches render
    let facet_only = query.terms.is_empty();
    let mut groups: Vec<(usize, SearchGroup)> = Vec::new();
    for (thread, mut records) in by_thread {
        records.sort_by_key(|r| r.order);
        let first = records.first().expect("groups hold a match").order;
        let total = records.len();
        let records: Vec<SearchRecord> = records
            .into_iter()
            .filter(|r| facets_pass(query, thread, r))
            .collect();
        if facet_only && records.is_empty() {
            continue;
        }
        groups.push((
            first,
            SearchGroup {
                task: thread,
                title: world.tasks[thread].task.name.as_str().to_string(),
                records,
                total,
            },
        ));
    }
    // threads enter by their first match, never by narrowing side effects
    groups.sort_by_key(|(first, g)| (*first, g.task));
    Ok(groups.into_iter().map(|(_, g)| g).collect())
}

fn facets_pass(query: &SearchQuery, thread: usize, record: &SearchRecord) -> bool {
    query.facets.iter().all(|facet| match facet {
        Facet::In(id) => thread == id.0,
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
    use crate::CommentKind;
    use crate::events::Event;
    use crate::objects::comment::CommentId;
    use crate::objects::task::Task;
    use crate::store::{Context, Record, RecordId, Tier, World};
    use crate::types::actor::ActorName;
    use crate::types::artifact::{Artifact, ContentHash};
    use crate::types::prose::Prose;

    fn ctx_of(state: TaskState) -> TaskContext {
        TaskContext {
            task: Task {
                state,
                name: Prose::new("implement foo".into()).unwrap(),
            },
            last_updated: RecordId(0),
            birth: RecordId(0),
            last_record_at: 0,
            offered_at: None,
            ownership: Vec::new(),
            current_offer: None,
            thread: Vec::new(),
            artifacts: Vec::new(),
            birth_actor: ActorName::new("human person".into()).unwrap(),
            workspace: None,
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
            TaskView::of(
                &world,
                TaskId(0),
                &ctx_of(TaskState::Cancelled(
                    Prose::new("no longer wanted".into()).unwrap()
                ))
            )
            .state,
            "cancelled"
        );
    }

    fn human() -> Context {
        Context {
            actor: ActorName::new("human person".into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        }
    }

    fn agent() -> Context {
        Context {
            actor: ActorName::new("saccade bot".into()).unwrap(),
            tier: Tier::Agent,
            scope: None,
        }
    }

    /// The thread is flat: comments in birth order at their homes, an
    /// artifact cluster standing wherever consecutive artifacts land.
    #[test]
    fn the_thread_is_flat_in_birth_order() {
        let human = human();
        let agent = agent();
        let note = |seq: usize, ctx: &Context, home: TaskId, body: &str| Record {
            id: RecordId(seq),
            timestamp: seq as u64,
            context: ctx.clone(),
            event: Event::Commented {
                home,
                kind: CommentKind::Note {
                    body: Prose::new(body.into()).unwrap(),
                },
                references: Vec::new(),
            },
        };
        let birth = |id: usize, ctx: &Context, name: &str| Record {
            id: RecordId(id),
            timestamp: id as u64,
            context: ctx.clone(),
            event: Event::TaskCreated {
                name: Prose::new(name.into()).unwrap(),
            },
        };
        let world = World::replay(vec![
            birth(0, &human, "real work"),
            birth(1, &human, "other work"),
            note(2, &agent, TaskId(0), "triage: how is sections, undecided"),
            note(
                3,
                &human,
                TaskId(0),
                "no - pure tree, canvas verdict pending",
            ),
            note(4, &agent, TaskId(0), "noted, parked with owner"),
            note(5, &human, TaskId(1), "belongs to the other thread"),
        ])
        .unwrap();

        assert_eq!(world.comments.len(), 4);
        assert_eq!(
            world.comments[&CommentId(RecordId(2))].comment.home,
            TaskId(0)
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(5))].comment.home,
            TaskId(1)
        );

        let thread = thread_view(&world, TaskId(0)).unwrap();
        let nums: Vec<(usize, usize)> = thread
            .items
            .iter()
            .filter_map(|item| match item {
                ThreadItem::Note(line) => Some((line.num, line.seq)),
                _ => None,
            })
            .collect();
        assert_eq!(nums, vec![(0, 2), (1, 3), (2, 4)]);

        // public numbers are comment birth order, shared across threads
        assert_eq!(
            thread_view(&world, TaskId(1))
                .unwrap()
                .items
                .iter()
                .filter_map(|item| match item {
                    ThreadItem::Note(line) => Some(line.num),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![3]
        );

        // the board lists live tasks flat, in identity order
        let rows = board(&world);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "t-0");
        assert_eq!(rows[1].id, "t-1");
    }

    #[test]
    fn thread_index_places_records_on_their_home_threads() {
        let human = human();
        let world = World::replay(vec![
            Record {
                id: RecordId(0),
                timestamp: 0,
                context: human.clone(),
                event: Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            },
            Record {
                id: RecordId(1),
                timestamp: 1,
                context: human.clone(),
                event: Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Note {
                        body: Prose::new("first deposit".into()).unwrap(),
                    },
                    references: Vec::new(),
                },
            },
            Record {
                id: RecordId(2),
                timestamp: 2,
                context: human.clone(),
                event: Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Note {
                        body: Prose::new("a later note".into()).unwrap(),
                    },
                    references: vec![RecordId(1)],
                },
            },
            Record {
                id: RecordId(3),
                timestamp: 3,
                context: human.clone(),
                event: Event::ArtifactAdded {
                    root: TaskId(0),
                    artifact: Artifact {
                        name: Prose::new("sweep figure".into()).unwrap(),
                        hash: ContentHash::of(b"figure bytes"),
                    },
                },
            },
            Record {
                id: RecordId(4),
                timestamp: 4,
                context: human,
                event: Event::CommentRevised {
                    id: CommentId(RecordId(1)),
                    body: Prose::new("first deposit, corrected".into()).unwrap(),
                },
            },
        ])
        .unwrap();

        let index = thread_index(&world);
        assert_eq!(index[&0], TaskId(0));
        assert_eq!(index[&1], TaskId(0));
        assert_eq!(index[&2], TaskId(0));
        assert_eq!(index[&3], TaskId(0));
        // record facts ride the provenance index: the revision's home
        // is the comment it repairs
        assert_eq!(index[&4], TaskId(0));
    }

    mod panels {
        use super::super::*;
        use crate::events::Event;
        use crate::objects::comment::CommentId;
        use crate::store::{Context, Record, RecordId, Tier, World};
        use crate::types::actor::ActorName;
        use crate::types::artifact::Artifact;
        use crate::types::prose::Prose;
        use crate::{ContentHash, Scope};

        fn ctx(tier: Tier) -> Context {
            Context {
                actor: ActorName::new(
                    match tier {
                        Tier::Human => "human person",
                        Tier::Agent => "saccade bot",
                        Tier::System => "saccade",
                    }
                    .into(),
                )
                .unwrap(),
                tier,
                scope: None,
            }
        }

        fn record(seq: usize, at: u64, tier: Tier, event: Event) -> Record {
            Record {
                id: RecordId(seq),
                timestamp: at,
                context: ctx(tier),
                event,
            }
        }

        fn task_at(seq: usize, at: u64, name: &str) -> Record {
            record(
                seq,
                at,
                Tier::Human,
                Event::TaskCreated {
                    name: Prose::new(name.into()).unwrap(),
                },
            )
        }

        fn comment_at(
            seq: usize,
            at: u64,
            tier: Tier,
            home: TaskId,
            kind: CommentKind,
            references: Vec<RecordId>,
        ) -> Record {
            Record {
                id: RecordId(seq),
                timestamp: at,
                context: ctx(tier),
                event: Event::Commented {
                    home,
                    kind,
                    references,
                },
            }
        }

        fn offer_body(text: &str) -> Prose {
            Prose::new(text.into()).unwrap()
        }

        fn clustered() -> World {
            World::replay(vec![
                task_at(0, 0, "implement foo"),
                comment_at(
                    1,
                    10,
                    Tier::Human,
                    TaskId(0),
                    CommentKind::Note {
                        body: offer_body("parked finding"),
                    },
                    Vec::new(),
                ),
                comment_at(
                    2,
                    20,
                    Tier::Agent,
                    TaskId(0),
                    CommentKind::Note {
                        body: offer_body("the reply, citing [^1]"),
                    },
                    vec![RecordId(1)],
                ),
                record(
                    3,
                    30,
                    Tier::Human,
                    Event::ArtifactAdded {
                        root: TaskId(0),
                        artifact: Artifact {
                            name: Prose::new("sweep over 28 decades".into()).unwrap(),
                            hash: ContentHash::of(b"figure bytes"),
                        },
                    },
                ),
            ])
            .unwrap()
        }

        fn entry_shape(entry: &ThreadItem) -> (&'static str, usize) {
            match entry {
                ThreadItem::Note(line) => ("note", line.num),
                ThreadItem::Artifacts(lines) => ("artifacts", lines[0].seq),
            }
        }

        #[test]
        fn artifacts_cluster_within_themselves() {
            let world = clustered();
            let thread = thread_view(&world, TaskId(0)).unwrap();
            let shapes: Vec<_> = thread.items.iter().map(entry_shape).collect();
            assert_eq!(
                shapes,
                vec![("note", 0), ("note", 1), ("artifacts", 3)],
                "two flat notes then a trailing artifact cluster"
            );
        }

        /// A reply is a comment citing its parent: the mapping carries
        /// the edge, repeated markers resolve the same target.
        #[test]
        fn a_reply_is_a_citation_not_a_parent() {
            let world = clustered();
            let thread = thread_view(&world, TaskId(0)).unwrap();
            match &thread.items[1] {
                ThreadItem::Note(line) => {
                    assert_eq!(line.seq, 2);
                    assert_eq!(line.home, TaskId(0));
                    assert_eq!(
                        line.references,
                        vec![RefTarget::Comment { num: 0, task: 0 }]
                    );
                }
                other => panic!("expected a note, got {other:?}"),
            }
            // the same marker twice resolves the same target
            let world = World::replay(vec![
                task_at(0, 0, "implement foo"),
                comment_at(
                    1,
                    10,
                    Tier::Agent,
                    TaskId(0),
                    CommentKind::Note {
                        body: offer_body("parked finding"),
                    },
                    Vec::new(),
                ),
                comment_at(
                    2,
                    20,
                    Tier::Human,
                    TaskId(0),
                    CommentKind::Note {
                        body: offer_body("again [^1] and again [^1]"),
                    },
                    vec![RecordId(1), RecordId(1)],
                ),
            ])
            .unwrap();
            let thread = thread_view(&world, TaskId(0)).unwrap();
            match &thread.items[1] {
                ThreadItem::Note(line) => assert_eq!(
                    line.references,
                    vec![
                        RefTarget::Comment { num: 0, task: 0 },
                        RefTarget::Comment { num: 0, task: 0 }
                    ]
                ),
                other => panic!("expected a note, got {other:?}"),
            }
        }

        /// Citations may name task births, artifact births, and
        /// incarnation births — not only comments.
        #[test]
        fn citations_name_any_birth() {
            use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};
            let world = World::replay(vec![
                task_at(0, 0, "implement foo"),
                record(
                    1,
                    1,
                    Tier::System,
                    Event::WorkspaceEstablished {
                        task_id: TaskId(0),
                        base: GitCommit::new("abc123".into()).unwrap(),
                        branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                        worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
                    },
                ),
                record(
                    2,
                    2,
                    Tier::Human,
                    Event::ArtifactAdded {
                        root: TaskId(0),
                        artifact: Artifact {
                            name: Prose::new("sweep figure".into()).unwrap(),
                            hash: ContentHash::of(b"figure bytes"),
                        },
                    },
                ),
                record(
                    3,
                    1,
                    Tier::Human,
                    Event::TaskClaimed {
                        id: TaskId(0),
                        incarnation: None,
                    },
                ),
                comment_at(
                    4,
                    4,
                    Tier::Human,
                    TaskId(0),
                    CommentKind::Note {
                        body: offer_body(
                            "the task [^1], the worker [^2], the figure [^3], the claim [^4]",
                        ),
                    },
                    vec![RecordId(0), RecordId(1), RecordId(2), RecordId(3)],
                ),
            ])
            .unwrap();
            let thread = thread_view(&world, TaskId(0)).unwrap();
            match &thread.items[1] {
                ThreadItem::Note(line) => assert_eq!(
                    line.references,
                    vec![
                        RefTarget::Task { task: 0 },
                        RefTarget::Incarnation {
                            num: 0,
                            task: Some(0)
                        },
                        RefTarget::Artifact {
                            seq: 2,
                            task: 0,
                            name: "sweep figure".into(),
                            hash: ContentHash::of(b"figure bytes").as_str().to_string(),
                        },
                        RefTarget::Record {
                            seq: 3,
                            task: Some(0),
                        },
                    ]
                ),
                other => panic!("expected a note, got {other:?}"),
            }
        }

        #[test]
        fn the_rail_archive_holds_concluded_and_cancelled_only() {
            let world = World::replay(vec![
                task_at(0, 0, "concluded work"),
                task_at(1, 1, "cancelled work"),
                task_at(2, 2, "open work"),
                task_at(3, 3, "offered work"),
                comment_at(
                    4,
                    4,
                    Tier::Agent,
                    TaskId(0),
                    CommentKind::Offer {
                        body: offer_body("suite green"),
                    },
                    Vec::new(),
                ),
                comment_at(
                    5,
                    5,
                    Tier::Human,
                    TaskId(0),
                    CommentKind::Accept {
                        offer: CommentId(RecordId(4)),
                    },
                    Vec::new(),
                ),
                record(
                    6,
                    6,
                    Tier::Human,
                    Event::TaskCancelled {
                        id: TaskId(1),
                        explanation: offer_body("no longer wanted"),
                    },
                ),
                comment_at(
                    7,
                    7,
                    Tier::Agent,
                    TaskId(3),
                    CommentKind::Offer {
                        body: offer_body("awaiting its consumer"),
                    },
                    Vec::new(),
                ),
            ])
            .unwrap();

            let rows = closed_tasks(&world);
            assert_eq!(
                rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
                vec!["t-1", "t-0"],
                "newest close first"
            );
            // the board holds the live pair
            let live: Vec<_> = board(&world).into_iter().map(|r| r.id).collect();
            assert_eq!(live, vec!["t-2".to_string(), "t-3".to_string()]);
        }

        #[test]
        fn a_revised_body_renders_with_the_mark() {
            let world = World::replay(vec![
                task_at(0, 0, "implement foo"),
                comment_at(
                    1,
                    10,
                    Tier::Agent,
                    TaskId(0),
                    CommentKind::Offer {
                        body: offer_body("suite green, 55 tests"),
                    },
                    Vec::new(),
                ),
                Record {
                    id: RecordId(2),
                    timestamp: 20,
                    context: ctx(Tier::Human),
                    event: Event::CommentRevised {
                        id: CommentId(RecordId(1)),
                        body: offer_body("suite green, 56 tests"),
                    },
                },
            ])
            .unwrap();

            let line = comment_line(&world, CommentId(RecordId(1))).unwrap();
            assert_eq!(line.kind, "offer");
            assert_eq!(line.body, "suite green, 56 tests");
            assert!(line.revised);
            assert_eq!(line.reviser.as_deref(), Some("human person"));
        }

        #[test]
        fn a_withdrawn_note_renders_its_tombstone() {
            let world = World::replay(vec![
                task_at(0, 0, "implement foo"),
                comment_at(
                    1,
                    10,
                    Tier::Agent,
                    TaskId(0),
                    CommentKind::Note {
                        body: offer_body("parked mid-flight"),
                    },
                    Vec::new(),
                ),
                Record {
                    id: RecordId(2),
                    timestamp: 20,
                    context: ctx(Tier::Human),
                    event: Event::CommentWithdrawn {
                        id: CommentId(RecordId(1)),
                        note: Prose::new("the parking was premature".into()).unwrap(),
                    },
                },
            ])
            .unwrap();

            let line = comment_line(&world, CommentId(RecordId(1))).unwrap();
            assert!(line.withdrawn);
            assert_eq!(line.withdrawer.as_deref(), Some("human person"));
            assert_eq!(
                line.withdrawal_note.as_deref(),
                Some("the parking was premature")
            );
            assert_eq!(line.kind, "note");
        }

        #[test]
        fn scope_rides_the_comment_context_as_disclosure() {
            let scoped = Context {
                actor: ActorName::new("pi".into()).unwrap(),
                tier: Tier::Agent,
                scope: Some(Scope {
                    task: Some(TaskId(0)),
                    incarnation: None,
                    workspace: None,
                }),
            };
            let world = World::replay(vec![
                task_at(0, 0, "implement foo"),
                Record {
                    id: RecordId(1),
                    timestamp: 10,
                    context: scoped,
                    event: Event::Commented {
                        home: TaskId(0),
                        kind: CommentKind::Note {
                            body: offer_body("scoped helper's note"),
                        },
                        references: Vec::new(),
                    },
                },
            ])
            .unwrap();

            // a scoped helper contributes without owning: no claim, no owner
            assert!(world.tasks[0].ownership.is_empty());
            let line = comment_line(&world, CommentId(RecordId(1))).unwrap();
            assert_eq!(line.actor, "pi");
        }

        /// The use line names its source task and, once bound, the
        /// selected conclusion with its grounds.
        #[test]
        fn a_use_renders_its_source_and_binding() {
            let world = World::replay(vec![
                task_at(0, 0, "consume the account"),
                task_at(1, 1, "produce the account"),
                comment_at(
                    2,
                    2,
                    Tier::Agent,
                    TaskId(1),
                    CommentKind::Offer {
                        body: offer_body("suite green"),
                    },
                    Vec::new(),
                ),
                comment_at(
                    3,
                    3,
                    Tier::Agent,
                    TaskId(0),
                    CommentKind::Use {
                        source: TaskId(1),
                        reason: offer_body("builds on the account"),
                        bound: None,
                    },
                    Vec::new(),
                ),
                Record {
                    id: RecordId(4),
                    timestamp: 4,
                    context: ctx(Tier::Human),
                    event: Event::UseBound {
                        id: CommentId(RecordId(3)),
                        offer: CommentId(RecordId(2)),
                        grounds: offer_body("the account that landed"),
                    },
                },
            ])
            .unwrap();

            let pending_world = World::replay(vec![
                task_at(0, 0, "consume the account"),
                task_at(1, 1, "produce the account"),
                comment_at(
                    2,
                    2,
                    Tier::Agent,
                    TaskId(1),
                    CommentKind::Offer {
                        body: offer_body("suite green"),
                    },
                    Vec::new(),
                ),
                comment_at(
                    3,
                    3,
                    Tier::Agent,
                    TaskId(0),
                    CommentKind::Use {
                        source: TaskId(1),
                        reason: offer_body("builds on the account"),
                        bound: None,
                    },
                    Vec::new(),
                ),
            ])
            .unwrap();
            let pending = comment_line(&pending_world, CommentId(RecordId(3))).unwrap();
            assert_eq!(pending.kind, "use");
            assert_eq!(pending.source, Some(TaskId(1)));
            assert_eq!(pending.binding, None);
            assert_eq!(pending.body, "builds on the account");

            let bound = comment_line(&world, CommentId(RecordId(3))).unwrap();
            assert_eq!(bound.kind, "use");
            assert_eq!(
                bound.binding.map(|b| (b.offer, b.grounds)),
                Some((0, "the account that landed".to_string()))
            );
        }
    }

    mod search {
        use super::super::*;
        use crate::ContentHash;
        use crate::events::Event;
        use crate::objects::comment::CommentId;
        use crate::store::{Context, Record, RecordId, Tier, World};
        use crate::types::actor::ActorName;
        use crate::types::artifact::Artifact;
        use crate::types::prose::Prose;

        fn ctx(tier: Tier, name: &str) -> Context {
            Context {
                actor: ActorName::new(name.into()).unwrap(),
                tier,
                scope: None,
            }
        }

        fn record(seq: usize, at: u64, ctx: &Context, event: Event) -> Record {
            Record {
                id: RecordId(seq),
                timestamp: at,
                context: ctx.clone(),
                event,
            }
        }

        fn task(seq: usize, at: u64, ctx: &Context, name: &str) -> Record {
            record(
                seq,
                at,
                ctx,
                Event::TaskCreated {
                    name: Prose::new(name.into()).unwrap(),
                },
            )
        }

        fn comment(
            seq: usize,
            at: u64,
            ctx: &Context,
            home: TaskId,
            kind: CommentKind,
            references: Vec<RecordId>,
        ) -> Record {
            record(
                seq,
                at,
                ctx,
                Event::Commented {
                    home,
                    kind,
                    references,
                },
            )
        }

        fn note(seq: usize, at: u64, ctx: &Context, home: TaskId, body: &str) -> Record {
            comment(
                seq,
                at,
                ctx,
                home,
                CommentKind::Note {
                    body: Prose::new(body.into()).unwrap(),
                },
                Vec::new(),
            )
        }

        /// The story: comments at seq 2,3,5,6,8,9 carry public numbers
        /// 0,1,2,3,4,5. c-3 and c-9 cite the finding at seq 2 (c-0);
        /// c-8 names it in prose.
        fn story() -> World {
            let jerry = ctx(Tier::Human, "jerry");
            let bot = ctx(Tier::Agent, "saccade bot");
            let pi = ctx(Tier::Agent, "pi");
            World::replay(vec![
                task(0, 0, &jerry, "implement foo"),
                task(1, 10, &bot, "migrate floop"),
                note(
                    2,
                    20,
                    &jerry,
                    TaskId(0),
                    "parked: the fluke blocks foo\ntelemetries abound here",
                ),
                comment(
                    3,
                    30,
                    &pi,
                    TaskId(0),
                    CommentKind::Note {
                        body: Prose::new("the fluke is real; floop proceeds".into()).unwrap(),
                    },
                    vec![RecordId(2)],
                ),
                task(4, 40, &jerry, "t-490 the id boundary guard"),
                note(5, 50, &pi, TaskId(1), "t-0 waits on this thread"),
                note(
                    6,
                    60,
                    &pi,
                    TaskId(1),
                    "the run answered\nsee c-123 for the trail",
                ),
                record(
                    7,
                    70,
                    &bot,
                    Event::TaskClaimed {
                        id: TaskId(1),
                        incarnation: None,
                    },
                ),
                comment(
                    8,
                    80,
                    &bot,
                    TaskId(1),
                    CommentKind::Offer {
                        body: Prose::new(
                            "suite 9 green; the floop migration landed per c-0".into(),
                        )
                        .unwrap(),
                    },
                    Vec::new(),
                ),
                comment(
                    9,
                    90,
                    &bot,
                    TaskId(0),
                    CommentKind::Note {
                        body: Prose::new("done, per the plan above".into()).unwrap(),
                    },
                    vec![RecordId(2)],
                ),
            ])
            .unwrap()
        }

        fn q(tokens: &[&str]) -> SearchQuery {
            let owned: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
            SearchQuery::parse(&owned).unwrap()
        }

        fn pointers(groups: &[SearchGroup]) -> Vec<(usize, Vec<String>)> {
            groups
                .iter()
                .map(|g| {
                    (
                        g.task,
                        g.records
                            .iter()
                            .map(|r| r.pointer.clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect()
        }

        #[test]
        fn artifact_names_are_searchable_and_kinded() {
            let jerry = ctx(Tier::Human, "jerry");
            let world = World::replay(vec![
                task(0, 0, &jerry, "hold the sweep"),
                record(
                    1,
                    1,
                    &jerry,
                    Event::ArtifactAdded {
                        root: TaskId(0),
                        artifact: Artifact {
                            name: Prose::new("sweep-overview".into()).unwrap(),
                            hash: ContentHash::of(b"figure bytes"),
                        },
                    },
                ),
                comment(
                    2,
                    2,
                    &jerry,
                    TaskId(0),
                    CommentKind::Note {
                        body: Prose::new("the verdict cites the figure above".into()).unwrap(),
                    },
                    vec![RecordId(1)],
                ),
            ])
            .unwrap();
            // the name matches as a whole token; the title's "sweep" does not
            let groups = search(&world, &q(&["sweep-overview"])).unwrap();
            assert_eq!(pointers(&groups), vec![(0, vec!["seq 1".into()])]);
            let groups = &groups[0];
            assert_eq!(groups.records[0].kind, "artifact");
            assert_eq!(groups.records[0].actor, None);
            assert_eq!(groups.records[0].body, "sweep-overview");
            // the kind facet narrows to artifacts alone
            let groups = search(&world, &q(&["sweep-overview", "kind:artifact"])).unwrap();
            assert_eq!(pointers(&groups), vec![(0, vec!["seq 1".into()])]);
            // the citing note's prose names no term of the query: a
            // citation's target is a pointer, not a text match
        }

        /// An id term is a reference search: structural citations match,
        /// prose mentions match, and the pointers carry public numbers.
        #[test]
        fn an_id_term_is_a_reference_search() {
            let world = story();
            // c-0 (seq 2): cited by c-3 and c-9 through their mappings,
            // named in prose by c-8
            assert_eq!(
                pointers(&search(&world, &q(&["c-0"])).unwrap()),
                vec![
                    (0, vec!["c-1".into(), "c-5".into()]),
                    (1, vec!["c-4".into()])
                ]
            );
            // t-0: the notes home on it, c-2 names it in prose
            assert_eq!(
                pointers(&search(&world, &q(&["t-0"])).unwrap()),
                vec![
                    (0, vec!["c-0".into(), "c-1".into(), "c-5".into()]),
                    (1, vec!["c-2".into()])
                ]
            );
            // a term naming no comment of this fold matches only prose
            assert_eq!(
                pointers(&search(&world, &q(&["c-9"])).unwrap()),
                Vec::<(usize, Vec<String>)>::new()
            );
        }

        #[test]
        fn a_revised_body_is_what_search_reads() {
            let jerry = ctx(Tier::Human, "jerry");
            let world = World::replay(vec![
                task(0, 0, &jerry, "hold the verdict"),
                note(2, 2, &jerry, TaskId(0), "the floop verdict lands"),
                record(
                    3,
                    3,
                    &jerry,
                    Event::CommentRevised {
                        id: CommentId(RecordId(2)),
                        body: Prose::new("the blorp verdict lands".into()).unwrap(),
                    },
                ),
            ])
            .unwrap();
            // the fold's latest body matches, the orphaned original does not
            assert_eq!(
                pointers(&search(&world, &q(&["blorp"])).unwrap()),
                vec![(0, vec!["c-0".into()])]
            );
            assert!(search(&world, &q(&["floop"])).unwrap().is_empty());
        }

        #[test]
        fn a_withdrawn_body_is_omitted_until_the_flag_lets_it_back() {
            let jerry = ctx(Tier::Human, "jerry");
            let pi = ctx(Tier::Agent, "pi");
            let world = World::replay(vec![
                task(0, 0, &jerry, "hold the verdict"),
                note(2, 2, &pi, TaskId(0), "the floop verdict lands"),
                record(
                    3,
                    3,
                    &jerry,
                    Event::CommentWithdrawn {
                        id: CommentId(RecordId(2)),
                        note: Prose::new("parked in the wrong place".into()).unwrap(),
                    },
                ),
                // a later record still cites the withdrawn comment: the
                // reference must never dangle
                comment(
                    4,
                    4,
                    &jerry,
                    TaskId(0),
                    CommentKind::Note {
                        body: Prose::new("the verdict lives on".into()).unwrap(),
                    },
                    vec![RecordId(2)],
                ),
            ])
            .unwrap();
            // the withdrawn body matches nothing by default — and the
            // tombstone's note never speaks for the body it evicted
            assert!(search(&world, &q(&["floop"])).unwrap().is_empty());
            assert!(search(&world, &q(&["parked"])).unwrap().is_empty());
            // the flag lets the body back in
            let mut flagged = q(&["floop"]);
            flagged.withdrawn = true;
            assert_eq!(
                pointers(&search(&world, &flagged).unwrap()),
                vec![(0, vec!["c-0".into()])]
            );
            // the citation still resolves: c-0's reference search finds
            // the record that cites it, withdrawn or not
            assert_eq!(
                pointers(&search(&world, &q(&["c-0"])).unwrap()),
                vec![(0, vec!["c-1".into()])]
            );
        }

        #[test]
        fn facets_narrow_with_visible_counts() {
            let world = story();
            // in: only t-0's records show; t-1 stays visible as a count row
            let groups = search(&world, &q(&["floop", "in:t-0"])).unwrap();
            assert_eq!(
                groups
                    .iter()
                    .map(|g| (g.task, g.records.len(), g.total))
                    .collect::<Vec<_>>(),
                vec![(1, 0, 2), (0, 1, 1)]
            );
            // by: only pi's notes match; the other authors stay as counts
            let groups = search(&world, &q(&["floop", "by:pi"])).unwrap();
            assert_eq!(
                groups
                    .iter()
                    .map(|g| (g.task, g.records.len(), g.total))
                    .collect::<Vec<_>>(),
                vec![(1, 0, 2), (0, 1, 1)]
            );
        }

        #[test]
        fn kind_facets_read_the_variant_vocabulary() {
            let world = story();
            let groups = search(&world, &q(&["floop", "kind:offer"])).unwrap();
            assert_eq!(
                pointers(&groups),
                vec![(1, vec!["c-4".into()]), (0, vec![])],
                "only the offer shows; the other thread holds its count"
            );
            let groups = search(&world, &q(&["kind:accept"])).unwrap();
            assert!(groups.is_empty());
            // the vocabulary is closed
            assert!(SearchQuery::parse(&["kind:demand".to_string()]).is_err());
        }

        #[test]
        fn an_unknown_task_facet_names_the_holds() {
            let world = story();
            let err = search(&world, &q(&["floop", "in:t-9"])).unwrap_err();
            assert!(err.to_string().contains("t-0 through t-2"), "{err}");
        }

        #[test]
        fn plain_terms_read_titles_bodies_and_explanations() {
            let jerry = ctx(Tier::Human, "jerry");
            let world = World::replay(vec![
                task(0, 0, &jerry, "implement foo"),
                record(
                    1,
                    1,
                    &jerry,
                    Event::TaskCancelled {
                        id: TaskId(0),
                        explanation: Prose::new("the funding for foo dissolved".into()).unwrap(),
                    },
                ),
            ])
            .unwrap();
            let groups = search(&world, &q(&["funding"])).unwrap();
            assert_eq!(pointers(&groups), vec![(0, vec!["t-0 explanation".into()])]);
        }
    }

    // -- the wait door's structural matching -------------------------

    mod wait {
        use super::*;

        fn ctx(name: &str) -> Context {
            Context {
                actor: ActorName::new(name.into()).unwrap(),
                tier: Tier::Human,
                scope: None,
            }
        }

        fn record(seq: usize, event: Event) -> (usize, Event) {
            (seq, event)
        }

        fn task(seq: usize, name: &str) -> (usize, Event) {
            record(
                seq,
                Event::TaskCreated {
                    name: Prose::new(name.into()).unwrap(),
                },
            )
        }

        fn comment(
            seq: usize,
            home: TaskId,
            kind: CommentKind,
            references: Vec<RecordId>,
        ) -> (usize, Event) {
            record(
                seq,
                Event::Commented {
                    home,
                    kind,
                    references,
                },
            )
        }

        fn note(seq: usize, home: TaskId, body: &str) -> (usize, Event) {
            comment(
                seq,
                home,
                CommentKind::Note {
                    body: Prose::new(body.into()).unwrap(),
                },
                Vec::new(),
            )
        }

        fn note_on(seq: usize, home: TaskId, body: &str, refs: Vec<RecordId>) -> (usize, Event) {
            comment(
                seq,
                home,
                CommentKind::Note {
                    body: Prose::new(body.into()).unwrap(),
                },
                refs,
            )
        }

        // t-0 holds the waited offer, its sibling note, a citing note,
        // revisions, a withdrawal, the accept, a claim, an artifact; t-1
        // a rejected offer and its concluded successor, a bound use
        fn story() -> (World, Vec<(usize, Event)>) {
            let events = vec![
                task(0, "implement foo"),
                comment(
                    1,
                    TaskId(0),
                    CommentKind::Offer {
                        body: Prose::new("suite green, ready for review".into()).unwrap(),
                    },
                    Vec::new(),
                ),
                note_on(2, TaskId(0), "sibling chatter", Vec::new()),
                note_on(
                    3,
                    TaskId(0),
                    "one question first, citing the offer [^1]",
                    vec![RecordId(1)],
                ),
                task(4, "migrate floop"),
                note_on(
                    5,
                    TaskId(0),
                    "answered in the body; see c-0 for the offer",
                    Vec::new(),
                ),
                (
                    6,
                    Event::CommentRevised {
                        id: CommentId(RecordId(5)),
                        body: Prose::new("answered more precisely".into()).unwrap(),
                    },
                ),
                comment(
                    7,
                    TaskId(0),
                    CommentKind::Accept {
                        offer: CommentId(RecordId(1)),
                    },
                    Vec::new(),
                ),
                note(8, TaskId(1), "the other thread"),
                comment(
                    9,
                    TaskId(1),
                    CommentKind::Use {
                        source: TaskId(0),
                        reason: Prose::new("floop rests on foo".into()).unwrap(),
                        bound: None,
                    },
                    Vec::new(),
                ),
                (
                    10,
                    Event::TaskClaimed {
                        id: TaskId(0),
                        incarnation: None,
                    },
                ),
                (
                    11,
                    Event::ArtifactAdded {
                        root: TaskId(0),
                        artifact: Artifact {
                            name: Prose::new("sweep figure".into()).unwrap(),
                            hash: ContentHash::of(b"figure bytes"),
                        },
                    },
                ),
                (
                    12,
                    Event::CommentRevised {
                        id: CommentId(RecordId(1)),
                        body: Prose::new("the account, restated".into()).unwrap(),
                    },
                ),
                (
                    13,
                    Event::CommentWithdrawn {
                        id: CommentId(RecordId(3)),
                        note: Prose::new("asked in the wrong place".into()).unwrap(),
                    },
                ),
                comment(
                    14,
                    TaskId(1),
                    CommentKind::Offer {
                        body: Prose::new("floop migrated, review welcome".into()).unwrap(),
                    },
                    Vec::new(),
                ),
                comment(
                    15,
                    TaskId(1),
                    CommentKind::Reject {
                        offer: CommentId(RecordId(14)),
                        grounds: Prose::new("the floop index is missing".into()).unwrap(),
                    },
                    Vec::new(),
                ),
                comment(
                    16,
                    TaskId(1),
                    CommentKind::Offer {
                        body: Prose::new("the index rides along now".into()).unwrap(),
                    },
                    Vec::new(),
                ),
                comment(
                    17,
                    TaskId(1),
                    CommentKind::Accept {
                        offer: CommentId(RecordId(16)),
                    },
                    Vec::new(),
                ),
                (
                    18,
                    Event::UseBound {
                        id: CommentId(RecordId(9)),
                        offer: CommentId(RecordId(1)),
                        grounds: Prose::new("the account that landed".into()).unwrap(),
                    },
                ),
            ];
            let records = events
                .iter()
                .map(|(seq, event)| Record {
                    id: RecordId(*seq),
                    timestamp: *seq as u64,
                    context: ctx("jerry"),
                    event: event.clone(),
                })
                .collect();
            (World::replay(records).unwrap(), events)
        }

        fn matching(
            world: &World,
            events: &[(usize, Event)],
            target: &Target,
            until: WaitUntil,
        ) -> Vec<usize> {
            events
                .iter()
                .filter(|(seq, event)| wait_matches(world, RecordId(*seq), event, target, until))
                .map(|(seq, _)| *seq)
                .collect()
        }

        // the task wait matches its own records and never another task's
        #[test]
        fn the_task_wait_matches_exactly_its_own_records() {
            let (world, events) = story();
            assert_eq!(
                matching(
                    &world,
                    &events,
                    &Target::Task(TaskId(0)),
                    WaitUntil::Activity
                ),
                vec![0, 1, 2, 3, 5, 6, 7, 10, 11, 12, 13]
            );
            assert_eq!(
                matching(
                    &world,
                    &events,
                    &Target::Task(TaskId(1)),
                    WaitUntil::Activity
                ),
                vec![4, 8, 9, 14, 15, 16, 17, 18]
            );
        }

        // the comment wait wakes on the comment's own repairs, the
        // records citing it, and the decisions naming it — not the
        // birth, sibling, prose-only mention, or a binding elsewhere
        #[test]
        fn the_comment_wait_matches_only_its_structural_edges() {
            let (world, events) = story();
            let offer = Target::Comment(CommentId(RecordId(1)));
            assert_eq!(
                matching(&world, &events, &offer, WaitUntil::Activity),
                vec![3, 7, 12, 13]
            );
            // a binding names its use's selection, never the offer:
            // waiting on the use wakes, waiting on the offer does not
            let use_target = Target::Comment(CommentId(RecordId(9)));
            assert_eq!(
                matching(&world, &events, &use_target, WaitUntil::Activity),
                vec![18]
            );
            // a prose-only mention (seq 5 says "see c-0") is not an edge
            // of the comment it names: only structural citations match
            let sibling = Target::Comment(CommentId(RecordId(2)));
            assert_eq!(
                matching(&world, &events, &sibling, WaitUntil::Activity),
                Vec::<usize>::new()
            );
        }

        // the offer wait matches every offer birth and nothing else
        #[test]
        fn the_offer_wait_matches_every_offer_birth_and_nothing_else() {
            let (world, events) = story();
            assert_eq!(
                matching(&world, &events, &Target::Task(TaskId(0)), WaitUntil::Offer),
                vec![1]
            );
            assert_eq!(
                matching(&world, &events, &Target::Task(TaskId(1)), WaitUntil::Offer),
                vec![14, 16]
            );
            assert_eq!(
                matching(
                    &world,
                    &events,
                    &Target::Comment(CommentId(RecordId(1))),
                    WaitUntil::Offer
                ),
                Vec::<usize>::new()
            );
        }
    }

    // -- the message fold's numbers and mailbox ----------------------

    mod messages {
        use super::*;
        use crate::objects::message::MessageState;
        use crate::store::Scope;
        use crate::types::pointers::{NativeSessionId, SessionLocator};

        fn ctx(name: &str) -> Context {
            Context {
                actor: ActorName::new(name.into()).unwrap(),
                tier: Tier::Human,
                scope: None,
            }
        }

        fn registered(_seq: usize, id: &str) -> Event {
            Event::IncarnationRegistered {
                native_id: NativeSessionId::new(id.into()).unwrap(),
                saved_session: SessionLocator::new(format!("/sessions/{id}.jsonl").into()).unwrap(),
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

        fn record(seq: usize, context: &Context, event: Event) -> Record {
            Record {
                id: RecordId(seq),
                timestamp: seq as u64,
                context: context.clone(),
                event,
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

        /// Messages take no thread home: a task context rides without
        /// publication, and the provenance index stays None.
        #[test]
        fn messages_take_no_thread_home() {
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
            // the task's thread stays empty whatever context rode
            assert!(world.tasks[0].thread.is_empty());
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
