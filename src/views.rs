//! Statements: the render shapes and their constructors, one per reader.
//! Views read objects and contexts; nothing in objects knows views exist.

use std::collections::BTreeMap;

use crate::events::Event;
use crate::objects::comment::{CommentContext, CommentId, CommentKind, CommentState, Target};
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestFacts {
    pub seq: usize,
    pub task: usize,
    pub workspace: Option<WorkspaceFacts>,
}

pub fn request_facts(world: &World, position: RecordId) -> Option<RequestFacts> {
    let comment = world.comments.get(&CommentId(position))?;
    let task = comment.comment.root.0;
    let workspace = world.tasks[task]
        .workspace
        .as_ref()
        .map(|w| WorkspaceFacts {
            task,
            base: w.workspace.base.as_str().to_string(),
            branch: w.workspace.branch.as_str().to_string(),
            path: w.workspace.worktree.as_path().display().to_string(),
        });
    Some(RequestFacts {
        seq: position.0,
        task,
        workspace,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitUntil {
    Activity,
    Offer,
}

/// A use never matches a comment wait: matching follows targets, and
/// a use targets its consuming task, never the offer it names.
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
                target: born,
            } if *born == *target
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
        Event::Commented { target: born, .. } => match target {
            Target::Task(task) => born_task(world, at) == Some(*task),
            Target::Comment(watched) => chain_contains(world, *born, *watched),
        },
        Event::CommentRevised { id, .. } | Event::CommentWithdrawn { id, .. } => match target {
            Target::Task(task) => world
                .comments
                .get(id)
                .is_some_and(|c| c.comment.root == *task),
            Target::Comment(watched) => {
                *id == *watched || chain_contains(world, target_of(world, *id), *watched)
            }
        },
        Event::ArtifactAdded { root, .. } => *target == Target::Task(*root),
        Event::WorkspaceEstablished { task_id, .. }
        | Event::TaskWorkspaceCheckpointed { task_id, .. } => *target == Target::Task(*task_id),
    }
}

fn born_task(world: &World, at: RecordId) -> Option<TaskId> {
    world.comments.get(&CommentId(at)).map(|c| c.comment.root)
}

fn target_of(world: &World, id: CommentId) -> Target {
    world
        .comments
        .get(&id)
        .expect("a chain member is resident")
        .comment
        .target
}

fn chain_contains(world: &World, born: Target, watched: CommentId) -> bool {
    let mut up = born;
    loop {
        match up {
            Target::Task(_) => return false,
            Target::Comment(id) => {
                if id == watched {
                    return true;
                }
                up = target_of(world, id);
            }
        }
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
    pub fn of(id: TaskId, ctx: &TaskContext) -> TaskView {
        TaskView {
            id: format!("t-{}", id.0),
            state: state_str(&ctx.task.state),
            name: ctx.task.name.as_str().to_string(),
            offer: ctx.current_offer.map(|c| c.0.0),
            owner: current_owner(ctx).map(|o| owner_str(&o.owner)),
            n_comments: ctx.thread.len(),
        }
    }
}

/// The ownership ledger's latest entry, the current holder.
pub fn current_owner(ctx: &TaskContext) -> Option<&OwnerEntry> {
    ctx.ownership.last()
}

fn owner_str(owner: &Owner) -> String {
    match owner {
        Owner::Human(actor) | Owner::External(actor) => actor.as_str().to_string(),
        Owner::Incarnation(id) => format!("i-{}", id.0.0),
    }
}

/// A row of the task's thread view. Depth is position in the rendered
/// thread — the task is the root, so a comment addressing it sits at
/// 1 — and target is the edge the comment carries: the task it
/// addresses or the comment it replies to.
#[derive(Debug)]
pub struct CommentLine {
    pub seq: usize,
    pub target: Target,
    pub depth: usize,
    pub actor: String,
    pub tier: String,
    pub kind: &'static str,
    pub body: String,
    pub offer: Option<usize>,
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

/// One conversation: a comment plus every reply hanging off it, in
/// record order. Bounded by the reply links, never by the renderer.
pub struct Conversation {
    pub root: CommentLine,
    pub replies: Vec<CommentLine>,
}

/// The focused task's thread in record order by first comment: a
/// conversation, or one unlinked annotation.
#[derive(Debug)]
pub enum ThreadItem {
    Group {
        root: CommentLine,
        replies: Vec<CommentLine>,
    },
    Note(CommentLine),
    /// Artifacts clustered within themselves: consecutive artifact
    /// records as one item, never attached to a comment's group —
    /// association with a comment is what citing is for.
    Artifacts(Vec<ArtifactLine>),
}

/// The focused task's thread as a view: the context's pointer index,
/// followed, grouped by reply links, ordered by first comment.
pub struct ThreadView {
    pub items: Vec<ThreadItem>,
}

/// The variant a comment carries — the kind never moves; a withdrawn
/// deposit is still the note it was born.
pub fn kind_of(kind: &CommentKind) -> &'static str {
    match kind {
        CommentKind::Note { .. } => "note",
        CommentKind::Request { .. } => "request",
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
    let mut depth = 1;
    let mut up = cctx.comment.target;
    while let Target::Comment(parent) = up {
        depth += 1;
        up = world
            .comments
            .get(&parent)
            .expect("apply received a dangling comment target")
            .comment
            .target;
    }
    CommentLine {
        seq: cid.0.0,
        target: cctx.comment.target,
        depth,
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
        offer: cctx.comment.kind.offer().map(|o| o.0.0),
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
    pub fn of(id: TaskId, ctx: &TaskContext) -> ShowView {
        ShowView {
            id: format!("t-{}", id.0),
            state: state_str(&ctx.task.state),
            name: ctx.task.name.as_str().to_string(),
            explanation: match &ctx.task.state {
                TaskState::Cancelled(why) => Some(why.as_str().to_string()),
                _ => None,
            },
            owner: current_owner(ctx).map(|o| owner_str(&o.owner)),
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
    world.tasks.get(id.0).map(|ctx| TaskView::of(id, ctx))
}

pub fn show_view(world: &World, id: TaskId) -> Option<ShowView> {
    world.tasks.get(id.0).map(|ctx| ShowView::of(id, ctx))
}

/// Follow the world's pointers to one task's thread: the conversation
/// each root opened, and the annotations no one replied to.
pub fn thread_view(world: &World, id: TaskId) -> Option<ThreadView> {
    let ctx = world.tasks.get(id.0)?;
    // building groups in record order; replies always come after their
    // root, so one pass suffices.
    enum Building {
        Group {
            root: CommentLine,
            replies: Vec<CommentLine>,
        },
    }
    enum Member {
        Comment(CommentId),
        Artifact(RecordId, Artifact),
    }
    // the thread's utterances merged by position; artifacts never join
    // a comment's group. Consecutive artifacts accumulate into one
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

    let mut groups: Vec<Building> = Vec::new();
    let mut root_index: BTreeMap<CommentId, usize> = BTreeMap::new();
    let mut clusters: Vec<(usize, Vec<ArtifactLine>)> = Vec::new();
    let mut open_cluster: Option<(usize, Vec<ArtifactLine>)> = None;
    for (_, member) in stream {
        match member {
            Member::Comment(cid) => {
                if let Some(cluster) = open_cluster.take() {
                    clusters.push(cluster);
                }
                let root = root_of(&world.comments, cid);
                if root == cid {
                    let line = line_of(world, cid);
                    root_index.insert(cid, groups.len());
                    groups.push(Building::Group {
                        root: line,
                        replies: Vec::new(),
                    });
                } else {
                    let g = root_index
                        .get(&root)
                        .copied()
                        .expect("a reply's root is resident and earlier");
                    let line = line_of(world, cid);
                    match &mut groups[g] {
                        Building::Group { replies, .. } => replies.push(line),
                    }
                }
            }
            Member::Artifact(rid, artifact) => {
                let line = artifact_line_of(rid.0, &artifact);
                match &mut open_cluster {
                    Some((_, lines)) => lines.push(line),
                    None => open_cluster = Some((rid.0, vec![line])),
                }
            }
        }
    }
    if let Some(cluster) = open_cluster.take() {
        clusters.push(cluster);
    }

    let group_items: Vec<(usize, ThreadItem)> = groups
        .into_iter()
        .map(|g| match g {
            Building::Group { root, replies } => {
                if replies.is_empty() {
                    (root.seq, ThreadItem::Note(root))
                } else {
                    (root.seq, ThreadItem::Group { root, replies })
                }
            }
        })
        .collect();
    let cluster_items: Vec<(usize, ThreadItem)> = clusters
        .into_iter()
        .map(|(seq, lines)| (seq, ThreadItem::Artifacts(lines)))
        .collect();
    // both lists are sorted by their first seq; one merge orders the items
    let mut items: Vec<ThreadItem> = Vec::with_capacity(group_items.len() + cluster_items.len());
    let mut groups_iter = group_items.into_iter().peekable();
    let mut clusters_iter = cluster_items.into_iter().peekable();
    loop {
        let take_group = match (groups_iter.peek(), clusters_iter.peek()) {
            (None, None) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some((gseq, _)), Some((cseq, _))) => gseq < cseq,
        };
        items.push(if take_group {
            groups_iter.next().expect("peeked").1
        } else {
            clusters_iter.next().expect("peeked").1
        });
    }
    Some(ThreadView { items })
}

/// The chain member whose target is the task — cid itself when it is a
/// root. Chains end at the task; the fold guarantees residency.
fn root_of(comments: &BTreeMap<CommentId, CommentContext>, cid: CommentId) -> CommentId {
    let mut current = cid;
    while let Target::Comment(parent) = comments[&current].comment.target {
        current = parent;
    }
    current
}

/// The board: live tasks in identity order, flat. An offered task
/// still takes findings and its accept.
pub fn board(world: &World) -> Vec<TaskView> {
    world
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, ctx)| matches!(ctx.task.state, TaskState::Open | TaskState::Offered))
        .map(|(i, ctx)| TaskView::of(TaskId(i), ctx))
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
        .map(|(i, ctx)| (ctx.last_record_at, i, TaskView::of(TaskId(i), ctx)))
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    rows.into_iter().map(|(_, _, row)| row).collect()
}

// -- the reading-side reference resolver's index --------------------

/// What a 'c-N' mention resolves to on the reading side: a comment's
/// home anchor, or an artifact's home card. The mention parse itself
/// lives in the view, never in the record.
#[derive(Clone, Debug, PartialEq)]
pub enum RefTarget {
    Comment {
        task: usize,
    },
    Artifact {
        task: usize,
        name: String,
        hash: String,
    },
}

/// Every seq the fold holds as a comment or an artifact, mapped to its
/// mention's render — the index the webui's body renders resolve
/// against. Births and machinery records stay unresolvable on purpose:
/// an unresolved token renders as plain text.
pub fn ref_index(world: &World) -> BTreeMap<usize, RefTarget> {
    let mut index = BTreeMap::new();
    for (id, cctx) in &world.comments {
        index.insert(
            id.0.0,
            RefTarget::Comment {
                task: cctx.comment.root.0,
            },
        );
    }
    for (i, ctx) in world.tasks.iter().enumerate() {
        for (rid, artifact) in &ctx.artifacts {
            index.insert(
                rid.0,
                RefTarget::Artifact {
                    task: i,
                    name: artifact.name.as_str().to_string(),
                    hash: artifact.hash.as_str().to_string(),
                },
            );
        }
    }
    index
}

/// Each record's home thread, where the fold gives it one: a task by
/// its birth, a comment by its root — replies included — an artifact
/// by the thread it rides. Records about a member, a revision or a
/// withdrawal, carry none: they fold into the member, not the thread.
pub fn thread_index(world: &World) -> BTreeMap<usize, TaskId> {
    let mut index = BTreeMap::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        index.insert(ctx.birth.0, TaskId(i));
        for cid in &ctx.thread {
            index.insert(cid.0.0, TaskId(i));
        }
        for (rid, _) in &ctx.artifacts {
            index.insert(rid.0, TaskId(i));
        }
    }
    index
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
pub const SEARCH_KINDS: [&str; 9] = [
    "task",
    "note",
    "request",
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
    /// c-N: prose citations of the comment, and replies addressing it
    Comment(RecordId),
    /// t-N: prose namings of the task, comments
    /// addressing it
    Task(TaskId),
}

impl Term {
    /// The string scanned for in prose.
    fn canonical(&self) -> String {
        match self {
            Term::Plain(s) => s.clone(),
            Term::Comment(id) => format!("c-{}", id.0),
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
                    .push(Term::Comment(RecordId(n.parse().expect("digits checked"))));
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
    Comment(RecordId),
    Task(TaskId),
}

/// Every term must be satisfied, each by the text or by a reference.
fn terms_match(query: &SearchQuery, text: &str, refs: &[Ref]) -> bool {
    query.terms.iter().all(|term| match term {
        Term::Plain(s) => token_in_field(s, text),
        Term::Comment(id) => {
            token_in_field(&Term::Comment(*id).canonical(), text)
                || refs.contains(&Ref::Comment(*id))
        }
        Term::Task(id) => {
            token_in_field(&Term::Task(*id).canonical(), text) || refs.contains(&Ref::Task(*id))
        }
    })
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

    let mut matches: Vec<(usize, SearchRecord)> = Vec::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        let title = ctx.task.name.as_str();
        if terms_match(query, title, &[]) {
            matches.push((
                i,
                SearchRecord {
                    pointer: format!("c-{}", ctx.birth.0),
                    kind: "task",
                    actor: Some(ctx.birth_actor.as_str().to_string()),
                    body: title.to_string(),
                    order: ctx.birth.0,
                },
            ));
        }
        if let TaskState::Cancelled(why) = &ctx.task.state {
            let text = why.as_str();
            if terms_match(query, text, &[]) {
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
            if terms_match(query, name, &[]) {
                matches.push((
                    i,
                    SearchRecord {
                        pointer: format!("c-{}", rid.0),
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
        let refs = [match cctx.comment.target {
            Target::Task(t) => Ref::Task(t),
            Target::Comment(c) => Ref::Comment(c.0),
        }];
        if terms_match(query, &body, &refs) {
            matches.push((
                cctx.comment.root.0,
                SearchRecord {
                    pointer: format!("c-{}", id.0.0),
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
    use crate::objects::comment::{CommentId, Target};
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
        assert_eq!(
            TaskView::of(TaskId(0), &ctx_of(TaskState::Open)).state,
            "open"
        );
        assert_eq!(
            TaskView::of(TaskId(0), &ctx_of(TaskState::Offered)).state,
            "offered"
        );
        assert_eq!(
            TaskView::of(TaskId(0), &ctx_of(TaskState::Concluded)).state,
            "concluded"
        );
        assert_eq!(
            TaskView::of(
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

    /// The board is flat: rows in identity order, and each task owns
    /// exactly its own thread.
    #[test]
    fn the_board_is_flat_in_identity_order() {
        let human = human();
        let agent = agent();
        let record = |id: usize, ctx: &Context, target: Target, body: &str| Record {
            id: RecordId(id),
            timestamp: id as u64,
            context: ctx.clone(),
            event: Event::Commented {
                target,
                kind: CommentKind::Note {
                    body: Prose::new(body.into()).unwrap(),
                },
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
            record(
                2,
                &agent,
                Target::Task(TaskId(0)),
                "triage: how is sections, undecided",
            ),
            record(
                3,
                &human,
                Target::Comment(CommentId(RecordId(2))),
                "no - pure tree, canvas verdict pending",
            ),
            record(
                4,
                &agent,
                Target::Comment(CommentId(RecordId(3))),
                "noted, parked with owner",
            ),
            record(
                5,
                &human,
                Target::Task(TaskId(1)),
                "belongs to the other thread",
            ),
        ])
        .unwrap();

        assert_eq!(world.comments.len(), 4);
        assert_eq!(
            world.comments[&CommentId(RecordId(2))].comment.target,
            Target::Task(TaskId(0))
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(3))].comment.target,
            Target::Comment(CommentId(RecordId(2)))
        );

        let thread = thread_view(&world, TaskId(0)).unwrap();
        match &thread.items[0] {
            ThreadItem::Group { root, replies } => {
                assert_eq!(root.seq, 2);
                assert_eq!(root.depth, 1);
                assert_eq!(root.target, Target::Task(TaskId(0)));
                assert_eq!(replies[0].seq, 3);
                assert_eq!(replies[0].depth, 2);
                assert_eq!(replies[0].target, Target::Comment(CommentId(RecordId(2))));
                assert_eq!(replies[1].seq, 4);
                assert_eq!(replies[1].depth, 3);
                assert_eq!(replies[1].target, Target::Comment(CommentId(RecordId(3))));
            }
            other => panic!("expected a group, got {other:?}"),
        }
        assert_eq!(thread_view(&world, TaskId(1)).unwrap().items.len(), 1);

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
                    target: Target::Task(TaskId(0)),
                    kind: CommentKind::Note {
                        body: Prose::new("first deposit".into()).unwrap(),
                    },
                },
            },
            Record {
                id: RecordId(2),
                timestamp: 2,
                context: human.clone(),
                event: Event::Commented {
                    target: Target::Comment(CommentId(RecordId(1))),
                    kind: CommentKind::Note {
                        body: Prose::new("a reply".into()).unwrap(),
                    },
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
        assert!(
            !index.contains_key(&4),
            "a revision carries no home of its own"
        );
    }

    mod panels {
        use super::super::*;
        use crate::events::Event;
        use crate::objects::comment::{CommentId, Target};
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
            target: Target,
            kind: CommentKind,
        ) -> Record {
            Record {
                id: RecordId(seq),
                timestamp: at,
                context: ctx(tier),
                event: Event::Commented { target, kind },
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
                    Target::Task(TaskId(0)),
                    CommentKind::Note {
                        body: offer_body("parked finding"),
                    },
                ),
                comment_at(
                    2,
                    20,
                    Tier::Agent,
                    Target::Comment(CommentId(RecordId(1))),
                    CommentKind::Note {
                        body: offer_body("the reply"),
                    },
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
                ThreadItem::Note(line) => ("note", line.seq),
                ThreadItem::Group { root, .. } => ("group", root.seq),
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
                vec![("group", 1), ("artifacts", 3)],
                "a group then a trailing artifact cluster"
            );
        }

        #[test]
        fn the_thread_clusters_by_reply_links() {
            let world = clustered();
            let thread = thread_view(&world, TaskId(0)).unwrap();
            match &thread.items[0] {
                ThreadItem::Group { root, replies } => {
                    assert_eq!(root.seq, 1);
                    assert_eq!(replies.len(), 1);
                    assert_eq!(replies[0].seq, 2);
                }
                other => panic!("expected a group, got {other:?}"),
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
                    Target::Task(TaskId(0)),
                    CommentKind::Offer {
                        body: offer_body("suite green"),
                    },
                ),
                comment_at(
                    5,
                    5,
                    Tier::Human,
                    Target::Comment(CommentId(RecordId(4))),
                    CommentKind::Accept {
                        offer: CommentId(RecordId(4)),
                    },
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
                    Target::Task(TaskId(3)),
                    CommentKind::Offer {
                        body: offer_body("awaiting its consumer"),
                    },
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
                    Target::Task(TaskId(0)),
                    CommentKind::Offer {
                        body: offer_body("suite green, 55 tests"),
                    },
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
                    Target::Task(TaskId(0)),
                    CommentKind::Note {
                        body: offer_body("parked mid-flight"),
                    },
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
                    task: TaskId(0),
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
                        target: Target::Task(TaskId(0)),
                        kind: CommentKind::Note {
                            body: offer_body("scoped helper's note"),
                        },
                    },
                },
            ])
            .unwrap();

            // a scoped helper contributes without owning: no claim, no owner
            assert!(world.tasks[0].ownership.is_empty());
            let line = comment_line(&world, CommentId(RecordId(1))).unwrap();
            assert_eq!(line.actor, "pi");
        }
    }

    mod search {
        use super::super::*;
        use crate::ContentHash;
        use crate::events::Event;
        use crate::objects::comment::{CommentId, Target};
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
            target: Target,
            kind: CommentKind,
        ) -> Record {
            record(seq, at, ctx, Event::Commented { target, kind })
        }

        fn note(seq: usize, at: u64, ctx: &Context, target: Target, body: &str) -> Record {
            comment(
                seq,
                at,
                ctx,
                target,
                CommentKind::Note {
                    body: Prose::new(body.into()).unwrap(),
                },
            )
        }

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
                    Target::Task(TaskId(0)),
                    "parked: the fluke blocks foo\ntelemetries abound here",
                ),
                note(
                    3,
                    30,
                    &pi,
                    Target::Comment(CommentId(RecordId(2))),
                    "the fluke is real; floop proceeds",
                ),
                task(4, 40, &jerry, "t-490 the id boundary guard"),
                note(
                    5,
                    50,
                    &pi,
                    Target::Task(TaskId(1)),
                    "t-0 waits on this thread",
                ),
                note(
                    6,
                    60,
                    &pi,
                    Target::Task(TaskId(1)),
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
                    Target::Task(TaskId(1)),
                    CommentKind::Offer {
                        body: Prose::new(
                            "suite 9 green; the floop migration landed per c-2".into(),
                        )
                        .unwrap(),
                    },
                ),
                note(
                    9,
                    90,
                    &bot,
                    Target::Comment(CommentId(RecordId(2))),
                    "done, per the plan above",
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
                note(
                    2,
                    2,
                    &jerry,
                    Target::Task(TaskId(0)),
                    "the verdict cites c-1 above",
                ),
            ])
            .unwrap();
            // the name matches as a whole token; the title's "sweep" does not
            let groups = search(&world, &q(&["sweep-overview"])).unwrap();
            assert_eq!(pointers(&groups), vec![(0, vec!["c-1".into()])]);
            let groups = &groups[0];
            assert_eq!(groups.records[0].kind, "artifact");
            assert_eq!(groups.records[0].actor, None);
            assert_eq!(groups.records[0].body, "sweep-overview");
            // the kind facet narrows to artifacts alone
            let groups = search(&world, &q(&["sweep-overview", "kind:artifact"])).unwrap();
            assert_eq!(pointers(&groups), vec![(0, vec!["c-1".into()])]);
            // and the mention is a reference search finding its namer
            let groups = search(&world, &q(&["c-1"])).unwrap();
            assert_eq!(pointers(&groups), vec![(0, vec!["c-2".into()])]);
        }

        #[test]
        fn an_id_term_is_a_reference_search() {
            let world = story();
            // 'c-2': the offer cites it in prose, c-3 and c-9 address it by reply
            assert_eq!(
                pointers(&search(&world, &q(&["c-2"])).unwrap()),
                vec![
                    (0, vec!["c-3".into(), "c-9".into()]),
                    (1, vec!["c-8".into()])
                ]
            );
            // 't-0': c-5 names it in prose, c-2 by address
            assert_eq!(
                pointers(&search(&world, &q(&["t-0"])).unwrap()),
                vec![(0, vec!["c-2".into()]), (1, vec!["c-5".into()])]
            );
        }

        #[test]
        fn a_revised_body_is_what_search_reads() {
            let jerry = ctx(Tier::Human, "jerry");
            let world = World::replay(vec![
                task(0, 0, &jerry, "hold the verdict"),
                note(
                    2,
                    2,
                    &jerry,
                    Target::Task(TaskId(0)),
                    "the floop verdict lands",
                ),
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
                vec![(0, vec!["c-2".into()])]
            );
            assert!(search(&world, &q(&["floop"])).unwrap().is_empty());
        }

        #[test]
        fn a_withdrawn_body_is_omitted_until_the_flag_lets_it_back() {
            let jerry = ctx(Tier::Human, "jerry");
            let pi = ctx(Tier::Agent, "pi");
            let world = World::replay(vec![
                task(0, 0, &jerry, "hold the verdict"),
                note(
                    2,
                    2,
                    &pi,
                    Target::Task(TaskId(0)),
                    "the floop verdict lands",
                ),
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
                note(
                    4,
                    4,
                    &jerry,
                    Target::Comment(CommentId(RecordId(2))),
                    "the verdict lives on per c-2",
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
                vec![(0, vec!["c-2".into()])]
            );
            // the citation still resolves: c-2's reference search finds the
            // record that cites it, withdrawn or not
            assert_eq!(
                pointers(&search(&world, &q(&["c-2"])).unwrap()),
                vec![(0, vec!["c-4".into()])]
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
            // by: only pi's reply matches; the title and offer are
            // other authors', visible as counts
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
                vec![(1, vec!["c-8".into()]), (0, vec![])],
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

        fn comment(seq: usize, target: Target, kind: CommentKind) -> (usize, Event) {
            record(seq, Event::Commented { target, kind })
        }

        fn note(seq: usize, target: Target, body: &str) -> (usize, Event) {
            comment(
                seq,
                target,
                CommentKind::Note {
                    body: Prose::new(body.into()).unwrap(),
                },
            )
        }

        // t-0 holds the waited offer, its sibling note, a two-link chain, revisions, a withdrawal, the accept, a claim, an artifact; t-1 a rejected offer and its concluded successor
        fn story() -> (World, Vec<(usize, Event)>) {
            let events = vec![
                task(0, "implement foo"),
                comment(
                    1,
                    Target::Task(TaskId(0)),
                    CommentKind::Offer {
                        body: Prose::new("suite green, ready for review".into()).unwrap(),
                    },
                ),
                note(2, Target::Task(TaskId(0)), "sibling chatter"),
                note(
                    3,
                    Target::Comment(CommentId(RecordId(1))),
                    "one question first",
                ),
                task(4, "migrate floop"),
                note(
                    5,
                    Target::Comment(CommentId(RecordId(3))),
                    "answered in the body; see c-1 for the offer",
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
                    Target::Comment(CommentId(RecordId(1))),
                    CommentKind::Accept {
                        offer: CommentId(RecordId(1)),
                    },
                ),
                note(8, Target::Task(TaskId(1)), "the other thread"),
                comment(
                    9,
                    Target::Task(TaskId(1)),
                    CommentKind::Use {
                        offer: CommentId(RecordId(1)),
                        reason: Prose::new("floop rests on foo".into()).unwrap(),
                    },
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
                    Target::Task(TaskId(1)),
                    CommentKind::Offer {
                        body: Prose::new("floop migrated, review welcome".into()).unwrap(),
                    },
                ),
                comment(
                    15,
                    Target::Comment(CommentId(RecordId(14))),
                    CommentKind::Reject {
                        offer: CommentId(RecordId(14)),
                        grounds: Prose::new("the floop index is missing".into()).unwrap(),
                    },
                ),
                comment(
                    16,
                    Target::Task(TaskId(1)),
                    CommentKind::Offer {
                        body: Prose::new("the index rides along now".into()).unwrap(),
                    },
                ),
                comment(
                    17,
                    Target::Comment(CommentId(RecordId(16))),
                    CommentKind::Accept {
                        offer: CommentId(RecordId(16)),
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
                vec![4, 8, 9, 14, 15, 16, 17]
            );
        }

        // the comment wait wakes only on its chain: replies, revisions of members, the accept — not the birth, sibling, use, or in-prose citation
        #[test]
        fn the_comment_wait_matches_only_its_chain() {
            let (world, events) = story();
            let offer = Target::Comment(CommentId(RecordId(1)));
            assert_eq!(
                matching(&world, &events, &offer, WaitUntil::Activity),
                vec![3, 5, 6, 7, 12, 13]
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

    // -- the request write's reply facts -----------------------------

    mod request_reply {
        use super::*;
        use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

        fn ctx(name: &str) -> Context {
            Context {
                actor: ActorName::new(name.into()).unwrap(),
                tier: Tier::Human,
                scope: None,
            }
        }

        fn record(seq: usize, ctx: &Context, event: Event) -> Record {
            Record {
                id: RecordId(seq),
                timestamp: seq as u64,
                context: ctx.clone(),
                event,
            }
        }

        fn comment(seq: usize, ctx: &Context, target: Target, kind: CommentKind) -> Record {
            record(seq, ctx, Event::Commented { target, kind })
        }

        // the request resolves a comment target through its root; a task with no workspace returns none
        #[test]
        fn the_request_facts_resolve_the_root_and_the_recorded_workspace() {
            let jerry = ctx("jerry");
            let world = World::replay(vec![
                record(
                    0,
                    &jerry,
                    Event::TaskCreated {
                        name: Prose::new("implement foo".into()).unwrap(),
                    },
                ),
                record(
                    1,
                    &jerry,
                    Event::TaskCreated {
                        name: Prose::new("migrate floop".into()).unwrap(),
                    },
                ),
                record(
                    2,
                    &jerry,
                    Event::WorkspaceEstablished {
                        task_id: TaskId(0),
                        base: GitCommit::new("abc123".into()).unwrap(),
                        branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                        worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
                    },
                ),
                comment(
                    3,
                    &jerry,
                    Target::Task(TaskId(1)),
                    CommentKind::Note {
                        body: Prose::new("context for the migration".into()).unwrap(),
                    },
                ),
                comment(
                    4,
                    &jerry,
                    Target::Task(TaskId(0)),
                    CommentKind::Request {
                        body: Prose::new("where do you stand?".into()).unwrap(),
                    },
                ),
                comment(
                    5,
                    &jerry,
                    Target::Comment(CommentId(RecordId(3))),
                    CommentKind::Request {
                        body: Prose::new("and on this note?".into()).unwrap(),
                    },
                ),
            ])
            .unwrap();

            let facts = request_facts(&world, RecordId(4)).expect("the request folds in");
            assert_eq!(facts.seq, 4);
            assert_eq!(facts.task, 0);
            let ws = facts.workspace.expect("the recorded workspace returns");
            assert_eq!(
                (ws.task, ws.branch, ws.base, ws.path),
                (
                    0,
                    "saccade/t-0".into(),
                    "abc123".into(),
                    "/repo/wt/t-0".into()
                )
            );

            // a comment target resolves through its root task, and a
            // task with no recorded workspace returns none
            let facts = request_facts(&world, RecordId(5)).expect("the request folds in");
            assert_eq!(facts.seq, 5);
            assert_eq!(facts.task, 1);
            assert!(facts.workspace.is_none());
        }
    }
}
