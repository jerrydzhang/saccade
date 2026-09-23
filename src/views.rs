//! Statements: the render shapes and their constructors, one per reader.
//! Views read objects and contexts; nothing in objects knows views exist.

use std::collections::BTreeMap;

use crate::objects::comment::{
    AgentAttemptState, CommentContext, CommentId, CommentState, Refusal, ResponseState, Target,
};
use crate::objects::proposal::{ProposalAction, ProposalContext, ProposalId, ProposalState};
use crate::objects::task::{TaskContext, TaskId, TaskState};
use crate::store::{RecordId, Tier, World};
use crate::types::artifact::Artifact;
use crate::types::prose::Prose;

/// A claimed task with no record movement for this long renders adrift.
pub const ADRIFT_AFTER_SECS: u64 = 24 * 3600;
/// The movement ribbon carries this much history, nothing older.
pub const RIBBON_WINDOW_SECS: u64 = 72 * 3600;

pub struct TaskView {
    pub id: String,
    pub state: &'static str,
    pub parent: Option<String>,
    pub name: String,
    /// The pending-judgment mark, when an open proposal targets this task.
    /// The row is an attention cue and a pointer; the full story lives at the seq.
    pub proposal: Option<ProposalMark>,
    /// Size of the attached comment thread; renders as a mere pointer (`#`).
    pub n_comments: usize,
}

pub struct ProposalMark {
    pub seq: usize,
    pub verb: &'static str,
}

/// The task state's identifier vocabulary, shared by every view that names it.
fn state_str(state: &TaskState) -> &'static str {
    match state {
        TaskState::Open => "open",
        TaskState::Claimed => "claimed",
        TaskState::Delivered(_) => "delivered",
        TaskState::Done(_) => "done",
        TaskState::Dropped => "dropped",
    }
}

impl TaskView {
    pub fn of(id: TaskId, ctx: &TaskContext, proposal: Option<&ProposalContext>) -> TaskView {
        TaskView {
            id: format!("t-{}", id.0),
            state: state_str(&ctx.task.state),
            parent: ctx.task.parent_id.map(|p| format!("t-{}", p.0)),
            name: ctx.task.name.as_str().to_string(),
            proposal: proposal.map(|p| {
                let seq = ctx
                    .proposal
                    .expect("a mark is only built when the pointer is set")
                    .0
                    .0;
                ProposalMark {
                    seq,
                    verb: match p.proposal.action {
                        ProposalAction::Drop { .. } => "drop",
                        ProposalAction::Release { .. } => "release",
                    },
                }
            }),
            n_comments: ctx.thread.len(),
        }
    }
}

pub struct ProposalView {
    pub id: usize,
    pub state: &'static str,
    pub action: &'static str,
    pub task: String,
    pub name: String,
}

impl ProposalView {
    /// `target_state` is the state of the task this proposal acts on; a
    /// missing target reads as stale — the act would be refused today.
    pub fn of(
        id: ProposalId,
        ctx: &ProposalContext,
        target_state: Option<&TaskState>,
    ) -> ProposalView {
        let stale = match ctx.proposal.state {
            ProposalState::Open => is_stale(ctx, target_state),
            _ => false,
        };
        ProposalView {
            id: id.0.0,
            state: match (&ctx.proposal.state, stale) {
                (ProposalState::Open, true) => "stale",
                (ProposalState::Open, false) => "open",
                (ProposalState::Accepted, _) => "accepted",
                (ProposalState::Rejected(_), _) => "rejected",
                (ProposalState::Withdrawn(_), _) => "withdrawn",
            },
            action: match ctx.proposal.action {
                ProposalAction::Drop { .. } => "drop",
                ProposalAction::Release { .. } => "release",
            },
            task: match ctx.proposal.action {
                ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => {
                    format!("t-{}", task_id.0)
                }
            },
            name: ctx.proposal.name.as_str().to_string(),
        }
    }
}

/// Derived staleness: would the embedded act be refused today?
/// The same probe decide uses at propose time — the quiet consumer to
/// accept's loud one.
fn is_stale(ctx: &ProposalContext, target_state: Option<&TaskState>) -> bool {
    let probe = Prose::new("probe".into()).expect("probe is non-empty");
    match target_state {
        None => true,
        Some(state) => state
            .transition(&ctx.proposal.action.target_event(&probe))
            .is_none(),
    }
}

/// A row of the task's thread view. Depth is position in the rendered
/// thread — the task is the root, so a comment addressing it sits at 1.
#[derive(Debug)]
pub struct CommentLine {
    pub seq: usize,
    pub depth: usize,
    pub actor: String,
    pub tier: String,
    /// The variant the comment carries: note, demand, steer, or ask
    pub kind: &'static str,
    pub body: String,
    pub state: Option<String>,
    pub born_at: u64,
    /// The machinery's refusal to run this demand, when it refused
    pub refusal: Option<RefusalView>,
}

/// The refusal as the thread renders it: the reason and the moment.
#[derive(Debug, PartialEq)]
pub struct RefusalView {
    pub reason: String,
    pub at: u64,
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

/// One position in a flattened thread: a comment or an artifact.
#[derive(Debug)]
pub enum ThreadEntry {
    Comment(CommentLine),
    Artifact(ArtifactLine),
}

/// One conversation: a comment plus every reply hanging off it, in
/// record order. Bounded by the reply links, never by the renderer.
pub struct Conversation {
    pub root: CommentLine,
    pub replies: Vec<CommentLine>,
}

/// The focused task's thread in record order by first comment: an
/// agent-addressed exchange with its run, any other conversation, or
/// one unlinked annotation.
#[derive(Debug)]
pub enum ThreadItem {
    Exchange {
        root: CommentLine,
        run: Option<RunView>,
        replies: Vec<ThreadEntry>,
    },
    Group {
        root: CommentLine,
        replies: Vec<ThreadEntry>,
    },
    Note(CommentLine),
    Artifact(ArtifactLine),
}

/// The focused task's thread as a view: the context's pointer index,
/// followed, grouped by reply links, ordered by first comment.
pub struct ThreadView {
    pub items: Vec<ThreadItem>,
}

/// The variant a comment state carries — the kind never moves.
pub fn kind_of(state: &CommentState) -> &'static str {
    match state {
        CommentState::Note => "note",
        CommentState::Demand { .. } => "demand",
        CommentState::Steer { .. } => "steer",
        CommentState::Ask { .. } => "ask",
    }
}

fn line_of(comments: &BTreeMap<CommentId, CommentContext>, cid: CommentId) -> CommentLine {
    let cctx = comments
        .get(&cid)
        .expect("apply guarantees thread members are resident");
    let mut depth = 1;
    let mut up = cctx.comment.target;
    while let Target::Comment(parent) = up {
        depth += 1;
        up = comments
            .get(&parent)
            .expect("apply received a dangling comment target")
            .comment
            .target;
    }
    CommentLine {
        seq: cid.0.0,
        depth,
        actor: cctx.actor.as_str().to_string(),
        tier: format!("{:?}", cctx.tier).to_lowercase(),
        kind: kind_of(&cctx.state),
        body: cctx.comment.body.as_str().to_string(),
        state: state_tag(&cctx.state, cctx.refusal.as_ref()),
        born_at: cctx.born_at,
        refusal: cctx.refusal.as_ref().map(|r| RefusalView {
            reason: r.reason.as_str().to_string(),
            at: r.at,
        }),
    }
}

/// The task's thread as a view: the context's pointer index, followed.
/// Membership is fixed at birth, so this walks pointers, never scans.
pub fn comment_thread(
    comments: &BTreeMap<CommentId, CommentContext>,
    ctx: &TaskContext,
) -> Vec<CommentLine> {
    ctx.thread
        .iter()
        .map(|cid| line_of(comments, *cid))
        .collect()
}

/// The thread flat in record order: comments and artifacts merged by
/// position — the show shape.
pub fn thread_entries(
    comments: &BTreeMap<CommentId, CommentContext>,
    ctx: &TaskContext,
) -> Vec<ThreadEntry> {
    let mut entries: Vec<(usize, ThreadEntry)> = ctx
        .thread
        .iter()
        .map(|cid| (cid.0.0, ThreadEntry::Comment(line_of(comments, *cid))))
        .collect();
    entries.extend(ctx.artifacts.iter().map(|(rid, artifact)| {
        (
            rid.0,
            ThreadEntry::Artifact(artifact_line_of(rid.0, artifact)),
        )
    }));
    entries.sort_by_key(|(seq, _)| *seq);
    entries.into_iter().map(|(_, entry)| entry).collect()
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
    world
        .comments
        .contains_key(&id)
        .then(|| line_of(&world.comments, id))
}

pub struct ShowView {
    pub id: String,
    pub state: &'static str,
    pub parent: Option<String>,
    pub name: String,
    pub receipt: Option<String>,
}

impl ShowView {
    pub fn of(id: TaskId, ctx: &TaskContext) -> ShowView {
        ShowView {
            id: format!("t-{}", id.0),
            state: state_str(&ctx.task.state),
            parent: ctx.task.parent_id.map(|p| format!("t-{}", p.0)),
            name: ctx.task.name.as_str().to_string(),
            receipt: match &ctx.task.state {
                TaskState::Delivered(r) | TaskState::Done(r) => Some(r.as_str().to_string()),
                _ => None,
            },
        }
    }
}

/// Read helpers: follow the world's pointers so emitters need not.
pub fn task_view(world: &World, id: TaskId) -> Option<TaskView> {
    let ctx = world.tasks.get(id.0)?;
    Some(TaskView::of(
        id,
        ctx,
        ctx.proposal.and_then(|p| world.proposals.get(&p)),
    ))
}

pub fn proposal_view(world: &World, id: ProposalId) -> Option<ProposalView> {
    let ctx = world.proposals.get(&id)?;
    let target = match ctx.proposal.action {
        ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => {
            world.tasks.get(task_id.0).map(|t| &t.task.state)
        }
    };
    Some(ProposalView::of(id, ctx, target))
}

pub fn show_view(world: &World, id: TaskId) -> Option<ShowView> {
    world.tasks.get(id.0).map(|ctx| ShowView::of(id, ctx))
}

/// The tag a thread row carries: its variant and where the response
/// stands. Notes carry nothing.
fn state_tag(state: &CommentState, refusal: Option<&Refusal>) -> Option<String> {
    match state {
        CommentState::Note => None,
        CommentState::Demand { response, attempt } => {
            if refusal.is_some() {
                return Some("demand, refused".into());
            }
            let response = match response {
                ResponseState::Awaiting => "awaiting",
                ResponseState::Responded { .. } => "responded",
            };
            let attempt = match attempt {
                AgentAttemptState::Authorized { .. } => "",
                AgentAttemptState::InFlight { .. } => ", in flight",
                AgentAttemptState::Spent => "",
            };
            Some(format!("demand, {response}{attempt}"))
        }
        CommentState::Steer { delivery } => Some(match delivery {
            crate::objects::comment::SteerDelivery::Standing => "steer, standing".into(),
            crate::objects::comment::SteerDelivery::Forwarded => "steer, forwarded".into(),
        }),
        CommentState::Ask { response } => Some(match response {
            ResponseState::Awaiting => "ask, awaiting".into(),
            ResponseState::Responded { .. } => "ask, answered".into(),
        }),
    }
}

/// Every open proposal, oldest first — the gate queue as rows.
pub fn open_proposals(world: &World) -> Vec<ProposalView> {
    world
        .proposals
        .keys()
        .filter_map(|id| proposal_view(world, *id))
        .filter(|v| v.state == "open")
        .collect()
}

/// Follow the world's pointers to one task's thread: the conversation
/// each root opened, and the annotations no one replied to.
pub fn thread_view(world: &World, id: TaskId) -> Option<ThreadView> {
    let ctx = world.tasks.get(id.0)?;
    // building groups in record order; replies always come after their
    // root, so one pass suffices. An agent-addressed root is an
    // exchange and carries the run that answered it, if one did.
    enum Building {
        Exchange {
            root: CommentLine,
            replies: Vec<ThreadEntry>,
        },
        Group {
            root: CommentLine,
            replies: Vec<ThreadEntry>,
        },
    }
    enum Member {
        Comment(CommentId),
        Artifact(RecordId, Artifact),
    }
    // the thread's utterances merged by position: an artifact rides the
    // stream beside the comments and lands in the group open at its
    // seq — position is the association
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
    let mut lead: Vec<ArtifactLine> = Vec::new();
    let mut last_group: Option<usize> = None;
    for (_, member) in stream {
        match member {
            Member::Comment(cid) => {
                let root = root_of(&world.comments, cid);
                if root == cid {
                    let line = line_of(&world.comments, cid);
                    root_index.insert(cid, groups.len());
                    if line.kind == "demand" {
                        groups.push(Building::Exchange {
                            root: line,
                            replies: Vec::new(),
                        });
                    } else {
                        groups.push(Building::Group {
                            root: line,
                            replies: Vec::new(),
                        });
                    }
                    last_group = Some(groups.len() - 1);
                } else {
                    let g = root_index
                        .get(&root)
                        .copied()
                        .expect("a reply's root is resident and earlier");
                    let line = line_of(&world.comments, cid);
                    match &mut groups[g] {
                        Building::Exchange { replies, .. } | Building::Group { replies, .. } => {
                            replies.push(ThreadEntry::Comment(line))
                        }
                    }
                    last_group = Some(g);
                }
            }
            Member::Artifact(rid, artifact) => {
                let line = artifact_line_of(rid.0, &artifact);
                match last_group {
                    Some(g) => match &mut groups[g] {
                        Building::Exchange { replies, .. } | Building::Group { replies, .. } => {
                            replies.push(ThreadEntry::Artifact(line))
                        }
                    },
                    // no comment precedes it: a standalone item at the top
                    None => lead.push(line),
                }
            }
        }
    }
    let mut items: Vec<ThreadItem> = lead.into_iter().map(ThreadItem::Artifact).collect();
    items.extend(groups.into_iter().map(|g| match g {
        Building::Exchange { root, replies } => {
            // the run that answered this demand, if one did
            let run = world
                .incarnations
                .iter()
                .find(|(_, r)| r.response_target.0.0 == root.seq)
                .map(|(id, r)| RunView {
                    incarnation: id.0.0,
                    task: r.task_id.0,
                    demand: r.response_target.0.0,
                    actor: r.actor.as_str().to_string(),
                    born_at: r.born_at,
                    done_at: r.done_at,
                });
            ThreadItem::Exchange { root, run, replies }
        }
        Building::Group { root, replies } => {
            if replies.is_empty() {
                ThreadItem::Note(root)
            } else {
                ThreadItem::Group { root, replies }
            }
        }
    }));
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

/// One row of the forest: a live task and its depth in the parent tree.
pub struct ForestRow {
    pub task: TaskView,
    pub depth: usize,
}

/// The live tree as navigation: open, claimed, and delivered tasks,
/// parents before children, each row carrying the open-proposal mark
/// it already has. A delivered task still takes findings and its accept.
pub fn forest(world: &World) -> Vec<ForestRow> {
    world
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, ctx)| {
            matches!(
                ctx.task.state,
                TaskState::Open | TaskState::Claimed | TaskState::Delivered(_)
            )
        })
        .map(|(i, ctx)| {
            let mut depth = 0;
            let mut up = ctx.task.parent_id;
            while let Some(parent) = up {
                depth += 1;
                up = world.tasks[parent.0].task.parent_id;
            }
            ForestRow {
                task: TaskView::of(
                    TaskId(i),
                    ctx,
                    ctx.proposal.and_then(|p| world.proposals.get(&p)),
                ),
                depth,
            }
        })
        .collect()
}

/// The rail's archive: done and dropped tasks, newest first — the sort
/// key is last touch, which never sits earlier than the settle itself.
/// Depth is flattened: the archive is a list, not a forest.
pub fn closed_tasks(world: &World) -> Vec<ForestRow> {
    let mut rows: Vec<(u64, usize, ForestRow)> = world
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, ctx)| matches!(ctx.task.state, TaskState::Done(_) | TaskState::Dropped))
        .map(|(i, ctx)| {
            (
                ctx.last_record_at,
                i,
                ForestRow {
                    task: TaskView::of(
                        TaskId(i),
                        ctx,
                        ctx.proposal.and_then(|p| world.proposals.get(&p)),
                    ),
                    depth: 0,
                },
            )
        })
        .collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    rows.into_iter().map(|(_, _, row)| row).collect()
}

/// A question awaiting its answer: the residual that reaches the
/// human, who is the dependency root of every ask.
pub struct AskedOfYou {
    pub comment: usize,
    pub task: usize,
    pub actor: String,
    pub body: String,
}

/// Every ask still awaiting its answer, oldest first — the residual
/// inbox; the waiter's own view of it is the wait release.
pub fn asked_of_you(world: &World) -> Vec<AskedOfYou> {
    world
        .comments
        .iter()
        .filter(|(_, c)| {
            matches!(
                c.state,
                CommentState::Ask {
                    response: ResponseState::Awaiting,
                }
            )
        })
        .map(|(id, c)| AskedOfYou {
            comment: id.0.0,
            task: c.comment.root.0,
            actor: c.actor.as_str().to_string(),
            body: c.comment.body.as_str().to_string(),
        })
        .collect()
}

// -- the reading-side reference resolver's index --------------------

/// What a '#N' mention resolves to on the reading side: a comment's
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

/// A run, as the strip and an exchange card name it.
#[derive(Debug)]
pub struct RunView {
    pub incarnation: usize,
    pub task: usize,
    /// The demand comment the run answers
    pub demand: usize,
    pub actor: String,
    pub born_at: u64,
    pub done_at: Option<u64>,
}

impl RunView {
    /// A run is in flight until its terminal record lands.
    pub fn in_flight(&self) -> bool {
        self.done_at.is_none()
    }
}

/// A claimed task and its two ages. Adrift is the t-53 tint: no record
/// movement for ADRIFT_AFTER_SECS.
pub struct Candidate {
    pub task: String,
    pub name: String,
    pub claim_age: u64,
    pub last_record_age: u64,
    pub adrift: bool,
}

/// The next panel: supervision facts only — runs in flight, demands
/// awaiting a human, claimed tasks with their ages.
pub fn next_panel(world: &World, now: u64) -> NextPanel {
    let mut runs = Vec::new();
    let mut candidates = Vec::new();
    let mut awaiting = Vec::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        if let Some(id) = ctx.active_incarnation {
            let run = &world.incarnations[&id];
            runs.push(RunView {
                incarnation: id.0.0,
                task: run.task_id.0,
                demand: run.response_target.0.0,
                actor: run.actor.as_str().to_string(),
                born_at: run.born_at,
                done_at: run.done_at,
            });
        }
        if let (TaskState::Claimed, Some(claimed_at)) = (&ctx.task.state, ctx.claimed_at) {
            let last_record_age = now.saturating_sub(ctx.last_record_at);
            candidates.push(Candidate {
                task: format!("t-{i}"),
                name: ctx.task.name.as_str().to_string(),
                claim_age: now.saturating_sub(claimed_at),
                last_record_age,
                adrift: last_record_age >= ADRIFT_AFTER_SECS,
            });
        }
        if let (TaskState::Delivered(_), Some(delivered_at)) = (&ctx.task.state, ctx.delivered_at) {
            let delivered_age = now.saturating_sub(delivered_at);
            awaiting.push(AwaitingAcceptance {
                task: format!("t-{i}"),
                name: ctx.task.name.as_str().to_string(),
                delivered_age,
                stale: delivered_age >= ADRIFT_AFTER_SECS,
            });
        }
    }
    NextPanel {
        runs,
        asked_of_you: asked_of_you(world),
        candidates,
        awaiting,
    }
}

pub struct NextPanel {
    pub runs: Vec<RunView>,
    pub asked_of_you: Vec<AskedOfYou>,
    pub candidates: Vec<Candidate>,
    /// Delivered tasks in id order — the accept door's rows.
    pub awaiting: Vec<AwaitingAcceptance>,
}

/// A delivered task and its wait for the accept. Stale is the delivered
/// tint: no accept for ADRIFT_AFTER_SECS.
pub struct AwaitingAcceptance {
    pub task: String,
    pub name: String,
    pub delivered_age: u64,
    pub stale: bool,
}

/// The kind a ribbon mark carries, keyed as the strip names them: a
/// note by its author's tier, a demand, a steer, an ask, or a run's birth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarkKind {
    Note { human: bool },
    Demand,
    Steer,
    Ask,
    Run,
}

/// One position on the movement ribbon; seq is the record the mark
/// clicks through to (a run marks the demand comment it answered).
pub struct RibbonMark {
    pub kind: MarkKind,
    pub at: u64,
    pub seq: usize,
    pub task: usize,
}

/// Every comment, demand, and run birth inside the 72h window ending
/// now, across all tasks, oldest first.
pub fn ribbon_marks(world: &World, now: u64) -> Vec<RibbonMark> {
    let from = now.saturating_sub(RIBBON_WINDOW_SECS);
    let mut marks: Vec<RibbonMark> = Vec::new();
    for (id, c) in &world.comments {
        if c.born_at < from || c.born_at > now {
            continue;
        }
        let kind = match c.state {
            CommentState::Note => MarkKind::Note {
                human: c.tier == Tier::Human,
            },
            CommentState::Demand { .. } => MarkKind::Demand,
            CommentState::Steer { .. } => MarkKind::Steer,
            CommentState::Ask { .. } => MarkKind::Ask,
        };
        marks.push(RibbonMark {
            kind,
            at: c.born_at,
            seq: id.0.0,
            task: c.comment.root.0,
        });
    }
    for run in world.incarnations.values() {
        if run.born_at >= from && run.born_at <= now {
            marks.push(RibbonMark {
                kind: MarkKind::Run,
                at: run.born_at,
                seq: run.response_target.0.0,
                task: run.task_id.0,
            });
        }
    }
    marks.sort_by_key(|m| m.at);
    marks
}

// -- The search door -------------------------------------------------

/// One only-show-me filter; none of them rank, suggest, or remember.
#[derive(Clone, Debug, PartialEq)]
pub enum Facet {
    /// Only records living on this thread
    In(TaskId),
    /// Only records whose author's name contains this, case-folded
    By(String),
    /// Only records of this kind; the vocabulary is SEARCH_KINDS
    Kind(&'static str),
    /// Only records on this thread's task or any of its descendants
    Under(TaskId),
}

/// The kinds a searched record can be — kind:'s whole vocabulary.
pub const SEARCH_KINDS: [&str; 7] = [
    "task", "note", "demand", "steer", "ask", "receipt", "artifact",
];

/// A term: plain words match text; ids are reference searches.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// Matches as a whole, word-bounded token, case-folded
    Plain(String),
    /// #N: prose citations of the comment, and replies addressing it
    Comment(RecordId),
    /// t-N: prose namings of the task, births under it, comments
    /// addressing it
    Task(TaskId),
}

impl Term {
    /// The string scanned for in prose.
    fn canonical(&self) -> String {
        match self {
            Term::Plain(s) => s.clone(),
            Term::Comment(id) => format!("#{}", id.0),
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

impl SearchQuery {
    pub fn parse(tokens: &[String]) -> Result<SearchQuery, SearchFail> {
        let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
        let mut query = SearchQuery::default();
        for token in tokens {
            if let Some(rest) = token.strip_prefix("in:") {
                query.facets.push(Facet::In(task_ref(rest, "in:")?));
            } else if let Some(rest) = token.strip_prefix("under:") {
                query.facets.push(Facet::Under(task_ref(rest, "under:")?));
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
                "give at least one term or a facet (in:, by:, kind:, under:); the moves:\n  sac search telemetry      find where it was decided\n  sac search '#907'         follow a reference (quote the hash)\n  sac search floop in:t-0   narrow with facets\n  sac search '#907' -C 3    read the neighborhood".into(),
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
    /// "#907" for a comment or a birth; "t-90 receipt" — the fold keeps
    /// no delivery seq, so the task is the receipt's address
    pub pointer: String,
    pub kind: &'static str,
    /// The author, when the fold keeps one; receipts carry none
    pub actor: Option<String>,
    pub body: String,
    /// The record's position; a receipt, having none, closes its group
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
            Facet::In(id) | Facet::Under(id) => Some(*id),
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
        let refs = ctx
            .task
            .parent_id
            .map(Ref::Task)
            .into_iter()
            .collect::<Vec<_>>();
        if terms_match(query, title, &refs) {
            matches.push((
                i,
                SearchRecord {
                    pointer: format!("#{}", ctx.birth.0),
                    kind: "task",
                    actor: Some(ctx.birth_actor.as_str().to_string()),
                    body: title.to_string(),
                    order: ctx.birth.0,
                },
            ));
        }
        if let TaskState::Delivered(receipt) | TaskState::Done(receipt) = &ctx.task.state {
            let text = receipt.as_str();
            if terms_match(query, text, &[]) {
                matches.push((
                    i,
                    SearchRecord {
                        pointer: format!("t-{i} receipt"),
                        kind: "receipt",
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
                        pointer: format!("#{}", rid.0),
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
        let body = cctx.comment.body.as_str();
        let refs = [match cctx.comment.target {
            Target::Task(t) => Ref::Task(t),
            Target::Comment(c) => Ref::Comment(c.0),
        }];
        if terms_match(query, body, &refs) {
            matches.push((
                cctx.comment.root.0,
                SearchRecord {
                    pointer: format!("#{}", id.0.0),
                    kind: kind_of(&cctx.state),
                    actor: Some(cctx.actor.as_str().to_string()),
                    body: body.to_string(),
                    order: id.0.0,
                },
            ));
        }
    }

    // group by thread; records by position, the receipt closing its group
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
            .filter(|r| facets_pass(world, query, thread, r))
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

fn facets_pass(world: &World, query: &SearchQuery, thread: usize, record: &SearchRecord) -> bool {
    query.facets.iter().all(|facet| match facet {
        Facet::In(id) => thread == id.0,
        Facet::Under(id) => under(world, thread, *id),
        Facet::By(name) => record
            .actor
            .as_deref()
            .is_some_and(|a| a.to_lowercase().contains(name)),
        Facet::Kind(kind) => record.kind == *kind,
    })
}

fn under(world: &World, thread: usize, root: TaskId) -> bool {
    let mut up = Some(TaskId(thread));
    while let Some(id) = up {
        if id == root {
            return true;
        }
        up = world.tasks[id.0].task.parent_id;
    }
    false
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

    fn ctx_of(state: TaskState) -> TaskContext {
        TaskContext {
            task: Task {
                state,
                name: Prose::new("implement foo".into()).unwrap(),
                parent_id: None,
            },
            last_updated: RecordId(0),
            birth: RecordId(0),
            claimed_at: None,
            last_record_at: 0,
            delivered_at: None,
            proposal: None,
            thread: Vec::new(),
            artifacts: Vec::new(),
            holder: None,
            birth_actor: ActorName::new("human person".into()).unwrap(),
            active_incarnation: None,
            workspace: None,
        }
    }

    #[test]
    fn state_strings_are_the_identifier_vocabulary() {
        assert_eq!(
            TaskView::of(TaskId(0), &ctx_of(TaskState::Open), None).state,
            "open"
        );
        assert_eq!(
            TaskView::of(TaskId(0), &ctx_of(TaskState::Claimed), None).state,
            "claimed"
        );
        assert_eq!(
            TaskView::of(
                TaskId(0),
                &ctx_of(TaskState::Delivered(
                    Prose::new("suite green".into()).unwrap()
                )),
                None
            )
            .state,
            "delivered"
        );
        assert_eq!(
            TaskView::of(
                TaskId(0),
                &ctx_of(TaskState::Done(Prose::new("suite green".into()).unwrap())),
                None
            )
            .state,
            "done"
        );
        assert_eq!(
            TaskView::of(TaskId(0), &ctx_of(TaskState::Dropped), None).state,
            "dropped"
        );
    }

    fn human() -> Context {
        Context {
            actor: ActorName::new("human person".into()).unwrap(),
            tier: Tier::Human,
        }
    }

    fn agent() -> Context {
        Context {
            actor: ActorName::new("saccade bot".into()).unwrap(),
            tier: Tier::Agent,
        }
    }

    /// The thread is a walk: targets are stored, depth and membership are
    /// derived, and each task owns exactly its own thread.
    #[test]
    fn comment_thread_is_derived_from_addresses() {
        let human = human();
        let agent = agent();
        let record = |id: usize, ctx: &Context, target: Target, body: &str| Record {
            id: RecordId(id),
            timestamp: id as u64,
            context: ctx.clone(),
            event: Event::Commented {
                target,
                body: Prose::new(body.into()).unwrap(),
                kind: CommentKind::Note,
            },
        };
        let birth = |id: usize, ctx: &Context, name: &str| Record {
            id: RecordId(id),
            timestamp: id as u64,
            context: ctx.clone(),
            event: Event::TaskCreated {
                name: Prose::new(name.into()).unwrap(),
                parent_id: None,
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

        let thread = comment_thread(&world.comments, &world.tasks[0]);
        assert_eq!(
            thread
                .iter()
                .map(|l| (l.seq, l.depth, l.actor.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (2, 1, "saccade bot"),
                (3, 2, "human person"),
                (4, 3, "saccade bot")
            ]
        );
        assert_eq!(comment_thread(&world.comments, &world.tasks[1]).len(), 1);
    }
}

#[cfg(test)]
mod panels {
    use super::*;
    use crate::CommentKind;
    use crate::events::Event;
    use crate::objects::comment::Target;
    use crate::objects::incarnation::IncarnationId;
    use crate::store::{Context, Record, RecordId, Tier, World};
    use crate::types::actor::ActorName;
    use crate::types::pointers::SessionPointer;

    const HOUR: u64 = 3600;

    fn ctx(tier: Tier) -> Context {
        Context {
            actor: ActorName::new(match tier {
                Tier::Human => "jerry".into(),
                _ => "pi".into(),
            })
            .unwrap(),
            tier,
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
                parent_id: None,
            },
        )
    }

    fn comment_at(seq: usize, at: u64, tier: Tier, target: Target, kind: CommentKind) -> Record {
        record(
            seq,
            at,
            tier,
            Event::Commented {
                target,
                body: Prose::new("a body worth keeping".into()).unwrap(),
                kind,
            },
        )
    }

    fn entry_seq(entry: &ThreadEntry) -> usize {
        match entry {
            ThreadEntry::Comment(c) => c.seq,
            ThreadEntry::Artifact(a) => a.seq,
        }
    }

    /// t-0 holds: c-2 (root, the demand) <- c-3 (its reply), c-4 (an
    /// orphan note), c-5 (a second root) <- c-6 (its reply), with c-4
    /// born between the two conversations.
    fn clustered() -> World {
        World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(
                2,
                2,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Demand,
            ),
            comment_at(
                3,
                3,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(2))),
                CommentKind::Note,
            ),
            comment_at(
                4,
                4,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Note,
            ),
            comment_at(
                5,
                5,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Note,
            ),
            comment_at(
                6,
                6,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(5))),
                CommentKind::Note,
            ),
        ])
        .unwrap()
    }

    fn artifact_record(seq: usize, at: u64, name: &str) -> Record {
        record(
            seq,
            at,
            Tier::Human,
            Event::ArtifactAdded {
                root: TaskId(0),
                artifact: crate::types::artifact::Artifact {
                    name: Prose::new(name.into()).unwrap(),
                    hash: crate::ContentHash::of(name.as_bytes()),
                },
            },
        )
    }

    fn entry_shape(entry: &ThreadEntry) -> (&'static str, usize) {
        match entry {
            ThreadEntry::Comment(c) => ("comment", c.seq),
            ThreadEntry::Artifact(a) => ("artifact", a.seq),
        }
    }

    /// Position is the association: artifacts ride the thread's seq
    /// order, inside the group open where they fall.
    #[test]
    fn artifacts_land_at_their_position_in_the_thread() {
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(
                2,
                2,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Demand,
            ),
            artifact_record(3, 3, "sweep figure"),
            artifact_record(4, 4, "spread figure"),
            comment_at(
                5,
                5,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(2))),
                CommentKind::Note,
            ),
            artifact_record(6, 6, "verdict figure"),
        ])
        .unwrap();
        let v = thread_view(&world, TaskId(0)).unwrap();
        match &v.items[0] {
            ThreadItem::Exchange { root, run, replies } => {
                assert_eq!(root.seq, 2);
                assert!(run.is_none());
                assert_eq!(
                    replies.iter().map(entry_shape).collect::<Vec<_>>(),
                    [
                        ("artifact", 3),
                        ("artifact", 4),
                        ("comment", 5),
                        ("artifact", 6),
                    ]
                );
            }
            other => panic!("expected an exchange, got {other:?}"),
        }
        // the flat show stream holds the same order
        let flat = thread_entries(&world.comments, &world.tasks[0]);
        assert_eq!(
            flat.iter().map(entry_shape).collect::<Vec<_>>(),
            [
                ("comment", 2),
                ("artifact", 3),
                ("artifact", 4),
                ("comment", 5),
                ("artifact", 6),
            ]
        );

        // an artifact no comment precedes stands alone at the top
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            artifact_record(1, 1, "lead figure"),
            comment_at(
                2,
                2,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Note,
            ),
        ])
        .unwrap();
        let v = thread_view(&world, TaskId(0)).unwrap();
        match &v.items[0] {
            ThreadItem::Artifact(line) => {
                assert_eq!(line.seq, 1);
                assert_eq!(line.name, "lead figure");
                assert_eq!(line.hash, crate::ContentHash::of(b"lead figure").as_str());
            }
            other => panic!("expected a lead artifact, got {other:?}"),
        }
        match &v.items[1] {
            ThreadItem::Note(line) => assert_eq!(line.seq, 2),
            other => panic!("expected a note, got {other:?}"),
        }
        // the pointer lookup show's record door uses
        let line = artifact_line(&world, RecordId(1)).expect("the artifact resolves");
        assert_eq!(line.short_hash().len(), 12);
        assert!(artifact_line(&world, RecordId(2)).is_none());
    }

    #[test]
    fn the_thread_clusters_by_reply_links() {
        let world = clustered();
        let v = thread_view(&world, TaskId(0)).unwrap();
        assert_eq!(v.items.len(), 3);
        match &v.items[0] {
            ThreadItem::Exchange { root, run, replies } => {
                // c-2 is an unanswered demand: the exchange opens, no run yet
                assert_eq!(root.seq, 2);
                assert!(run.is_none());
                assert_eq!(replies.iter().map(entry_seq).collect::<Vec<_>>(), [3]);
            }
            other => panic!("expected an exchange, got {other:?}"),
        }
        match &v.items[1] {
            ThreadItem::Note(line) => assert_eq!(line.seq, 4),
            other => panic!("expected a note, got {other:?}"),
        }
        match &v.items[2] {
            ThreadItem::Group { root, replies } => {
                // c-5 is unaddressed: a plain group with its reply
                assert_eq!(root.seq, 5);
                assert_eq!(replies.iter().map(entry_seq).collect::<Vec<_>>(), [6]);
            }
            other => panic!("expected a group, got {other:?}"),
        }
    }

    #[test]
    fn an_answered_demand_carries_its_run() {
        let bind = |seq: usize, at: u64| {
            record(
                seq,
                at,
                Tier::System,
                Event::IncarnationBound {
                    task_id: TaskId(0),
                    response_target: CommentId(RecordId(2)),
                    trigger: RecordId(2),
                    actor: ActorName::new("pi".into()).unwrap(),
                    session: SessionPointer::new("/tmp/s".into()).unwrap(),
                },
            )
        };
        let settle = |seq: usize, at: u64| {
            record(
                seq,
                at,
                Tier::System,
                Event::IncarnationSettled {
                    id: IncarnationId(RecordId(3)),
                },
            )
        };
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(
                2,
                2 * HOUR,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Demand,
            ),
            bind(3, 3 * HOUR),
            record(
                4,
                3 * HOUR + 30 * 60,
                Tier::System,
                Event::IncarnationPromptAccepted {
                    id: IncarnationId(RecordId(3)),
                },
            ),
            comment_at(
                5,
                4 * HOUR,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(2))),
                CommentKind::Note,
            ),
            settle(6, 5 * HOUR),
        ])
        .unwrap();
        let v = thread_view(&world, TaskId(0)).unwrap();
        match &v.items[0] {
            ThreadItem::Exchange { root, run, replies } => {
                assert_eq!(root.seq, 2);
                let run = run.as_ref().expect("the bind answered the demand");
                assert_eq!(run.actor, "pi");
                assert_eq!(run.born_at, 3 * HOUR);
                assert_eq!(run.done_at, Some(5 * HOUR));
                assert!(!run.in_flight());
                assert_eq!(replies.iter().map(entry_seq).collect::<Vec<_>>(), [5]);
            }
            other => panic!("expected an exchange, got {other:?}"),
        }
    }

    #[test]
    fn the_rail_archive_holds_done_and_dropped_only() {
        let world = World::replay(vec![
            task_at(0, 0, "finished work"),
            record(1, 10, Tier::Human, Event::TaskClaimed { id: TaskId(0) }),
            record(
                2,
                20,
                Tier::Human,
                Event::TaskDone {
                    id: TaskId(0),
                    receipt: Prose::new("suite green".into()).unwrap(),
                },
            ),
            task_at(3, 30, "live work"),
            task_at(4, 40, "void work"),
            record(
                5,
                50,
                Tier::Human,
                Event::TaskDropped {
                    id: TaskId(1),
                    note: Prose::new("void".into()).unwrap(),
                },
            ),
            // delivered work awaits its accept: it browses with the live
            task_at(6, 60, "delivered work"),
            record(7, 70, Tier::Human, Event::TaskClaimed { id: TaskId(3) }),
            record(
                8,
                80,
                Tier::Human,
                Event::TaskDelivered {
                    id: TaskId(3),
                    receipt: Prose::new("awaiting the asker".into()).unwrap(),
                },
            ),
        ])
        .unwrap();
        let closed = closed_tasks(&world);
        assert_eq!(closed.len(), 2);
        // most recently settled first: t-1 dropped at 50, t-0 done at 20
        assert_eq!(closed[0].task.id, "t-1");
        assert_eq!(closed[0].task.state, "dropped");
        assert_eq!(closed[1].task.id, "t-0");
        assert_eq!(closed[1].task.state, "done");
        // the live tree holds what is open, claimed, or delivered
        let live = forest(&world);
        assert_eq!(live.len(), 2);
        assert_eq!(live[0].task.id, "t-2");
        assert_eq!(live[1].task.id, "t-3");
        assert_eq!(live[1].task.state, "delivered");
    }

    #[test]
    fn an_active_incarnation_yields_a_run_row() {
        let bind = |seq: usize, at: u64| {
            record(
                seq,
                at,
                Tier::System,
                Event::IncarnationBound {
                    task_id: TaskId(0),
                    response_target: CommentId(RecordId(2)),
                    trigger: RecordId(2),
                    actor: ActorName::new("pi".into()).unwrap(),
                    session: SessionPointer::new("/tmp/s".into()).unwrap(),
                },
            )
        };
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(
                2,
                2 * HOUR,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Demand,
            ),
            bind(3, 3 * HOUR),
            record(
                4,
                3 * HOUR + 30 * 60,
                Tier::System,
                Event::IncarnationPromptAccepted {
                    id: IncarnationId(RecordId(3)),
                },
            ),
        ])
        .unwrap();
        // bound and accepted, not yet settled: the strip names the run
        let next = next_panel(&world, 4 * HOUR);
        assert_eq!(next.runs.len(), 1);
        let r = &next.runs[0];
        assert_eq!(r.incarnation, 3);
        assert_eq!(r.task, 0);
        assert_eq!(r.demand, 2);
        assert_eq!(r.actor, "pi");
        assert!(r.in_flight());
    }

    #[test]
    fn a_refused_demand_carries_its_refusal_on_the_thread() {
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(
                2,
                2 * HOUR,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Demand,
            ),
            record(
                3,
                3 * HOUR,
                Tier::System,
                Event::DemandRefused {
                    demand: CommentId(RecordId(2)),
                    reason: Prose::new(
                        "the worktree is a disk-only leftover; reconcile it through the human"
                            .into(),
                    )
                    .unwrap(),
                },
            ),
        ])
        .unwrap();
        let v = thread_view(&world, TaskId(0)).unwrap();
        match &v.items[0] {
            ThreadItem::Exchange { root, run, replies } => {
                assert_eq!(root.seq, 2);
                assert!(run.is_none(), "a refused demand never bound a run");
                assert!(replies.is_empty());
                assert_eq!(root.state.as_deref(), Some("demand, refused"));
                let refusal = root.refusal.as_ref().expect("the refusal rides the line");
                assert_eq!(refusal.at, 3 * HOUR);
                assert!(refusal.reason.contains("disk-only leftover"));
            }
            other => panic!("expected an exchange, got {other:?}"),
        }
    }

    #[test]
    fn asked_of_you_scans_awaiting_human_demands() {
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            // agent asks the human: it shows up
            comment_at(1, 1, Tier::Agent, Target::Task(TaskId(0)), CommentKind::Ask),
            // an unaddressed note never asks
            comment_at(
                2,
                2,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Note,
            ),
        ])
        .unwrap();
        let asked = asked_of_you(&world);
        assert_eq!(asked.len(), 1);
        assert_eq!(asked[0].comment, 1);
        assert_eq!(asked[0].task, 0);
        assert_eq!(asked[0].actor, "pi");

        // the human's reply answers it and the scan empties
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(1, 1, Tier::Agent, Target::Task(TaskId(0)), CommentKind::Ask),
            comment_at(
                2,
                2,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Note,
            ),
            comment_at(
                3,
                3,
                Tier::Human,
                Target::Comment(CommentId(RecordId(1))),
                CommentKind::Note,
            ),
        ])
        .unwrap();
        assert!(asked_of_you(&world).is_empty());
    }

    #[test]
    fn claimed_tasks_carry_ages_and_drift_after_a_day() {
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            record(1, 100, Tier::Human, Event::TaskClaimed { id: TaskId(0) }),
            task_at(2, 200, "quiet work"),
        ])
        .unwrap();
        let now = 100 + ADRIFT_AFTER_SECS - 1;
        let next = next_panel(&world, now);
        assert_eq!(next.candidates.len(), 1);
        let c = &next.candidates[0];
        assert_eq!(c.task, "t-0");
        assert_eq!(c.claim_age, ADRIFT_AFTER_SECS - 1);
        assert_eq!(c.last_record_age, ADRIFT_AFTER_SECS - 1);
        assert!(!c.adrift);
        // one more second of silence is adrift
        let next = next_panel(&world, now + 1);
        assert!(next.candidates[0].adrift);
        // the untouched open task never becomes a candidate
        assert_eq!(next.runs.len(), 0);
    }

    #[test]
    fn ribbon_marks_kinds_and_window() {
        let bind = |seq: usize, at: u64, trigger: RecordId| {
            record(
                seq,
                at,
                Tier::System,
                Event::IncarnationBound {
                    task_id: TaskId(0),
                    response_target: CommentId(RecordId(2)),
                    trigger,
                    actor: ActorName::new("pi".into()).unwrap(),
                    session: SessionPointer::new("/tmp/s".into()).unwrap(),
                },
            )
        };
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(
                2,
                10 * HOUR,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Demand,
            ),
            bind(3, 11 * HOUR, RecordId(2)),
            comment_at(
                4,
                80 * HOUR,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Note,
            ),
            comment_at(
                5,
                90 * HOUR,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Note,
            ),
        ])
        .unwrap();
        let now = 100 * HOUR;
        let marks = ribbon_marks(&world, now);
        // the 10h comment is older than 72h; 80h and 90h stay
        assert_eq!(marks.iter().map(|m| m.seq).collect::<Vec<_>>(), [4, 5]);
        assert!(
            marks
                .iter()
                .all(|m| m.kind == MarkKind::Note { human: true })
        );

        let fresh = World::replay(vec![
            task_at(0, 0, "real work"),
            comment_at(
                2,
                10 * HOUR,
                Tier::Human,
                Target::Task(TaskId(0)),
                CommentKind::Demand,
            ),
            bind(3, 11 * HOUR, RecordId(2)),
            comment_at(
                4,
                12 * HOUR,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(2))),
                CommentKind::Note,
            ),
        ])
        .unwrap();
        let marks = ribbon_marks(&fresh, 13 * HOUR);
        // the run marks the demand it answered, at the bind's birth time
        assert_eq!(
            marks.iter().map(|m| (m.seq, m.kind)).collect::<Vec<_>>(),
            [
                (2, MarkKind::Demand),
                (2, MarkKind::Run),
                (4, MarkKind::Note { human: false }),
            ]
        );
    }
}

#[cfg(test)]
mod search {
    use super::*;
    use crate::CommentKind;
    use crate::events::Event;
    use crate::objects::comment::Target;
    use crate::store::{Context, Record, RecordId, Tier, World};
    use crate::types::actor::ActorName;
    use crate::types::prose::Prose;

    fn ctx(tier: Tier, name: &str) -> Context {
        Context {
            actor: ActorName::new(name.into()).unwrap(),
            tier,
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

    fn task(seq: usize, at: u64, ctx: &Context, name: &str, parent: Option<TaskId>) -> Record {
        record(
            seq,
            at,
            ctx,
            Event::TaskCreated {
                name: Prose::new(name.into()).unwrap(),
                parent_id: parent,
            },
        )
    }

    fn note(seq: usize, at: u64, ctx: &Context, target: Target, body: &str) -> Record {
        record(
            seq,
            at,
            ctx,
            Event::Commented {
                target,
                body: Prose::new(body.into()).unwrap(),
                kind: CommentKind::Note,
            },
        )
    }

    /// t-0 "implement foo" holds the demand #2, its reply #3 (pi), and
    /// the later answer #9 (bot, citing #2 by address alone); t-1
    /// "migrate floop" is born under t-0, named in prose by #5, carried
    /// a run-name line in #6, and was delivered by the bot with a
    /// receipt citing #2; t-2's title guards the id boundary.
    fn story() -> World {
        let jerry = ctx(Tier::Human, "jerry");
        let bot = ctx(Tier::Agent, "saccade bot");
        let pi = ctx(Tier::Agent, "pi");
        World::replay(vec![
            task(0, 0, &jerry, "implement foo", None),
            task(1, 10, &bot, "migrate floop", Some(TaskId(0))),
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
            task(4, 40, &jerry, "t-490 the id boundary guard", None),
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
                "the run pi/t-90-1 answered\nsee #123 for the trail",
            ),
            record(7, 70, &bot, Event::TaskClaimed { id: TaskId(1) }),
            record(
                8,
                80,
                &bot,
                Event::TaskDelivered {
                    id: TaskId(1),
                    receipt: Prose::new("suite 9 green; the floop migration landed per #2".into())
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
            task(0, 0, &jerry, "hold the sweep", None),
            record(
                1,
                1,
                &jerry,
                Event::ArtifactAdded {
                    root: TaskId(0),
                    artifact: crate::types::artifact::Artifact {
                        name: Prose::new("sweep-overview".into()).unwrap(),
                        hash: crate::ContentHash::of(b"figure bytes"),
                    },
                },
            ),
            note(
                2,
                2,
                &jerry,
                Target::Task(TaskId(0)),
                "the verdict cites #1 above",
            ),
        ])
        .unwrap();
        // the name matches as a whole token; the title's "sweep" does not
        let groups = search(&world, &q(&["sweep-overview"])).unwrap();
        assert_eq!(pointers(&groups), vec![(0, vec!["#1".into()])]);
        let groups = &groups[0];
        assert_eq!(groups.records[0].kind, "artifact");
        assert_eq!(groups.records[0].actor, None);
        assert_eq!(groups.records[0].body, "sweep-overview");
        // the kind facet narrows to artifacts alone
        let groups = search(&world, &q(&["sweep-overview", "kind:artifact"])).unwrap();
        assert_eq!(pointers(&groups), vec![(0, vec!["#1".into()])]);
        // and the mention is a reference search finding its namer
        let groups = search(&world, &q(&["#1"])).unwrap();
        assert_eq!(pointers(&groups), vec![(0, vec!["#2".into()])]);
    }

    #[test]
    fn a_plain_term_reads_titles_bodies_and_receipts() {
        let groups = search(&story(), &q(&["floop"])).unwrap();
        // threads enter by first match: t-1's title at #1 before t-0's #3
        assert_eq!(
            pointers(&groups),
            vec![
                (1, vec!["#1".into(), "t-1 receipt".into()]),
                (0, vec!["#3".into()])
            ]
        );
        let title = &groups[0].records[0];
        assert_eq!(
            (title.kind, title.actor.as_deref()),
            ("task", Some("saccade bot"))
        );
        let receipt = &groups[0].records[1];
        // the receipt carries no author in the fold and closes its group
        assert_eq!((receipt.kind, receipt.actor.as_deref()), ("receipt", None));
        assert_eq!(
            receipt.body,
            "suite 9 green; the floop migration landed per #2"
        );
        assert_eq!(groups[0].title, "migrate floop");
    }

    #[test]
    fn matching_is_case_folded_and_word_bounded() {
        let world = story();
        assert_eq!(search(&world, &q(&["FLOOP"])).unwrap().len(), 2);
        // plurals and embedded ids never match
        assert!(search(&world, &q(&["telemetry"])).unwrap().is_empty());
        assert!(search(&world, &q(&["t-49"])).unwrap().is_empty());
        assert!(search(&world, &q(&["t-90"])).unwrap().is_empty());
        assert!(search(&world, &q(&["#12"])).unwrap().is_empty());
        // the tokens themselves do
        assert_eq!(
            pointers(&search(&world, &q(&["t-490"])).unwrap()),
            vec![(2, vec!["#4".into()])]
        );
        let groups = search(&world, &q(&["#123"])).unwrap();
        assert_eq!(pointers(&groups), vec![(1, vec!["#6".into()])]);
        // the record carries the field whole; the matched line is a
        // render cut of it
        assert_eq!(
            groups[0].records[0].body,
            "the run pi/t-90-1 answered\nsee #123 for the trail"
        );
        assert_eq!(
            matched_line(&q(&["#123"]).terms, &groups[0].records[0].body),
            "see #123 for the trail"
        );
    }

    #[test]
    fn an_id_term_is_a_reference_search() {
        let world = story();
        // '#2': the receipt cites it in prose, #3 and #9 address it by reply
        assert_eq!(
            pointers(&search(&world, &q(&["#2"])).unwrap()),
            vec![
                (0, vec!["#3".into(), "#9".into()]),
                (1, vec!["t-1 receipt".into()])
            ]
        );
        // 't-0': #1 names it as parent, #5 in prose, #2 by address
        assert_eq!(
            pointers(&search(&world, &q(&["t-0"])).unwrap()),
            vec![(1, vec!["#1".into(), "#5".into()]), (0, vec!["#2".into()])]
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
        // by: a receipt carries no author, so the filter narrows it out
        let groups = search(&world, &q(&["floop", "by:pi"])).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|g| (g.task, g.records.len(), g.total))
                .collect::<Vec<_>>(),
            vec![(1, 0, 2), (0, 1, 1)]
        );
        // kind: receipts only
        let groups = search(&world, &q(&["floop", "kind:receipt"])).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|g| (g.task, g.records.iter().map(|r| r.kind).collect::<Vec<_>>()))
                .collect::<Vec<_>>(),
            vec![(1, vec!["receipt"]), (0, vec![])]
        );
        // under: t-0's subtree holds t-0 and t-1; t-2's match stays
        // visible as a count-only row, never silently excluded
        let groups = search(&world, &q(&["t-490", "under:t-0"])).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|g| (g.task, g.records.len(), g.total))
                .collect::<Vec<_>>(),
            vec![(2, 0, 1)]
        );
        let groups = search(&world, &q(&["floop", "under:t-0"])).unwrap();
        assert_eq!(
            groups.iter().map(|g| (g.task, g.total)).collect::<Vec<_>>(),
            vec![(1, 2), (0, 1)]
        );
    }

    #[test]
    fn a_facet_may_browse_a_whole_thread() {
        // a facet alone matches everything, narrowed by the facet; only
        // the threads the facet matches render — no count-row flood
        let groups = search(&story(), &q(&["in:t-0"])).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|g| (
                    g.task,
                    g.records
                        .iter()
                        .map(|r| r.pointer.clone())
                        .collect::<Vec<_>>(),
                    g.total
                ))
                .collect::<Vec<_>>(),
            vec![(
                0,
                vec!["#0".into(), "#2".into(), "#3".into(), "#9".into()],
                4
            )]
        );
        // a facet alone that nothing passes renders nothing at all
        assert!(search(&story(), &q(&["by:nobody"])).unwrap().is_empty());
    }

    #[test]
    fn term_matches_filtered_by_facets_stay_as_count_rows() {
        // a term's matches stay visible as counts when a facet filters
        // them out — narrowing is visible, never silent
        let groups = search(&story(), &q(&["floop", "by:jerry"])).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|g| (g.task, g.records.len(), g.total))
                .collect::<Vec<_>>(),
            vec![(1, 0, 2), (0, 0, 1)]
        );
    }

    #[test]
    fn unknown_facet_tasks_and_grammar_refuse() {
        let world = story();
        assert!(matches!(
            search(&world, &q(&["in:t-9"])),
            Err(SearchFail::NoSuchTask {
                asked: TaskId(9),
                holds: 3
            })
        ));
        let owned = |t: &str| vec![t.to_string()];
        assert!(
            SearchQuery::parse(&owned("kind:bogus"))
                .unwrap_err()
                .to_string()
                .contains("the kinds are")
        );
        assert!(matches!(SearchQuery::parse(&[]), Err(SearchFail::Usage(_))));
        assert!(matches!(
            SearchQuery::parse(&owned("in:x")),
            Err(SearchFail::Usage(_))
        ));
        assert!(matches!(
            SearchQuery::parse(&owned("by:")),
            Err(SearchFail::Usage(_))
        ));
        assert_eq!(
            q(&["#2", "t-3", "Floop"]),
            SearchQuery {
                terms: vec![
                    Term::Comment(RecordId(2)),
                    Term::Task(TaskId(3)),
                    Term::Plain("floop".into())
                ],
                facets: Vec::new()
            }
        );
    }
}
