//! Statements: the render shapes and their constructors, one per reader.
//! Views read objects and contexts; nothing in objects knows views exist.

use std::collections::BTreeMap;

use crate::objects::comment::{
    AgentAttemptState, CommentContext, CommentId, CommentState, ResponseState, Target,
};
use crate::objects::proposal::{ProposalAction, ProposalContext, ProposalId, ProposalState};
use crate::objects::task::{TaskContext, TaskId, TaskState};
use crate::store::{Tier, World};
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
    pub body: String,
    pub state: Option<String>,
    pub born_at: u64,
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
        replies: Vec<CommentLine>,
    },
    Group {
        root: CommentLine,
        replies: Vec<CommentLine>,
    },
    Note(CommentLine),
}

/// The focused task's thread as a view: the context's pointer index,
/// followed, grouped by reply links, ordered by first comment.
pub struct ThreadView {
    pub items: Vec<ThreadItem>,
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
        body: cctx.comment.body.as_str().to_string(),
        state: demand_tag(&cctx.state),
        born_at: cctx.born_at,
    }
}

/// The task's thread as a view: the context's pointer index, followed.
/// Membership is fixed at birth, so this walks pointers, never scans.
// TODO: this walks the parent chain for depth while the Commented fold
// walks the same chain for root; maybe stamp depth at birth instead
pub fn comment_thread(
    comments: &BTreeMap<CommentId, CommentContext>,
    ctx: &TaskContext,
) -> Vec<CommentLine> {
    ctx.thread
        .iter()
        .map(|cid| line_of(comments, *cid))
        .collect()
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
                TaskState::Done(r) => Some(r.as_str().to_string()),
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

/// The demand tag a thread row carries: who it addresses and where the
/// response stands. Unaddressed rows carry nothing.
fn demand_tag(state: &CommentState) -> Option<String> {
    match state {
        CommentState::Unaddressed => None,
        CommentState::AddressedToHuman { response } => match response {
            ResponseState::Awaiting => Some("to human, awaiting".into()),
            ResponseState::Responded { .. } => Some("to human, responded".into()),
        },
        CommentState::AddressedToAgent { response, attempt } => {
            let response = match response {
                ResponseState::Awaiting => "awaiting",
                ResponseState::Responded { .. } => "responded",
            };
            let attempt = match attempt {
                AgentAttemptState::Authorized { .. } => "",
                AgentAttemptState::InFlight { .. } => ", in flight",
                AgentAttemptState::Spent => "",
            };
            Some(format!("to agent, {response}{attempt}"))
        }
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
            replies: Vec<CommentLine>,
        },
        Group {
            root: CommentLine,
            replies: Vec<CommentLine>,
        },
    }
    let mut groups: Vec<Building> = Vec::new();
    let mut root_index: BTreeMap<CommentId, usize> = BTreeMap::new();
    for cid in &ctx.thread {
        let root = root_of(&world.comments, *cid);
        if root == *cid {
            let line = line_of(&world.comments, *cid);
            root_index.insert(*cid, groups.len());
            if is_agent_demand(&line.state) {
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
        } else {
            let g = root_index
                .get(&root)
                .copied()
                .expect("a reply's root is resident and earlier");
            let line = line_of(&world.comments, *cid);
            match &mut groups[g] {
                Building::Exchange { replies, .. } | Building::Group { replies, .. } => {
                    replies.push(line)
                }
            }
        }
    }
    let items = groups
        .into_iter()
        .map(|g| match g {
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
        })
        .collect();
    Some(ThreadView { items })
}

/// An agent-addressed demand opens an exchange; a human-addressed or
/// unaddressed root is an ordinary group.
fn is_agent_demand(state: &Option<String>) -> bool {
    state.as_deref().is_some_and(|s| s.starts_with("to agent"))
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

/// The live tree as navigation: open and claimed tasks, parents before
/// children, each row carrying the open-proposal mark it already has.
pub fn forest(world: &World) -> Vec<ForestRow> {
    world
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, ctx)| matches!(ctx.task.state, TaskState::Open | TaskState::Claimed))
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

/// A human-addressed demand awaiting an answer.
pub struct AskedOfYou {
    pub comment: usize,
    pub task: usize,
    pub actor: String,
}

/// Every AddressedToHuman demand still awaiting, oldest first — the
/// mirror of the supervisor's runnable-demand scan.
pub fn asked_of_you(world: &World) -> Vec<AskedOfYou> {
    world
        .comments
        .iter()
        .filter(|(_, c)| {
            matches!(
                c.state,
                CommentState::AddressedToHuman {
                    response: ResponseState::Awaiting,
                }
            )
        })
        .map(|(id, c)| AskedOfYou {
            comment: id.0.0,
            task: c.comment.root.0,
            actor: c.actor.as_str().to_string(),
        })
        .collect()
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
    }
    NextPanel {
        runs,
        asked_of_you: asked_of_you(world),
        candidates,
    }
}

pub struct NextPanel {
    pub runs: Vec<RunView>,
    pub asked_of_you: Vec<AskedOfYou>,
    pub candidates: Vec<Candidate>,
}

/// The kind a ribbon mark carries, keyed as the strip names them: a
/// note by its author's tier, a demand, or a run's birth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarkKind {
    Note { human: bool },
    Demand,
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
            CommentState::Unaddressed => MarkKind::Note {
                human: c.tier == Tier::Human,
            },
            _ => MarkKind::Demand,
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

#[cfg(test)]
mod test {
    use super::*;
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
            claimed_at: None,
            last_record_at: 0,
            proposal: None,
            thread: Vec::new(),
            holder: None,
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
                addressee: None,
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
    use crate::Addressee;
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

    fn comment_at(
        seq: usize,
        at: u64,
        tier: Tier,
        target: Target,
        to: Option<Addressee>,
    ) -> Record {
        record(
            seq,
            at,
            tier,
            Event::Commented {
                target,
                body: Prose::new("a body worth keeping".into()).unwrap(),
                addressee: to,
            },
        )
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
                Some(Addressee::Agent),
            ),
            comment_at(
                3,
                3,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(2))),
                None,
            ),
            comment_at(4, 4, Tier::Human, Target::Task(TaskId(0)), None),
            comment_at(5, 5, Tier::Human, Target::Task(TaskId(0)), None),
            comment_at(
                6,
                6,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(5))),
                None,
            ),
        ])
        .unwrap()
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
                assert_eq!(replies.iter().map(|r| r.seq).collect::<Vec<_>>(), [3]);
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
                assert_eq!(replies.iter().map(|r| r.seq).collect::<Vec<_>>(), [6]);
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
                Some(Addressee::Agent),
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
                None,
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
                assert_eq!(replies.iter().map(|r| r.seq).collect::<Vec<_>>(), [5]);
            }
            other => panic!("expected an exchange, got {other:?}"),
        }
    }

    #[test]
    fn asked_of_you_scans_awaiting_human_demands() {
        let world = World::replay(vec![
            task_at(0, 0, "real work"),
            // agent asks the human: it shows up
            comment_at(
                1,
                1,
                Tier::Agent,
                Target::Task(TaskId(0)),
                Some(Addressee::Human),
            ),
            // an unaddressed note never asks
            comment_at(2, 2, Tier::Human, Target::Task(TaskId(0)), None),
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
            comment_at(
                1,
                1,
                Tier::Agent,
                Target::Task(TaskId(0)),
                Some(Addressee::Human),
            ),
            comment_at(2, 2, Tier::Human, Target::Task(TaskId(0)), None),
            comment_at(
                3,
                3,
                Tier::Human,
                Target::Comment(CommentId(RecordId(1))),
                None,
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
                Some(Addressee::Agent),
            ),
            bind(3, 11 * HOUR, RecordId(2)),
            comment_at(4, 80 * HOUR, Tier::Human, Target::Task(TaskId(0)), None),
            comment_at(5, 90 * HOUR, Tier::Human, Target::Task(TaskId(0)), None),
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
                Some(Addressee::Agent),
            ),
            bind(3, 11 * HOUR, RecordId(2)),
            comment_at(
                4,
                12 * HOUR,
                Tier::Agent,
                Target::Comment(CommentId(RecordId(2))),
                None,
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
