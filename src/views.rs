//! Statements: the render shapes and their constructors, one per reader.
//! Views read objects and contexts; nothing in objects knows views exist.

use std::collections::BTreeMap;

use crate::objects::comment::{
    AgentAttemptState, CommentContext, CommentId, CommentState, ResponseState, Target,
};
use crate::objects::proposal::{ProposalAction, ProposalContext, ProposalId, ProposalState};
use crate::objects::task::{TaskContext, TaskId, TaskState};
use crate::store::World;
use crate::types::prose::Prose;

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
pub struct CommentLine {
    pub seq: usize,
    pub depth: usize,
    pub actor: String,
    pub body: String,
    pub state: Option<String>,
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
        .map(|cid| {
            let cctx = comments
                .get(cid)
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
                body: cctx.comment.body.as_str().to_string(),
                state: demand_tag(&cctx.state),
            }
        })
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

/// Follow the world's pointers to one task's thread.
pub fn thread_view(world: &World, id: TaskId) -> Option<Vec<CommentLine>> {
    world
        .tasks
        .get(id.0)
        .map(|ctx| comment_thread(&world.comments, ctx))
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
