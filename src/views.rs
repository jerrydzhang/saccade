//! Statements: the render shapes and their constructors, one per reader.
//! Views read objects and contexts; nothing in objects knows views exist.

use std::collections::BTreeMap;

use crate::objects::comment::{CommentContext, CommentId, Target};
use crate::objects::proposal::{ProposalAction, ProposalContext, ProposalId, ProposalState};
use crate::objects::task::{TaskContext, TaskId, TaskState};
use crate::prose::Prose;
use crate::store::World;

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

impl TaskView {
    pub fn of(id: TaskId, ctx: &TaskContext, proposal: Option<&ProposalContext>) -> TaskView {
        TaskView {
            id: format!("t-{}", id.0),
            state: match ctx.task.state {
                TaskState::Open => "open",
                TaskState::Claimed => "claimed",
                TaskState::Done(_) => "done",
                TaskState::Dropped => "dropped",
            },
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
            .validate(&ctx.proposal.action.target_event(&probe))
            .is_err(),
    }
}

/// A row of the task's thread view. Depth is position in the rendered
/// thread — the task is the root, so a comment addressing it sits at 1.
pub struct CommentLine {
    pub seq: usize,
    pub depth: usize,
    pub actor: String,
    pub body: String,
}

/// The task's thread as a view: the context's pointer index, followed.
/// Membership is fixed at birth, so this walks pointers, never scans.
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
                actor: cctx.actor.clone(),
                body: cctx.comment.body.as_str().to_string(),
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
            state: match ctx.task.state {
                TaskState::Open => "open",
                TaskState::Claimed => "claimed",
                TaskState::Done(_) => "done",
                TaskState::Dropped => "dropped",
            },
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
    use crate::objects::task::Task;
    use crate::store::RecordId;

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
}
