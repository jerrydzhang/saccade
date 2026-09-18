use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::decide::{decide, enforce_tier, expand};
use crate::events::{Command, Event};
use crate::objects::comment::{
    Addressee, AgentAttemptState, Comment, CommentContext, CommentId, CommentState, ResponseState,
};
use crate::objects::incarnation::{IncarnationContext, IncarnationId, IncarnationState};
use crate::objects::proposal::{Proposal, ProposalContext, ProposalId, ProposalState};
use crate::objects::task::{Task, TaskContext, TaskState};
use crate::objects::workspace::{WorkspaceContext, WorktreeState};
use crate::types::actor::ActorName;
use crate::{ProposalAction, Prose, Reject, Target};

#[derive(Debug)]
pub struct FoldError {
    pub at: RecordId,
    pub reason: Reason,
}

#[derive(Debug)]
pub enum Reason {
    /// The parent task of a TaskCreated event does not exist.
    InvalidParentTaskId,
    /// The record's author is not the current claim holder.
    NotClaimHolder,
    InvalidIncarnationId,
    /// The task's run slot is taken: one live incarnation per task.
    IncarnationAlreadyActive,
    /// A binding names a demand that lives on another task.
    DemandNotOnTask,
    /// The task already holds a workspace: one lineage per task.
    WorkspaceAlreadyExists,
    /// The event names a task whose workspace was never created.
    WorkspaceMissing,
    /// A second physical creation while the worktree is present.
    WorktreeAlreadyPresent,
    InvalidTaskId,
    InvalidProposalId,
    InvalidCommentId,
    InvalidStateTransition,
    /// A task already holds an open judgment proposal
    ProposalAlreadyOpen,
}

/// Command-path translation of fold failures into refusals; total over Reason
/// since every fold failure is command-reachable.
impl From<Reason> for Reject {
    fn from(reason: Reason) -> Reject {
        match reason {
            Reason::InvalidTaskId => Reject::InvalidTaskId,
            Reason::InvalidProposalId => Reject::InvalidProposalId,
            Reason::InvalidCommentId => Reject::InvalidCommentId,
            Reason::InvalidParentTaskId => Reject::InvalidParentTaskId,
            Reason::InvalidStateTransition => Reject::InvalidStateTransition,
            Reason::NotClaimHolder => Reject::NotClaimHolder,
            Reason::InvalidIncarnationId => Reject::InvalidIncarnationId,
            Reason::IncarnationAlreadyActive => Reject::IncarnationAlreadyActive,
            Reason::DemandNotOnTask => Reject::DemandNotOnTask,
            Reason::WorkspaceAlreadyExists => Reject::WorkspaceAlreadyExists,
            Reason::WorkspaceMissing => Reject::WorkspaceMissing,
            Reason::WorktreeAlreadyPresent => Reject::WorktreeAlreadyPresent,
            Reason::ProposalAlreadyOpen => Reject::ProposalAlreadyOpen,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Human,
    Agent,
    System,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub actor: ActorName,
    pub tier: Tier,
}

impl Context {
    /// The fixed author machinery records under; no caller presents this.
    pub fn system() -> Self {
        Context {
            actor: ActorName::new("saccade".into()).expect("the fixed system actor name is valid"),
            tier: Tier::System,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecordId(pub usize);

#[derive(Clone, Debug)]
pub struct Record {
    pub id: RecordId,
    pub timestamp: u64,
    pub context: Context,
    pub event: Event,
}

#[derive(Clone, Debug, PartialEq)]
pub struct World {
    pub tasks: Vec<TaskContext>,
    pub proposals: BTreeMap<ProposalId, ProposalContext>,
    pub comments: BTreeMap<CommentId, CommentContext>,
    pub incarnations: BTreeMap<IncarnationId, IncarnationContext>,
}

impl World {
    pub fn new() -> Self {
        World {
            tasks: Vec::new(),
            proposals: BTreeMap::new(),
            comments: BTreeMap::new(),
            incarnations: BTreeMap::new(),
        }
    }

    pub fn replay(records: Vec<Record>) -> Result<World, FoldError> {
        World::new().fold(records)
    }

    pub fn fold(self, records: Vec<Record>) -> Result<World, FoldError> {
        let mut world = self.clone();
        for record in records {
            world = world.apply(record)?;
        }
        Ok(world)
    }

    /// Only valid path for world mutation
    pub fn apply(mut self, record: Record) -> Result<Self, FoldError> {
        let at = record.id;
        self.apply_record(record)
            .map_err(|reason| FoldError { at, reason })?;
        Ok(self)
    }

    /// A run ending: the task's slot frees and the demand's attempt spends
    fn terminalize(&mut self, id: IncarnationId, record: &Record) {
        if let Some(run) = self.incarnations.get(&id) {
            if let Some(task_ctx) = self.tasks.get_mut(run.task_id.0)
                && task_ctx.active_incarnation == Some(id)
            {
                task_ctx.active_incarnation = None;
                task_ctx.last_updated = record.id;
            }
            if let Some(demand) = self.comments.get_mut(&run.response_target)
                && let Some(next) = demand.state.transition(&record.event, record)
            {
                demand.state = next;
            }
        }
    }

    fn apply_record(&mut self, record: Record) -> Result<(), Reason> {
        match record.event {
            // Task events
            Event::TaskCreated {
                name: task_name,
                parent_id,
            } => {
                if parent_id.is_some_and(|parent| parent.0 >= self.tasks.len()) {
                    return Err(Reason::InvalidParentTaskId);
                }

                self.tasks.push(TaskContext {
                    task: Task {
                        state: TaskState::Open,
                        name: task_name,
                        parent_id,
                    },
                    last_updated: record.id,
                    proposal: None,
                    thread: Vec::new(),
                    holder: None,
                    active_incarnation: None,
                    workspace: None,
                });
            }
            ref event @ Event::TaskClaimed { id } => {
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_updated = record.id;
                task_ctx.task.state = task_ctx
                    .task
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
                task_ctx.holder = Some(record.context.actor.clone());
            }
            ref event @ Event::TaskDone { id, .. } => {
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.task.state = task_ctx
                    .task
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
                // only the holder completes the claim
                if task_ctx.holder.as_ref() != Some(&record.context.actor) {
                    return Err(Reason::NotClaimHolder);
                }
                task_ctx.last_updated = record.id;
                task_ctx.holder = None;
            }
            ref event @ Event::TaskReleased { id, .. } => {
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.task.state = task_ctx
                    .task
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
                // humans may release any claim, agents only their own
                if record.context.tier != Tier::Human
                    && task_ctx.holder.as_ref() != Some(&record.context.actor)
                {
                    return Err(Reason::NotClaimHolder);
                }
                task_ctx.last_updated = record.id;
                task_ctx.holder = None;
            }
            ref event @ Event::TaskDropped { id, .. } => {
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_updated = record.id;
                task_ctx.task.state = task_ctx
                    .task
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
            }
            // Incarnation events
            ref event @ Event::IncarnationBound {
                ref task_id,
                ref response_target,
                ref trigger,
                // ref required since actor and session are not Copy, but we don't want to move them out of the record
                ref actor,
                ref session,
            } => {
                let task_ctx = self.tasks.get_mut(task_id.0).ok_or(Reason::InvalidTaskId)?;
                if task_ctx.active_incarnation.is_some() {
                    return Err(Reason::IncarnationAlreadyActive);
                }
                let demand = self
                    .comments
                    .get_mut(response_target)
                    .ok_or(Reason::InvalidCommentId)?;
                // the run binds a demand on its own task
                if demand.comment.root != *task_id {
                    return Err(Reason::DemandNotOnTask);
                }
                demand.state = demand
                    .state
                    .transition(event, &record)
                    .ok_or(Reason::InvalidStateTransition)?;
                self.incarnations.insert(
                    IncarnationId(record.id),
                    IncarnationContext {
                        task_id: *task_id,
                        response_target: *response_target,
                        trigger: *trigger,
                        actor: actor.clone(),
                        session: session.clone(),
                        state: IncarnationState::Bound,
                        produced: Vec::new(),
                    },
                );
                task_ctx.active_incarnation = Some(IncarnationId(record.id));
                task_ctx.last_updated = record.id;
            }
            ref event @ Event::IncarnationPromptAccepted { id } => {
                let run = self
                    .incarnations
                    .get_mut(&id)
                    .ok_or(Reason::InvalidIncarnationId)?;
                run.state = run
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
            }
            ref event @ Event::IncarnationPromptRejected { id, .. } => {
                let run = self
                    .incarnations
                    .get_mut(&id)
                    .ok_or(Reason::InvalidIncarnationId)?;
                run.state = run
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
                self.terminalize(id, &record);
            }
            ref event @ Event::IncarnationSettled { id } => {
                let run = self
                    .incarnations
                    .get_mut(&id)
                    .ok_or(Reason::InvalidIncarnationId)?;
                run.state = run
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
                self.terminalize(id, &record);
            }
            ref event @ Event::RecordProducedBy {
                record_id,
                incarnation_id,
            } => {
                let run = self
                    .incarnations
                    .get_mut(&incarnation_id)
                    .ok_or(Reason::InvalidIncarnationId)?;
                run.state = run
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
                // produced pointers point backward, at records already born
                if record_id.0 >= record.id.0 {
                    return Err(Reason::InvalidStateTransition);
                }
                run.produced.push(record_id);
            }
            // Workspace events
            Event::TaskWorkspaceCreated {
                task_id,
                ref base,
                ref branch,
            } => {
                let task_ctx = self.tasks.get_mut(task_id.0).ok_or(Reason::InvalidTaskId)?;
                if task_ctx.workspace.is_some() {
                    return Err(Reason::WorkspaceAlreadyExists);
                }
                task_ctx.workspace = Some(WorkspaceContext {
                    base: base.clone(),
                    branch: branch.clone(),
                    // the checkpoint starts at the base the workspace was cut from
                    checkpoint: base.clone(),
                    worktree: WorktreeState::Absent,
                });
                task_ctx.last_updated = record.id;
            }
            ref event @ Event::TaskWorktreeCreated { task_id, .. } => {
                let task_ctx = self.tasks.get_mut(task_id.0).ok_or(Reason::InvalidTaskId)?;
                let workspace = task_ctx
                    .workspace
                    .as_mut()
                    .ok_or(Reason::WorkspaceMissing)?;
                workspace.worktree = workspace
                    .worktree
                    .transition(event)
                    .ok_or(Reason::WorktreeAlreadyPresent)?;
                task_ctx.last_updated = record.id;
            }
            Event::TaskWorkspaceCheckpointed {
                task_id,
                ref checkpoint,
            } => {
                let task_ctx = self.tasks.get_mut(task_id.0).ok_or(Reason::InvalidTaskId)?;
                let workspace = task_ctx
                    .workspace
                    .as_mut()
                    .ok_or(Reason::WorkspaceMissing)?;
                workspace.checkpoint = checkpoint.clone();
                task_ctx.last_updated = record.id;
            }

            // Proposal events
            Event::ProposalCreated {
                name: proposal_name,
                action,
            } => {
                let proposal_id = ProposalId(record.id);

                // one judgment at a time
                match action {
                    ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => {
                        let task_ctx =
                            self.tasks.get_mut(task_id.0).ok_or(Reason::InvalidTaskId)?;
                        if task_ctx.proposal.is_some() {
                            return Err(Reason::ProposalAlreadyOpen);
                        }
                        let probe = Prose::new("probe".into()).expect("probe is non-empty");
                        if task_ctx
                            .task
                            .state
                            .transition(&action.target_event(&probe))
                            .is_none()
                        {
                            return Err(Reason::InvalidStateTransition);
                        }
                        task_ctx.proposal = Some(proposal_id);
                    }
                }

                self.proposals.insert(
                    proposal_id,
                    ProposalContext {
                        proposal: Proposal {
                            state: ProposalState::Open,
                            name: proposal_name,
                            action,
                        },
                    },
                );
            }
            ref event @ (Event::ProposalWithdrawn { id, .. }
            | Event::ProposalRejected { id, .. }
            | Event::ProposalAccepted { id, .. }) => {
                let proposal_ctx = self
                    .proposals
                    .get_mut(&id)
                    .ok_or(Reason::InvalidProposalId)?;
                proposal_ctx.proposal.state = proposal_ctx
                    .proposal
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;

                match &proposal_ctx.proposal.action {
                    ProposalAction::Drop { task_id } | ProposalAction::Release { task_id } => {
                        self.tasks
                            .get_mut(task_id.0)
                            .ok_or(Reason::InvalidTaskId)?
                            .proposal = None;
                    }
                }
            }
            // Comment events
            Event::Commented {
                ref target,
                ref body,
                addressee,
            } => {
                let mut up = *target;
                let root_task_id = loop {
                    match up {
                        Target::Task(t) => break t,
                        Target::Comment(parent) => {
                            up = self
                                .comments
                                .get(&parent)
                                .ok_or(Reason::InvalidCommentId)?
                                .comment
                                .target;
                        }
                    }
                };

                let state = match addressee {
                    None => CommentState::Unaddressed,
                    Some(Addressee::Human) => CommentState::AddressedToHuman {
                        response: ResponseState::Awaiting,
                    },
                    Some(Addressee::Agent) => CommentState::AddressedToAgent {
                        response: ResponseState::Awaiting,
                        attempt: AgentAttemptState::Authorized { trigger: record.id },
                    },
                };

                self.comments.insert(
                    CommentId(record.id),
                    CommentContext {
                        comment: Comment {
                            target: *target,
                            body: body.clone(),
                            root: root_task_id,
                        },
                        actor: record.context.actor.clone(),
                        state,
                    },
                );

                if let Target::Comment(parent) = target {
                    let parent_ctx = self
                        .comments
                        .get_mut(parent)
                        .expect("apply validated the parent above");

                    if let Some(next) = parent_ctx.state.transition(&record.event, &record) {
                        parent_ctx.state = next;
                    }
                }

                self.tasks
                    .get_mut(root_task_id.0)
                    .ok_or(Reason::InvalidTaskId)?
                    .thread
                    .push(CommentId(record.id));
            }
        }

        Ok(())
    }

    /// Stamps events as records onto a scratch copy: the fold is the
    /// validator, so nothing lands unless every record folds
    pub(crate) fn stage(
        &self,
        base: usize,
        context: &Context,
        time: u64,
        events: &[Event],
    ) -> Result<(World, Vec<Record>), Reject> {
        let records: Vec<Record> = events
            .iter()
            .enumerate()
            .map(|(i, event)| Record {
                id: RecordId(base + i),
                timestamp: time,
                context: context.clone(),
                event: event.clone(),
            })
            .collect();

        let folded = self
            .clone()
            .fold(records.clone())
            .map_err(|err| Reject::from(err.reason))?;
        Ok((folded, records))
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for Log {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Log {
    records: Vec<Record>,
    world: World,
}

impl Log {
    pub fn new() -> Self {
        Log {
            records: Vec::new(),
            world: World::new(),
        }
    }

    /// Machinery verbs enter through the same pipeline under the fixed
    /// system authorship: the tier comes from the role, never from input.
    pub fn execute_system(&mut self, command: Command, now: u64) -> Result<Vec<Record>, Reject> {
        self.execute(Context::system(), command, now)
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn execute(
        &mut self,
        context: Context,
        command: Command,
        now: u64,
    ) -> Result<Vec<Record>, Reject> {
        let events = decide(command);
        enforce_tier(&context, &events)?;
        let events = expand(&self.world, &events)?;

        let (folded, records) = self
            .world
            .stage(self.records.len(), &context, now, &events)?;
        self.records.extend(records.iter().cloned());
        self.world = folded;

        Ok(records)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::task::TaskId;
    use crate::types::prose::Prose;

    fn agent() -> Context {
        Context {
            actor: ActorName::new("saccade bot".into()).unwrap(),
            tier: Tier::Agent,
        }
    }

    #[test]
    fn command_translation_maps_every_reachable_reason() {
        assert!(matches!(
            Reject::from(Reason::InvalidTaskId),
            Reject::InvalidTaskId
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidProposalId),
            Reject::InvalidProposalId
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidCommentId),
            Reject::InvalidCommentId
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidParentTaskId),
            Reject::InvalidParentTaskId
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidStateTransition),
            Reject::InvalidStateTransition
        ));
        assert!(matches!(
            Reject::from(Reason::NotClaimHolder),
            Reject::NotClaimHolder
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidIncarnationId),
            Reject::InvalidIncarnationId
        ));
        assert!(matches!(
            Reject::from(Reason::IncarnationAlreadyActive),
            Reject::IncarnationAlreadyActive
        ));
        assert!(matches!(
            Reject::from(Reason::ProposalAlreadyOpen),
            Reject::ProposalAlreadyOpen
        ));
        assert!(matches!(
            Reject::from(Reason::DemandNotOnTask),
            Reject::DemandNotOnTask
        ));
        assert!(matches!(
            Reject::from(Reason::WorkspaceAlreadyExists),
            Reject::WorkspaceAlreadyExists
        ));
        assert!(matches!(
            Reject::from(Reason::WorkspaceMissing),
            Reject::WorkspaceMissing
        ));
        assert!(matches!(
            Reject::from(Reason::WorktreeAlreadyPresent),
            Reject::WorktreeAlreadyPresent
        ));
    }

    #[test]
    fn corrupted_task_id_fails_the_fold() {
        let record = Record {
            id: RecordId(0),
            timestamp: 1,
            context: agent(),
            event: Event::TaskClaimed { id: TaskId(0) }, // no task exists yet
        };
        let err = World::new().apply(record).unwrap_err();
        assert!(matches!(err.reason, Reason::InvalidTaskId));
    }

    #[test]
    fn dangling_parent_fails_the_fold() {
        let record = Record {
            id: RecordId(0),
            timestamp: 1,
            context: agent(),
            event: Event::TaskCreated {
                name: Prose::new("orphan task".into()).unwrap(),
                parent_id: Some(TaskId(3)),
            },
        };
        let err = World::new().apply(record).unwrap_err();
        assert!(matches!(err.reason, Reason::InvalidParentTaskId));
    }

    #[test]
    fn second_open_judgment_fails_the_fold() {
        let created = Record {
            id: RecordId(0),
            timestamp: 1,
            context: agent(),
            event: Event::TaskCreated {
                name: Prose::new("migrate floop".into()).unwrap(),
                parent_id: None,
            },
        };
        let first = Record {
            id: RecordId(1),
            timestamp: 1,
            context: agent(),
            event: Event::ProposalCreated {
                name: Prose::new("drop floop instead".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
        };
        let second = Record {
            id: RecordId(2),
            timestamp: 1,
            context: agent(),
            event: Event::ProposalCreated {
                name: Prose::new("drop floop again".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
        };
        let err = World::new()
            .apply(created)
            .unwrap()
            .apply(first)
            .unwrap()
            .apply(second)
            .unwrap_err();
        assert!(matches!(err.reason, Reason::ProposalAlreadyOpen));
    }

    #[test]
    fn invalid_task_transition_fails_the_fold() {
        let created = Record {
            id: RecordId(0),
            timestamp: 1,
            context: agent(),
            event: Event::TaskCreated {
                name: Prose::new("new task".into()).unwrap(),
                parent_id: None,
            },
        };
        let done = Record {
            id: RecordId(1),
            timestamp: 1,
            context: agent(),
            event: Event::TaskDone {
                id: TaskId(0),
                receipt: Prose::new("jumping straight to done".into()).unwrap(),
            },
        };
        let err = World::new()
            .apply(created)
            .unwrap()
            .apply(done)
            .unwrap_err();
        assert!(matches!(err.reason, Reason::InvalidStateTransition));
    }
}
