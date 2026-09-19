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

/// Steps the world by a command, returning the new world and the records
/// that were produced. Takes the world as given — the caller owns its
/// truth; `db::record` reconstructs it from the log under the write lock
/// before calling.
pub fn execute(
    world: &World,
    base: usize,
    context: &Context,
    command: Command,
    now: u64,
) -> Result<(World, Vec<Record>), Reject> {
    let events = decide(command);
    enforce_tier(context, &events)?;
    let events = expand(world, &events)?;
    world.stage(base, context, now, &events)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Event;
    use crate::objects::comment::{
        Addressee, AgentAttemptState, CommentId, CommentState, ResponseState, Target,
    };
    use crate::objects::incarnation::{IncarnationId, IncarnationState};
    use crate::objects::proposal::{ProposalAction, ProposalId, ProposalState};
    use crate::objects::task::{TaskId, TaskState};
    use crate::objects::workspace::WorktreeState;
    use crate::types::actor::ActorName;
    use crate::types::failure::{FailureCode, FailureEvidence};
    use crate::types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
    use crate::types::prose::Prose;

    /// Test fixture over `execute`: the in-memory records+world pair, for
    /// contract tests that run the law with no storage layer present.
    pub struct Log {
        records: Vec<Record>,
        world: World,
    }

    impl Default for Log {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Log {
        pub fn new() -> Self {
            Log {
                records: Vec::new(),
                world: World::new(),
            }
        }

        pub fn execute_system(
            &mut self,
            command: Command,
            now: u64,
        ) -> Result<Vec<Record>, Reject> {
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
            let (folded, records) =
                execute(&self.world, self.records.len(), &context, command, now)?;
            self.records.extend(records.iter().cloned());
            self.world = folded;

            Ok(records)
        }
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
    #[test]
    fn rejected_command_appends_nothing() {
        let mut log = Log::new();
        populate_log(&mut log);

        // t-3 is dropped: the fold refuses the claim and nothing lands
        let refused = log.execute(agent(), Command::ClaimTask { id: TaskId(3) }, 99);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        assert_eq!(log.records().len(), RECORD_COUNT);
    }

    #[test]
    fn replay_split_equals_whole() {
        let mut log = Log::new();
        populate_log(&mut log);

        for i in 0..log.records().len() {
            let (head, tail) = log.records().split_at(i);
            let mut staged = World::replay(head.to_vec()).unwrap();
            for record in tail {
                staged = staged
                    .apply(record.clone())
                    .expect("decide emitted an unfoldable event");
            }

            assert_eq!(staged, *log.world());
        }
    }

    #[test]
    fn done_and_release_are_holder_gated() {
        let mut log = Log::new();
        populate_log(&mut log);

        let other_agent = Context {
            actor: ActorName::new("other agent".into()).unwrap(),
            tier: Tier::Agent,
        };

        // t-3 is claimed by the agent: only the holder completes
        for (ctx, command) in [
            (
                other_agent.clone(),
                Command::CompleteTask {
                    id: TaskId(3),
                    receipt: Prose::new("not my claim".into()).unwrap(),
                },
            ),
            (
                human(),
                Command::CompleteTask {
                    id: TaskId(3),
                    receipt: Prose::new("not my claim".into()).unwrap(),
                },
            ),
            (
                other_agent.clone(),
                Command::ReleaseTask {
                    id: TaskId(3),
                    note: Prose::new("not my claim".into()).unwrap(),
                },
            ),
        ] {
            let refused = log.execute(ctx, command, 99);
            assert!(matches!(refused, Err(Reject::NotClaimHolder)));
        }
        assert_eq!(log.records().len(), RECORD_COUNT);

        // a human may release any claim
        log.execute(
            human(),
            Command::ReleaseTask {
                id: TaskId(3),
                note: Prose::new("reclaiming for the other agent".into()).unwrap(),
            },
            20,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);
        assert_eq!(log.world().tasks[3].holder, None);

        // an agent may release only its own claim
        log.execute(
            other_agent.clone(),
            Command::ClaimTask { id: TaskId(3) },
            21,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].holder, Some(other_agent.actor.clone()));
        log.execute(
            other_agent.clone(),
            Command::ReleaseTask {
                id: TaskId(3),
                note: Prose::new("handing back".into()).unwrap(),
            },
            22,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);
    }

    #[test]
    fn authority_supersedes_existence_in_rejections() {
        let mut log = Log::new();
        let drop = || Command::DropTask {
            id: TaskId(0),
            note: Prose::new("invalid drop".into()).unwrap(),
        };
        let accept = || Command::AcceptProposal {
            id: ProposalId(RecordId(99)),
        };
        assert!(matches!(
            log.execute(agent(), drop(), 1),
            Err(Reject::HumanOnly)
        ));
        assert!(matches!(
            log.execute(human(), drop(), 1),
            Err(Reject::InvalidTaskId)
        ));
        assert!(matches!(
            log.execute(agent(), accept(), 1),
            Err(Reject::HumanOnly)
        ));
        assert!(matches!(
            log.execute(human(), accept(), 1),
            Err(Reject::InvalidProposalId)
        ));
    }

    #[test]
    fn second_open_proposal_on_one_task_is_refused() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("migrate floop".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            human(),
            Command::CreateProposal {
                name: Prose::new("drop floop instead".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            2,
        )
        .unwrap();

        let second = Command::CreateProposal {
            name: Prose::new("drop floop again".into()).unwrap(),
            action: ProposalAction::Drop { task_id: TaskId(0) },
        };
        assert!(matches!(
            log.execute(human(), second, 3),
            Err(Reject::ProposalAlreadyOpen)
        ));

        log.execute(
            human(),
            Command::RejectProposal {
                id: ProposalId(RecordId(1)),
                note: Prose::new("floop stays".into()).unwrap(),
            },
            4,
        )
        .unwrap();
        let third = Command::CreateProposal {
            name: Prose::new("drop floop for real".into()).unwrap(),
            action: ProposalAction::Drop { task_id: TaskId(0) },
        };
        assert!(log.execute(human(), third, 5).is_ok());
    }

    #[test]
    fn accept_compound_write_two_records_in_order() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("I am going to do floop again".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        let proposed = log
            .execute(
                agent(),
                Command::CreateProposal {
                    name: Prose::new("this is a duplicated task".into()).unwrap(),
                    action: ProposalAction::Drop { task_id: TaskId(0) },
                },
                2,
            )
            .unwrap();

        // A proposal's identity is the record id of its creation act, so the proposal's id is 1
        assert_eq!(proposed[0].id.0, 1);
        assert!(log.world().proposals.contains_key(&ProposalId(RecordId(1))));

        let accepted = log
            .execute(
                human(),
                Command::AcceptProposal {
                    id: ProposalId(RecordId(1)),
                },
                3,
            )
            .unwrap();

        // accepting produces two records: the proposal acceptance and the act of dropping the task
        assert_eq!(accepted.len(), 2);
        assert!(matches!(
            &accepted[0].event,
            Event::ProposalAccepted { id } if id.0.0 == 1
        ));
        let Event::TaskDropped { id, note } = &accepted[1].event else {
            panic!("the embedded act must ride the accept");
        };
        assert_eq!(id.0, 0);
        // the proposal's name becomes the drop's note
        assert_eq!(note.as_str(), "this is a duplicated task");
        assert_eq!(log.world().tasks[0].task.state, TaskState::Dropped);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))]
                .proposal
                .state,
            ProposalState::Accepted
        );
    }

    #[test]
    fn agent_accept_writes_nothing() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("duplicate corpse".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: Prose::new("duplicate of the sibling".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            2,
        )
        .unwrap();

        let refused = log.execute(
            agent(),
            Command::AcceptProposal {
                id: ProposalId(RecordId(1)),
            },
            3,
        );
        assert!(matches!(refused, Err(Reject::HumanOnly)));
        assert_eq!(log.records().len(), 2);
        assert_eq!(log.world().tasks[0].task.state, TaskState::Open);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))]
                .proposal
                .state,
            ProposalState::Open
        );
    }

    #[test]
    fn open_proposals_are_inert_and_stale_accept_fails_atomically() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: Prose::new("real work".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CreateProposal {
                name: Prose::new("not real work".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(0) },
            },
            2,
        )
        .unwrap();

        // claiming the task makes the proposal stale but it still valid
        log.execute(agent(), Command::ClaimTask { id: TaskId(0) }, 3)
            .unwrap();

        // drop from claimed is illegal therefore the accept fails
        let refused = log.execute(
            human(),
            Command::AcceptProposal {
                id: ProposalId(RecordId(1)),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        assert_eq!(log.records().len(), 3);
        assert_eq!(log.world().tasks[0].task.state, TaskState::Claimed);
        assert_eq!(
            log.world().proposals[&ProposalId(RecordId(1))]
                .proposal
                .state,
            ProposalState::Open
        );
    }

    #[test]
    fn incarnation_lifecycle_runs_through_the_machinery_verbs() {
        let mut log = Log::new();
        populate_log(&mut log);
        let session = SessionPointer::new("/tmp/pi-session.jsonl".into()).unwrap();

        // the demand
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("run the suite".into()).unwrap(),
                addressee: Some(Addressee::Agent),
            },
            20,
        )
        .unwrap();
        let demand = CommentId(RecordId(14));
        let trigger = RecordId(14);

        // machinery verbs reject judgment tiers: the role is the only door
        let refused = log.execute(
            agent(),
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger,
                actor: ActorName::new("pi".into()).unwrap(),
                session: session.clone(),
            },
            21,
        );
        assert!(matches!(refused, Err(Reject::HumanOnly)));

        log.execute_system(
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger,
                actor: ActorName::new("pi".into()).unwrap(),
                session: session.clone(),
            },
            21,
        )
        .unwrap();
        let run = IncarnationId(RecordId(15));
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::Bound
        );
        assert_eq!(log.world().tasks[0].active_incarnation, Some(run));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::InFlight { incarnation: run }
            }
        );

        // a second live run on one task is refused
        let refused = log.execute_system(
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger,
                actor: ActorName::new("pi".into()).unwrap(),
                session: session.clone(),
            },
            22,
        );
        assert!(matches!(refused, Err(Reject::IncarnationAlreadyActive)));

        // settling before an accepted prompt is refused
        let refused = log.execute_system(Command::SettleIncarnation { id: run }, 23);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        log.execute_system(Command::AcceptPrompt { id: run }, 24)
            .unwrap();
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::PromptAccepted
        );

        // the session answers; the reply responds but the run holds the slot
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("55 green, nothing flaky".into()).unwrap(),
                addressee: None,
            },
            25,
        )
        .unwrap();
        let reply = CommentId(RecordId(17));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::InFlight { incarnation: run }
            }
        );

        // the run produced its reply, then settled
        log.execute_system(
            Command::MarkRecord {
                incarnation_id: run,
                record_id: RecordId(17),
            },
            26,
        )
        .unwrap();
        log.execute_system(Command::SettleIncarnation { id: run }, 27)
            .unwrap();
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::Settled
        );
        assert_eq!(log.world().tasks[0].active_incarnation, None);
        assert_eq!(log.world().incarnations[&run].produced, vec![RecordId(17)]);
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::Spent
            }
        );
    }

    #[test]
    fn rejected_prompt_terminalizes_and_spends() {
        let mut log = Log::new();
        populate_log(&mut log);
        let session = SessionPointer::new("/tmp/pi-session.jsonl".into()).unwrap();
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("run the flaky one".into()).unwrap(),
                addressee: Some(Addressee::Agent),
            },
            20,
        )
        .unwrap();
        let demand = CommentId(RecordId(14));
        log.execute_system(
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: demand,
                trigger: RecordId(14),
                actor: ActorName::new("pi".into()).unwrap(),
                session,
            },
            21,
        )
        .unwrap();
        log.execute_system(
            Command::RejectPrompt {
                id: IncarnationId(RecordId(15)),
                evidence: FailureEvidence::new(
                    FailureCode::PromptRejected,
                    Some("session refused the pointer prompt".into()),
                ),
            },
            22,
        )
        .unwrap();
        let run = IncarnationId(RecordId(15));
        assert_eq!(
            log.world().incarnations[&run].state,
            IncarnationState::Interrupted
        );
        assert_eq!(log.world().tasks[0].active_incarnation, None);
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Spent
            }
        );

        // a second prompt outcome never lands on the same run
        let refused = log.execute_system(Command::AcceptPrompt { id: run }, 23);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
    }

    #[test]
    fn agent_demands_fold_and_answer_by_exact_tier() {
        let mut log = Log::new();
        populate_log(&mut log);
        let before = log.records().len();

        // the demand is born authorized on its own birth record
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("what is the fold count?".into()).unwrap(),
                addressee: Some(Addressee::Agent),
            },
            20,
        )
        .unwrap();
        let demand = CommentId(RecordId(before));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Authorized {
                    trigger: RecordId(before)
                }
            }
        );

        // a wrong-tier reply lands but does not answer
        log.execute(
            human(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("asking the agent, not you".into()).unwrap(),
                addressee: None,
            },
            21,
        )
        .unwrap();
        assert!(matches!(
            &log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                ..
            }
        ));

        // a deeper descendant never satisfies the ancestor
        let mid = CommentId(RecordId(before + 1));
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(mid),
                body: Prose::new("still gathering".into()).unwrap(),
                addressee: None,
            },
            22,
        )
        .unwrap();
        assert!(matches!(
            &log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Awaiting,
                ..
            }
        ));

        // the first exact-tier direct reply answers and spends the attempt
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("fourteen, fixtures unchanged".into()).unwrap(),
                addressee: None,
            },
            23,
        )
        .unwrap();
        let reply = CommentId(RecordId(before + 3));
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::Spent,
            }
        );

        // a second exact-tier reply changes nothing
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Comment(demand),
                body: Prose::new("also fourteen".into()).unwrap(),
                addressee: None,
            },
            24,
        )
        .unwrap();
        assert_eq!(
            log.world().comments[&demand].state,
            CommentState::AddressedToAgent {
                response: ResponseState::Responded { reply },
                attempt: AgentAttemptState::Spent,
            }
        );
    }

    #[test]
    fn comments_have_no_state_gate() {
        let mut log = Log::new();
        for name in ["open work", "done work", "dropped work"] {
            log.execute(
                human(),
                Command::CreateTask {
                    name: Prose::new(name.into()).unwrap(),
                    parent_id: None,
                },
                1,
            )
            .unwrap();
        }
        log.execute(human(), Command::ClaimTask { id: TaskId(1) }, 2)
            .unwrap();
        log.execute(
            human(),
            Command::CompleteTask {
                id: TaskId(1),
                receipt: Prose::new("shipped".into()).unwrap(),
            },
            3,
        )
        .unwrap();
        log.execute(
            human(),
            Command::DropTask {
                id: TaskId(2),
                note: Prose::new("run dead".into()).unwrap(),
            },
            4,
        )
        .unwrap();

        let before = log.records().len();
        for (id, state) in [(0, "open"), (1, "done"), (2, "dropped")] {
            log.execute(
                agent(),
                Command::Comment {
                    target: Target::Task(TaskId(id)),
                    body: Prose::new(format!("for the record, on the {state} task")).unwrap(),
                    addressee: None,
                },
                9,
            )
            .unwrap_or_else(|e| panic!("comment on {state} task refused: {e:?}"));
        }
        assert_eq!(log.records().len(), before + 3);

        // only an unknown comment id refuses, writing nothing
        let refused = log.execute(
            human(),
            Command::Comment {
                target: Target::Comment(CommentId(RecordId(99))),
                body: Prose::new("addresses nothing".into()).unwrap(),
                addressee: None,
            },
            10,
        );
        assert!(matches!(refused, Err(Reject::InvalidCommentId)));
        assert_eq!(log.records().len(), before + 3);
    }

    #[test]
    fn workspace_records_run_through_the_machinery_verbs() {
        let mut log = Log::new();
        log.execute(
            Context {
                actor: ActorName::new("human person".into()).unwrap(),
                tier: Tier::Human,
            },
            Command::CreateTask {
                name: Prose::new("run managed demand in a worktree".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();

        // a task with no workspace refuses worktree and checkpoint records
        let base = GitCommit::new("abc123".into()).unwrap();
        let branch = GitBranch::new("saccade/t-0".into()).unwrap();
        let worktree = WorktreePath::new("/repo/wt/t-0".into()).unwrap();
        let refused = log.execute_system(
            Command::CreateWorktree {
                task_id: TaskId(0),
                worktree: worktree.clone(),
            },
            2,
        );
        assert!(matches!(refused, Err(Reject::WorkspaceMissing)));

        // provisioning births the lineage: checkpoint starts at the base
        log.execute_system(
            Command::CreateWorkspace {
                task_id: TaskId(0),
                base: base.clone(),
                branch: branch.clone(),
            },
            2,
        )
        .unwrap();
        let ctx = &log.world().tasks[0];
        assert_eq!(ctx.workspace.as_ref().unwrap().checkpoint, base);
        assert!(matches!(
            ctx.workspace.as_ref().unwrap().worktree,
            WorktreeState::Absent
        ));

        // one lineage per task
        let refused = log.execute_system(
            Command::CreateWorkspace {
                task_id: TaskId(0),
                base: base.clone(),
                branch: branch.clone(),
            },
            3,
        );
        assert!(matches!(refused, Err(Reject::WorkspaceAlreadyExists)));

        // the physical creation is recorded, once
        log.execute_system(
            Command::CreateWorktree {
                task_id: TaskId(0),
                worktree: worktree.clone(),
            },
            4,
        )
        .unwrap();
        let refused = log.execute_system(
            Command::CreateWorktree {
                task_id: TaskId(0),
                worktree: worktree.clone(),
            },
            5,
        );
        assert!(matches!(refused, Err(Reject::WorktreeAlreadyPresent)));
        assert!(matches!(
            log.world().tasks[0].workspace.as_ref().unwrap().worktree,
            WorktreeState::Present(_)
        ));

        // a checkpoint advances; it may advance again
        let head = GitCommit::new("def456".into()).unwrap();
        log.execute_system(
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: head.clone(),
            },
            6,
        )
        .unwrap();
        assert_eq!(
            log.world().tasks[0].workspace.as_ref().unwrap().checkpoint,
            head
        );
    }

    const RECORD_COUNT: usize = 14;

    fn populate_log(log: &mut Log) {
        let agent_ctx = agent();
        let human_ctx = human();

        log.execute(
            agent_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("implement foo".into()).unwrap(),
                parent_id: None,
            },
            1,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(0) }, 2)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("fix bar".into()).unwrap(),
                parent_id: None,
            },
            3,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(1) }, 4)
            .unwrap();

        log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(1),
                receipt: Prose::new("bar fixed".into()).unwrap(),
            },
            5,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("improve baz".into()).unwrap(),
                parent_id: Some(TaskId(0)),
            },
            6,
        )
        .unwrap();

        log.execute(
            agent_ctx.clone(),
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Prose::new("foo completed successfully".into()).unwrap(),
            },
            7,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("migrate floop".into()).unwrap(),
                parent_id: None,
            },
            8,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(3) }, 9)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::ReleaseTask {
                id: TaskId(3),
                note: Prose::new("run dead, reclaim".into()).unwrap(),
            },
            10,
        )
        .unwrap();

        log.execute(agent_ctx.clone(), Command::ClaimTask { id: TaskId(3) }, 11)
            .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::DropTask {
                id: TaskId(2),
                note: Prose::new("scope covered by fix bar".into()).unwrap(),
            },
            12,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::DropTask {
                id: TaskId(1),
                note: Prose::new("covered by fix bar; kept only as context".into()).unwrap(),
            },
            13,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: Prose::new("open work".into()).unwrap(),
                parent_id: None,
            },
            14,
        )
        .unwrap();
    }
}
