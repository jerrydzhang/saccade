use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Reject;
use crate::decide::{decide, enforce_tier};
use crate::events::{Command, Event};
use crate::objects::comment::{
    Comment, CommentContext, CommentId, CommentKind, CommentState, Revision, Target, Withdrawal,
};
use crate::objects::incarnation::{Incarnation, IncarnationContext, IncarnationId};
use crate::objects::task::{Owner, OwnerEntry, Task, TaskContext, TaskId, TaskState};
use crate::objects::workspace::{Workspace, WorkspaceContext};
use crate::types::actor::ActorName;
use crate::types::pointers::WorktreePath;

#[derive(Debug)]
pub struct FoldError {
    pub at: RecordId,
    pub reason: Reason,
}

#[derive(Debug)]
pub enum Reason {
    InvalidTaskId,
    InvalidCommentId,
    /// A decision or reliance names a comment that is not an offer.
    NotAnOffer,
    /// A decision or reliance names an offer that is not its task's
    /// standing Offered or Concluded account.
    NotCurrentOffer,
    /// The reliance names its own task's conclusion: a task cannot
    /// consume itself.
    SelfReliance,
    /// The actor's declared task scope covers the offered task: a
    /// task-bound producer cannot accept its own conclusion.
    SelfAcceptance,
    /// A claim names an incarnation the record never birthed, or one
    /// bound to another task.
    InvalidIncarnationId,
    InvalidStateTransition,
    /// The task already holds a workspace: one lineage per task.
    WorkspaceAlreadyExists,
    /// The event names a task whose workspace was never established.
    WorkspaceMissing,
    /// An agent- or human-tier checkpoint names a head the workspace
    /// already left; the explicit door never rewinds.
    CheckpointRewind,
}

/// Command-path translation of fold failures into refusals; total over Reason
/// since every fold failure is command-reachable.
impl From<Reason> for Reject {
    fn from(reason: Reason) -> Reject {
        match reason {
            Reason::InvalidTaskId => Reject::InvalidTaskId,
            Reason::InvalidCommentId => Reject::InvalidCommentId,
            Reason::NotAnOffer => Reject::NotAnOffer,
            Reason::NotCurrentOffer => Reject::NotCurrentOffer,
            Reason::SelfReliance => Reject::SelfReliance,
            Reason::SelfAcceptance => Reject::SelfAcceptance,
            Reason::InvalidIncarnationId => Reject::InvalidIncarnationId,
            Reason::InvalidStateTransition => Reject::InvalidStateTransition,
            Reason::WorkspaceAlreadyExists => Reject::WorkspaceAlreadyExists,
            Reason::WorkspaceMissing => Reject::WorkspaceMissing,
            Reason::CheckpointRewind => Reject::CheckpointRewind,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Human,
    Agent,
    System,
}

/// An environment-declared binding: what the acting session's
/// environment claims it is scoped to. Disclosure riding the record,
/// never proof — the one gate it feeds is the self-acceptance refusal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scope {
    pub task: TaskId,
    pub incarnation: Option<IncarnationId>,
    pub workspace: Option<WorktreePath>,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub actor: ActorName,
    pub tier: Tier,
    pub scope: Option<Scope>,
}

impl Context {
    /// The fixed author machinery records under; no caller presents this.
    pub fn system() -> Self {
        Context {
            actor: ActorName::new("saccade".into()).expect("the fixed system actor name is valid"),
            tier: Tier::System,
            scope: None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    pub comments: BTreeMap<CommentId, CommentContext>,
    /// Record-derived worker identities, born one-to-one with the
    /// workspace establishments
    pub incarnations: BTreeMap<IncarnationId, IncarnationContext>,
}

impl World {
    pub fn new() -> Self {
        World {
            tasks: Vec::new(),
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

    /// The task whose birth record landed at this seq: a create reply's
    /// reference, resolved against the post-write fold.
    pub fn task_born_at(&self, birth: RecordId) -> Option<TaskId> {
        self.tasks.iter().position(|t| t.birth == birth).map(TaskId)
    }

    /// Only valid path for world mutation
    pub fn apply(mut self, record: Record) -> Result<Self, FoldError> {
        let at = record.id;
        self.apply_record(record)
            .map_err(|reason| FoldError { at, reason })?;
        Ok(self)
    }

    /// The owner a claim names — exactly the ratified kinds. The
    /// incarnation reference, never the caller's tier, names a
    /// task-bound worker: a claim that rides one holds as the
    /// incarnation whoever requested it; a claim without the
    /// reference holds as the caller.
    fn owner_of(context: &Context, incarnation: Option<IncarnationId>) -> Result<Owner, Reason> {
        if let Some(id) = incarnation {
            return Ok(Owner::Incarnation(id));
        }
        match context.tier {
            Tier::Human => Ok(Owner::Human(context.actor.clone())),
            Tier::Agent => Ok(Owner::External(context.actor.clone())),
            // machinery holds no whole-task responsibility
            Tier::System => Err(Reason::InvalidStateTransition),
        }
    }

    fn apply_record(&mut self, record: Record) -> Result<(), Reason> {
        match record.event {
            // Task events
            Event::TaskCreated { name: task_name } => {
                self.tasks.push(TaskContext {
                    task: Task {
                        state: TaskState::Open,
                        name: task_name,
                    },
                    birth: record.id,
                    last_updated: record.id,
                    last_record_at: record.timestamp,
                    offered_at: None,
                    ownership: Vec::new(),
                    current_offer: None,
                    thread: Vec::new(),
                    artifacts: Vec::new(),
                    birth_actor: record.context.actor.clone(),
                    workspace: None,
                });
            }
            Event::TaskClaimed { id, incarnation } => {
                let owner = Self::owner_of(&record.context, incarnation)?;
                if let Some(inc) = incarnation {
                    let born = self
                        .incarnations
                        .get(&inc)
                        .ok_or(Reason::InvalidIncarnationId)?;
                    if born.incarnation.task != id {
                        return Err(Reason::InvalidIncarnationId);
                    }
                }
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.ownership.push(OwnerEntry {
                    owner,
                    record: record.id,
                    at: record.timestamp,
                });
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::TaskCancelled { id, .. } | Event::TaskReopened { id, .. } => {
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.task.state = task_ctx
                    .task
                    .state
                    .transition(&record.event)
                    .ok_or(Reason::InvalidStateTransition)?;
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
                if matches!(record.event, Event::TaskReopened { .. }) {
                    task_ctx.current_offer = None;
                    task_ctx.offered_at = None;
                }
            }
            // Comment events
            Event::Commented {
                ref target,
                ref kind,
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

                match kind {
                    CommentKind::Note { .. } | CommentKind::Request { .. } => {}
                    // the broadcast addresses its whole task
                    CommentKind::Offer { .. } => {
                        if !matches!(target, Target::Task(_)) {
                            return Err(Reason::InvalidStateTransition);
                        }
                    }
                    CommentKind::Accept { offer } | CommentKind::Reject { offer, .. } => {
                        // a decision addresses the offer it names
                        if *target != Target::Comment(*offer) {
                            return Err(Reason::InvalidStateTransition);
                        }
                        let offer_ctx = self.comments.get(offer).ok_or(Reason::InvalidCommentId)?;
                        if !matches!(offer_ctx.comment.kind, CommentKind::Offer { .. }) {
                            return Err(Reason::NotAnOffer);
                        }
                        if matches!(kind, CommentKind::Accept { .. })
                            && record.context.tier == Tier::Agent
                            && record
                                .context
                                .scope
                                .as_ref()
                                .is_some_and(|scope| scope.task == offer_ctx.comment.root)
                        {
                            return Err(Reason::SelfAcceptance);
                        }
                        let offered = self
                            .tasks
                            .get(offer_ctx.comment.root.0)
                            .ok_or(Reason::InvalidTaskId)?;
                        if offered.current_offer != Some(*offer) {
                            return Err(Reason::NotCurrentOffer);
                        }
                    }
                    CommentKind::Use { offer, .. } => {
                        let offer_ctx = self.comments.get(offer).ok_or(Reason::InvalidCommentId)?;
                        if !matches!(offer_ctx.comment.kind, CommentKind::Offer { .. }) {
                            return Err(Reason::NotAnOffer);
                        }
                        let source = offer_ctx.comment.root;
                        if source == root_task_id {
                            return Err(Reason::SelfReliance);
                        }
                        let src_ctx = self.tasks.get(source.0).ok_or(Reason::InvalidTaskId)?;
                        if src_ctx.current_offer != Some(*offer)
                            || !matches!(
                                src_ctx.task.state,
                                TaskState::Offered | TaskState::Concluded
                            )
                        {
                            return Err(Reason::NotCurrentOffer);
                        }
                    }
                }

                self.comments.insert(
                    CommentId(record.id),
                    CommentContext {
                        comment: Comment {
                            target: *target,
                            kind: kind.clone(),
                            root: root_task_id,
                            state: CommentState::of(kind),
                        },
                        actor: record.context.actor.clone(),
                        tier: record.context.tier,
                        scope: record.context.scope.clone(),
                        born_at: record.timestamp,
                        revised: None,
                        withdrawn: None,
                    },
                );

                let task_ctx = self
                    .tasks
                    .get_mut(root_task_id.0)
                    .ok_or(Reason::InvalidTaskId)?;
                if let Some(next) = task_ctx.task.state.transition(&record.event) {
                    task_ctx.task.state = next;
                    task_ctx.last_updated = record.id;
                } else if matches!(
                    kind,
                    CommentKind::Offer { .. }
                        | CommentKind::Accept { .. }
                        | CommentKind::Reject { .. }
                ) {
                    // the lifecycle kinds must move the task: a second
                    // offer needs the reject door, a decided task no door
                    return Err(Reason::InvalidStateTransition);
                }
                match kind {
                    CommentKind::Offer { .. } => {
                        task_ctx.current_offer = Some(CommentId(record.id));
                        task_ctx.offered_at = Some(record.timestamp);
                    }
                    CommentKind::Reject { .. } => {
                        task_ctx.current_offer = None;
                        task_ctx.offered_at = None;
                    }
                    _ => {}
                }
                task_ctx.thread.push(CommentId(record.id));
                task_ctx.last_record_at = record.timestamp;
            }
            Event::CommentRevised { id, ref body } => {
                let comment_ctx = self.comments.get_mut(&id).ok_or(Reason::InvalidCommentId)?;
                // the gate is the state: a withdrawn deposit admits no repair
                if matches!(comment_ctx.comment.state, CommentState::Withdrawn(_)) {
                    return Err(Reason::InvalidStateTransition);
                }
                // the bodyless Accept carries no prose to repair
                match &mut comment_ctx.comment.kind {
                    CommentKind::Note { body: prose }
                    | CommentKind::Request { body: prose }
                    | CommentKind::Offer { body: prose }
                    | CommentKind::Reject { grounds: prose, .. }
                    | CommentKind::Use { reason: prose, .. } => *prose = body.clone(),
                    CommentKind::Accept { .. } => return Err(Reason::InvalidStateTransition),
                }
                comment_ctx.revised = Some(Revision {
                    record: record.id,
                    reviser: record.context.actor.clone(),
                });
                let root = comment_ctx.comment.root;
                let task_ctx = self.tasks.get_mut(root.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_record_at = record.timestamp;
            }
            ref event @ Event::CommentWithdrawn { id, .. } => {
                let comment_ctx = self.comments.get_mut(&id).ok_or(Reason::InvalidCommentId)?;
                comment_ctx.comment.state = comment_ctx
                    .comment
                    .state
                    .transition(event)
                    .ok_or(Reason::InvalidStateTransition)?;
                comment_ctx.withdrawn = Some(Withdrawal {
                    record: record.id,
                    withdrawer: record.context.actor.clone(),
                });
                let root = comment_ctx.comment.root;
                let task_ctx = self.tasks.get_mut(root.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::ArtifactAdded { root, ref artifact } => {
                let task_ctx = self.tasks.get_mut(root.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.artifacts.push((record.id, artifact.clone()));
                task_ctx.last_record_at = record.timestamp;
            }
            // Workspace events
            Event::WorkspaceEstablished {
                task_id,
                ref base,
                ref branch,
                ref worktree,
            } => {
                let task_ctx = self.tasks.get_mut(task_id.0).ok_or(Reason::InvalidTaskId)?;
                if task_ctx.workspace.is_some() {
                    return Err(Reason::WorkspaceAlreadyExists);
                }
                task_ctx.workspace = Some(WorkspaceContext {
                    workspace: Workspace {
                        base: base.clone(),
                        branch: branch.clone(),
                        checkpoint: base.clone(),
                        worktree: worktree.clone(),
                    },
                    birth: record.id,
                    heads: vec![base.clone()],
                });
                // the same birth is the task-bound worker's identity:
                // one record position, both names
                self.incarnations.insert(
                    IncarnationId(record.id),
                    IncarnationContext {
                        incarnation: Incarnation {
                            task: task_id,
                            workspace: worktree.clone(),
                        },
                        born_at: record.timestamp,
                    },
                );
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
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
                // the machinery (boot recovery) rebases the checkpoint;
                // the explicit door never returns to a head the record left
                if record.context.tier != Tier::System
                    && *checkpoint != workspace.workspace.checkpoint
                    && workspace.heads.contains(checkpoint)
                {
                    return Err(Reason::CheckpointRewind);
                }
                if *checkpoint != workspace.workspace.checkpoint {
                    workspace.heads.push(checkpoint.clone());
                    workspace.workspace.checkpoint = checkpoint.clone();
                }
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
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
    world.stage(base, context, now, &events)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Event;
    use crate::objects::comment::{CommentId, CommentKind, CommentState, Target};
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::task::{Owner, TaskId, TaskState};
    use crate::types::actor::ActorName;
    use crate::types::artifact::Artifact;
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};
    use crate::types::prose::Prose;
    use crate::{ContentHash, Scope};

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

    /// An agent whose environment declares the given task binding.
    fn scoped_agent(task: usize) -> Context {
        Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope: Some(Scope {
                task: TaskId(task),
                incarnation: None,
                workspace: None,
            }),
        }
    }

    fn prose(text: &str) -> Prose {
        Prose::new(text.into()).unwrap()
    }

    fn offer(id: TaskId, body: &str) -> Command {
        Command::Comment {
            target: Target::Task(id),
            kind: CommentKind::Offer { body: prose(body) },
        }
    }

    fn accept(offer: usize) -> Command {
        Command::Comment {
            target: Target::Comment(CommentId(RecordId(offer))),
            kind: CommentKind::Accept {
                offer: CommentId(RecordId(offer)),
            },
        }
    }

    fn reject(offer: usize, grounds: &str) -> Command {
        Command::Comment {
            target: Target::Comment(CommentId(RecordId(offer))),
            kind: CommentKind::Reject {
                offer: CommentId(RecordId(offer)),
                grounds: prose(grounds),
            },
        }
    }

    fn note(target: Target, body: &str) -> Command {
        Command::Comment {
            target,
            kind: CommentKind::Note { body: prose(body) },
        }
    }

    const RECORD_COUNT: usize = 18;

    /// foo accepted; bar re-offered; floop reopened; a consumer uses foo.
    fn populate_log(log: &mut Log) {
        let human_ctx = human();
        let agent_ctx = agent();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: prose("implement foo"),
            },
            1,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: None,
            },
            2,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            note(Target::Task(TaskId(0)), "mid-flight finding parked"),
            2,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            offer(TaskId(0), "suite green, 55 tests"),
            3,
        )
        .unwrap();
        log.execute(human_ctx.clone(), accept(3), 4).unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: prose("fix bar"),
            },
            5,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            6,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            offer(TaskId(1), "bar fixed the quick way"),
            7,
        )
        .unwrap();
        log.execute(human_ctx.clone(), reject(7, "the edge case still fails"), 8)
            .unwrap();
        log.execute(
            agent_ctx.clone(),
            offer(TaskId(1), "edge case held, suite green"),
            9,
        )
        .unwrap();
        log.execute(human_ctx.clone(), accept(9), 10).unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: prose("migrate floop"),
            },
            11,
        )
        .unwrap();
        log.execute(
            human_ctx.clone(),
            Command::CancelTask {
                id: TaskId(2),
                explanation: prose("the migration is no longer wanted"),
            },
            12,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::Comment {
                target: Target::Task(TaskId(2)),
                kind: CommentKind::Request {
                    body: prose("please resume the migration"),
                },
            },
            13,
        )
        .unwrap();
        log.execute(
            human_ctx.clone(),
            Command::ReopenTask {
                id: TaskId(2),
                explanation: prose("the blocker cleared"),
            },
            14,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: prose("adopt foo's account"),
            },
            15,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::ClaimTask {
                id: TaskId(3),
                incarnation: None,
            },
            16,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::Comment {
                target: Target::Task(TaskId(3)),
                kind: CommentKind::Use {
                    offer: CommentId(RecordId(3)),
                    reason: prose("builds on foo's landed account"),
                },
            },
            17,
        )
        .unwrap();
    }

    #[test]
    fn authority_supersedes_existence_in_rejections() {
        let mut log = Log::new();
        let cancel = || Command::CancelTask {
            id: TaskId(0),
            explanation: prose("stop nothing"),
        };
        let provision = || Command::EstablishWorkspace {
            task_id: TaskId(0),
            base: GitCommit::new("abc123".into()).unwrap(),
            branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
        };
        // the tier gate fires before the fold ever reads existence
        assert!(matches!(
            log.execute(agent(), cancel(), 1),
            Err(Reject::HumanOnly)
        ));
        assert!(matches!(
            log.execute(human(), cancel(), 1),
            Err(Reject::InvalidTaskId)
        ));
        assert!(matches!(
            log.execute_system(provision(), 1),
            Err(Reject::InvalidTaskId)
        ));
        assert!(matches!(
            log.execute(human(), provision(), 1),
            Err(Reject::HumanOnly)
        ));
        assert!(matches!(
            log.execute_system(provision(), 1),
            Err(Reject::InvalidTaskId)
        ));
    }

    #[test]
    fn command_translation_maps_every_reachable_reason() {
        assert!(matches!(
            Reject::from(Reason::InvalidTaskId),
            Reject::InvalidTaskId
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidCommentId),
            Reject::InvalidCommentId
        ));
        assert!(matches!(
            Reject::from(Reason::NotAnOffer),
            Reject::NotAnOffer
        ));
        assert!(matches!(
            Reject::from(Reason::NotCurrentOffer),
            Reject::NotCurrentOffer
        ));
        assert!(matches!(
            Reject::from(Reason::SelfReliance),
            Reject::SelfReliance
        ));
        assert!(matches!(
            Reject::from(Reason::SelfAcceptance),
            Reject::SelfAcceptance
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidIncarnationId),
            Reject::InvalidIncarnationId
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidStateTransition),
            Reject::InvalidStateTransition
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
            Reject::from(Reason::CheckpointRewind),
            Reject::CheckpointRewind
        ));
    }

    #[test]
    fn corrupted_task_id_fails_the_fold() {
        let record = Record {
            id: RecordId(0),
            timestamp: 1,
            context: agent(),
            event: Event::TaskClaimed {
                id: TaskId(0),
                incarnation: None,
            },
        };
        let err = World::new().apply(record).unwrap_err();
        assert!(matches!(err.reason, Reason::InvalidTaskId));
    }

    #[test]
    fn an_offer_without_a_task_transition_fails_the_fold() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("implement foo"),
            },
            1,
        )
        .unwrap();
        log.execute(agent(), offer(TaskId(0), "the first account"), 2)
            .unwrap();
        // offered tasks admit no second offer: reject, then offer anew
        let refused = log.execute(agent(), offer(TaskId(0), "a replacement account"), 3);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        assert_eq!(log.records().len(), 2);
    }

    #[test]
    fn rejected_command_appends_nothing() {
        let mut log = Log::new();
        populate_log(&mut log);

        // t-9 names no task: the fold refuses and nothing lands
        let refused = log.execute(
            human(),
            Command::ClaimTask {
                id: TaskId(9),
                incarnation: None,
            },
            99,
        );
        assert!(matches!(refused, Err(Reject::InvalidTaskId)));
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

    /// The story folds the ratified lifecycle.
    #[test]
    fn the_story_folds_the_ratified_lifecycle() {
        let mut log = Log::new();
        populate_log(&mut log);
        let world = log.world();

        assert_eq!(world.tasks[0].task.state, TaskState::Concluded);
        assert_eq!(world.tasks[0].current_offer, Some(CommentId(RecordId(3))));
        assert_eq!(world.tasks[1].task.state, TaskState::Concluded);
        assert_eq!(world.tasks[1].current_offer, Some(CommentId(RecordId(9))));
        assert_eq!(world.tasks[2].task.state, TaskState::Open);
        assert_eq!(world.tasks[3].task.state, TaskState::Open);
        // the reopen cleared the standing pointer
        assert_eq!(world.tasks[2].current_offer, None);

        // the create-reply query: births resolve, non-births don't
        assert_eq!(world.task_born_at(RecordId(0)), Some(TaskId(0)));
        assert_eq!(world.task_born_at(RecordId(2)), None);
    }

    #[test]
    fn stale_decisions_are_refused_not_absorbed() {
        let mut log = Log::new();
        populate_log(&mut log);

        // t-1's first offer was rejected: neither decision reaches it
        for command in [accept(7), reject(7, "changed my mind again")] {
            let refused = log.execute(human(), command, 99);
            assert!(matches!(refused, Err(Reject::NotCurrentOffer)));
        }
        // a note is not an offer; an offer on another thread is not
        // this thread's current one
        let refused = log.execute(human(), accept(2), 99);
        assert!(matches!(refused, Err(Reject::NotAnOffer)));
        assert_eq!(log.records().len(), RECORD_COUNT);

        // t-0 concluded on c-3: its own current offer no longer moves
        for command in [accept(3), reject(3, "a fault found later")] {
            let refused = log.execute(human(), command, 99);
            assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        }
        assert_eq!(log.records().len(), RECORD_COUNT);
    }

    #[test]
    fn declared_scope_refuses_only_the_self_acceptance() {
        let mut log = Log::new();
        populate_log(&mut log);

        // the scoped agent accepts its own task's offer: refused
        let refused = log.execute(scoped_agent(0), accept(3), 99);
        assert!(matches!(refused, Err(Reject::SelfAcceptance)));

        // the same agent rejects on grounds: any participant may
        log.execute(agent(), offer(TaskId(3), "the consumer's account"), 18)
            .unwrap();
        log.execute(
            scoped_agent(0),
            reject(18, "the reason no longer holds"),
            19,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);

        // a scope on another task accepts freely; humans ignore scope
        log.execute(agent(), offer(TaskId(3), "the second account"), 20)
            .unwrap();
        log.execute(scoped_agent(2), accept(20), 21).unwrap();
        let mut scoped_human = scoped_agent(3);
        scoped_human.tier = Tier::Human;
        log.execute(agent(), offer(TaskId(2), "the floop account"), 22)
            .unwrap();
        log.execute(scoped_human, accept(22), 23).unwrap();
        assert_eq!(log.world().tasks[2].task.state, TaskState::Concluded);
        assert_eq!(log.world().tasks[3].task.state, TaskState::Concluded);
    }

    #[test]
    fn claim_transfers_ownership_without_touching_state() {
        let mut log = Log::new();
        populate_log(&mut log);

        assert_eq!(
            log.world().tasks[0].ownership,
            vec![OwnerEntry {
                owner: Owner::External(agent().actor),
                record: RecordId(1),
                at: 2,
            }]
        );

        // a claim on the cancelled task transfers at any state
        let before = log.world().tasks[2].task.state.clone();
        log.execute(
            human(),
            Command::ClaimTask {
                id: TaskId(2),
                incarnation: None,
            },
            99,
        )
        .unwrap();
        assert_eq!(log.world().tasks[2].task.state, before);
        assert_eq!(log.world().tasks[2].ownership.len(), 1);
        assert!(matches!(
            log.world().tasks[2].ownership[0].owner,
            Owner::Human(_)
        ));

        // a second claim transfers and preserves history
        log.execute(
            agent(),
            Command::ClaimTask {
                id: TaskId(2),
                incarnation: None,
            },
            100,
        )
        .unwrap();
        assert_eq!(log.world().tasks[2].ownership.len(), 2);
        assert!(matches!(
            log.world().tasks[2].ownership[1].owner,
            Owner::External(_)
        ));

        // an incarnation reference the record never birthed refuses
        let refused = log.execute(
            agent(),
            Command::ClaimTask {
                id: TaskId(2),
                incarnation: Some(crate::objects::incarnation::IncarnationId(RecordId(9))),
            },
            101,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
    }

    #[test]
    fn cancellation_and_reopening_are_human_authored_prose_acts() {
        let mut log = Log::new();
        populate_log(&mut log);

        // agents refuse both doors
        for command in [
            Command::CancelTask {
                id: TaskId(3),
                explanation: prose("agent cannot stop"),
            },
            Command::ReopenTask {
                id: TaskId(2),
                explanation: prose("agent cannot resume"),
            },
        ] {
            let refused = log.execute(agent(), command, 99);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }

        // the human cancels open t-3 with the explanation riding state
        log.execute(
            human(),
            Command::CancelTask {
                id: TaskId(3),
                explanation: prose("the consumer need dissolved"),
            },
            99,
        )
        .unwrap();
        assert_eq!(
            log.world().tasks[3].task.state,
            TaskState::Cancelled(prose("the consumer need dissolved"))
        );

        // a request never reopens a cancelled task
        log.execute(
            agent(),
            Command::Comment {
                target: Target::Task(TaskId(3)),
                kind: CommentKind::Request {
                    body: prose("please resume the adoption"),
                },
            },
            100,
        )
        .unwrap();
        assert!(matches!(
            log.world().tasks[3].task.state,
            TaskState::Cancelled(_)
        ));

        // reopening restores open with the explanation recorded
        log.execute(
            human(),
            Command::ReopenTask {
                id: TaskId(3),
                explanation: prose("the need returned"),
            },
            101,
        )
        .unwrap();
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);

        // concluded tasks admit neither door: faults go to new tasks
        for command in [
            Command::CancelTask {
                id: TaskId(0),
                explanation: prose("too late to stop"),
            },
            Command::ReopenTask {
                id: TaskId(0),
                explanation: prose("too late to resume"),
            },
        ] {
            let refused = log.execute(human(), command, 102);
            assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        }
        let refused = log.execute(
            human(),
            Command::ReopenTask {
                id: TaskId(3),
                explanation: prose("only cancelled tasks reopen"),
            },
            103,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
    }

    #[test]
    fn use_guards_its_source_reference() {
        let mut log = Log::new();
        populate_log(&mut log);

        // the story's use names foo's accepted account from t-3's thread
        let used = &log.world().comments[&CommentId(RecordId(17))];
        assert_eq!(used.comment.root, TaskId(3));
        assert!(matches!(used.comment.kind, CommentKind::Use { .. }));
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);

        // a non-offer refuses
        let refused = log.execute(
            agent(),
            Command::Comment {
                target: Target::Task(TaskId(3)),
                kind: CommentKind::Use {
                    offer: CommentId(RecordId(2)),
                    reason: prose("that is a note"),
                },
            },
            99,
        );
        assert!(matches!(refused, Err(Reject::NotAnOffer)));

        // a rejected account is not a standing one
        let refused = log.execute(
            agent(),
            Command::Comment {
                target: Target::Task(TaskId(3)),
                kind: CommentKind::Use {
                    offer: CommentId(RecordId(7)),
                    reason: prose("bar's rejected account"),
                },
            },
            100,
        );
        assert!(matches!(refused, Err(Reject::NotCurrentOffer)));

        // t-3 stands offered on c-18: a use from its own thread refuses
        log.execute(agent(), offer(TaskId(3), "the consumer's account"), 18)
            .unwrap();
        let refused = log.execute(
            agent(),
            Command::Comment {
                target: Target::Task(TaskId(3)),
                kind: CommentKind::Use {
                    offer: CommentId(RecordId(18)),
                    reason: prose("relying on myself"),
                },
            },
            101,
        );
        assert!(matches!(refused, Err(Reject::SelfReliance)));

        // a standing Offered account on another task lands
        log.execute(
            human(),
            Command::Comment {
                target: Target::Task(TaskId(2)),
                kind: CommentKind::Use {
                    offer: CommentId(RecordId(18)),
                    reason: prose("floop builds on the consumer's account"),
                },
            },
            102,
        )
        .unwrap();
        assert_eq!(log.records().len(), RECORD_COUNT + 2);
    }

    #[test]
    fn revision_repairs_prose_and_never_references() {
        let mut log = Log::new();
        populate_log(&mut log);

        // the offer's account is revised: the fold swaps the prose
        log.execute(
            agent(),
            Command::ReviseComment {
                id: CommentId(RecordId(9)),
                body: prose("edge case held, suite green — 56 tests"),
            },
            99,
        )
        .unwrap();
        let ctx = &log.world().comments[&CommentId(RecordId(9))];
        assert!(matches!(
            &ctx.comment.kind,
            CommentKind::Offer { body } if body.as_str().contains("56 tests")
        ));
        assert_eq!(
            ctx.revised,
            Some(Revision {
                record: RecordId(18),
                reviser: agent().actor,
            })
        );

        // the bodyless Accept carries no prose to repair
        let refused = log.execute(
            human(),
            Command::ReviseComment {
                id: CommentId(RecordId(10)),
                body: prose("no prose to swap"),
            },
            100,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        // the addressed position must be a comment's birth
        let refused = log.execute(
            human(),
            Command::ReviseComment {
                id: CommentId(RecordId(99)),
                body: prose("addresses nothing"),
            },
            101,
        );
        assert!(matches!(refused, Err(Reject::InvalidCommentId)));
        assert_eq!(log.records().len(), RECORD_COUNT + 1);
    }

    #[test]
    fn withdrawal_retires_a_note_at_any_tier_and_keeps_the_withdrawer() {
        let mut log = Log::new();
        populate_log(&mut log);
        let note_id = CommentId(RecordId(2));

        // a human withdraws the agent's note: the tombstone names the withdrawer
        log.execute(
            human(),
            Command::WithdrawComment {
                id: note_id,
                note: prose("parked in the wrong place"),
            },
            99,
        )
        .unwrap();
        let ctx = &log.world().comments[&note_id];
        assert_eq!(
            ctx.withdrawn,
            Some(Withdrawal {
                record: RecordId(18),
                withdrawer: human().actor,
            })
        );
        assert!(matches!(
            &ctx.comment.state,
            CommentState::Withdrawn(note) if note.as_str() == "parked in the wrong place"
        ));

        // terminal: a second withdrawal and a revision both refuse
        for command in [
            Command::WithdrawComment {
                id: note_id,
                note: prose("changed my mind again"),
            },
            Command::ReviseComment {
                id: note_id,
                body: prose("repair in place"),
            },
        ] {
            let refused = log.execute(agent(), command, 100);
            assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        }

        // the variants keep their verdicts: the offer, the accept, the
        // reject, and the use all refuse the door
        for id in [3usize, 10, 8, 17] {
            let refused = log.execute(
                human(),
                Command::WithdrawComment {
                    id: CommentId(RecordId(id)),
                    note: prose("not a note deposit"),
                },
                101,
            );
            assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        }

        // the addressed position must be a comment's birth
        let refused = log.execute(
            human(),
            Command::WithdrawComment {
                id: CommentId(RecordId(99)),
                note: prose("addresses nothing"),
            },
            102,
        );
        assert!(matches!(refused, Err(Reject::InvalidCommentId)));
        assert_eq!(log.records().len(), RECORD_COUNT + 1);
    }

    #[test]
    fn comments_have_no_state_gate() {
        let mut log = Log::new();
        populate_log(&mut log);

        let before = log.records().len();
        // t-0 concluded, t-2 cancelled: notes land on both
        for (id, state) in [(0, "concluded"), (2, "cancelled")] {
            log.execute(
                agent(),
                note(
                    Target::Task(TaskId(id)),
                    &format!("for the record, on the {state} task"),
                ),
                99,
            )
            .unwrap_or_else(|e| panic!("note on {state} task refused: {e:?}"));
        }
        assert_eq!(log.records().len(), before + 2);

        // only an unknown comment id refuses, writing nothing
        let refused = log.execute(
            human(),
            note(
                Target::Comment(CommentId(RecordId(99))),
                "addresses nothing",
            ),
            100,
        );
        assert!(matches!(refused, Err(Reject::InvalidCommentId)));
        assert_eq!(log.records().len(), before + 2);
    }

    #[test]
    fn artifacts_land_seq_ordered_beside_comments() {
        let mut log = Log::new();
        populate_log(&mut log);
        let figure = Artifact {
            name: prose("sweep over 28 decades"),
            hash: ContentHash::of(b"figure bytes"),
        };

        // both tiers author the act, like notes
        for ctx in [agent(), human()] {
            log.execute(
                ctx,
                Command::Artifact {
                    root: TaskId(3),
                    artifact: figure.clone(),
                },
                20,
            )
            .unwrap();
        }

        // an unknown task refuses, writing nothing
        let before = log.records().len();
        let refused = log.execute(
            human(),
            Command::Artifact {
                root: TaskId(9),
                artifact: figure.clone(),
            },
            21,
        );
        assert!(matches!(refused, Err(Reject::InvalidTaskId)));
        assert_eq!(log.records().len(), before);

        // the thread moved but the state did not: t-3 keeps its claim
        assert_eq!(log.world().tasks[3].task.state, TaskState::Open);
        assert_eq!(log.world().tasks[3].last_record_at, 20);

        // the pointers land in record order with their seqs
        let artifacts = &log.world().tasks[3].artifacts;
        assert_eq!(artifacts.len(), 2);
        assert_eq!(artifacts[0].0, RecordId(RECORD_COUNT));
        assert_eq!(artifacts[1].0, RecordId(RECORD_COUNT + 1));
        assert_eq!(artifacts[0].1, figure);
        assert_eq!(artifacts[1].1, figure);
        // and only the named thread holds them
        assert!(log.world().tasks[0].artifacts.is_empty());

        // no state gate: a cancelled task holds the pointer too
        log.execute(
            agent(),
            Command::Artifact {
                root: TaskId(2),
                artifact: figure,
            },
            22,
        )
        .unwrap();
        assert_eq!(log.world().tasks[2].artifacts.len(), 1);
    }

    #[test]
    fn the_establishment_births_the_shared_workspace_identity() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("carry a merge through the worktree"),
            },
            1,
        )
        .unwrap();

        // a task with no workspace refuses the checkpoint record
        let base = GitCommit::new("abc123".into()).unwrap();
        let branch = GitBranch::new("saccade/t-0".into()).unwrap();
        let worktree = WorktreePath::new("/repo/wt/t-0".into()).unwrap();
        let refused = log.execute_system(
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: base.clone(),
            },
            2,
        );
        assert!(matches!(refused, Err(Reject::WorkspaceMissing)));

        // the establishment births the lineage; the same position names the worker
        log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: base.clone(),
                branch: branch.clone(),
                worktree: worktree.clone(),
            },
            2,
        )
        .unwrap();
        let birth = RecordId(1);
        let ctx = &log.world().tasks[0];
        assert_eq!(ctx.workspace.as_ref().unwrap().workspace.checkpoint, base);
        assert_eq!(ctx.workspace.as_ref().unwrap().workspace.worktree, worktree);
        assert_eq!(ctx.workspace.as_ref().unwrap().birth, birth);
        let inc = &log.world().incarnations[&IncarnationId(birth)];
        assert_eq!(
            (inc.incarnation.task, inc.incarnation.workspace.as_path()),
            (TaskId(0), worktree.as_path())
        );

        // one lineage per task
        let refused = log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: base.clone(),
                branch: branch.clone(),
                worktree: worktree.clone(),
            },
            3,
        );
        assert!(matches!(refused, Err(Reject::WorkspaceAlreadyExists)));

        // the establishment verb rejects judgment tiers
        let refused = log.execute(
            human(),
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base,
                branch: GitBranch::new("saccade/t-1".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-1".into()).unwrap(),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::HumanOnly)));
    }

    /// An incarnation claim names the incarnation at every judgment tier.
    #[test]
    fn an_incarnation_claim_names_the_incarnation_whoever_requests_it() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("carry a merge through the worktree"),
            },
            1,
        )
        .unwrap();
        let birth = RecordId(1);
        log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            2,
        )
        .unwrap();

        // the human requests the start: ownership names the incarnation
        log.execute(
            human(),
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: Some(IncarnationId(birth)),
            },
            3,
        )
        .unwrap();
        assert!(matches!(
            log.world().tasks[0].ownership[0].owner,
            Owner::Incarnation(IncarnationId(id)) if id == birth
        ));

        // a claim whose incarnation is bound to another task refuses
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("adopt the account"),
            },
            4,
        )
        .unwrap();
        let refused = log.execute(
            agent(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: Some(IncarnationId(birth)),
            },
            5,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));

        // ordinary claims keep their owners: the human, the external agent
        log.execute(
            human(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            6,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            7,
        )
        .unwrap();
        let owners: Vec<&Owner> = log.world().tasks[1]
            .ownership
            .iter()
            .map(|e| &e.owner)
            .collect();
        assert!(matches!(owners[0], Owner::Human(_)));
        assert!(matches!(owners[1], Owner::External(_)));
    }

    #[test]
    fn checkpoints_never_rewind_at_the_explicit_door() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("carry a merge through the worktree"),
            },
            1,
        )
        .unwrap();
        let base = GitCommit::new("abc123".into()).unwrap();
        log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: base.clone(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            2,
        )
        .unwrap();

        // the machinery rebases freely: boot recovery names an
        // unrelated head and lands
        let recovery = GitCommit::new("777000".into()).unwrap();
        log.execute_system(
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: recovery.clone(),
            },
            3,
        )
        .unwrap();

        // the explicit door is agent- and human-held: an advance lands
        let merged = GitCommit::new("def456".into()).unwrap();
        for ctx in [agent(), human()] {
            log.execute(
                ctx,
                Command::CheckpointWorkspace {
                    task_id: TaskId(0),
                    checkpoint: merged.clone(),
                },
                4,
            )
            .unwrap();
        }
        assert_eq!(
            log.world().tasks[0]
                .workspace
                .as_ref()
                .unwrap()
                .workspace
                .checkpoint,
            merged
        );

        // restating the recorded head is a no-op that succeeds
        let before = log.records().len();
        log.execute(
            agent(),
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: merged.clone(),
            },
            5,
        )
        .unwrap();
        assert_eq!(log.records().len(), before + 1);
        assert_eq!(
            log.world().tasks[0]
                .workspace
                .as_ref()
                .unwrap()
                .workspace
                .checkpoint,
            merged
        );

        // a rewind names a head the record left: the base, the recovered
        // head — refused at both explicit tiers, writing nothing
        for rewind in [base, recovery] {
            for ctx in [agent(), human()] {
                let refused = log.execute(
                    ctx,
                    Command::CheckpointWorkspace {
                        task_id: TaskId(0),
                        checkpoint: rewind.clone(),
                    },
                    6,
                );
                assert!(matches!(refused, Err(Reject::CheckpointRewind)));
            }
        }
        assert_eq!(log.records().len(), before + 1);
    }
}
