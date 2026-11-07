use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Reject;
use crate::decide::{decide, enforce_tier};
use crate::events::{Command, Event};
use crate::objects::incarnation::{Incarnation, IncarnationContext, IncarnationId};
use crate::objects::message::{Message, MessageContext, MessageId, MessageState};
use crate::objects::task::{BoardCommit, Owner, OwnerEntry, Task, TaskContext, TaskId, TaskState};
use crate::objects::workspace::{
    CreatedWorkspace, CreatedWorkspaceContext, RootSpaceBinding, Workspace, WorkspaceChange,
    WorkspaceContext, WorkspaceId, WorkspaceStage, WorkspaceStageFact,
};
use crate::types::actor::ActorName;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

#[derive(Debug)]
pub struct FoldError {
    pub at: RecordId,
    pub reason: Reason,
}

#[derive(Debug)]
pub enum Reason {
    InvalidTaskId,
    /// A board commit names a parent that is not the task's current
    /// commit: revisions chain onto what actually stands.
    NotCurrentParent {
        expected: Option<GitCommit>,
    },
    /// An incorporation cites a source commit that is no recorded
    /// TaskCommitted of the source task.
    UnrecordedSourceCommit {
        source: TaskId,
        cited: GitCommit,
    },
    /// A board offer names a commit that is not the task's current one.
    NotCurrentCommit {
        current: Option<GitCommit>,
    },
    /// The actor's declared task scope covers the offered task: a
    /// task-bound producer cannot accept its own board.
    SelfAcceptance,
    /// An accepted board's incorporation cites a source commit that is
    /// not that source's accepted final commit.
    SourceNotAccepted {
        source: TaskId,
        cited: GitCommit,
    },
    /// A claim names an incarnation the record never birthed, or a
    /// historical establishment bound to another task.
    InvalidIncarnationId,
    /// A registration names a native session an incarnation already
    /// carries: the effect resolves the existing incarnation, it
    /// never births again.
    DuplicateNativeSession,
    /// A registration names a native session already recorded under a
    /// different saved locator: only the exact recorded association
    /// resolves, a reused id never rebinds to other history.
    SavedSessionMismatch,
    /// A bind names an incarnation that already carries its saved
    /// locator: the attach completes a launch-door birth exactly once.
    LocatorAlreadyBound,
    /// A message's reply or submit names a message the record never
    /// birthed, or a submission by an incarnation it does not
    /// address.
    InvalidMessageId,
    InvalidStateTransition,
    /// The task already holds a workspace: one lineage per task.
    WorkspaceAlreadyExists,
    /// The event names a task whose workspace was never established.
    WorkspaceMissing,
    /// An agent- or human-tier checkpoint names a head the workspace
    /// already left; the explicit door never rewinds.
    CheckpointRewind,
    /// A lifecycle fact names a record that is no workspace's birth:
    /// neither an independent creation nor a task-bound establishment.
    InvalidWorkspaceId,
    /// A cleanup names a workspace that is already Cleaned, or a
    /// restoration names one that is Available: the stage is not there
    /// to move, and no duplicate lifecycle fact lands.
    WorkspaceStageMismatch,
}

/// Command-path translation of fold failures into refusals; total over Reason
/// since every fold failure is command-reachable.
impl From<Reason> for Reject {
    fn from(reason: Reason) -> Reject {
        match reason {
            Reason::InvalidTaskId => Reject::InvalidTaskId,
            Reason::NotCurrentParent { expected } => Reject::NotCurrentParent {
                expected: expected.as_ref().map(|c| c.as_str().to_string()),
            },
            Reason::UnrecordedSourceCommit { source, cited } => Reject::UnrecordedSourceCommit {
                source_task: source,
                cited: cited.as_str().to_string(),
            },
            Reason::NotCurrentCommit { current } => Reject::NotCurrentCommit {
                current: current.as_ref().map(|c| c.as_str().to_string()),
            },
            Reason::SourceNotAccepted { source, cited } => Reject::SourceNotAccepted {
                source_task: source,
                cited: cited.as_str().to_string(),
            },
            Reason::SelfAcceptance => Reject::SelfAcceptance,
            Reason::InvalidIncarnationId => Reject::InvalidIncarnationId,
            Reason::DuplicateNativeSession => Reject::DuplicateNativeSession,
            Reason::SavedSessionMismatch => Reject::SavedSessionMismatch,
            Reason::InvalidMessageId => Reject::InvalidMessageId,
            Reason::InvalidStateTransition => Reject::InvalidStateTransition,
            Reason::WorkspaceAlreadyExists => Reject::WorkspaceAlreadyExists,
            Reason::WorkspaceMissing => Reject::WorkspaceMissing,
            Reason::CheckpointRewind => Reject::CheckpointRewind,
            Reason::InvalidWorkspaceId => Reject::InvalidWorkspaceId,
            Reason::WorkspaceStageMismatch => Reject::WorkspaceStageMismatch,
            Reason::LocatorAlreadyBound => Reject::LocatorAlreadyBound,
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
/// never proof - the one gate it feeds is the self-acceptance
/// refusal. Native participation can precede any task: the task
/// binding is optional, the incarnation and workspace disclosures
/// stand alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scope {
    pub task: Option<TaskId>,
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

/// A workspace birth's recorded facts, whichever way it was born: an
/// independent creation or a historical task-bound establishment. The
/// birth record's position is the identity; the task home is the
/// establishment's provenance, never a revival of task binding.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceFacts {
    pub birth: RecordId,
    /// The immutable commit the workspace was cut from
    pub base: GitCommit,
    /// The branch preserving the work
    pub branch: GitBranch,
    /// The workspace's absolute address
    pub worktree: WorktreePath,
    /// The establishment's home task; None for an independent creation
    pub home: Option<TaskId>,
    /// The stage the workspace object itself carries
    pub stage: WorkspaceStage,
}

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
    /// Record-derived identities: historical establishment births and
    /// registered native sessions, in birth order
    pub incarnations: BTreeMap<IncarnationId, IncarnationContext>,
    /// Durable addressed messages, in birth order
    pub messages: BTreeMap<MessageId, MessageContext>,
    /// Workspaces born by their own creation records: independent of
    /// any task or incarnation, identity at the birth record. No
    /// human-facing number derives from these births.
    pub workspaces: BTreeMap<WorkspaceId, CreatedWorkspaceContext>,
    /// The latest confirmed saved-directory change per native
    /// incarnation: a machinery fact cache, never a lifecycle stage.
    pub workspace_changes: BTreeMap<IncarnationId, WorkspaceChange>,
    /// The primary checkout's bound Herdr destination: the latest
    /// explicit human choice, None after a clear or before any bind.
    /// A fact cache like the workspace changes; the log keeps the
    /// history.
    pub root_space: Option<RootSpaceBinding>,
    /// Every applied record: existence for the citation guard, its
    /// home thread where the fold gives one. A record without a home
    /// still exists. No residency - the named record stays the event
    /// it is.
    pub provenance: BTreeMap<RecordId, Option<TaskId>>,
}

impl World {
    pub fn new() -> Self {
        World {
            tasks: Vec::new(),
            incarnations: BTreeMap::new(),
            messages: BTreeMap::new(),
            workspaces: BTreeMap::new(),
            workspace_changes: BTreeMap::new(),
            root_space: None,
            provenance: BTreeMap::new(),
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

    /// The boards caches' current state, from the fold: the ref key
    /// (the birth record) and the standing commit per task that holds
    /// published revisions. Startup repair sets every cache to this.
    pub fn boards_current(&self) -> Vec<(String, String)> {
        self.tasks
            .iter()
            .filter(|ctx| !ctx.commits.is_empty())
            .map(|ctx| {
                (
                    ctx.birth.0.to_string(),
                    ctx.commits
                        .last()
                        .expect("the list is nonempty")
                        .commit
                        .as_str()
                        .to_string(),
                )
            })
            .collect()
    }

    /// The incarnation's public number: its position in incarnation
    /// birth order, the number the human-facing i-N carries.
    pub fn incarnation_num(&self, id: IncarnationId) -> Option<usize> {
        if !self.incarnations.contains_key(&id) {
            return None;
        }
        Some(self.incarnations.range(..id).count())
    }

    /// The incarnation holding this public number, resolved from the fold.
    pub fn incarnation_at_num(&self, n: usize) -> Option<IncarnationId> {
        self.incarnations.keys().nth(n).copied()
    }

    /// The message's public number: its position in message birth
    /// order, the number the human-facing m-N carries. Births do not
    /// carry their own ids; the fold derives them.
    pub fn message_num(&self, id: MessageId) -> Option<usize> {
        if !self.messages.contains_key(&id) {
            return None;
        }
        Some(self.messages.range(..id).count())
    }

    /// The message holding this public number, resolved from the fold.
    pub fn message_at_num(&self, n: usize) -> Option<MessageId> {
        self.messages.keys().nth(n).copied()
    }

    /// A workspace birth's recorded facts, whichever way it was born:
    /// an independent creation or a historical task-bound
    /// establishment. The birth record's position is the identity;
    /// the task home is the establishment's provenance, never a
    /// revival of task binding.
    pub fn workspace_facts(&self, birth: RecordId) -> Option<WorkspaceFacts> {
        if let Some(context) = self.workspaces.get(&WorkspaceId(birth)) {
            return Some(WorkspaceFacts {
                birth,
                base: context.workspace.base.clone(),
                branch: context.workspace.branch.clone(),
                worktree: context.workspace.worktree.clone(),
                home: None,
                stage: context.workspace.stage,
            });
        }
        self.tasks.iter().enumerate().find_map(|(position, ctx)| {
            let established = ctx.workspace.as_ref()?;
            if established.birth != birth {
                return None;
            }
            Some(WorkspaceFacts {
                birth,
                base: established.workspace.base.clone(),
                branch: established.workspace.branch.clone(),
                worktree: established.workspace.worktree.clone(),
                home: Some(TaskId(position)),
                stage: established.workspace.stage,
            })
        })
    }

    /// Every workspace birth recorded at this exact path, in birth
    /// order. The caller compares canonical forms; two births on one
    /// path are the caller's ambiguity to refuse.
    pub fn workspace_births_at(&self, path: &Path) -> Vec<RecordId> {
        let mut births: Vec<RecordId> = self
            .workspaces
            .iter()
            .filter(|(_, ctx)| ctx.workspace.worktree.as_path() == path)
            .map(|(id, _)| id.0)
            .collect();
        for ctx in &self.tasks {
            if let Some(established) = &ctx.workspace
                && established.workspace.worktree.as_path() == path
            {
                births.push(established.birth);
            }
        }
        births.sort();
        births
    }

    /// Every workspace birth in birth order, both kinds - the
    /// reconciliation and resolution walks' enumeration.
    pub fn workspace_births(&self) -> Vec<RecordId> {
        let mut births: Vec<RecordId> = self.workspaces.keys().map(|id| id.0).collect();
        for ctx in &self.tasks {
            if let Some(established) = &ctx.workspace {
                births.push(established.birth);
            }
        }
        births.sort();
        births
    }

    /// The workspace birth's current lifecycle stage, owned by the
    /// workspace object; None when the record names no workspace birth.
    pub fn workspace_stage(&self, birth: RecordId) -> Option<WorkspaceStage> {
        self.workspace_facts(birth).map(|facts| facts.stage)
    }

    /// The recorded workspace at this exact path whose stage is
    /// Cleaned, when one stands: the ordinary-use guard's whole
    /// question. Unrecognized directories name nothing.
    pub fn cleaned_workspace_at(&self, path: &Path) -> Option<RecordId> {
        self.workspace_births_at(path)
            .into_iter()
            .find(|birth| self.workspace_stage(*birth) == Some(WorkspaceStage::Cleaned))
    }

    /// Only valid path for world mutation
    pub fn apply(mut self, record: Record) -> Result<Self, FoldError> {
        let at = record.id;
        self.apply_record(record)
            .map_err(|reason| FoldError { at, reason })?;
        Ok(self)
    }

    /// The owner a claim names - exactly the ratified kinds. The
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
                let birth = record.id;
                self.provenance
                    .insert(birth, Some(TaskId(self.tasks.len())));
                self.tasks.push(TaskContext {
                    task: Task {
                        state: TaskState::Open,
                        name: task_name,
                    },
                    birth: record.id,
                    last_updated: record.id,
                    last_record_at: record.timestamp,
                    ownership: Vec::new(),
                    commits: Vec::new(),
                    birth_actor: record.context.actor.clone(),
                    workspace: None,
                });
            }
            Event::TaskClaimed { id, incarnation } => {
                self.provenance.insert(record.id, Some(id));
                let owner = Self::owner_of(&record.context, incarnation)?;
                if let Some(inc) = incarnation {
                    let born = self
                        .incarnations
                        .get(&inc)
                        .ok_or(Reason::InvalidIncarnationId)?;
                    // a historical establishment is bound to its birth
                    // task; a native identity is task-free by law
                    if let Incarnation::Established { task, .. } = &born.incarnation
                        && *task != id
                    {
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
            Event::TaskCommitted {
                task,
                ref commit,
                ref parent,
                ref incorporations,
            } => {
                self.provenance.insert(record.id, Some(task));
                // the incorporation sources resolve before the mutable
                // context borrow: their recorded revisions are the guard
                for (source, cited) in incorporations {
                    let src = self.tasks.get(source.0).ok_or(Reason::InvalidTaskId)?;
                    if !src.commits.iter().any(|c| &c.commit == cited) {
                        return Err(Reason::UnrecordedSourceCommit {
                            source: *source,
                            cited: cited.clone(),
                        });
                    }
                }
                let task_ctx = self.tasks.get_mut(task.0).ok_or(Reason::InvalidTaskId)?;
                // a revision publishes on an open task; offered and
                // concluded boards publish through reject and reopen
                if !matches!(task_ctx.task.state, TaskState::Open) {
                    return Err(Reason::InvalidStateTransition);
                }
                let current = task_ctx.commits.last().map(|c| c.commit.clone());
                if *parent != current {
                    return Err(Reason::NotCurrentParent { expected: current });
                }
                task_ctx.commits.push(BoardCommit {
                    commit: commit.clone(),
                    incorporations: incorporations.clone(),
                    record: record.id,
                    at: record.timestamp,
                });
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::TaskOffered { task, ref commit } => {
                self.provenance.insert(record.id, Some(task));
                let task_ctx = self.tasks.get_mut(task.0).ok_or(Reason::InvalidTaskId)?;
                if let Some(next) = task_ctx.task.state.transition(&record.event) {
                    task_ctx.task.state = next;
                } else {
                    return Err(Reason::InvalidStateTransition);
                }
                let current = task_ctx.commits.last().map(|c| c.commit.clone());
                if Some(commit.clone()) != current {
                    return Err(Reason::NotCurrentCommit { current });
                }
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::TaskRejected { task, grounds: _ } => {
                self.provenance.insert(record.id, Some(task));
                let task_ctx = self.tasks.get_mut(task.0).ok_or(Reason::InvalidTaskId)?;
                if let Some(next) = task_ctx.task.state.transition(&record.event) {
                    task_ctx.task.state = next;
                } else {
                    return Err(Reason::InvalidStateTransition);
                }
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::TaskAccepted { task } => {
                self.provenance.insert(record.id, Some(task));
                // the scope disclosure feeds the one gate: an agent
                // whose declared task covers this one cannot accept it
                if record.context.tier == Tier::Agent
                    && record
                        .context
                        .scope
                        .as_ref()
                        .and_then(|scope| scope.task)
                        .is_some_and(|scoped| scoped == task)
                {
                    return Err(Reason::SelfAcceptance);
                }
                // the offered stage admits the accept, and an offered
                // task always holds the commit its offer named; the
                // stage gates before the offered commit is read
                let (next, incorporations) = {
                    let task_ctx = self.tasks.get(task.0).ok_or(Reason::InvalidTaskId)?;
                    let next = task_ctx
                        .task
                        .state
                        .transition(&record.event)
                        .ok_or(Reason::InvalidStateTransition)?;
                    let incorporations = task_ctx
                        .commits
                        .last()
                        .expect("an offered task holds the commit its offer named")
                        .incorporations
                        .clone();
                    (next, incorporations)
                };
                // every incorporation cites exactly its source's
                // accepted final commit
                for (source, cited) in &incorporations {
                    let src = self.tasks.get(source.0).ok_or(Reason::InvalidTaskId)?;
                    let accepted = matches!(src.task.state, TaskState::Concluded)
                        && src.commits.last().is_some_and(|last| &last.commit == cited);
                    if !accepted {
                        return Err(Reason::SourceNotAccepted {
                            source: *source,
                            cited: cited.clone(),
                        });
                    }
                }
                let task_ctx = self.tasks.get_mut(task.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.task.state = next;
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::TaskCancelled { id, .. } | Event::TaskReopened { id, .. } => {
                self.provenance.insert(record.id, Some(id));
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.task.state = task_ctx
                    .task
                    .state
                    .transition(&record.event)
                    .ok_or(Reason::InvalidStateTransition)?;
                task_ctx.last_updated = record.id;
                task_ctx.last_record_at = record.timestamp;
            }
            // Workspace events
            Event::WorkspaceEstablished {
                task_id,
                ref base,
                ref branch,
                ref worktree,
            } => {
                self.provenance.insert(record.id, Some(task_id));
                let task_ctx = self.tasks.get_mut(task_id.0).ok_or(Reason::InvalidTaskId)?;
                if task_ctx.workspace.is_some() {
                    return Err(Reason::WorkspaceAlreadyExists);
                }
                // the object carries its own stage, born Available; the
                // birth itself is the fact, so the context names none
                task_ctx.workspace = Some(WorkspaceContext {
                    workspace: Workspace {
                        base: base.clone(),
                        branch: branch.clone(),
                        checkpoint: base.clone(),
                        worktree: worktree.clone(),
                        stage: WorkspaceStage::Available,
                    },
                    birth: record.id,
                    heads: vec![base.clone()],
                    stage_fact: None,
                });
                // the same birth is the historical attribution identity:
                // one record position, both names. It carries no native
                // session binding and births no message sender.
                self.incarnations.insert(
                    IncarnationId(record.id),
                    IncarnationContext {
                        incarnation: Incarnation::Established {
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
                self.provenance.insert(record.id, Some(task_id));
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
            Event::WorkspaceCreated {
                ref base,
                ref branch,
                ref worktree,
            } => {
                // a birth with no task home still exists; no task, no
                // incarnation, no numbering rides it
                self.provenance.insert(record.id, None);
                self.workspaces.insert(
                    WorkspaceId(record.id),
                    CreatedWorkspaceContext {
                        workspace: CreatedWorkspace {
                            base: base.clone(),
                            branch: branch.clone(),
                            worktree: worktree.clone(),
                            stage: WorkspaceStage::Available,
                        },
                        born_at: record.timestamp,
                        stage_fact: None,
                    },
                );
            }
            Event::WorkspaceChanged {
                incarnation,
                ref worktree,
            } => {
                // the change names a recorded native incarnation: a
                // historical attribution identity has no saved session
                // to move
                let born = self
                    .incarnations
                    .get(&incarnation)
                    .ok_or(Reason::InvalidIncarnationId)?;
                if born.incarnation.native_id().is_none() {
                    return Err(Reason::InvalidIncarnationId);
                }
                // a fact with no task home: it rides the incarnation
                self.provenance.insert(record.id, None);
                self.workspace_changes.insert(
                    incarnation,
                    WorkspaceChange {
                        worktree: worktree.clone(),
                        record: record.id,
                        at: record.timestamp,
                    },
                );
            }
            Event::RootSpaceBound { space, tab } => {
                // the human's chosen root destination: latest choice
                // wins, history stays in the log; no task home rides it
                self.provenance.insert(record.id, None);
                self.root_space = Some(RootSpaceBinding { space, tab });
            }
            Event::RootSpaceCleared {} => {
                self.provenance.insert(record.id, None);
                self.root_space = None;
            }
            Event::WorkspaceCleaned { workspace } | Event::WorkspaceRestored { workspace } => {
                // the fact references an existing workspace birth of
                // either kind: the store keeps only the existence guard
                // and the provenance landing - the stage's transition
                // is the object's own pure table, and the setting
                // fact's record and time live on the context
                if let Some(context) = self.workspaces.get_mut(&WorkspaceId(workspace)) {
                    context.workspace.stage = context
                        .workspace
                        .stage
                        .transition(&record.event)
                        .ok_or(Reason::WorkspaceStageMismatch)?;
                    context.stage_fact = Some(WorkspaceStageFact {
                        record: record.id,
                        at: record.timestamp,
                    });
                } else {
                    let established = self
                        .tasks
                        .iter_mut()
                        .find_map(|ctx| ctx.workspace.as_mut().filter(|w| w.birth == workspace))
                        .ok_or(Reason::InvalidWorkspaceId)?;
                    established.workspace.stage = established
                        .workspace
                        .stage
                        .transition(&record.event)
                        .ok_or(Reason::WorkspaceStageMismatch)?;
                    established.stage_fact = Some(WorkspaceStageFact {
                        record: record.id,
                        at: record.timestamp,
                    });
                }
                // the fact rides no task home, like every machinery
                // workspace fact
                self.provenance.insert(record.id, None);
            }
            // Incarnation and message events
            Event::IncarnationRegistered {
                ref native_id,
                ref saved_session,
                ref address,
                adopted,
                harness,
            } => {
                // registering an existing exact native session resolves
                // at the effect; the fold refuses the second birth, and
                // a reused id with a different locator never rebinds.
                // Identity is interpreted within its harness: the same
                // id under another harness is another session
                let known = self.incarnations.values().find(|ctx| {
                    ctx.incarnation.native_id() == Some(native_id)
                        && ctx.incarnation.harness() == Some(harness)
                });
                match known {
                    Some(ctx) if ctx.incarnation.saved_session() == saved_session.as_ref() => {
                        return Err(Reason::DuplicateNativeSession);
                    }
                    Some(_) => return Err(Reason::SavedSessionMismatch),
                    None => {}
                }
                // a birth without a home still exists
                self.provenance.insert(record.id, None);
                self.incarnations.insert(
                    IncarnationId(record.id),
                    IncarnationContext {
                        incarnation: Incarnation::Native {
                            native_id: native_id.clone(),
                            saved_session: saved_session.clone(),
                            address: address.clone(),
                            adopted,
                            harness,
                        },
                        born_at: record.timestamp,
                    },
                );
            }
            Event::SavedSessionBound {
                incarnation,
                ref saved_session,
            } => {
                // the attach completes a launch-door birth exactly once:
                // the incarnation must stand, be native, and be unbound
                let ctx = self
                    .incarnations
                    .get_mut(&incarnation)
                    .ok_or(Reason::InvalidIncarnationId)?;
                let Incarnation::Native {
                    saved_session: bound,
                    ..
                } = &mut ctx.incarnation
                else {
                    return Err(Reason::InvalidIncarnationId);
                };
                if bound.is_some() {
                    return Err(Reason::LocatorAlreadyBound);
                }
                *bound = Some(saved_session.clone());
            }
            Event::HostAddressAssigned {
                incarnation,
                ref address,
                adopted,
            } => {
                // the association's address moves with a relaunch; the
                // incarnation must stand and be native
                let ctx = self
                    .incarnations
                    .get_mut(&incarnation)
                    .ok_or(Reason::InvalidIncarnationId)?;
                let Incarnation::Native {
                    address: carried,
                    adopted: carried_kind,
                    ..
                } = &mut ctx.incarnation
                else {
                    return Err(Reason::InvalidIncarnationId);
                };
                *carried = address.clone();
                *carried_kind = adopted;
            }
            Event::MessageSent {
                from,
                to,
                reply_to,
                task,
                ref body,
            } => {
                // the sender is the session's own registered identity:
                // machine-supplied from the declared scope, never prose
                if record.context.scope.as_ref().and_then(|s| s.incarnation) != Some(from) {
                    return Err(Reason::InvalidIncarnationId);
                }
                let sender = self
                    .incarnations
                    .get(&from)
                    .ok_or(Reason::InvalidIncarnationId)?;
                if sender.incarnation.native_id().is_none() {
                    return Err(Reason::InvalidIncarnationId);
                }
                if !self.incarnations.contains_key(&to) {
                    return Err(Reason::InvalidIncarnationId);
                }
                // the recipient must be a registered native session:
                // historical attribution identities have no execution
                // boundary, so mail to them could never be submitted
                if self.incarnations[&to].incarnation.native_id().is_none() {
                    return Err(Reason::InvalidIncarnationId);
                }
                if reply_to.is_some_and(|m| !self.messages.contains_key(&m)) {
                    return Err(Reason::InvalidMessageId);
                }
                if task.is_some_and(|t| self.tasks.get(t.0).is_none()) {
                    return Err(Reason::InvalidTaskId);
                }
                // messages take no task-thread home, whatever context rides
                self.provenance.insert(record.id, None);
                self.messages.insert(
                    MessageId(record.id),
                    MessageContext {
                        message: Message {
                            from,
                            to,
                            reply_to,
                            task,
                            body: body.clone(),
                        },
                        state: MessageState::Pending,
                        actor: record.context.actor.clone(),
                        tier: record.context.tier,
                        born_at: record.timestamp,
                    },
                );
            }
            Event::MessageSubmitted { id, by } => {
                // submission is the recipient's machinery act: the
                // submitter is the addressee, the mail is pending
                let msg = self.messages.get(&id).ok_or(Reason::InvalidMessageId)?;
                if msg.message.to != by {
                    return Err(Reason::InvalidMessageId);
                }
                if !self.incarnations.contains_key(&by) {
                    return Err(Reason::InvalidIncarnationId);
                }
                self.provenance.insert(record.id, None);
                let msg = self.messages.get_mut(&id).expect("resident above");
                msg.state = msg
                    .state
                    .transition(&record.event)
                    .ok_or(Reason::InvalidStateTransition)?;
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
/// that were produced. Takes the world as given - the caller owns its
/// truth; `db::record` reconstructs it from the log under the write lock
/// before calling.
pub fn execute(
    world: &World,
    base: usize,
    context: &Context,
    command: Command,
    now: u64,
) -> Result<(World, Vec<Record>), Reject> {
    let events = decide(command.clone());
    enforce_tier(context, &events)?;
    world.stage(base, context, now, &events)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Event;
    use crate::Scope;
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::task::{Owner, TaskId, TaskState};
    use crate::types::actor::ActorName;
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};
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
                task: Some(TaskId(task)),
                incarnation: None,
                workspace: None,
            }),
        }
    }

    fn prose(text: &str) -> Prose {
        Prose::new(text.into()).unwrap()
    }

    const RECORD_COUNT: usize = 12;

    /// foo is committed, offered, rejected, recommitted, accepted; the
    /// consumer bar commits citing foo's accepted revision, and is
    /// offered in turn.
    fn populate_log(log: &mut Log) {
        let human_ctx = human();
        let agent_ctx = agent();
        let revision = |n: u8| GitCommit::new(format!("{n:02}{}", "a".repeat(38))).unwrap();

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
            Command::CommitBoard {
                task: TaskId(0),
                commit: revision(1),
                parent: None,
                incorporations: Vec::new(),
            },
            3,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::OfferBoard {
                task: TaskId(0),
                commit: revision(1),
            },
            4,
        )
        .unwrap();
        log.execute(
            human_ctx.clone(),
            Command::RejectBoard {
                task: TaskId(0),
                grounds: prose("the edge case still fails"),
            },
            5,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::CommitBoard {
                task: TaskId(0),
                commit: revision(2),
                parent: Some(revision(1)),
                incorporations: Vec::new(),
            },
            6,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::OfferBoard {
                task: TaskId(0),
                commit: revision(2),
            },
            7,
        )
        .unwrap();
        log.execute(
            human_ctx.clone(),
            Command::AcceptBoard { task: TaskId(0) },
            8,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: prose("adopt foo's accounts"),
            },
            9,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            10,
        )
        .unwrap();
        // the consumer cites foo's accepted revision r2 exactly
        log.execute(
            agent_ctx.clone(),
            Command::CommitBoard {
                task: TaskId(1),
                commit: revision(3),
                parent: None,
                incorporations: vec![(TaskId(0), revision(2))],
            },
            11,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            Command::OfferBoard {
                task: TaskId(1),
                commit: revision(3),
            },
            12,
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
            Reject::from(Reason::NotCurrentParent { expected: None }),
            Reject::NotCurrentParent { expected: None }
        ));
        assert!(matches!(
            Reject::from(Reason::UnrecordedSourceCommit {
                source: TaskId(0),
                cited: GitCommit::new("a".repeat(40)).unwrap()
            }),
            Reject::UnrecordedSourceCommit { .. }
        ));
        assert!(matches!(
            Reject::from(Reason::NotCurrentCommit { current: None }),
            Reject::NotCurrentCommit { current: None }
        ));
        assert!(matches!(
            Reject::from(Reason::SourceNotAccepted {
                source: TaskId(0),
                cited: GitCommit::new("a".repeat(40)).unwrap()
            }),
            Reject::SourceNotAccepted { .. }
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
            Reject::from(Reason::DuplicateNativeSession),
            Reject::DuplicateNativeSession
        ));
        assert!(matches!(
            Reject::from(Reason::SavedSessionMismatch),
            Reject::SavedSessionMismatch
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidMessageId),
            Reject::InvalidMessageId
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
        assert!(matches!(
            Reject::from(Reason::InvalidWorkspaceId),
            Reject::InvalidWorkspaceId
        ));
        assert!(matches!(
            Reject::from(Reason::WorkspaceStageMismatch),
            Reject::WorkspaceStageMismatch
        ));
        assert!(matches!(
            Reject::from(Reason::LocatorAlreadyBound),
            Reject::LocatorAlreadyBound
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

    /// The story folds the ratified lifecycle, including the use's
    /// prospective birth, explicit binding, and explicit rebinding.
    #[test]
    fn the_story_folds_the_ratified_lifecycle() {
        let mut log = Log::new();
        populate_log(&mut log);
        let world = log.world();

        // foo: committed r1, offered, rejected, recommitted r2,
        // accepted - the revisions stay listed, the state concluded
        assert_eq!(world.tasks[0].task.state, TaskState::Concluded);
        assert_eq!(world.tasks[0].commits.len(), 2);
        assert_eq!(
            world.tasks[0].commits[0].commit.as_str(),
            "01aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert_eq!(
            world.tasks[0].commits[1].commit.as_str(),
            "02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert_eq!(world.tasks[0].revision_of(RecordId(2)), Some(1));
        assert_eq!(world.tasks[0].revision_of(RecordId(5)), Some(2));

        // bar: the consumer stacked on foo's accepted r2
        assert_eq!(world.tasks[1].task.state, TaskState::Offered);
        assert_eq!(world.tasks[1].commits.len(), 1);
        assert_eq!(
            world.tasks[1].commits[0].incorporations,
            vec![(
                TaskId(0),
                GitCommit::new("02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()).unwrap()
            )]
        );
        assert_eq!(world.tasks[1].commits[0].record, RecordId(10));
        assert_eq!(world.tasks[1].revision_of(RecordId(10)), Some(1));

        // the create-reply query: births resolve, non-births don't
        assert_eq!(world.task_born_at(RecordId(0)), Some(TaskId(0)));
        assert_eq!(world.task_born_at(RecordId(2)), None);

        // the boards caches' current state: foo at r2's commit, bar at
        // its own
        assert_eq!(
            world.boards_current(),
            vec![
                (
                    "0".to_string(),
                    "02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()
                ),
                (
                    "8".to_string(),
                    "03aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()
                ),
            ]
        );
    }

    #[test]
    fn claim_transfers_ownership_without_touching_state() {
        let mut log = Log::new();
        populate_log(&mut log);

        // bar's agent claim is the recorded ownership the story seeds
        assert_eq!(
            log.world().tasks[1].ownership,
            vec![OwnerEntry {
                owner: Owner::External(agent().actor),
                record: RecordId(9),
                at: 10,
            }]
        );

        // a claim transfers at any state: offered accepts the takeover
        let before = log.world().tasks[1].task.state.clone();
        log.execute(
            human(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            99,
        )
        .unwrap();
        assert_eq!(log.world().tasks[1].task.state, before);
        assert_eq!(log.world().tasks[1].ownership.len(), 2);
        assert!(matches!(
            log.world().tasks[1].ownership[0].owner,
            Owner::External(_)
        ));
        assert!(matches!(
            log.world().tasks[1].ownership[1].owner,
            Owner::Human(_)
        ));

        // a second claim transfers and preserves history
        log.execute(
            agent(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            100,
        )
        .unwrap();
        assert_eq!(log.world().tasks[1].ownership.len(), 3);
        assert!(matches!(
            log.world().tasks[1].ownership[2].owner,
            Owner::External(_)
        ));

        // an incarnation reference the record never birthed refuses
        let refused = log.execute(
            agent(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: Some(crate::objects::incarnation::IncarnationId(RecordId(3))),
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
                id: TaskId(1),
                explanation: prose("agent cannot stop"),
            },
            Command::ReopenTask {
                id: TaskId(1),
                explanation: prose("agent cannot resume"),
            },
        ] {
            let refused = log.execute(agent(), command, 99);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }

        // the human cancels offered t-1 with the explanation riding state
        log.execute(
            human(),
            Command::CancelTask {
                id: TaskId(1),
                explanation: prose("the consumer need dissolved"),
            },
            99,
        )
        .unwrap();
        assert_eq!(
            log.world().tasks[1].task.state,
            TaskState::Cancelled(prose("the consumer need dissolved"))
        );

        // ordinary activity never reopens a cancelled task
        log.execute(
            agent(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            100,
        )
        .unwrap();
        assert!(matches!(
            log.world().tasks[1].task.state,
            TaskState::Cancelled(_)
        ));

        // reopening restores open with the explanation recorded
        log.execute(
            human(),
            Command::ReopenTask {
                id: TaskId(1),
                explanation: prose("the need returned"),
            },
            101,
        )
        .unwrap();
        assert_eq!(log.world().tasks[1].task.state, TaskState::Open);

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
                id: TaskId(0),
                explanation: prose("only cancelled tasks reopen"),
            },
            103,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
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
        let base = GitCommit::new(format!("{}{}", "abc123", "0".repeat(34))).unwrap();
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
        let recovery = GitCommit::new(format!("{}{}", "777000", "0".repeat(34))).unwrap();
        log.execute_system(
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: recovery.clone(),
            },
            3,
        )
        .unwrap();

        // the explicit door is agent- and human-held: an advance lands
        let merged = GitCommit::new(format!("{}{}", "def456", "0".repeat(34))).unwrap();
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
        // head - refused at both explicit tiers, writing nothing
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

    /// The root destination is the human's latest explicit choice:
    /// agents and machinery refuse both acts, a rebind replaces, a
    /// clear removes, and the log keeps every recorded choice.
    #[test]
    fn the_root_destination_is_the_humans_latest_explicit_choice() {
        let mut log = Log::new();
        populate_log(&mut log);

        // agents and machinery refuse both acts
        for ctx in [agent(), Context::system()] {
            let refused = log.execute(
                ctx.clone(),
                Command::BindRootSpace {
                    space: "spaAgent".into(),
                    tab: "tbAgent".into(),
                },
                100,
            );
            assert!(matches!(refused, Err(Reject::HumanOnly)));
            let refused = log.execute(ctx, Command::ClearRootSpace {}, 100);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        assert!(log.world().root_space.is_none());

        // the human binds; the pair stands on the world
        log.execute(
            human(),
            Command::BindRootSpace {
                space: "spaManual".into(),
                tab: "tbManual".into(),
            },
            100,
        )
        .unwrap();
        assert_eq!(
            log.world()
                .root_space
                .as_ref()
                .map(|b| (b.space.as_str(), b.tab.as_str())),
            Some(("spaManual", "tbManual"))
        );

        // a rebind replaces the standing choice
        log.execute(
            human(),
            Command::BindRootSpace {
                space: "spaOther".into(),
                tab: "tbOther".into(),
            },
            101,
        )
        .unwrap();
        assert_eq!(
            log.world()
                .root_space
                .as_ref()
                .map(|b| (b.space.as_str(), b.tab.as_str())),
            Some(("spaOther", "tbOther"))
        );

        // clear removes it; the record history keeps all three choices
        log.execute(human(), Command::ClearRootSpace {}, 102)
            .unwrap();
        assert!(log.world().root_space.is_none());
        let choices = log
            .records()
            .iter()
            .filter(|r| {
                matches!(
                    r.event,
                    Event::RootSpaceBound { .. } | Event::RootSpaceCleared {}
                )
            })
            .count();
        assert_eq!(choices, 3);
    }

    /// The use's guards: self-reliance and source existence at birth;
    /// the binding's eligibility at every selection. A bound use keeps
    /// its selection when the source replaces its conclusion.
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
            inc.incarnation,
            Incarnation::Established {
                task: TaskId(0),
                workspace: worktree.clone(),
            }
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

    // -- the task-free workspace contracts ---------------------------

    fn created(label_branch: &str, path: &str) -> Command {
        Command::CreateWorkspace {
            base: GitCommit::new("abc123".into()).unwrap(),
            branch: GitBranch::new(label_branch.into()).unwrap(),
            worktree: WorktreePath::new(path.into()).unwrap(),
        }
    }

    /// A created workspace births its own identity: no task, no
    /// incarnation, no claim or message - and no i-N advance.
    #[test]
    fn creation_births_a_task_free_workspace_identity() {
        let mut log = Log::new();
        log.execute_system(created("feature-abc123ab", "/repo/wt/feature-abc123ab"), 1)
            .unwrap();
        log.execute_system(
            created("workspace-deadbeef", "/repo/wt/workspace-deadbeef"),
            2,
        )
        .unwrap();

        let world = log.world();
        assert_eq!(world.tasks.len(), 0);
        // no incarnation rides a creation: the i-N numbering is untouched
        assert_eq!(world.incarnations.len(), 0);
        assert_eq!(world.workspaces.len(), 2);
        let birth = RecordId(0);
        assert_eq!(world.provenance[&birth], None);
        assert_eq!(
            world.workspaces[&crate::WorkspaceId(birth)].workspace,
            crate::CreatedWorkspace {
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("feature-abc123ab".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/feature-abc123ab".into()).unwrap(),
                stage: crate::WorkspaceStage::Available,
            }
        );
        // several sessions may select one workspace: a second creation
        // naming the same directory is another birth, never a refusal
        log.execute_system(created("feature-abc123ab", "/repo/wt/feature-abc123ab"), 3)
            .unwrap();
        assert_eq!(log.world().workspaces.len(), 3);

        // the creation verb rejects judgment tiers
        for ctx in [human(), agent()] {
            let refused = log.execute(ctx, created("rejected-label", "/repo/wt/x"), 4);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        assert_eq!(log.records().len(), 3);
    }

    /// The lifecycle stage rides the workspace birth, both birth kinds
    /// alike: Available at birth, Cleaned on the completed fact,
    /// Available again on restoration, and every cycle's fact kept.
    /// Historical attribution stands beside the stage, untouched.
    #[test]
    fn the_lifecycle_rides_the_workspace_birth_across_both_kinds() {
        let mut log = Log::new();
        log.execute_system(created("feature-abc123ab", "/repo/wt/feature-abc123ab"), 1)
            .unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("establish a lineage"),
            },
            2,
        )
        .unwrap();
        log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            3,
        )
        .unwrap();

        let world = log.world();
        let created_birth = RecordId(0);
        let established_birth = RecordId(2);
        assert_eq!(
            world.workspace_stage(created_birth),
            Some(crate::WorkspaceStage::Available)
        );
        assert_eq!(
            world.workspace_stage(established_birth),
            Some(crate::WorkspaceStage::Available)
        );
        assert_eq!(
            world.workspace_births_at(Path::new("/repo/wt/feature-abc123ab")),
            vec![created_birth]
        );
        assert_eq!(
            world.workspace_births_at(Path::new("/repo/wt/t-0")),
            vec![established_birth]
        );
        let facts = world.workspace_facts(created_birth).unwrap();
        assert_eq!(facts.home, None);
        let facts = world.workspace_facts(established_birth).unwrap();
        assert_eq!(facts.home, Some(TaskId(0)));
        assert_eq!(facts.branch.as_str(), "saccade/t-0");
        // the historical attribution identity stands beside the stage
        assert!(
            world
                .incarnations
                .contains_key(&IncarnationId(established_birth))
        );
        assert_eq!(
            world.tasks[0].workspace.as_ref().unwrap().birth,
            established_birth
        );

        // the created one cycles: clean, restore, clean again
        log.execute_system(
            Command::CleanWorkspace {
                workspace: created_birth,
            },
            4,
        )
        .unwrap();
        assert_eq!(
            log.world().workspace_stage(created_birth),
            Some(crate::WorkspaceStage::Cleaned)
        );
        // the lifecycle fact rides no task home, like every machinery
        // workspace fact
        assert_eq!(log.world().provenance[&RecordId(3)], None);
        assert_eq!(
            log.world()
                .cleaned_workspace_at(Path::new("/repo/wt/feature-abc123ab")),
            Some(created_birth)
        );
        log.execute_system(
            Command::RestoreWorkspace {
                workspace: created_birth,
            },
            5,
        )
        .unwrap();
        assert_eq!(
            log.world().workspace_stage(created_birth),
            Some(crate::WorkspaceStage::Available)
        );
        assert_eq!(
            log.world()
                .cleaned_workspace_at(Path::new("/repo/wt/feature-abc123ab")),
            None
        );
        log.execute_system(
            Command::CleanWorkspace {
                workspace: created_birth,
            },
            6,
        )
        .unwrap();
        // earlier facts stay in the log; the stage is the latest's
        assert_eq!(log.records().len(), 6);

        // the established one cycles through the same door
        log.execute_system(
            Command::CleanWorkspace {
                workspace: established_birth,
            },
            7,
        )
        .unwrap();
        assert_eq!(
            log.world().workspace_stage(established_birth),
            Some(crate::WorkspaceStage::Cleaned)
        );
        // the task's workspace lineage is untouched by the cleanup
        assert_eq!(
            log.world().tasks[0].workspace.as_ref().unwrap().birth,
            established_birth
        );
        log.execute_system(
            Command::RestoreWorkspace {
                workspace: established_birth,
            },
            8,
        )
        .unwrap();
        assert_eq!(
            log.world().workspace_stage(established_birth),
            Some(crate::WorkspaceStage::Available)
        );
    }

    /// The lifecycle gates know what they move: an unknown birth
    /// refuses, a duplicate cleanup refuses, restoring an Available
    /// workspace is the held seam, and the door is machinery-only.
    #[test]
    fn the_lifecycle_gates_know_what_they_move() {
        let mut log = Log::new();
        log.execute_system(created("feature-abc123ab", "/repo/wt/feature-abc123ab"), 1)
            .unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("carry a merge through the worktree"),
            },
            2,
        )
        .unwrap();
        let birth = RecordId(0);
        let task_birth = RecordId(1);

        // a record that is no workspace's birth - a task birth - refuses
        let refused = log.execute_system(
            Command::CleanWorkspace {
                workspace: task_birth,
            },
            3,
        );
        assert!(matches!(refused, Err(Reject::InvalidWorkspaceId)));
        let refused = log.execute_system(
            Command::CleanWorkspace {
                workspace: RecordId(99),
            },
            3,
        );
        assert!(matches!(refused, Err(Reject::InvalidWorkspaceId)));

        // the cleanup lands once; the duplicate stage move refuses
        log.execute_system(Command::CleanWorkspace { workspace: birth }, 4)
            .unwrap();
        let refused = log.execute_system(Command::CleanWorkspace { workspace: birth }, 5);
        assert!(matches!(refused, Err(Reject::WorkspaceStageMismatch)));

        // a fresh workspace stands Available: restoration is idempotent
        // by postcondition - the fact lands and the stage stays what it
        // already gives, with the latest fact named on the lifecycle
        log.execute_system(
            created("workspace-deadbeef", "/repo/wt/workspace-deadbeef"),
            6,
        )
        .unwrap();
        log.execute_system(
            Command::RestoreWorkspace {
                workspace: RecordId(3),
            },
            7,
        )
        .unwrap();
        assert_eq!(
            log.world().workspace_stage(RecordId(3)),
            Some(crate::WorkspaceStage::Available)
        );
        // the latest fact's provenance lives on the context
        assert_eq!(
            log.world().workspaces[&crate::WorkspaceId(RecordId(3))]
                .stage_fact
                .as_ref()
                .map(|fact| fact.record),
            Some(RecordId(4))
        );

        // judgment tiers refuse the door: the completed facts are the
        // machinery's, never caller-authored
        for ctx in [human(), agent()] {
            let refused = log.execute(
                ctx.clone(),
                Command::CleanWorkspace {
                    workspace: RecordId(3),
                },
                8,
            );
            assert!(matches!(refused, Err(Reject::HumanOnly)));
            let refused = log.execute(ctx, Command::RestoreWorkspace { workspace: birth }, 8);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        assert_eq!(log.records().len(), 5);
    }

    /// The confirmed change names a native incarnation and lands as a
    /// fact: the cache holds the latest, the log keeps every one, and
    /// a historical attribution identity is no change target.
    #[test]
    fn a_confirmed_change_rides_a_native_incarnation() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        let to = |path: &str| Command::ChangeWorkspace {
            incarnation: IncarnationId(RecordId(0)),
            worktree: WorktreePath::new(path.into()).unwrap(),
        };
        log.execute_system(to("/repo/wt/feature-abc123ab"), 2)
            .unwrap();
        let first = &log.world().workspace_changes[&IncarnationId(RecordId(0))];
        assert_eq!(first.record, RecordId(1));
        assert_eq!(
            first.worktree,
            WorktreePath::new("/repo/wt/feature-abc123ab".into()).unwrap()
        );

        // a second change replaces the cache and keeps both records
        log.execute_system(to("/repo/wt/workspace-deadbeef"), 3)
            .unwrap();
        assert_eq!(log.records().len(), 3);
        let latest = &log.world().workspace_changes[&IncarnationId(RecordId(0))];
        assert_eq!(latest.record, RecordId(2));
        assert_eq!(
            latest.worktree,
            WorktreePath::new("/repo/wt/workspace-deadbeef".into()).unwrap()
        );

        // the change fact takes no task home
        assert_eq!(log.world().provenance[&RecordId(1)], None);
        assert_eq!(log.world().provenance[&RecordId(2)], None);

        // an unknown incarnation refuses: the fact needs its session
        let refused = log.execute_system(
            Command::ChangeWorkspace {
                incarnation: IncarnationId(RecordId(9)),
                worktree: WorktreePath::new("/repo/wt/nowhere".into()).unwrap(),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));

        // a historical attribution identity has no saved session to
        // move: the establishment birth is no change target
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("carry a merge through the worktree"),
            },
            5,
        )
        .unwrap();
        log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            6,
        )
        .unwrap();
        let establishment = log.records().len() - 1;
        let refused = log.execute_system(
            Command::ChangeWorkspace {
                incarnation: IncarnationId(RecordId(establishment)),
                worktree: WorktreePath::new("/repo/wt/elsewhere".into()).unwrap(),
            },
            7,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
        assert_eq!(log.world().workspace_changes.len(), 1);

        // the change verb rejects judgment tiers: it is machinery's
        // post-effect record, never a caller's claim
        for ctx in [human(), agent()] {
            let refused = log.execute(ctx, to("/repo/wt/rejected"), 8);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        assert_eq!(log.records().len(), 5);
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

    // -- the native communication contracts --------------------------

    use crate::NativeSessionId;
    use crate::SessionLocator;
    use crate::harness::Harness;
    use crate::objects::message::MessageState;

    fn registered(id: &str) -> Command {
        Command::RegisterIncarnation {
            native_id: NativeSessionId::new(id.into()).unwrap(),
            saved_session: Some(
                SessionLocator::new(format!("/sessions/{id}.jsonl").into()).unwrap(),
            ),
            address: None,
            adopted: false,
            harness: Harness::Pi,
        }
    }

    /// The session a registered native incarnation sends and waits as.
    fn native_session(birth: usize) -> Context {
        Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope: Some(Scope {
                task: None,
                incarnation: Some(IncarnationId(RecordId(birth))),
                workspace: None,
            }),
        }
    }

    fn message(from: usize, to: usize, reply_to: Option<usize>) -> Command {
        Command::SendMessage {
            from: IncarnationId(RecordId(from)),
            to: IncarnationId(RecordId(to)),
            reply_to: reply_to.map(|r| crate::objects::message::MessageId(RecordId(r))),
            task: None,
            body: prose("please look at the failing edge case"),
        }
    }

    /// A task-free registration births a native identity with no home:
    /// the record exists, the provenance index holds None, and the
    /// same native session never registers twice.
    #[test]
    fn registration_births_a_homeless_native_identity_once() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        log.execute_system(registered("bbbb2222"), 2).unwrap();

        let world = log.world();
        assert_eq!(world.incarnations.len(), 2);
        let birth = RecordId(0);
        assert_eq!(world.provenance[&birth], None);
        assert_eq!(
            world.incarnations[&IncarnationId(birth)].incarnation,
            Incarnation::Native {
                native_id: NativeSessionId::new("aaaa1111".into()).unwrap(),
                saved_session: Some(
                    SessionLocator::new("/sessions/aaaa1111.jsonl".into()).unwrap()
                ),
                address: None,
                adopted: false,
                harness: Harness::Pi,
            }
        );

        // the same native session resolves at the effect; a second
        // birth refuses the fold
        let refused = log.execute_system(registered("aaaa1111"), 3);
        assert!(matches!(refused, Err(Reject::DuplicateNativeSession)));
        assert_eq!(log.records().len(), 2);

        // a reused native id with a different saved locator is not the
        // recorded association: it refuses, it never rebinds history
        let relocated = Command::RegisterIncarnation {
            native_id: NativeSessionId::new("aaaa1111".into()).unwrap(),
            saved_session: Some(SessionLocator::new("/elsewhere/aaaa1111.jsonl".into()).unwrap()),
            address: None,
            adopted: false,
            harness: Harness::Pi,
        };
        let refused = log.execute_system(relocated, 3);
        assert!(matches!(refused, Err(Reject::SavedSessionMismatch)));
        assert_eq!(log.records().len(), 2);

        // the registration verb rejects judgment tiers
        for ctx in [human(), agent()] {
            let refused = log.execute(ctx, registered("cccc3333"), 3);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }

        // the launch-door birth: an unbound locator and the opaque
        // host address ride the birth itself
        log.execute_system(
            Command::RegisterIncarnation {
                native_id: NativeSessionId::new("door-dddd".into()).unwrap(),
                saved_session: None,
                address: Some("/tty5".into()),
                adopted: false,
                harness: Harness::Pi,
            },
            3,
        )
        .unwrap();
        assert_eq!(
            log.world().incarnations[&IncarnationId(RecordId(2))].incarnation,
            Incarnation::Native {
                native_id: NativeSessionId::new("door-dddd".into()).unwrap(),
                saved_session: None,
                address: Some("/tty5".into()),
                adopted: false,
                harness: Harness::Pi,
            }
        );
    }

    /// Identity is interpreted within its harness: the same native id
    /// under another harness is another session, and the exact
    /// association - harness, id, locator - guards uniqueness per
    /// harness.
    #[test]
    fn the_same_native_id_under_another_harness_is_another_session() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        let claude_code = Command::RegisterIncarnation {
            native_id: NativeSessionId::new("aaaa1111".into()).unwrap(),
            saved_session: Some(SessionLocator::new("/sessions/aaaa1111.jsonl".into()).unwrap()),
            address: None,
            adopted: false,
            harness: Harness::ClaudeCode,
        };
        // the pi registration never matches a claude-code lookup: the
        // same id births again under its own harness
        log.execute_system(claude_code.clone(), 2).unwrap();
        assert_eq!(log.world().incarnations.len(), 2);
        assert_eq!(
            log.world().incarnations[&IncarnationId(RecordId(1))]
                .incarnation
                .harness(),
            Some(Harness::ClaudeCode)
        );

        // the exact claude-code association resolves once, never twice
        let refused = log.execute_system(claude_code, 3);
        assert!(matches!(refused, Err(Reject::DuplicateNativeSession)));

        // a reused claude-code id with another locator never rebinds
        let relocated = Command::RegisterIncarnation {
            native_id: NativeSessionId::new("aaaa1111".into()).unwrap(),
            saved_session: Some(SessionLocator::new("/elsewhere/aaaa1111.jsonl".into()).unwrap()),
            address: None,
            adopted: false,
            harness: Harness::ClaudeCode,
        };
        let refused = log.execute_system(relocated, 3);
        assert!(matches!(refused, Err(Reject::SavedSessionMismatch)));
        assert_eq!(log.records().len(), 2);
    }

    /// The attach completes a door-birth exactly once: the locator
    /// binds onto an unbound native incarnation, never over a bound
    /// one, and never onto a non-incarnation record.
    #[test]
    fn the_locator_binds_an_unbound_birth_once() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        log.execute_system(
            Command::RegisterIncarnation {
                native_id: NativeSessionId::new("door-dddd".into()).unwrap(),
                saved_session: None,
                address: Some("/tty5".into()),
                adopted: false,
                harness: Harness::Pi,
            },
            2,
        )
        .unwrap();
        let unbound = IncarnationId(RecordId(1));

        // the binding is machinery's act, like the birth it completes
        for ctx in [human(), agent()] {
            let refused = log.execute(
                ctx,
                Command::BindSavedSession {
                    incarnation: unbound,
                    saved_session: SessionLocator::new("/sessions/door.jsonl".into()).unwrap(),
                },
                3,
            );
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        log.execute_system(
            Command::BindSavedSession {
                incarnation: unbound,
                saved_session: SessionLocator::new("/sessions/door.jsonl".into()).unwrap(),
            },
            3,
        )
        .unwrap();
        assert_eq!(
            log.world().incarnations[&unbound]
                .incarnation
                .saved_session(),
            Some(&SessionLocator::new("/sessions/door.jsonl".into()).unwrap())
        );

        // a second bind refuses: the attach binds once, never overwrites
        let refused = log.execute_system(
            Command::BindSavedSession {
                incarnation: unbound,
                saved_session: SessionLocator::new("/sessions/other.jsonl".into()).unwrap(),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::LocatorAlreadyBound)));

        // an extension-registered birth already carries its locator:
        // the attach resolves it, it never rebinds
        let bound = IncarnationId(RecordId(0));
        let refused = log.execute_system(
            Command::BindSavedSession {
                incarnation: bound,
                saved_session: SessionLocator::new("/sessions/aaaa1111-b.jsonl".into()).unwrap(),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::LocatorAlreadyBound)));

        // no such incarnation: the bind names a birth the record never
        // carried
        let refused = log.execute_system(
            Command::BindSavedSession {
                incarnation: IncarnationId(RecordId(99)),
                saved_session: SessionLocator::new("/sessions/none.jsonl".into()).unwrap(),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
        // an established identity is no bind target
        let refused = log.execute_system(
            Command::BindSavedSession {
                incarnation: IncarnationId(RecordId(50)),
                saved_session: SessionLocator::new("/sessions/none.jsonl".into()).unwrap(),
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
    }

    /// The association's address moves with a relaunch: the same
    /// incarnation, another terminal, recorded by the machinery.
    #[test]
    fn the_address_moves_with_a_relaunch() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        let id = IncarnationId(RecordId(0));

        for ctx in [human(), agent()] {
            let refused = log.execute(
                ctx,
                Command::AssignHostAddress {
                    incarnation: id,
                    address: Some("/tty7".into()),
                    adopted: false,
                },
                2,
            );
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        log.execute_system(
            Command::AssignHostAddress {
                incarnation: id,
                address: Some("/tty7".into()),
                adopted: false,
            },
            2,
        )
        .unwrap();
        assert_eq!(
            log.world().incarnations[&id].incarnation.address(),
            Some("/tty7")
        );
        log.execute_system(
            Command::AssignHostAddress {
                incarnation: id,
                address: Some("/tty9".into()),
                adopted: false,
            },
            3,
        )
        .unwrap();
        assert_eq!(
            log.world().incarnations[&id].incarnation.address(),
            Some("/tty9")
        );
        // a birth the record never carried is no assignment target
        let refused = log.execute_system(
            Command::AssignHostAddress {
                incarnation: IncarnationId(RecordId(9)),
                address: Some("/tty1".into()),
                adopted: false,
            },
            4,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
    }

    /// A message's guards: the sender is the session's own registered
    /// native identity; the recipient, reply, and task references all
    /// resolve; the message takes no thread home.
    #[test]
    fn messages_guard_their_references_and_take_no_home() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        log.execute_system(registered("bbbb2222"), 2).unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("implement foo"),
            },
            3,
        )
        .unwrap();

        // a scoped registered sender delivers
        log.execute(native_session(0), message(0, 1, None), 4)
            .unwrap();
        assert_eq!(log.world().messages.len(), 1);
        let mctx = &log.world().messages[&crate::objects::message::MessageId(RecordId(3))];
        assert_eq!(mctx.state, MessageState::Pending);
        assert_eq!(log.world().provenance[&RecordId(3)], None);

        // an unscoped caller claims no sender identity
        let refused = log.execute(agent(), message(0, 1, None), 5);
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
        // the declared identity is the sender: another session's birth refuses
        let refused = log.execute(native_session(1), message(0, 1, None), 5);
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
        // a historical establishment is no message sender
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("migrate floop"),
            },
            5,
        )
        .unwrap();
        log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(1),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-1".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-1".into()).unwrap(),
            },
            6,
        )
        .unwrap();
        let establishment = log.records().len() - 1;
        let refused = log.execute(
            native_session(establishment),
            message(establishment, 0, None),
            7,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));

        // the recipient must exist
        let refused = log.execute(native_session(0), message(0, 9, None), 7);
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
        // the recipient must be a registered native session: a
        // historical establishment has no boundary mail could reach
        let refused = log.execute(native_session(0), message(0, establishment, None), 7);
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
        // a session may address itself: delivery filters on the
        // addressee, never on a no-self authority rule
        log.execute(native_session(0), message(0, 0, None), 7)
            .unwrap();
        let self_mail = &log.world().messages[&crate::objects::message::MessageId(RecordId(6))];
        assert_eq!(self_mail.message.from, self_mail.message.to);
        assert_eq!(self_mail.state, MessageState::Pending);
        // the reply reference must exist
        let refused = log.execute(native_session(1), message(1, 0, Some(99)), 7);
        assert!(matches!(refused, Err(Reject::InvalidMessageId)));
        // the task context must exist
        let refused = log.execute(
            native_session(0),
            Command::SendMessage {
                from: IncarnationId(RecordId(0)),
                to: IncarnationId(RecordId(1)),
                reply_to: None,
                task: Some(TaskId(9)),
                body: prose("context on nothing"),
            },
            7,
        );
        assert!(matches!(refused, Err(Reject::InvalidTaskId)));
        assert_eq!(log.records().len(), 7);
    }

    /// The reply chain and the delivery record: a reply names an
    /// existing message, submission is the addressee's machinery act,
    /// and submission never repeats.
    #[test]
    fn replies_and_submissions_bookkeep_the_delivery() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        log.execute_system(registered("bbbb2222"), 2).unwrap();
        log.execute(native_session(0), message(0, 1, None), 3)
            .unwrap();
        // the recipient answers: the reply names the message it answers
        log.execute(native_session(1), message(1, 0, Some(2)), 4)
            .unwrap();

        // submission is the addressee's act, machinery-recorded
        log.execute_system(
            Command::SubmitMessage {
                id: crate::objects::message::MessageId(RecordId(2)),
                by: IncarnationId(RecordId(1)),
            },
            5,
        )
        .unwrap();
        let mctx = &log.world().messages[&crate::objects::message::MessageId(RecordId(2))];
        assert_eq!(mctx.state, MessageState::Submitted);
        assert_eq!(log.world().provenance[&RecordId(4)], None);

        // the sender's own mail is not the recipient's to submit
        let refused = log.execute_system(
            Command::SubmitMessage {
                id: crate::objects::message::MessageId(RecordId(2)),
                by: IncarnationId(RecordId(0)),
            },
            6,
        );
        assert!(matches!(refused, Err(Reject::InvalidMessageId)));
        // an unknown message refuses
        let refused = log.execute_system(
            Command::SubmitMessage {
                id: crate::objects::message::MessageId(RecordId(99)),
                by: IncarnationId(RecordId(1)),
            },
            6,
        );
        assert!(matches!(refused, Err(Reject::InvalidMessageId)));
        // submission never repeats: the table holds on submitted mail
        let refused = log.execute_system(
            Command::SubmitMessage {
                id: crate::objects::message::MessageId(RecordId(2)),
                by: IncarnationId(RecordId(1)),
            },
            6,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        // judgment tiers refuse the delivery record at the authority table
        for ctx in [human(), agent()] {
            let refused = log.execute(
                ctx,
                Command::SubmitMessage {
                    id: crate::objects::message::MessageId(RecordId(3)),
                    by: IncarnationId(RecordId(0)),
                },
                7,
            );
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        assert_eq!(log.records().len(), 5);
    }

    /// A task-free scope accepts what it did not produce: the
    /// self-acceptance gate reads the task binding alone.
    #[test]
    fn a_task_free_scope_accepts_freely() {
        let mut log = Log::new();
        log.execute_system(registered("aaaa1111"), 1).unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("implement foo"),
            },
            2,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::CommitBoard {
                task: TaskId(0),
                commit: GitCommit::new("a".repeat(40)).unwrap(),
                parent: None,
                incorporations: Vec::new(),
            },
            3,
        )
        .unwrap();
        log.execute(
            agent(),
            Command::OfferBoard {
                task: TaskId(0),
                commit: GitCommit::new("a".repeat(40)).unwrap(),
            },
            4,
        )
        .unwrap();

        // the native session declares no task: its scope covers nothing
        let mut session = native_session(0);
        session.actor = agent().actor;
        log.execute(session, Command::AcceptBoard { task: TaskId(0) }, 5)
            .unwrap();
        assert_eq!(log.world().tasks[0].task.state, TaskState::Concluded);

        // the task-scoped producer still cannot accept its own board
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("fix bar"),
            },
            6,
        )
        .unwrap();
        let scoped = scoped_agent(1);
        log.execute(
            scoped.clone(),
            Command::CommitBoard {
                task: TaskId(1),
                commit: GitCommit::new("b".repeat(40)).unwrap(),
                parent: None,
                incorporations: Vec::new(),
            },
            7,
        )
        .unwrap();
        log.execute(
            scoped.clone(),
            Command::OfferBoard {
                task: TaskId(1),
                commit: GitCommit::new("b".repeat(40)).unwrap(),
            },
            8,
        )
        .unwrap();
        let refused = log.execute(scoped, Command::AcceptBoard { task: TaskId(1) }, 9);
        assert!(matches!(refused, Err(Reject::SelfAcceptance)));
    }

    /// A native identity is task-free: it claims through the explicit
    /// door alone, and never inherits an establishment's task.
    #[test]
    fn native_identities_claim_through_the_explicit_door() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("carry a merge through the worktree"),
            },
            1,
        )
        .unwrap();
        log.execute_system(registered("aaaa1111"), 2).unwrap();
        log.execute_system(registered("bbbb2222"), 3).unwrap();
        log.execute_system(
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            4,
        )
        .unwrap();
        let establishment = log.records().len() - 1;

        // a native identity claims any task that exists
        log.execute(
            native_session(1),
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: Some(IncarnationId(RecordId(1))),
            },
            5,
        )
        .unwrap();
        assert!(matches!(
            log.world().tasks[0].ownership[0].owner,
            Owner::Incarnation(IncarnationId(RecordId(1)))
        ));

        // the historical establishment is bound to its birth task
        let refused = log.execute(
            human(),
            Command::ClaimTask {
                id: TaskId(9),
                incarnation: Some(IncarnationId(RecordId(establishment))),
            },
            6,
        );
        assert!(matches!(refused, Err(Reject::InvalidIncarnationId)));
    }

    #[test]
    fn a_commit_guards_its_parent_its_stage_and_its_sources() {
        let mut log = Log::new();
        populate_log(&mut log);

        // the task stands concluded after the acceptance: a commit refuses
        let refused = log.execute(
            agent(),
            Command::CommitBoard {
                task: TaskId(0),
                commit: GitCommit::new("f".repeat(40)).unwrap(),
                parent: Some(
                    GitCommit::new("02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()).unwrap(),
                ),
                incorporations: Vec::new(),
            },
            99,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        // a parent that is not the task's current commit refuses
        let mut fresh = Log::new();
        fresh
            .execute(
                human(),
                Command::CreateTask {
                    name: prose("implement foo"),
                },
                1,
            )
            .unwrap();
        fresh
            .execute(
                agent(),
                Command::CommitBoard {
                    task: TaskId(0),
                    commit: GitCommit::new("a".repeat(40)).unwrap(),
                    parent: None,
                    incorporations: Vec::new(),
                },
                2,
            )
            .unwrap();
        let refused = fresh.execute(
            agent(),
            Command::CommitBoard {
                task: TaskId(0),
                commit: GitCommit::new("b".repeat(40)).unwrap(),
                parent: None,
                incorporations: Vec::new(),
            },
            3,
        );
        match refused {
            Err(Reject::NotCurrentParent { expected }) => {
                assert_eq!(
                    expected.as_deref(),
                    Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                );
            }
            other => panic!("expected the parent guard, got {other:?}"),
        }
    }

    #[test]
    fn an_accept_guards_the_stage_the_scope_and_the_sources() {
        let mut log = Log::new();
        populate_log(&mut log);

        // bar stands offered; the accept lands the conclusion
        assert_eq!(log.world().tasks[1].task.state, TaskState::Offered);
        log.execute(human(), Command::AcceptBoard { task: TaskId(1) }, 99)
            .unwrap();
        assert_eq!(log.world().tasks[1].task.state, TaskState::Concluded);

        // a scoped agent whose declared task covers the offered board
        // cannot accept it
        let mut scoped = Log::new();
        scoped
            .execute(
                human(),
                Command::CreateTask {
                    name: prose("implement foo"),
                },
                1,
            )
            .unwrap();
        scoped
            .execute(
                agent(),
                Command::CommitBoard {
                    task: TaskId(0),
                    commit: GitCommit::new("a".repeat(40)).unwrap(),
                    parent: None,
                    incorporations: Vec::new(),
                },
                2,
            )
            .unwrap();
        scoped
            .execute(
                agent(),
                Command::OfferBoard {
                    task: TaskId(0),
                    commit: GitCommit::new("a".repeat(40)).unwrap(),
                },
                3,
            )
            .unwrap();
        let refused = scoped.execute(
            Context {
                actor: ActorName::new("pi".into()).unwrap(),
                tier: Tier::Agent,
                scope: Some(Scope {
                    task: Some(TaskId(0)),
                    incarnation: None,
                    workspace: None,
                }),
            },
            Command::AcceptBoard { task: TaskId(0) },
            4,
        );
        assert!(matches!(refused, Err(Reject::SelfAcceptance)));

        // an incorporation citing a source commit that is not the
        // source's accepted final commit refuses
        let mut stale = Log::new();
        stale
            .execute(
                human(),
                Command::CreateTask {
                    name: prose("implement foo"),
                },
                1,
            )
            .unwrap();
        stale
            .execute(
                agent(),
                Command::CommitBoard {
                    task: TaskId(0),
                    commit: GitCommit::new("a".repeat(40)).unwrap(),
                    parent: None,
                    incorporations: Vec::new(),
                },
                2,
            )
            .unwrap();
        stale
            .execute(
                agent(),
                Command::OfferBoard {
                    task: TaskId(0),
                    commit: GitCommit::new("a".repeat(40)).unwrap(),
                },
                3,
            )
            .unwrap();
        // the consumer commits citing foo's r1 (recorded), offers, and
        // the accept refuses: r1 is not foo's accepted final commit -
        // foo never accepted at all
        stale
            .execute(
                human(),
                Command::CreateTask {
                    name: prose("consume foo"),
                },
                5,
            )
            .unwrap();
        stale
            .execute(
                agent(),
                Command::CommitBoard {
                    task: TaskId(1),
                    commit: GitCommit::new("c".repeat(40)).unwrap(),
                    parent: None,
                    incorporations: vec![(TaskId(0), GitCommit::new("a".repeat(40)).unwrap())],
                },
                6,
            )
            .unwrap();
        stale
            .execute(
                agent(),
                Command::OfferBoard {
                    task: TaskId(1),
                    commit: GitCommit::new("c".repeat(40)).unwrap(),
                },
                7,
            )
            .unwrap();
        let refused = stale.execute(human(), Command::AcceptBoard { task: TaskId(1) }, 8);
        assert!(matches!(refused, Err(Reject::SourceNotAccepted { .. })));

        // a consumer commit citing a commit never recorded on its
        // source task refuses at the commit guard
        stale
            .execute(
                human(),
                Command::CreateTask {
                    name: prose("consume again"),
                },
                9,
            )
            .unwrap();
        let refused = stale.execute(
            agent(),
            Command::CommitBoard {
                task: TaskId(2),
                commit: GitCommit::new("d".repeat(40)).unwrap(),
                parent: None,
                incorporations: vec![(TaskId(0), GitCommit::new("f".repeat(40)).unwrap())],
            },
            10,
        );
        assert!(matches!(
            refused,
            Err(Reject::UnrecordedSourceCommit { .. })
        ));
    }
}
