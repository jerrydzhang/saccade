use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Reject;
use crate::decide::{decide, enforce_tier};
use crate::events::{Command, Event};
use crate::objects::incarnation::{Incarnation, IncarnationContext, IncarnationId};
use crate::objects::message::{Message, MessageContext, MessageId, MessageState};
use crate::objects::task::{Owner, OwnerEntry, Task, TaskContext, TaskId, TaskState};
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
