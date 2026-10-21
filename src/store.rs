use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Reject;
use crate::decide::{decide, enforce_tier};
use crate::events::{Command, Event};
use crate::objects::comment::{
    Binding, Comment, CommentContext, CommentId, CommentKind, CommentState, Revision, Withdrawal,
};
use crate::objects::incarnation::{Incarnation, IncarnationContext, IncarnationId};
use crate::objects::message::{Message, MessageContext, MessageId, MessageState};
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
    /// A citation names a record that is not an earlier birth: stable
    /// internal references identify birth records.
    InvalidReference,
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
            Reason::DuplicateNativeSession => Reject::DuplicateNativeSession,
            Reason::SavedSessionMismatch => Reject::SavedSessionMismatch,
            Reason::InvalidReference => Reject::InvalidReference,
            Reason::InvalidMessageId => Reject::InvalidMessageId,
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
    /// Record-derived identities: historical establishment births and
    /// registered native sessions, in birth order
    pub incarnations: BTreeMap<IncarnationId, IncarnationContext>,
    /// Durable addressed messages, in birth order
    pub messages: BTreeMap<MessageId, MessageContext>,
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
            comments: BTreeMap::new(),
            incarnations: BTreeMap::new(),
            messages: BTreeMap::new(),
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

    /// The comment's public number: its position in comment birth
    /// order, the number the human-facing c-N carries. Births do not
    /// carry their own ids; the fold derives them.
    pub fn comment_num(&self, id: CommentId) -> Option<usize> {
        if !self.comments.contains_key(&id) {
            return None;
        }
        Some(self.comments.range(..id).count())
    }

    /// The comment holding this public number, resolved from the fold.
    pub fn comment_at_num(&self, n: usize) -> Option<CommentId> {
        self.comments.keys().nth(n).copied()
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

    /// The eligibility a binding preserves at the moment it lands: the
    /// offer named is an Offer of the use's declared source task and
    /// that task's standing Offered or Concluded account. A bound use
    /// keeps its selection when that offer is later rejected or
    /// replaced - nothing here follows the source forward.
    fn check_binding(&self, source: TaskId, offer: CommentId) -> Result<(), Reason> {
        let offer_ctx = self.comments.get(&offer).ok_or(Reason::InvalidCommentId)?;
        if !matches!(offer_ctx.comment.kind, CommentKind::Offer { .. }) {
            return Err(Reason::NotAnOffer);
        }
        if offer_ctx.comment.home != source {
            return Err(Reason::InvalidStateTransition);
        }
        let src = self.tasks.get(source.0).ok_or(Reason::InvalidTaskId)?;
        if src.current_offer != Some(offer)
            || !matches!(src.task.state, TaskState::Offered | TaskState::Concluded)
        {
            return Err(Reason::NotCurrentOffer);
        }
        Ok(())
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
                if matches!(record.event, Event::TaskReopened { .. }) {
                    task_ctx.current_offer = None;
                    task_ctx.offered_at = None;
                }
            }
            // Comment events
            Event::Commented {
                home,
                ref kind,
                ref references,
            } => {
                if self.tasks.get(home.0).is_none() {
                    return Err(Reason::InvalidTaskId);
                }
                for rid in references {
                    if rid.0 >= record.id.0 || !self.provenance.contains_key(rid) {
                        return Err(Reason::InvalidReference);
                    }
                }
                match kind {
                    CommentKind::Note { .. } | CommentKind::Offer { .. } => {}
                    CommentKind::Accept { offer } | CommentKind::Reject { offer, .. } => {
                        // a decision lives on the task of the offer it names
                        let offer_ctx = self.comments.get(offer).ok_or(Reason::InvalidCommentId)?;
                        if !matches!(offer_ctx.comment.kind, CommentKind::Offer { .. }) {
                            return Err(Reason::NotAnOffer);
                        }
                        if offer_ctx.comment.home != home {
                            return Err(Reason::InvalidStateTransition);
                        }
                        if matches!(kind, CommentKind::Accept { .. })
                            && record.context.tier == Tier::Agent
                            && record
                                .context
                                .scope
                                .as_ref()
                                .and_then(|scope| scope.task)
                                .is_some_and(|scoped| scoped == offer_ctx.comment.home)
                        {
                            return Err(Reason::SelfAcceptance);
                        }
                        let offered = self
                            .tasks
                            .get(offer_ctx.comment.home.0)
                            .ok_or(Reason::InvalidTaskId)?;
                        if offered.current_offer != Some(*offer) {
                            return Err(Reason::NotCurrentOffer);
                        }
                    }
                    CommentKind::Use { source, bound, .. } => {
                        if *source == home {
                            return Err(Reason::SelfReliance);
                        }
                        if self.tasks.get(source.0).is_none() {
                            return Err(Reason::InvalidTaskId);
                        }
                        // a birth may carry its initial binding - the
                        // shape a historical log migrates into - under
                        // the same eligibility a later binding preserves
                        if let Some(binding) = bound {
                            self.check_binding(*source, binding.offer)?;
                        }
                    }
                }

                let binding = match kind {
                    CommentKind::Use { bound: Some(_), .. } => Some(Binding {
                        record: record.id,
                        actor: record.context.actor.clone(),
                    }),
                    _ => None,
                };

                self.provenance.insert(record.id, Some(home));
                let task_ctx = self.tasks.get_mut(home.0).ok_or(Reason::InvalidTaskId)?;
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

                self.comments.insert(
                    CommentId(record.id),
                    CommentContext {
                        comment: Comment {
                            home,
                            kind: kind.clone(),
                            state: CommentState::of(kind),
                        },
                        actor: record.context.actor.clone(),
                        tier: record.context.tier,
                        scope: record.context.scope.clone(),
                        born_at: record.timestamp,
                        revised: None,
                        withdrawn: None,
                        binding,
                        references: references.clone(),
                    },
                );
            }
            Event::CommentRevised { id, ref body } => {
                let home = self
                    .comments
                    .get(&id)
                    .ok_or(Reason::InvalidCommentId)?
                    .comment
                    .home;
                self.provenance.insert(record.id, Some(home));
                let comment_ctx = self.comments.get_mut(&id).ok_or(Reason::InvalidCommentId)?;
                // the gate is the state: a withdrawn deposit admits no repair
                if matches!(comment_ctx.comment.state, CommentState::Withdrawn(_)) {
                    return Err(Reason::InvalidStateTransition);
                }
                // the bodyless Accept carries no prose to repair
                match &mut comment_ctx.comment.kind {
                    CommentKind::Note { body: prose }
                    | CommentKind::Offer { body: prose }
                    | CommentKind::Reject { grounds: prose, .. }
                    | CommentKind::Use { reason: prose, .. } => *prose = body.clone(),
                    CommentKind::Accept { .. } => return Err(Reason::InvalidStateTransition),
                }
                comment_ctx.revised = Some(Revision {
                    record: record.id,
                    reviser: record.context.actor.clone(),
                });
                let home = comment_ctx.comment.home;
                let task_ctx = self.tasks.get_mut(home.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_record_at = record.timestamp;
            }
            ref event @ Event::CommentWithdrawn { id, .. } => {
                let home = self
                    .comments
                    .get(&id)
                    .ok_or(Reason::InvalidCommentId)?
                    .comment
                    .home;
                self.provenance.insert(record.id, Some(home));
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
                let home = comment_ctx.comment.home;
                let task_ctx = self.tasks.get_mut(home.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::UseBound { id, offer, .. } => {
                let comment_ctx = self.comments.get(&id).ok_or(Reason::InvalidCommentId)?;
                // binding is the use's door: every other variant holds
                let CommentKind::Use { source, .. } = &comment_ctx.comment.kind else {
                    return Err(Reason::InvalidStateTransition);
                };
                let source = *source;
                let home = comment_ctx.comment.home;
                self.check_binding(source, offer)?;
                self.provenance.insert(record.id, Some(home));
                let comment_ctx = self.comments.get_mut(&id).expect("resident above");
                comment_ctx.comment.state = comment_ctx
                    .comment
                    .state
                    .transition(&record.event)
                    .ok_or(Reason::InvalidStateTransition)?;
                comment_ctx.binding = Some(Binding {
                    record: record.id,
                    actor: record.context.actor.clone(),
                });
                let task_ctx = self.tasks.get_mut(home.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_record_at = record.timestamp;
            }
            Event::ArtifactAdded { root, ref artifact } => {
                self.provenance.insert(record.id, Some(root));
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
                self.provenance.insert(record.id, Some(task_id));
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
            // Incarnation and message events
            Event::IncarnationRegistered {
                ref native_id,
                ref saved_session,
            } => {
                // registering an existing exact native session resolves
                // at the effect; the fold refuses the second birth, and
                // a reused id with a different locator never rebinds
                let known = self
                    .incarnations
                    .values()
                    .find(|ctx| ctx.incarnation.native_id() == Some(native_id));
                match known {
                    Some(ctx) if ctx.incarnation.saved_session() == Some(saved_session) => {
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
                        },
                        born_at: record.timestamp,
                    },
                );
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
    // The authoring ingress, after the tier gate: a live use is born
    // prospective (the born-bound shape is migrated history's), and
    // every marker resolves through its mapping. The fold replays
    // historical text unlinked.
    match &command {
        Command::Comment {
            kind: CommentKind::Use { bound: Some(_), .. },
            ..
        } => return Err(Reject::InvalidStateTransition),
        Command::Comment {
            kind, references, ..
        } => {
            if let Some(prose) = kind.prose()
                && crate::markers::first_unresolved(prose.as_str(), references.len()).is_some()
            {
                return Err(Reject::InvalidReference);
            }
        }
        Command::ReviseComment { id, body } => {
            if let Some(ctx) = world.comments.get(id)
                && ctx.comment.kind.prose().is_some()
                && !matches!(&ctx.comment.state, CommentState::Withdrawn(_))
                && crate::markers::first_unresolved(body.as_str(), ctx.references.len()).is_some()
            {
                return Err(Reject::InvalidReference);
            }
        }
        _ => {}
    }
    world.stage(base, context, now, &events)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Event;
    use crate::objects::comment::{CommentId, CommentKind, CommentState};
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
                task: Some(TaskId(task)),
                incarnation: None,
                workspace: None,
            }),
        }
    }

    fn prose(text: &str) -> Prose {
        Prose::new(text.into()).unwrap()
    }

    fn note(home: TaskId, body: &str) -> Command {
        Command::Comment {
            home,
            kind: CommentKind::Note { body: prose(body) },
            references: Vec::new(),
        }
    }

    /// A note citing the given births through its markers.
    fn citing_note(home: TaskId, body: &str, references: Vec<RecordId>) -> Command {
        Command::Comment {
            home,
            kind: CommentKind::Note { body: prose(body) },
            references,
        }
    }

    fn offer(id: TaskId, body: &str) -> Command {
        Command::Comment {
            home: id,
            kind: CommentKind::Offer { body: prose(body) },
            references: Vec::new(),
        }
    }

    fn accept_on(home: TaskId, offer: usize) -> Command {
        Command::Comment {
            home,
            kind: CommentKind::Accept {
                offer: CommentId(RecordId(offer)),
            },
            references: Vec::new(),
        }
    }

    fn reject_on(home: TaskId, offer: usize, grounds: &str) -> Command {
        Command::Comment {
            home,
            kind: CommentKind::Reject {
                offer: CommentId(RecordId(offer)),
                grounds: prose(grounds),
            },
            references: Vec::new(),
        }
    }

    fn use_of(home: TaskId, source: TaskId, reason: &str) -> Command {
        Command::Comment {
            home,
            kind: CommentKind::Use {
                source,
                reason: prose(reason),
                bound: None,
            },
            references: Vec::new(),
        }
    }

    fn bind(id: usize, offer: usize, grounds: &str) -> Command {
        Command::BindUse {
            id: CommentId(RecordId(id)),
            offer: CommentId(RecordId(offer)),
            grounds: prose(grounds),
        }
    }

    const RECORD_COUNT: usize = 27;

    /// foo accepted; bar re-offered and accepted; floop reopened; a
    /// consumer cites foo, uses foo, and follows bar's rework through a
    /// replacement conclusion.
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
            note(TaskId(0), "mid-flight finding parked"),
            2,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            offer(TaskId(0), "suite green, 55 tests"),
            3,
        )
        .unwrap();
        log.execute(human_ctx.clone(), accept_on(TaskId(0), 3), 4)
            .unwrap();

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
        log.execute(
            human_ctx.clone(),
            reject_on(TaskId(1), 7, "the edge case still fails"),
            8,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            offer(TaskId(1), "edge case held, suite green"),
            9,
        )
        .unwrap();
        log.execute(human_ctx.clone(), accept_on(TaskId(1), 9), 10)
            .unwrap();

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
            note(TaskId(2), "the migration waits on a decision"),
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
                name: prose("adopt the accounts"),
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
        // the prospective use names its source task; the conclusion it
        // rests on is a separate explicit binding
        log.execute(
            agent_ctx.clone(),
            use_of(TaskId(3), TaskId(0), "builds on foo's landed account"),
            17,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            bind(17, 3, "foo's accepted account is the one that landed"),
            18,
        )
        .unwrap();
        // a citation: marker [^1] names foo's account, same target twice
        log.execute(
            agent_ctx.clone(),
            citing_note(
                TaskId(3),
                "rests on the conclusion above [^1], again [^1]",
                vec![RecordId(3)],
            ),
            19,
        )
        .unwrap();

        log.execute(
            human_ctx.clone(),
            Command::CreateTask {
                name: prose("rewrite bar properly"),
            },
            20,
        )
        .unwrap();
        log.execute(agent_ctx.clone(), offer(TaskId(4), "the deeper fix"), 21)
            .unwrap();
        log.execute(
            agent_ctx.clone(),
            use_of(TaskId(3), TaskId(4), "bar's rework is the real foundation"),
            22,
        )
        .unwrap();
        log.execute(agent_ctx.clone(), bind(22, 21, "the standing account"), 23)
            .unwrap();
        // the source replaces its conclusion: the binding keeps it
        log.execute(
            human_ctx.clone(),
            reject_on(TaskId(4), 21, "the fix misses the index"),
            24,
        )
        .unwrap();
        log.execute(
            agent_ctx.clone(),
            offer(TaskId(4), "the fix that holds"),
            25,
        )
        .unwrap();
        // the explicit rebind selects the replacement
        log.execute(
            agent_ctx.clone(),
            bind(22, 25, "the replacement held under review"),
            26,
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
            Reject::from(Reason::DuplicateNativeSession),
            Reject::DuplicateNativeSession
        ));
        assert!(matches!(
            Reject::from(Reason::SavedSessionMismatch),
            Reject::SavedSessionMismatch
        ));
        assert!(matches!(
            Reject::from(Reason::InvalidReference),
            Reject::InvalidReference
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

    /// The story folds the ratified lifecycle, including the use's
    /// prospective birth, explicit binding, and explicit rebinding.
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
        assert_eq!(world.tasks[4].task.state, TaskState::Offered);
        // the reopen cleared the standing pointer
        assert_eq!(world.tasks[2].current_offer, None);
        assert_eq!(world.tasks[4].current_offer, Some(CommentId(RecordId(25))));

        // the create-reply query: births resolve, non-births don't
        assert_eq!(world.task_born_at(RecordId(0)), Some(TaskId(0)));
        assert_eq!(world.task_born_at(RecordId(2)), None);

        // public numbers derive from birth order: the comments in the
        // story are seq 2,3,4,7,8,9,10,13,17,19,21,22,24,25 - the use
        // born at 17 is the ninth comment, c-8 to a human
        assert_eq!(world.comment_at_num(8), Some(CommentId(RecordId(17))));
        assert_eq!(world.comment_num(CommentId(RecordId(17))), Some(8));
        assert_eq!(world.comment_at_num(world.comments.len()), None);

        // the use was born prospective, bound to foo's account explicitly
        let use_ctx = &world.comments[&CommentId(RecordId(17))];
        assert_eq!(use_ctx.comment.home, TaskId(3));
        assert_eq!(use_ctx.comment.kind.source(), Some(TaskId(0)));
        assert_eq!(
            use_ctx.comment.state.payload_offer(),
            Some(CommentId(RecordId(3)))
        );
        assert_eq!(
            use_ctx.binding.as_ref().map(|b| b.record),
            Some(RecordId(18))
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(17))].comment.state,
            CommentState::UseBound(crate::objects::comment::UseBinding {
                offer: CommentId(RecordId(3)),
                grounds: prose("foo's accepted account is the one that landed"),
            })
        );

        // the second use followed bar's rework: replaced, then rebound
        let rework = &world.comments[&CommentId(RecordId(22))];
        assert_eq!(
            rework.comment.state.payload_offer(),
            Some(CommentId(RecordId(25)))
        );
        assert_eq!(
            rework.binding.as_ref().map(|b| b.record),
            Some(RecordId(26))
        );

        // the citation note carries its mapping: one marker, one target
        let citing = &world.comments[&CommentId(RecordId(19))];
        assert!(matches!(
            &citing.comment.kind,
            CommentKind::Note { body } if body.as_str().contains("[^1]")
        ));
    }

    #[test]
    fn stale_decisions_are_refused_not_absorbed() {
        let mut log = Log::new();
        populate_log(&mut log);

        // t-1's first offer was rejected: neither decision reaches it
        for command in [
            accept_on(TaskId(1), 7),
            reject_on(TaskId(1), 7, "changed my mind again"),
        ] {
            let refused = log.execute(human(), command, 99);
            assert!(matches!(refused, Err(Reject::NotCurrentOffer)));
        }
        // a note is not an offer
        let refused = log.execute(human(), accept_on(TaskId(0), 2), 99);
        assert!(matches!(refused, Err(Reject::NotAnOffer)));
        assert_eq!(log.records().len(), RECORD_COUNT);

        // t-0 concluded on c-3: its own current offer no longer moves
        for command in [
            accept_on(TaskId(0), 3),
            reject_on(TaskId(0), 3, "a fault found later"),
        ] {
            let refused = log.execute(human(), command, 99);
            assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        }
        assert_eq!(log.records().len(), RECORD_COUNT);
    }

    #[test]
    fn declared_scope_refuses_only_the_self_acceptance() {
        let mut log = Log::new();
        populate_log(&mut log);

        // t-4 stands offered on c-25: the scoped producer cannot accept
        let scoped = scoped_agent(4);
        let refused = log.execute(scoped.clone(), accept_on(TaskId(4), 25), 99);
        assert!(matches!(refused, Err(Reject::SelfAcceptance)));

        // the same agent rejects on grounds: any participant may
        log.execute(
            scoped,
            reject_on(TaskId(4), 25, "the reason no longer holds"),
            27,
        )
        .unwrap();
        assert_eq!(log.world().tasks[4].task.state, TaskState::Open);

        // a scope on another task accepts freely; humans ignore scope
        log.execute(agent(), offer(TaskId(4), "the second account"), 28)
            .unwrap();
        log.execute(scoped_agent(2), accept_on(TaskId(4), 28), 29)
            .unwrap();
        let mut scoped_human = scoped_agent(3);
        scoped_human.tier = Tier::Human;
        log.execute(agent(), offer(TaskId(2), "the floop account"), 30)
            .unwrap();
        log.execute(scoped_human, accept_on(TaskId(2), 30), 32)
            .unwrap();
        assert_eq!(log.world().tasks[2].task.state, TaskState::Concluded);
        assert_eq!(log.world().tasks[4].task.state, TaskState::Concluded);
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

        // a comment never reopens a cancelled task
        log.execute(agent(), note(TaskId(3), "please resume the adoption"), 100)
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

    /// The use's guards: self-reliance and source existence at birth;
    /// the binding's eligibility at every selection. A bound use keeps
    /// its selection when the source replaces its conclusion.
    #[test]
    fn use_guards_its_source_and_binding() {
        let mut log = Log::new();
        populate_log(&mut log);

        // a use names its own task: refused
        let refused = log.execute(
            agent(),
            use_of(TaskId(3), TaskId(3), "relying on myself"),
            99,
        );
        assert!(matches!(refused, Err(Reject::SelfReliance)));

        // a use names an unborn task: refused
        let refused = log.execute(
            agent(),
            use_of(TaskId(3), TaskId(9), "relying on nothing"),
            100,
        );
        assert!(matches!(refused, Err(Reject::InvalidTaskId)));

        // a binding names a note: not an offer
        let refused = log.execute(agent(), bind(17, 2, "that is a note"), 101);
        assert!(matches!(refused, Err(Reject::NotAnOffer)));

        // a binding names another task's offer than the declared source
        let refused = log.execute(agent(), bind(17, 9, "bar's account is not foo's"), 102);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        // bar's rejected account stands on a task the use never named
        let refused = log.execute(agent(), bind(22, 7, "bar's rejected account"), 103);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        // binding a note is not the use's door
        let refused = log.execute(agent(), bind(2, 3, "a note binds nothing"), 104);
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        // a binding names an unknown comment
        let refused = log.execute(agent(), bind(99, 3, "no such use"), 105);
        assert!(matches!(refused, Err(Reject::InvalidCommentId)));

        // the replaced conclusion: rebinding to it refuses, and the
        // bound use kept it through the replacement
        let refused = log.execute(agent(), bind(22, 21, "the account bar replaced"), 106);
        assert!(matches!(refused, Err(Reject::NotCurrentOffer)));
        let rework = &log.world().comments[&CommentId(RecordId(22))];
        assert_eq!(
            rework.comment.state.payload_offer(),
            Some(CommentId(RecordId(25)))
        );
        assert_eq!(log.records().len(), RECORD_COUNT);
    }

    /// Source replacement alone leaves the binding; the explicit rebind
    /// selects the replacement; acceptance never invalidates it.
    #[test]
    fn a_bound_use_keeps_its_selection_through_source_replacement() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("consume the rework"),
            },
            1,
        )
        .unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("produce the rework"),
            },
            2,
        )
        .unwrap();
        log.execute(agent(), offer(TaskId(1), "the first account"), 3)
            .unwrap();
        log.execute(
            agent(),
            use_of(TaskId(0), TaskId(1), "builds on the rework"),
            4,
        )
        .unwrap();

        // pending: no selection yet
        assert_eq!(
            log.world().comments[&CommentId(RecordId(3))].comment.state,
            CommentState::Use
        );
        log.execute(agent(), bind(3, 2, "the standing account"), 5)
            .unwrap();

        // the source replaces its conclusion: the binding keeps it
        log.execute(
            human(),
            reject_on(TaskId(1), 2, "the fix misses the index"),
            6,
        )
        .unwrap();
        assert_eq!(
            log.world().comments[&CommentId(RecordId(3))]
                .comment
                .state
                .payload_offer(),
            Some(CommentId(RecordId(2)))
        );
        // rebinding to the replaced account refuses: not current
        let refused = log.execute(agent(), bind(3, 2, "gone"), 7);
        assert!(matches!(refused, Err(Reject::NotCurrentOffer)));

        log.execute(agent(), offer(TaskId(1), "the replacement"), 8)
            .unwrap();
        log.execute(agent(), bind(3, 6, "the replacement held under review"), 9)
            .unwrap();
        assert_eq!(
            log.world().comments[&CommentId(RecordId(3))]
                .comment
                .state
                .payload_offer(),
            Some(CommentId(RecordId(6)))
        );

        // acceptance of the selected conclusion changes nothing: no
        // follow, no cascade
        log.execute(human(), accept_on(TaskId(1), 6), 10).unwrap();
        assert_eq!(
            log.world().comments[&CommentId(RecordId(3))]
                .comment
                .state
                .payload_offer(),
            Some(CommentId(RecordId(6)))
        );
        assert_eq!(log.world().tasks[0].task.state, TaskState::Open);
    }

    /// The born-bound use replays as history and refuses at authoring.
    #[test]
    fn a_use_may_be_born_already_bound_in_history_alone() {
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("consume foo"),
            },
            1,
        )
        .unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("produce foo"),
            },
            2,
        )
        .unwrap();
        log.execute(agent(), offer(TaskId(1), "suite green"), 3)
            .unwrap();
        let record = Record {
            id: RecordId(3),
            timestamp: 4,
            context: agent(),
            event: Event::Commented {
                home: TaskId(0),
                kind: CommentKind::Use {
                    source: TaskId(1),
                    reason: prose("builds on foo's account"),
                    bound: Some(crate::objects::comment::UseBinding {
                        offer: CommentId(RecordId(2)),
                        grounds: prose("the account that landed"),
                    }),
                },
                references: Vec::new(),
            },
        };
        log.world = log
            .world
            .clone()
            .apply(record)
            .expect("history replays the born-bound shape");
        let use_ctx = &log.world().comments[&CommentId(RecordId(3))];
        assert_eq!(
            use_ctx.comment.state,
            CommentState::UseBound(crate::objects::comment::UseBinding {
                offer: CommentId(RecordId(2)),
                grounds: prose("the account that landed"),
            })
        );
        assert_eq!(
            use_ctx.binding.as_ref().map(|b| b.record),
            Some(RecordId(3))
        );

        // the same shape refuses at the authoring ingress, whatever
        // eligibility the world holds
        let refused = log.execute(
            agent(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Use {
                    source: TaskId(1),
                    reason: prose("builds on foo's account"),
                    bound: Some(crate::objects::comment::UseBinding {
                        offer: CommentId(RecordId(2)),
                        grounds: prose("the account that landed"),
                    }),
                },
                references: Vec::new(),
            },
            5,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        // and a born binding to a rejected account refuses the replay
        // too: history keeps the law it kept
        let mut log = Log::new();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("consume bar"),
            },
            1,
        )
        .unwrap();
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("produce bar"),
            },
            2,
        )
        .unwrap();
        log.execute(agent(), offer(TaskId(1), "the account"), 3)
            .unwrap();
        log.execute(human(), reject_on(TaskId(1), 2, "not good enough"), 4)
            .unwrap();
        let record = Record {
            id: RecordId(5),
            timestamp: 5,
            context: agent(),
            event: Event::Commented {
                home: TaskId(0),
                kind: CommentKind::Use {
                    source: TaskId(1),
                    reason: prose("builds on bar's account"),
                    bound: Some(crate::objects::comment::UseBinding {
                        offer: CommentId(RecordId(2)),
                        grounds: prose("the account that landed"),
                    }),
                },
                references: Vec::new(),
            },
        };
        let refused = log.world.clone().apply(record);
        assert!(matches!(
            refused.map_err(|e| e.reason),
            Err(Reason::NotCurrentOffer)
        ));
    }

    /// Markers resolve at authoring; code and math are literal; history replays unlinked.
    #[test]
    fn authoring_validates_markers_after_authority() {
        let mut log = Log::new();
        populate_log(&mut log);

        // a marker beyond the mapping refuses at the ingress
        let refused = log.execute(
            agent(),
            citing_note(TaskId(3), "an unresolvable [^2]", vec![RecordId(3)]),
            40,
        );
        assert!(matches!(refused, Err(Reject::InvalidReference)));
        // zero is no marker position either
        let refused = log.execute(
            agent(),
            citing_note(TaskId(3), "no zeroth [^0]", vec![RecordId(3)]),
            40,
        );
        assert!(matches!(refused, Err(Reject::InvalidReference)));
        // a number too large for an index refuses, never panics
        let oversized = "9".repeat(40);
        let refused = log.execute(
            agent(),
            citing_note(
                TaskId(3),
                &format!("oversized [^{oversized}]"),
                vec![RecordId(3)],
            ),
            40,
        );
        assert!(matches!(refused, Err(Reject::InvalidReference)));

        // code spans and math are literal: the body lands with no
        // mapping at all
        log.execute(
            agent(),
            note(TaskId(3), "the marker syntax is `[^1]` and $[^1]$ is math"),
            40,
        )
        .unwrap();

        // a revision keeps the mapping it was born with: the story's
        // citing note (one reference) takes [^1], never [^2]
        log.execute(
            agent(),
            Command::ReviseComment {
                id: CommentId(RecordId(19)),
                body: prose("reworded, still [^1]"),
            },
            41,
        )
        .unwrap();
        let refused = log.execute(
            agent(),
            Command::ReviseComment {
                id: CommentId(RecordId(19)),
                body: prose("retargeted [^2]"),
            },
            42,
        );
        assert!(matches!(refused, Err(Reject::InvalidReference)));

        // the fold replays what authoring refused: the oversized
        // marker lands in history and renders unresolved
        let len = log.records().len();
        let record = Record {
            id: RecordId(len),
            timestamp: 43,
            context: agent(),
            event: Event::Commented {
                home: TaskId(3),
                kind: CommentKind::Note {
                    body: prose(&format!("historical [^{oversized}]")),
                },
                references: vec![RecordId(3)],
            },
        };
        log.world = log
            .world
            .clone()
            .apply(record)
            .expect("history replays unlinked");
    }

    /// Citations name earlier applied records: births and record facts alike.
    #[test]
    fn citations_name_earlier_records_only() {
        let mut log = Log::new();
        populate_log(&mut log);

        // a comment birth, a task birth, an artifact birth, and the
        // record facts - a claim and a binding - all cite
        let artifact = Artifact {
            name: prose("sweep figure"),
            hash: ContentHash::of(b"figure bytes"),
        };
        log.execute(
            agent(),
            Command::Artifact {
                root: TaskId(3),
                artifact: artifact.clone(),
            },
            27,
        )
        .unwrap();
        let artifact_seq = log.records().len() - 1;
        log.execute(
            agent(),
            citing_note(
                TaskId(3),
                "foo's account [^1], the task [^2], the figure [^3], the claim [^4], the binding [^5]",
                vec![RecordId(3), RecordId(0), RecordId(artifact_seq), RecordId(1), RecordId(18)],
            ),
            28,
        )
        .unwrap();

        // a revision record is a record fact too
        log.execute(
            agent(),
            Command::ReviseComment {
                id: CommentId(RecordId(2)),
                body: prose("mid-flight finding parked, corrected"),
            },
            29,
        )
        .unwrap();
        let revision_seq = log.records().len() - 1;
        log.execute(
            agent(),
            citing_note(TaskId(3), "the repair [^1]", vec![RecordId(revision_seq)]),
            30,
        )
        .unwrap();

        // a future position refuses, and so does a gap the log never
        // filled: earlier numeric position alone is not existence
        let refused = log.execute(
            agent(),
            citing_note(TaskId(3), "out of reach [^1]", vec![RecordId(999)]),
            31,
        );
        assert!(matches!(refused, Err(Reject::InvalidReference)));
        let record = Record {
            id: RecordId(500),
            timestamp: 32,
            context: agent(),
            event: Event::Commented {
                home: TaskId(3),
                kind: CommentKind::Note {
                    body: prose("a hole in the log [^1]"),
                },
                references: vec![RecordId(499)],
            },
        };
        let refused = log.world.clone().apply(record);
        assert!(matches!(
            refused.map_err(|e| e.reason),
            Err(Reason::InvalidReference)
        ));
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
                body: prose("edge case held, suite green - 56 tests"),
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
                record: RecordId(RECORD_COUNT),
                reviser: agent().actor,
            })
        );

        // the bodyless Accept carries no prose to repair: marker
        // validation skips it, the transition refusal still lands
        let refused = log.execute(
            human(),
            Command::ReviseComment {
                id: CommentId(RecordId(10)),
                body: prose("no prose to swap [^1]"),
            },
            100,
        );
        assert!(matches!(refused, Err(Reject::InvalidStateTransition)));

        // the addressed position must be a comment's birth
        let refused = log.execute(
            human(),
            Command::ReviseComment {
                id: CommentId(RecordId(99)),
                body: prose("addresses nothing [^1]"),
            },
            101,
        );
        assert!(matches!(refused, Err(Reject::InvalidCommentId)));

        // a revision never retargets references: the citing note's
        // mapping survives its own repair untouched
        log.execute(
            agent(),
            Command::ReviseComment {
                id: CommentId(RecordId(19)),
                body: prose("rests on the conclusion above, reworded [^1]"),
            },
            102,
        )
        .unwrap();
        let birth = log
            .records()
            .iter()
            .find(|r| matches!(r.event, Event::Commented { .. } if r.id == RecordId(19)))
            .expect("the birth stands in the log");
        match &birth.event {
            Event::Commented { references, .. } => assert_eq!(*references, vec![RecordId(3)]),
            other => panic!("the birth keeps its shape: {other:?}"),
        }
        assert_eq!(log.records().len(), RECORD_COUNT + 2);
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
                record: RecordId(RECORD_COUNT),
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
                body: prose("repair in place [^1]"),
            },
        ] {
            let refused = log.execute(agent(), command, 100);
            assert!(matches!(refused, Err(Reject::InvalidStateTransition)));
        }

        // the variants keep their verdicts: the offer, the accept, the
        // reject, and the uses all refuse the door
        for id in [3usize, 10, 8, 17, 22] {
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
                note(TaskId(id), &format!("for the record, on the {state} task")),
                99,
            )
            .unwrap_or_else(|e| panic!("note on {state} task refused: {e:?}"));
        }
        assert_eq!(log.records().len(), before + 2);

        // only an unknown home task refuses, writing nothing
        let refused = log.execute(agent(), note(TaskId(9), "no such home"), 100);
        assert!(matches!(refused, Err(Reject::InvalidTaskId)));
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
    use crate::objects::message::MessageState;

    fn registered(id: &str) -> Command {
        Command::RegisterIncarnation {
            native_id: NativeSessionId::new(id.into()).unwrap(),
            saved_session: SessionLocator::new(format!("/sessions/{id}.jsonl").into()).unwrap(),
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
                saved_session: SessionLocator::new("/sessions/aaaa1111.jsonl".into()).unwrap(),
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
            saved_session: SessionLocator::new("/elsewhere/aaaa1111.jsonl".into()).unwrap(),
        };
        let refused = log.execute_system(relocated, 3);
        assert!(matches!(refused, Err(Reject::SavedSessionMismatch)));
        assert_eq!(log.records().len(), 2);

        // the registration verb rejects judgment tiers
        for ctx in [human(), agent()] {
            let refused = log.execute(ctx, registered("cccc3333"), 3);
            assert!(matches!(refused, Err(Reject::HumanOnly)));
        }
        assert_eq!(log.records().len(), 2);
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
        log.execute(agent(), offer(TaskId(0), "suite green, 55 tests"), 3)
            .unwrap();

        // the native session declares no task: its scope covers nothing
        let mut session = native_session(0);
        session.actor = agent().actor;
        log.execute(session, accept_on(TaskId(0), 2), 4).unwrap();
        assert_eq!(log.world().tasks[0].task.state, TaskState::Concluded);

        // the task-scoped producer still cannot accept its own offer
        log.execute(
            human(),
            Command::CreateTask {
                name: prose("fix bar"),
            },
            5,
        )
        .unwrap();
        let scoped = scoped_agent(1);
        log.execute(scoped.clone(), offer(TaskId(1), "bar fixed"), 6)
            .unwrap();
        let refused = log.execute(scoped, accept_on(TaskId(1), 5), 7);
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
}
