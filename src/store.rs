use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::decide::{decide, enforce_tier, expand};
use crate::events::{Command, Event};
use crate::objects::comment::{Comment, CommentContext, CommentId};
use crate::objects::proposal::{Proposal, ProposalContext, ProposalId, ProposalState};
use crate::objects::task::{Task, TaskContext, TaskState};
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
    InvalidTaskId,
    InvalidProposalId,
    InvalidCommentId,
    InvalidStateTransition,
    /// A task already holds an open judgment proposal
    ProposalAlreadyOpen,
}

/// Command-path translation of fold failures into refusals; total over Reason
/// since every fold failure is command-reachable.
pub(crate) fn to_reject(reason: Reason) -> Reject {
    match reason {
        Reason::InvalidTaskId => Reject::InvalidTaskId,
        Reason::InvalidProposalId => Reject::InvalidProposalId,
        Reason::InvalidCommentId => Reject::InvalidCommentId,
        Reason::InvalidParentTaskId => Reject::InvalidParentTaskId,
        Reason::InvalidStateTransition => Reject::InvalidStateTransition,
        Reason::ProposalAlreadyOpen => Reject::ProposalAlreadyOpen,
    }
}

#[derive(Clone, Debug, PartialEq)]
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
}

impl World {
    pub fn new() -> Self {
        World {
            tasks: Vec::new(),
            proposals: BTreeMap::new(),
            comments: BTreeMap::new(),
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
                });
            }
            ref event @ (Event::TaskClaimed { id }
            | Event::TaskDone { id, .. }
            | Event::TaskDropped { id, .. }
            | Event::TaskReleased { id, .. }) => {
                let task_ctx = self.tasks.get_mut(id.0).ok_or(Reason::InvalidTaskId)?;
                task_ctx.last_updated = record.id;
                task_ctx.task = task_ctx
                    .task
                    .apply(event)
                    .ok_or(Reason::InvalidStateTransition)?;
            }
            // Proposal events
            Event::ProposalCreated {
                name: proposal_name,
                action,
            } => {
                let proposal_id = ProposalId(record.id);

                // one judgment at a time: the task's proposal pointer is its open judgment;
                // the act must be legal at birth — staleness acquired later is designed state
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
                proposal_ctx.proposal = proposal_ctx
                    .proposal
                    .apply(event)
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
            Event::Commented { target, body } => {
                let mut up = target;
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

                self.comments.insert(
                    CommentId(record.id),
                    CommentContext {
                        comment: Comment { target, body },
                        actor: record.context.actor.as_str().into(),
                    },
                );
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
            .map_err(|err| to_reject(err.reason))?;
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
            to_reject(Reason::InvalidTaskId),
            Reject::InvalidTaskId
        ));
        assert!(matches!(
            to_reject(Reason::InvalidProposalId),
            Reject::InvalidProposalId
        ));
        assert!(matches!(
            to_reject(Reason::InvalidCommentId),
            Reject::InvalidCommentId
        ));
        assert!(matches!(
            to_reject(Reason::InvalidParentTaskId),
            Reject::InvalidParentTaskId
        ));
        assert!(matches!(
            to_reject(Reason::InvalidStateTransition),
            Reject::InvalidStateTransition
        ));
        assert!(matches!(
            to_reject(Reason::ProposalAlreadyOpen),
            Reject::ProposalAlreadyOpen
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
