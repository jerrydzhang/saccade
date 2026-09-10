use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Reject;
use crate::decide::decide;
use crate::events::{Command, Event};
use crate::objects::comment::{Comment, CommentId};
use crate::objects::proposal::{Proposal, ProposalId, ProposalState};
use crate::objects::task::{Task, TaskId, TaskState};

#[derive(Clone, Debug, PartialEq)]
pub enum Tier {
    Human,
    Agent,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub actor: String,
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

#[derive(Debug, PartialEq)]
pub struct World {
    pub tasks: Vec<Task>,
    pub proposals: BTreeMap<ProposalId, Proposal>,
    pub comments: BTreeMap<CommentId, Comment>,
}

impl World {
    pub fn new() -> Self {
        World {
            tasks: Vec::new(),
            proposals: BTreeMap::new(),
            comments: BTreeMap::new(),
        }
    }

    pub fn replay(records: Vec<Record>) -> Self {
        let mut world = World::new();

        for record in &records {
            world.apply(record.clone());
        }

        world
    }

    /// Only valid path for world mutation
    pub fn apply(&mut self, record: Record) {
        match record.event {
            // Task events
            Event::TaskCreated {
                id,
                name: task_name,
                parent_id,
            } => {
                assert_eq!(id.0, self.tasks.len(), "non-dense TaskCreated id");

                self.tasks.push(Task {
                    state: TaskState::Open,
                    name: task_name,
                    parent_id,
                });
            }
            ref event @ (Event::TaskClaimed { id }
            | Event::TaskDone { id, .. }
            | Event::TaskDropped { id, .. }
            | Event::TaskReleased { id, .. }) => {
                let task = self
                    .tasks
                    .get_mut(id.0)
                    .expect("apply received an invalid task id");

                *task = task
                    .apply(event)
                    .expect("decide emitted an unfoldable event");
            }
            // Proposal events
            Event::ProposalCreated {
                name: proposal_name,
                action,
            } => {
                let proposal_id = ProposalId(record.id);

                self.proposals.insert(
                    proposal_id,
                    Proposal {
                        state: ProposalState::Open,
                        name: proposal_name,
                        action,
                    },
                );
            }
            ref event @ (Event::ProposalWithdrawn { id, .. }
            | Event::ProposalRejected { id, .. }
            | Event::ProposalAccepted { id, .. }) => {
                let proposal = self
                    .proposals
                    .get_mut(&id)
                    .expect("apply received an invalid proposal id");

                *proposal = proposal
                    .apply(event)
                    .expect("decide emitted an unfoldable event");
            }
            // Comment events
            Event::Commented { target, body } => {
                self.comments.insert(
                    CommentId(record.id),
                    Comment {
                        target,
                        body,
                        actor: record.context.actor.clone(),
                    },
                );
            }
        }
    }

    pub(crate) fn next_task_id(&self) -> TaskId {
        TaskId(self.tasks.len())
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
        let events = decide(&self.world, &context, command)?;

        let current_record_len = self.records.len();
        let records: Vec<Record> = events
            .into_iter()
            .enumerate()
            .map(|(i, event)| Record {
                id: RecordId(current_record_len + i),
                timestamp: now,
                context: context.clone(),
                event,
            })
            .collect();

        self.records.extend(records.iter().cloned());

        for record in &records {
            self.world.apply(record.clone());
        }

        Ok(records)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prose::Prose;
    
    fn agent() -> Context {
        Context {
            actor: "saccade bot".into(),
            tier: Tier::Agent,
        }
    }

    #[test]
    #[should_panic(expected = "apply received an invalid task id")]
    fn corrupted_task_id_panics() {
        let record = Record {
            id: RecordId(0),
            timestamp: 1,
            context: agent(),
            event: Event::TaskClaimed { id: TaskId(0) }, // no task exists yet
        };
        World::replay(vec![record]);
    }

    #[test]
    #[should_panic(expected = "non-dense TaskCreated id")]
    fn corrupted_new_task_id_record_panics() {
        let record = Record {
            id: RecordId(0),
            timestamp: 1,
            context: agent(),
            event: Event::TaskCreated {
                id: TaskId(1),
                name: Prose::new("invalid task".into()).unwrap(),
                parent_id: None,
            },
        };
        World::replay(vec![record]);
    }

    #[test]
    #[should_panic(expected = "non-dense TaskCreated id")]
    fn duplicated_task_id_records_panics() {
        let records = vec![
            Record {
                id: RecordId(0),
                timestamp: 1,
                context: agent(),
                event: Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("new task".into()).unwrap(),
                    parent_id: None,
                },
            },
            Record {
                id: RecordId(0),
                timestamp: 1,
                context: agent(),
                event: Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("new task again".into()).unwrap(),
                    parent_id: None,
                },
            },
        ];
        World::replay(records);
    }

    #[test]
    #[should_panic(expected = "decide emitted an unfoldable event")]
    fn invalid_task_transition_record_panics() {
        let records = vec![
            Record {
                id: RecordId(0),
                timestamp: 1,
                context: agent(),
                event: Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("new task".into()).unwrap(),
                    parent_id: None,
                },
            },
            Record {
                id: RecordId(0),
                timestamp: 1,
                context: agent(),
                event: Event::TaskDone {
                    id: TaskId(0),
                    receipt: Prose::new("jumping straight to done".into()).unwrap(),
                },
            },
        ];
        World::replay(records);
    }
}
