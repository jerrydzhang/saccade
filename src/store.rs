use crate::decide::decide;
use crate::events::{Command, Event};
use crate::task::{Reject, Task, TaskId, TaskState};

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

#[derive(Clone, Debug)]
pub struct RecordId(pub(crate) usize);

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
}

impl World {
    pub fn new() -> Self {
        World { tasks: Vec::new() }
    }

    pub fn replay(records: Vec<Record>) -> Self {
        let mut world = World { tasks: Vec::new() };

        for record in &records {
            world.apply(record.clone());
        }

        world
    }

    /// Only valid path for world mutation
    pub fn apply(&mut self, record: Record) {
        match record.event {
            Event::TaskCreated {
                id,
                task_name,
                parent_id,
            } => {
                assert_eq!(id.0, self.tasks.len(), "non-dense TaskCreated id");

                self.tasks.push(Task {
                    id,
                    state: TaskState::Open,
                    task_name,
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
    use crate::task::Receipt;

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
                task_name: "invalid task".into(),
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
                    task_name: "new task".into(),
                    parent_id: None,
                },
            },
            Record {
                id: RecordId(0),
                timestamp: 1,
                context: agent(),
                event: Event::TaskCreated {
                    id: TaskId(0),
                    task_name: "new task again".into(),
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
                    task_name: "new task".into(),
                    parent_id: None,
                },
            },
            Record {
                id: RecordId(0),
                timestamp: 1,
                context: agent(),
                event: Event::TaskDone {
                    id: TaskId(0),
                    receipt: Receipt("jumping straight to done".into()),
                },
            },
        ];
        World::replay(records);
    }
}
