use crate::events::Event;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

/// A worktree's residency: born absent, present once physically created.
#[derive(Clone, Debug, PartialEq)]
pub enum WorktreeState {
    Absent,
    Present(WorktreePath),
}

impl WorktreeState {
    /// The state machine table, all state transitions must go through this
    pub fn transition(&self, event: &Event) -> Option<WorktreeState> {
        match (self, event) {
            (WorktreeState::Absent, Event::TaskWorktreeCreated { worktree, .. }) => {
                Some(WorktreeState::Present(worktree.clone()))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceContext {
    /// The immutable commit the workspace was cut from
    pub base: GitCommit,
    /// The branch preserving advancement between checkpoints
    pub branch: GitBranch,
    /// The latest recorded clean head; born at base
    pub checkpoint: GitCommit,
    /// Every head the record has named, base first; the explicit
    /// checkpoint door returns to none of them
    pub heads: Vec<GitCommit>,
    pub worktree: WorktreeState,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::task::TaskId;

    #[test]
    fn transition_table_admits_exactly_the_legal_cells() {
        let states = [
            WorktreeState::Absent,
            WorktreeState::Present(WorktreePath::new("/wt".into()).unwrap()),
        ];
        let events = [
            Event::TaskWorkspaceCreated {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            },
            Event::TaskWorktreeCreated {
                task_id: TaskId(0),
                worktree: WorktreePath::new("/wt".into()).unwrap(),
            },
            Event::TaskWorkspaceCheckpointed {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("def456".into()).unwrap(),
            },
        ];

        for state in &states {
            for event in &events {
                let legal = matches!(
                    (state, event),
                    (WorktreeState::Absent, Event::TaskWorktreeCreated { .. })
                );
                assert_eq!(
                    state.transition(event).is_some(),
                    legal,
                    "table disagrees at ({state:?}, {event:?})"
                );
            }
        }
    }
}
