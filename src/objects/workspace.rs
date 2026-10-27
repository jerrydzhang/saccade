use serde::{Deserialize, Serialize};

use crate::RecordId;
use crate::events::Event;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

/// The record position of a created workspace's birth is its identity.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct WorkspaceId(pub RecordId);

/// A workspace born by its own creation record: independent of any task
/// or incarnation, its identity the birth record's position. The facts
/// are what the provisioner established, never caller-claimed values.
/// The stage is the object's own, born Available.
#[derive(Clone, Debug, PartialEq)]
pub struct CreatedWorkspace {
    /// The immutable commit the workspace was cut from
    pub base: GitCommit,
    /// The branch provisioning created
    pub branch: GitBranch,
    /// The workspace's absolute address
    pub worktree: WorktreePath,
    pub stage: WorkspaceStage,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreatedWorkspaceContext {
    pub workspace: CreatedWorkspace,
    /// Event time of the birth record
    pub born_at: u64,
    /// The completed fact that set the current stage; None at a birth,
    /// where the birth itself is the fact
    pub stage_fact: Option<WorkspaceStageFact>,
}

/// A workspace's lifecycle stage, owned by the workspace: Available
/// stands at birth, Cleaned stands after a completed cleanup, and a
/// completed restoration returns Available - idempotent by
/// postcondition. The latest completed fact after the birth is the
/// stage; earlier facts stay in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkspaceStage {
    Available,
    Cleaned,
}

impl WorkspaceStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkspaceStage::Available => "available",
            WorkspaceStage::Cleaned => "cleaned",
        }
    }

    /// The pure transition table: the stage a completed lifecycle fact
    /// moves this workspace to, or None when the fact does not apply -
    /// a cleanup of an already Cleaned workspace names a stage already
    /// there, and a restoration is idempotent by postcondition, so it
    /// re-states Available over Available.
    pub fn transition(&self, event: &Event) -> Option<WorkspaceStage> {
        match (self, event) {
            (WorkspaceStage::Available, Event::WorkspaceCleaned { .. }) => {
                Some(WorkspaceStage::Cleaned)
            }
            (WorkspaceStage::Available, Event::WorkspaceRestored { .. }) => {
                Some(WorkspaceStage::Available)
            }
            (WorkspaceStage::Cleaned, Event::WorkspaceRestored { .. }) => {
                Some(WorkspaceStage::Available)
            }
            // every other event moves no workspace stage
            _ => None,
        }
    }
}

/// The completed lifecycle fact that set a workspace's current stage:
/// the record that landed it and its event time - provenance on the
/// context, never on the stage.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceStageFact {
    pub record: RecordId,
    pub at: u64,
}

/// The latest confirmed saved-directory change of a native incarnation:
/// the selected directory, the record that confirmed it, and its event
/// time. A machinery fact riding the incarnation, never a lifecycle
/// stage; earlier changes stay in the log.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceChange {
    pub worktree: WorktreePath,
    pub record: RecordId,
    pub at: u64,
}

/// The task's working environment, born once by the establishment
/// record: the facts that record established, never a promise that
/// a directory or process still exists - existence checks live in
/// effects, not the fold. The birth record's position is the shared
/// workspace/incarnation identity. The stage is the object's own,
/// born Available, the same stage the created family carries.
#[derive(Clone, Debug, PartialEq)]
pub struct Workspace {
    /// The immutable commit the workspace was cut from
    pub base: GitCommit,
    /// The branch preserving advancement between checkpoints
    pub branch: GitBranch,
    /// The latest recorded clean head; born at base
    pub checkpoint: GitCommit,
    /// The workspace's absolute address
    pub worktree: WorktreePath,
    pub stage: WorkspaceStage,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceContext {
    pub workspace: Workspace,
    /// The establishment record's position: the shared
    /// workspace/incarnation identity
    pub birth: RecordId,
    /// Every head the record has named, base first; the explicit
    /// checkpoint door returns to none of them
    pub heads: Vec<GitCommit>,
    /// The completed fact that set the current stage; None at a birth
    pub stage_fact: Option<WorkspaceStageFact>,
}

#[cfg(test)]
mod test {
    use super::*;

    /// The stage transition table is an exhaustive inventory: every
    /// stage paired with every completed lifecycle fact, pinned where
    /// the object lives.
    #[test]
    fn the_stage_transition_table_admits_exactly_the_legal_cells() {
        let stages = [WorkspaceStage::Available, WorkspaceStage::Cleaned];
        let facts = [
            Event::WorkspaceCleaned {
                workspace: RecordId(0),
            },
            Event::WorkspaceRestored {
                workspace: RecordId(0),
            },
        ];
        let expected = [
            // available: cleanup cleans, restoration re-states available
            Some(WorkspaceStage::Cleaned),
            Some(WorkspaceStage::Available),
            // cleaned: a second cleanup names a stage already there,
            // and restoration returns available
            None,
            Some(WorkspaceStage::Available),
        ];
        for (cell, stage) in stages.iter().enumerate() {
            for (fact_cell, fact) in facts.iter().enumerate() {
                let moved = stage.transition(fact);
                assert_eq!(
                    moved,
                    expected[cell * 2 + fact_cell],
                    "{stage:?} under {:?} moves wrong",
                    fact
                );
            }
        }
    }
}
