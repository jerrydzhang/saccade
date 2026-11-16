//! The workspace doors' wire: request and reply shapes for the
//! workspace create and clean doors, one definition shared by the
//! server (serve/workspace.rs) and the client half (serve/client.rs).

use serde::{Deserialize, Serialize};

use crate::db::StoredRecord;
use crate::store::WorkspaceFacts;

/// What a creation provisioned: actual Git facts and the birth record
/// that carries them - the workspace's identity, never a numbering.
/// The same facts ride the clean reply: the birth identity and what
/// the birth carried.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceBirth {
    /// The birth record's position: the workspace's stable identity
    pub seq: usize,
    pub base: String,
    pub branch: String,
    pub path: String,
}

impl WorkspaceBirth {
    /// The recorded facts of the lifecycle's target, read off a
    /// workspace birth.
    pub(crate) fn of(facts: WorkspaceFacts) -> Self {
        WorkspaceBirth {
            seq: facts.birth.0,
            base: facts.base.as_str().to_string(),
            branch: facts.branch.as_str().to_string(),
            path: facts.worktree.as_path().display().to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateReply {
    pub created: bool,
    pub workspace: WorkspaceBirth,
    pub records: Vec<StoredRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRequest {
    /// The descriptive label of the c-366 name: optional, falling
    /// back to workspace-<token>
    #[serde(default)]
    pub label: Option<String>,
    /// The canonical base: a ref, defaulting to the primary HEAD
    #[serde(default)]
    pub base: Option<String>,
    #[serde(default)]
    pub client: Option<String>,
}

/// Why a kept branch stands after a clean that acted: the merged
/// rule named no candidate containing its tip, or git holds no local
/// branch under the recorded name at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BranchKept {
    Unmerged,
    Missing,
}

/// The recorded branch's disposition at a clean that acted: whether
/// this clean deleted it, every local branch the merged rule named
/// (the first renders in the text face), and why it stands when
/// kept. Additive wire: the Cleaned event is untouched - the record's
/// story already names the branch and base.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchOutcome {
    pub deleted: bool,
    pub merged_into: Vec<String>,
    /// Why the branch stands: unmerged or missing; absent when deleted
    pub kept: Option<BranchKept>,
}

impl BranchOutcome {
    /// The branch went: git branch -d accepted it after the verified
    /// checkout removal.
    pub(crate) fn deleted(merged_into: Vec<String>) -> Self {
        BranchOutcome {
            deleted: true,
            merged_into,
            kept: None,
        }
    }

    /// The branch stands: no candidate contained its tip.
    pub(crate) fn kept_unmerged() -> Self {
        BranchOutcome {
            deleted: false,
            merged_into: Vec::new(),
            kept: Some(BranchKept::Unmerged),
        }
    }

    /// Nothing to act on: git holds no local branch under the name.
    pub(crate) fn kept_missing() -> Self {
        BranchOutcome {
            deleted: false,
            merged_into: Vec::new(),
            kept: Some(BranchKept::Missing),
        }
    }
}

/// The lifecycle controller's reply: clean names the exact recorded
/// workspace path; the server runs the ordered effects and lands the
/// completed facts itself.
#[derive(Debug, Serialize, Deserialize)]
pub struct CleanReply {
    /// A Cleaned fact landed with this call
    pub cleaned: bool,
    /// Recorded cleaned and every verified resource already absent:
    /// idempotent success, no duplicate transition
    pub already_cleaned: bool,
    pub workspace: WorkspaceBirth,
    /// The branch disposition of a clean that acted; null on the
    /// already-cleaned face, where this call acted no branch
    #[serde(default)]
    pub branch: Option<BranchOutcome>,
    /// The honest step list: what the door asked, what it verified,
    /// and any host ask that failed before the list confirmed closure
    pub verified: Vec<String>,
    pub records: Vec<StoredRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CleanRequest {
    /// The exact recorded workspace path; never inferred from a task
    /// or incarnation
    pub path: String,
    #[serde(default)]
    pub client: Option<String>,
}
