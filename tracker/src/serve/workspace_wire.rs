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
