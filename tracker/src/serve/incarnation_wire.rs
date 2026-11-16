//! The incarnation doors' wire: request and reply shapes for the
//! native attach and the incarnation register doors, one definition
//! shared by the server (serve/incarnation.rs) and the client half
//! (serve/client.rs).

use serde::{Deserialize, Serialize};

use crate::db::StoredRecord;

/// What the integration learns when its session attaches.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttachReply {
    /// Whether this call birthed the registration; an existing exact
    /// native session resolves, never a second incarnation
    pub registered: bool,
    /// The birth record position: the machine identity channel
    pub incarnation: usize,
    /// The incarnation's dense public number, the human-facing i-N
    pub num: usize,
    pub records: Vec<StoredRecord>,
}

/// The attaching process's presented terminal: the host-injected
/// self-address of the terminal it runs in, with launched-versus-
/// adopted provenance from the door's session-scoped launch word.
/// Machine-supplied disclosure at the contract boundary - honest
/// deployment evidence of where the process runs, never proof of
/// which launch spawned it; identity rides the exact resolution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PresentedTerminal {
    pub address: String,
    pub adopted: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AttachRequest {
    pub native_id: crate::record::types::pointers::NativeSessionId,
    pub session_file: crate::record::types::pointers::SessionLocator,
    /// The harness the attaching session runs under; the pi
    /// extension sends none and folds as pi. Absent stays absent on
    /// the wire: the pi client's hand-built request named no key, and
    /// the field's default folds the same.
    #[serde(default)]
    #[serde(skip_serializing_if = "harness_is_default")]
    pub harness: crate::host::harness::Harness,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal: Option<PresentedTerminal>,
    #[serde(default)]
    pub client: Option<String>,
}

/// The wire keeps the default off the attach request: the client's
/// hand-built body named no harness key.
fn harness_is_default(harness: &crate::host::harness::Harness) -> bool {
    *harness == crate::host::harness::Harness::default()
}

/// The launch door's own birth: the pre-allocated native id and the
/// opaque host address, recorded before the harness's first save - the
/// association exists at launch time, no correlation token, no wait.
#[derive(Debug, Serialize, Deserialize)]
pub struct IncarnationRegisterRequest {
    pub native_id: crate::record::types::pointers::NativeSessionId,
    pub address: String,
    /// The address kind: the here door adopts the caller terminal,
    /// every other birth launches a new terminal.
    #[serde(default)]
    pub adopted: bool,
    /// The harness the fresh session runs under; absent folds as pi.
    #[serde(default)]
    pub harness: crate::host::harness::Harness,
    /// The launch's name, recorded as the birth's first naming. It
    /// rides only when the launch carries one, so a server that
    /// predates naming still answers an unnamed launch.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<crate::record::objects::incarnation::IncarnationName>,
    #[serde(default)]
    pub client: Option<String>,
}

/// The registration's reply, the keys the register door answered
/// inline until 2.4.
#[derive(Debug, Serialize, Deserialize)]
pub struct IncarnationRegisterReply {
    pub incarnation: usize,
    pub records: Vec<StoredRecord>,
    pub registered: bool,
}
