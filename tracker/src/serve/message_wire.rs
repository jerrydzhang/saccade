//! The messages doors' wire: request and reply shapes for the send
//! door and the mailbox read, one definition shared by the server
//! (serve/message.rs) and the client half (serve/client.rs).

use serde::{Deserialize, Serialize};

use crate::api::WireContext;
use crate::db::StoredRecord;

#[derive(Serialize, Deserialize)]
pub struct MessageSendRequest {
    pub context: WireContext,
    /// The recipient's public number, the human-facing i-N - the same
    /// token every envelope and contact names
    pub to: usize,
    #[serde(default)]
    pub reply_to: Option<usize>,
    #[serde(default)]
    pub task: Option<usize>,
    pub body: crate::record::types::prose::Prose,
    #[serde(default)]
    pub client: Option<String>,
}

/// One sent message's facts: its stable birth position and the dense
/// m-N, plus the sender's resolved identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MessageFacts {
    pub seq: usize,
    pub num: Option<usize>,
}

/// The native-submission attempt's truth: the state always, the
/// refusal's code and evidence on a refused submission, the
/// unrecorded state's disclosure on an unrecorded one. The optional
/// keys stay absent when nothing rides them, the shapes the send
/// door answered inline until 2.4.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmissionFacts {
    pub state: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SendMessageReply {
    pub records: Vec<StoredRecord>,
    pub message: MessageFacts,
    pub from: MessageFacts,
    /// The native-submission attempt's truth; absent only from a
    /// pre-attempt server, never from this wire's own success reply
    #[serde(default)]
    pub submission: Option<SubmissionFacts>,
}

/// One endpoint of a message as the mailbox read returns it. The
/// sender's endpoint names its actor; the recipient's names none,
/// so `actor` stays absent when no name rides.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MailboxEndpoint {
    pub seq: usize,
    pub num: Option<usize>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
}

/// One message as the mailbox read returns it: the stable birth
/// position, the dense endpoints, and the delivery state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MailboxMessage {
    pub seq: usize,
    pub num: Option<usize>,
    pub from: MailboxEndpoint,
    pub to: MailboxEndpoint,
    pub reply_to: Option<usize>,
    pub task: Option<usize>,
    pub state: String,
    pub body: String,
    pub born_at: u64,
}

/// The mailbox read's reply: the reader's mail in birth order, any
/// state, both directions, with the log's cursor.
#[derive(Debug, Serialize, Deserialize)]
pub struct MailboxReply {
    pub messages: Vec<MailboxMessage>,
    #[serde(default)]
    pub cursor: Option<usize>,
}
