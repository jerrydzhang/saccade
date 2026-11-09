//! The board viewer's herdr-core client.
//!
//! One wire contract, pinned here against the pinned herdr-core 0.1.17
//! schema (`hc schema`): JSON-RPC 2.0, one message per line on the
//! core's Unix socket. `file.read` for text, `machine.exec` with fixed
//! argv for bytes and remote path resolution, `machine.list` to enrich
//! machine-state errors, and - the suite's one write, the explainer
//! annotation submit (t-24@r21) - `file.edit`, exact-once and
//! all-or-none. No machine management, no shell strings, no retries:
//! every machine state is a structured, human-actionable error.
//!
//! The crate is pure over a `CoreTransport`, so the full contract runs
//! against fakes in the fast check loop; the socket framing itself is
//! exercised against a local test socket, never a core.

pub mod base64;
pub mod client;
pub mod error;
pub mod guard;
pub mod transport;

pub use client::{CoreClient, ExecOutput, FileEdit, FileEditOutcome, ASSET_CAP};
pub use error::CoreFail;
pub use guard::guard_relative;
pub use transport::{CoreTransport, SocketTransport};
