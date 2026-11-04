//! The host contract: the six verbs every persistent-session host
//! answers, with the shared payload and refusal types they speak.
//! Each host is one compiled-in dialect implementing the interface;
//! doors consume the interface, never a host's specifics, so a host
//! swap is a dialect module plus its own claim.

use std::path::Path;
use std::time::Duration;

/// A host address returned at launch: opaque bytes, asserting nothing
/// beyond host-reported liveness at probe time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HostAddress(pub String);

/// Host-reported liveness at probe time. Unknown is honest and is
/// never read as gone by any consumer.
#[derive(Clone, Debug, PartialEq)]
pub enum Liveness {
    Live,
    Gone,
    Unknown(String),
}

/// One bounded host wait's answer. The wake is advisory only: truth
/// about arrivals is the session's own mailbox read.
#[derive(Clone, Debug, PartialEq)]
pub enum WaitWake {
    Message,
    Quiet,
}

/// A launch refusal, naming the evidence.
#[derive(Debug)]
pub enum LaunchRefusal {
    /// The cwd is missing or not a directory; never asked the host.
    UnusableCwd(String),
    /// The host could not be asked (spawn failure).
    Unavailable(String),
    /// The host was asked and refused.
    HostRefused(String),
    /// The host started a terminal but named no readable address.
    NoAddress(String),
}

/// A submission refusal, naming the evidence. Submitted means the
/// host accepted the ordered submission - written and submitted,
/// nothing more; Unresolved means the bounded attempt expired and the
/// caller's message stays pending, evidence of the attempt, never a
/// promise.
#[derive(Debug)]
pub enum SubmitRefusal {
    /// The host named the address absent or finished.
    AddressNotLive(String),
    /// The host was asked and refused for its own reasons.
    HostRefused(String),
    /// The host could not be asked or its answer could not be read.
    Unknown(String),
    /// The adapter-side bound expired; the attempt is unresolved.
    Unresolved,
}

/// The finished-state fact a close reply carried: the host's own read
/// of the program it just ended, never a re-list. Unconfirmed carries
/// the raw evidence when the reply named no finished state - the
/// consumer refuses rather than guessing.
#[derive(Debug)]
pub enum CloseFact {
    Finished,
    Unconfirmed(String),
}

/// A close refusal, naming the evidence.
#[derive(Debug)]
pub enum CloseRefusal {
    AddressNotLive(String),
    HostRefused(String),
    Unknown(String),
}

/// A placement refusal, naming the evidence. The terminal exists
/// either way - the launch cannot be undone - so the caller discloses
/// and carries on rather than failing a launch that already landed.
#[derive(Debug)]
pub enum PlaceRefusal {
    /// The host named the address absent or finished.
    AddressNotLive(String),
    /// The host was asked and refused for its own reasons.
    HostRefused(String),
    /// The host could not be asked, its answer could not be read, or
    /// the bounded ask expired.
    Unknown(String),
}

/// The five verbs. dyn-compatible: doors hold the dialect behind the
/// interface, never a host type.
pub trait HostContract: Sync {
    /// One persistent session running argv in cwd; the adapter never
    /// supervises what it launched, and every launch passes the
    /// host's no-parent form (settled law).
    fn launch(&self, cwd: &Path, argv: &[String]) -> Result<HostAddress, LaunchRefusal>;
    /// One placement declaration on a terminal the caller just
    /// launched: the host's own grouping reads it and the terminal
    /// shows where the author says it lives. One ask, right after the
    /// launch's address returns; ids are local to that machine.
    fn place(&self, address: &HostAddress, space: &str, tab: &str) -> Result<(), PlaceRefusal>;
    /// Liveness gates submission and nothing else; no liveness fact
    /// ever enters the log.
    fn live(&self, address: &HostAddress) -> Liveness;
    /// One ordered submission into a live session: queue-style, the
    /// write never waits for a pause and never observes a turn.
    fn submit(&self, address: &HostAddress, text: &str) -> Result<(), SubmitRefusal>;
    /// One bounded host wait at an explicit address; the arrival-wait
    /// door loops over this verb with the mailbox as truth.
    fn wait(&self, address: &HostAddress, bound: Duration) -> WaitWake;
    /// Ends the session program and carries back the finished-state
    /// fact the reply named. Consumed only by the workspace clean
    /// effect's ordered contract, never as a standalone door.
    fn close(&self, address: &HostAddress) -> Result<CloseFact, CloseRefusal>;
}

impl std::fmt::Display for LaunchRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchRefusal::UnusableCwd(detail) => write!(f, "{detail}"),
            LaunchRefusal::Unavailable(detail) => {
                write!(f, "the host could not be asked: {detail}")
            }
            LaunchRefusal::HostRefused(detail) => {
                write!(f, "the host refused the launch: {detail}")
            }
            LaunchRefusal::NoAddress(detail) => {
                write!(f, "the host named no readable address: {detail}")
            }
        }
    }
}

impl std::fmt::Display for SubmitRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitRefusal::AddressNotLive(detail) => write!(f, "the address is not live: {detail}"),
            SubmitRefusal::HostRefused(detail) => {
                write!(f, "the host refused the submission: {detail}")
            }
            SubmitRefusal::Unknown(detail) => {
                write!(f, "the submission could not be asked: {detail}")
            }
            SubmitRefusal::Unresolved => write!(
                f,
                "the submission did not resolve within its bound; the message stays pending"
            ),
        }
    }
}

impl std::fmt::Display for CloseRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloseRefusal::AddressNotLive(detail) => write!(f, "the address is not live: {detail}"),
            CloseRefusal::HostRefused(detail) => write!(f, "the host refused the close: {detail}"),
            CloseRefusal::Unknown(detail) => write!(f, "the close could not be asked: {detail}"),
        }
    }
}

impl std::fmt::Display for PlaceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaceRefusal::AddressNotLive(detail) => write!(f, "the address is not live: {detail}"),
            PlaceRefusal::HostRefused(detail) => {
                write!(f, "the host refused the placement: {detail}")
            }
            PlaceRefusal::Unknown(detail) => {
                write!(f, "the placement could not be asked: {detail}")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn the_trait_stays_dyn_compatible() {
        fn _assert_dyn(_: &dyn HostContract) {}
    }
}
