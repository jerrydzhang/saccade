//! Every refusal the core client can produce, each with the stable code
//! the bridge hands to the renderer and a human-readable line. Machine
//! states are human-actionable facts, so the messages name the human's
//! own host verbs; the client itself never retries and never performs
//! any of them.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CoreFail {
    /// The local core socket is missing, refused the connection, or the
    /// round trip timed out.
    Socket { path: String, detail: String },
    /// The core spoke something this client cannot parse, or answered
    /// with a shape it never promised.
    Protocol { detail: String },
    /// The machine is not in the core's machine list.
    MachineUnknown { machine: String },
    /// The machine's link is still coming up.
    MachineConnecting { machine: String },
    /// The machine needs its login repeated.
    MachineNeedsLogin { machine: String },
    /// The machine cannot be reached at all.
    MachineUnreachable {
        machine: String,
        why: Option<String>,
    },
    /// The machine is known but not ready, and the core said nothing
    /// more specific.
    MachineUnavailable { machine: String, detail: String },
    /// The machine runs a herdr-core this client cannot talk to.
    MachineNeedsUpdate { machine: String },
    /// The file is not there, or the program that reads it said no.
    NotFound {
        path: String,
        machine: String,
        detail: String,
    },
    /// The file exists but is not UTF-8 text.
    NotText { path: String, machine: String },
    /// The core refused the request's shape; a bridge bug, never a retry.
    InvalidParams { detail: String },
    /// The core could not even start the program.
    SpawnFailed { detail: String },
    /// The program ran and exited nonzero.
    ExecFailed {
        argv: String,
        machine: String,
        code: Option<i32>,
        stderr: String,
    },
    /// The program ran out of its time budget.
    ExecTimeout { argv: String, machine: String },
    /// Bytes beyond the transport cap; refused, never rendered.
    AssetTooLarge { path: String, machine: String },
    /// The resolved asset path left the declaration directory on the
    /// remote side, or failed the structural guard.
    Containment { path: String, machine: String },
    /// A structural path refusal at the boundary.
    Denied { reason: String },
    /// The text read stopped inside the file: partial text never
    /// becomes a board.
    TextTruncated {
        path: String,
        machine: String,
        got_last: u64,
        total_lines: u64,
    },
}

impl CoreFail {
    /// The structured code the renderer sees. The list is closed: both
    /// sides of the bridge ship from this repository.
    pub fn code(&self) -> &'static str {
        match self {
            CoreFail::Socket { .. } => "core-unavailable",
            CoreFail::Protocol { .. } => "core-protocol",
            CoreFail::MachineUnknown { .. } => "machine-unknown",
            CoreFail::MachineConnecting { .. } => "machine-unavailable",
            CoreFail::MachineNeedsLogin { .. } => "machine-needs-login",
            CoreFail::MachineUnreachable { .. } => "machine-unreachable",
            CoreFail::MachineUnavailable { .. } => "machine-unavailable",
            CoreFail::MachineNeedsUpdate { .. } => "machine-needs-update",
            CoreFail::NotFound { .. } => "asset-missing",
            CoreFail::NotText { .. } => "asset-denied",
            CoreFail::InvalidParams { .. } => "bridge-invalid",
            CoreFail::SpawnFailed { .. } => "exec-spawn-failed",
            CoreFail::ExecFailed { .. } => "asset-missing",
            CoreFail::ExecTimeout { .. } => "exec-timeout",
            CoreFail::AssetTooLarge { .. } => "asset-too-large",
            CoreFail::Containment { .. } => "asset-denied",
            CoreFail::Denied { .. } => "asset-denied",
            CoreFail::TextTruncated { .. } => "text-truncated",
        }
    }
}

fn optional_why(why: &Option<String>) -> String {
    match why {
        Some(why) if !why.is_empty() => format!(": {why}"),
        _ => String::new(),
    }
}

impl fmt::Display for CoreFail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreFail::Socket { path, detail } => write!(
                f,
                "cannot reach your herdr-core at {path}: {detail}; is the herdr core running?"
            ),
            CoreFail::Protocol { detail } => {
                write!(f, "herdr-core answered something unexpected: {detail}")
            }
            CoreFail::MachineUnknown { machine } => write!(
                f,
                "unknown machine {machine:?}: it is not in your core's machine list; \
                 the viewer never adds machines, that is your own host setup (hc machine add)"
            ),
            CoreFail::MachineConnecting { machine } => write!(
                f,
                "machine {machine:?} is still connecting; it may need a moment"
            ),
            CoreFail::MachineNeedsLogin { machine } => write!(
                f,
                "machine {machine:?} needs login; run `hc machine login {machine}` yourself"
            ),
            CoreFail::MachineUnreachable { machine, why } => write!(
                f,
                "machine {machine:?} is unreachable{}; check it with `hc machine ls`",
                optional_why(why)
            ),
            CoreFail::MachineUnavailable { machine, detail } => {
                write!(f, "machine {machine:?} is not ready: {detail}")
            }
            CoreFail::MachineNeedsUpdate { machine } => write!(
                f,
                "machine {machine:?} runs a herdr-core this viewer cannot talk to; \
                 `hc machine update {machine}` updates it"
            ),
            CoreFail::NotFound {
                path,
                machine,
                detail,
            } => write!(
                f,
                "{path:?} on machine {machine:?} is missing or unreadable{detail}",
                detail = if detail.is_empty() {
                    String::new()
                } else {
                    format!(" ({detail})")
                }
            ),
            CoreFail::NotText { path, machine } => {
                write!(f, "{path:?} on machine {machine:?} is not UTF-8 text")
            }
            CoreFail::InvalidParams { detail } => {
                write!(f, "the viewer sent a malformed request: {detail}")
            }
            CoreFail::SpawnFailed { detail } => {
                write!(f, "the machine could not start the read: {detail}")
            }
            CoreFail::ExecFailed {
                argv,
                machine,
                code,
                stderr,
            } => write!(
                f,
                "{argv:?} failed on machine {machine:?} (exit {code:?}){stderr}",
                stderr = if stderr.is_empty() {
                    String::new()
                } else {
                    format!(": {stderr}")
                }
            ),
            CoreFail::ExecTimeout { argv, machine } => write!(
                f,
                "{argv:?} on machine {machine:?} ran out of its time budget"
            ),
            CoreFail::AssetTooLarge { path, machine } => write!(
                f,
                "{path:?} on machine {machine:?} exceeds the 8 MB transfer cap; \
                 it is refused, never partially rendered"
            ),
            CoreFail::Containment { path, machine } => write!(
                f,
                "{path:?} resolved outside the declaration directory on machine {machine:?}"
            ),
            CoreFail::Denied { reason } => write!(f, "refused: {reason}"),
            CoreFail::TextTruncated {
                path,
                machine,
                got_last,
                total_lines,
            } => write!(
                f,
                "the text read stopped inside {path:?} on machine {machine:?}: \
                 it got through line {got_last} of {total_lines}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CoreFail;

    #[test]
    fn every_variant_carries_its_pinned_code() {
        let inventory = vec![
            (
                CoreFail::Socket {
                    path: "/s".into(),
                    detail: "d".into(),
                },
                "core-unavailable",
            ),
            (CoreFail::Protocol { detail: "d".into() }, "core-protocol"),
            (
                CoreFail::MachineUnknown {
                    machine: "m".into(),
                },
                "machine-unknown",
            ),
            (
                CoreFail::MachineConnecting {
                    machine: "m".into(),
                },
                "machine-unavailable",
            ),
            (
                CoreFail::MachineNeedsLogin {
                    machine: "m".into(),
                },
                "machine-needs-login",
            ),
            (
                CoreFail::MachineUnreachable {
                    machine: "m".into(),
                    why: None,
                },
                "machine-unreachable",
            ),
            (
                CoreFail::MachineUnavailable {
                    machine: "m".into(),
                    detail: "d".into(),
                },
                "machine-unavailable",
            ),
            (
                CoreFail::MachineNeedsUpdate {
                    machine: "m".into(),
                },
                "machine-needs-update",
            ),
            (
                CoreFail::NotFound {
                    path: "/p".into(),
                    machine: "m".into(),
                    detail: String::new(),
                },
                "asset-missing",
            ),
            (
                CoreFail::NotText {
                    path: "/p".into(),
                    machine: "m".into(),
                },
                "asset-denied",
            ),
            (
                CoreFail::InvalidParams { detail: "d".into() },
                "bridge-invalid",
            ),
            (
                CoreFail::SpawnFailed { detail: "d".into() },
                "exec-spawn-failed",
            ),
            (
                CoreFail::ExecFailed {
                    argv: "cat".into(),
                    machine: "m".into(),
                    code: Some(1),
                    stderr: String::new(),
                },
                "asset-missing",
            ),
            (
                CoreFail::ExecTimeout {
                    argv: "cat".into(),
                    machine: "m".into(),
                },
                "exec-timeout",
            ),
            (
                CoreFail::AssetTooLarge {
                    path: "/p".into(),
                    machine: "m".into(),
                },
                "asset-too-large",
            ),
            (
                CoreFail::Containment {
                    path: "/p".into(),
                    machine: "m".into(),
                },
                "asset-denied",
            ),
            (CoreFail::Denied { reason: "r".into() }, "asset-denied"),
            (
                CoreFail::TextTruncated {
                    path: "/p".into(),
                    machine: "m".into(),
                    got_last: 1,
                    total_lines: 2,
                },
                "text-truncated",
            ),
        ];
        for (fail, code) in &inventory {
            assert_eq!(&fail.code(), code, "{fail:?}");
            assert!(!fail.to_string().is_empty(), "{fail:?}");
        }
    }

    #[test]
    fn machine_states_name_the_human_verb_and_never_a_retry() {
        let login = CoreFail::MachineNeedsLogin {
            machine: "tired".into(),
        }
        .to_string();
        assert!(login.contains("hc machine login tired"), "{login}");
        let update = CoreFail::MachineNeedsUpdate {
            machine: "tired".into(),
        }
        .to_string();
        assert!(update.contains("hc machine update tired"), "{update}");
        let unknown = CoreFail::MachineUnknown {
            machine: "x".into(),
        }
        .to_string();
        assert!(unknown.contains("never adds machines"), "{unknown}");
    }

    #[test]
    fn unreachable_carries_the_machine_reason() {
        let fail = CoreFail::MachineUnreachable {
            machine: "tired".into(),
            why: Some("ssh: connection refused".into()),
        };
        assert!(
            fail.to_string()
                .contains("is unreachable: ssh: connection refused")
        );
        let bare = CoreFail::MachineUnreachable {
            machine: "t".into(),
            why: None,
        }
        .to_string();
        assert!(bare.contains("is unreachable;"), "{bare}");
    }
}
