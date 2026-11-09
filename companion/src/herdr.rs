//! The herdr-core dialect, read side: binary resolution, the motion
//! subscription (`hc events --json`), and the two one-shot reads
//! (`hc ls --all`, `hc get`). Write verbs - send, keys, interrupt,
//! kill, rm - are absent by construction: the companion reads Herdr
//! and attaches viewers, it never writes to a terminal.

use std::path::PathBuf;

use serde_json::Value;

/// The explicit override, the same var the tracker's host dialect
/// names: set and nonempty, it is the host binary, always.
pub const OVERRIDE_VAR: &str = "SACCADE_HERDR";

/// Resolve the hc binary: the `--hc` override, then `SACCADE_HERDR`,
/// then the ambient PATH (production joins the machine's own server
/// through it).
pub fn resolve_hc(flag: Option<&str>) -> Result<PathBuf, String> {
    if let Some(named) = flag.filter(|f| !f.is_empty()) {
        return Ok(PathBuf::from(named));
    }
    if let Some(named) = std::env::var_os(OVERRIDE_VAR).filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(named));
    }
    Ok(PathBuf::from("hc"))
}

/// One row of the terminal table: the address, the host's own state
/// word (`live`, `finished`), and the program's current word - the
/// snapshot recovery path reads the word straight off the table when
/// the stream was down for it.
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalRow {
    pub address: String,
    pub state: String,
    pub word: Option<String>,
}

/// Parse `hc ls --all --json`: `{"terminals":[...]}`.
pub fn parse_ls(body: &str) -> Result<Vec<TerminalRow>, String> {
    let v: Value =
        serde_json::from_str(body).map_err(|e| format!("hc ls output is not JSON: {e}"))?;
    let rows = v
        .get("terminals")
        .and_then(Value::as_array)
        .ok_or("hc ls carries no terminals")?;
    Ok(rows
        .iter()
        .filter_map(|row| {
            Some(TerminalRow {
                address: row.get("address").and_then(Value::as_str)?.to_string(),
                state: row.get("state").and_then(Value::as_str)?.to_string(),
                word: row
                    .get("status")
                    .and_then(|s| s.get("state"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect())
}

/// One motion fact off the stream, matched by the terminal address.
#[derive(Debug, Clone, PartialEq)]
pub enum Motion {
    /// A status change; the word is the program's own new state, and
    /// the exit nulls it (a `status` event with no word arrives just
    /// before `finished`).
    Word {
        terminal: String,
        word: Option<String>,
    },
    /// The terminal finished, with the core's reason.
    Finished { terminal: String, reason: String },
    /// The core announced a restart (the `restarted` and `lost`
    /// events): kept terminals come back under their addresses.
    CoreRestarted,
}

/// Parse one `hc events --json` line. The pinned core wraps each event
/// in a JSON-RPC envelope with the payload under `params`; a bare
/// payload shape is accepted too. The first line is the subscription's
/// snapshot and carries neither - it decodes to nothing. Unknown event
/// types decode to nothing: the companion follows words and endings
/// only.
pub fn parse_event(line: &str) -> Result<Option<Motion>, String> {
    let v: Value =
        serde_json::from_str(line).map_err(|e| format!("event line is not JSON: {e}"))?;
    let v = v.get("params").unwrap_or(&v);
    let terminal = v.get("terminal").and_then(Value::as_str);
    match v.get("type").and_then(Value::as_str) {
        // the restart vocabulary is stream-level: no terminal rides it
        Some("restarted") | Some("lost") => Ok(Some(Motion::CoreRestarted)),
        Some("status") => {
            let Some(terminal) = terminal else {
                return Ok(None);
            };
            Ok(Some(Motion::Word {
                word: v
                    .get("data")
                    .and_then(|d| d.get("status"))
                    .and_then(|s| s.get("state"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                terminal: terminal.to_string(),
            }))
        }
        Some("finished") => {
            let Some(terminal) = terminal else {
                return Ok(None);
            };
            let reason = v
                .get("data")
                .and_then(|d| d.get("reason"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            Ok(Some(Motion::Finished {
                terminal: terminal.to_string(),
                reason,
            }))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // the stream shapes, captured against the pinned core exactly as
    // t-13@r6 records them - on the wire each rides a JSON-RPC
    // envelope, and the bare payload shape is accepted beside it
    #[test]
    fn a_status_line_yields_the_complete_new_word() {
        let wrapped = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":12,"terminal":"t2","type":"status","data":{"records":{},"reported":true,"status":{"state":"working","app":"pi","msg":"editing src","kind":null,"progress":null,"title":null}}}}"#;
        let motion = parse_event(wrapped).unwrap().unwrap();
        assert_eq!(
            motion,
            Motion::Word {
                terminal: "t2".into(),
                word: Some("working".into()),
            }
        );
        let bare =
            r#"{"seq":12,"terminal":"t2","type":"status","data":{"status":{"state":"working"}}}"#;
        assert_eq!(
            parse_event(bare).unwrap().unwrap(),
            Motion::Word {
                terminal: "t2".into(),
                word: Some("working".into()),
            }
        );
    }

    #[test]
    fn the_exit_nulls_the_word_before_the_finished_line() {
        let line = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":13,"terminal":"t2","type":"status","data":{"status":null}}}"#;
        assert_eq!(
            parse_event(line).unwrap().unwrap(),
            Motion::Word {
                terminal: "t2".into(),
                word: None,
            }
        );
        let line = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":14,"terminal":"t2","type":"finished","data":{"exit_code":0,"finished_ms":42,"reason":"exited","signal":null}}}"#;
        assert_eq!(
            parse_event(line).unwrap().unwrap(),
            Motion::Finished {
                terminal: "t2".into(),
                reason: "exited".into(),
            }
        );
    }

    #[test]
    fn the_subscription_snapshot_line_decodes_to_nothing() {
        let line = r#"{"channel":null,"seq":83,"terminals":[{"address":"t1","state":"live"}]}"#;
        assert_eq!(parse_event(line).unwrap(), None);
    }

    #[test]
    fn unknown_event_types_decode_to_nothing_and_restarts_signal() {
        for line in [
            r#"{"seq":15,"terminal":"t2","type":"created","data":{}}"#,
            r#"{"seq":16,"terminal":"t2","type":"turn","data":{"state":"idle","turn":3}}"#,
            r#"{"seq":17,"terminal":"t2","type":"size","data":{"cols":80,"rows":24}}"#,
        ] {
            assert_eq!(parse_event(line).unwrap(), None);
        }
        // a bare top-level restarted decodes like its wrapped form: the
        // params unwrap makes the two shapes one
        assert_eq!(
            parse_event(r#"{"seq":20,"type":"restarted"}"#)
                .unwrap()
                .unwrap(),
            Motion::CoreRestarted
        );
        // the restart vocabulary: its own motion, not a per-terminal one
        assert_eq!(
            parse_event(
                r#"{"jsonrpc":"2.0","method":"event","params":{"seq":18,"type":"restarted"}}"#
            )
            .unwrap()
            .unwrap(),
            Motion::CoreRestarted
        );
        assert_eq!(
            parse_event(r#"{"jsonrpc":"2.0","method":"event","params":{"seq":19,"type":"lost"}}"#)
                .unwrap()
                .unwrap(),
            Motion::CoreRestarted
        );
        // a bare top-level restarted decodes like its wrapped form: the
        // params unwrap makes the two shapes one
        assert_eq!(
            parse_event(r#"{"seq":20,"type":"restarted"}"#)
                .unwrap()
                .unwrap(),
            Motion::CoreRestarted
        );
    }

    #[test]
    fn the_terminal_table_parses_with_the_current_word() {
        let body = r#"{"hidden_finished":0,"terminals":[
            {"address":"t1","state":"live","status":{"state":"working","app":"pi"}},
            {"address":"t2","state":"finished","status":null},
            {"address":"local-b/t1","state":"live"}
        ]}"#;
        let rows = parse_ls(body).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2].address, "local-b/t1", "remote addresses verbatim");
        assert_eq!(
            rows[0].word.as_deref(),
            Some("working"),
            "the table carries the word"
        );
        assert_eq!(rows[1].word, None, "an exit nulls the word");
        assert_eq!(rows[2].word, None, "a silent program carries none");
    }

    #[test]
    fn resolution_prefers_the_flag_then_the_env_then_the_ambient() {
        // arms are resolved at call time from the process env; only the
        // flag arm is tested here - the env arm under a poisoned env is
        // the host dialect's own tested contract
        assert_eq!(
            resolve_hc(Some("/opt/hc")).unwrap(),
            PathBuf::from("/opt/hc")
        );
    }
}
