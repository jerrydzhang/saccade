//! Request shapes, error interpretation and remote containment over any
//! transport. Three read-only calls exist, exactly the pinned
//! herdr-core 0.1.17 schema's fields:
//!
//! - `file.read`   {"path", "on"}                    - whole-file text
//! - `machine.exec` {"argv", "on", "timeout": 30.0}  - fixed argv, no shell
//! - `machine.list` {}                               - error enrichment
//!
//! Containment is proven on the remote side: the host enforces nothing,
//! so both the declaration directory and the joined candidate are
//! canonicalized THERE with one fixed `realpath` argv and the prefix
//! relation is checked against the canonical forms before any bytes
//! move; the read itself uses the canonical path. Bytes ride one fixed
//! `cat` argv; over-cap and truncated reads are refused, never rendered.

use crate::base64;
use crate::error::CoreFail;
use crate::guard::guard_relative;
use crate::transport::CoreTransport;
use serde_json::{Value, json};

/// The exec byte cap, from the schema: stdout keeps the first 8 MB and
/// flags the rest. A flagged read is refused here, never rendered.
pub const ASSET_CAP: u64 = 8 * 1024 * 1024;

/// The bounded time budget for every fixed-argv read on a machine.
const EXEC_TIMEOUT_SECONDS: f64 = 30.0;

/// The stderr excerpt carried in failure messages.
const STDERR_EXCERPT: usize = 200;

#[derive(Debug)]
pub struct ExecOutput {
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub truncated: bool,
    pub timed_out: bool,
}

pub struct CoreClient<T: CoreTransport> {
    transport: T,
}

impl<T: CoreTransport> CoreClient<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    /// One whole-file text read: `file.read` with no paging. A read
    /// that stopped inside the file (first != 1 or last != lines) is
    /// refused; partial text never becomes a board.
    pub fn read_text(&self, machine: &str, path: &str) -> Result<String, CoreFail> {
        let result = self.request(
            "file.read",
            &json!({"path": path, "on": machine}),
            machine,
            Some(path),
        )?;
        let text = str_field(&result, "text")?;
        let first = number_field(&result, "first")?;
        let last = number_field(&result, "last")?;
        let lines = number_field(&result, "lines")?;
        if first != 1 || last != lines {
            return Err(CoreFail::TextTruncated {
                path: path.into(),
                machine: machine.into(),
                got_last: last,
                total_lines: lines,
            });
        }
        Ok(text.to_string())
    }

    /// One fixed-argv program on a machine: argv as given, no shell,
    /// bounded time. A nonzero exit or a timeout is a result, not an
    /// error.
    pub fn exec(&self, machine: &str, argv: &[&str]) -> Result<ExecOutput, CoreFail> {
        let result = self.request(
            "machine.exec",
            &json!({"argv": argv, "on": machine, "timeout": EXEC_TIMEOUT_SECONDS}),
            machine,
            None,
        )?;
        let stdout =
            base64::decode(str_field(&result, "stdout")?).map_err(|e| CoreFail::Protocol {
                detail: format!("exec stdout: {e}"),
            })?;
        let stderr =
            base64::decode(str_field(&result, "stderr")?).map_err(|e| CoreFail::Protocol {
                detail: format!("exec stderr: {e}"),
            })?;
        Ok(ExecOutput {
            code: result.get("code").and_then(Value::as_i64).map(|c| c as i32),
            stdout,
            stderr,
            truncated: bool_field(&result, "truncated")?,
            timed_out: bool_field(&result, "timed_out")?,
        })
    }

    /// The machine's own state line (state, why), or `None` when the
    /// core does not know the name at all.
    pub fn machine_state(
        &self,
        machine: &str,
    ) -> Result<Option<(String, Option<String>)>, CoreFail> {
        let result = self.request("machine.list", &json!({}), machine, None)?;
        let machines = result
            .get("machines")
            .and_then(Value::as_array)
            .ok_or_else(|| CoreFail::Protocol {
                detail: "machine.list result carries no machines".into(),
            })?;
        for entry in machines {
            if entry.get("name").and_then(Value::as_str) == Some(machine) {
                let state = entry
                    .get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let why = entry
                    .get("why")
                    .and_then(Value::as_str)
                    .map(|why| why.to_string());
                return Ok(Some((state, why)));
            }
        }
        Ok(None)
    }

    /// One asset on a machine, with containment proven there. The
    /// canonical prefix check holds on the remote side because a remote
    /// symlink inside the board directory can point anywhere the user
    /// can read; a local canonical check would not transfer.
    pub fn asset_bytes(
        &self,
        machine: &str,
        base: &str,
        relative: &str,
    ) -> Result<Vec<u8>, CoreFail> {
        guard_relative(relative).map_err(|reason| CoreFail::Denied {
            reason: reason.to_string(),
        })?;
        let joined = if base.ends_with('/') {
            format!("{base}{relative}")
        } else {
            format!("{base}/{relative}")
        };
        // one fixed argv answers with the canonical base and the
        // canonical candidate, in request order
        let resolve_argv = ["realpath", "-e", "--", base, joined.as_str()];
        let resolved = self.exec(machine, &resolve_argv)?;
        if resolved.timed_out {
            return Err(CoreFail::ExecTimeout {
                argv: argv_line(&resolve_argv),
                machine: machine.into(),
            });
        }
        if resolved.code != Some(0) {
            return Err(CoreFail::NotFound {
                path: joined.clone(),
                machine: machine.into(),
                detail: stderr_excerpt(&resolved.stderr),
            });
        }
        if resolved.truncated {
            return Err(CoreFail::Protocol {
                detail: "realpath output was cut short".into(),
            });
        }
        let text = String::from_utf8(resolved.stdout).map_err(|_| CoreFail::Protocol {
            detail: "realpath wrote non-UTF-8 paths".into(),
        })?;
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() != 2 {
            return Err(CoreFail::Protocol {
                detail: "realpath did not answer with exactly two paths".into(),
            });
        }
        let canon_base = lines[0].trim_end_matches('\r');
        let canon_candidate = lines[1].trim_end_matches('\r');
        // the trailing-slash strip makes the root directory say exactly
        // what it is: a declaration at "/" bounds nothing
        let prefix = canon_base.trim_end_matches('/');
        let inside =
            canon_candidate == prefix || canon_candidate.starts_with(&format!("{prefix}/"));
        if !inside {
            return Err(CoreFail::Containment {
                path: joined.clone(),
                machine: machine.into(),
            });
        }
        let read_argv = ["cat", "--", canon_candidate];
        let bytes = self.exec(machine, &read_argv)?;
        if bytes.timed_out {
            return Err(CoreFail::ExecTimeout {
                argv: argv_line(&read_argv),
                machine: machine.into(),
            });
        }
        if bytes.code != Some(0) {
            return Err(CoreFail::NotFound {
                path: canon_candidate.into(),
                machine: machine.into(),
                detail: stderr_excerpt(&bytes.stderr),
            });
        }
        if bytes.truncated || bytes.stdout.len() as u64 > ASSET_CAP {
            return Err(CoreFail::AssetTooLarge {
                path: joined.clone(),
                machine: machine.into(),
            });
        }
        Ok(bytes.stdout)
    }

    fn request(
        &self,
        method: &str,
        params: &Value,
        machine: &str,
        subject: Option<&str>,
    ) -> Result<Value, CoreFail> {
        let response = self.transport.call(method, params)?;
        if let Some(error) = response.get("error") {
            return Err(self.interpret_error(error, machine, subject));
        }
        response
            .get("result")
            .cloned()
            .ok_or_else(|| CoreFail::Protocol {
                detail: format!("{method} response carries neither result nor error"),
            })
    }

    /// The pinned schema pins `data.kind`; the live CLI has also been
    /// seen carrying `kind` and `hint` flat on the error object, so
    /// both shapes are accepted.
    fn interpret_error(&self, error: &Value, machine: &str, subject: Option<&str>) -> CoreFail {
        let kind = error
            .get("data")
            .and_then(|data| data.get("kind"))
            .and_then(Value::as_str)
            .or_else(|| error.get("kind").and_then(Value::as_str))
            .unwrap_or("");
        let detail = error
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| error.get("hint").and_then(Value::as_str))
            .unwrap_or("")
            .to_string();
        match kind {
            "machine_unknown" => CoreFail::MachineUnknown {
                machine: machine.into(),
            },
            "unavailable" => self.machine_unavailable(machine, detail),
            "needs_update" => CoreFail::MachineNeedsUpdate {
                machine: machine.into(),
            },
            "not_found" => CoreFail::NotFound {
                path: subject.unwrap_or("(the core did not name it)").into(),
                machine: machine.into(),
                detail,
            },
            "not_text" => CoreFail::NotText {
                path: subject.unwrap_or("(the core did not name it)").into(),
                machine: machine.into(),
            },
            "io" => CoreFail::NotFound {
                path: subject.unwrap_or("(the core did not name it)").into(),
                machine: machine.into(),
                detail,
            },
            "invalid_params" => CoreFail::InvalidParams { detail },
            "spawn_failed" => CoreFail::SpawnFailed { detail },
            "" => CoreFail::Protocol {
                detail: format!("the core error carries no kind: {detail}"),
            },
            other => CoreFail::Protocol {
                detail: format!("core error kind {other:?}: {detail}"),
            },
        }
    }

    /// One read-only machine.list turns a bare "unavailable" into the
    /// machine's own state line. This is error enrichment, not a retry:
    /// nothing is asked twice, and the answer changes only the message.
    fn machine_unavailable(&self, machine: &str, detail: String) -> CoreFail {
        match self.machine_state(machine) {
            Ok(Some((state, why))) => match state.as_str() {
                "needs_login" => CoreFail::MachineNeedsLogin {
                    machine: machine.into(),
                },
                "unreachable" => CoreFail::MachineUnreachable {
                    machine: machine.into(),
                    why,
                },
                "connecting" => CoreFail::MachineConnecting {
                    machine: machine.into(),
                },
                other => CoreFail::MachineUnavailable {
                    machine: machine.into(),
                    detail: match why {
                        Some(why) => format!("state {other:?}: {why}"),
                        None => format!("state {other:?}"),
                    },
                },
            },
            Ok(None) => CoreFail::MachineUnknown {
                machine: machine.into(),
            },
            Err(_) => CoreFail::MachineUnavailable {
                machine: machine.into(),
                detail,
            },
        }
    }
}

fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, CoreFail> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| CoreFail::Protocol {
            detail: format!("result carries no {key} string"),
        })
}

fn number_field(value: &Value, key: &str) -> Result<u64, CoreFail> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| CoreFail::Protocol {
            detail: format!("result carries no {key} number"),
        })
}

fn bool_field(value: &Value, key: &str) -> Result<bool, CoreFail> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| CoreFail::Protocol {
            detail: format!("result carries no {key} flag"),
        })
}

fn argv_line(argv: &[&str]) -> String {
    argv.join(" ")
}

fn stderr_excerpt(stderr: &[u8]) -> String {
    if stderr.is_empty() {
        return String::new();
    }
    let text = String::from_utf8_lossy(stderr);
    let excerpt: String = text.chars().take(STDERR_EXCERPT).collect();
    let trimmed = excerpt.trim_end();
    if trimmed.is_empty() {
        String::new()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// A fake core: canned answers, every request recorded. The contract
    /// tests run entirely against this - no socket, no process.
    struct Fake {
        answers: Mutex<VecDeque<Result<Value, CoreFail>>>,
        seen: Mutex<Vec<(String, Value)>>,
    }

    impl Fake {
        fn new(answers: &[Value]) -> Self {
            Self {
                answers: Mutex::new(
                    answers
                        .iter()
                        .map(|value| Ok(value.clone()))
                        .collect::<VecDeque<_>>(),
                ),
                seen: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<(String, Value)> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl CoreTransport for Fake {
        fn call(&self, method: &str, params: &Value) -> Result<Value, CoreFail> {
            self.seen
                .lock()
                .unwrap()
                .push((method.to_string(), params.clone()));
            self.answers
                .lock()
                .unwrap()
                .pop_front()
                .expect("fake ran out of canned answers")
        }
    }

    fn file_read_ok(text: &str, lines: u64) -> Value {
        json!({"text": text, "first": 1, "last": lines, "lines": lines})
    }

    fn exec_ok(stdout_b64: &str) -> Value {
        json!({
            "code": 0,
            "stdout": stdout_b64,
            "stderr": "",
            "truncated": false,
            "timed_out": false,
            "ms": 3,
        })
    }

    fn error_envelope(kind: &str) -> Value {
        json!({"error": {"code": -32000, "data": {"kind": kind}, "message": "boom"}})
    }

    fn machine_list(state: &str, why: Option<&str>) -> Value {
        json!({"result": {"machines": [
            {"name": "tired", "target": "tired", "id": "core-1", "host": "tired",
             "version": "0.1.17", "state": state,
             "why": why}
        ]}})
    }

    fn realpath_result(base: &str, candidate: &str) -> Value {
        exec_ok(&base64_of(&format!("{base}\n{candidate}\n")))
    }

    fn base64_of(text: &str) -> String {
        // a tiny reference encoder for fixtures: standard alphabet
        let bytes = text.as_bytes();
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = *chunk.get(1).unwrap_or(&0) as u32;
            let b2 = *chunk.get(2).unwrap_or(&0) as u32;
            let window = (b0 << 16) | (b1 << 8) | b2;
            let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            out.push(alphabet[(window >> 18) as usize & 63] as char);
            out.push(alphabet[(window >> 12) as usize & 63] as char);
            if chunk.len() > 1 {
                out.push(alphabet[(window >> 6) as usize & 63] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(alphabet[window as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    fn assert_fails<T: std::fmt::Debug>(
        result: Result<T, CoreFail>,
        code: &str,
        needle: &str,
    ) -> CoreFail {
        let fail = result.unwrap_err();
        assert_eq!(fail.code(), code, "{fail}");
        assert!(fail.to_string().contains(needle), "{fail}");
        fail
    }

    // --- request shapes ---------------------------------------------------

    #[test]
    fn read_text_sends_exactly_the_schema_params() {
        let fake = Fake::new(&[json!({"result": file_read_ok("# board\n", 1)})]);
        let client = CoreClient::new(&fake);
        let text = client.read_text("tired", "/srv/b.yaml").unwrap();
        assert_eq!(text, "# board\n");
        let calls = fake.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "file.read");
        assert_eq!(calls[0].1, json!({"path": "/srv/b.yaml", "on": "tired"}));
    }

    #[test]
    fn exec_sends_a_fixed_argv_on_and_an_explicit_timeout() {
        let fake = Fake::new(&[json!({"result": exec_ok(&base64_of("hi"))})]);
        let client = CoreClient::new(&fake);
        client.exec("tired", &["cat", "--", "/x"]).unwrap();
        let calls = fake.calls();
        assert_eq!(calls[0].0, "machine.exec");
        assert_eq!(
            calls[0].1,
            json!({"argv": ["cat", "--", "/x"], "on": "tired", "timeout": 30.0})
        );
        // never a shell line
        assert!(calls[0].1.get("line").is_none());
    }

    #[test]
    fn exec_decodes_base64_stdout_and_stderr() {
        let fake = Fake::new(&[json!({"result": json!({
            "code": 0,
            "stdout": base64_of("bytes!"),
            "stderr": base64_of("warn"),
            "truncated": false,
            "timed_out": false,
            "ms": 4,
        })})]);
        let client = CoreClient::new(&fake);
        let output = client.exec("tired", &["cat", "--", "/x"]).unwrap();
        assert_eq!(output.stdout, b"bytes!");
        assert_eq!(output.stderr, b"warn");
        assert_eq!(output.code, Some(0));
    }

    // --- text reads -------------------------------------------------------

    #[test]
    fn a_partial_text_read_is_refused() {
        let fake = Fake::new(&[json!({"result": json!({
            "text": "partial\n", "first": 1, "last": 1, "lines": 900})})]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/big.yaml"),
            "text-truncated",
            "line 1 of 900",
        );
    }

    #[test]
    fn not_text_is_a_denial_with_the_path_named() {
        let fake = Fake::new(&[json!({"error": {
            "code": -32000, "data": {"kind": "not_text"}, "message": "nope"}})]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/img.png"),
            "asset-denied",
            "/img.png",
        );
    }

    #[test]
    fn both_error_envelope_shapes_are_accepted() {
        for envelope in [
            error_envelope("not_found"),
            json!({"error": {"code": -32000, "kind": "not_found", "hint": "gone"}}),
        ] {
            let fake = Fake::new(&[envelope]);
            let client = CoreClient::new(&fake);
            let fail = client.read_text("tired", "/gone.yaml").unwrap_err();
            assert_eq!(fail.code(), "asset-missing", "{fail}");
        }
    }

    // --- machine-state error mapping ---------------------------------------

    #[test]
    fn machine_unknown_maps_directly() {
        let fake = Fake::new(&[error_envelope("machine_unknown")]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("ghost", "/b.yaml"),
            "machine-unknown",
            "ghost",
        );
        assert_eq!(
            fake.calls().len(),
            1,
            "no enrichment call for a precise kind"
        );
    }

    #[test]
    fn unavailable_becomes_the_machine_state_line() {
        // needs_login
        let fake = Fake::new(&[
            error_envelope("unavailable"),
            machine_list("needs_login", None),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "machine-needs-login",
            "hc machine login tired",
        );
        // unreachable, with the reason quoted
        let fake = Fake::new(&[
            error_envelope("unavailable"),
            machine_list("unreachable", Some("ssh: port 22: refused")),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "machine-unreachable",
            "ssh: port 22: refused",
        );
        // still connecting
        let fake = Fake::new(&[
            error_envelope("unavailable"),
            machine_list("connecting", None),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "machine-unavailable",
            "connecting",
        );
        // a name the core does not know surfaces as unknown
        let fake = Fake::new(&[
            error_envelope("unavailable"),
            json!({"result": {"machines": []}}),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("ghost", "/b.yaml"),
            "machine-unknown",
            "ghost",
        );
        // enrichment that itself fails keeps the core's own detail
        let fake = Fake::new(&[error_envelope("unavailable"), error_envelope("io")]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "machine-unavailable",
            "boom",
        );
    }

    #[test]
    fn needs_update_maps_directly_without_a_machine_list() {
        let fake = Fake::new(&[error_envelope("needs_update")]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "machine-needs-update",
            "tired",
        );
        assert_eq!(
            fake.calls().len(),
            1,
            "no enrichment call for a precise kind"
        );
    }

    #[test]
    fn unknown_and_missing_kinds_are_protocol_refusals() {
        let fake = Fake::new(&[
            json!({"error": {"code": -32000, "data": {"kind": "mystery"}, "message": "?"}}),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "core-protocol",
            "mystery",
        );
        let fake = Fake::new(&[json!({"error": {"code": -32000, "message": "bare"}})]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "core-protocol",
            "no kind",
        );
    }

    // --- remote containment and bytes ---------------------------------------

    #[test]
    fn a_held_asset_resolves_both_sides_remotely_then_reads_the_canonical_path() {
        let fake = Fake::new(&[
            json!({"result": realpath_result("/srv/b", "/srv/b/fig.png")}),
            json!({"result": exec_ok(&base64_of("PNGDATA"))}),
        ]);
        let client = CoreClient::new(&fake);
        let bytes = client.asset_bytes("tired", "/srv/b", "fig.png").unwrap();
        assert_eq!(bytes, b"PNGDATA");
        let calls = fake.calls();
        assert_eq!(calls.len(), 2);
        // the containment proof ran remotely, one fixed argv, no shell
        assert_eq!(
            calls[0].1,
            json!({"argv": ["realpath", "-e", "--", "/srv/b", "/srv/b/fig.png"],
                   "on": "tired", "timeout": 30.0})
        );
        // the read used the canonical candidate
        assert_eq!(
            calls[1].1,
            json!({"argv": ["cat", "--", "/srv/b/fig.png"],
                   "on": "tired", "timeout": 30.0})
        );
    }

    #[test]
    fn an_escaped_candidate_is_refused_before_any_bytes_move() {
        let fake = Fake::new(&[json!({"result": realpath_result("/srv/b", "/etc/passwd")})]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.asset_bytes("tired", "/srv/b", "fig.png"),
            "asset-denied",
            "fig.png",
        );
        assert_eq!(fake.calls().len(), 1, "no cat after a containment refusal");
    }

    #[test]
    fn a_symlinked_declaration_directory_canonizes_first() {
        // the human opened /srv/link/b; the real directory is /srv/real/b
        let fake = Fake::new(&[
            json!({"result": realpath_result("/srv/real/b", "/srv/real/b/fig.png")}),
            json!({"result": exec_ok(&base64_of("x"))}),
        ]);
        let client = CoreClient::new(&fake);
        client
            .asset_bytes("tired", "/srv/link/b", "fig.png")
            .unwrap();
        let calls = fake.calls();
        assert_eq!(
            calls[1].1,
            json!({"argv": ["cat", "--", "/srv/real/b/fig.png"],
                   "on": "tired", "timeout": 30.0})
        );
    }

    #[test]
    fn a_declaration_at_the_root_bounds_nothing() {
        // containment against "/" admits every absolute path; the human
        // chose it, and the prefix check says so rather than guessing
        let fake = Fake::new(&[
            json!({"result": realpath_result("/", "/etc/passwd")}),
            json!({"result": exec_ok(&base64_of("rooty"))}),
        ]);
        let client = CoreClient::new(&fake);
        let bytes = client.asset_bytes("tired", "/", "etc/passwd").unwrap();
        assert_eq!(bytes, b"rooty");
        assert_eq!(fake.calls().len(), 2);
    }

    #[test]
    fn structural_guard_refusals_never_reach_the_transport() {
        let attempts = [
            ("../secret", "parent traversal"),
            ("/abs", "absolute path"),
            ("a\\b", "backslash"),
            ("", "empty path"),
        ];
        for (attempt, reason) in attempts {
            let fake = Fake::new(&[]);
            let client = CoreClient::new(&fake);
            assert_fails(
                client.asset_bytes("tired", "/srv/b", attempt),
                "asset-denied",
                reason,
            );
            assert_eq!(fake.calls().len(), 0, "{attempt}");
        }
    }

    #[test]
    fn a_missing_candidate_is_missing_not_denied() {
        let fake = Fake::new(&[json!({"result": json!({
            "code": 1, "stdout": "", "stderr": base64_of("realpath: No such file"),
            "truncated": false, "timed_out": false, "ms": 2})})]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.asset_bytes("tired", "/srv/b", "gone.png"),
            "asset-missing",
            "No such file",
        );
    }

    #[test]
    fn a_failed_read_carries_the_stderr_excerpt() {
        let fake = Fake::new(&[
            json!({"result": realpath_result("/srv/b", "/srv/b/fig.png")}),
            json!({"result": json!({
                "code": 1, "stdout": "", "stderr": base64_of("cat: fig.png: Is a directory"),
                "truncated": false, "timed_out": false, "ms": 2})}),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.asset_bytes("tired", "/srv/b", "fig.png"),
            "asset-missing",
            "Is a directory",
        );
    }

    #[test]
    fn truncated_and_oversized_bytes_are_refused_never_rendered() {
        let fake = Fake::new(&[
            json!({"result": realpath_result("/srv/b", "/srv/b/fig.png")}),
            json!({"result": json!({
                "code": 0, "stdout": base64_of("half a picture"), "stderr": "",
                "truncated": true, "timed_out": false, "ms": 9})}),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.asset_bytes("tired", "/srv/b", "fig.png"),
            "asset-too-large",
            "8 MB",
        );
    }

    #[test]
    fn a_timed_out_read_is_an_honest_timeout() {
        let fake = Fake::new(&[
            json!({"result": realpath_result("/srv/b", "/srv/b/fig.png")}),
            json!({"result": json!({
                "code": null, "stdout": "", "stderr": "",
                "truncated": false, "timed_out": true, "ms": 30000})}),
        ]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.asset_bytes("tired", "/srv/b", "fig.png"),
            "exec-timeout",
            "cat -- /srv/b/fig.png",
        );
    }

    #[test]
    fn machine_level_failures_during_resolution_keep_the_machine_codes() {
        let fake = Fake::new(&[error_envelope("needs_update")]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.asset_bytes("tired", "/srv/b", "fig.png"),
            "machine-needs-update",
            "tired",
        );
    }

    #[test]
    fn malformed_core_results_are_protocol_refusals() {
        // bad base64 in exec output
        let fake = Fake::new(&[json!({"result": json!({
            "code": 0, "stdout": "**", "stderr": "",
            "truncated": false, "timed_out": false, "ms": 1})})]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.exec("tired", &["cat", "--", "/x"]),
            "core-protocol",
            "stdout",
        );
        // missing schema-required flags
        let fake = Fake::new(&[json!({"result": json!({
            "code": 0, "stdout": "", "stderr": "", "timed_out": false, "ms": 1})})]);
        let client = CoreClient::new(&fake);
        assert_fails(
            client.exec("tired", &["cat", "--", "/x"]),
            "core-protocol",
            "truncated",
        );
        // a response with neither result nor error
        struct Bare;
        impl CoreTransport for Bare {
            fn call(&self, _method: &str, _params: &Value) -> Result<Value, CoreFail> {
                Ok(json!({"jsonrpc": "2.0", "id": 1}))
            }
        }
        let client = CoreClient::new(Bare);
        assert_fails(
            client.read_text("tired", "/b.yaml"),
            "core-protocol",
            "neither result nor error",
        );
    }
}
