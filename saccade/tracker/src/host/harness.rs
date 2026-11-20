//! The recorded harness distinction and its compiled-in native session
//! adapters: the pinned binary, the fresh and resume command lines, and
//! the saved-transcript header facts ordinary resume validates. The
//! caller owns process quiescence and the native startup itself; these
//! compose one argv and read one file, never the record or the tracker.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::pi_session::{HandoffFail, SessionHeader};

/// The harness a native session runs under; the default is what
/// registrations born before the field carried.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Harness {
    #[default]
    Pi,
    ClaudeCode,
}

impl Harness {
    /// The wire token the registration and attach requests carry.
    pub fn as_str(&self) -> &'static str {
        match self {
            Harness::Pi => "pi",
            Harness::ClaudeCode => "claude-code",
        }
    }

    /// The pinned binary; a PATH guess is never taken.
    pub fn pinned_binary(&self) -> Result<PathBuf, String> {
        match self {
            Harness::Pi => crate::host::pinned_pi().map_err(|e| e.to_string()),
            Harness::ClaudeCode => resolve_pin(
                std::env::var_os("SACCADE_CLAUDE_CODE"),
                option_env!("SACCADE_CLAUDE_CODE"),
                "no pinned claude-code: this binary carries no flake pin; export SACCADE_CLAUDE_CODE=<claude-code> to name one",
            ),
        }
    }

    /// The fresh-launch argv; pi saves under the tracker's own session
    /// dir, claude-code takes only the UUID.
    pub fn fresh_argv(
        &self,
        repo: &Path,
        native_id: &str,
        prompt: &[String],
    ) -> Result<Vec<String>, String> {
        let bin = self.pinned_binary()?;
        self.fresh_line(&bin, repo, native_id, prompt)
    }

    /// The resume argv; the saved locator is each harness's own
    /// resume token.
    pub fn resume_argv(
        &self,
        native_id: &str,
        session: &Path,
        prompt: &[String],
    ) -> Result<Vec<String>, String> {
        let bin = self.pinned_binary()?;
        Ok(self.resume_line(&bin, native_id, session, prompt))
    }

    /// The saved transcript's header facts; the cwd is the launch-time
    /// fallback, a recorded workspace change stays the authority.
    pub fn read_header(&self, session: &Path) -> Result<SessionHeader, String> {
        match self {
            Harness::Pi => crate::pi_session::read_header(session).map_err(|e| e.to_string()),
            Harness::ClaudeCode => claude_code_header(session).map_err(|e| e.to_string()),
        }
    }

    fn fresh_line(
        &self,
        bin: &Path,
        repo: &Path,
        native_id: &str,
        prompt: &[String],
    ) -> Result<Vec<String>, String> {
        if *self == Harness::ClaudeCode && !is_uuid(native_id) {
            return Err(format!(
                "claude-code --session-id takes a valid UUID; {native_id} is not one"
            ));
        }
        let mut argv = vec![bin.display().to_string()];
        match self {
            Harness::Pi => {
                argv.push("--session-id".into());
                argv.push(native_id.to_string());
                argv.push("--session-dir".into());
                argv.push(crate::pi_session::sessions_dir(repo).display().to_string());
            }
            Harness::ClaudeCode => {
                argv.push("--session-id".into());
                argv.push(native_id.to_string());
            }
        }
        argv.extend(prompt_tail(prompt));
        Ok(argv)
    }

    fn resume_line(
        &self,
        bin: &Path,
        native_id: &str,
        session: &Path,
        prompt: &[String],
    ) -> Vec<String> {
        let mut argv = vec![bin.display().to_string()];
        match self {
            Harness::Pi => {
                argv.push("--session".into());
                argv.push(native_id.to_string());
                argv.push("--session-dir".into());
                argv.push(
                    session
                        .parent()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                );
            }
            // the transcript path is claude-code's own resume token
            Harness::ClaudeCode => {
                argv.push("--resume".into());
                argv.push(session.display().to_string());
            }
        }
        argv.extend(prompt_tail(prompt));
        argv
    }
}

fn prompt_tail(prompt: &[String]) -> Vec<String> {
    if prompt.is_empty() {
        Vec::new()
    } else {
        vec![prompt.join(" ")]
    }
}

// pure so the precedence is testable without touching the environment
fn resolve_pin(
    env: Option<std::ffi::OsString>,
    compile: Option<&str>,
    missing: &str,
) -> Result<PathBuf, String> {
    if let Some(path) = env.filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = compile {
        return Ok(PathBuf::from(path));
    }
    Err(missing.to_string())
}

// claude-code refuses or renames a session whose --session-id is no UUID
fn is_uuid(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (at, b) in bytes.iter().enumerate() {
        let dashed = matches!(at, 8 | 13 | 18 | 23);
        match (dashed, b) {
            (true, b'-') => {}
            (false, b'0'..=b'9') => {}
            (false, b'a'..=b'f' | b'A'..=b'F') => {}
            _ => return false,
        }
    }
    true
}

// claude-code writes no header line: the first record carrying both
// sessionId and cwd answers, from existing transcript fields only
fn claude_code_header(path: &Path) -> Result<SessionHeader, HandoffFail> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(HandoffFail::Missing(path.to_path_buf()));
        }
        Err(e) => {
            return Err(HandoffFail::Unreadable {
                path: path.to_path_buf(),
                detail: e.to_string(),
            });
        }
    };
    if bytes.is_empty() {
        return Err(HandoffFail::Empty(path.to_path_buf()));
    }
    let mut saw_record = false;
    let mut saw_id = false;
    let mut header: Option<SessionHeader> = None;
    for line in bytes.split(|&b| b == b'\n') {
        let text = match std::str::from_utf8(line) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let value: serde_json::Value = match serde_json::from_str(text) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(map) = value.as_object() else {
            continue;
        };
        saw_record = true;
        let id = map.get("sessionId").and_then(|v| v.as_str());
        let cwd = map.get("cwd").and_then(|v| v.as_str());
        if let Some(id) = id.filter(|id| !id.is_empty()) {
            saw_id = true;
            if let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) {
                header = Some(SessionHeader {
                    id: id.to_string(),
                    cwd: cwd.to_string(),
                });
                break;
            }
        }
    }
    match header {
        Some(header) => Ok(header),
        None if !saw_record => Err(HandoffFail::MalformedHeader {
            path: path.to_path_buf(),
            detail: "no transcript records".to_string(),
        }),
        None if !saw_id => Err(HandoffFail::MissingId(path.to_path_buf())),
        None => Err(HandoffFail::MissingCwd(path.to_path_buf())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("saccade-harness-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // one claude-code-shaped transcript: records naming sessionId and cwd
    fn cc_story(session_id: &str, cwd: &str) -> Vec<u8> {
        let records = [
            r#"{"type":"summary","summary":"implement the adapter","leafUuid":"11111111-1111-4111-8111-111111111111"}"#
                .to_string(),
            format!(
                r#"{{"type":"user","uuid":"22222222-2222-4222-8222-222222222222","parentUuid":null,"timestamp":"2026-10-21T03:40:04.770Z","sessionId":"{session_id}","cwd":"{cwd}","gitBranch":"saccade/t-53","version":"2.1.238","message":{{"role":"user","content":"implement the harness adapter"}}}}"#
            ),
            format!(
                r#"{{"type":"assistant","uuid":"33333333-3333-4333-8333-333333333333","parentUuid":"22222222-2222-4222-8222-222222222222","timestamp":"2026-10-21T03:40:05.100Z","sessionId":"{session_id}","cwd":"{cwd}","message":{{"id":"msg_01","model":"claude-sonnet-4-5","content":[{{"type":"text","text":"starting from the harness module"}}],"usage":{{}}}}}}"#
            ),
        ];
        let mut bytes = Vec::new();
        for record in &records {
            bytes.extend_from_slice(record.as_bytes());
            bytes.push(b'\n');
        }
        bytes
    }

    fn variant(f: &HandoffFail) -> &'static str {
        match f {
            HandoffFail::Missing(_) => "Missing",
            HandoffFail::Empty(_) => "Empty",
            HandoffFail::NotUtf8(_) => "NotUtf8",
            HandoffFail::Unreadable { .. } => "Unreadable",
            HandoffFail::MalformedHeader { .. } => "MalformedHeader",
            HandoffFail::NotASession { .. } => "NotASession",
            HandoffFail::MissingId(_) => "MissingId",
            HandoffFail::MissingCwd(_) => "MissingCwd",
            HandoffFail::InvalidCwd(_) => "InvalidCwd",
        }
    }

    // the wire vocabulary lives here and nowhere else
    #[test]
    fn the_wire_tokens_are_kebab_and_round_trip() {
        assert_eq!(Harness::Pi.as_str(), "pi");
        assert_eq!(Harness::ClaudeCode.as_str(), "claude-code");
        for (token, harness) in [("pi", Harness::Pi), ("claude-code", Harness::ClaudeCode)] {
            let wire = serde_json::to_value(harness).unwrap();
            assert_eq!(wire.as_str().unwrap(), token);
            assert_eq!(serde_json::from_value::<Harness>(wire).unwrap(), harness);
        }
        assert!(serde_json::from_value::<Harness>(serde_json::json!("codex")).is_err());
    }

    // the non-empty runtime override wins, then the compile pin
    #[test]
    fn pin_resolution_names_its_own_precedence() {
        assert_eq!(
            resolve_pin(
                Some("/override/bin".into()),
                Some("/compile/bin"),
                "missing"
            )
            .unwrap(),
            PathBuf::from("/override/bin")
        );
        assert_eq!(
            resolve_pin(Some("".into()), Some("/compile/bin"), "missing").unwrap(),
            PathBuf::from("/compile/bin")
        );
        let missed = resolve_pin(
            None,
            None,
            "no pinned claude-code: export SACCADE_CLAUDE_CODE",
        );
        assert_eq!(
            missed.unwrap_err(),
            "no pinned claude-code: export SACCADE_CLAUDE_CODE"
        );
    }

    // the argv inventory: each harness speaks its own fresh and resume flags
    #[test]
    fn the_argv_inventory_carries_each_harnesss_own_flags() {
        let repo = Path::new("/repo/wt");
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let prompt = ["first words".to_string(), "second words".to_string()];
        assert_eq!(
            Harness::Pi
                .fresh_line(Path::new("pi"), repo, "native-id-1", &prompt)
                .unwrap(),
            vec![
                "pi".to_string(),
                "--session-id".to_string(),
                "native-id-1".to_string(),
                "--session-dir".to_string(),
                crate::pi_session::sessions_dir(repo).display().to_string(),
                "first words second words".to_string(),
            ]
        );
        assert_eq!(
            Harness::ClaudeCode
                .fresh_line(Path::new("/compile/claude-code"), repo, uuid, &prompt)
                .unwrap(),
            vec![
                "/compile/claude-code".to_string(),
                "--session-id".to_string(),
                uuid.to_string(),
                "first words second words".to_string(),
            ]
        );
        // an empty prompt rides no positional word
        let quiet = Harness::ClaudeCode
            .fresh_line(Path::new("/compile/claude-code"), repo, uuid, &[])
            .unwrap();
        assert_eq!(quiet.len(), 3);
        // pi's ids are its own vocabulary: any saved shape
        assert!(
            Harness::Pi
                .fresh_line(Path::new("pi"), repo, "handoff-smoke-1a2b3c4d", &[])
                .is_ok()
        );
        let refused = Harness::ClaudeCode.fresh_line(
            Path::new("/compile/claude-code"),
            repo,
            "handoff-smoke-1a2b3c4d",
            &[],
        );
        assert_eq!(
            refused.unwrap_err(),
            "claude-code --session-id takes a valid UUID; handoff-smoke-1a2b3c4d is not one"
        );
        assert!(is_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(is_uuid("AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"));
        assert!(!is_uuid("handoff-smoke-1a2b3c4d"));
        assert!(!is_uuid("550e8400e29b41d4a716446655440000"));
        assert!(!is_uuid("550e8400-e29b-41d4-a716-44665544000g"));

        let session = Path::new("/sessions/550e8400-e29b-41d4-a716-446655440000.jsonl");
        assert_eq!(
            Harness::Pi.resume_line(Path::new("pi"), uuid, session, &[]),
            vec![
                "pi".to_string(),
                "--session".to_string(),
                uuid.to_string(),
                "--session-dir".to_string(),
                "/sessions".to_string(),
            ]
        );
        // claude-code resumes by the transcript path; the id rides the file
        assert_eq!(
            Harness::ClaudeCode.resume_line(
                Path::new("/compile/claude-code"),
                uuid,
                session,
                &["continue the adapter".to_string()],
            ),
            vec![
                "/compile/claude-code".to_string(),
                "--resume".to_string(),
                "/sessions/550e8400-e29b-41d4-a716-446655440000.jsonl".to_string(),
                "continue the adapter".to_string(),
            ]
        );
    }

    // the saved-header inventory: each harness's facts and refusals
    #[test]
    fn the_saved_header_inventory_reads_each_harnesss_own_facts() {
        let dir = scratch("header");
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let cc = dir.join(format!("{uuid}.jsonl"));
        std::fs::write(&cc, cc_story(uuid, "/repo/wt")).unwrap();
        assert_eq!(
            Harness::ClaudeCode.read_header(&cc).unwrap(),
            SessionHeader {
                id: uuid.into(),
                cwd: "/repo/wt".into(),
            }
        );
        assert_eq!(std::fs::read(&cc).unwrap(), cc_story(uuid, "/repo/wt"));
        assert!(matches!(
            claude_code_header(&dir.join("absent.jsonl")),
            Err(HandoffFail::Missing(_))
        ));

        // a moved session: the first carrying record keeps the launch-time cwd
        let moved = dir.join("moved.jsonl");
        let mut bytes = cc_story("aaaa1111-1111-4111-8111-111111111111", "/repo/born");
        bytes.extend_from_slice(
            br#"{"type":"user","uuid":"44444444-4444-4444-8444-444444444444","sessionId":"aaaa1111-1111-4111-8111-111111111111","cwd":"/repo/moved","message":{"role":"user","content":"now from the new workspace"}}"#,
        );
        bytes.push(b'\n');
        std::fs::write(&moved, &bytes).unwrap();
        assert_eq!(claude_code_header(&moved).unwrap().cwd, "/repo/born");

        let pi = dir.join("pi-session.jsonl");
        std::fs::write(
            &pi,
            br#"{"type":"session","version":3,"id":"handoff-smoke-1a2b3c4d","cwd":"/repo/wt"}
{"type":"message","id":"4ba0a050","parentId":null}
"#,
        )
        .unwrap();
        assert_eq!(
            Harness::Pi.read_header(&pi).unwrap(),
            SessionHeader {
                id: "handoff-smoke-1a2b3c4d".into(),
                cwd: "/repo/wt".into(),
            }
        );
        // claude-code validation refuses a pi session: its records carry no sessionId
        let cc_on_pi = Harness::ClaudeCode.read_header(&pi).unwrap_err();
        assert!(cc_on_pi.contains("carries no id"), "{cc_on_pi}");

        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            ("empty.jsonl", Vec::new(), "Empty"),
            ("garbage.jsonl", b"\xff\xfe not json\n".to_vec(), "MalformedHeader"),
            (
                "no-id.jsonl",
                br#"{"type":"user","cwd":"/repo/wt","message":{"role":"user","content":"who am I"}}"#.to_vec(),
                "MissingId",
            ),
            (
                "no-cwd.jsonl",
                br#"{"type":"user","sessionId":"bbbb2222-2222-4222-8222-222222222222","message":{"role":"user","content":"where am I"}}"#.to_vec(),
                "MissingCwd",
            ),
            (
                "blank-cwd.jsonl",
                br#"{"type":"user","sessionId":"bbbb2222-2222-4222-8222-222222222222","cwd":"","message":{}}"#.to_vec(),
                "MissingCwd",
            ),
        ];
        for (name, bytes, expected) in &cases {
            let path = dir.join(name);
            std::fs::write(&path, bytes).unwrap();
            let outcome = claude_code_header(&path);
            assert_eq!(variant(&outcome.unwrap_err()), *expected, "{name}");
            assert_eq!(std::fs::read(&path).unwrap(), *bytes, "{name}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
