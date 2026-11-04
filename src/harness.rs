//! The recorded harness distinction and its compiled-in native session
//! adapters: the pinned binary, the fresh and resume command lines,
//! and the saved-transcript header facts ordinary resume validates.
//! Small compiled-in functions, never a plugin framework. The caller
//! owns process quiescence and the native startup itself; these
//! compose one argv and read one file, never the record or the
//! tracker.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::pi_session::{HandoffFail, SessionHeader};

/// The harness a native session runs under. Recorded at the
/// registration and the attach; native identity is interpreted within
/// its harness, and registrations born before the field existed fold
/// as pi.
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

    /// The pinned binary: the runtime override env first, then the
    /// flake pin this binary was built with. Never an arbitrary PATH
    /// guess; every harness resolves by the same shape.
    pub fn pinned_binary(&self) -> Result<PathBuf, String> {
        match self {
            // pi's pin is the host launch mechanics' own
            Harness::Pi => crate::host::pinned_pi().map_err(|e| e.to_string()),
            Harness::ClaudeCode => resolve_pin(
                std::env::var_os("SACCADE_CLAUDE_CODE"),
                option_env!("SACCADE_CLAUDE_CODE"),
                "no pinned claude-code: this binary carries no flake pin; export SACCADE_CLAUDE_CODE=<claude-code> to name one",
            ),
        }
    }

    /// The composed fresh-launch argv: one pinned binary started at
    /// the exact repo, carrying the pre-allocated native id. Pi saves
    /// under the tracker's own session dir; claude-code takes only the
    /// UUID and lands the transcript in its own project tree.
    pub fn fresh_argv(
        &self,
        repo: &Path,
        native_id: &str,
        prompt: &[String],
    ) -> Result<Vec<String>, String> {
        let bin = self.pinned_binary()?;
        self.fresh_line(&bin, repo, native_id, prompt)
    }

    /// The composed resume argv: the saved locator is each harness's
    /// own resume token. Pi re-speaks the native id and its session
    /// dir; claude-code resumes by the absolute transcript path.
    pub fn resume_argv(
        &self,
        native_id: &str,
        session: &Path,
        prompt: &[String],
    ) -> Result<Vec<String>, String> {
        let bin = self.pinned_binary()?;
        Ok(self.resume_line(&bin, native_id, session, prompt))
    }

    /// The saved transcript's validated header facts: its native id
    /// and its initial cwd. The cwd is the launch-time workspace
    /// fallback only - a recorded workspace change stays the
    /// authority, and the file is never rewritten to follow.
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
            // the transcript path is claude-code's own resume token:
            // --resume accepts the absolute .jsonl file in place of an id
            Harness::ClaudeCode => {
                argv.push("--resume".into());
                argv.push(session.display().to_string());
            }
        }
        argv.extend(prompt_tail(prompt));
        argv
    }
}

/// One prompt word list rides as the harnesses' single positional
/// text, the composition the launch doors already speak.
fn prompt_tail(prompt: &[String]) -> Vec<String> {
    if prompt.is_empty() {
        Vec::new()
    } else {
        vec![prompt.join(" ")]
    }
}

/// Pin resolution as pure data so the precedence is testable: the
/// non-empty runtime override wins, then the compile pin, and
/// otherwise the harness names what it missed.
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

/// claude-code's --session-id takes a valid UUID; the tracker's fresh
/// native ids already are, and anything else would launch a session
/// claude-code refuses or renames.
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

/// A claude-code transcript carries no session header line: every
/// record names its own sessionId and cwd. Validation reads the first
/// record carrying both - existing transcript fields only, nothing
/// invented - and strict saved-id matching stays with the caller.
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

    // a claude-code-shaped transcript: no header line, records naming
    // their own sessionId and cwd, then a story of turns
    fn cc_story(session_id: &str, cwd: &str) -> Vec<u8> {
        let records = [
            format!(
                r#"{{"type":"summary","summary":"implement the adapter","leafUuid":"11111111-1111-4111-8111-111111111111"}}"#
            ),
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

    #[test]
    fn the_wire_tokens_are_kebab_and_round_trip() {
        // the wire vocabulary lives here and nowhere else
        assert_eq!(Harness::Pi.as_str(), "pi");
        assert_eq!(Harness::ClaudeCode.as_str(), "claude-code");
        for (token, harness) in [("pi", Harness::Pi), ("claude-code", Harness::ClaudeCode)] {
            let wire = serde_json::to_value(&harness).unwrap();
            assert_eq!(wire.as_str().unwrap(), token);
            assert_eq!(serde_json::from_value::<Harness>(wire).unwrap(), harness);
        }
        // anything else is version skew, never a silent pi
        assert!(serde_json::from_value::<Harness>(serde_json::json!("codex")).is_err());
    }

    #[test]
    fn pin_resolution_names_its_own_precedence() {
        // the runtime override wins over the compile pin
        assert_eq!(
            resolve_pin(
                Some("/override/bin".into()),
                Some("/compile/bin"),
                "missing"
            )
            .unwrap(),
            PathBuf::from("/override/bin")
        );
        // an empty override is no override
        assert_eq!(
            resolve_pin(Some("".into()), Some("/compile/bin"), "missing").unwrap(),
            PathBuf::from("/compile/bin")
        );
        // neither pin names the miss
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

    #[test]
    fn fresh_argv_speaks_each_harnesss_own_flags() {
        let repo = Path::new("/repo/wt");
        let prompt = ["first words".to_string(), "second words".to_string()];
        let pi = Harness::Pi
            .fresh_line(Path::new("pi"), repo, "native-id-1", &prompt)
            .unwrap();
        assert_eq!(
            pi,
            vec![
                "pi".to_string(),
                "--session-id".to_string(),
                "native-id-1".to_string(),
                "--session-dir".to_string(),
                crate::pi_session::sessions_dir(repo).display().to_string(),
                "first words second words".to_string(),
            ]
        );
        // the bin comes from the pin resolution, so the shape is read
        // off the line after the program word
        let cc_uuid = "550e8400-e29b-41d4-a716-446655440000";
        let cc = Harness::ClaudeCode
            .fresh_line(Path::new("/compile/claude-code"), repo, cc_uuid, &prompt)
            .unwrap();
        assert_eq!(
            cc,
            vec![
                "/compile/claude-code".to_string(),
                "--session-id".to_string(),
                cc_uuid.to_string(),
                "first words second words".to_string(),
            ]
        );
        // an empty prompt rides no positional word
        let quiet = Harness::ClaudeCode
            .fresh_line(Path::new("/compile/claude-code"), repo, cc_uuid, &[])
            .unwrap();
        assert_eq!(quiet.len(), 3);
    }

    #[test]
    fn claude_code_fresh_launch_takes_only_valid_uuids() {
        let repo = Path::new("/repo/wt");
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
        // pi's ids are the harness's own vocabulary: any saved shape
        let accepted = Harness::Pi.fresh_line(Path::new("pi"), repo, "handoff-smoke-1a2b3c4d", &[]);
        assert!(accepted.is_ok());
    }

    #[test]
    fn resume_argv_speaks_each_harnesss_own_token() {
        let session = Path::new("/sessions/550e8400-e29b-41d4-a716-446655440000.jsonl");
        let pi = Harness::Pi.resume_line(
            Path::new("pi"),
            "550e8400-e29b-41d4-a716-446655440000",
            session,
            &[],
        );
        assert_eq!(
            pi,
            vec![
                "pi".to_string(),
                "--session".to_string(),
                "550e8400-e29b-41d4-a716-446655440000".to_string(),
                "--session-dir".to_string(),
                "/sessions".to_string(),
            ]
        );
        // claude-code resumes by the absolute transcript path; the id
        // rides the file it names
        let cc = Harness::ClaudeCode.resume_line(
            Path::new("/compile/claude-code"),
            "550e8400-e29b-41d4-a716-446655440000",
            session,
            &["continue the adapter".to_string()],
        );
        assert_eq!(
            cc,
            vec![
                "/compile/claude-code".to_string(),
                "--resume".to_string(),
                "/sessions/550e8400-e29b-41d4-a716-446655440000.jsonl".to_string(),
                "continue the adapter".to_string(),
            ]
        );
    }

    #[test]
    fn the_cc_transcript_reads_its_facts_from_the_first_carrying_record() {
        let dir = scratch("cc-read");
        let session = dir.join("550e8400-e29b-41d4-a716-446655440000.jsonl");
        std::fs::write(
            &session,
            cc_story("550e8400-e29b-41d4-a716-446655440000", "/repo/wt"),
        )
        .unwrap();
        let header = claude_code_header(&session).unwrap();
        // the transcript's own fields answer for identity and the
        // launch-time cwd; nothing durable is invented
        assert_eq!(
            header,
            SessionHeader {
                id: "550e8400-e29b-41d4-a716-446655440000".into(),
                cwd: "/repo/wt".into(),
            }
        );
        assert_eq!(
            std::fs::read(&session).unwrap(),
            cc_story("550e8400-e29b-41d4-a716-446655440000", "/repo/wt")
        );
        assert!(matches!(
            claude_code_header(&dir.join("absent.jsonl")),
            Err(HandoffFail::Missing(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_cc_cwd_is_the_launch_time_fallback_only() {
        // a session that moved: the first carrying record names where
        // it was born, and no later record rewrites that fact
        let dir = scratch("cc-moved");
        let session = dir.join("moved.jsonl");
        let mut bytes = cc_story("aaaa1111-1111-4111-8111-111111111111", "/repo/born");
        bytes.extend_from_slice(
            br#"{"type":"user","uuid":"44444444-4444-4444-8444-444444444444","sessionId":"aaaa1111-1111-4111-8111-111111111111","cwd":"/repo/moved","message":{"role":"user","content":"now from the new workspace"}}"#,
        );
        bytes.push(b'\n');
        std::fs::write(&session, &bytes).unwrap();
        let header = claude_code_header(&session).unwrap();
        assert_eq!(header.cwd, "/repo/born");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failed_cc_validation_names_its_variant_and_touches_nothing() {
        let dir = scratch("cc-refusals");
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            // an empty file carries no transcript
            ("empty.jsonl", Vec::new(), "Empty"),
            // binary noise parses to no record at all
            ("garbage.jsonl", b"\xff\xfe not json\n".to_vec(), "MalformedHeader"),
            // records without a sessionId carry no identity
            (
                "no-id.jsonl",
                br#"{"type":"user","cwd":"/repo/wt","message":{"role":"user","content":"who am I"}}"#.to_vec(),
                "MissingId",
            ),
            // records with an id but no cwd carry no launch fact
            (
                "no-cwd.jsonl",
                br#"{"type":"user","sessionId":"bbbb2222-2222-4222-8222-222222222222","message":{"role":"user","content":"where am I"}}"#.to_vec(),
                "MissingCwd",
            ),
            // an empty-string field is no fact
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
            // the original bytes survive the refused read
            assert_eq!(std::fs::read(&path).unwrap(), *bytes, "{name}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_pi_header_reads_through_the_same_face() {
        let dir = scratch("pi-delegate");
        let session = dir.join("session.jsonl");
        std::fs::write(
            &session,
            br#"{"type":"session","version":3,"id":"handoff-smoke-1a2b3c4d","cwd":"/repo/wt"}
{"type":"message","id":"4ba0a050","parentId":null}
"#,
        )
        .unwrap();
        let header = Harness::Pi.read_header(&session).unwrap();
        assert_eq!(
            header,
            SessionHeader {
                id: "handoff-smoke-1a2b3c4d".into(),
                cwd: "/repo/wt".into(),
            }
        );
        // claude-code validation refuses a pi session: its records
        // carry no sessionId
        let cc = Harness::ClaudeCode.read_header(&session).unwrap_err();
        assert!(cc.contains("carries no id"), "{cc}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn uuid_check_takes_the_saved_shapes() {
        assert!(is_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(is_uuid("AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"));
        assert!(!is_uuid("handoff-smoke-1a2b3c4d"));
        assert!(!is_uuid("550e8400e29b41d4a716446655440000"));
        assert!(!is_uuid("550e8400-e29b-41d4-a716-44665544000g"));
    }
}
