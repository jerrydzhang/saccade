//! Native Pi session-file mechanics for workspace handoff: one saved
//! session (JSONL in Pi's documented session format) keeps its native
//! identity while the cwd in its first-line header moves to a selected
//! workspace. This is an adapter over the documented format, not a
//! native API: the caller owns process quiescence and the ordinary
//! native startup (`--session <full-id> --session-dir <dir>` launched
//! from the selected workspace). One file is touched; the record, the
//! tracker, and any running Pi are not.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::types::pointers::WorktreePath;

/// A saved session's validated header facts: its native id and cwd.
#[derive(Debug, PartialEq)]
pub struct SessionHeader {
    pub id: String,
    pub cwd: String,
}

/// A handoff refusal; every variant leaves the session file untouched.
#[derive(Debug, thiserror::Error)]
pub enum HandoffFail {
    #[error("no session file at {0}")]
    Missing(PathBuf),
    #[error("session file {0} is empty")]
    Empty(PathBuf),
    #[error("session header at {0} is not UTF-8")]
    NotUtf8(PathBuf),
    #[error("session file {path} could not be read: {detail}")]
    Unreadable { path: PathBuf, detail: String },
    #[error("first line of {path} is not a JSON object: {detail}")]
    MalformedHeader { path: PathBuf, detail: String },
    #[error("first line of {path} is not a session header: type {found}")]
    NotASession { path: PathBuf, found: String },
    #[error("session header at {0} carries no id")]
    MissingId(PathBuf),
    #[error("session at {path} carries id {found:?}, not {expected:?}")]
    IdMismatch {
        path: PathBuf,
        expected: String,
        found: String,
    },
    #[error("session header at {0} carries no cwd")]
    MissingCwd(PathBuf),
    #[error("session header at {0} carries no usable cwd value")]
    InvalidCwd(PathBuf),
    #[error("retargeting {path} failed, original untouched: {detail}")]
    Write { path: PathBuf, detail: String },
}

/// The saved session's header facts.
pub fn read_header(path: &Path) -> Result<SessionHeader, HandoffFail> {
    Ok(load(path)?.facts)
}

/// Move one stopped session's header cwd to the selected workspace
/// address, replacing the file atomically and preserving its
/// permissions; the id, every other header field, and all non-header
/// bytes are unchanged. Returns the cwd the session carried before.
///
/// The caller confirms the previous native process has exited before
/// calling; a readable file establishes no quiescence.
pub fn retarget_cwd(
    path: &Path,
    session_id: &str,
    workspace: &WorktreePath,
) -> Result<String, HandoffFail> {
    let loaded = load(path)?;
    if loaded.facts.id != session_id {
        return Err(HandoffFail::IdMismatch {
            path: path.to_path_buf(),
            expected: session_id.to_string(),
            found: loaded.facts.id,
        });
    }
    let previous = loaded.facts.cwd.clone();
    let bytes = retargeted_bytes(path, loaded, workspace)?;
    atomic_replace(path, bytes)?;
    Ok(previous)
}

struct LoadedSession {
    facts: SessionHeader,
    header: Map<String, Value>,
    line_ended_with_cr: bool,
    rest: Vec<u8>,
}

fn load(path: &Path) -> Result<LoadedSession, HandoffFail> {
    let bytes = match fs::read(path) {
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
    let (line, rest) = match bytes.iter().position(|&b| b == b'\n') {
        Some(at) => (&bytes[..at], bytes[at..].to_vec()),
        None => (&bytes[..], Vec::new()),
    };
    let line_ended_with_cr = line.ends_with(b"\r");
    let text = std::str::from_utf8(line).map_err(|_| HandoffFail::NotUtf8(path.to_path_buf()))?;
    let value: Value = serde_json::from_str(text).map_err(|e| HandoffFail::MalformedHeader {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    let header = match value {
        Value::Object(map) => map,
        _ => {
            return Err(HandoffFail::MalformedHeader {
                path: path.to_path_buf(),
                detail: "not a JSON object".to_string(),
            });
        }
    };
    let facts = validate(path, &header)?;
    Ok(LoadedSession {
        facts,
        header,
        line_ended_with_cr,
        rest,
    })
}

fn validate(path: &Path, header: &Map<String, Value>) -> Result<SessionHeader, HandoffFail> {
    match header.get("type") {
        Some(Value::String(t)) if t == "session" => {}
        other => {
            return Err(HandoffFail::NotASession {
                path: path.to_path_buf(),
                found: other
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(absent)".into()),
            });
        }
    }
    let id = match header.get("id") {
        Some(Value::String(id)) if !id.is_empty() => id.clone(),
        _ => return Err(HandoffFail::MissingId(path.to_path_buf())),
    };
    let cwd = match header.get("cwd") {
        None => return Err(HandoffFail::MissingCwd(path.to_path_buf())),
        Some(Value::String(cwd)) if !cwd.is_empty() => cwd.clone(),
        _ => return Err(HandoffFail::InvalidCwd(path.to_path_buf())),
    };
    Ok(SessionHeader { id, cwd })
}

fn retargeted_bytes(
    path: &Path,
    mut loaded: LoadedSession,
    workspace: &WorktreePath,
) -> Result<Vec<u8>, HandoffFail> {
    loaded.header.insert(
        "cwd".into(),
        Value::String(workspace.as_path().to_string_lossy().into_owned()),
    );
    let mut line =
        serde_json::to_vec(&Value::Object(loaded.header)).map_err(|e| HandoffFail::Write {
            path: path.to_path_buf(),
            detail: e.to_string(),
        })?;
    if loaded.line_ended_with_cr {
        line.push(b'\r');
    }
    let mut bytes = line;
    bytes.extend_from_slice(&loaded.rest);
    Ok(bytes)
}

fn atomic_replace(path: &Path, bytes: Vec<u8>) -> Result<(), HandoffFail> {
    let fail = |detail: String| HandoffFail::Write {
        path: path.to_path_buf(),
        detail,
    };
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let permissions = fs::metadata(path)
        .map_err(|e| fail(e.to_string()))?
        .permissions();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let temp = dir.unwrap_or_else(|| Path::new(".")).join(format!(
        ".saccade-retarget-{}-{}.tmp",
        std::process::id(),
        nanos
    ));
    let written = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.set_permissions(permissions)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        if let Some(dir) = dir
            && let Ok(handle) = fs::File::open(dir)
        {
            let _ = handle.sync_all();
        }
        Ok(())
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temp);
    }
    written.map_err(|e: std::io::Error| fail(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("saccade-pi-session-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn workspace(path: &str) -> WorktreePath {
        WorktreePath::new(PathBuf::from(path)).unwrap()
    }

    fn write_session(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
    }

    fn dir_files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    fn header_of(bytes: &[u8]) -> Value {
        let line = bytes.split(|&b| b == b'\n').next().unwrap();
        serde_json::from_slice(line).unwrap()
    }

    // A native-shaped session: header with an extra field, then a
    // story of tree-linked entries through a final newline.
    fn story_session() -> Vec<u8> {
        let header = r#"{"type":"session","version":3,"id":"handoff-smoke-1a2b3c4d","timestamp":"2026-10-21T03:40:04.770Z","cwd":"/story/worktrees/implement-handoff","parentSession":"/story/sessions/earlier.jsonl"}"#;
        let entries = [
            r#"{"type":"session_info","id":"f48a46a4","parentId":null,"timestamp":"2026-10-21T03:40:04.770Z","name":"handoff smoke run"}"#,
            r#"{"type":"message","id":"4ba0a050","parentId":null,"timestamp":"2026-10-21T03:40:04.880Z","message":{"role":"user","content":"implement the handoff helper","timestamp":1761008404880}}"#,
            r#"{"type":"message","id":"d4e5f6g7","parentId":"4ba0a050","timestamp":"2026-10-21T03:40:05.100Z","message":{"role":"assistant","content":[{"type":"text","text":"starting from the helper module"}],"api":"anthropic-messages","provider":"anthropic","model":"claude-sonnet-4-5","usage":{},"stopReason":"stop","timestamp":1761008405100}}"#,
            r#"{"type":"model_change","id":"520892be","parentId":"d4e5f6g7","timestamp":"2026-10-21T03:40:06.000Z","provider":"anthropic","modelId":"claude-opus-4-6"}"#,
            r#"{"type":"message","id":"ae18ab9e","parentId":"520892be","timestamp":"2026-10-21T03:40:07.000Z","message":{"role":"user","content":"switch to the selected workspace now","timestamp":1761008407000}}"#,
        ];
        let mut bytes = header.as_bytes().to_vec();
        for entry in entries {
            bytes.push(b'\n');
            bytes.extend_from_slice(entry.as_bytes());
        }
        bytes.push(b'\n');
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
            HandoffFail::IdMismatch { .. } => "IdMismatch",
            HandoffFail::MissingCwd(_) => "MissingCwd",
            HandoffFail::InvalidCwd(_) => "InvalidCwd",
            HandoffFail::Write { .. } => "Write",
        }
    }

    #[test]
    fn retarget_moves_only_the_header_cwd_between_workspaces() {
        let dir = scratch("retarget");
        let session = dir.join("2026-10-21T03-40-04-770Z_handoff-smoke-1a2b3c4d.jsonl");
        write_session(&session, &story_session());
        let original = fs::read(&session).unwrap();
        let select = workspace("/story/worktrees/select-workspace");

        let previous = retarget_cwd(&session, "handoff-smoke-1a2b3c4d", &select).unwrap();
        // the reply states the cwd the session carried before the move
        assert_eq!(previous, "/story/worktrees/implement-handoff");

        let rewritten = fs::read(&session).unwrap();
        let header = header_of(&rewritten);
        // the session keeps its native id through the move
        assert_eq!(header["id"], "handoff-smoke-1a2b3c4d");
        // the first line stays a session header
        assert_eq!(header["type"], "session");
        // the header cwd now names the selected workspace address
        assert_eq!(header["cwd"], select.as_path().to_str().unwrap());
        // header fields outside cwd, known and unknown, keep their values
        assert_eq!(header["version"], 3);
        assert_eq!(header["timestamp"], "2026-10-21T03:40:04.770Z");
        assert_eq!(header["parentSession"], "/story/sessions/earlier.jsonl");
        // the conversation body after the header line is byte-identical
        let body_at = original.iter().position(|&b| b == b'\n').unwrap();
        assert!(rewritten.ends_with(&original[body_at..]));
        // the rewrite leaves exactly the session file behind, no temporary
        assert_eq!(
            dir_files(&dir),
            vec![session.file_name().unwrap().to_string_lossy()]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn retarget_preserves_the_first_line_terminator_shape() {
        let dir = scratch("terminators");
        let with_newline = dir.join("with-newline.jsonl");
        write_session(&with_newline, &story_session());
        retarget_cwd(
            &with_newline,
            "handoff-smoke-1a2b3c4d",
            &workspace("/story/wt-b"),
        )
        .unwrap();
        // a session ending in a newline still ends in a newline
        assert!(fs::read(&with_newline).unwrap().ends_with(b"\n"));

        let mut bare = story_session();
        bare.truncate(bare.iter().position(|&b| b == b'\n').unwrap());
        let header_only = dir.join("header-only.jsonl");
        write_session(&header_only, &bare);
        retarget_cwd(
            &header_only,
            "handoff-smoke-1a2b3c4d",
            &workspace("/story/wt-b"),
        )
        .unwrap();
        let rewritten = fs::read(&header_only).unwrap();
        // a header-only session keeps the absence of a trailing newline
        assert!(!rewritten.ends_with(b"\n"));
        let header = header_of(&rewritten);
        assert_eq!(header["cwd"], "/story/wt-b");
        assert_eq!(header["id"], "handoff-smoke-1a2b3c4d");
        assert_eq!(header["parentSession"], "/story/sessions/earlier.jsonl");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn retarget_twice_lands_on_one_session_file() {
        let dir = scratch("twice");
        let session = dir.join("session.jsonl");
        write_session(&session, &story_session());
        let select = workspace("/story/worktrees/select-workspace");
        retarget_cwd(&session, "handoff-smoke-1a2b3c4d", &select).unwrap();
        retarget_cwd(&session, "handoff-smoke-1a2b3c4d", &select).unwrap();
        // a second retarget replaces in place, never a second session file
        assert_eq!(dir_files(&dir), vec!["session.jsonl"]);
        assert_eq!(
            header_of(&fs::read(&session).unwrap())["cwd"],
            select.as_path().to_str().unwrap()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failed_validation_leaves_the_original_session_file_untouched() {
        let dir = scratch("refusals");
        let select = workspace("/story/worktrees/select-workspace");
        let expected_id = "handoff-smoke-1a2b3c4d";
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            // a conversation line first is not a JSON header
            (
                "not-json.jsonl",
                b"{\"type\":\"message\"\nrest".to_vec(),
                "MalformedHeader",
            ),
            // a scalar first line is not a header object
            ("scalar.jsonl", b"42\n".to_vec(), "MalformedHeader"),
            // a message entry first is not a session header
            (
                "wrong-type.jsonl",
                br#"{"type":"message","id":"4ba0a050"}"#.to_vec(),
                "NotASession",
            ),
            // a header naming another session refuses to move
            (
                "wrong-id.jsonl",
                br#"{"type":"session","id":"other-9988ccdd","cwd":"/story/wt-a"}"#.to_vec(),
                "IdMismatch",
            ),
            // a header with no id refuses to move
            (
                "no-id.jsonl",
                br#"{"type":"session","cwd":"/story/wt-a"}"#.to_vec(),
                "MissingId",
            ),
            // a header with no cwd refuses to move
            (
                "no-cwd.jsonl",
                br#"{"type":"session","id":"handoff-smoke-1a2b3c4d"}"#.to_vec(),
                "MissingCwd",
            ),
            // a header with a null cwd refuses to move
            (
                "null-cwd.jsonl",
                br#"{"type":"session","id":"handoff-smoke-1a2b3c4d","cwd":null}"#.to_vec(),
                "InvalidCwd",
            ),
            // a header with an empty-string cwd refuses to move
            (
                "empty-cwd.jsonl",
                br#"{"type":"session","id":"handoff-smoke-1a2b3c4d","cwd":""}"#.to_vec(),
                "InvalidCwd",
            ),
            // an empty file carries no header
            ("empty.jsonl", Vec::new(), "Empty"),
        ];
        for (name, bytes, expected) in &cases {
            let path = dir.join(name);
            write_session(&path, bytes);
            let outcome = retarget_cwd(&path, expected_id, &select);
            // the refusal names its variant
            assert_eq!(variant(&outcome.unwrap_err()), *expected, "{name}");
            // the original bytes survive every refused move
            assert_eq!(fs::read(&path).unwrap(), *bytes, "{name}");
        }
        // a missing session file refuses without creating one
        let missing = dir.join("absent.jsonl");
        assert!(matches!(
            retarget_cwd(&missing, expected_id, &select),
            Err(HandoffFail::Missing(_))
        ));
        // refused moves left no temporary files behind
        assert_eq!(dir_files(&dir).len(), cases.len());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_header_reports_the_session_facts() {
        let dir = scratch("read");
        let session = dir.join("session.jsonl");
        write_session(&session, &story_session());
        let header = read_header(&session).unwrap();
        // inspection reads the native id and cwd without touching bytes
        assert_eq!(
            header,
            SessionHeader {
                id: "handoff-smoke-1a2b3c4d".into(),
                cwd: "/story/worktrees/implement-handoff".into(),
            }
        );
        assert_eq!(fs::read(&session).unwrap(), story_session());
        assert!(matches!(
            read_header(&dir.join("absent.jsonl")),
            Err(HandoffFail::Missing(_))
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn retarget_preserves_file_access_permissions() {
        let dir = scratch("permissions");
        let session = dir.join("session.jsonl");
        write_session(&session, &story_session());
        fs::set_permissions(&session, fs::Permissions::from_mode(0o600)).unwrap();
        retarget_cwd(
            &session,
            "handoff-smoke-1a2b3c4d",
            &workspace("/story/wt-b"),
        )
        .unwrap();
        // the moved session keeps its access mode
        let mode = fs::metadata(&session).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        fs::remove_dir_all(&dir).unwrap();
    }
}
