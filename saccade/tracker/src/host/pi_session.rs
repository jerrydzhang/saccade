//! Native Pi session-file mechanics: the door-owned session dir, fresh
//! native ids, and the header facts ordinary resume validates. An
//! adapter over Pi's documented session format, not a native API. The
//! caller owns process quiescence and the native startup itself; this
//! module reads one file, never the record or the tracker.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde_json::{Map, Value};

use crate::paths;

/// The door-owned directory native sessions save into: under the
/// tracker's state root, so the launch door knows the exact tree and
/// the locator binds against it.
pub fn sessions_dir(repo_root: &std::path::Path) -> std::path::PathBuf {
    paths::state_dir(repo_root).join("native-sessions")
}

/// A fresh native session id, uuid v4: pi creates the exact session
/// under it, and the launch door's birth carries it as the durable
/// identity - no correlation token, no attach wait.
pub fn generate_session_id() -> String {
    let mut bytes = [0u8; 16];
    // exactly sixteen bytes: /dev/urandom never ends, so anything
    // read-to-EOF here would allocate forever
    let filled = std::fs::File::open("/dev/urandom")
        .is_ok_and(|mut file| std::io::Read::read_exact(&mut file, &mut bytes).is_ok());
    if !filled {
        // no urandom: nanos and pid distinguish concurrent doors well
        // enough, and pi refuses a colliding id rather than merging
        let nanos = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mix = nanos ^ ((std::process::id() as u128) << 64);
        for (at, piece) in bytes.chunks_mut(8).enumerate() {
            let shift = 8 * at;
            for (i, slot) in piece.iter_mut().enumerate() {
                *slot = (mix >> (shift + 8 * i)) as u8;
            }
        }
    }
    // uuid v4 version and variant bits
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

/// A saved session's validated header facts: its native id and the
/// cwd it was launched at. The cwd answers only for sessions whose
/// fold carries no workspace change; a recorded change is the
/// authority, and the file is never rewritten to follow it.
#[derive(Debug, PartialEq)]
pub struct SessionHeader {
    pub id: String,
    pub cwd: String,
}

/// A session-file refusal; every variant leaves the file untouched.
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
    #[error("session header at {0} carries no cwd")]
    MissingCwd(PathBuf),
    #[error("session header at {0} carries no usable cwd value")]
    InvalidCwd(PathBuf),
}

/// Read a saved session's validated header facts; the bytes are never
/// written back.
pub fn read_header(path: &Path) -> Result<SessionHeader, HandoffFail> {
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
    let line = match bytes.iter().position(|&b| b == b'\n') {
        Some(at) => &bytes[..at],
        None => &bytes[..],
    };
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
    validate(path, &header)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("saccade-pi-session-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_session(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
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

    // a native-shaped session: header with an extra field, then a story of tree-linked entries
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
    fn failed_validation_names_its_variant_and_touches_nothing() {
        let dir = scratch("refusals");
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
            // a header with no id carries no identity
            (
                "no-id.jsonl",
                br#"{"type":"session","cwd":"/story/wt-a"}"#.to_vec(),
                "MissingId",
            ),
            // a header with no cwd carries no launch fact
            (
                "no-cwd.jsonl",
                br#"{"type":"session","id":"handoff-smoke-1a2b3c4d"}"#.to_vec(),
                "MissingCwd",
            ),
            // a header with a null cwd carries no launch fact
            (
                "null-cwd.jsonl",
                br#"{"type":"session","id":"handoff-smoke-1a2b3c4d","cwd":null}"#.to_vec(),
                "InvalidCwd",
            ),
            // a header with an empty-string cwd carries no launch fact
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
            let outcome = read_header(&path);
            // the refusal names its variant
            assert_eq!(variant(&outcome.unwrap_err()), *expected, "{name}");
            // the original bytes survive the refused read
            assert_eq!(fs::read(&path).unwrap(), *bytes, "{name}");
        }
        // a missing session file refuses without creating one
        let missing = dir.join("absent.jsonl");
        assert!(matches!(
            read_header(&missing),
            Err(HandoffFail::Missing(_))
        ));
        fs::remove_dir_all(&dir).unwrap();
    }
}
