//! pi's RPC protocol, the subset saccade speaks: JSONL over stdio with
//! strict LF framing. Commands are prompt, steer, and abort; events are
//! agent_start, turn events, agent_settled, and the command responses.
//! The fake-Pi stub in tests/executor.rs is the contract; pi is an
//! implementation of it. Nothing here knows a process exists — the
//! codec is pure bytes.

use serde_json::{Value, json};

/// One command to the executor session, framed for its stdin.
#[derive(Clone, Debug, PartialEq)]
pub enum ClientCommand {
    Prompt { message: String },
    Steer { message: String },
    Abort,
}

impl ClientCommand {
    /// The wire frame: one JSON object, one LF.
    pub fn frame(&self) -> String {
        let value = match self {
            ClientCommand::Prompt { message } => json!({"type": "prompt", "message": message}),
            ClientCommand::Steer { message } => json!({"type": "steer", "message": message}),
            ClientCommand::Abort => json!({"type": "abort"}),
        };
        format!("{value}\n")
    }
}

/// An event from the executor session. The subset names what the
/// runner consumes; everything else arrives as `Other` and stays
/// runner-internal, never reaching a human-facing surface.
#[derive(Clone, Debug, PartialEq)]
pub enum ServerEvent {
    Response {
        command: String,
        success: bool,
        error: Option<String>,
    },
    AgentStart,
    TurnStart,
    TurnEnd,
    AgentSettled,
    Other(String),
}

impl ServerEvent {
    /// Parse one framed line. Lenient by design: an unknown event type
    /// is forward compatibility, not corruption.
    pub fn parse(line: &str) -> ServerEvent {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            return ServerEvent::Other("<unparseable>".into());
        };
        match value.get("type").and_then(|t| t.as_str()) {
            Some("response") => ServerEvent::Response {
                command: value
                    .get("command")
                    .and_then(|c| c.as_str())
                    .unwrap_or_default()
                    .to_string(),
                success: value
                    .get("success")
                    .and_then(|s| s.as_bool())
                    .unwrap_or(false),
                error: value
                    .get("error")
                    .and_then(|e| e.as_str())
                    .map(str::to_string),
            },
            Some("agent_start") => ServerEvent::AgentStart,
            Some("turn_start") => ServerEvent::TurnStart,
            Some("turn_end") => ServerEvent::TurnEnd,
            Some("agent_settled") => ServerEvent::AgentSettled,
            Some(other) => ServerEvent::Other(other.to_string()),
            None => ServerEvent::Other("<typeless>".into()),
        }
    }
}

/// Split framed lines out of a byte buffer: records end at LF only,
/// an optional trailing CR is accepted and stripped. Leftover bytes
/// without a terminator stay buffered for the next chunk.
pub fn frames(buffer: &mut Vec<u8>) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(i) = buffer.iter().position(|b| *b == b'\n') {
        let line: Vec<u8> = buffer.drain(..=i).collect();
        let mut line = &line[..line.len() - 1];
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        out.push(String::from_utf8_lossy(line).into_owned());
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    /// The codec's own contract: frames are strict LF, a trailing CR is
    /// tolerated, partial lines buffer, and every command parses back.
    #[test]
    fn framing_splits_on_lf_only_and_strips_one_cr() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"{\"a\":1}\r\n{\"b\":");
        assert_eq!(frames(&mut buffer), vec![r#"{"a":1}"#.to_string()]);
        buffer.extend_from_slice(b"2}\n\n{\"c\":\"x\\u2028y\"}\n");
        assert_eq!(
            frames(&mut buffer),
            vec![
                r#"{"b":2}"#.to_string(),
                String::new(),
                r#"{"c":"x\u2028y"}"#.to_string(),
            ]
        );
        // the U+2028 inside a string is one line's content, not a frame
        let parsed = ServerEvent::parse(r#"{"c":"x y"}"#);
        assert!(matches!(parsed, ServerEvent::Other(_)));
    }

    #[test]
    fn commands_frame_and_events_parse_over_the_subset() {
        assert_eq!(
            ClientCommand::Prompt {
                message: "do the work".into()
            }
            .frame(),
            "{\"message\":\"do the work\",\"type\":\"prompt\"}\n"
        );
        assert_eq!(
            ClientCommand::Steer {
                message: "change course".into()
            }
            .frame(),
            "{\"message\":\"change course\",\"type\":\"steer\"}\n"
        );
        assert_eq!(ClientCommand::Abort.frame(), "{\"type\":\"abort\"}\n");

        assert_eq!(
            ServerEvent::parse(r#"{"type":"response","command":"prompt","success":true}"#),
            ServerEvent::Response {
                command: "prompt".into(),
                success: true,
                error: None,
            }
        );
        assert_eq!(
            ServerEvent::parse(
                r#"{"type":"response","command":"steer","success":false,"error":"busy"}"#
            ),
            ServerEvent::Response {
                command: "steer".into(),
                success: false,
                error: Some("busy".into()),
            }
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"agent_start"}"#),
            ServerEvent::AgentStart
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"turn_start"}"#),
            ServerEvent::TurnStart
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"turn_end"}"#),
            ServerEvent::TurnEnd
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"agent_settled"}"#),
            ServerEvent::AgentSettled
        );
        // outside the subset: tolerated, opaque
        assert_eq!(
            ServerEvent::parse(r#"{"type":"queue_update","steering":["x"]}"#),
            ServerEvent::Other("queue_update".into())
        );
        assert_eq!(
            ServerEvent::parse("not json"),
            ServerEvent::Other("<unparseable>".into())
        );
    }
}
