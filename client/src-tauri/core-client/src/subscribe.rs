//! The deck's cue feed: one persistent `events.subscribe` connection,
//! read line by line, yielding only the doc events the caller names.
//! The pinned core 0.1.17 sends the snapshot reply first, then one
//! notification per line - `event` methods whose params carry
//! `{"type": "doc", "data": {"name", "version"}}` for a document
//! landing, heartbeats every 15 s that keep the read alive. Everything
//! else - other terminals' facts, other documents' events - is skipped
//! here, so the renderer never sees another name's traffic.

use crate::error::CoreFail;
use crate::transport::DEFAULT_READ_TIMEOUT;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// One document landing.
#[derive(Debug, Clone, PartialEq)]
pub struct DocEvent {
    pub name: String,
    pub version: u64,
}

#[derive(Debug)]
pub struct Subscription {
    path: PathBuf,
    reader: BufReader<UnixStream>,
    _writer: UnixStream,
}

fn socket_fail(path: &Path, e: std::io::Error) -> CoreFail {
    CoreFail::Socket {
        path: path.display().to_string(),
        detail: e.to_string(),
    }
}

impl Subscription {
    /// Open the subscription: one fresh connection, `events.subscribe`
    /// with no filters (doc events ride the stream whatever the
    /// terminal filters say), `on` omitted for the local core's own
    /// machine - the same rule every call here makes. The reply line
    /// is checked; the notifications follow.
    pub fn open(path: &Path, machine: &str) -> Result<Subscription, CoreFail> {
        let stream = UnixStream::connect(path).map_err(|e| socket_fail(path, e))?;
        stream
            .set_read_timeout(Some(DEFAULT_READ_TIMEOUT))
            .map_err(|e| socket_fail(path, e))?;
        let mut writer = stream.try_clone().map_err(|e| socket_fail(path, e))?;
        let mut params = json!({});
        if !machine.is_empty() {
            params["on"] = json!(machine);
        }
        writeln!(
            writer,
            "{}",
            json!({"jsonrpc": "2.0", "id": 1, "method": "events.subscribe", "params": params})
        )
        .map_err(|e| socket_fail(path, e))?;
        writer.flush().map_err(|e| socket_fail(path, e))?;
        let mut reader = BufReader::new(stream);
        // the snapshot reply: one line carrying an id. A missing or
        // refused reply is a refusal, never a silent feed
        let reply = Subscription::read_line(&mut reader, path)?;
        let reply = reply.ok_or_else(|| CoreFail::Protocol {
            detail: "the subscription closed before its reply".into(),
        })?;
        let parsed: Value = serde_json::from_str(&reply).map_err(|e| CoreFail::Protocol {
            detail: format!("the subscription reply is not JSON: {e}"),
        })?;
        if parsed.get("error").is_some() {
            return Err(CoreFail::Protocol {
                detail: "the core refused the subscription".into(),
            });
        }
        if parsed.get("result").is_none() {
            return Err(CoreFail::Protocol {
                detail: "the subscription reply carries neither result nor error".into(),
            });
        }
        Ok(Subscription {
            path: path.to_path_buf(),
            reader,
            _writer: writer,
        })
    }

    /// Set the read budget of the next bounded read (the deck feed's
    /// hold). The core's 15 s heartbeat sits far inside the 40 s
    /// default; a hold may ask for less, and a quiet expiry answers
    /// as [`CoreFail::HoldQuiet`], not as a socket failure.
    pub fn hold_timeout(&mut self, timeout: Duration) -> Result<(), CoreFail> {
        self.reader
            .get_ref()
            .set_read_timeout(Some(timeout))
            .map_err(|e| socket_fail(&self.path, e))
    }

    /// True when a failure is the bounded read's quiet expiry.
    pub fn timed_out(fail: &CoreFail) -> bool {
        matches!(fail, CoreFail::HoldQuiet)
    }

    /// One document landing for `name`, or None when the core closed
    /// the stream cleanly. Every other line - heartbeats, other
    /// terminals' facts, other documents' events - is skipped; a
    /// `missed` marker names the subscriber as fallen behind, and the
    /// caller's next doc.get answers whatever changed, so skipping it
    /// costs only the ordinary next event. A read error (the read
    /// budget is 40 s against a 15 s heartbeat) is the core stopping
    /// talking: an error, never silence.
    pub fn next_doc_event(&mut self, name: &str) -> Result<Option<DocEvent>, CoreFail> {
        loop {
            let line = Subscription::read_line(&mut self.reader, &self.path)?;
            let line = match line {
                None => return Ok(None),
                Some(line) => line,
            };
            let parsed: Value = match serde_json::from_str(&line) {
                Ok(parsed) => parsed,
                Err(_) => continue,
            };
            if parsed.get("method").and_then(Value::as_str) != Some("event") {
                continue;
            }
            let params = parsed.get("params").cloned().unwrap_or(Value::Null);
            if params.get("type").and_then(Value::as_str) != Some("doc") {
                continue;
            }
            let data = params.get("data").cloned().unwrap_or(Value::Null);
            let event_name = data.get("name").and_then(Value::as_str).unwrap_or("");
            // across a core-to-core hop the news is machine-qualified
            // (the pinned schema's hello: "news about t7 reaches it as
            // rohan/t7"), so the deck's own event arrives as
            // <machine>/<name> there; both forms match
            let matches = event_name == name || event_name.ends_with(&format!("/{name}"));
            if !matches {
                continue;
            }
            let version = data.get("version").and_then(Value::as_u64).unwrap_or(0);
            return Ok(Some(DocEvent {
                name: event_name.to_string(),
                version,
            }));
        }
    }

    fn read_line(
        reader: &mut BufReader<UnixStream>,
        path: &Path,
    ) -> Result<Option<String>, CoreFail> {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(line)),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                Err(CoreFail::HoldQuiet)
            }
            Err(e) => Err(socket_fail(path, e)),
        }
    }
}

/// The read budget, re-exported for the bridge's documentation: the
/// subscription's only timer is the socket's own read timeout, and
/// the core's 15 s heartbeat sits far inside it.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{BufRead, BufReader, Write};
    use std::sync::mpsc;

    /// A local test socket speaking the framing: it records the request
    /// line, writes the scripted reply and notifications, and closes.
    /// Never a core process.
    fn framed_socket(reply: Value, notifications: Vec<Value>) -> (PathBuf, mpsc::Receiver<String>) {
        let dir = std::env::temp_dir().join(format!(
            "deck-sub-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe.sock");
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let (request_tx, request_rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            let (server, _peer) = listener.accept().unwrap();
            let mut reader = BufReader::new(server.try_clone().unwrap());
            let mut line = String::new();
            if reader.read_line(&mut line).is_ok() {
                let _ = request_tx.send(line);
            }
            let mut writer = server;
            writeln!(writer, "{reply}").unwrap();
            for notification in &notifications {
                writeln!(writer, "{notification}").unwrap();
            }
        });
        (path, request_rx)
    }

    #[test]
    fn a_subscription_carries_only_the_named_documents_events() {
        let doc_event = json!({
            "jsonrpc": "2.0", "method": "event",
            "params": {"type": "doc", "seq": 2, "data": {"name": "deck-ab", "version": 4}}
        });
        let other_event = json!({
            "jsonrpc": "2.0", "method": "event",
            "params": {"type": "doc", "seq": 3, "data": {"name": "deck-cd", "version": 1}}
        });
        let terminal_event = json!({
            "jsonrpc": "2.0", "method": "event",
            "params": {"type": "status", "seq": 4, "data": {}}
        });
        let heartbeat = json!({"jsonrpc": "2.0", "method": "heartbeat", "params": {"seq": 5}});
        let (path, _requests) = framed_socket(
            json!({"jsonrpc": "2.0", "id": 1, "result": {"terminals": [], "seq": 1}}),
            vec![heartbeat, other_event, terminal_event, doc_event],
        );
        let mut subscription = Subscription::open(&path, "").unwrap();
        let event = subscription.next_doc_event("deck-ab").unwrap().unwrap();
        assert_eq!(event.name, "deck-ab");
        assert_eq!(event.version, 4);
    }

    #[test]
    fn a_local_subscription_omits_on_and_a_named_one_carries_it() {
        let reply = json!({"jsonrpc": "2.0", "id": 1, "result": {"terminals": [], "seq": 1}});
        let (local_path, local_requests) = framed_socket(reply.clone(), vec![]);
        Subscription::open(&local_path, "").unwrap();
        let request = local_requests
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(!request.contains("\"on\""), "{request}");
        let (named_path, named_requests) = framed_socket(reply, vec![]);
        Subscription::open(&named_path, "tired").unwrap();
        let request = named_requests
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(request.contains("\"on\":\"tired\""), "{request}");
    }

    #[test]
    fn a_refused_subscription_is_a_refusal_not_a_silent_feed() {
        let (path, _requests) = framed_socket(
            json!({"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "x"}}),
            vec![],
        );
        let opened = Subscription::open(&path, "");
        assert!(opened.is_err(), "{opened:?}");
    }

    #[test]
    fn a_relayed_event_names_the_document_through_its_machine_prefix() {
        let relayed = json!({
            "jsonrpc": "2.0", "method": "event",
            "params": {"type": "doc", "seq": 2,
                       "data": {"name": "b/deck-ab", "version": 4}}
        });
        let (path, _requests) = framed_socket(
            json!({"jsonrpc": "2.0", "id": 1, "result": {"terminals": [], "seq": 1}}),
            vec![relayed],
        );
        let mut subscription = Subscription::open(&path, "").unwrap();
        let event = subscription.next_doc_event("deck-ab").unwrap().unwrap();
        assert_eq!(event.version, 4);
        // another machine's document still does not match
        let (path2, _requests2) = framed_socket(
            json!({"jsonrpc": "2.0", "id": 1, "result": {"terminals": [], "seq": 1}}),
            vec![json!({
                "jsonrpc": "2.0", "method": "event",
                "params": {"type": "doc", "seq": 2,
                           "data": {"name": "b/deck-cd", "version": 4}}
            })],
        );
        let mut subscription2 = Subscription::open(&path2, "").unwrap();
        assert!(subscription2.next_doc_event("deck-ab").unwrap().is_none());
    }

    #[test]
    fn a_cleanly_closed_stream_ends_the_feed() {
        let (path, _requests) = framed_socket(
            json!({"jsonrpc": "2.0", "id": 1, "result": {"terminals": [], "seq": 1}}),
            vec![],
        );
        let mut subscription = Subscription::open(&path, "").unwrap();
        let event = subscription.next_doc_event("deck-ab").unwrap();
        assert!(event.is_none());
    }
}
