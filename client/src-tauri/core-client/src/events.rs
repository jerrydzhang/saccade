//! The event stream's read face: `events.subscribe` over the local
//! core's socket, one persistent connection per opener. The client
//! holds ONE subscription per core per app process (t-24@r22: on
//! herdr-core 0.1.17 a second concurrent `events.subscribe` on one
//! core never answers), and fans what arrives out to every window and
//! instrument; nothing else here may open a second one. Read-only by
//! construction: a subscription sends nothing to any terminal.

use crate::error::CoreFail;
use crate::transport::SocketTransport;
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// The response id of the subscription's request line.
const REQUEST_ID: u64 = 1;

/// The bounded time the subscribe request itself may take: the core
/// answers a subscription at once with its snapshot, so a quiet read
/// past this is a socket failure, not a quiet stream.
const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(10);

/// One event off the stream, the fields the instruments read. The
/// pinned core wraps each event in a JSON-RPC notification with the
/// payload under `params`; a bare payload shape is accepted too (the
/// same dialect `hc events --json` prints). Unknown event types decode
/// to `Other`: the feed carries them, the instruments ignore them.
#[derive(Debug, Clone, PartialEq)]
pub enum CoreEvent {
    /// The subscription's own first answer: the terminal table as it
    /// stands. The listing carries no words (t-21@r1) - words fill as
    /// status events arrive.
    Snapshot { terminals: Vec<SnapshotTerminal> },
    /// A status change: the program's word and its own message,
    /// progress and title exactly as sent (a missing field is absent,
    /// never invented). The exit nulls the word: a status with no
    /// status arrives just before `finished`.
    Status {
        terminal: String,
        state: Option<String>,
        app: Option<String>,
        msg: Option<String>,
        progress: Option<String>,
        title: Option<String>,
    },
    /// The terminal finished, with the core's reason.
    Finished { terminal: String, reason: String },
    /// The core announced a restart (`restarted`, `lost`): kept
    /// terminals come back under their addresses; the feed's consumer
    /// re-derives its table from the next snapshot.
    CoreRestarted,
    /// An event this build does not interpret. It rides the stream so
    /// a later consumer (the deck's document events, t-21) can join
    /// the fan-out without a second subscription.
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotTerminal {
    pub address: String,
    pub state: String,
}

impl CoreEvent {
    /// True when the event can change an instrument's live join.
    pub fn names_terminal(&self) -> Option<&str> {
        match self {
            CoreEvent::Status { terminal, .. } => Some(terminal),
            CoreEvent::Finished { terminal, .. } => Some(terminal),
            _ => None,
        }
    }
}

/// The plain value a field carries as text, exactly as sent: strings
/// as strings, numbers and booleans in their JSON form, null and
/// missing as absent. Objects render as compact JSON - the program
/// sent a shape, not a sentence.
fn field_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Number(number) => Some(number.to_string()),
        other => Some(other.to_string()),
    }
}

/// Parse one stream line. The first line of a subscription is the
/// JSON-RPC response (the snapshot); every later line is a
/// notification. An error envelope on the response is a failure.
pub fn parse_line(line: &str, first: bool) -> Result<CoreEvent, CoreFail> {
    let value: Value = serde_json::from_str(line).map_err(|e| CoreFail::Protocol {
        detail: format!("event line is not JSON: {e}"),
    })?;
    if first {
        if let Some(error) = value.get("error") {
            return Err(CoreFail::Protocol {
                detail: format!("events.subscribe refused: {error}"),
            });
        }
        let result = value.get("result").ok_or_else(|| CoreFail::Protocol {
            detail: "the subscribe answer carries no result".into(),
        })?;
        let terminals = result
            .get("terminals")
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| {
                        Some(SnapshotTerminal {
                            address: row.get("address")?.as_str()?.to_string(),
                            state: row
                                .get("state")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown")
                                .to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        return Ok(CoreEvent::Snapshot { terminals });
    }
    let value = value.get("params").unwrap_or(&value);
    let terminal = value
        .get("terminal")
        .and_then(Value::as_str)
        .map(str::to_string);
    match value.get("type").and_then(Value::as_str) {
        Some("restarted") | Some("lost") => Ok(CoreEvent::CoreRestarted),
        Some("status") => {
            let Some(terminal) = terminal else {
                return Ok(CoreEvent::Other);
            };
            let status = value
                .get("data")
                .and_then(|d| d.get("status"))
                .and_then(|s| s.as_object());
            Ok(CoreEvent::Status {
                terminal,
                state: field_text(status.and_then(|s| s.get("state"))),
                app: field_text(status.and_then(|s| s.get("app"))),
                msg: field_text(status.and_then(|s| s.get("msg"))),
                progress: field_text(status.and_then(|s| s.get("progress"))),
                title: field_text(status.and_then(|s| s.get("title"))),
            })
        }
        Some("finished") => {
            let Some(terminal) = terminal else {
                return Ok(CoreEvent::Other);
            };
            Ok(CoreEvent::Finished {
                terminal,
                reason: value
                    .get("data")
                    .and_then(|d| d.get("reason"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
            })
        }
        _ => Ok(CoreEvent::Other),
    }
}

/// One open subscription: the socket connection, owned for the feed's
/// life. `next_event` returns `None` at a clean end (EOF); a framing
/// or socket failure is an error and the connection is done - the
/// consumer reconnects by opening a fresh subscription.
pub struct EventsConnection {
    reader: BufReader<UnixStream>,
    stream: UnixStream,
}

impl EventsConnection {
    /// Open one subscription on the core's socket: the request line,
    /// then the snapshot as the first read. ONE per core per process.
    pub fn open(socket: &SocketTransport) -> Result<Self, CoreFail> {
        Self::open_on(socket, |path, timeout| {
            let stream = UnixStream::connect(path)?;
            stream.set_read_timeout(Some(timeout))?;
            Ok(stream)
        })
    }

    /// The open, over a connect step the tests hold: the framing is
    /// what the contract pins, not how the socket arrives.
    pub fn open_on<F>(socket: &SocketTransport, connect: F) -> Result<Self, CoreFail>
    where
        F: FnOnce(&std::path::Path, Duration) -> std::io::Result<UnixStream>,
    {
        let path = socket.path().to_path_buf();
        let mut stream = connect(&path, SUBSCRIBE_TIMEOUT).map_err(|e| CoreFail::Socket {
            path: path.display().to_string(),
            detail: format!("connect: {e}"),
        })?;
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": REQUEST_ID,
            "method": "events.subscribe",
            "params": {},
        });
        stream
            .set_write_timeout(Some(SUBSCRIBE_TIMEOUT))
            .map_err(|e| CoreFail::Socket {
                path: path.display().to_string(),
                detail: format!("write timeout: {e}"),
            })?;
        writeln!(stream, "{request}").map_err(|e| CoreFail::Socket {
            path: path.display().to_string(),
            detail: format!("subscribe write: {e}"),
        })?;
        // the subscription's life is unbounded: the request's bounded
        // timeout hands over to the consumer's own read policy
        stream
            .set_read_timeout(None)
            .map_err(|e| CoreFail::Socket {
                path: path.display().to_string(),
                detail: format!("read timeout: {e}"),
            })?;
        let reader = BufReader::new(stream.try_clone().map_err(|e| CoreFail::Socket {
            path: path.display().to_string(),
            detail: format!("clone: {e}"),
        })?);
        Ok(Self { reader, stream })
    }

    /// The next event: `None` at a clean EOF, an error on a broken
    /// frame. The first read is the subscription's snapshot.
    pub fn next_event(&mut self, first: bool) -> Result<CoreEvent, CoreFail> {
        let mut line = String::new();
        let read = self
            .reader
            .read_line(&mut line)
            .map_err(|e| CoreFail::Socket {
                path: String::new(),
                detail: format!("stream read: {e}"),
            })?;
        if read == 0 {
            return Ok(CoreEvent::Other);
        }
        parse_line(line.trim_end_matches(['\n', '\r']), first)
    }

    /// The owned socket, for the tests' cleanup.
    pub fn into_stream(self) -> UnixStream {
        self.stream
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::thread;

    /// One fake core: answers the subscribe line with the snapshot,
    /// then prints the given lines. The socket lives in a fresh temp
    /// directory, retained (no deletion sweeps).
    fn fake_core(lines: &[String]) -> (std::path::PathBuf, std::path::PathBuf) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .subsec_nanos();
        // a short name: a Unix socket path must fit SUN_LEN; unique
        // creation, retained (no deletion sweeps)
        let dir = std::env::temp_dir().join(format!("sace-{}-{}-{}", std::process::id(), n, nanos));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("test.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        let served = lines.to_vec();
        thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut writer = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream);
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            assert!(request.contains("events.subscribe"), "{request}");
            for line in served {
                writeln!(writer, "{line}").unwrap();
            }
        });
        (socket_path, dir)
    }

    fn transport(path: &std::path::Path) -> SocketTransport {
        SocketTransport::new(path.to_path_buf())
    }

    fn open_at(path: &std::path::Path) -> EventsConnection {
        EventsConnection::open_on(&transport(path), |path, timeout| {
            let stream = UnixStream::connect(path)?;
            stream.set_read_timeout(Some(timeout))?;
            Ok(stream)
        })
        .unwrap()
    }

    #[test]
    fn the_subscribe_request_names_the_method_and_the_snapshot_answers_first() {
        let snapshot = r#"{"jsonrpc":"2.0","id":1,"result":{"channel":null,"seq":83,"terminals":[{"address":"t1","state":"live"},{"address":"t2","state":"live"}]}}"#;
        let status = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":12,"terminal":"t2","type":"status","data":{"records":{},"reported":true,"status":{"state":"working","app":"pi","msg":"editing src","kind":null,"progress":42,"title":null}}}}"#;
        let (path, _dir) = fake_core(&[snapshot.into(), status.into()]);
        let mut connection = open_at(&path);
        let event = connection.next_event(true).unwrap();
        assert_eq!(
            event,
            CoreEvent::Snapshot {
                terminals: vec![
                    SnapshotTerminal {
                        address: "t1".into(),
                        state: "live".into()
                    },
                    SnapshotTerminal {
                        address: "t2".into(),
                        state: "live".into()
                    },
                ]
            }
        );
        let event = connection.next_event(false).unwrap();
        assert_eq!(
            event,
            CoreEvent::Status {
                terminal: "t2".into(),
                state: Some("working".into()),
                app: Some("pi".into()),
                msg: Some("editing src".into()),
                progress: Some("42".into()),
                title: None,
            }
        );
    }

    #[test]
    fn the_exit_nulls_the_word_and_finished_names_its_reason() {
        let snapshot = r#"{"jsonrpc":"2.0","id":1,"result":{"terminals":[]}}"#;
        let exit = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":13,"terminal":"t2","type":"status","data":{"status":null}}}"#;
        let finished = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":14,"terminal":"t2","type":"finished","data":{"exit_code":0,"finished_ms":42,"reason":"exited","signal":null}}}"#;
        let (path, _dir) = fake_core(&[snapshot.into(), exit.into(), finished.into()]);
        let mut connection = open_at(&path);
        assert!(matches!(
            connection.next_event(true).unwrap(),
            CoreEvent::Snapshot { .. }
        ));
        assert_eq!(
            connection.next_event(false).unwrap(),
            CoreEvent::Status {
                terminal: "t2".into(),
                state: None,
                app: None,
                msg: None,
                progress: None,
                title: None,
            }
        );
        assert_eq!(
            connection.next_event(false).unwrap(),
            CoreEvent::Finished {
                terminal: "t2".into(),
                reason: "exited".into(),
            }
        );
    }

    #[test]
    fn the_restart_vocabulary_is_stream_level_and_unknown_types_ride_through() {
        let snapshot = r#"{"jsonrpc":"2.0","id":1,"result":{"terminals":[]}}"#;
        let restarted =
            r#"{"jsonrpc":"2.0","method":"event","params":{"seq":18,"type":"restarted"}}"#;
        let lost = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":19,"type":"lost"}}"#;
        let created = r#"{"jsonrpc":"2.0","method":"event","params":{"seq":20,"terminal":"t9","type":"created","data":{}}}"#;
        let (path, _dir) = fake_core(&[
            snapshot.into(),
            restarted.into(),
            lost.into(),
            created.into(),
        ]);
        let mut connection = open_at(&path);
        let _ = connection.next_event(true).unwrap();
        assert_eq!(
            connection.next_event(false).unwrap(),
            CoreEvent::CoreRestarted
        );
        assert_eq!(
            connection.next_event(false).unwrap(),
            CoreEvent::CoreRestarted
        );
        assert_eq!(connection.next_event(false).unwrap(), CoreEvent::Other);
    }

    #[test]
    fn a_refused_subscription_is_an_error_and_a_bare_payload_shape_decodes() {
        let refused =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"no second subscription"}}"#;
        let (path, _dir) = fake_core(&[refused.into()]);
        let mut connection = open_at(&path);
        let error = connection.next_event(true).unwrap_err();
        assert!(error.to_string().contains("no second subscription"));
        // the bare shape the CLI prints decodes the same
        let bare =
            r#"{"seq":12,"terminal":"t2","type":"status","data":{"status":{"state":"idle"}}}"#;
        assert_eq!(
            parse_line(bare, false).unwrap(),
            CoreEvent::Status {
                terminal: "t2".into(),
                state: Some("idle".into()),
                app: None,
                msg: None,
                progress: None,
                title: None,
            }
        );
    }
}
