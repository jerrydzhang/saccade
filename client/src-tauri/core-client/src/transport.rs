//! The wire: JSON-RPC 2.0, one message per line, over the local core's
//! Unix socket - the pinned herdr-core 0.1.17 socket face (`hc schema`:
//! "one message per line on the hc socket (or framed after `hello`)";
//! framed-after-hello exists for core-to-core links, not for a
//! same-user local client). One fresh connection per call: stateless on
//! the viewer's side, nothing to supervise or reap.

use crate::error::CoreFail;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The response id of a fresh per-call connection.
const REQUEST_ID: u64 = 1;

/// How long one round trip may take; above any single exec budget, so a
/// slow machine surfaces as the exec's own timeout, not the socket's.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(40);

pub trait CoreTransport {
    /// One JSON-RPC round trip. Framing failures come back as
    /// `CoreFail::Socket`/`CoreFail::Protocol`; the JSON-RPC error
    /// envelope is returned as-is for the client to interpret.
    fn call(&self, method: &str, params: &Value) -> Result<Value, CoreFail>;
}

impl<T: CoreTransport + ?Sized> CoreTransport for &T {
    fn call(&self, method: &str, params: &Value) -> Result<Value, CoreFail> {
        (**self).call(method, params)
    }
}

pub struct SocketTransport {
    path: PathBuf,
    read_timeout: Duration,
}

impl SocketTransport {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            read_timeout: DEFAULT_READ_TIMEOUT,
        }
    }

    /// The socket path this transport dials: the subscription face
    /// names it in its failures and the tests hold it.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A bounded read timeout, for tests and for honesty about the
    /// default.
    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = timeout;
        self
    }

    /// $XDG_RUNTIME_DIR/herdr-core/<session>.sock, the session being
    /// HC_SESSION or "default". A session carrying a path separator is
    /// refused before it can name another directory.
    pub fn default_path() -> Result<PathBuf, CoreFail> {
        let runtime = std::env::var("XDG_RUNTIME_DIR").map_err(|_| CoreFail::Socket {
            path: "(unset)".into(),
            detail: "XDG_RUNTIME_DIR is not set".into(),
        })?;
        let session = std::env::var("HC_SESSION")
            .ok()
            .filter(|session| !session.is_empty())
            .unwrap_or_else(|| "default".to_string());
        if session.contains('/') || session.contains('\0') {
            return Err(CoreFail::Socket {
                path: runtime.clone(),
                detail: format!("HC_SESSION {session:?} carries a path separator"),
            });
        }
        Ok(Path::new(&runtime)
            .join("herdr-core")
            .join(format!("{session}.sock")))
    }
}

impl CoreTransport for SocketTransport {
    fn call(&self, method: &str, params: &Value) -> Result<Value, CoreFail> {
        let socket_fail = |detail: String| CoreFail::Socket {
            path: self.path.display().to_string(),
            detail,
        };
        let stream = UnixStream::connect(&self.path).map_err(|e| socket_fail(e.to_string()))?;
        stream
            .set_read_timeout(Some(self.read_timeout))
            .map_err(|e| socket_fail(e.to_string()))?;
        stream
            .set_write_timeout(Some(self.read_timeout))
            .map_err(|e| socket_fail(e.to_string()))?;
        let mut writer = stream.try_clone().map_err(|e| socket_fail(e.to_string()))?;
        let request = json!({
            "jsonrpc": "2.0",
            "id": REQUEST_ID,
            "method": method,
            "params": params,
        });
        writeln!(writer, "{request}").map_err(|e| socket_fail(e.to_string()))?;
        let mut reader = BufReader::new(stream);
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = reader
                .read_until(b'\n', &mut line)
                .map_err(|e| socket_fail(e.to_string()))?;
            if read == 0 {
                return Err(CoreFail::Protocol {
                    detail: "the core closed the connection before a response".into(),
                });
            }
            let parsed: Value = serde_json::from_slice(&line).map_err(|e| CoreFail::Protocol {
                detail: format!("unparseable line from the core: {e}"),
            })?;
            if parsed.get("id") == Some(&json!(REQUEST_ID)) {
                return Ok(parsed);
            }
            // a method-only line is a core notification; keep reading
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;

    /// A local test socket speaking the framing back at the client:
    /// accepts connections until closed, reads one request line per
    /// connection, and either runs the canned reply script or holds the
    /// connection silently. No core, no network anywhere.
    struct TestServer {
        path: PathBuf,
    }

    impl TestServer {
        fn start(script: &[&str]) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "bvc-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("test.sock");
            let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
            let script: Vec<String> = script.iter().map(|r| r.to_string()).collect();
            std::thread::spawn(move || {
                while let Ok((mut stream, _)) = listener.accept() {
                    let mut buf = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        match stream.read(&mut byte) {
                            Ok(0) | Err(_) => break,
                            Ok(_) if byte[0] == b'\n' => break,
                            Ok(_) => buf.push(byte[0]),
                        }
                    }
                    if std::str::from_utf8(&buf)
                        .map(|text| text.contains("\"method\""))
                        .unwrap_or(false)
                    {
                        for response in &script {
                            if stream.write_all(response.as_bytes()).is_err() {
                                break;
                            }
                        }
                    }
                    // the stream drops here: an empty script answers the
                    // request with an immediate close, which is exactly
                    // the framing failure under test
                }
            });
            Self { path }
        }

        fn start_silent() -> Self {
            let dir = std::env::temp_dir().join(format!(
                "bvc-silent-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("test.sock");
            let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
            std::thread::spawn(move || {
                if let Ok((stream, _)) = listener.accept() {
                    // hold the connection open, never answering
                    std::mem::forget(stream);
                    std::thread::sleep(Duration::from_secs(3600));
                }
            });
            Self { path }
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
            if let Some(dir) = self.path.parent() {
                let _ = std::fs::remove_dir(dir);
            }
        }
    }

    const NOTIFICATION: &str =
        "{\"jsonrpc\":\"2.0\",\"method\":\"events.changed\",\"params\":{}}\n";
    const RESPONSE: &str = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n";

    #[test]
    fn the_request_is_one_json_rpc_line_and_the_result_comes_back() {
        let server = TestServer::start(&[RESPONSE]);
        let transport = SocketTransport::new(server.path.clone());
        let response = transport
            .call("file.read", &json!({"path": "/b.yaml"}))
            .unwrap();
        assert_eq!(response.get("id"), Some(&json!(1)));
        assert_eq!(response.get("result"), Some(&json!({"ok": true})));
    }

    #[test]
    fn notifications_are_skipped_until_the_matching_id() {
        let server = TestServer::start(&[NOTIFICATION, NOTIFICATION, RESPONSE]);
        let transport = SocketTransport::new(server.path.clone());
        let response = transport.call("file.read", &json!({})).unwrap();
        assert_eq!(response.get("result"), Some(&json!({"ok": true})));
    }

    #[test]
    fn an_error_envelope_is_returned_for_the_client_to_interpret() {
        let error = "{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32000,\"data\":{\"kind\":\"not_found\"},\"message\":\"no such file\"}}\n";
        let server = TestServer::start(&[error]);
        let transport = SocketTransport::new(server.path.clone());
        let response = transport.call("file.read", &json!({})).unwrap();
        assert!(response.get("error").is_some());
    }

    #[test]
    fn a_silent_socket_times_out_as_unavailable() {
        let server = TestServer::start_silent();
        let transport =
            SocketTransport::new(server.path.clone()).with_read_timeout(Duration::from_millis(150));
        let fail = transport.call("file.read", &json!({})).unwrap_err();
        assert_eq!(fail.code(), "core-unavailable", "{fail}");
    }

    #[test]
    fn a_closed_connection_is_a_protocol_refusal() {
        let server = TestServer::start(&[]);
        let transport = SocketTransport::new(server.path.clone());
        let fail = transport.call("file.read", &json!({})).unwrap_err();
        assert_eq!(fail.code(), "core-protocol", "{fail}");
    }

    #[test]
    fn a_missing_socket_is_unavailable_with_the_path_named() {
        let transport = SocketTransport::new(PathBuf::from("/definitely/not/here.sock"));
        let fail = transport.call("file.read", &json!({})).unwrap_err();
        assert_eq!(fail.code(), "core-unavailable");
        assert!(fail.to_string().contains("/definitely/not/here.sock"));
    }

    #[test]
    fn the_default_path_is_the_session_socket_under_the_runtime_dir() {
        let previous_runtime = std::env::var("XDG_RUNTIME_DIR").ok();
        let previous_session = std::env::var("HC_SESSION").ok();
        std::env::set_var("XDG_RUNTIME_DIR", "/run/user/1000");
        std::env::remove_var("HC_SESSION");
        assert_eq!(
            SocketTransport::default_path().unwrap(),
            PathBuf::from("/run/user/1000/herdr-core/default.sock")
        );
        std::env::set_var("HC_SESSION", "lane");
        assert_eq!(
            SocketTransport::default_path().unwrap(),
            PathBuf::from("/run/user/1000/herdr-core/lane.sock")
        );
        std::env::set_var("HC_SESSION", "../escape");
        assert_eq!(
            SocketTransport::default_path().unwrap_err().code(),
            "core-unavailable"
        );
        match previous_runtime {
            Some(value) => std::env::set_var("XDG_RUNTIME_DIR", value),
            None => std::env::remove_var("XDG_RUNTIME_DIR"),
        }
        match previous_session {
            Some(value) => std::env::set_var("HC_SESSION", value),
            None => std::env::remove_var("HC_SESSION"),
        }
    }
}
