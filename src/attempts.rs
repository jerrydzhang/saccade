//! The write-attempt record: one line per request the server judges,
//! appended beside the tracker. Refusals are fat — the request as
//! received plus the storage-seq cursor they died against, because a
//! refusal enters no journal and the line is the only place its shape
//! survives. Acceptances are thin — the seqs the request became; the
//! journal already holds the payload, and the record-duplicate law
//! cuts anything it can restate. The reader is the incident-time
//! diagnoser; the moment is the judgment.

use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::json;

/// What the judging door knows about the request as received: the
/// client binary that built it, when the door knows one, and the raw
/// body bytes.
pub struct AsReceived {
    pub client: Option<String>,
    pub raw: String,
}

/// One judged write's outcome half; the common fields ride the line
/// either way.
pub enum Outcome<'a> {
    /// The seqs the request became in the journal.
    Landed { seqs: &'a [usize] },
    /// The wire's refusal code, the storage seq cursor the request
    /// died against (clone at it to reproduce the world judged), and
    /// the request as received.
    Refused {
        code: &'a str,
        cursor: Option<usize>,
        request: &'a str,
    },
}

/// The append-only sink. A file that cannot open is a silent no-op: a
/// dead sink never taxes the write path.
#[derive(Clone)]
pub struct Attempts {
    file: Arc<Mutex<Option<std::fs::File>>>,
}

impl Attempts {
    pub fn beside(db_path: &Path) -> Self {
        let path = crate::paths::attempts_at(db_path);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok();
        Attempts {
            file: Arc::new(Mutex::new(file)),
        }
    }

    pub fn append(
        &self,
        actor: Option<&str>,
        client: Option<&str>,
        duration_ms: u64,
        outcome: Outcome,
    ) {
        let mut line = json!({
            "ts": crate::db::now_epoch(),
            "actor": actor,
            "client": client,
            "server": env!("CARGO_PKG_VERSION"),
            "duration_ms": duration_ms,
        });
        match outcome {
            Outcome::Landed { seqs } => {
                line["outcome"] = json!("landed");
                line["seqs"] = json!(seqs);
            }
            Outcome::Refused {
                code,
                cursor,
                request,
            } => {
                line["outcome"] = json!("refused");
                line["code"] = json!(code);
                line["cursor"] = json!(cursor);
                line["request"] = json!(request);
            }
        }
        let Ok(text) = serde_json::to_string(&line) else {
            return;
        };
        if let Some(file) = self
            .file
            .lock()
            .expect("the attempts sink is not poisoned")
            .as_mut()
            && writeln!(file, "{text}").is_err()
        {
            // a sink that cannot write loses its line, never the write
        }
    }
}

/// Serve's warns land beside the tracker with the attempts, so one
/// surface greps everything; info and below stay stderr's. A file that
/// cannot open leaves the subscriber stderr-only.
pub struct WarnsTee {
    file: Option<Arc<Mutex<std::fs::File>>>,
}

impl WarnsTee {
    pub fn beside(db_path: &Path) -> Self {
        let path = crate::paths::attempts_at(db_path);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(|f| Arc::new(Mutex::new(f)));
        WarnsTee { file }
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for WarnsTee {
    type Writer = Box<dyn std::io::Write>;

    fn make_writer(&'a self) -> Self::Writer {
        Box::new(std::io::stderr())
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        // Level orders by verbosity, ERROR least: WARN and worse tee
        if *meta.level() <= tracing::Level::WARN
            && let Some(file) = &self.file
        {
            return Box::new(TeeWrite { file: file.clone() });
        }
        Box::new(std::io::stderr())
    }
}

struct TeeWrite {
    file: Arc<Mutex<std::fs::File>>,
}

impl std::io::Write for TeeWrite {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stderr().write_all(buf);
        let mut file = self.file.lock().expect("the attempts sink is not poisoned");
        let _ = file.write_all(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut file = self.file.lock().expect("the attempts sink is not poisoned");
        let _ = file.flush();
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::Value;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sac-attempts-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("saccade.db")
    }

    fn lines_of(db: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(crate::paths::attempts_at(db))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).expect("one json object per line"))
            .collect()
    }

    /// The record-duplicate law: a landed line is the correlation and
    /// nothing else — no payload, no request echo, no fields the
    /// journal can restate.
    #[test]
    fn a_landed_line_carries_no_payload() {
        let db = scratch("landed");
        let attempts = Attempts::beside(&db);
        attempts.append(
            Some("human person"),
            Some("0.3.0"),
            3,
            Outcome::Landed { seqs: &[41, 42] },
        );
        let lines = lines_of(&db);
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        assert_eq!(line["outcome"], "landed");
        assert_eq!(line["seqs"], json!([41, 42]));
        assert_eq!(line["actor"], "human person");
        assert_eq!(line["client"], "0.3.0");
        assert_eq!(line["server"], env!("CARGO_PKG_VERSION"));
        let mut keys = line
            .as_object()
            .expect("a line is an object")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "actor",
                "client",
                "duration_ms",
                "outcome",
                "seqs",
                "server",
                "ts"
            ]
        );
        std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
    }

    /// A refusal enters no journal: the line carries the full request
    /// as received and the cursor to clone against, so its shape
    /// survives exactly once.
    #[test]
    fn a_refused_line_restates_the_request_and_its_cursor() {
        let db = scratch("refused");
        let attempts = Attempts::beside(&db);
        let raw = r#"{"context":{"actor":"pi","tier":"agent"},"command":{"claim_task":{"id":9}}}"#;
        attempts.append(
            Some("pi"),
            None,
            1,
            Outcome::Refused {
                code: "invalid_task_id",
                cursor: Some(824),
                request: raw,
            },
        );
        let line = &lines_of(&db)[0];
        assert_eq!(line["outcome"], "refused");
        assert_eq!(line["code"], "invalid_task_id");
        assert_eq!(line["cursor"], 824);
        assert_eq!(line["request"], raw);
        // the request rides as a string: replay re-posts these bytes
        assert!(line["request"].is_string());
        std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
    }

    /// A sink that cannot open is silent: the write path never pays
    /// for a dead attempts file.
    #[test]
    fn a_dead_sink_costs_nothing() {
        let db = scratch("dead");
        // a directory where the file would open: the open fails
        std::fs::create_dir_all(crate::paths::attempts_at(&db)).unwrap();
        let attempts = Attempts::beside(&db);
        attempts.append(Some("pi"), None, 1, Outcome::Landed { seqs: &[0] });
        attempts.append(
            None,
            None,
            0,
            Outcome::Refused {
                code: "malformed_request",
                cursor: None,
                request: "not json",
            },
        );
        std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
    }

    /// Serve's warns land beside the tracker with the attempts; info
    /// stays stderr's.
    #[test]
    fn the_tee_lands_warns_and_leaves_info() {
        let db = scratch("tee");
        let tee = WarnsTee::beside(&db);
        use tracing_subscriber::util::SubscriberInitExt;
        let _guard = tracing_subscriber::fmt()
            .json()
            .with_env_filter("info")
            .with_writer(tee)
            .set_default();
        tracing::warn!(task = 3, "the warn lands");
        tracing::info!("the info stays stderr's");
        let text = std::fs::read_to_string(crate::paths::attempts_at(&db)).unwrap();
        assert!(text.contains("the warn lands"), "{text}");
        assert!(!text.contains("the info stays"), "{text}");
        std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
    }
}
