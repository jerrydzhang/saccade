//! The app's ONE Herdr event feed, by this client's own design
//! (t-24@r23: subscriptions are per connection and unlimited on
//! herdr-core 0.1.17 - the earlier one-subscription rule is
//! withdrawn; sharing one across windows remains allowed where
//! simpler, and simpler is what this client holds): every window and
//! instrument reads the table this feed keeps - one reconnect, one
//! backoff, one table to trust. The feed owns the connection's life:
//! one persistent
//! subscribe on the local core's socket, the table updated as events
//! arrive, every consumer waking through one bounded hold (the
//! wait_look_config pattern: no timer poll, quiet is a normal answer).
//! A dead stream reconnects with backoff and clears the table first -
//! a stale word would lie - and words refill from the stream. The
//! feed reads only: a subscription sends nothing to any terminal.
//! The socket is resolved once by the shell's setup and handed in; a
//! device with no core hands in none, and the feed dials nothing.

use saccade_app_core::{CoreEvent, EventsConnection, SocketTransport};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

/// The bounded hold a wait gets without asking, and the ceiling any
/// ask may set - the pages' contract, kept here too.
const DEFAULT_HOLD: Duration = Duration::from_secs(25);
const MAX_HOLD: Duration = Duration::from_secs(55);

/// One live terminal's words, exactly as the program sent them: a
/// field the program did not send stays absent (t-24@r19: nothing is
/// shown when nothing was sent). `finished` carries the core's reason
/// once the terminal ended; the word is gone then.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WordRow {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub msg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished: Option<String>,
    /// The moment this row last changed, epoch seconds: the time the
    /// agents glance renders.
    pub at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WordsSnapshot {
    pub generation: u64,
    pub words: BTreeMap<String, WordRow>,
    /// Whether the feed holds a live connection to Herdr core: the
    /// strip says "Herdr unreachable" when it does not (t-34@r6 change 6).
    pub connected: bool,
    /// Why not, when not: the socket's path and the failure, or the
    /// missing runtime directory - what the human needs to fix it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
}

#[derive(Debug, Default)]
struct Table {
    generation: u64,
    words: BTreeMap<String, WordRow>,
    connected: bool,
    problem: Option<String>,
}

#[derive(Debug)]
struct Shared {
    table: Mutex<Table>,
    changed: Condvar,
}

impl Shared {
    /// The connection's state and, when down, why; a change is news for
    /// the waiters and a line in the app's log.
    fn set_connected(&self, connected: bool, problem: Option<String>) {
        let mut table = self.table.lock().expect("the words table is not poisoned");
        if table.connected == connected && table.problem == problem {
            return;
        }
        match (&problem, connected) {
            (_, true) => eprintln!("[bridge] herdr words feed connected"),
            (Some(why), false) => eprintln!("[bridge] herdr words feed down: {why}"),
            (None, false) => eprintln!("[bridge] herdr words feed down"),
        }
        table.connected = connected;
        table.problem = if connected { None } else { problem };
        table.generation += 1;
        drop(table);
        self.changed.notify_all();
    }

    fn publish(&self, update: impl FnOnce(&mut BTreeMap<String, WordRow>)) {
        let mut table = self.table.lock().expect("the words table is not poisoned");
        table.generation += 1;
        update(&mut table.words);
        drop(table);
        self.changed.notify_all();
    }
}

/// The app's feed handle: the shared table and the thread that keeps
/// it. The feed runs for the app's life, like the look's follower.
pub struct WordsFeed {
    shared: Arc<Shared>,
}

impl WordsFeed {
    /// Start the feed on ONE socket: the resolved core socket, or
    /// none - a device with no core parks the feed, and it dials
    /// nothing. Reconnects use the same socket with backoff.
    pub fn spawn(socket: Option<PathBuf>, backoff: Duration) -> Self {
        let shared = Arc::new(Shared {
            table: Mutex::new(Table::default()),
            changed: Condvar::new(),
        });
        let thread_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("herdr-words-feed".into())
            .spawn(move || match socket {
                Some(path) => feed_loop(thread_shared, path, backoff),
                None => park_loop(thread_shared),
            })
            .expect("the words feed thread starts");
        Self { shared }
    }

    /// The bounded hold: answer at once when the table moved past
    /// `since`, else park on the feed's wake until it does or the hold
    /// expires. Quiet is a normal answer.
    pub fn wait(&self, since: u64, hold: Duration) -> WordsSnapshot {
        let table = self
            .shared
            .table
            .lock()
            .expect("the words table is not poisoned");
        if table.generation > since {
            return WordsSnapshot {
                generation: table.generation,
                words: table.words.clone(),
                connected: table.connected,
                problem: table.problem.clone(),
            };
        }
        let deadline = std::time::Instant::now() + hold;
        let mut table = table;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            let (locked, _timed_out) = self
                .shared
                .changed
                .wait_timeout(table, remaining)
                .expect("the words table is not poisoned");
            table = locked;
            if table.generation > since {
                break;
            }
        }
        WordsSnapshot {
            generation: table.generation,
            words: table.words.clone(),
            connected: table.connected,
            problem: table.problem.clone(),
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0)
}

/// No core handed in: the feed keeps the empty table and waits. It
/// never dials, never re-resolves - the process starts once, against
/// the core its setup found.
fn park_loop(shared: Arc<Shared>) {
    loop {
        std::thread::sleep(Duration::from_secs(3600));
        let _ = &shared;
    }
}

fn feed_loop(shared: Arc<Shared>, socket: PathBuf, backoff: Duration) {
    loop {
        // words are live state: every (re)connect starts from the
        // empty table, and what the core still knows refills the
        // moment its stream says so
        shared.publish(|words| words.clear());
        let transport = SocketTransport::new(socket.clone());
        let mut connection = match EventsConnection::open(&transport) {
            Ok(connection) => connection,
            Err(fail) => {
                shared.set_connected(false, Some(fail.to_string()));
                std::thread::sleep(backoff);
                continue;
            }
        };
        shared.set_connected(true, None);
        let mut first = true;
        let why = loop {
            match connection.next_event(first) {
                // a clean EOF, before any frame or mid-stream: the
                // connection is done; reconnect on the backoff
                Ok(CoreEvent::Other) => {
                    break format!("the core at {} closed the event stream", socket.display())
                }
                Ok(event) => {
                    first = false;
                    apply(&shared, event);
                }
                Err(fail) => break fail.to_string(),
            }
        };
        shared.set_connected(false, Some(why));
        std::thread::sleep(backoff);
    }
}

fn apply(shared: &Shared, event: CoreEvent) {
    match event {
        // the restart vocabulary and a fresh snapshot both mean the
        // table restarts: kept terminals come back under their
        // addresses as their events arrive
        CoreEvent::CoreRestarted | CoreEvent::Snapshot { .. } => {
            shared.publish(|words| words.clear());
        }
        CoreEvent::Finished { terminal, reason } => {
            let at = now_secs();
            shared.publish(move |words| {
                let row = words.entry(terminal).or_insert(WordRow {
                    state: None,
                    app: None,
                    msg: None,
                    progress: None,
                    title: None,
                    finished: None,
                    at,
                });
                row.state = None;
                row.app = None;
                row.msg = None;
                row.progress = None;
                row.title = None;
                row.finished = Some(reason);
                row.at = at;
            });
        }
        CoreEvent::Status {
            terminal,
            state,
            app,
            msg,
            progress,
            title,
        } => {
            let at = now_secs();
            shared.publish(move |words| {
                let row = words.entry(terminal).or_insert(WordRow {
                    state: None,
                    app: None,
                    msg: None,
                    progress: None,
                    title: None,
                    finished: None,
                    at,
                });
                row.state = state;
                row.app = app;
                row.msg = msg;
                row.progress = progress;
                row.title = title;
                row.finished = None;
                row.at = at;
            });
        }
        CoreEvent::Other => {}
    }
}

// -- the bridge command --------------------------------------------------

static FEED: OnceLock<WordsFeed> = OnceLock::new();

/// Start the app's feed, once, from the shell's setup: the socket the
/// setup resolved, or none.
/// Start the app's feed on the resolved socket, or - when none could be
/// resolved - park it with the reason, so the strip can say why.
pub fn start(socket: Result<PathBuf, String>) {
    let feed = match socket {
        Ok(path) => WordsFeed::spawn(Some(path), Duration::from_secs(2)),
        Err(why) => {
            let feed = WordsFeed::spawn(None, Duration::from_secs(2));
            feed.shared.set_connected(false, Some(why));
            feed
        }
    };
    let _ = FEED.set(feed);
}

/// The app's feed, started on demand when a command arrives first:
/// with no socket handed in it dials nothing.
fn feed() -> &'static WordsFeed {
    FEED.get_or_init(|| WordsFeed::spawn(None, Duration::from_secs(2)))
}

#[tauri::command]
pub async fn words_wait(since: u64, timeout: Option<u64>) -> WordsSnapshot {
    // the bounded hold rides a blocking thread; the async command
    // never blocks the main thread - a sync command with a hold would
    // freeze the window for the hold's whole length, exactly what the
    // look's wait avoids
    let hold = timeout
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_HOLD)
        .min(MAX_HOLD);
    let handle = tauri::async_runtime::spawn_blocking(move || feed().wait(since, hold));
    handle.await.expect("words wait task")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;

    /// One fake core: answers the subscribe line, prints the lines,
    /// then holds until the peer hangs up - the feed's EOF is the
    /// reconnect signal. The temp directory is retained: no deletion
    /// sweeps.
    fn fake_core(lines: &[String]) -> (PathBuf, std::path::PathBuf) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .subsec_nanos();
        // short name: a Unix socket path must fit SUN_LEN
        let dir = std::env::temp_dir().join(format!("sacw-{}-{}-{}", std::process::id(), n, nanos));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("core.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        let served = lines.to_vec();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut writer = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream);
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            assert!(request.contains("events.subscribe"), "{request}");
            for line in served {
                writeln!(writer, "{line}").unwrap();
            }
            let _ = reader.read_line(&mut String::new());
        });
        (socket_path, dir)
    }

    fn status_line(seq: u64, terminal: &str, state: &str, msg: &str) -> String {
        format!(
            r#"{{"jsonrpc":"2.0","method":"event","params":{{"seq":{seq},"terminal":"{terminal}","type":"status","data":{{"status":{{"state":"{state}","app":"pi","msg":"{msg}","kind":null,"progress":null,"title":null}}}}}}}}"#
        )
    }

    #[test]
    fn the_feed_holds_one_subscription_and_the_hold_wakes_on_the_landing() {
        let lines = vec![
            r#"{"jsonrpc":"2.0","id":1,"result":{"channel":null,"seq":1,"terminals":[{"address":"t1","state":"live"}]}}"#.into(),
            status_line(2, "t1", "working", "editing src"),
        ];
        let (socket, _dir) = fake_core(&lines);
        let feed = WordsFeed::spawn(Some(socket), Duration::from_millis(10));
        // the feed publishes at least its connect's clear within the
        // hold; the status may land beside it, so the first wait's
        // rows are timing and assert nothing
        let quiet = feed.wait(0, Duration::from_millis(400));
        assert!(
            quiet.generation >= 1,
            "the feed's landing published at least once"
        );
        // walk forward until the word is on the table: each hold
        // returns on a landing or expires, and the fake core holds the
        // connection open so the landings settle
        let deadline = std::time::Instant::now() + Duration::from_secs(4);
        let mut snapshot = feed.wait(quiet.generation, Duration::from_secs(2));
        while snapshot
            .words
            .get("t1")
            .map(|row| row.state.is_none())
            .unwrap_or(true)
        {
            assert!(
                std::time::Instant::now() < deadline,
                "the word never landed"
            );
            snapshot = feed.wait(snapshot.generation, Duration::from_secs(1));
        }
        assert!(
            snapshot.connected,
            "a feed on a live core says it is connected"
        );
        let row = snapshot.words.get("t1").expect("the terminal's row");
        assert_eq!(row.state.as_deref(), Some("working"));
        assert_eq!(row.msg.as_deref(), Some("editing src"));
        assert_eq!(row.app.as_deref(), Some("pi"));
        assert_eq!(row.finished, None);
        // quiet is a normal answer: no landing, no move
        let stale = feed.wait(snapshot.generation, Duration::from_millis(120));
        assert_eq!(stale.generation, snapshot.generation);
    }

    #[test]
    fn a_feed_with_no_core_dials_nothing_and_answers_quiet() {
        let feed = WordsFeed::spawn(None, Duration::from_millis(10));
        let snapshot = feed.wait(0, Duration::from_millis(80));
        assert_eq!(snapshot.generation, 0);
        assert!(snapshot.words.is_empty());
        assert!(
            !snapshot.connected,
            "no core: the strip can say Herdr is unreachable"
        );
    }

    #[test]
    fn a_feed_whose_core_is_gone_says_it_is_not_connected() {
        let socket = std::env::temp_dir().join(format!("sac-no-core-{}.sock", std::process::id()));
        let feed = WordsFeed::spawn(Some(socket), Duration::from_millis(10));
        // the first landing is the feed's own clear; the failed dial
        // follows it
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let mut snapshot = feed.wait(0, Duration::from_millis(200));
        while snapshot.problem.is_none() && std::time::Instant::now() < deadline {
            snapshot = feed.wait(snapshot.generation, Duration::from_millis(200));
        }
        assert!(!snapshot.connected);
        let why = snapshot.problem.expect("the strip is told why");
        assert!(
            why.contains("sac-no-core"),
            "the reason names the socket: {why}"
        );
    }
}
