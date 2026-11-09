//! `sac-companion`: the agents companion. One daemon per tracker on the
//! hub: it keeps one tmux session per tracker holding one window per
//! live agent session, each an interactive viewer on that agent's
//! Herdr terminal. Enrollment follows the record through
//! `/view/v1/agents`; motion follows Herdr's stream; a reconciler owns
//! every tmux mutation. The tracker supervises nothing and neither
//! does this program - see the package README for the law it carries.

mod herdr;
mod page;
mod reconcile;
mod tmux;

use std::io::BufRead;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;
use reconcile::{Companion, Feed};

/// The env vars a window pane needs to reach the same core the daemon
/// reaches: carried from the daemon's own environment into the pane
/// command. Everything else in the pane's environment is tmux's own.
const PANE_ENV_KEYS: &[&str] = &[
    "PATH",
    "HOME",
    "HC_SESSION",
    "XDG_CONFIG_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
];

/// The agents page's hold bound plus slack: the server parks on its
/// landing wake up to `timeout`, so the client's own bound must exceed
/// it and never cut a held request short.
const HOLD_BOUND: Duration = Duration::from_secs(70);

#[derive(Debug, Parser)]
#[command(
    name = "sac-companion",
    about = "Keep one tmux agents session per tracker: one window per live agent, attached as a viewer to its Herdr terminal",
    version
)]
struct Args {
    /// The tracker's repository root; its state directory holds the
    /// server address hint, the same discovery `sac` uses
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Serve discovery: read the agents page here instead
    #[arg(long)]
    server: Option<String>,

    /// A private tmux server: a bare name means `tmux -L <name>`, an
    /// absolute path means `tmux -S <path>`; the default is the default
    /// tmux server, whose other sessions are never touched
    #[arg(long)]
    socket: Option<String>,

    /// The herdr-core CLI; default is ambient `hc`, or SACCADE_HERDR
    #[arg(long)]
    hc: Option<String>,
}

fn fail(message: &str) -> ! {
    eprintln!("sac-companion: {message}");
    std::process::exit(2);
}

/// The server URL: the flag, or the address hint beside the repo's
/// database (the state directory's composition mirrors the tracker's
/// own: basename plus a short FNV-1a of the resolved root).
fn server_url(args: &Args) -> String {
    if let Some(url) = &args.server {
        return url.clone();
    }
    let Some(repo) = &args.repo else {
        fail("no server: pass --repo (its address hint names the server) or --server <url>")
    };
    let repo = std::fs::canonicalize(repo).unwrap_or_else(|e| fail(&format!("--repo: {e}")));
    let parent = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| {
            let home =
                std::env::var_os("HOME").unwrap_or_else(|| std::ffi::OsString::from("no HOME"));
            PathBuf::from(home).join(".local").join("state")
        });
    let basename = repo
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "repo".into());
    let mut hash: u32 = 0x811c_9dc5;
    for byte in repo.to_string_lossy().as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    let db = parent
        .join("saccade")
        .join(format!("{basename}-{hash:06x}"))
        .join("saccade.db");
    let hint = db.with_file_name(format!(
        "{}.addr",
        db.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("saccade.db")
    ));
    std::fs::read_to_string(&hint)
        .map(|url| url.trim().to_string())
        .ok()
        .filter(|url| !url.is_empty())
        .unwrap_or_else(|| {
            fail(&format!(
                "no server discovered for {}: no address hint at {}; start one beside that tracker (`sac serve`)",
                repo.display(),
                hint.display()
            ))
        })
}

fn pane_env() -> Vec<(String, String)> {
    PANE_ENV_KEYS
        .iter()
        .filter_map(|key| {
            std::env::var(key)
                .ok()
                .filter(|value| !value.is_empty())
                .map(|value| (key.to_string(), value))
        })
        .collect()
}

/// One GET of a `/view/v1` location: the body plus its status, for the
/// refusal decoding. Errors are strings; the watcher retries.
fn fetch(agent: &ureq::Agent, url: &str) -> Result<(u16, String), String> {
    let mut response = agent
        .get(url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .map_err(|e| format!("request failed: {e}"))?;
    let status = response.status().as_u16();
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("body read failed: {e}"))?;
    Ok((status, body))
}

fn main() {
    let args = Args::parse();
    let url = server_url(&args);
    let hc = herdr::resolve_hc(args.hc.as_deref()).unwrap_or_else(|e| fail(&e));
    let server = match &args.socket {
        Some(name) if name.starts_with('/') => tmux::Server::At(PathBuf::from(name)),
        Some(name) => tmux::Server::Named(name.clone()),
        None => tmux::Server::Default,
    };
    let env = pane_env();

    let (tx, rx) = mpsc::channel::<Feed>();

    // the stream reader's pid, so a signal can end the child we started
    // (and only it) before the daemon exits
    static EVENTS_CHILD: AtomicI32 = AtomicI32::new(-1);
    extern "C" fn end_child_and_exit(_: libc::c_int) {
        let pid = EVENTS_CHILD.load(Ordering::SeqCst);
        // SAFETY: libc calls only; the handler replaces the default
        // termination with: end the child we started, then exit
        unsafe {
            if pid > 0 {
                libc::kill(pid, libc::SIGKILL);
            }
            libc::_exit(0);
        }
    }
    unsafe {
        libc::signal(
            libc::SIGINT,
            end_child_and_exit as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            end_child_and_exit as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }

    // enrollment: the record, through the agents page. The server
    // parks each `after` request on its landing wake, so the watcher
    // never timer-polls; the only client-side sleep is failure backoff,
    // and a refusal or failure re-reads the page whole. Before it, the
    // identity phase parks on the root the same way until machine id
    // and repo path name the tracker (t-24@r18: no identity, no
    // session, nothing touched in tmux - keep reading, keep watching,
    // log the reason once).
    let page_tx = tx.clone();
    let page_url = url.clone();
    std::thread::spawn(move || {
        // this thread's IO runs on poll, which no restart flag saves:
        // every signal but the two shutdown ones stays off it, so a
        // stray delivery cannot cut a held page request
        unsafe {
            let mut mask: libc::sigset_t = std::mem::zeroed();
            libc::sigfillset(&mut mask);
            libc::sigdelset(&mut mask, libc::SIGINT);
            libc::sigdelset(&mut mask, libc::SIGTERM);
            libc::pthread_sigmask(libc::SIG_SETMASK, &mask, std::ptr::null_mut());
        }
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(HOLD_BOUND))
            .build()
            .new_agent();
        let mut identity_reason_logged = false;
        let mut root_cursor: Option<u64> = None;
        let mut root_backoff = 1u64;
        let (base, identity) = loop {
            let request = match root_cursor {
                Some(seq) => format!("{page_url}/view/v1/?after={seq}&timeout=55"),
                None => format!("{page_url}/view/v1/"),
            };
            match fetch(&agent, &request) {
                Ok((status, body)) => match page::decode(&body, status) {
                    Ok(Ok(root)) => {
                        root_cursor = root.seq.or(root_cursor);
                        match (&root.machine_id, &root.repo_path) {
                            (Some(machine), Some(repo)) => {
                                let base = root.tracker.clone().unwrap_or_else(|| {
                                    std::path::Path::new(repo)
                                        .file_name()
                                        .map(|n| n.to_string_lossy().into_owned())
                                        .unwrap_or_else(|| "tracker".into())
                                });
                                break (base, format!("{machine}|{repo}"));
                            }
                            _ => {
                                if !identity_reason_logged {
                                    eprintln!(
                                        "sac-companion: the root names no machine id or repo path; no agents session until it does"
                                    );
                                    identity_reason_logged = true;
                                }
                            }
                        }
                    }
                    Ok(Err(refusal)) => eprintln!(
                        "sac-companion: the root page refused ({}, {})",
                        refusal.code, refusal.detail
                    ),
                    Err(bad) => eprintln!("sac-companion: undecodable root: {bad}"),
                },
                Err(e) => {
                    eprintln!("sac-companion: the record side is down ({e}); retrying");
                    std::thread::sleep(Duration::from_secs(root_backoff));
                    root_backoff = (root_backoff * 2).min(15);
                    continue;
                }
            }
            // the landing wake parks the next hold; no timer poll
            std::thread::sleep(Duration::from_secs(root_backoff));
            root_backoff = (root_backoff * 2).min(15);
        };
        if page_tx
            .send(Feed::Tracker {
                base: base.clone(),
                identity: identity.clone(),
            })
            .is_err()
        {
            return;
        }
        let mut cursor: Option<u64> = None;
        let mut backoff = 1u64;
        loop {
            let request = match cursor {
                Some(seq) => format!("{page_url}/view/v1/agents?after={seq}&timeout=55"),
                None => format!("{page_url}/view/v1/agents"),
            };
            match fetch(&agent, &request) {
                Ok((status, body)) => match page::decode(&body, status) {
                    Ok(Ok(p)) => {
                        let rows = page::agents(&p);
                        if page_tx.send(Feed::Snapshot(rows)).is_err() {
                            return;
                        }
                        cursor = p.seq.or(cursor);
                        backoff = 1;
                        continue;
                    }
                    Ok(Err(refusal)) => {
                        eprintln!(
                            "sac-companion: the agents page refused ({}, {}); retrying",
                            refusal.code, refusal.detail
                        );
                        // a refusal is a failed read: the recovery read
                        // is whole
                        cursor = None;
                    }
                    Err(bad) => {
                        eprintln!("sac-companion: undecodable page: {bad}");
                        cursor = None;
                    }
                },
                Err(e) => {
                    eprintln!("sac-companion: the record side is down ({e}); retrying");
                    // the recovery read is whole (t-24@r5): a failure
                    // drops the cursor, so the next request re-reads the
                    // page fresh instead of waiting on a stale wake
                    cursor = None;
                }
            }
            std::thread::sleep(Duration::from_secs(backoff));
            backoff = (backoff * 2).min(15);
        }
    });

    // motion: one Herdr subscription; every EOF is a restart signal
    let motion_hc = hc.clone();
    std::thread::spawn(move || {
        loop {
            let child = Command::new(&motion_hc)
                .args(["events", "--json"])
                .stdout(std::process::Stdio::piped())
                .spawn();
            match child {
                Ok(mut child) => {
                    EVENTS_CHILD.store(child.id() as i32, Ordering::SeqCst);
                    let stdout = child.stdout.take().expect("events stdout was piped");
                    let reader = std::io::BufReader::new(stdout);
                    for line in reader.lines() {
                        let Ok(line) = line else {
                            break;
                        };
                        // the subscription's own snapshot is the fresh
                        // picture after any (re)subscribe: it restarts the
                        // re-attach pass exactly like a restarted event
                        if line.contains("\"terminals\"") {
                            if tx.send(Feed::HerdrRestart).is_err() {
                                return;
                            }
                            continue;
                        }
                        // the subscription's own snapshot is the fresh
                        // picture after any (re)subscribe: structurally
                        // decoded, current words and states included
                        if let Ok(rows) = herdr::parse_ls(&line) {
                            if tx.send(Feed::TerminalSnapshot(rows)).is_err() {
                                return;
                            }
                            continue;
                        }
                        match herdr::parse_event(&line) {
                            Ok(Some(motion)) => {
                                let feed = match motion {
                                    herdr::Motion::Word { terminal, word } => {
                                        Feed::Word { terminal, word }
                                    }
                                    herdr::Motion::Finished { terminal, reason } => {
                                        Feed::Finished { terminal, reason }
                                    }
                                    herdr::Motion::CoreRestarted => Feed::HerdrRestart,
                                };
                                if tx.send(feed).is_err() {
                                    return;
                                }
                            }
                            Ok(None) => {}
                            Err(e) => eprintln!("sac-companion: undecodable event: {e}"),
                        }
                    }
                    // EOF: the subscription ended - core restart or loss.
                    EVENTS_CHILD.store(-1, Ordering::SeqCst);
                    let _ = child.wait();
                    eprintln!(
                        "sac-companion: the Herdr stream ended; treating it as a restart signal"
                    );
                    if tx.send(Feed::HerdrRestart).is_err() {
                        return;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
                Err(e) => {
                    eprintln!(
                        "sac-companion: cannot start {} events ({e}); retrying",
                        motion_hc.display()
                    );
                    std::thread::sleep(Duration::from_secs(5));
                }
            }
        }
    });

    // the reconciler: the only tmux mutations in the process
    let mut companion = Companion::new(
        tmux::Tmux {
            server: server.clone(),
        },
        &hc.to_string_lossy(),
        server,
        env,
    );
    for feed in rx {
        companion.accept(feed);
    }
}
