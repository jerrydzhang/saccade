//! The one host-activation mechanism, shared by every send entrance:
//! the wire handler after a landed SendMessage, and the local-mode send
//! door. Activation lands no records and changes no meaning - it is
//! the machinery deciding whether a stopped peer's exact saved session
//! should be relaunched, and holding the mail pending on every doubt.

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use crate::IncarnationId;
use crate::api::AppState;

/// How long a just-submitted resume launch blocks further activation of
/// the same native session: the launched Pi needs those seconds to
/// start and register, and no second sender may relaunch it meanwhile.
const LAUNCH_GRACE_SECS: u64 = 30;

/// How long activation waits for another holder of the per-session
/// lock before standing down: bounded, never indefinite.
const LOCK_WAIT_MILLIS: u64 = 10_000;

/// Whether a native Pi process lives in a workspace right now, read
/// from the host's process table. Workspace-granular by host
/// limitation, not by choice: a running Pi neither holds its session
/// file open nor names its session in process metadata, so no
/// session-specific quiescence proof exists - the gap is stated where
/// it bites, never papered over. Entries whose cwd is unreadable carry
/// no evidence: kernel threads have no cwd, and another user's process
/// cannot be this user's session.
#[derive(Debug)]
pub enum Presence {
    /// the scan completed and saw no native Pi in this workspace
    Absent,
    /// a native Pi process runs in this workspace
    Present,
    /// the host's process table is unavailable: nothing was learned
    Unreadable,
}

pub fn pi_presence_in(dir: &Path) -> Presence {
    let mut names = vec!["pi".to_string(), "saccade-pi".to_string()];
    if let Ok(pinned) = std::env::var("SACCADE_PI") {
        if let Some(base) = Path::new(&pinned)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
        {
            names.push(base);
        }
    }
    let entries = match std::fs::read_dir("/proc") {
        Ok(entries) => entries,
        Err(_) => return Presence::Unreadable,
    };
    for entry in entries.flatten() {
        let pid = match entry.file_name().to_string_lossy().parse::<u32>() {
            Ok(pid) => pid,
            Err(_) => continue,
        };
        let same_workspace =
            std::fs::read_link(format!("/proc/{pid}/cwd")).is_ok_and(|cwd| cwd == dir);
        if !same_workspace {
            continue;
        }
        let argv0 = std::fs::read(format!("/proc/{pid}/cmdline"))
            .ok()
            .and_then(|raw| {
                raw.split(|b| *b == 0)
                    .next()
                    .map(|argv0| {
                        let os: &std::ffi::OsStr = std::os::unix::ffi::OsStrExt::from_bytes(argv0);
                        Path::new(os)
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                    })
                    .unwrap_or(None)
            });
        if argv0.is_some_and(|base| names.contains(&base)) {
            return Presence::Present;
        }
    }
    Presence::Absent
}

/// Everything the decision needs about one recipient, resolved by the
/// caller against its own view of the world.
#[derive(Debug)]
pub struct Recipient {
    /// the recipient's public number, addressing the resume door
    pub num: usize,
    pub native_id: String,
    pub saved_session: PathBuf,
}

/// What activation decided. Every arm is descriptive; nothing infers
/// the recipient's work or a message's fate beyond "pending". A
/// submitted launch is exactly that - submitted; reattachment is
/// observed later, never assumed here.
#[derive(Debug)]
pub enum Decision {
    /// a delivery bridge holds a connection: in-session delivery owns it
    Connected,
    /// the saved session cannot be validated or the host state cannot
    /// be learned: mail stays pending, the reason travels
    Hold(String),
    /// a native Pi runs in the workspace without a bridge connection:
    /// active or unknown, never resumed into
    LiveUnbridged,
    /// no bridge and no Pi process: the exact session's launch went out
    ResumeSubmitted,
    /// another activation holds this session's lock or a launch just
    /// went out: mail stays pending while it registers
    InFlight,
}

/// The per-native-session serialization: one flock file per native
/// session, so simultaneous senders cannot both decide absent before
/// either launch registers. Runtime cache beside the tracker's state
/// (or a temp fallback when none is known) - never a record, never a
/// lifecycle object.
fn activation_paths_in(base: PathBuf, native_id: &str) -> (PathBuf, PathBuf) {
    let base = if base.is_dir() || std::fs::create_dir_all(&base).is_ok() {
        base
    } else {
        std::env::temp_dir()
    };
    let base = base.join("saccade-activation");
    let _ = std::fs::create_dir_all(&base);
    (
        base.join(format!("{native_id}.lock")),
        base.join(format!("{native_id}.launched")),
    )
}

/// Whether a launch for this session just went out: the marker is
/// written under the lock after a submitted launch and expires on the
/// grace window, so a later sender reads in-flight instead of
/// relaunching a Pi that has not registered yet.
fn launch_in_flight(launched_path: &Path) -> bool {
    match std::fs::read_to_string(launched_path) {
        Ok(text) => text
            .trim()
            .parse::<u64>()
            .map(|at| {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                now.saturating_sub(at) < LAUNCH_GRACE_SECS
            })
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn mark_launch_submitted(launched_path: &Path) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = std::fs::write(launched_path, format!("{now}\n"));
}

/// The activation decision for one recipient, under the session's
/// lock: bridge first, then the saved header, then the process table,
/// then the in-flight marker. The launch closure runs INSIDE the lock,
/// so the decision, the submission, and the grace marker are one
/// atomic act - no second sender can decide absent in between.
/// Unknown stays unknown; the gap is the answer.
pub fn activate(
    recipient: &Recipient,
    bridge_connected: bool,
    launch: &mut impl FnMut() -> Result<(), String>,
) -> Decision {
    activate_in(
        std::env::var("SACCADE_REPO")
            .ok()
            .map(|repo| crate::paths::state_dir(Path::new(&repo)))
            .unwrap_or_else(std::env::temp_dir),
        recipient,
        bridge_connected,
        launch,
    )
}

fn activate_in(
    base: PathBuf,
    recipient: &Recipient,
    bridge_connected: bool,
    launch: &mut impl FnMut() -> Result<(), String>,
) -> Decision {
    let (lock_path, launched_path) = activation_paths_in(base, &recipient.native_id);
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
    {
        Ok(file) => file,
        Err(_) => return Decision::Hold("the activation lock is unopenable".into()),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(LOCK_WAIT_MILLIS);
    loop {
        let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if taken == 0 {
            break;
        }
        if std::time::Instant::now() >= deadline {
            return Decision::InFlight;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    let decision = if bridge_connected {
        Decision::Connected
    } else {
        match crate::pi_session::read_header(&recipient.saved_session) {
            Err(e) => Decision::Hold(format!("their saved session cannot be read: {e}")),
            Ok(header) if header.id != recipient.native_id => Decision::Hold(format!(
                "their saved session carries id {}, not {}",
                header.id, recipient.native_id
            )),
            Ok(header) => {
                let workspace = PathBuf::from(&header.cwd);
                if !workspace.is_dir() {
                    Decision::Hold(format!(
                        "their workspace {} is not on this host; safety cannot be verified",
                        workspace.display()
                    ))
                } else {
                    match pi_presence_in(&workspace) {
                        Presence::Present => Decision::LiveUnbridged,
                        Presence::Unreadable => Decision::Hold(
                            "the host process table is unreadable; liveness cannot be learned"
                                .into(),
                        ),
                        Presence::Absent => {
                            if launch_in_flight(&launched_path) {
                                Decision::InFlight
                            } else {
                                match launch() {
                                    Ok(()) => {
                                        mark_launch_submitted(&launched_path);
                                        Decision::ResumeSubmitted
                                    }
                                    Err(e) => Decision::Hold(format!("resume launch refused: {e}")),
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    drop(file);
    decision
}

/// The wire's activation after a landed SendMessage: resolve the
/// recipient against the server's own fold, decide under the session's
/// lock, and submit the resume launch through this binary's own door
/// when safe. Fire and forget - the send reply never waits on host
/// mechanics, and a failed activation leaves the mail pending exactly
/// as it is.
pub fn after_send(app: &AppState, to_id: IncarnationId, to_num: usize) {
    let snapshot = match app.snapshot() {
        Ok(snapshot) => snapshot,
        Err(_) => return,
    };
    let Some(ctx) = snapshot.world.incarnations.get(&to_id) else {
        return;
    };
    let crate::Incarnation::Native {
        native_id,
        saved_session,
    } = &ctx.incarnation
    else {
        return;
    };
    let recipient = Recipient {
        num: to_num,
        native_id: native_id.as_str().to_string(),
        saved_session: saved_session.as_path().to_path_buf(),
    };
    let connected = app
        .bridge
        .connected()
        .iter()
        .any(|(incarnation, _)| *incarnation == to_id.0.0);
    // the resume launch rides this binary's own door; the address file
    // names the url this server serves, published at bind
    let address = crate::paths::address_at(app.db_path());
    let url = std::fs::read_to_string(&address).unwrap_or_default();
    let url = url.trim().to_string();
    if url.is_empty() {
        return;
    }
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("sac"));
    let mut launch = || -> Result<(), String> {
        std::process::Command::new(&exe)
            .args([
                "--server",
                &url,
                "integrate",
                "--resume",
                &format!("i-{to_num}"),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("the resume launch could not start: {e}"))
    };
    let _ = activate(&recipient, connected, &mut launch);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipient(saved_session: &Path) -> Recipient {
        Recipient {
            num: 4,
            native_id: "aaaa1111".into(),
            saved_session: saved_session.to_path_buf(),
        }
    }

    fn header_file(dir: &Path, id: &str, cwd: &Path) -> PathBuf {
        let path = dir.join("session.jsonl");
        std::fs::write(
            &path,
            format!(
                "{{\"type\":\"session\",\"id\":\"{id}\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
                cwd.display()
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn a_live_pi_in_the_workspace_is_present_by_argv0_and_cwd() {
        let dir = std::env::temp_dir().join(format!("sac-presence-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        use std::os::unix::process::CommandExt;
        let mut child = std::process::Command::new("sh")
            .arg0("pi")
            .arg("-c")
            // the conjunction keeps dash from exec-rebranding the last command
            .arg("sleep 5 && true")
            .current_dir(&dir)
            .spawn()
            .expect("a renamed sh spawns");
        std::thread::sleep(std::time::Duration::from_millis(150));
        let seen = matches!(pi_presence_in(&dir), Presence::Present);
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(seen, "a pi-named process in the workspace reads Present");
    }

    #[test]
    fn an_empty_workspace_reads_absent() {
        let dir = std::env::temp_dir().join(format!("sac-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let seen = matches!(pi_presence_in(&dir), Presence::Absent);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(seen);
    }

    #[test]
    fn a_missing_session_file_holds_the_mail_and_launches_nothing() {
        let base = std::env::temp_dir().join(format!("sac-act-{}", std::process::id()));
        let recipient = recipient(&base.join("nope").join("missing.jsonl"));
        let mut launched = 0;
        let mut launch = || -> Result<(), String> {
            launched += 1;
            Ok(())
        };
        let decision = activate_in(base.clone(), &recipient, false, &mut launch);
        assert!(matches!(decision, Decision::Hold(_)), "{decision:?}");
        assert_eq!(launched, 0);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_header_id_mismatch_holds_the_mail() {
        let base = std::env::temp_dir().join(format!("sac-act-m{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let saved = header_file(&base, "bbbb2222", Path::new("/nowhere"));
        let recipient = recipient(&saved);
        let mut launch = || -> Result<(), String> { Ok(()) };
        let decision = activate_in(base.clone(), &recipient, false, &mut launch);
        assert!(matches!(decision, Decision::Hold(_)), "{decision:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_absent_workspace_launches_once_and_marks_the_grace() {
        let base = std::env::temp_dir().join(format!("sac-act-g{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let workspace = base.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let saved = header_file(&base, "aaaa1111", &workspace);
        let recipient = recipient(&saved);
        let launched = std::cell::Cell::new(0);
        let mut launch = || -> Result<(), String> {
            launched.set(launched.get() + 1);
            Ok(())
        };
        let first = activate_in(base.clone(), &recipient, false, &mut launch);
        assert!(matches!(first, Decision::ResumeSubmitted), "{first:?}");
        assert_eq!(launched.get(), 1);
        // a second sender inside the grace window stands down: the
        // launch is in flight, the mail stays pending, no relaunch
        let second = activate_in(base.clone(), &recipient, false, &mut launch);
        assert!(matches!(second, Decision::InFlight), "{second:?}");
        assert_eq!(launched.get(), 1);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_connected_bridge_never_launches() {
        let base = std::env::temp_dir().join(format!("sac-act-c{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let recipient = recipient(&base.join("whatever.jsonl"));
        let mut launched = 0;
        let mut launch = || -> Result<(), String> {
            launched += 1;
            Ok(())
        };
        let decision = activate_in(base.clone(), &recipient, true, &mut launch);
        assert!(matches!(decision, Decision::Connected));
        assert_eq!(launched, 0);
        let _ = std::fs::remove_dir_all(&base);
    }
}
