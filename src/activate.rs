use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use crate::IncarnationId;
use crate::api::AppState;

/// How long activation waits for another holder of the per-session
/// lock before standing down: bounded, never indefinite.
const LOCK_WAIT_MILLIS: u64 = 10_000;

/// How long the resume door waits for the launched session's guard to
/// be taken: bounded handoff, after which the door reports failure and
/// the mail stays relaunchable. The bound never proves quiescence - a
/// late boot is contained by the attach-side duplicate refusal, and a
/// late attacher still associates, drains, and submits. Cold first
/// boots of the composed harness have been observed past two minutes.
pub const GUARD_HANDOFF_SECS: u64 = 300;

/// The canonical per-native-session guard file: the session's
/// integration holds an exclusive kernel flock on it from association
/// until death, and the key is the stable hash of the opaque id -
/// never the id itself as a path.
pub fn guard_path(repo_root: &Path, native_id: &str) -> PathBuf {
    crate::paths::state_dir(repo_root)
        .join("saccade-activation")
        .join(format!("{}.guard", crate::paths::session_key(native_id)))
}

/// Whether the native session's own guard is held by a live process:
/// the session's integration holds an exclusive kernel flock on this
/// file from association until death, so a failed non-blocking take is
/// session-specific live evidence - a held lock means a live holder,
/// and any death releases it. None when the guard cannot be probed.
pub fn guard_held(guard: &Path) -> Option<bool> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(guard)
        .ok()?;
    let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    Some(taken != 0)
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
    /// the session's own guard is held: a live process owns this exact
    /// native session
    LiveGuard,
    /// no live association, no held guard, valid header: the exact
    /// session's launch was observed through to the guard handoff
    ResumeSubmitted,
    /// another activation holds this session's lock: mail stays
    /// pending while it observes its outcome
    InFlight,
}

/// The per-native-session serialization: one flock file per native
/// session (keyed by the session's stable hash), so simultaneous
/// senders cannot both decide absent before either launch registers.
/// Runtime cache under the tracker's state directory - never a record,
/// never a lifecycle object.
fn activation_paths_in(repo_root: &Path, native_id: &str) -> (PathBuf, PathBuf) {
    let base = crate::paths::state_dir(repo_root).join("saccade-activation");
    let _ = std::fs::create_dir_all(&base);
    let key = crate::paths::session_key(native_id);
    (
        base.join(format!("{key}.lock")),
        base.join(format!("{key}.launched")),
    )
}

/// The activation decision for one recipient, under the session's
/// lock: the runtime association (live bridge connections) first, then
/// the session's own guard, then the saved header, then the launch.
/// The launch closure runs INSIDE the lock and is observed to
/// completion, so the decision and the launch are one atomic act - no
/// second sender can decide absent in between, and the lock is not
/// released until the outcome is known. Unknown stays unknown.
pub fn activate(
    repo_root: &Path,
    recipient: &Recipient,
    bridge_connected: bool,
    launch: &mut impl FnMut() -> Result<(), String>,
) -> Decision {
    let (lock_path, _launched) = activation_paths_in(repo_root, &recipient.native_id);
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
        // the session's own guard: a held guard is live evidence at
        // native-session granularity, which no directory scan offers
        let guard = guard_path(repo_root, &recipient.native_id);
        match guard_held(&guard) {
            Some(true) => Decision::LiveGuard,
            None => Decision::Hold(
                "the session's launch guard cannot be probed; liveness is unknown".into(),
            ),
            Some(false) => match crate::pi_session::read_header(&recipient.saved_session) {
                Err(e) => Decision::Hold(format!("their saved session cannot be read: {e}")),
                Ok(header) if header.id != recipient.native_id => Decision::Hold(format!(
                    "their saved session carries id {}, not {}",
                    header.id, recipient.native_id
                )),
                Ok(_) => match launch() {
                    Ok(()) => Decision::ResumeSubmitted,
                    Err(e) => Decision::Hold(format!("resume launch refused: {e}")),
                },
            },
        }
    };
    drop(file);
    decision
}

/// The wire's activation after a landed SendMessage: resolve the
/// recipient against the server's own fold, decide under the session's
/// lock, and - when safe - run the resume launch through this binary's
/// own door to its observed outcome. The door exits success only when
/// the launched session's guard is held (the rigorous handoff), so the
/// child's exit is what claims a submitted launch. Bounded and reaped:
/// pipes are drained while waiting, the child never outlives the bound,
/// and a failure leaves the mail pending with nothing recorded. Fire
/// and forget from the reply's perspective - the send never waits on
/// host mechanics.
pub fn after_send(app: &AppState, to_id: IncarnationId, to_num: usize) {
    eprintln!("ACTIVATE: entered for i-{to_num}");
    // activation routes through the host: a server serving no repo has
    // no state directory to key guards by and no repo to hand the door
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        eprintln!("ACTIVATE: no repo root");
        return;
    };
    eprintln!("ACTIVATE: repo {}", repo_root.display());
    let address = crate::paths::address_at(app.db_path());
    let url = std::fs::read_to_string(&address).unwrap_or_default();
    let url = url.trim().to_string();
    eprintln!("ACTIVATE: url {url}");
    if url.is_empty() {
        return;
    }
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
    // the launcher is the real door; a test supplies an explicit
    // launcher seam rather than a nested test binary posing as one
    let launcher = match std::env::var("SACCADE_ACTIVATION_LAUNCHER") {
        Ok(path) if !path.trim().is_empty() => PathBuf::from(path),
        _ => match std::env::current_exe() {
            Ok(exe) => exe,
            Err(_) => return,
        },
    };
    let mut launch = || -> Result<(), String> {
        let mut child = std::process::Command::new(&launcher)
            .args([
                "--server",
                &url,
                "--repo",
                repo_root.display().to_string().as_str(),
                "integrate",
                "--resume",
                &format!("i-{to_num}"),
            ])
            .current_dir(&repo_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("the resume launch could not start: {e}"))?;
        // drain the pipes while waiting: a full pipe would stall the
        // child, and the output is the door's own face - observed, reaped
        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();
        let out_drain = std::thread::spawn(move || {
            let mut sink = String::new();
            if let Some(pipe) = stdout_pipe.as_mut() {
                let _ = std::io::Read::read_to_string(pipe, &mut sink);
            }
            sink
        });
        let err_drain = std::thread::spawn(move || {
            let mut sink = String::new();
            if let Some(pipe) = stderr_pipe.as_mut() {
                let _ = std::io::Read::read_to_string(pipe, &mut sink);
            }
            sink
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(420);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = out_drain.join();
                    let _ = err_drain.join();
                    return Err("the resume launch door exceeded its bound".into());
                }
                Err(e) => return Err(format!("the resume launch could not be observed: {e}")),
            }
        };
        let _ = out_drain.join();
        let _ = err_drain.join();
        if status.success() {
            Ok(())
        } else {
            Err("the resume launch door exited unsuccessfully".into())
        }
    };
    let decision = activate(repo_root.as_path(), &recipient, connected, &mut launch);
    eprintln!("ACTIVATE: decision {decision:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;

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
    fn a_free_guard_reads_not_held_and_a_taken_one_reads_held() {
        let base = std::env::temp_dir().join(format!("sac-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let guard = guard_path(&base, "aaaa1111");
        std::fs::create_dir_all(guard.parent().unwrap()).unwrap();
        assert_eq!(
            guard_held(&guard),
            Some(false),
            "an untouched guard is free"
        );
        // a live holder: this test takes the lock and keeps the fd open
        let holder = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&guard)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert_eq!(guard_held(&guard), Some(true), "a held lock reads held");
        drop(holder);
        assert_eq!(guard_held(&guard), Some(false), "death released it");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_held_guard_is_live_evidence_and_launches_nothing() {
        let base = std::env::temp_dir().join(format!("sac-held-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let workspace = base.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let saved = header_file(&base, "aaaa1111", &workspace);
        let guard = guard_path(&base, "aaaa1111");
        std::fs::create_dir_all(guard.parent().unwrap()).unwrap();
        let holder = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&guard)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let recipient = recipient(&saved);
        let mut launched = 0;
        let mut launch = || -> Result<(), String> {
            launched += 1;
            Ok(())
        };
        let decision = activate(&base, &recipient, false, &mut launch);
        assert!(matches!(decision, Decision::LiveGuard), "{decision:?}");
        assert_eq!(launched, 0, "a live holder is never resumed into");
        drop(holder);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_free_guard_with_a_valid_header_observes_the_launch_through() {
        let base = std::env::temp_dir().join(format!("sac-free-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let workspace = base.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let saved = header_file(&base, "aaaa1111", &workspace);
        let recipient = recipient(&saved);
        let mut launch = || -> Result<(), String> { Ok(()) };
        let decision = activate(&base, &recipient, false, &mut launch);
        assert!(
            matches!(decision, Decision::ResumeSubmitted),
            "{decision:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_connected_bridge_launches_nothing_even_with_a_free_guard() {
        let base = std::env::temp_dir().join(format!("sac-conn-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let recipient = recipient(&base.join("whatever.jsonl"));
        let mut launched = 0;
        let mut launch = || -> Result<(), String> {
            launched += 1;
            Ok(())
        };
        let decision = activate(&base, &recipient, true, &mut launch);
        assert!(matches!(decision, Decision::Connected));
        assert_eq!(launched, 0);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_header_mismatch_holds_and_launches_nothing() {
        let base = std::env::temp_dir().join(format!("sac-mis-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let workspace = base.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let saved = header_file(&base, "bbbb2222", &workspace);
        let recipient = recipient(&saved);
        let mut launched = 0;
        let mut launch = || -> Result<(), String> {
            launched += 1;
            Ok(())
        };
        let decision = activate(&base, &recipient, false, &mut launch);
        assert!(matches!(decision, Decision::Hold(_)), "{decision:?}");
        assert_eq!(launched, 0);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn the_session_key_hides_arbitrary_ids_from_paths() {
        assert_ne!(
            crate::paths::session_key("../../etc/passwd"),
            "../../etc/passwd"
        );
        assert_eq!(
            crate::paths::session_key("same-id"),
            crate::paths::session_key("same-id"),
        );
        assert_ne!(
            crate::paths::session_key("one"),
            crate::paths::session_key("two"),
        );
    }
}
