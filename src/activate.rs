use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// How long activation waits for another holder of the per-session
/// lock before standing down: bounded, never indefinite.
const LOCK_WAIT_MILLIS: u64 = 10_000;

/// How long the resume door waits for the launched session's guard to
/// be taken: bounded handoff, after which the door reports the launch
/// unobserved. The bound never proves quiescence - a late boot is
/// contained by the attach-side duplicate refusal, and a late attacher
/// still associates. Cold first boots of the composed harness have
/// been observed past two minutes.
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
/// and any death releases it. Only the documented would-block conflict
/// reads as held; every other failure leaves the answer unknown.
pub fn guard_held(guard: &Path) -> Option<bool> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(guard)
        .ok()?;
    let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if taken == 0 {
        return Some(false);
    }
    let errno = std::io::Error::last_os_error().raw_os_error()?;
    match errno {
        libc::EWOULDBLOCK => Some(true),
        _ => None,
    }
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

/// Why a session's activation lock is not ours. Unavailable means the
/// lock file could not even be opened - the state root is broken;
/// InFlight means another holder kept it past the bound.
#[derive(Debug)]
pub enum LockFail {
    Unavailable(String),
    InFlight,
}

/// The per-native-session activation lock, held for as long as the
/// returned value lives: a stopped-session controller serializes its
/// guard proof, header retarget, and record against automatic mail
/// activation on the very same file, so neither can interleave. Bounded
/// wait, never indefinite.
pub struct SessionLock {
    /// The open descriptor: dropping it releases the flock
    _file: std::fs::File,
}

impl std::fmt::Debug for SessionLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SessionLock")
    }
}

pub fn lock_session(repo_root: &Path, native_id: &str) -> Result<SessionLock, LockFail> {
    lock_session_within(
        repo_root,
        native_id,
        std::time::Duration::from_millis(LOCK_WAIT_MILLIS),
    )
}

/// The same lock under an explicit wait bound; the test seam keeps the
/// in-flight refusal out of the standard bound's shadow.
pub fn lock_session_within(
    repo_root: &Path,
    native_id: &str,
    wait: std::time::Duration,
) -> Result<SessionLock, LockFail> {
    let (lock_path, _launched) = activation_paths_in(repo_root, native_id);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .map_err(|e| LockFail::Unavailable(e.to_string()))?;
    let deadline = std::time::Instant::now() + wait;
    loop {
        let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if taken == 0 {
            return Ok(SessionLock { _file: file });
        }
        if std::time::Instant::now() >= deadline {
            return Err(LockFail::InFlight);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;

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

    #[test]
    fn the_session_lock_serializes_and_reports_a_refused_open() {
        let base = std::env::temp_dir().join(format!("sac-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();

        // a free lock is taken and held for the value's lifetime
        let held = lock_session(&base, "aaaa1111").expect("the free lock is taken");
        // a second taker finds it in flight, inside a test-sized bound
        let second = lock_session_within(&base, "aaaa1111", std::time::Duration::from_millis(300))
            .unwrap_err();
        assert!(matches!(second, LockFail::InFlight), "{second:?}");
        drop(held);
        let again = lock_session(&base, "aaaa1111");
        assert!(again.is_ok(), "release lets the next holder in");
        drop(again);

        // an unopenable lock file is unavailability, never a timeout:
        // the lock sits beside the guard under the same session key
        let guard = guard_path(&base, "bbbb2222");
        std::fs::create_dir_all(guard.parent().unwrap()).unwrap();
        let lock = guard.with_file_name(format!(
            "{}.lock",
            guard.file_stem().unwrap().to_string_lossy()
        ));
        std::fs::create_dir(&lock).unwrap();
        let refused = lock_session(&base, "bbbb2222").unwrap_err();
        assert!(matches!(refused, LockFail::Unavailable(_)), "{refused:?}");
        let _ = std::fs::remove_dir_all(&base);
    }
}
