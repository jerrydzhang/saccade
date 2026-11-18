use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;

use crate::attempts::{AsReceived, Attempts, Outcome};
use crate::db::{self, ExecuteFail, StoredRecord};
use crate::record::events::Command;
use crate::store::{Context, World};

use super::api::TAURI_ORIGINS;
use super::fail::ApiFail;

pub(super) enum ServerState {
    Ready(Inner),
    Degraded(String, rusqlite::Connection),
}

pub(super) struct Inner {
    pub(super) conn: rusqlite::Connection,
    /// The tracker's own db: the worktree home derives from it
    pub(super) db_path: std::path::PathBuf,
}
/// The sole writer's submit serialization: one lock per recipient
/// incarnation, so concurrent sends to one recipient submit in record
/// order even when its address moves mid-flight - an address key
/// would let sends on either side of a move take different locks and
/// interleave. Runtime facts only - never record identity, authority,
/// or obligation; lost on restart.
pub struct SubmitLocks {
    locks: std::sync::Mutex<HashMap<usize, Arc<Mutex<()>>>>,
}

impl Default for SubmitLocks {
    fn default() -> Self {
        SubmitLocks {
            locks: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl SubmitLocks {
    /// Acquire the per-recipient lock: concurrent sends to one
    /// recipient serialize here; different recipients never contend.
    /// The lock is synchronous because the adapter's submit is a
    /// blocking subprocess call (the bounded attempt's own bound
    /// contains it).
    pub fn acquire(
        &self,
        to: crate::record::objects::incarnation::IncarnationId,
    ) -> Arc<Mutex<()>> {
        let mut map = self.locks.lock().expect("the submit locks hold");
        map.entry(to.0.0)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}
/// The landing wake: a parked read's fast path, fired when this
/// server's own door lands a write. Never a timer, never delivery -
/// parked reads re-check their predicate against the fold.
#[derive(Clone, Default)]
pub struct LandingWake {
    notify: Arc<tokio::sync::Notify>,
}

impl LandingWake {
    /// Fire the wake: parked holds re-evaluate. Called where a write
    /// this wake watches has landed.
    pub fn fire(&self) {
        self.notify.notify_waiters();
    }

    /// A parked read's hold on the landing wake, used by the page
    /// door. `decide` runs once the registration has entered the
    /// wake's wait list, so a write landing between the deciding read
    /// and the park wakes the hold - it can never be missed. The
    /// hold answers `None` when the wake fired or the bound expired;
    /// the caller re-evaluates the page either way. Wake-driven: one
    /// registration, one decide, one wake-or-deadline wait, no
    /// periodic re-checks.
    pub async fn hold<R>(
        &self,
        mut decide: impl FnMut() -> Option<R>,
        bound: Duration,
    ) -> Option<R> {
        let mut notified = std::pin::pin!(self.notify.notified());
        let _ = notified.as_mut().enable();
        if let Some(answer) = decide() {
            return Some(answer);
        }
        let _ = tokio::time::timeout(bound, notified).await;
        None
    }
}

/// The sole writer's state: every face loads the log fresh under the
/// one lock, so no read can serve a world the file contradicts.
#[derive(Clone)]
pub struct AppState {
    pub(super) submit_locks: Arc<SubmitLocks>,
    /// the landing wake: parked reads' fast path when this server's
    /// own door lands a write
    pub landing: LandingWake,
    pub(super) inner: Arc<Mutex<ServerState>>,
    pub(super) db_path: std::path::PathBuf,
    pub(super) attempts: Attempts,
    /// The serving repo's root, when boot resolved one.
    repo_root: Option<std::path::PathBuf>,
    /// This server's own URL, published at bind: the routing a launched
    /// session needs to reach the tracker it was started from.
    server_url: Option<String>,
    /// The origins whose browser pages may read this server: the Tauri
    /// shells by default, the served frontend's origin configured at
    /// serve time. A request naming another origin refuses; a request
    /// naming none (the CLI, curl) never did come from a page.
    pub(crate) allowed_origins: Arc<[String]>,
    /// The extra Host headers a request may name, configured at serve
    /// time: loopback names and the bind address are always accepted,
    /// anything else is a rebound name and refuses before routing.
    pub(crate) allowed_hosts: Arc<[String]>,
}

pub struct Snapshot {
    pub world: World,
    pub rows: Vec<StoredRecord>,
}

/// Degraded still serves the raw rows: the stream is the log's face when
/// the world will not fold.
#[derive(Debug)]
pub struct Degraded {
    pub reason: String,
    pub rows: Vec<StoredRecord>,
}

/// A read that cannot present the world: the log will not fold (its
/// raw rows still travel), or will not even load - corruption and
/// storage failures are database errors, never an empty successful log.
#[derive(Debug)]
pub enum ReadFail {
    Degraded(Degraded),
    Db(String),
}

impl ReadFail {
    pub fn reason(&self) -> &str {
        match self {
            ReadFail::Degraded(d) => &d.reason,
            ReadFail::Db(reason) => reason,
        }
    }
}

impl AppState {
    pub fn open(db_path: &Path) -> Result<Self, String> {
        // the worktree home derives from the db's own directory, so the
        // path is absolute from the start - a fresh tracker resolves
        // through its parent, an existing one canonically
        let canonical = if db_path.exists() {
            std::fs::canonicalize(db_path)
                .map_err(|e| format!("database {}: {e}", db_path.display()))?
        } else {
            let parent = db_path.parent().unwrap_or_else(|| Path::new("."));
            let parent = if parent.as_os_str().is_empty() {
                Path::new(".").to_path_buf()
            } else {
                parent.to_path_buf()
            };
            let canonical_parent = std::fs::canonicalize(&parent)
                .map_err(|e| format!("database {}: {e}", db_path.display()))?;
            canonical_parent.join(db_path.file_name().unwrap_or_default())
        };
        let conn = db::open(&canonical).map_err(|e| format!("database: {e}"))?;
        let state = match db::load(&conn).map_err(|e| format!("database: {e}"))?.state {
            db::LoadState::Full(_) => ServerState::Ready(Inner {
                conn,
                db_path: canonical.clone(),
            }),
            db::LoadState::Degraded(reason) => ServerState::Degraded(reason, conn),
        };
        Ok(AppState {
            submit_locks: Arc::new(SubmitLocks::default()),
            landing: LandingWake::default(),
            inner: Arc::new(Mutex::new(state)),
            db_path: canonical,
            attempts: Attempts::beside(db_path),
            repo_root: None,
            server_url: None,
            allowed_origins: TAURI_ORIGINS
                .iter()
                .map(|origin| origin.to_string())
                .collect(),
            allowed_hosts: Arc::from([]),
        })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Name the serving repo's root.
    pub fn with_repo_root(mut self, root: std::path::PathBuf) -> Self {
        self.repo_root = Some(root);
        self
    }

    /// Extend the origin allowlist with serve-time entries: the served
    /// frontend's origin in the lanes, a component's origin in deploy.
    pub fn with_allowed_origins(mut self, origins: Vec<String>) -> Self {
        let mut allowed = self.allowed_origins.to_vec();
        allowed.extend(origins);
        self.allowed_origins = allowed.into();
        self
    }

    /// Name the extra hosts a request may name (serve-time entries).
    pub fn with_allowed_hosts(mut self, hosts: Vec<String>) -> Self {
        let mut allowed = self.allowed_hosts.to_vec();
        allowed.extend(hosts);
        self.allowed_hosts = allowed.into();
        self
    }

    /// Name this server's own published URL.
    pub fn with_server_url(mut self, url: String) -> Self {
        self.server_url = Some(url);
        self
    }

    /// This server's own published URL, when bound.
    pub fn server_url(&self) -> Option<&str> {
        self.server_url.as_deref()
    }

    /// The serving repo's root, when boot resolved one.
    pub fn repo_root(&self) -> Option<&Path> {
        self.repo_root.as_deref()
    }

    /// The judging seam: one write attempt, one line in the attempts
    /// log, appended while the writer lock is held so the file's order
    /// is the judgment order. The line names the door that judged
    /// (`door`); duration includes the lock wait - the queueing a
    /// request met is part of what the diagnoser wants.
    pub fn execute(
        &self,
        door: &str,
        context: &Context,
        command: Command,
        at: Option<u64>,
        received: AsReceived,
    ) -> Result<Vec<StoredRecord>, ExecuteFail> {
        let started = Instant::now();
        let mut guard = self.inner.lock().expect("the writer lock is not poisoned");
        // the cursor names the log as it stands at judgment, external
        // writes included
        let cursor = match &*guard {
            ServerState::Ready(inner) => Some(db::last_seq(&inner.conn).map_err(ExecuteFail::Db)?),
            ServerState::Degraded(..) => None,
        }
        .flatten();
        let outcome = match &mut *guard {
            ServerState::Degraded(reason, _) => Err(ExecuteFail::Degraded(reason.clone())),
            ServerState::Ready(inner) => db::record(
                &mut inner.conn,
                context,
                command,
                at.unwrap_or_else(db::now_epoch),
            )
            .map(|(stored, _)| stored),
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        let code = match &outcome {
            Ok(_) => None,
            Err(ExecuteFail::Reject(reject)) => Some(reject.code()),
            Err(ExecuteFail::Degraded(_)) => Some("degraded"),
            Err(ExecuteFail::Db(_)) => Some("storage"),
        };
        match &outcome {
            Ok(stored) => {
                // a landed write may be a parked read's predicate: the
                // landing is the wake, never a timer
                self.wake_landing();
                self.attempts.append(
                    door,
                    Some(context.actor.as_str()),
                    received.client.as_deref(),
                    duration_ms,
                    Outcome::Landed {
                        seqs: &stored.iter().map(|s| s.seq).collect::<Vec<_>>(),
                    },
                );
            }
            Err(_) => self.attempts.append(
                door,
                Some(context.actor.as_str()),
                received.client.as_deref(),
                duration_ms,
                Outcome::Refused {
                    code: code.expect("a refusal names its code"),
                    cursor,
                    request: &received.raw,
                },
            ),
        }
        outcome
    }

    /// Fire the landing wake: call wherever this server's own door
    /// landed a write outside `execute`. A parked page hold wakes and
    /// re-reads the fold.
    pub(crate) fn wake_landing(&self) {
        self.landing.notify.notify_waiters();
    }

    pub fn snapshot(&self) -> Result<Snapshot, ReadFail> {
        let guard = self.inner.lock().expect("the writer lock is not poisoned");
        let (boot_reason, conn) = match &*guard {
            ServerState::Ready(inner) => (None, &inner.conn),
            ServerState::Degraded(reason, conn) => (Some(reason.clone()), conn),
        };
        let loadout = db::load(conn).map_err(|e| ReadFail::Db(e.to_string()))?;
        match loadout.state {
            db::LoadState::Full(world) => match boot_reason {
                None => Ok(Snapshot {
                    world,
                    rows: loadout.rows,
                }),
                Some(reason) => Err(ReadFail::Degraded(Degraded {
                    reason,
                    rows: loadout.rows,
                })),
            },
            db::LoadState::Degraded(fresh) => Err(ReadFail::Degraded(Degraded {
                reason: boot_reason.unwrap_or(fresh),
                rows: loadout.rows,
            })),
        }
    }
}
pub(super) fn boards_of(app: &AppState) -> Result<crate::boards::Boards, ApiFail> {
    let root = app.repo_root().ok_or(ApiFail::NoRepo)?;
    crate::boards::Boards::open(&crate::paths::state_dir(root)).map_err(|fail| {
        ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            fail.code(),
            fail.detail(),
        )
    })
}

pub(super) fn snapshot_of(app: &AppState) -> Result<Snapshot, ApiFail> {
    app.snapshot().map_err(|fail| match fail {
        ReadFail::Degraded(degraded) => ApiFail::Degraded(degraded.reason),
        ReadFail::Db(reason) => ApiFail::Db(reason),
    })
}
#[cfg(test)]
mod hold_test {
    use super::*;

    // The page hold's own contract: one registration, one decide, one
    // wake-or-deadline wait - never a periodic re-check, and a
    // landing after the registration is never missed.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_hold_runs_wake_driven_to_its_deadline_without_periodic_rechecks() {
        let wake = LandingWake::default();
        // no landing: the hold runs its full bound, deciding once
        let started = std::time::Instant::now();
        let decides = std::cell::Cell::new(0u32);
        let answer = wake
            .hold(
                || {
                    decides.set(decides.get() + 1);
                    None::<u8>
                },
                std::time::Duration::from_millis(600),
            )
            .await;
        assert_eq!(answer, None);
        assert_eq!(
            decides.get(),
            1,
            "one decide per hold, no periodic re-checks"
        );
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(590),
            "the hold ran to its deadline"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_millis(1500),
            "the hold did not overshoot its deadline"
        );

        // a landing during the decide wakes the hold at once: the
        // registration was entered before the decide ran
        let waker = wake.clone();
        let started = std::time::Instant::now();
        let answer = wake
            .hold(
                || {
                    let waker = waker.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        waker.notify.notify_waiters();
                    });
                    None::<u8>
                },
                std::time::Duration::from_secs(30),
            )
            .await;
        assert_eq!(
            answer, None,
            "the wake ends the hold, the caller re-decides"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "the landing woke the hold long before the bound"
        );

        // the decide answering Some ends the hold at once
        let answer = wake
            .hold(|| Some(7u8), std::time::Duration::from_secs(30))
            .await;
        assert_eq!(answer, Some(7));
    }
}
