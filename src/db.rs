use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use crate::Reject;
use crate::events::{Command, Event};
use crate::store::{Context, Record, RecordId, World, execute};
use crate::types::actor::ActorName;
use crate::wire;

/// file-header stamp identifying a Saccade database.
const APPLICATION_ID: i32 = i32::from_be_bytes(*b"sacd");
/// The schema this binary reads and writes.
pub const SCHEMA_VERSION: i32 = 4;
/// The schema the offline migration reads: Saccade 0.13.0's stamp.
pub const MIGRATE_FROM_VERSION: i32 = 3;

#[derive(Debug)]
pub enum DbError {
    Sqlite(rusqlite::Error),
    /// creating the tracker's parent directories failed
    Io(String),
    /// application_id mismatch: not a Saccade database.
    NotSaccade,
    /// user_version ahead of this binary: written by a newer Saccade.
    NewerSchema(i32),
    /// user_version behind this binary: an unmigrated log.
    OlderSchema(i32),
    /// the migration's source is not the schema it reads
    /// Known kind, unparseable payload: corruption. Loading refuses.
    /// Read opened a path with no database; mutating commands create one.
    Missing(std::path::PathBuf),
    Corrupt {
        seq: usize,
        kind: String,
        detail: String,
    },
}

impl From<rusqlite::Error> for DbError {
    fn from(e: rusqlite::Error) -> Self {
        DbError::Sqlite(e)
    }
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::Sqlite(e) => write!(f, "sqlite: {e}"),
            DbError::Io(e) => write!(f, "filesystem: {e}"),
            DbError::NotSaccade => write!(f, "file is not a saccade database"),
            DbError::NewerSchema(v) => {
                write!(f, "database schema v{v} is newer than this binary; upgrade")
            }
            DbError::OlderSchema(v) => {
                write!(f, "database schema v{v} predates this binary")
            }
            DbError::Missing(path) => write!(
                f,
                "no saccade database at {}\nany mutating command creates one",
                path.display()
            ),
            DbError::Corrupt { seq, kind, detail } => {
                write!(f, "corrupt record at seq {seq} (kind {kind}): {detail}")
            }
        }
    }
}

/// A single row of the records table, and the serde shape of a record on
/// every surface: the wire, the API response, and `--json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRecord {
    pub seq: usize,
    pub event_time: u64,
    pub logged_time: u64,
    pub actor: String,
    pub tier: String,
    pub kind: String,
    #[serde(with = "payload_json")]
    pub payload: String,
    /// The record's environment-declared scope, when its author carried one
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// Payloads are stored as JSON text and travel as the JSON itself.
mod payload_json {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(payload: &str, ser: S) -> Result<S::Ok, S::Error> {
        match serde_json::from_str::<serde_json::Value>(payload) {
            Ok(value) => value.serialize(ser),
            Err(_) => payload.serialize(ser),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<String, D::Error> {
        let value = serde_json::Value::deserialize(de)?;
        Ok(value.to_string())
    }
}

pub enum LoadState {
    /// Every row understood: the world folds.
    Full(World),
    /// Rows retained raw; world projection and writes refused: version skew
    /// or a record that does not fold.
    Degraded(String),
}

pub struct Loadout {
    /// Raw rows in seq order
    pub rows: Vec<StoredRecord>,
    pub state: LoadState,
}

impl std::error::Error for DbError {}

#[derive(Debug, thiserror::Error)]
pub enum ExecuteFail {
    #[error("storage: {0}")]
    Db(#[from] DbError),
    #[error("world projection unavailable: {0}")]
    Degraded(String),
    #[error("{0}")]
    Reject(#[from] Reject),
}

impl From<rusqlite::Error> for ExecuteFail {
    fn from(e: rusqlite::Error) -> Self {
        ExecuteFail::Db(e.into())
    }
}

pub fn open(path: &Path) -> Result<Connection, DbError> {
    // a tracker's state root may not exist yet; creating it is part of
    // creating the tracker
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DbError::Io(e.to_string()))?;
    }
    let conn = Connection::open(path)?;
    configure(&conn)?;
    // journal_mode returns the resulting mode as a row; read and discard.
    conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    init(&conn)?;
    Ok(conn)
}

/// Read path: never creates, never mutates. A missing file is a named error,
/// not an empty log.
pub fn open_read(path: &Path) -> Result<Connection, DbError> {
    if !path.exists() {
        return Err(DbError::Missing(path.to_path_buf()));
    }
    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(Duration::from_millis(5000))?;
    verify(&conn)?;
    Ok(conn)
}

/// The non-creating half of init's match: named rejections for foreign,
/// unstamped-but-nonempty, and newer-schema files.
fn verify(conn: &Connection) -> Result<(), DbError> {
    let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let app_id: i32 = conn.query_row("PRAGMA application_id", [], |r| r.get(0))?;
    match (app_id, version) {
        (APPLICATION_ID, SCHEMA_VERSION) => Ok(()),
        (APPLICATION_ID, v) if v > SCHEMA_VERSION => Err(DbError::NewerSchema(v)),
        (APPLICATION_ID, v) if v > 0 => Err(DbError::OlderSchema(v)),
        _ => Err(DbError::NotSaccade),
    }
}

fn configure(conn: &Connection) -> Result<(), DbError> {
    conn.busy_timeout(Duration::from_millis(5000))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

/// Stamp the file header and create the log if needed. One write transaction,
/// so a crash mid-init leaves the file untouched rather than half-stamped.
fn init(conn: &Connection) -> Result<(), DbError> {
    let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let app_id: i32 = conn.query_row("PRAGMA application_id", [], |r| r.get(0))?;
    if (app_id, version) != (0, 0) {
        return verify(conn);
    }

    // Zero means unclaimed, not empty: SQLite zero-initializes both
    // header fields and never touches them. A foreign database that
    // has never set its stamps also reads (0, 0)
    let user_objects: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'",
        [],
        |r| r.get(0),
    )?;
    if user_objects > 0 {
        return Err(DbError::NotSaccade);
    }

    let sql = format!(
        "BEGIN IMMEDIATE;
         CREATE TABLE events (
             seq         INTEGER PRIMARY KEY,
             event_time  INTEGER NOT NULL,
             logged_time INTEGER NOT NULL,
             actor       TEXT    NOT NULL,
             tier        TEXT    NOT NULL,
             kind        TEXT    NOT NULL,
             payload     TEXT    NOT NULL,
             scope       TEXT    NULL
         );
         CREATE TRIGGER events_no_update BEFORE UPDATE ON events
             BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
         CREATE TRIGGER events_no_delete BEFORE DELETE ON events
             BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
         PRAGMA application_id = {APPLICATION_ID};
         PRAGMA user_version = {SCHEMA_VERSION};
         COMMIT;",
    );
    conn.execute_batch(&sql).map_err(DbError::from)
}

pub fn load(conn: &Connection) -> Result<Loadout, DbError> {
    fold_rows(read_rows(conn)?)
}

pub fn fold_rows(rows: Vec<StoredRecord>) -> Result<Loadout, DbError> {
    let (records, degraded) = parse_rows(&rows)?;
    let state = match degraded {
        Some(reason) => LoadState::Degraded(reason),
        None => match World::replay(records) {
            Ok(world) => LoadState::Full(world),
            Err(err) => LoadState::Degraded(format!(
                "record at seq {} does not fold: {:?}",
                err.at.0, err.reason
            )),
        },
    };
    Ok(Loadout { rows, state })
}

fn read_rows(conn: &Connection) -> Result<Vec<StoredRecord>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT seq, event_time, logged_time, actor, tier, kind, payload, scope
         FROM events ORDER BY seq",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok(StoredRecord {
                seq: row.get::<_, i64>(0)? as usize,
                event_time: row.get::<_, i64>(1)? as u64,
                logged_time: row.get::<_, i64>(2)? as u64,
                actor: row.get(3)?,
                tier: row.get(4)?,
                kind: row.get(5)?,
                payload: row.get(6)?,
                scope: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Rows to records; unknown kind/tier stops at the first as a
/// degradation, malformed payloads of known kinds are corruption.
fn parse_rows(rows: &[StoredRecord]) -> Result<(Vec<Record>, Option<String>), DbError> {
    let mut degraded: Option<String> = None;
    let mut records = Vec::with_capacity(rows.len());
    for row in rows {
        let tier = match wire::tier_from(&row.tier) {
            Ok(t) => t,
            Err(_) => {
                degraded = Some(format!("unknown tier '{}' at seq {}", row.tier, row.seq));
                break;
            }
        };
        let event = match wire::assemble(&row.kind, &row.payload) {
            Ok(event) => event,
            Err(wire::ParseFail::UnknownKind(kind)) => {
                degraded = Some(format!("unknown kind '{kind}' at seq {}", row.seq));
                break;
            }
            Err(wire::ParseFail::UnknownTier(tier)) => {
                degraded = Some(format!("unknown tier '{tier}' at seq {}", row.seq));
                break;
            }
            Err(wire::ParseFail::Malformed { detail, .. }) => {
                return Err(DbError::Corrupt {
                    seq: row.seq,
                    kind: row.kind.clone(),
                    detail,
                });
            }
        };
        records.push(Record {
            id: RecordId(row.seq),
            timestamp: row.event_time,
            context: Context {
                actor: ActorName::new(row.actor.clone()).map_err(|_| DbError::Corrupt {
                    seq: row.seq,
                    kind: row.kind.clone(),
                    detail: "invalid actor name".into(),
                })?,
                tier,
                scope: row
                    .scope
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(|_| DbError::Corrupt {
                        seq: row.seq,
                        kind: row.kind.clone(),
                        detail: "invalid scope".into(),
                    })?,
            },
            event,
        });
    }
    Ok((records, degraded))
}

/// The log's last storage seq, the cursor a refusal names: read fresh
/// so it counts writes the file took outside this process.
pub fn last_seq(conn: &Connection) -> Result<Option<usize>, DbError> {
    conn.query_row("SELECT max(seq) FROM events", [], |r| {
        Ok(r.get::<_, Option<i64>>(0)?.map(|s| s as usize))
    })
    .map_err(DbError::from)
}

/// An event plus the context claiming it, before it is a record
pub struct RecordDraft {
    pub context: Context,
    pub event: Event,
}

/// A mixed-context append: every draft folds or none of them lands
pub struct AtomicBatch {
    pub drafts: Vec<RecordDraft>,
}

pub fn record(
    conn: &mut Connection,
    context: &Context,
    command: Command,
    event_time: u64,
) -> Result<(Vec<StoredRecord>, World), ExecuteFail> {
    let txn = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let loadout = load(&txn)?;
    let world = match loadout.state {
        LoadState::Full(world) => world,
        LoadState::Degraded(reason) => return Err(ExecuteFail::Degraded(reason)),
    };

    let (world, records) = execute(&world, loadout.rows.len(), context, command, event_time)?;
    let stored = persist(&txn, &records, event_time)?;
    txn.commit()?;
    Ok((stored, world))
}

/// Clone the log's prefix through `at` into a fresh tracker at `out`:
/// the schema and identity stamps first, then the rows in order - the
/// cursor-cloning recipe as one verb. The cursor is the storage seq
/// read from the rows, never a record id; the two series need not
/// agree. In-place truncation is impossible (the append-only triggers),
/// so a clone is always a fresh file.
pub fn clone(source: &Path, at: usize, out: &Path) -> Result<usize, DbError> {
    let src = open_read(source)?;
    let rows = load(&src)?.rows;
    clone_rows(&rows, at, out)
}

pub fn clone_rows(rows: &[StoredRecord], at: usize, out: &Path) -> Result<usize, DbError> {
    if out.exists() {
        return Err(DbError::Io(format!(
            "refusing to overwrite {}; clone to a fresh path",
            out.display()
        )));
    }
    let Some(last) = rows.last() else {
        return Err(DbError::Io(format!(
            "the log holds no records; --at {at} names nothing"
        )));
    };
    if at > last.seq {
        return Err(DbError::Io(format!(
            "--at {at} is beyond the log's last seq {}",
            last.seq
        )));
    }
    let mut dst = open(out)?;
    let txn = dst.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let mut copied = 0usize;
    for row in rows.iter().take_while(|r| r.seq <= at) {
        txn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                row.seq as i64,
                row.event_time as i64,
                row.logged_time as i64,
                row.actor,
                row.tier,
                row.kind,
                row.payload,
                row.scope
            ],
        )?;
        copied += 1;
    }
    txn.commit()?;
    Ok(copied)
}

/// Fold-validated all-or-none append of a mixed-context batch; positional ids
/// are assigned inside the transaction. No authority here: batches are
/// programmatic, tier gates live in the command path.
pub fn execute_batch(
    conn: &mut Connection,
    batch: AtomicBatch,
    event_time: u64,
) -> Result<Vec<StoredRecord>, ExecuteFail> {
    let txn = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let loadout = load(&txn)?;
    let world = match loadout.state {
        LoadState::Full(world) => world,
        LoadState::Degraded(reason) => return Err(ExecuteFail::Degraded(reason)),
    };

    let records: Vec<Record> = batch
        .drafts
        .into_iter()
        .enumerate()
        .map(|(i, draft)| Record {
            id: RecordId(loadout.rows.len() + i),
            timestamp: event_time,
            context: draft.context,
            event: draft.event,
        })
        .collect();
    world
        .clone()
        .fold(records.clone())
        .map_err(|err| ExecuteFail::Reject(err.reason.into()))?;
    let stored = persist(&txn, &records, event_time)?;
    txn.commit()?;
    Ok(stored)
}

/// Writes records into the database, It implicitly trusts that the records fold validly, the
/// validation check must be done before calling this function
fn persist(
    txn: &Transaction,
    records: &[Record],
    event_time: u64,
) -> Result<Vec<StoredRecord>, ExecuteFail> {
    let logged_time = now_epoch();
    let mut stored = Vec::with_capacity(records.len());
    for record in records {
        let tier = wire::tier_of(&record.context.tier);
        let (kind, payload) = wire::disassemble(&record.event);
        let scope = record
            .context
            .scope
            .as_ref()
            .map(|s| serde_json::to_string(s).expect("scope is plain data"));
        stored.push(StoredRecord {
            seq: record.id.0,
            event_time,
            logged_time,
            actor: record.context.actor.as_str().to_string(),
            tier: tier.to_string(),
            kind: kind.clone(),
            payload: payload.clone(),
            scope: scope.clone(),
        });
        txn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                record.id.0 as i64,
                event_time as i64,
                logged_time as i64,
                record.context.actor.as_str(),
                tier,
                kind,
                payload,
                scope
            ],
        )?;
    }
    Ok(stored)
}

pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before 1970")
        .as_secs()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::events::{Command, Event};
    use crate::objects::comment::CommentKind;
    use crate::objects::comment::{CommentState, Revision, Withdrawal};
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::task::{TaskId, TaskState};
    use crate::store::Tier;
    use crate::types::artifact::Artifact;
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};
    use crate::{CommentId, ContentHash, Prose, RecordId, Scope};

    fn memory_db() -> Connection {
        let conn = Connection::open_in_memory().expect("open memory db");
        conn.busy_timeout(Duration::from_millis(5000)).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        init(&conn).expect("init schema");
        conn
    }

    fn agent() -> Context {
        Context {
            actor: ActorName::new("saccade bot".into()).unwrap(),
            tier: Tier::Agent,
            scope: None,
        }
    }

    fn human() -> Context {
        Context {
            actor: ActorName::new("human person".into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        }
    }

    /// A task-bound session's context.
    fn scoped_agent() -> Context {
        Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope: Some(Scope {
                task: Some(TaskId(0)),
                incarnation: None,
                workspace: Some(WorktreePath::new("/repo/wt/t-0".into()).unwrap()),
            }),
        }
    }

    fn create(name: &str) -> Command {
        Command::CreateTask {
            name: Prose::new(name.into()).unwrap(),
        }
    }

    fn offer(task: usize, body: &str) -> Command {
        Command::Comment {
            home: TaskId(task),
            kind: CommentKind::Offer {
                body: Prose::new(body.into()).unwrap(),
            },
            references: Vec::new(),
        }
    }

    fn accept_on(home: usize, seq: usize) -> Command {
        Command::Comment {
            home: TaskId(home),
            kind: CommentKind::Accept {
                offer: CommentId(RecordId(seq)),
            },
            references: Vec::new(),
        }
    }

    fn reject_on(home: usize, seq: usize, grounds: &str) -> Command {
        Command::Comment {
            home: TaskId(home),
            kind: CommentKind::Reject {
                offer: CommentId(RecordId(seq)),
                grounds: Prose::new(grounds.into()).unwrap(),
            },
            references: Vec::new(),
        }
    }

    fn prose(text: &str) -> Prose {
        Prose::new(text.into()).unwrap()
    }

    fn note(body: &str) -> Command {
        Command::Comment {
            home: TaskId(0),
            kind: CommentKind::Note {
                body: Prose::new(body.into()).unwrap(),
            },
            references: Vec::new(),
        }
    }

    #[test]
    fn execute_then_load_round_trips_the_world() {
        let mut conn = memory_db();
        record(&mut conn, &human(), create("implement foo"), 10).unwrap();
        record(
            &mut conn,
            &agent(),
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: None,
            },
            11,
        )
        .unwrap();
        record(&mut conn, &scoped_agent(), note("mid-flight finding"), 11).unwrap();
        record(&mut conn, &agent(), offer(0, "suite green, 55 tests"), 12).unwrap();
        record(&mut conn, &human(), accept_on(0, 3), 12).unwrap();
        record(&mut conn, &human(), create("fix bar"), 13).unwrap();
        record(&mut conn, &agent(), offer(1, "bar fixed the quick way"), 14).unwrap();
        record(
            &mut conn,
            &human(),
            reject_on(1, 6, "the edge case still fails"),
            15,
        )
        .unwrap();
        record(&mut conn, &agent(), offer(1, "edge case held"), 16).unwrap();
        record(&mut conn, &human(), accept_on(1, 8), 16).unwrap();
        record(&mut conn, &human(), create("migrate floop"), 17).unwrap();
        record(
            &mut conn,
            &human(),
            Command::CancelTask {
                id: TaskId(2),
                explanation: prose("no longer wanted"),
            },
            18,
        )
        .unwrap();
        record(
            &mut conn,
            &human(),
            Command::ReopenTask {
                id: TaskId(2),
                explanation: prose("the blocker cleared"),
            },
            19,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            note("the migration waits on a decision"),
            20,
        )
        .unwrap();
        record(&mut conn, &human(), create("adopt foo's account"), 21).unwrap();
        // the prospective use names its source; the binding is its own record
        record(
            &mut conn,
            &agent(),
            Command::Comment {
                home: TaskId(3),
                kind: CommentKind::Use {
                    source: TaskId(0),
                    reason: prose("builds on the landed account"),
                    bound: None,
                },
                references: Vec::new(),
            },
            21,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::BindUse {
                id: CommentId(RecordId(15)),
                offer: CommentId(RecordId(3)),
                grounds: prose("foo's accepted account is the one that landed"),
            },
            22,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::Artifact {
                root: TaskId(0),
                artifact: Artifact {
                    name: Prose::new("sweep figure".into()).unwrap(),
                    hash: ContentHash::of(b"figure bytes"),
                },
            },
            23,
        )
        .unwrap();
        let system = Context::system();
        record(
            &mut conn,
            &system,
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            22,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("def456".into()).unwrap(),
            },
            22,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: Some(IncarnationId(RecordId(18))),
            },
            22,
        )
        .unwrap();
        let batch = AtomicBatch {
            drafts: vec![
                RecordDraft {
                    context: agent(),
                    event: Event::TaskCreated {
                        name: Prose::new("batchwork".into()).unwrap(),
                    },
                },
                RecordDraft {
                    context: human(),
                    event: Event::TaskClaimed {
                        id: TaskId(4),
                        incarnation: None,
                    },
                },
            ],
        };
        let batched = execute_batch(&mut conn, batch, 24).unwrap();
        assert_eq!(batched.len(), 2);
        record(&mut conn, &agent(), note("post-fold receipt"), 25).unwrap();
        record(
            &mut conn,
            &agent(),
            Command::ReviseComment {
                id: CommentId(RecordId(23)),
                body: prose("post-fold receipt, corrected"),
            },
            26,
        )
        .unwrap();
        record(&mut conn, &agent(), note("parked mid-flight"), 27).unwrap();
        record(
            &mut conn,
            &human(),
            Command::WithdrawComment {
                id: CommentId(RecordId(25)),
                note: prose("the parking was premature"),
            },
            27,
        )
        .unwrap();
        // a second source, replaced and rebound: the rebind family rides
        record(&mut conn, &human(), create("rewrite bar properly"), 28).unwrap();
        record(&mut conn, &agent(), offer(5, "the deeper fix"), 28).unwrap();
        record(
            &mut conn,
            &agent(),
            Command::Comment {
                home: TaskId(3),
                kind: CommentKind::Use {
                    source: TaskId(5),
                    reason: prose("bar's rework is the real foundation"),
                    bound: None,
                },
                references: Vec::new(),
            },
            29,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::BindUse {
                id: CommentId(RecordId(29)),
                offer: CommentId(RecordId(28)),
                grounds: prose("the standing account"),
            },
            29,
        )
        .unwrap();
        record(
            &mut conn,
            &human(),
            reject_on(5, 28, "the fix misses the index"),
            30,
        )
        .unwrap();
        record(&mut conn, &agent(), offer(5, "the fix that holds"), 30).unwrap();
        record(
            &mut conn,
            &agent(),
            Command::BindUse {
                id: CommentId(RecordId(29)),
                offer: CommentId(RecordId(32)),
                grounds: prose("the replacement held under review"),
            },
            31,
        )
        .unwrap();
        // a citation rides its mapping through the wire
        let _ = record(
            &mut conn,
            &agent(),
            Command::Comment {
                home: TaskId(3),
                kind: CommentKind::Note {
                    body: prose("rests on the conclusion above [^1], again [^1]"),
                },
                references: vec![RecordId(3)],
            },
            32,
        )
        .unwrap();

        // the native families ride the same round-trip: registration,
        // an addressed message, the recipient's submission
        record(
            &mut conn,
            &system,
            Command::RegisterIncarnation {
                native_id: crate::NativeSessionId::new("aaaa1111".into()).unwrap(),
                saved_session: crate::SessionLocator::new("/sessions/aaaa1111.jsonl".into())
                    .unwrap(),
            },
            33,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::RegisterIncarnation {
                native_id: crate::NativeSessionId::new("bbbb2222".into()).unwrap(),
                saved_session: crate::SessionLocator::new("/sessions/bbbb2222.jsonl".into())
                    .unwrap(),
            },
            33,
        )
        .unwrap();
        let sender = Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope: Some(Scope {
                task: None,
                incarnation: Some(IncarnationId(RecordId(35))),
                workspace: None,
            }),
        };
        record(
            &mut conn,
            &sender,
            Command::SendMessage {
                from: IncarnationId(RecordId(35)),
                to: IncarnationId(RecordId(36)),
                reply_to: None,
                task: Some(TaskId(0)),
                body: prose(
                    "please look at the failing edge case: caf\u{e9} floop, na\u{ef}ve fix",
                ),
            },
            34,
        )
        .unwrap();
        let (_, returned) = record(
            &mut conn,
            &system,
            Command::SubmitMessage {
                id: crate::MessageId(RecordId(37)),
                by: IncarnationId(RecordId(36)),
            },
            35,
        )
        .unwrap();

        let loadout = load(&conn).unwrap();
        let LoadState::Full(world) = loadout.state else {
            panic!("expected a full load");
        };
        assert_eq!(returned, world);
        assert_eq!(loadout.rows.len(), 39);
        // the task families ride their own kinds
        assert_eq!(loadout.rows[0].kind, "task_created");
        assert_eq!(loadout.rows[1].kind, "task_claimed");
        assert_eq!(loadout.rows[11].kind, "task_cancelled");
        assert_eq!(loadout.rows[12].kind, "task_reopened");
        assert_eq!(loadout.rows[21].kind, "task_created");
        assert_eq!(loadout.rows[22].kind, "task_claimed");
        // the workspace establishment is machinery-authored and births
        // the shared identity the claim names
        assert_eq!(loadout.rows[18].kind, "workspace_established");
        assert_eq!(loadout.rows[19].kind, "task_workspace_checkpointed");
        assert_eq!(loadout.rows[18].actor.as_str(), "saccade");
        assert_eq!(
            world.incarnations[&IncarnationId(RecordId(18))].incarnation,
            crate::Incarnation::Established {
                task: TaskId(0),
                workspace: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            }
        );
        assert_eq!(
            world.tasks[0].workspace.as_ref().unwrap().birth,
            RecordId(18)
        );
        // the scoped record carries its declared scope; unscoped rows none
        assert!(
            loadout.rows[2]
                .scope
                .as_deref()
                .is_some_and(|s| s.contains("wt/t-0")),
            "{}",
            loadout.rows[2].scope.clone().unwrap_or_default()
        );
        assert_eq!(loadout.rows[0].scope, None);
        assert_eq!(
            world.comments[&CommentId(RecordId(2))].scope,
            scoped_agent().scope
        );
        // the comment families ride the commented kind with their
        // variant payloads
        for (seq, variant) in [
            (3, "offer"),
            (4, "accept"),
            (7, "reject"),
            (13, "note"),
            (15, "use"),
        ] {
            assert_eq!(loadout.rows[seq].kind, "commented");
            assert!(
                loadout.rows[seq].payload.contains(variant),
                "seq {seq}: {}",
                loadout.rows[seq].payload
            );
        }
        assert!(
            loadout.rows[7]
                .payload
                .contains("the edge case still fails")
        );
        // a prospective use names its source task, never an offer
        assert!(loadout.rows[15].payload.contains("\"source\":0"));
        // the binding family: the use, the selected offer, the grounds
        for seq in [16, 33] {
            assert_eq!(loadout.rows[seq].kind, "use_bound", "seq {seq}");
            assert!(
                loadout.rows[seq].payload.contains("grounds"),
                "seq {seq}: {}",
                loadout.rows[seq].payload
            );
        }
        // the world folds the bindings: the state carries the
        // selection, the context its attribution - foo's account,
        // then the replacement
        assert_eq!(
            world.comments[&CommentId(RecordId(15))]
                .comment
                .state
                .payload_offer(),
            Some(CommentId(RecordId(3)))
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(29))]
                .binding
                .as_ref()
                .map(|b| b.record),
            Some(RecordId(33))
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(29))]
                .comment
                .state
                .payload_offer(),
            Some(CommentId(RecordId(32)))
        );
        // the citation mapping rides the birth payload whole
        assert!(loadout.rows[34].payload.contains("\"references\":[3]"));
        // the revision family: pointer only, the fold presents the swap
        assert_eq!(loadout.rows[24].kind, "comment_revised");
        assert!(
            loadout.rows[24]
                .payload
                .contains("post-fold receipt, corrected")
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(23))]
                .comment
                .kind
                .prose()
                .unwrap()
                .as_str(),
            "post-fold receipt, corrected"
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(23))].revised,
            Some(Revision {
                record: RecordId(24),
                reviser: agent().actor,
            })
        );
        assert!(loadout.rows[23].payload.contains("post-fold receipt"));
        // the withdrawal family: the note and the tombstone's pointer
        assert_eq!(loadout.rows[26].kind, "comment_withdrawn");
        assert!(
            loadout.rows[26]
                .payload
                .contains("the parking was premature")
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(25))].withdrawn,
            Some(Withdrawal {
                record: RecordId(26),
                withdrawer: human().actor,
            })
        );
        assert!(matches!(
            world.comments[&CommentId(RecordId(25))].comment.state,
            CommentState::Withdrawn(_)
        ));
        assert!(loadout.rows[25].payload.contains("parked mid-flight"));
        // the native families: machinery registration, the addressed
        // message from its scoped sender, the recipient's submission
        assert_eq!(loadout.rows[35].kind, "incarnation_registered");
        assert_eq!(loadout.rows[36].kind, "incarnation_registered");
        assert_eq!(loadout.rows[35].actor.as_str(), "saccade");
        assert_eq!(loadout.rows[37].kind, "message_sent");
        // real Unicode user data rides storage and the fold byte-exact, by escape
        assert!(
            loadout.rows[37]
                .payload
                .contains("please look at the failing edge case: caf\u{e9} floop, na\u{ef}ve fix")
        );
        assert!(
            loadout.rows[37]
                .scope
                .as_deref()
                .is_some_and(|s| s.contains("35"))
        );
        assert_eq!(
            loadout.rows[37].scope.as_deref(),
            sender
                .scope
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap())
                .as_deref()
        );
        assert_eq!(loadout.rows[38].kind, "message_submitted");
        // the registration births a native identity; the message
        // answers by reference and the submission reads submitted
        assert_eq!(
            world.incarnations[&IncarnationId(RecordId(35))].incarnation,
            crate::Incarnation::Native {
                native_id: crate::NativeSessionId::new("aaaa1111".into()).unwrap(),
                saved_session: crate::SessionLocator::new("/sessions/aaaa1111.jsonl".into())
                    .unwrap(),
            }
        );
        assert_eq!(world.messages.len(), 1);
        let mctx = &world.messages[&crate::MessageId(RecordId(37))];
        assert_eq!(mctx.state, crate::MessageState::Submitted);
        assert_eq!(mctx.message.task, Some(TaskId(0)));
        // the artifact family: pointer only
        assert_eq!(loadout.rows[17].kind, "artifact_added");
        assert!(loadout.rows[17].payload.contains("sweep figure"));
        assert!(
            loadout.rows[17]
                .payload
                .contains(ContentHash::of(b"figure bytes").as_str())
        );
        assert_eq!(
            world.tasks[0].artifacts,
            vec![(
                RecordId(17),
                Artifact {
                    name: Prose::new("sweep figure".into()).unwrap(),
                    hash: ContentHash::of(b"figure bytes"),
                }
            )]
        );
        // the world folded the full story
        assert_eq!(world.tasks.len(), 6);
        assert!(matches!(world.tasks[0].task.state, TaskState::Concluded));
        assert!(matches!(world.tasks[1].task.state, TaskState::Concluded));
        assert!(matches!(world.tasks[2].task.state, TaskState::Open));
        assert!(matches!(world.tasks[3].task.state, TaskState::Open));
        assert!(matches!(world.tasks[5].task.state, TaskState::Offered));
        assert_eq!(world.tasks[0].current_offer, Some(CommentId(RecordId(3))));
        assert_eq!(world.tasks[1].current_offer, Some(CommentId(RecordId(8))));
        assert_eq!(world.tasks[2].current_offer, None);
        assert!(matches!(
            world.tasks[0]
                .workspace
                .as_ref()
                .map(|w| &w.workspace.checkpoint),
            Some(checkpoint) if *checkpoint == GitCommit::new("def456".into()).unwrap()
        ));

        // bi-temporal: event time is caller-supplied, logged time is ours
        assert_eq!(loadout.rows[0].event_time, 10);
        assert!(loadout.rows[0].logged_time >= 10);
    }

    /// test against a real file, not just an in-memory database
    #[test]
    fn file_backed_log_reopens_with_its_world() {
        let path = std::env::temp_dir().join("saccade-reopen-test.db");
        let _ = std::fs::remove_file(&path);

        {
            let mut conn = open(&path).expect("create the file-backed log");
            record(&mut conn, &human(), create("implement foo"), 10).unwrap();
            record(&mut conn, &agent(), offer(0, "suite green"), 11).unwrap();
            record(&mut conn, &human(), accept_on(0, 1), 12).unwrap();
        } // connection dropped: WAL checkpoints back into the main file

        // reopen runs the full open path: pragmas, header-stamp verify
        let conn = open(&path).expect("reopen the same file");
        let loadout = load(&conn).unwrap();
        let LoadState::Full(world) = loadout.state else {
            panic!("expected a full load after reopen");
        };
        assert_eq!(loadout.rows.len(), 3);
        assert_eq!(world.tasks.len(), 1);
        assert!(matches!(world.tasks[0].task.state, TaskState::Concluded));

        // the read-only path sees the same file
        let ro = open_read(&path).expect("read-only open of a real file");
        assert_eq!(load(&ro).unwrap().rows.len(), 3);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn batch_failure_appends_none() {
        let mut conn = memory_db();
        record(&mut conn, &agent(), create("survivor"), 1).unwrap();

        // the second draft does not fold, so nothing may land
        let batch = AtomicBatch {
            drafts: vec![
                RecordDraft {
                    context: agent(),
                    event: Event::TaskCreated {
                        name: Prose::new("doomed sibling".into()).unwrap(),
                    },
                },
                RecordDraft {
                    context: agent(),
                    event: Event::TaskClaimed {
                        id: TaskId(99),
                        incarnation: None,
                    },
                },
            ],
        };
        let refused = execute_batch(&mut conn, batch, 2);
        assert!(matches!(
            refused,
            Err(ExecuteFail::Reject(Reject::InvalidTaskId))
        ));

        let loadout = load(&conn).unwrap();
        assert_eq!(loadout.rows.len(), 1);
        assert_eq!(loadout.rows[0].kind, "task_created");

        // seq stayed dense: the refused batch consumed no positions
        record(&mut conn, &agent(), create("next"), 3).unwrap();
        assert_eq!(load(&conn).unwrap().rows[1].seq, 1);
    }

    #[test]
    fn seq_stays_dense_across_rejections() {
        let mut conn = memory_db();
        record(&mut conn, &agent(), create("a"), 1).unwrap();

        let rejected = record(
            &mut conn,
            &agent(),
            Command::ClaimTask {
                id: TaskId(9),
                incarnation: None,
            },
            2,
        );
        assert!(matches!(
            rejected,
            Err(ExecuteFail::Reject(Reject::InvalidTaskId))
        ));

        record(&mut conn, &agent(), create("b"), 3).unwrap();
        let loadout = load(&conn).unwrap();
        assert_eq!(
            loadout.rows.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn unknown_kind_degrades_reads_and_blocks_writes() {
        let mut conn = memory_db();
        record(&mut conn, &agent(), create("a"), 1).unwrap();
        conn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (1, 1, 1, 'future binary', 'agent', 'task_zapped', '{}')",
            [],
        )
        .unwrap();

        // reads never fail: rows come back raw
        let loadout = load(&conn).unwrap();
        let LoadState::Degraded(reason) = &loadout.state else {
            panic!("expected degraded mode");
        };
        assert!(
            reason.contains("task_zapped") && reason.contains("seq 1"),
            "{reason}"
        );
        assert_eq!(loadout.rows.len(), 2);

        // writes refuse
        let refused = record(&mut conn, &agent(), create("b"), 2);
        assert!(matches!(refused, Err(ExecuteFail::Degraded(_))));
    }

    /// A schema stamp this binary does not hold refuses by name; a foreign stamp is not a saccade database.
    #[test]
    fn pre_replacement_schema_is_refused() {
        let stamped = |version: i32| {
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "CREATE TABLE events (seq INTEGER PRIMARY KEY);
                 PRAGMA application_id = 1935762276;
                 PRAGMA user_version = {version};"
            ))
            .unwrap();
            conn
        };
        assert!(matches!(
            verify(&stamped(SCHEMA_VERSION - 1)),
            Err(DbError::OlderSchema(v)) if v == SCHEMA_VERSION - 1
        ));
        assert!(matches!(
            verify(&stamped(SCHEMA_VERSION + 1)),
            Err(DbError::NewerSchema(v)) if v == SCHEMA_VERSION + 1
        ));
        let foreign = Connection::open_in_memory().unwrap();
        foreign
            .execute_batch(
                "CREATE TABLE events (seq INTEGER PRIMARY KEY);
                 PRAGMA application_id = 99;
                 PRAGMA user_version = 1;",
            )
            .unwrap();
        assert!(matches!(verify(&foreign), Err(DbError::NotSaccade)));
    }

    #[test]
    fn unknown_tier_degrades_the_same_way() {
        let conn = memory_db();
        conn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (0, 1, 1, 'future binary', 'daemon', 'task_created', '{}')",
            [],
        )
        .unwrap();
        let loadout = load(&conn).unwrap();
        let LoadState::Degraded(reason) = &loadout.state else {
            panic!("expected degraded mode");
        };
        assert!(reason.contains("daemon"), "{reason}");
    }

    #[test]
    fn malformed_payload_of_known_kind_refuses_to_load() {
        let conn = memory_db();
        conn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (0, 1, 1, 'vandal', 'human', 'task_cancelled', '{')",
            [],
        )
        .unwrap();
        assert!(matches!(load(&conn), Err(DbError::Corrupt { .. })));
    }

    #[test]
    fn unstamped_foreign_database_is_not_adopted() {
        // (0, 0) means unclaimed, not empty: a foreign table must not be
        // adopted into a saccade log.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE their_stuff (x)", []).unwrap();
        assert!(matches!(init(&conn), Err(DbError::NotSaccade)));
    }

    #[test]
    fn append_only_triggers_block_mutation() {
        let mut conn = memory_db();
        record(&mut conn, &agent(), create("a"), 1).unwrap();

        assert!(
            conn.execute("UPDATE events SET actor = 'vandal'", [])
                .is_err()
        );
        assert!(
            conn.execute("DELETE FROM events WHERE seq = 0", [])
                .is_err()
        );
        assert_eq!(load(&conn).unwrap().rows.len(), 1);
    }

    /// The cursor-clone recipe: schema and stamps first, then the
    /// prefix in order, into a file nobody else holds.
    #[test]
    fn clone_copies_the_prefix_into_a_fresh_tracker() {
        let dir = std::env::temp_dir().join(format!("sac-clone-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("saccade.db");
        {
            let mut conn = open(&source).unwrap();
            for name in ["migrate floop", "write the receipt", "accept it"] {
                record(&mut conn, &human(), create(name), 1).unwrap();
            }
        }

        let out = dir.join("prefix.db");
        let copied = clone(&source, 1, &out).unwrap();
        assert_eq!(copied, 2);
        let loadout = load(&open(&out).unwrap()).unwrap();
        assert_eq!(
            loadout.rows.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(loadout.rows[1].kind, "task_created");
        // the identity stamps ride: the clone is a tracker, not a raw copy
        assert!(open_read(&out).is_ok());

        // the prefix refolds to its own world
        let LoadState::Full(world) = loadout.state else {
            panic!("a prefix folds");
        };
        assert_eq!(world.tasks.len(), 2);

        // a fresh path only, and a cursor inside the log only
        assert!(matches!(clone(&source, 0, &out), Err(DbError::Io(_))));
        let beyond = dir.join("beyond.db");
        assert!(matches!(clone(&source, 9, &beyond), Err(DbError::Io(_))));
        assert!(!beyond.exists(), "a refused clone writes nothing");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
