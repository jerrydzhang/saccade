use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, TransactionBehavior, params};

use crate::Reject;
use crate::decide::decide;
use crate::events::Command;
use crate::store::{Context, Record, RecordId, World};
use crate::wire;

/// file-header stamp identifying a Saccade database.
const APPLICATION_ID: i32 = i32::from_be_bytes(*b"sacd");
const SCHEMA_VERSION: i32 = 1;

#[derive(Debug)]
pub enum DbError {
    Sqlite(rusqlite::Error),
    /// application_id mismatch: not a Saccade database.
    NotSaccade,
    /// user_version ahead of this binary: written by a newer Saccade.
    NewerSchema(i32),
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
            DbError::NotSaccade => write!(f, "file is not a saccade database"),
            DbError::NewerSchema(v) => {
                write!(f, "database schema v{v} is newer than this binary; upgrade")
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

/// A single row of the records table
#[derive(Debug, Clone)]
pub struct StoredRecord {
    pub seq: usize,
    pub event_time: u64,
    pub logged_time: u64,
    pub actor: String,
    pub tier: String,
    pub kind: String,
    pub payload: String,
}

pub enum LoadState {
    /// Every row understood: the world folds.
    Full(World),
    /// Version skew: rows retained raw; world projection and writes refused.
    Degraded(String),
}

pub struct Loadout {
    /// Raw rows in seq order
    pub rows: Vec<StoredRecord>,
    pub state: LoadState,
}

#[derive(Debug)]
pub enum ExecuteFail {
    Db(DbError),
    Degraded(String),
    Reject(Reject),
}

impl From<DbError> for ExecuteFail {
    fn from(e: DbError) -> Self {
        ExecuteFail::Db(e)
    }
}

impl From<Reject> for ExecuteFail {
    fn from(r: Reject) -> Self {
        ExecuteFail::Reject(r)
    }
}

impl From<rusqlite::Error> for ExecuteFail {
    fn from(e: rusqlite::Error) -> Self {
        ExecuteFail::Db(e.into())
    }
}

pub fn open(path: &Path) -> Result<Connection, DbError> {
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
             payload     TEXT    NOT NULL
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
    let mut stmt = conn.prepare(
        "SELECT seq, event_time, logged_time, actor, tier, kind, payload FROM events ORDER BY seq",
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
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    // Unknown kind/tier: stop at the first, keep rows, refuse the fold.
    // Malformed payload of a known kind: loud error, refuse everything.
    let mut degraded: Option<String> = None;
    let mut records = Vec::with_capacity(rows.len());
    for row in &rows {
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
                actor: row.actor.clone(),
                tier,
            },
            event,
        });
    }

    let state = match degraded {
        Some(reason) => LoadState::Degraded(reason),
        None => LoadState::Full(World::replay(records)),
    };
    Ok(Loadout { rows, state })
}

pub fn execute(
    conn: &mut Connection,
    context: &Context,
    command: Command,
    event_time: u64,
) -> Result<Vec<StoredRecord>, ExecuteFail> {
    let txn = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let loadout = load(&txn)?;
    let world = match loadout.state {
        LoadState::Full(world) => world,
        LoadState::Degraded(reason) => return Err(ExecuteFail::Degraded(reason)),
    };

    let events = decide(&world, context, command)?;
    let logged_time = now_epoch();
    let tier = wire::tier_of(&context.tier);

    let mut stored = Vec::with_capacity(events.len());
    for (i, event) in events.into_iter().enumerate() {
        let seq = loadout.rows.len() + i;
        let (kind, payload) = wire::disassemble(&event);
        stored.push(StoredRecord {
            seq,
            event_time,
            logged_time,
            actor: context.actor.clone(),
            tier: tier.to_string(),
            kind: kind.to_string(),
            payload: payload.clone(),
        });
        txn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                seq as i64,
                event_time as i64,
                logged_time as i64,
                context.actor,
                tier,
                kind,
                payload
            ],
        )?;
    }
    txn.commit()?;
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
    use crate::events::Command;
    use crate::objects::task::{Receipt, TaskId, TaskState};
    use crate::store::Tier;
    use crate::{ProposalAction, ProposalId, ProposalState, RecordId};

    fn memory_db() -> Connection {
        let conn = Connection::open_in_memory().expect("open memory db");
        conn.busy_timeout(Duration::from_millis(5000)).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        init(&conn).expect("init schema");
        conn
    }

    fn agent() -> Context {
        Context {
            actor: "saccade bot".into(),
            tier: Tier::Agent,
        }
    }

    fn human() -> Context {
        Context {
            actor: "human person".into(),
            tier: Tier::Human,
        }
    }

    fn create(name: &str) -> Command {
        Command::CreateTask {
            task_name: name.into(),
            parent_id: None,
        }
    }

    #[test]
    fn execute_then_load_round_trips_the_world() {
        let mut conn = memory_db();
        execute(&mut conn, &agent(), create("implement foo"), 10).unwrap();
        execute(
            &mut conn,
            &agent(),
            Command::ClaimTask { id: TaskId(0) },
            11,
        )
        .unwrap();
        execute(
            &mut conn,
            &human(),
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Receipt("tests green".into()),
            },
            12,
        )
        .unwrap();

        execute(&mut conn, &agent(), create("duplicate corpse"), 13).unwrap();
        execute(
            &mut conn,
            &agent(),
            Command::CreateProposal {
                name: "duplicate of t-0".into(),
                action: ProposalAction::Drop { task_id: TaskId(1) },
            },
            14,
        )
        .unwrap();
        let compound = execute(
            &mut conn,
            &human(),
            Command::AcceptProposal {
                id: ProposalId(RecordId(4)),
            },
            15,
        )
        .unwrap();
        assert_eq!(compound.len(), 2);

        let loadout = load(&conn).unwrap();
        let LoadState::Full(world) = loadout.state else {
            panic!("expected a full load");
        };
        assert_eq!(loadout.rows.len(), 7);
        assert_eq!(loadout.rows[4].kind, "proposal_created");
        assert_eq!(loadout.rows[5].kind, "proposal_accepted");
        assert_eq!(loadout.rows[6].kind, "task_dropped");
        assert_eq!(world.tasks.len(), 2);
        assert!(matches!(world.tasks[0].state, TaskState::Done(_)));
        assert!(matches!(world.tasks[1].state, TaskState::Dropped));
        assert_eq!(
            world.proposals[&ProposalId(RecordId(4))].state,
            ProposalState::Accepted
        );

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
            execute(&mut conn, &agent(), create("implement foo"), 10).unwrap();
            execute(
                &mut conn,
                &agent(),
                Command::ClaimTask { id: TaskId(0) },
                11,
            )
            .unwrap();
            execute(
                &mut conn,
                &human(),
                Command::CompleteTask {
                    id: TaskId(0),
                    receipt: Receipt("tests green".into()),
                },
                12,
            )
            .unwrap();
        } // connection dropped: WAL checkpoints back into the main file

        // reopen runs the full open path: pragmas, header-stamp verify
        let conn = open(&path).expect("reopen the same file");
        let loadout = load(&conn).unwrap();
        let LoadState::Full(world) = loadout.state else {
            panic!("expected a full load after reopen");
        };
        assert_eq!(loadout.rows.len(), 3);
        assert_eq!(world.tasks.len(), 1);
        assert!(matches!(world.tasks[0].state, TaskState::Done(_)));

        // the read-only path sees the same file
        let ro = open_read(&path).expect("read-only open of a real file");
        assert_eq!(load(&ro).unwrap().rows.len(), 3);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn seq_stays_dense_across_rejections() {
        let mut conn = memory_db();
        execute(&mut conn, &agent(), create("a"), 1).unwrap();

        let rejected = execute(&mut conn, &agent(), Command::ClaimTask { id: TaskId(9) }, 2);
        assert!(matches!(
            rejected,
            Err(ExecuteFail::Reject(Reject::InvalidTaskId))
        ));

        execute(&mut conn, &agent(), create("b"), 3).unwrap();
        let loadout = load(&conn).unwrap();
        assert_eq!(
            loadout.rows.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn unknown_kind_degrades_reads_and_blocks_writes() {
        let mut conn = memory_db();
        execute(&mut conn, &agent(), create("a"), 1).unwrap();
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
        let refused = execute(&mut conn, &agent(), create("b"), 2);
        assert!(matches!(refused, Err(ExecuteFail::Degraded(_))));
    }

    #[test]
    fn unknown_tier_degrades_the_same_way() {
        let conn = memory_db();
        conn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (0, 1, 1, 'future binary', 'system', 'task_created', '{}')",
            [],
        )
        .unwrap();
        let loadout = load(&conn).unwrap();
        let LoadState::Degraded(reason) = &loadout.state else {
            panic!("expected degraded mode");
        };
        assert!(reason.contains("system"), "{reason}");
    }

    #[test]
    fn malformed_payload_of_known_kind_refuses_to_load() {
        let conn = memory_db();
        conn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (0, 1, 1, 'vandal', 'human', 'task_done', '{')",
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
        execute(&mut conn, &agent(), create("a"), 1).unwrap();

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
}
