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
const SCHEMA_VERSION: i32 = 1;

#[derive(Debug)]
pub enum DbError {
    Sqlite(rusqlite::Error),
    /// creating the tracker's parent directories failed
    Io(String),
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
            DbError::Io(e) => write!(f, "filesystem: {e}"),
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

/// A single row of the records table, and the serde shape of a record on
/// every surface: the wire, the API response, and `--json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRecord {
    pub seq: usize,
    pub event_time: u64,
    pub logged_time: u64,
    pub actor: String,
    pub tier: String,
    pub kind: String,
    #[serde(with = "payload_json")]
    pub payload: String,
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
                actor: ActorName::new(row.actor.clone()).map_err(|_| DbError::Corrupt {
                    seq: row.seq,
                    kind: row.kind.clone(),
                    detail: "invalid actor name".into(),
                })?,
                tier,
            },
            event,
        });
    }

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
/// the schema and identity stamps first, then the rows in order — the
/// cursor-cloning recipe as one verb. The cursor is the storage seq
/// read from the rows, never a record id; the two series need not
/// agree. In-place truncation is impossible (the append-only triggers),
/// so a clone is always a fresh file.
pub fn clone(source: &Path, at: usize, out: &Path) -> Result<usize, DbError> {
    if out.exists() {
        return Err(DbError::Io(format!(
            "refusing to overwrite {}; clone to a fresh path",
            out.display()
        )));
    }
    let src = open_read(source)?;
    let rows = load(&src)?.rows;
    let Some(last) = rows.last() else {
        return Err(DbError::Io(format!(
            "{} holds no records; --at {at} names nothing",
            source.display()
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
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                row.seq as i64,
                row.event_time as i64,
                row.logged_time as i64,
                row.actor,
                row.tier,
                row.kind,
                row.payload
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
        stored.push(StoredRecord {
            seq: record.id.0,
            event_time,
            logged_time,
            actor: record.context.actor.as_str().to_string(),
            tier: tier.to_string(),
            kind: kind.clone(),
            payload: payload.clone(),
        });
        txn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                record.id.0 as i64,
                event_time as i64,
                logged_time as i64,
                record.context.actor.as_str(),
                tier,
                kind,
                payload
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
    use crate::objects::comment::{AgentAttemptState, CommentState, ResponseState};
    use crate::objects::incarnation::{IncarnationId, IncarnationState};
    use crate::objects::task::{TaskId, TaskState};
    use crate::store::Tier;
    use crate::types::artifact::Artifact;
    use crate::types::pointers::{GitBranch, GitCommit, SessionPointer, WorktreePath};
    use crate::{
        CommentId, ContentHash, ProposalAction, ProposalId, ProposalState, Prose, RecordId, Target,
    };

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
        }
    }

    fn human() -> Context {
        Context {
            actor: ActorName::new("human person".into()).unwrap(),
            tier: Tier::Human,
        }
    }

    fn create(name: &str) -> Command {
        Command::CreateTask {
            name: Prose::new(name.into()).unwrap(),
            parent_id: None,
        }
    }

    #[test]
    fn execute_then_load_round_trips_the_world() {
        let mut conn = memory_db();
        record(&mut conn, &agent(), create("implement foo"), 10).unwrap();
        record(
            &mut conn,
            &human(),
            Command::ClaimTask { id: TaskId(0) },
            11,
        )
        .unwrap();
        record(
            &mut conn,
            &human(),
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Prose::new("tests green".into()).unwrap(),
            },
            12,
        )
        .unwrap();
        // the delivered deposit passes through the accept door
        record(
            &mut conn,
            &human(),
            Command::AcceptTask { id: TaskId(0) },
            12,
        )
        .unwrap();

        record(&mut conn, &agent(), create("duplicate corpse"), 13).unwrap();
        record(
            &mut conn,
            &agent(),
            Command::CreateProposal {
                name: Prose::new("duplicate of t-0".into()).unwrap(),
                action: ProposalAction::Drop { task_id: TaskId(1) },
            },
            14,
        )
        .unwrap();
        let (compound, _) = record(
            &mut conn,
            &human(),
            Command::AcceptProposal {
                id: ProposalId(RecordId(5)),
            },
            15,
        )
        .unwrap();
        assert_eq!(compound.len(), 2);

        // the thread round-trips: root on the task, reply to the record
        record(
            &mut conn,
            &agent(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("receipt verified against receipts test".into()).unwrap(),
                kind: CommentKind::Note,
            },
            30,
        )
        .unwrap();
        record(
            &mut conn,
            &human(),
            Command::Comment {
                target: Target::Comment(CommentId(RecordId(8))),
                body: Prose::new("agreed, closing".into()).unwrap(),
                kind: CommentKind::Note,
            },
            31,
        )
        .unwrap();
        // a mixed-context batch appends atomically with per-draft tiers
        let batch = AtomicBatch {
            drafts: vec![
                RecordDraft {
                    context: agent(),
                    event: Event::TaskCreated {
                        name: Prose::new("batchwork".into()).unwrap(),
                        parent_id: None,
                    },
                },
                RecordDraft {
                    context: human(),
                    event: Event::TaskClaimed { id: TaskId(2) },
                },
                RecordDraft {
                    context: human(),
                    event: Event::TaskDone {
                        id: TaskId(2),
                        receipt: Prose::new("batched receipt".into()).unwrap(),
                    },
                },
            ],
        };
        let batched = execute_batch(&mut conn, batch, 32).unwrap();
        assert_eq!(batched.len(), 3);

        // an agent demand runs its course: bind, accept, reply, produce, settle
        record(
            &mut conn,
            &human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("who folded the receipt?".into()).unwrap(),
                kind: CommentKind::Demand,
            },
            31,
        )
        .unwrap();
        let system = Context::system();
        record(
            &mut conn,
            &system,
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: CommentId(RecordId(13)),
                trigger: RecordId(13),
                actor: ActorName::new("pi".into()).unwrap(),
                session: SessionPointer::new("/tmp/pi-session.jsonl".into()).unwrap(),
            },
            31,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::AcceptPrompt {
                id: IncarnationId(RecordId(14)),
            },
            31,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::Comment {
                target: Target::Comment(CommentId(RecordId(13))),
                body: Prose::new("the fold did, at seq 9".into()).unwrap(),
                kind: CommentKind::Note,
            },
            31,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::MarkRecord {
                record_id: RecordId(16),
                incarnation_id: IncarnationId(RecordId(14)),
            },
            31,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::SettleIncarnation {
                id: IncarnationId(RecordId(14)),
            },
            31,
        )
        .unwrap();

        // the demand reopened done t-0; the fired session claims, delivers,
        // and the asker accepts again
        record(
            &mut conn,
            &agent(),
            Command::ClaimTask { id: TaskId(0) },
            32,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Prose::new("refolded the receipt".into()).unwrap(),
            },
            32,
        )
        .unwrap();
        record(
            &mut conn,
            &human(),
            Command::AcceptTask { id: TaskId(0) },
            32,
        )
        .unwrap();

        // the workspace records its lineage, worktree, and checkpoints
        record(
            &mut conn,
            &system,
            Command::CreateWorkspace {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            },
            33,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::CreateWorktree {
                task_id: TaskId(0),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            33,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::CheckpointWorkspace {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("def456".into()).unwrap(),
            },
            33,
        )
        .unwrap();

        // a second demand the machinery refuses: the fact lands on the
        // thread, the authorization spends
        record(
            &mut conn,
            &human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("one more round".into()).unwrap(),
                kind: CommentKind::Demand,
            },
            33,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::RefuseDemand {
                demand: CommentId(RecordId(25)),
                reason: Prose::new(
                    "t-0 branch saccade/t-0 diverged from the recorded checkpoint".into(),
                )
                .unwrap(),
            },
            33,
        )
        .unwrap();

        // the refused demand reopened done t-0; the re-ask cycle closes it
        record(
            &mut conn,
            &agent(),
            Command::ClaimTask { id: TaskId(0) },
            33,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Prose::new("reconciled the branch by hand".into()).unwrap(),
            },
            33,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::AcceptTask { id: TaskId(0) },
            34,
        )
        .unwrap();

        // the variant cycle rides the same round-trip: a demand fires a
        // run, a steer reaches it and is consumed, an ask is answered,
        // the run settles
        record(
            &mut conn,
            &human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("serve the re-cut".into()).unwrap(),
                kind: CommentKind::Demand,
            },
            34,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::BindIncarnation {
                task_id: TaskId(0),
                response_target: CommentId(RecordId(30)),
                trigger: RecordId(30),
                actor: ActorName::new("pi/t-0-2".into()).unwrap(),
                session: SessionPointer::new("/tmp/pi-session-2.jsonl".into()).unwrap(),
            },
            34,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::AcceptPrompt {
                id: IncarnationId(RecordId(31)),
            },
            34,
        )
        .unwrap();
        record(
            &mut conn,
            &human(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("also cover the offline path".into()).unwrap(),
                kind: CommentKind::Steer,
            },
            34,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::ForwardSteer {
                steer: CommentId(RecordId(33)),
            },
            34,
        )
        .unwrap();
        record(
            &mut conn,
            &agent(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("which offline path, break-glass or the copy?".into()).unwrap(),
                kind: CommentKind::Ask,
            },
            34,
        )
        .unwrap();
        record(
            &mut conn,
            &human(),
            Command::Comment {
                target: Target::Comment(CommentId(RecordId(35))),
                body: Prose::new("break-glass; the copy is dead".into()).unwrap(),
                kind: CommentKind::Note,
            },
            34,
        )
        .unwrap();
        record(
            &mut conn,
            &system,
            Command::SettleIncarnation {
                id: IncarnationId(RecordId(31)),
            },
            34,
        )
        .unwrap();

        // an artifact parks its pointer on the thread: name and hash,
        // the bytes never ride the columns
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
            35,
        )
        .unwrap();

        // the returned world is the one a full reload produces: the
        // revision is the last write, so its world carries the swap
        record(
            &mut conn,
            &agent(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("post-fold receipt".into()).unwrap(),
                kind: CommentKind::Note,
            },
            35,
        )
        .unwrap();

        // the revision swaps the body the fold presents; the pointer
        // names the record that did it, the birth bytes stay in their row
        let (_, returned) = record(
            &mut conn,
            &agent(),
            Command::ReviseComment {
                id: CommentId(RecordId(39)),
                body: Prose::new("post-fold receipt, corrected".into()).unwrap(),
            },
            36,
        )
        .unwrap();

        let loadout = load(&conn).unwrap();
        let LoadState::Full(world) = loadout.state else {
            panic!("expected a full load");
        };
        assert_eq!(returned, world);
        assert_eq!(loadout.rows.len(), 41);
        // the revision family rides the same columns: pointer only
        assert_eq!(loadout.rows[40].kind, "comment_revised");
        assert!(
            loadout.rows[40]
                .payload
                .contains("post-fold receipt, corrected")
        );
        // the fold presents the revised body, the birth row keeps the original
        assert_eq!(
            world.comments[&CommentId(RecordId(39))]
                .comment
                .body
                .as_str(),
            "post-fold receipt, corrected"
        );
        assert_eq!(
            world.comments[&CommentId(RecordId(39))].revised,
            Some(RecordId(40))
        );
        assert!(loadout.rows[39].payload.contains("post-fold receipt"));
        // the artifact family rides the same columns: pointer only
        assert_eq!(loadout.rows[38].kind, "artifact_added");
        assert!(loadout.rows[38].payload.contains("sweep figure"));
        assert!(
            loadout.rows[38]
                .payload
                .contains(ContentHash::of(b"figure bytes").as_str())
        );
        assert_eq!(
            world.tasks[0].artifacts,
            vec![(
                RecordId(38),
                Artifact {
                    name: Prose::new("sweep figure".into()).unwrap(),
                    hash: ContentHash::of(b"figure bytes"),
                }
            )]
        );
        assert_eq!(loadout.rows[2].kind, "task_delivered");
        assert_eq!(loadout.rows[3].kind, "task_accepted");
        assert_eq!(loadout.rows[5].kind, "proposal_created");
        assert_eq!(loadout.rows[6].kind, "proposal_accepted");
        assert_eq!(loadout.rows[7].kind, "task_dropped");
        assert_eq!(loadout.rows[8].actor.as_str(), "saccade bot");
        assert_eq!(loadout.rows[8].tier, "agent");
        assert_eq!(loadout.rows[9].actor.as_str(), "human person");
        assert_eq!(loadout.rows[9].tier, "human");
        assert_eq!(loadout.rows[12].kind, "task_done");
        assert_eq!(loadout.rows[26].kind, "demand_refused");
        assert_eq!(loadout.rows[26].actor.as_str(), "saccade");
        assert_eq!(loadout.rows[28].kind, "task_delivered");
        assert_eq!(loadout.rows[29].kind, "task_accepted");
        assert_eq!(loadout.rows[22].kind, "task_workspace_created");
        assert_eq!(loadout.rows[23].kind, "task_worktree_created");
        assert_eq!(loadout.rows[24].kind, "task_workspace_checkpointed");
        assert_eq!(loadout.rows[22].actor.as_str(), "saccade");
        // the variant cycle round-trips: the steer's consumption and the
        // ask's answer ride the same columns
        assert_eq!(loadout.rows[34].kind, "steer_forwarded");
        assert_eq!(loadout.rows[34].actor.as_str(), "saccade");
        assert!(matches!(
            world.comments[&CommentId(RecordId(33))].state,
            crate::objects::comment::CommentState::Steer {
                delivery: crate::objects::comment::SteerDelivery::Forwarded,
            }
        ));
        assert!(matches!(
            world.comments[&CommentId(RecordId(35))].state,
            crate::objects::comment::CommentState::Ask {
                response: ResponseState::Responded {
                    reply: CommentId(RecordId(36))
                },
            }
        ));
        assert_eq!(world.tasks[0].active_incarnation, None);
        assert!(matches!(
            world.tasks[0].workspace.as_ref().map(|w| &w.checkpoint),
            Some(checkpoint) if *checkpoint == GitCommit::new("def456".into()).unwrap()
        ));
        assert_eq!(world.tasks.len(), 3);
        // the variant cycle's demand reopened t-0 and its run settled
        // unanswered: open, holding a spent ask
        assert!(matches!(world.tasks[0].task.state, TaskState::Open));
        assert!(matches!(world.tasks[1].task.state, TaskState::Dropped));
        assert!(matches!(world.tasks[2].task.state, TaskState::Done(_)));
        assert_eq!(
            world.proposals[&ProposalId(RecordId(5))].proposal.state,
            ProposalState::Accepted
        );
        let reply = &world.comments[&CommentId(RecordId(9))];
        let root = &world.comments[&CommentId(RecordId(8))];
        assert_eq!(root.actor.as_str(), "saccade bot");
        assert_eq!(reply.actor.as_str(), "human person");
        let demand = &world.comments[&CommentId(RecordId(13))];
        assert_eq!(
            demand.state,
            CommentState::Demand {
                response: ResponseState::Responded {
                    reply: CommentId(RecordId(16))
                },
                attempt: AgentAttemptState::Spent,
            }
        );
        let run = &world.incarnations[&IncarnationId(RecordId(14))];
        assert_eq!(run.state, IncarnationState::Settled);
        assert_eq!(run.produced, vec![RecordId(16)]);
        assert_eq!(world.tasks[0].active_incarnation, None);
        // the refusal round-trips through the wire columns
        assert_eq!(loadout.rows[26].kind, "demand_refused");
        assert_eq!(loadout.rows[26].actor.as_str(), "saccade");
        let refused = &world.comments[&CommentId(RecordId(25))];
        assert_eq!(
            refused.state,
            CommentState::Demand {
                response: ResponseState::Awaiting,
                attempt: AgentAttemptState::Spent,
            }
        );
        let refusal = refused.refusal.as_ref().expect("the refusal fact folded");
        assert!(refusal.reason.as_str().contains("diverged"));
        assert_eq!(refusal.at, 33);

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
            record(&mut conn, &agent(), create("implement foo"), 10).unwrap();
            record(
                &mut conn,
                &human(),
                Command::ClaimTask { id: TaskId(0) },
                11,
            )
            .unwrap();
            record(
                &mut conn,
                &human(),
                Command::CompleteTask {
                    id: TaskId(0),
                    receipt: Prose::new("tests green".into()).unwrap(),
                },
                12,
            )
            .unwrap();
            record(
                &mut conn,
                &human(),
                Command::AcceptTask { id: TaskId(0) },
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
        assert_eq!(loadout.rows.len(), 4);
        assert_eq!(world.tasks.len(), 1);
        assert!(matches!(world.tasks[0].task.state, TaskState::Done(_)));

        // the read-only path sees the same file
        let ro = open_read(&path).expect("read-only open of a real file");
        assert_eq!(load(&ro).unwrap().rows.len(), 4);

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
                        parent_id: None,
                    },
                },
                RecordDraft {
                    context: agent(),
                    event: Event::TaskClaimed { id: TaskId(99) },
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

        let rejected = record(&mut conn, &agent(), Command::ClaimTask { id: TaskId(9) }, 2);
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
