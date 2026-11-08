//! The cutover's bounded import: carry the workspace facts, the saved
//! native sessions, and the root destination association from the
//! pre-board log into a fresh one. No tasks, no comments, no messages -
//! those start fresh at the cutover. The old database is read raw and
//! read-only: its schema stamp predates this binary on purpose, and the
//! import is the one reader that means to.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rusqlite::Connection;
use serde_json::Value;

use crate::IncarnationId;
use crate::db::DbError;
use crate::db::RecordDraft;
use crate::events::Event;
use crate::harness::Harness;
use crate::store::{RecordId, Tier};
use crate::types::actor::ActorName;
use crate::types::pointers::{GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath};

/// The import's own refusal: counts, references, and stamps.
#[derive(Debug)]
pub enum ImportFail {
    /// The source is not a Saccade log at all.
    NotSaccade,
    /// The source's schema is newer than the shapes this import reads.
    NewerSchema(i32),
    /// The destination already holds records: the import lands on a
    /// fresh tracker only, so the identity mapping is total.
    NotFresh {
        records: usize,
    },
    /// A carried fact names a birth the import did not carry.
    UnmappedReference {
        kind: String,
        reference: usize,
    },
    /// The fold refused a carried fact; the whole import rolled back.
    FoldRefused {
        code: &'static str,
    },
    Sqlite(rusqlite::Error),
    Db(DbError),
    Json {
        seq: usize,
        detail: String,
    },
    Actor {
        seq: usize,
        name: String,
    },
}

impl From<rusqlite::Error> for ImportFail {
    fn from(e: rusqlite::Error) -> Self {
        ImportFail::Sqlite(e)
    }
}

impl std::fmt::Display for ImportFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportFail::NotSaccade => write!(f, "the source is not a saccade database"),
            ImportFail::NewerSchema(v) => write!(
                f,
                "the source's schema v{v} is newer than this import reads; open it with its own binary"
            ),
            ImportFail::NotFresh { records } => write!(
                f,
                "the destination holds {records} records; the import lands on a fresh tracker only"
            ),
            ImportFail::UnmappedReference { kind, reference } => write!(
                f,
                "a {kind} names birth {reference}, which the import did not carry"
            ),
            ImportFail::FoldRefused { code } => write!(
                f,
                "the fold refused the carried facts ({code}); nothing landed"
            ),
            ImportFail::Sqlite(e) => write!(f, "sqlite: {e}"),
            ImportFail::Db(e) => write!(f, "database: {e}"),
            ImportFail::Json { seq, detail } => {
                write!(f, "record {seq}: the payload does not read: {detail}")
            }
            ImportFail::Actor { seq, name } => {
                write!(f, "record {seq}: '{name}' is not an actor name")
            }
        }
    }
}

/// What the import carried, for the door's reply.
#[derive(Debug, Default, PartialEq)]
pub struct Carried {
    pub workspaces: usize,
    pub workspace_stages: usize,
    pub saved_sessions: usize,
    pub session_binds: usize,
    pub root_association: usize,
    pub records: usize,
}

/// Open the pre-board log read-only and read it as it is: the stamp
/// check the ordinary readers run is deliberately absent here, with
/// one guard - this import reads the shapes up to the board cutover's
/// schema, and a newer stamp refuses.
fn open_old(path: &Path) -> Result<Connection, ImportFail> {
    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(std::time::Duration::from_millis(5000))?;
    let app_id: i32 = conn
        .query_row("PRAGMA application_id", [], |r| r.get(0))
        .map_err(ImportFail::Sqlite)?;
    const APPLICATION_ID: i32 = i32::from_be_bytes(*b"sacd");
    if app_id != APPLICATION_ID {
        return Err(ImportFail::NotSaccade);
    }
    let version: i32 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(ImportFail::Sqlite)?;
    if version > 5 {
        return Err(ImportFail::NewerSchema(version));
    }
    Ok(conn)
}

fn parse_tier(tier: &str) -> Tier {
    match tier {
        "human" => Tier::Human,
        "system" => Tier::System,
        _ => Tier::Agent,
    }
}

fn string_at(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

struct Row {
    seq: usize,
    actor: String,
    tier: String,
    kind: String,
    payload: Value,
}

fn read_rows(old: &Connection) -> Result<Vec<Row>, ImportFail> {
    let mut statement = old
        .prepare("SELECT seq, event_time, actor, tier, kind, payload FROM events ORDER BY seq")
        .map_err(ImportFail::Sqlite)?;
    let mut rows = statement.query([]).map_err(ImportFail::Sqlite)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(ImportFail::Sqlite)? {
        let seq: i64 = row.get(0).map_err(ImportFail::Sqlite)?;
        let actor: String = row.get(2).map_err(ImportFail::Sqlite)?;
        let tier: String = row.get(3).map_err(ImportFail::Sqlite)?;
        let kind: String = row.get(4).map_err(ImportFail::Sqlite)?;
        let payload_text: String = row.get(5).map_err(ImportFail::Sqlite)?;
        let payload: Value = serde_json::from_str(&payload_text).map_err(|e| ImportFail::Json {
            seq: seq.max(0) as usize,
            detail: e.to_string(),
        })?;
        out.push(Row {
            seq: seq.max(0) as usize,
            actor,
            tier,
            kind,
            payload,
        });
    }
    Ok(out)
}

/// Carry the cutover facts from `source` into `destination`: one
/// all-or-none batch in the old log's order, timestamps preserved,
/// contexts carried without their scope (the scope named dropped
/// objects). Returns what landed.
pub fn import_workspaces(source: &Path, destination: &Path) -> Result<Carried, ImportFail> {
    let old = open_old(source)?;
    let rows = read_rows(&old)?;

    // the fresh tracker is read through the ordinary door: a missing
    // file reads as the fresh empty tracker the import lands on, an
    // existing one carries no records
    if destination.exists() {
        let fresh = crate::db::open_read(destination).map_err(ImportFail::Db)?;
        let existing = crate::db::load(&fresh).map_err(ImportFail::Db)?;
        if !existing.rows.is_empty() {
            return Err(ImportFail::NotFresh {
                records: existing.rows.len(),
            });
        }
    }

    // the carried incarnations: those born with a saved session, plus
    // those a bind names - resolved before the walk, since a bind can
    // never precede its registration but the decision needs the whole
    // log
    let mut carried_incs: HashSet<usize> = HashSet::new();
    for row in &rows {
        match row.kind.as_str() {
            "incarnation_registered" => {
                if string_at(&row.payload, "saved_session").is_some() {
                    carried_incs.insert(row.seq);
                }
            }
            "saved_session_bound" => {
                if let Some(incarnation) = row.payload.get("incarnation").and_then(|v| v.as_u64()) {
                    carried_incs.insert(incarnation as usize);
                }
            }
            _ => {}
        }
    }

    let mut drafts: Vec<RecordDraft> = Vec::new();
    // old birth seq -> new record seq (a fresh log numbers its first
    // batch from zero), for the references stages and binds name
    let mut workspace_births: HashMap<usize, usize> = HashMap::new();
    let mut incarnation_births: HashMap<usize, usize> = HashMap::new();
    let mut carried = Carried::default();

    for row in &rows {
        let context = crate::store::Context {
            actor: ActorName::new(row.actor.clone()).map_err(|_| ImportFail::Actor {
                seq: row.seq,
                name: row.actor.clone(),
            })?,
            tier: parse_tier(&row.tier),
            scope: None,
        };
        match row.kind.as_str() {
            // both workspace births carry identity (branch, path, base);
            // the task-bound establishment drops its task reference -
            // the fresh log carries no tasks
            "workspace_created" | "workspace_established" => {
                let base = string_at(&row.payload, "base").ok_or(ImportFail::Json {
                    seq: row.seq,
                    detail: "no base".into(),
                })?;
                let branch = string_at(&row.payload, "branch").ok_or(ImportFail::Json {
                    seq: row.seq,
                    detail: "no branch".into(),
                })?;
                let worktree = string_at(&row.payload, "worktree").ok_or(ImportFail::Json {
                    seq: row.seq,
                    detail: "no worktree".into(),
                })?;
                let new_position = drafts.len();
                drafts.push(RecordDraft {
                    context,
                    event: Event::WorkspaceCreated {
                        base: GitCommit::new(base).map_err(|_| ImportFail::Json {
                            seq: row.seq,
                            detail: "bad base commit".into(),
                        })?,
                        branch: GitBranch::new(branch).map_err(|_| ImportFail::Json {
                            seq: row.seq,
                            detail: "bad branch".into(),
                        })?,
                        worktree: WorktreePath::new(worktree.into()).map_err(|_| {
                            ImportFail::Json {
                                seq: row.seq,
                                detail: "bad worktree path".into(),
                            }
                        })?,
                    },
                });
                workspace_births.insert(row.seq, new_position);
                carried.workspaces += 1;
            }
            "workspace_cleaned" | "workspace_restored" => {
                let reference = row
                    .payload
                    .get("workspace")
                    .and_then(|v| v.as_u64())
                    .ok_or(ImportFail::Json {
                        seq: row.seq,
                        detail: "no workspace reference".into(),
                    })? as usize;
                let mapped = workspace_births.get(&reference).copied().ok_or(
                    ImportFail::UnmappedReference {
                        kind: row.kind.clone(),
                        reference,
                    },
                )?;
                drafts.push(RecordDraft {
                    context,
                    event: if row.kind == "workspace_cleaned" {
                        Event::WorkspaceCleaned {
                            workspace: RecordId(mapped),
                        }
                    } else {
                        Event::WorkspaceRestored {
                            workspace: RecordId(mapped),
                        }
                    },
                });
                carried.workspace_stages += 1;
            }
            "incarnation_registered" if carried_incs.contains(&row.seq) => {
                let native_id = string_at(&row.payload, "native_id").ok_or(ImportFail::Json {
                    seq: row.seq,
                    detail: "no native id".into(),
                })?;
                let saved_session = string_at(&row.payload, "saved_session");
                let adopted = row
                    .payload
                    .get("adopted")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let harness = row
                    .payload
                    .get("harness")
                    .and_then(|v| v.as_str())
                    .map(|name| serde_json::from_str(&format!("\"{name}\"")).ok())
                    .unwrap_or(Some(Harness::Pi))
                    .unwrap_or(Harness::Pi);
                let new_position = drafts.len();
                drafts.push(RecordDraft {
                    context,
                    event: Event::IncarnationRegistered {
                        native_id: NativeSessionId::new(native_id).map_err(|_| {
                            ImportFail::Json {
                                seq: row.seq,
                                detail: "bad native id".into(),
                            }
                        })?,
                        saved_session: saved_session
                            .clone()
                            .map(|path| SessionLocator::new(path.into()))
                            .transpose()
                            .map_err(|_| ImportFail::Json {
                                seq: row.seq,
                                detail: "bad saved session".into(),
                            })?,
                        address: None,
                        adopted,
                        harness,
                    },
                });
                incarnation_births.insert(row.seq, new_position);
                if saved_session.is_some() {
                    carried.saved_sessions += 1;
                }
            }
            "saved_session_bound" => {
                let incarnation = row
                    .payload
                    .get("incarnation")
                    .and_then(|v| v.as_u64())
                    .ok_or(ImportFail::Json {
                        seq: row.seq,
                        detail: "no incarnation".into(),
                    })? as usize;
                let mapped = incarnation_births.get(&incarnation).copied().ok_or(
                    ImportFail::UnmappedReference {
                        kind: row.kind.clone(),
                        reference: incarnation,
                    },
                )?;
                let saved_session =
                    string_at(&row.payload, "saved_session").ok_or(ImportFail::Json {
                        seq: row.seq,
                        detail: "no saved session".into(),
                    })?;
                drafts.push(RecordDraft {
                    context,
                    event: Event::SavedSessionBound {
                        incarnation: IncarnationId(RecordId(mapped)),
                        saved_session: SessionLocator::new(saved_session.into()).map_err(|_| {
                            ImportFail::Json {
                                seq: row.seq,
                                detail: "bad saved session".into(),
                            }
                        })?,
                    },
                });
                carried.session_binds += 1;
            }
            "root_space_bound" => {
                let space = string_at(&row.payload, "space").ok_or(ImportFail::Json {
                    seq: row.seq,
                    detail: "no space".into(),
                })?;
                let tab = string_at(&row.payload, "tab").ok_or(ImportFail::Json {
                    seq: row.seq,
                    detail: "no tab".into(),
                })?;
                drafts.push(RecordDraft {
                    context,
                    event: Event::RootSpaceBound { space, tab },
                });
                carried.root_association += 1;
            }
            "root_space_cleared" => {
                drafts.push(RecordDraft {
                    context,
                    event: Event::RootSpaceCleared {},
                });
            }
            // tasks, comments, messages, artifacts, checkpoints, use
            // bindings, host addresses: the cutover starts them fresh
            _ => {}
        }
    }

    carried.records = drafts.len();
    if drafts.is_empty() {
        return Ok(carried);
    }

    // land the batch through the ordinary write door: the fold judges
    // every event, and one refusal rolls the whole import back
    let mut conn = crate::db::open(destination).map_err(ImportFail::Db)?;
    crate::db::execute_batch(
        &mut conn,
        crate::db::AtomicBatch { drafts },
        crate::db::now_epoch(),
    )
    .map_err(|e| match e {
        crate::db::ExecuteFail::Reject(reject) => ImportFail::FoldRefused {
            code: reject.code(),
        },
        crate::db::ExecuteFail::Degraded(reason) => ImportFail::Json {
            seq: 0,
            detail: reason,
        },
        crate::db::ExecuteFail::Db(e) => ImportFail::Db(e),
    })?;
    Ok(carried)
}
