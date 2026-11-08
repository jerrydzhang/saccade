//! The cutover's bounded import: carry the workspace facts and the
//! root destination association from the pre-board log into a fresh
//! one. No tasks, no comments, no messages, no incarnations, no saved
//! sessions - those start fresh at the cutover. The old database is
//! read raw and read-only: its schema stamp predates this binary on
//! purpose, and the import is the one reader that means to.

use std::collections::HashMap;
use std::path::Path;

use rusqlite::Connection;
use serde_json::Value;

use crate::db::DbError;
use crate::db::RecordDraft;
use crate::events::Event;
use crate::store::{RecordId, Tier};
use crate::types::actor::ActorName;
use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

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

    let mut drafts: Vec<RecordDraft> = Vec::new();
    // old birth seq -> new record seq (a fresh log numbers its first
    // batch from zero), for the references stages name
    let mut workspace_births: HashMap<usize, usize> = HashMap::new();
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
            // tasks, comments, messages, incarnations, saved sessions,
            // binds, artifacts, checkpoints, use bindings, host
            // addresses: the cutover starts them fresh
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // the pre-board log the import reads: the schema-5 events table
    // with the old kinds and their payload shapes
    fn old_fixture(path: &Path) -> Connection {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE events (
                seq INTEGER PRIMARY KEY,
                event_time INTEGER NOT NULL,
                logged_time INTEGER NOT NULL,
                actor TEXT NOT NULL,
                tier TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                scope TEXT NULL
            );
            PRAGMA application_id = 1935762276;
            PRAGMA user_version = 5;",
        )
        .unwrap();
        conn
    }

    fn put(conn: &Connection, seq: i64, actor: &str, tier: &str, kind: &str, payload: &str) {
        conn.execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload)
             VALUES (?1, 1791657084 + ?1, 1791657085 + ?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![seq, actor, tier, kind, payload],
        )
        .unwrap();
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sac-import-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_cutover_import_carries_workspaces_and_the_root() {
        let dir = scratch("carry");
        let old = old_fixture(&dir.join("old.db"));
        put(
            &old,
            0,
            "codex",
            "agent",
            "workspace_created",
            r#"{"base":"6d7e0ef629c03cddfc65722b325634b83a825055","branch":"fix-message-behavior-a542d73c","worktree":"/run/saccade-import-fixture/worktrees/fix-message-behavior-a542d73c"}"#,
        );
        put(
            &old,
            1,
            "codex",
            "agent",
            "workspace_cleaned",
            r#"{"workspace":0}"#,
        );
        put(
            &old,
            2,
            "codex",
            "agent",
            "workspace_created",
            r#"{"base":"6d7e0ef629c03cddfc65722b325634b83a825055","branch":"board-lifecycle-9a01b7d2","worktree":"/run/saccade-import-fixture/worktrees/board-lifecycle-9a01b7d2"}"#,
        );
        put(
            &old,
            3,
            "ida person",
            "human",
            "root_space_bound",
            r#"{"space":"work","tab":"tracker"}"#,
        );

        let dest = dir.join("fresh.db");
        let carried = import_workspaces(&dir.join("old.db"), &dest).unwrap();
        assert_eq!(
            carried,
            Carried {
                workspaces: 2,
                workspace_stages: 1,
                root_association: 1,
                records: 4,
            }
        );

        let conn = crate::db::open_read(&dest).unwrap();
        let loadout = crate::db::load(&conn).unwrap();
        let kinds: Vec<&str> = loadout.rows.iter().map(|r| r.kind.as_str()).collect();
        assert_eq!(
            kinds,
            [
                "workspace_created",
                "workspace_cleaned",
                "workspace_created",
                "root_space_bound"
            ]
        );
        // the carried contexts keep actor and tier from the old log
        assert_eq!(loadout.rows[3].actor, "ida person");
        assert_eq!(loadout.rows[3].tier, "human");
        // the fold reads the carried facts: the cleaned stage and the
        // human's root destination
        let (records, _) = crate::db::parse_rows(&loadout.rows).unwrap();
        let world = crate::store::World::replay(records).unwrap();
        assert_eq!(
            world.workspace_stage(RecordId(0)),
            Some(crate::WorkspaceStage::Cleaned)
        );
        assert_eq!(
            world.workspace_stage(RecordId(2)),
            Some(crate::WorkspaceStage::Available)
        );
        let root = world.root_space.unwrap();
        assert_eq!(root.space, "work");
        assert_eq!(root.tab, "tracker");

        // a second import refuses: the destination holds records now
        let again = import_workspaces(&dir.join("old.db"), &dest).unwrap_err();
        assert!(
            matches!(again, ImportFail::NotFresh { records: 4 }),
            "the non-empty destination refuses: {again:?}"
        );
    }

    #[test]
    fn the_import_refuses_a_source_stamped_above_its_shapes() {
        let dir = scratch("stamp");
        let old = old_fixture(&dir.join("old.db"));
        put(
            &old,
            0,
            "codex",
            "agent",
            "workspace_created",
            r#"{"base":"6d7e0ef629c03cddfc65722b325634b83a825055","branch":"future-branch-1a2b3c4d","worktree":"/run/saccade-import-fixture/worktrees/future-branch-1a2b3c4d"}"#,
        );
        old.execute_batch("PRAGMA user_version = 6;").unwrap();

        let refused = import_workspaces(&dir.join("old.db"), &dir.join("fresh.db")).unwrap_err();
        assert!(
            matches!(refused, ImportFail::NewerSchema(6)),
            "a newer stamp refuses: {refused:?}"
        );
        assert!(!dir.join("fresh.db").exists());
    }
}
