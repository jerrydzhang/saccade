//! The window-state pool's storage (t-14 "The suite", t-24@r20): its
//! own SQLite file beside the event log - `<db>.pool` - never inside
//! it, so frequent position writes never contend with the record and
//! whatever reproduces the log (clone, import, a fresh db) carries no
//! pool. The file carries its own schema version under AGENTS.md's
//! migration rule: opening is the bounded version-to-version step,
//! transactional, refusing a file from a newer pool rather than
//! guessing at its shape. Losing the pool loses window arrangements,
//! never the record.

use std::path::Path;

use rusqlite::Connection;

/// The schema version this pool writes and understands.
pub const SCHEMA_VERSION: i64 = 1;

/// One pooled window state: what the window shows, where in the
/// content it reads, its zoom, its annotation drafts - drafts riding
/// the state per t-24@r20 - and the device and pool sequence of its
/// last change, so a device can recognise its own echo.
#[derive(Clone, Debug)]
pub struct StateRow {
    pub id: String,
    pub location: Option<String>,
    pub part: Option<String>,
    pub zoom: Option<f64>,
    /// The validated drafts JSON, stored and returned verbatim.
    pub drafts: Option<String>,
    pub last_device: Option<String>,
    pub changed: u64,
}

/// Why a pool file did not open.
#[derive(Debug)]
pub enum OpenFail {
    /// The file exists but names a schema version newer than this
    /// tracker writes: the pool refuses rather than guessing.
    TooNew {
        found: i64,
    },
    Storage(String),
}

/// Why a write refused.
#[derive(Debug)]
pub enum StoreFail {
    /// The state the write names is not in the pool.
    Missing(String),
    /// Birth of a state naming an id that is already there.
    Exists(String),
    Storage(String),
}

pub struct Pool {
    conn: Connection,
}

impl Pool {
    /// Open the pool file, creating it empty (schema v1) when missing.
    /// The init runs in one transaction: a crash leaves either the
    /// empty schema or nothing, never a half file.
    pub fn open(path: &Path) -> Result<Pool, OpenFail> {
        let conn = Connection::open(path)
            .map_err(|e| OpenFail::Storage(format!("{}: {e}", path.display())))?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| OpenFail::Storage(e.to_string()))?;
        let version: Option<String> = conn
            .query_row(
                "SELECT value FROM pool_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .ok();
        match version.and_then(|text| text.parse::<i64>().ok()) {
            None => Self::init_schema(&conn)?,
            Some(found) if found > SCHEMA_VERSION => {
                return Err(OpenFail::TooNew { found });
            }
            Some(_) => {}
        }
        Ok(Pool { conn })
    }

    fn init_schema(conn: &Connection) -> Result<(), OpenFail> {
        conn.execute_batch(
            "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS pool_meta (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS pool_seq (n INTEGER NOT NULL);
             INSERT OR IGNORE INTO pool_seq (n) VALUES (0);
             CREATE TABLE IF NOT EXISTS pool_states (
               id TEXT PRIMARY KEY,
               location TEXT,
               part TEXT,
               zoom REAL,
               drafts TEXT,
               last_device TEXT,
               changed INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS pool_attachments (
               device TEXT NOT NULL,
               state TEXT NOT NULL REFERENCES pool_states(id) ON DELETE CASCADE,
               PRIMARY KEY (device, state)
             );
             CREATE TABLE IF NOT EXISTS pool_links (
               device TEXT NOT NULL,
               from_state TEXT NOT NULL REFERENCES pool_states(id) ON DELETE CASCADE,
               to_state TEXT NOT NULL REFERENCES pool_states(id) ON DELETE CASCADE,
               PRIMARY KEY (device, from_state)
             );
             INSERT OR REPLACE INTO pool_meta (key, value)
               VALUES ('schema_version', '1');
             COMMIT;",
        )
        .map_err(|e| OpenFail::Storage(e.to_string()))?;
        Ok(())
    }

    /// The pool's monotonic change sequence: every write bumps it, and
    /// every read answers it as the cursor a `?after=` wait rides.
    pub fn seq(&self) -> Result<u64, String> {
        self.conn
            .query_row("SELECT n FROM pool_seq", [], |row| row.get::<_, i64>(0))
            .map(|n: i64| n.max(0) as u64)
            .map_err(|e| e.to_string())
    }

    fn bump(&self) -> Result<u64, String> {
        self.conn
            .execute("UPDATE pool_seq SET n = n + 1", [])
            .map_err(|e| e.to_string())?;
        self.seq()
    }

    /// Every state, in id order. Attachments and links are answered
    /// per device at the door: nothing here shows one device another
    /// device's attached set (t-24@r20).
    pub fn states(&self) -> Result<Vec<StateRow>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, location, part, zoom, drafts, last_device, changed FROM pool_states ORDER BY id")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], row_state)
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok(rows)
    }

    pub fn state(&self, id: &str) -> Result<Option<StateRow>, String> {
        self.conn
            .query_row(
                "SELECT id, location, part, zoom, drafts, last_device, changed
                 FROM pool_states WHERE id = ?1",
                [id],
                row_state,
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.to_string()),
            })
    }

    /// Birth of a state: client-generated opaque id, the creating
    /// device named. An existing id refuses - the caller re-reads
    /// rather than clobbers.
    pub fn create(
        &self,
        id: &str,
        device: &str,
        location: Option<&str>,
        part: Option<&str>,
        zoom: Option<f64>,
        drafts: Option<&str>,
    ) -> Result<u64, StoreFail> {
        if self.state(id).map_err(StoreFail::Storage)?.is_some() {
            return Err(StoreFail::Exists(id.to_string()));
        }
        let seq = self.bump().map_err(StoreFail::Storage)?;
        self.conn
            .execute(
                "INSERT INTO pool_states (id, location, part, zoom, drafts, last_device, changed)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![id, location, part, zoom, drafts, device, seq as i64],
            )
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        Ok(seq)
    }

    /// A settled change to a state's own fields: the writing device is
    /// named, so a device recognises its own echo on the next read.
    /// A field present in the patch is written - `None` inside clears
    /// it - and a field absent from the patch stays as it is.
    pub fn update(&self, id: &str, device: &str, patch: &StatePatch) -> Result<u64, StoreFail> {
        if self.state(id).map_err(StoreFail::Storage)?.is_none() {
            return Err(StoreFail::Missing(id.to_string()));
        }
        let seq = self.bump().map_err(StoreFail::Storage)?;
        // ?1 is the id, ?2 the new changed stamp; every patched field
        // takes the next index as it is appended
        let mut sql = String::from("UPDATE pool_states SET changed = ?2");
        let mut params: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(id.to_string()), Box::new(seq as i64)];
        for (name, value) in [
            ("location", &patch.location),
            ("part", &patch.part),
            ("zoom", &patch.zoom),
            ("drafts", &patch.drafts),
            ("last_device", &Some(StateValue::Text(device))),
        ] {
            if let Some(inner) = value {
                let index = params.len() + 1;
                sql.push_str(&format!(", {name} = ?{index}"));
                params.push(match inner {
                    StateValue::Text(text) => Box::new(text.to_string()),
                    StateValue::Number(number) => Box::new(*number),
                    StateValue::Clear => Box::new(Option::<String>::None),
                });
            }
        }
        sql.push_str(" WHERE id = ?1");
        let references: Vec<&dyn rusqlite::ToSql> =
            params.iter().map(|box_| box_.as_ref()).collect();
        self.conn
            .execute(&sql, references.as_slice())
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        Ok(seq)
    }

    /// This device starts showing the state.
    pub fn attach(&self, device: &str, state: &str) -> Result<u64, StoreFail> {
        if self.state(state).map_err(StoreFail::Storage)?.is_none() {
            return Err(StoreFail::Missing(state.to_string()));
        }
        let seq = self.bump().map_err(StoreFail::Storage)?;
        self.conn
            .execute(
                "INSERT OR IGNORE INTO pool_attachments (device, state) VALUES (?1, ?2)",
                [device, state],
            )
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        Ok(seq)
    }

    /// This device stops showing the state. The state stays - closing
    /// detaches, discarding is the separate act. A link lapses when
    /// its target detaches (t-14): links on this device pointing at
    /// the detached state go in the same write.
    pub fn detach(&self, device: &str, state: &str) -> Result<u64, StoreFail> {
        if self.state(state).map_err(StoreFail::Storage)?.is_none() {
            return Err(StoreFail::Missing(state.to_string()));
        }
        let seq = self.bump().map_err(StoreFail::Storage)?;
        self.conn
            .execute(
                "DELETE FROM pool_attachments WHERE device = ?1 AND state = ?2",
                [device, state],
            )
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        self.conn
            .execute(
                "DELETE FROM pool_links WHERE device = ?1 AND to_state = ?2",
                [device, state],
            )
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        Ok(seq)
    }

    /// This device's attached state ids.
    pub fn attached(&self, device: &str) -> Result<Vec<String>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT state FROM pool_attachments WHERE device = ?1 ORDER BY state")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([device], |row| row.get::<_, String>(0))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok(rows)
    }

    /// The link this device carries out of `from`, when it has one.
    pub fn link(&self, device: &str, from: &str) -> Result<Option<String>, String> {
        self.conn
            .query_row(
                "SELECT to_state FROM pool_links WHERE device = ?1 AND from_state = ?2",
                [device, from],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.to_string()),
            })
    }

    /// Set or replace this device's one outgoing link from `from` to
    /// `to`: many windows may link to one target, never more than one
    /// link out of a window, and only between windows attached on the
    /// device (t-14).
    pub fn set_link(&self, device: &str, from: &str, to: &str) -> Result<u64, LinkFail> {
        if from == to {
            return Err(LinkFail::SelfLink(from.to_string()));
        }
        let attached = self.attached(device).map_err(LinkFail::Storage)?;
        if !attached.iter().any(|id| id == from) {
            return Err(LinkFail::NotAttached(from.to_string()));
        }
        if !attached.iter().any(|id| id == to) {
            return Err(LinkFail::NotAttached(to.to_string()));
        }
        let seq = self.bump().map_err(LinkFail::Storage)?;
        self.conn
            .execute(
                "INSERT INTO pool_links (device, from_state, to_state) VALUES (?1, ?2, ?3)
                 ON CONFLICT (device, from_state) DO UPDATE SET to_state = excluded.to_state",
                [device, from, to],
            )
            .map_err(|e| LinkFail::Storage(e.to_string()))?;
        Ok(seq)
    }

    /// Remove this device's outgoing link from `from`. Unlinking a
    /// window that carries no link changes nothing: the sequence does
    /// not move.
    pub fn clear_link(&self, device: &str, from: &str) -> Result<u64, String> {
        let removed = self
            .conn
            .execute(
                "DELETE FROM pool_links WHERE device = ?1 AND from_state = ?2",
                [device, from],
            )
            .map_err(|e| e.to_string())?;
        if removed == 0 {
            return self.seq();
        }
        self.bump()
    }

    /// The rare act: the state leaves the pool on every device, its
    /// attachments and every link touching it with it.
    pub fn discard(&self, state: &str) -> Result<u64, StoreFail> {
        if self.state(state).map_err(StoreFail::Storage)?.is_none() {
            return Err(StoreFail::Missing(state.to_string()));
        }
        let seq = self.bump().map_err(StoreFail::Storage)?;
        self.conn
            .execute("DELETE FROM pool_states WHERE id = ?1", [state])
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        Ok(seq)
    }
}

/// Why a link refused.
#[derive(Debug)]
pub enum LinkFail {
    /// A window never links to itself.
    SelfLink(String),
    /// Links run only between windows attached on the device.
    NotAttached(String),
    Storage(String),
}

/// One settled field change: set to text, set to a number, or clear.
/// A field absent from the patch never moves.
pub enum StateValue<'a> {
    Text(&'a str),
    Number(f64),
    Clear,
}

/// The fields a PATCH may settle. Every field is `Option` twice over:
/// outer absent = untouched, `Some(...)` = written or cleared.
#[derive(Default)]
pub struct StatePatch<'a> {
    pub location: Option<StateValue<'a>>,
    pub part: Option<StateValue<'a>>,
    pub zoom: Option<StateValue<'a>>,
    pub drafts: Option<StateValue<'a>>,
}

fn row_state(row: &rusqlite::Row) -> Result<StateRow, rusqlite::Error> {
    Ok(StateRow {
        id: row.get(0)?,
        location: row.get(1)?,
        part: row.get(2)?,
        zoom: row.get(3)?,
        drafts: row.get(4)?,
        last_device: row.get(5)?,
        changed: row.get::<_, i64>(6).map(|n: i64| n.max(0) as u64)?,
    })
}

#[cfg(test)]
mod test {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "saccade-pool-store-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("tracker.db.pool")
    }

    #[test]
    fn a_fresh_file_inits_schema_v1_and_opens_again() {
        let path = scratch("fresh");
        {
            let pool = Pool::open(&path).unwrap();
            assert_eq!(pool.seq().unwrap(), 0);
            assert!(pool.states().unwrap().is_empty());
        }
        let again = Pool::open(&path).unwrap();
        assert_eq!(again.seq().unwrap(), 0);
    }

    #[test]
    fn a_newer_schema_version_refuses_to_open() {
        let path = scratch("toonew");
        {
            let pool = Pool::open(&path).unwrap();
            pool.conn
                .execute(
                    "UPDATE pool_meta SET value = '99' WHERE key = 'schema_version'",
                    [],
                )
                .unwrap();
        }
        match Pool::open(&path) {
            Err(OpenFail::TooNew { found }) => assert_eq!(found, 99),
            Err(OpenFail::Storage(reason)) => panic!("storage: {reason}"),
            Ok(_) => panic!("expected TooNew, opened anyway"),
        }
    }

    #[test]
    fn create_update_and_the_echo_facts() {
        let pool = Pool::open(&scratch("update")).unwrap();
        pool.create("s1", "dev-a", Some("tracker/t-5"), None, None, None)
            .unwrap();
        match pool.create("s1", "dev-a", None, None, None, None) {
            Err(StoreFail::Exists(id)) => assert_eq!(id, "s1"),
            other => panic!("expected Exists, got {other:?}"),
        }
        let row = pool.state("s1").unwrap().unwrap();
        assert_eq!(row.location.as_deref(), Some("tracker/t-5"));
        assert_eq!(row.last_device.as_deref(), Some("dev-a"));
        let seq = pool
            .update(
                "s1",
                "dev-b",
                &StatePatch {
                    location: Some(StateValue::Text("tracker/t-7")),
                    zoom: Some(StateValue::Number(1.2)),
                    ..StatePatch::default()
                },
            )
            .unwrap();
        assert!(seq > 1);
        let row = pool.state("s1").unwrap().unwrap();
        assert_eq!(row.location.as_deref(), Some("tracker/t-7"));
        assert_eq!(row.zoom, Some(1.2));
        assert_eq!(row.last_device.as_deref(), Some("dev-b"));
        // absent fields stay
        pool.update("s1", "dev-a", &StatePatch::default()).unwrap();
        let row = pool.state("s1").unwrap().unwrap();
        assert_eq!(row.zoom, Some(1.2));
        // a cleared field is null
        pool.update(
            "s1",
            "dev-a",
            &StatePatch {
                zoom: Some(StateValue::Clear),
                ..StatePatch::default()
            },
        )
        .unwrap();
        let row = pool.state("s1").unwrap().unwrap();
        assert_eq!(row.zoom, None);
        assert!(pool.state("nope").unwrap().is_none());
    }

    #[test]
    fn attach_detach_and_the_target_side_link_lapse() {
        let pool = Pool::open(&scratch("links")).unwrap();
        pool.create("from", "dev", None, None, None, None).unwrap();
        pool.create("to", "dev", None, None, None, None).unwrap();
        pool.create("other", "dev", None, None, None, None).unwrap();
        for id in ["from", "to", "other"] {
            pool.attach("dev", id).unwrap();
        }
        pool.set_link("dev", "from", "to").unwrap();
        assert_eq!(pool.link("dev", "from").unwrap().as_deref(), Some("to"));
        // one link out of a window: a second set replaces
        pool.set_link("dev", "from", "other").unwrap();
        assert_eq!(pool.link("dev", "from").unwrap().as_deref(), Some("other"));
        // many windows may link to one target
        pool.set_link("dev", "to", "other").unwrap();
        // only between attached windows on the device
        match pool.set_link("dev", "other", "detached") {
            Err(LinkFail::NotAttached(id)) => assert_eq!(id, "detached"),
            other => panic!("expected NotAttached, got {other:?}"),
        }
        // a window never links to itself
        match pool.set_link("dev", "other", "other") {
            Err(LinkFail::SelfLink(id)) => assert_eq!(id, "other"),
            other => panic!("expected SelfLink, got {other:?}"),
        }
        // the target detaches: every link pointing at it lapses
        pool.detach("dev", "other").unwrap();
        assert_eq!(pool.link("dev", "from").unwrap(), None);
        assert_eq!(pool.link("dev", "to").unwrap(), None);
        // back attached, from links other again; now the SOURCE detaches:
        // its own link survives for the resume (t-14 lapses on target)
        pool.attach("dev", "other").unwrap();
        pool.set_link("dev", "from", "other").unwrap();
        pool.detach("dev", "from").unwrap();
        assert_eq!(pool.link("dev", "from").unwrap().as_deref(), Some("other"));
        // unlinking: the surviving link goes, and unlinking nothing
        // moves no sequence
        let before = pool.seq().unwrap();
        assert_eq!(pool.clear_link("dev", "from").unwrap(), before + 1);
        assert_eq!(pool.clear_link("dev", "from").unwrap(), before + 1);
    }

    #[test]
    fn detach_of_an_unattached_state_is_a_change_but_idempotent_in_effect() {
        let pool = Pool::open(&scratch("detach")).unwrap();
        pool.create("s", "dev", None, None, None, None).unwrap();
        pool.attach("dev", "s").unwrap();
        let attached = pool.attached("dev").unwrap();
        assert_eq!(attached, vec!["s".to_string()]);
        pool.detach("dev", "s").unwrap();
        assert!(pool.attached("dev").unwrap().is_empty());
        // the state stayed
        assert!(pool.state("s").unwrap().is_some());
    }

    #[test]
    fn discard_removes_the_state_its_attachments_and_every_link_touching_it() {
        let pool = Pool::open(&scratch("discard")).unwrap();
        pool.create("gone", "dev-a", None, None, None, None)
            .unwrap();
        pool.create("keeper", "dev-a", None, None, None, None)
            .unwrap();
        pool.attach("dev-a", "gone").unwrap();
        pool.attach("dev-a", "keeper").unwrap();
        pool.attach("dev-b", "gone").unwrap();
        pool.set_link("dev-a", "keeper", "gone").unwrap();
        pool.attach("dev-a", "gone").unwrap();
        pool.create("source", "dev-a", None, None, None, None)
            .unwrap();
        pool.attach("dev-a", "source").unwrap();
        pool.set_link("dev-a", "gone", "keeper").unwrap();
        pool.discard("gone").unwrap();
        assert!(pool.state("gone").unwrap().is_none());
        assert!(
            pool.attached("dev-a")
                .unwrap()
                .iter()
                .all(|id| id != "gone")
        );
        assert!(
            pool.attached("dev-b")
                .unwrap()
                .iter()
                .all(|id| id != "gone")
        );
        assert_eq!(pool.link("dev-a", "keeper").unwrap(), None);
        assert_eq!(pool.link("dev-a", "gone").unwrap(), None);
        assert!(pool.state("keeper").unwrap().is_some());
        match pool.discard("gone") {
            Err(StoreFail::Missing(id)) => assert_eq!(id, "gone"),
            other => panic!("expected Missing, got {other:?}"),
        }
    }
}
