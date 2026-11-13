//! The window store's storage (t-14 "The suite", t-24@r20): its
//! own SQLite file beside the event log - `<db>.windows` - never inside
//! it, so frequent position writes never contend with the record and
//! whatever reproduces the log (clone, import, a fresh db) carries no
//! window store. The file carries its own schema version under AGENTS.md's
//! migration rule: opening is the bounded version-to-version step,
//! transactional, refusing a file from a newer window store rather than
//! guessing at its shape. Losing the window store loses window arrangements,
//! never the record.

use std::path::Path;

use rusqlite::Connection;

/// The schema version this window store writes and understands.
pub const SCHEMA_VERSION: i64 = 1;

/// One stored window state: what the window shows, where in the
/// content it reads, its zoom, its annotation drafts - drafts riding
/// the state per t-24@r20 - and the device and window store sequence of its
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

/// Why a window store file did not open.
#[derive(Debug)]
pub enum OpenFail {
    /// The file exists but names a schema version newer than this
    /// tracker writes: the window store refuses rather than guessing.
    TooNew {
        found: i64,
    },
    Storage(String),
}

/// Why a write refused.
#[derive(Debug)]
pub enum StoreFail {
    /// The state the write names is not in the window store.
    Missing(String),
    /// Birth of a state naming an id that is already there.
    Exists(String),
    Storage(String),
}

pub struct WindowStore {
    conn: Connection,
}

impl WindowStore {
    /// Open the window store file, creating it empty (schema v1) when missing.
    /// The init runs in one transaction: a crash leaves either the
    /// empty schema or nothing, never a half file.
    pub fn open(path: &Path) -> Result<WindowStore, OpenFail> {
        let conn = Connection::open(path)
            .map_err(|e| OpenFail::Storage(format!("{}: {e}", path.display())))?;
        // the record db's own durability settings (record/db.rs): WAL with
        // synchronous=NORMAL commits without the rollback journal's
        // fsyncs per write, and a busy wait instead of an immediate
        // refusal; the default journal made every window act pay several
        // fsyncs (t-36 m-556)
        let storage = |e: rusqlite::Error| OpenFail::Storage(e.to_string());
        conn.busy_timeout(std::time::Duration::from_millis(5000))
            .map_err(storage)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(storage)?;
        // journal_mode answers the resulting mode as a row: read, discard
        conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))
            .map_err(storage)?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(storage)?;
        let version: Option<String> = conn
            .query_row(
                "SELECT value FROM windows_meta WHERE key = 'schema_version'",
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
        Ok(WindowStore { conn })
    }

    fn init_schema(conn: &Connection) -> Result<(), OpenFail> {
        conn.execute_batch(
            "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS windows_meta (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS windows_seq (n INTEGER NOT NULL);
             INSERT OR IGNORE INTO windows_seq (n) VALUES (0);
             CREATE TABLE IF NOT EXISTS window_states (
               id TEXT PRIMARY KEY,
               location TEXT,
               part TEXT,
               zoom REAL,
               drafts TEXT,
               last_device TEXT,
               changed INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS window_attachments (
               device TEXT NOT NULL,
               state TEXT NOT NULL REFERENCES window_states(id) ON DELETE CASCADE,
               PRIMARY KEY (device, state)
             );
             CREATE TABLE IF NOT EXISTS window_links (
               device TEXT NOT NULL,
               from_state TEXT NOT NULL REFERENCES window_states(id) ON DELETE CASCADE,
               to_state TEXT NOT NULL REFERENCES window_states(id) ON DELETE CASCADE,
               PRIMARY KEY (device, from_state)
             );
             INSERT OR REPLACE INTO windows_meta (key, value)
               VALUES ('schema_version', '1');
             COMMIT;",
        )
        .map_err(|e| OpenFail::Storage(e.to_string()))?;
        Ok(())
    }

    /// The window store's monotonic change sequence: every write bumps it, and
    /// every read answers it as the cursor a `?after=` wait rides.
    pub fn seq(&self) -> Result<u64, String> {
        self.conn
            .query_row("SELECT n FROM windows_seq", [], |row| row.get::<_, i64>(0))
            .map(|n: i64| n.max(0) as u64)
            .map_err(|e| e.to_string())
    }

    fn bump(&self) -> Result<u64, String> {
        self.conn
            .execute("UPDATE windows_seq SET n = n + 1", [])
            .map_err(|e| e.to_string())?;
        self.seq()
    }

    /// Every state, in id order. Attachments and links are answered
    /// per device at the door: nothing here shows one device another
    /// device's attached set (t-24@r20).
    pub fn states(&self) -> Result<Vec<StateRow>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, location, part, zoom, drafts, last_device, changed FROM window_states ORDER BY id")
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
                 FROM window_states WHERE id = ?1",
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
                "INSERT INTO window_states (id, location, part, zoom, drafts, last_device, changed)
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
        let mut sql = String::from("UPDATE window_states SET changed = ?2");
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
                "INSERT OR IGNORE INTO window_attachments (device, state) VALUES (?1, ?2)",
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
                "DELETE FROM window_attachments WHERE device = ?1 AND state = ?2",
                [device, state],
            )
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        self.conn
            .execute(
                "DELETE FROM window_links WHERE device = ?1 AND to_state = ?2",
                [device, state],
            )
            .map_err(|e| StoreFail::Storage(e.to_string()))?;
        Ok(seq)
    }

    /// This device's attached state ids.
    pub fn attached(&self, device: &str) -> Result<Vec<String>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT state FROM window_attachments WHERE device = ?1 ORDER BY state")
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
                "SELECT to_state FROM window_links WHERE device = ?1 AND from_state = ?2",
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
                "INSERT INTO window_links (device, from_state, to_state) VALUES (?1, ?2, ?3)
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
                "DELETE FROM window_links WHERE device = ?1 AND from_state = ?2",
                [device, from],
            )
            .map_err(|e| e.to_string())?;
        if removed == 0 {
            return self.seq();
        }
        self.bump()
    }

    /// The rare act: the state leaves the window store on every device, its
    /// attachments and every link touching it with it.
    pub fn discard(&self, state: &str) -> Result<u64, StoreFail> {
        if self.state(state).map_err(StoreFail::Storage)?.is_none() {
            return Err(StoreFail::Missing(state.to_string()));
        }
        let seq = self.bump().map_err(StoreFail::Storage)?;
        self.conn
            .execute("DELETE FROM window_states WHERE id = ?1", [state])
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
            "saccade-windows-store-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("tracker.db.windows")
    }

    #[test]
    fn the_store_opens_with_the_record_dbs_durability_settings() {
        // WAL with synchronous=NORMAL and a busy timeout, as the record db
        // opens (record/db.rs): the default rollback journal with full
        // sync costs several fsyncs per write, which on a contended disk
        // made every window act wait (t-36 m-556: the browser flake)
        let path = scratch("durability");
        let store = WindowStore::open(&path).unwrap();
        let mode: String = store
            .conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
        let synchronous: i64 = store
            .conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .unwrap();
        assert_eq!(synchronous, 1, "synchronous=NORMAL");
        let timeout: i64 = store
            .conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    #[test]
    fn a_fresh_file_inits_schema_v1_and_opens_again() {
        let path = scratch("fresh");
        {
            let store = WindowStore::open(&path).unwrap();
            assert_eq!(store.seq().unwrap(), 0);
            assert!(store.states().unwrap().is_empty());
        }
        let again = WindowStore::open(&path).unwrap();
        assert_eq!(again.seq().unwrap(), 0);
    }

    #[test]
    fn a_newer_schema_version_refuses_to_open() {
        let path = scratch("toonew");
        {
            let store = WindowStore::open(&path).unwrap();
            store
                .conn
                .execute(
                    "UPDATE windows_meta SET value = '99' WHERE key = 'schema_version'",
                    [],
                )
                .unwrap();
        }
        match WindowStore::open(&path) {
            Err(OpenFail::TooNew { found }) => assert_eq!(found, 99),
            Err(OpenFail::Storage(reason)) => panic!("storage: {reason}"),
            Ok(_) => panic!("expected TooNew, opened anyway"),
        }
    }

    #[test]
    fn create_update_and_the_echo_facts() {
        let store = WindowStore::open(&scratch("update")).unwrap();
        store
            .create("s1", "dev-a", Some("tracker/t-5"), None, None, None)
            .unwrap();
        match store.create("s1", "dev-a", None, None, None, None) {
            Err(StoreFail::Exists(id)) => assert_eq!(id, "s1"),
            other => panic!("expected Exists, got {other:?}"),
        }
        let row = store.state("s1").unwrap().unwrap();
        assert_eq!(row.location.as_deref(), Some("tracker/t-5"));
        assert_eq!(row.last_device.as_deref(), Some("dev-a"));
        let seq = store
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
        let row = store.state("s1").unwrap().unwrap();
        assert_eq!(row.location.as_deref(), Some("tracker/t-7"));
        assert_eq!(row.zoom, Some(1.2));
        assert_eq!(row.last_device.as_deref(), Some("dev-b"));
        // absent fields stay
        store.update("s1", "dev-a", &StatePatch::default()).unwrap();
        let row = store.state("s1").unwrap().unwrap();
        assert_eq!(row.zoom, Some(1.2));
        // a cleared field is null
        store
            .update(
                "s1",
                "dev-a",
                &StatePatch {
                    zoom: Some(StateValue::Clear),
                    ..StatePatch::default()
                },
            )
            .unwrap();
        let row = store.state("s1").unwrap().unwrap();
        assert_eq!(row.zoom, None);
        assert!(store.state("nope").unwrap().is_none());
    }

    #[test]
    fn attach_detach_and_the_target_side_link_lapse() {
        let store = WindowStore::open(&scratch("links")).unwrap();
        store.create("from", "dev", None, None, None, None).unwrap();
        store.create("to", "dev", None, None, None, None).unwrap();
        store
            .create("other", "dev", None, None, None, None)
            .unwrap();
        for id in ["from", "to", "other"] {
            store.attach("dev", id).unwrap();
        }
        store.set_link("dev", "from", "to").unwrap();
        assert_eq!(store.link("dev", "from").unwrap().as_deref(), Some("to"));
        // one link out of a window: a second set replaces
        store.set_link("dev", "from", "other").unwrap();
        assert_eq!(store.link("dev", "from").unwrap().as_deref(), Some("other"));
        // many windows may link to one target
        store.set_link("dev", "to", "other").unwrap();
        // only between attached windows on the device
        match store.set_link("dev", "other", "detached") {
            Err(LinkFail::NotAttached(id)) => assert_eq!(id, "detached"),
            other => panic!("expected NotAttached, got {other:?}"),
        }
        // a window never links to itself
        match store.set_link("dev", "other", "other") {
            Err(LinkFail::SelfLink(id)) => assert_eq!(id, "other"),
            other => panic!("expected SelfLink, got {other:?}"),
        }
        // the target detaches: every link pointing at it lapses
        store.detach("dev", "other").unwrap();
        assert_eq!(store.link("dev", "from").unwrap(), None);
        assert_eq!(store.link("dev", "to").unwrap(), None);
        // back attached, from links other again; now the SOURCE detaches:
        // its own link survives for the resume (t-14 lapses on target)
        store.attach("dev", "other").unwrap();
        store.set_link("dev", "from", "other").unwrap();
        store.detach("dev", "from").unwrap();
        assert_eq!(store.link("dev", "from").unwrap().as_deref(), Some("other"));
        // unlinking: the surviving link goes, and unlinking nothing
        // moves no sequence
        let before = store.seq().unwrap();
        assert_eq!(store.clear_link("dev", "from").unwrap(), before + 1);
        assert_eq!(store.clear_link("dev", "from").unwrap(), before + 1);
    }

    #[test]
    fn detach_of_an_unattached_state_is_a_change_but_idempotent_in_effect() {
        let store = WindowStore::open(&scratch("detach")).unwrap();
        store.create("s", "dev", None, None, None, None).unwrap();
        store.attach("dev", "s").unwrap();
        let attached = store.attached("dev").unwrap();
        assert_eq!(attached, vec!["s".to_string()]);
        store.detach("dev", "s").unwrap();
        assert!(store.attached("dev").unwrap().is_empty());
        // the state stayed
        assert!(store.state("s").unwrap().is_some());
    }

    #[test]
    fn discard_removes_the_state_its_attachments_and_every_link_touching_it() {
        let store = WindowStore::open(&scratch("discard")).unwrap();
        store
            .create("gone", "dev-a", None, None, None, None)
            .unwrap();
        store
            .create("keeper", "dev-a", None, None, None, None)
            .unwrap();
        store.attach("dev-a", "gone").unwrap();
        store.attach("dev-a", "keeper").unwrap();
        store.attach("dev-b", "gone").unwrap();
        store.set_link("dev-a", "keeper", "gone").unwrap();
        store.attach("dev-a", "gone").unwrap();
        store
            .create("source", "dev-a", None, None, None, None)
            .unwrap();
        store.attach("dev-a", "source").unwrap();
        store.set_link("dev-a", "gone", "keeper").unwrap();
        store.discard("gone").unwrap();
        assert!(store.state("gone").unwrap().is_none());
        assert!(
            store
                .attached("dev-a")
                .unwrap()
                .iter()
                .all(|id| id != "gone")
        );
        assert!(
            store
                .attached("dev-b")
                .unwrap()
                .iter()
                .all(|id| id != "gone")
        );
        assert_eq!(store.link("dev-a", "keeper").unwrap(), None);
        assert_eq!(store.link("dev-a", "gone").unwrap(), None);
        assert!(store.state("keeper").unwrap().is_some());
        match store.discard("gone") {
            Err(StoreFail::Missing(id)) => assert_eq!(id, "gone"),
            other => panic!("expected Missing, got {other:?}"),
        }
    }
}
