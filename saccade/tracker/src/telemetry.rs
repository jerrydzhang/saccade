//! The command telemetry (D16): one line per CLI door, appended
//! beside the tracker's database. A line carries the verb, the flag
//! names (never values), the outcome or refusal code, the duration,
//! and the tier, for reads and writes alike. Never prose, payloads,
//! or board content.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// The telemetry file's home: beside the database it observes, so a
/// `--db`-named tracker gets its telemetry named the same way.
pub fn at(db_path: &Path) -> PathBuf {
    db_path.with_file_name("telemetry.jsonl")
}

/// One door's telemetry line.
#[derive(Serialize)]
pub struct Entry<'a> {
    pub verb: &'a str,
    pub flags: &'a [String],
    pub outcome: String,
    pub duration_ms: u64,
    pub tier: &'a str,
}

/// Append one line. The reader is the post-cutover component review;
/// the moment is that review's read of this file. A door with no
/// resolvable tracker (no --db, no repo) emits nothing: a line with no
/// destination names no reader.
pub fn emit(db_path: Option<&Path>, entry: &Entry) {
    let Some(db_path) = db_path else {
        return;
    };
    let line = serde_json::to_string(entry).expect("telemetry is plain data");
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(at(db_path))
    else {
        return;
    };
    let _ = writeln!(file, "{line}");
}
