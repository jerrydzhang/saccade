//! The offline migration: one schema-3 log rewritten into schema 4.
//! The source is never touched; seq and timestamps carry over
//! unchanged; typed stored reference values stay birth references.
//! Prose is renumbered onto the fold's public numbering — comment
//! births to their dense c-N, non-comment records to `seq N`, ranges
//! to `seq N–M` — and every token that cannot be identified stays
//! visibly legacy, reported for human review rather than reassigned.

use std::collections::BTreeMap;
use std::path::Path;

use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};

use crate::db::{self, StoredRecord};

/// The schema this migration reads, verified before anything else.
const FROM_VERSION: i32 = db::MIGRATE_FROM_VERSION;

/// The v3 log's event vocabulary, exactly as Saccade 0.13.0 wrote it:
/// read here, never folded — the shapes live only as long as the
/// migration needs them.
mod v3 {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Target {
        Task(usize),
        Comment(usize),
    }
}

#[derive(Debug)]
pub enum MigrateFail {
    Db(db::DbError),
    /// the source does not hold the schema this migration reads
    WrongSchema {
        found: i32,
    },
    /// a source row's payload does not parse as its kind claims
    Corrupt {
        seq: usize,
        detail: String,
    },
    /// a rewritten record does not fold under the new law — the
    /// output is never written
    Unfoldable {
        seq: usize,
        detail: String,
    },
    /// the output path already holds a tracker
    OutputExists(std::path::PathBuf),
    Io(String),
}

impl From<db::DbError> for MigrateFail {
    fn from(e: db::DbError) -> Self {
        MigrateFail::Db(e)
    }
}

impl From<rusqlite::Error> for MigrateFail {
    fn from(e: rusqlite::Error) -> Self {
        MigrateFail::Db(e.into())
    }
}

impl std::fmt::Display for MigrateFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrateFail::Db(e) => write!(f, "storage: {e}"),
            MigrateFail::WrongSchema { found } => write!(
                f,
                "the source holds schema v{found}; this migration reads v{FROM_VERSION}"
            ),
            MigrateFail::Corrupt { seq, detail } => {
                write!(f, "corrupt record at seq {seq}: {detail}")
            }
            MigrateFail::Unfoldable { seq, detail } => write!(
                f,
                "the rewritten record at seq {seq} does not fold: {detail}; nothing was written"
            ),
            MigrateFail::OutputExists(path) => write!(
                f,
                "refusing to overwrite {}; migrate into a fresh path",
                path.display()
            ),
            MigrateFail::Io(detail) => write!(f, "filesystem: {detail}"),
        }
    }
}

/// What the migration did, as its report reads.
#[derive(Default, Debug, PartialEq)]
pub struct Report {
    pub records: usize,
    /// Requests that became notes, their bodies and homes preserved
    pub requests_became_notes: usize,
    /// Replies that became home-task comments with empty mappings
    pub replies_rehomed: usize,
    /// Uses migrated as born-bound to their historical selection
    pub uses_born_bound: usize,
    /// c-N tokens renumbered onto the dense comment numbering
    pub comment_refs_renumbered: usize,
    /// c-N tokens naming non-comment records, rewritten as `seq N`
    pub record_refs_rewritten: usize,
    /// ranges rewritten as `seq N–M`
    pub ranges_rewritten: usize,
    /// i-N tokens renumbered onto the dense incarnation numbering
    pub incarnation_refs_renumbered: usize,
    /// every token that could not be identified, kept visibly legacy
    pub legacy: Vec<LegacyRef>,
}

#[derive(Debug, PartialEq)]
pub struct LegacyRef {
    pub seq: usize,
    pub field: String,
    pub token: String,
}

/// The old log as the migration walks it: enough of a fold to resolve
/// reply chains into home tasks and to name each birth.
struct V3Walk {
    /// each comment's target and resolved root task
    comments: BTreeMap<usize, (v3::Target, usize)>,
    /// task ids by birth record
    task_births: BTreeMap<usize, usize>,
    /// incarnation births (the establishment records), in order
    incarnation_births: Vec<usize>,
    /// the log's own length
    records: usize,
}

impl V3Walk {
    fn root_of(&self, target: &v3::Target) -> Option<usize> {
        match target {
            v3::Target::Task(t) => Some(*t),
            v3::Target::Comment(c) => {
                let mut up = *c;
                loop {
                    let (next, _) = self.comments.get(&up)?;
                    match next {
                        v3::Target::Task(t) => return Some(*t),
                        v3::Target::Comment(parent) => up = *parent,
                    }
                }
            }
        }
    }
}

/// Open the source read-only and verify it holds the migration's
/// schema. No write path ever opens it.
fn open_source(path: &Path) -> Result<Connection, MigrateFail> {
    if !path.exists() {
        return Err(MigrateFail::Db(db::DbError::Missing(path.to_path_buf())));
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let app_id: i32 = conn.query_row("PRAGMA application_id", [], |r| r.get(0))?;
    let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if app_id != i32::from_be_bytes(*b"sacd") || version != FROM_VERSION {
        return Err(MigrateFail::WrongSchema { found: version });
    }
    Ok(conn)
}

/// Migrate one schema-3 log into a fresh schema-4 tracker: read every
/// row, rewrite the payloads onto the new law, renumber the prose,
/// fold the result, and only then write. The source is untouched.
pub fn migrate(source: &Path, out: &Path) -> Result<Report, MigrateFail> {
    if out.exists() {
        return Err(MigrateFail::OutputExists(out.to_path_buf()));
    }
    let src = open_source(source)?;
    let mut stmt = src.prepare(
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
    drop(stmt);
    drop(src);

    let mut walk = V3Walk {
        comments: BTreeMap::new(),
        task_births: BTreeMap::new(),
        incarnation_births: Vec::new(),
        records: rows.len(),
    };
    // dense numbering: each birth's position among births of its kind
    let mut comment_births: Vec<usize> = Vec::new();
    let mut artifact_births: Vec<usize> = Vec::new();
    for row in &rows {
        match row.kind.as_str() {
            "task_created" => {
                walk.task_births.insert(row.seq, walk.task_births.len());
            }
            "commented" => {
                walk.comments.insert(row.seq, parse_comment(row)?);
                comment_births.push(row.seq);
            }
            "artifact_added" => artifact_births.push(row.seq),
            "workspace_established" => walk.incarnation_births.push(row.seq),
            _ => {}
        }
    }
    let comment_number: BTreeMap<usize, usize> = comment_births
        .iter()
        .enumerate()
        .map(|(n, seq)| (*seq, n))
        .collect();
    let incarnation_number: BTreeMap<usize, usize> = walk
        .incarnation_births
        .iter()
        .enumerate()
        .map(|(n, seq)| (*seq, n))
        .collect();
    let mut report = Report {
        records: rows.len(),
        ..Default::default()
    };

    // the rewriter every prose field passes through
    let is_comment = |seq: usize| comment_number.contains_key(&seq);
    let is_record = |seq: usize| seq < walk.records;
    let context = Renumber {
        comment_number: &comment_number,
        incarnation_number: &incarnation_number,
        is_comment: &is_comment,
        is_record: &is_record,
        records: walk.records,
    };
    let mut renumber = |text: &str, seq: usize, field: &str, report: &mut Report| -> String {
        rewrite_prose(text, seq, field, &context, report)
    };

    let mut out_rows: Vec<StoredRecord> = Vec::with_capacity(rows.len());
    for row in &rows {
        let payload: serde_json::Value =
            serde_json::from_str(&row.payload).map_err(|e| MigrateFail::Corrupt {
                seq: row.seq,
                detail: e.to_string(),
            })?;
        let (kind, payload) = match row.kind.as_str() {
            "task_created" => ("task_created".to_string(), {
                let mut p = payload;
                rewrite_field(&mut p, "name", row.seq, &mut renumber, &mut report);
                p.to_string()
            }),
            "task_claimed" => ("task_claimed".to_string(), payload.to_string()),
            "task_cancelled" | "task_reopened" => {
                let mut p = payload;
                rewrite_field(&mut p, "explanation", row.seq, &mut renumber, &mut report);
                (row.kind.clone(), p.to_string())
            }
            "commented" => rewrite_commented(row, payload, &walk, &mut renumber, &mut report)?,
            "comment_revised" => {
                let mut p = payload;
                rewrite_field(&mut p, "body", row.seq, &mut renumber, &mut report);
                ("comment_revised".to_string(), p.to_string())
            }
            "comment_withdrawn" => {
                let mut p = payload;
                rewrite_field(&mut p, "note", row.seq, &mut renumber, &mut report);
                ("comment_withdrawn".to_string(), p.to_string())
            }
            "use_bound" => ("use_bound".to_string(), payload.to_string()),
            "artifact_added" => {
                let mut p = payload;
                if let Some(artifact) = p.get_mut("artifact") {
                    rewrite_field(artifact, "name", row.seq, &mut renumber, &mut report);
                }
                ("artifact_added".to_string(), p.to_string())
            }
            "workspace_established" | "task_workspace_checkpointed" => {
                (row.kind.clone(), payload.to_string())
            }
            other => {
                return Err(MigrateFail::Corrupt {
                    seq: row.seq,
                    detail: format!("unknown kind {other}"),
                });
            }
        };
        out_rows.push(StoredRecord {
            seq: row.seq,
            event_time: row.event_time,
            logged_time: row.logged_time,
            actor: row.actor.clone(),
            tier: row.tier.clone(),
            kind,
            payload,
            scope: row.scope.clone(),
        });
    }
    // scope JSON rides typed values: birth references, never renumbered

    // the rewritten log must fold under the new law before anything
    // is written — a migration that cannot fold writes nothing
    let loadout = db::fold_rows(out_rows.clone()).map_err(MigrateFail::Db)?;
    match loadout.state {
        db::LoadState::Full(_) => {}
        db::LoadState::Degraded(reason) => {
            return Err(MigrateFail::Unfoldable {
                seq: 0,
                detail: reason,
            });
        }
    }

    let mut dst = db::open(out)?;
    let txn = dst.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    for row in &out_rows {
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
    }
    txn.commit()?;
    Ok(report)
}

fn parse_comment(row: &StoredRecord) -> Result<(v3::Target, usize), MigrateFail> {
    #[derive(Deserialize)]
    struct Commented {
        target: v3::Target,
    }
    let parsed: Commented =
        serde_json::from_str(&row.payload).map_err(|e| MigrateFail::Corrupt {
            seq: row.seq,
            detail: e.to_string(),
        })?;
    Ok((parsed.target, 0))
}

/// One prose field's renumbering pass.
fn rewrite_field(
    payload: &mut serde_json::Value,
    field: &str,
    seq: usize,
    renumber: &mut impl FnMut(&str, usize, &str, &mut Report) -> String,
    report: &mut Report,
) {
    if let Some(text) = payload.get(field).and_then(|v| v.as_str()) {
        let rewritten = renumber(text, seq, field, report);
        if let Some(slot) = payload.get_mut(field) {
            *slot = serde_json::Value::String(rewritten);
        }
    }
}

/// The v3 commented payload onto the v4 law: one home task, the
/// variant (requests become notes, uses carry their historical
/// binding at birth), and an empty citation mapping — old comments
/// never gain graph edges.
fn rewrite_commented(
    row: &StoredRecord,
    payload: serde_json::Value,
    walk: &V3Walk,
    renumber: &mut impl FnMut(&str, usize, &str, &mut Report) -> String,
    report: &mut Report,
) -> Result<(String, String), MigrateFail> {
    #[derive(Deserialize, Serialize)]
    struct V4Commented {
        home: usize,
        kind: serde_json::Value,
        references: Vec<usize>,
    }
    #[derive(Deserialize)]
    struct Parsed {
        target: v3::Target,
        kind: serde_json::Value,
    }
    let parsed: Parsed = serde_json::from_value(payload).map_err(|e| MigrateFail::Corrupt {
        seq: row.seq,
        detail: e.to_string(),
    })?;
    let root = walk.root_of(&parsed.target).ok_or(MigrateFail::Corrupt {
        seq: row.seq,
        detail: "a reply chain resolves to no task".into(),
    })?;
    if !matches!(parsed.target, v3::Target::Task(_)) {
        report.replies_rehomed += 1;
    }
    let mut kind = parsed.kind;
    let variant = kind
        .as_object()
        .and_then(|m| m.keys().next().cloned())
        .unwrap_or_default();
    // the prose lives inside the variant: descend before rewriting
    let rewrite_variant = |kind: &mut serde_json::Value,
                           field: &str,
                           renumber: &mut dyn FnMut(&str, usize, &str, &mut Report) -> String,
                           report: &mut Report| {
        let key = kind
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .unwrap_or_default();
        if let Some(inner) = kind.get_mut(&key) {
            rewrite_field(
                inner,
                field,
                row.seq,
                &mut |t, s, f, r| renumber(t, s, f, r),
                report,
            );
        }
    };
    match variant.as_str() {
        "note" | "offer" => {
            rewrite_variant(&mut kind, "body", renumber, report);
        }
        "request" => {
            rewrite_variant(&mut kind, "body", renumber, report);
            // a historical request becomes the note it reads as, its
            // body, attribution, and home preserved — no fabricated
            // message or graph edge
            let body = kind
                .get("request")
                .and_then(|v| v.get("body"))
                .cloned()
                .unwrap_or_default();
            kind = serde_json::json!({ "note": { "body": body } });
            report.requests_became_notes += 1;
        }
        "reject" => {
            rewrite_variant(&mut kind, "grounds", renumber, report);
        }
        "accept" => {}
        "use" => {
            #[derive(Deserialize)]
            struct UsePayload {
                offer: usize,
                reason: String,
            }
            let use_payload: UsePayload = serde_json::from_value(kind.get("use").cloned().unwrap())
                .map_err(|e| MigrateFail::Corrupt {
                    seq: row.seq,
                    detail: e.to_string(),
                })?;
            let reason = renumber(&use_payload.reason, row.seq, "reason", report);
            let source = walk
                .root_of(&v3::Target::Comment(use_payload.offer))
                .ok_or(MigrateFail::Corrupt {
                    seq: row.seq,
                    detail: "the relied-on offer resolves to no task".into(),
                })?;
            // born bound to the historical selection, the original
            // grounds preserved — no new human judgment anywhere
            kind = serde_json::json!({
                "use": {
                    "source": source,
                    "reason": reason,
                    "bound": {
                        "offer": use_payload.offer,
                        "grounds": reason,
                    }
                }
            });
            report.uses_born_bound += 1;
        }
        other => {
            return Err(MigrateFail::Corrupt {
                seq: row.seq,
                detail: format!("unknown comment variant {other}"),
            });
        }
    }
    let out = V4Commented {
        home: root,
        kind,
        references: Vec::new(),
    };
    Ok((
        "commented".to_string(),
        serde_json::to_string(&out).map_err(|e| MigrateFail::Corrupt {
            seq: row.seq,
            detail: e.to_string(),
        })?,
    ))
}

/// The renumbering context: what each token can resolve to.
struct Renumber<'a> {
    comment_number: &'a BTreeMap<usize, usize>,
    incarnation_number: &'a BTreeMap<usize, usize>,
    is_comment: &'a dyn Fn(usize) -> bool,
    is_record: &'a dyn Fn(usize) -> bool,
    records: usize,
}

/// Rewrite one prose field's c-N and i-N tokens onto the public
/// numbering. Identified tokens renumber; ranges of local records
/// ride the raw spelling end to end (`seq N–M`) unless both ends are
/// comment births; tokens this log cannot identify — and tokens
/// already carrying the legacy mark — stay visibly legacy and join
/// the report, never re-qualified.
fn rewrite_prose(
    text: &str,
    seq: usize,
    field: &str,
    ctx: &Renumber,
    report: &mut Report,
) -> String {
    let chars: Vec<char> = text.chars().collect();
    let word = |c: char| c.is_alphanumeric() || matches!(c, '-' | '#' | '_');
    // the local membership a digit string can prove, fallibly
    let local = |digits: &str| digits.parse::<usize>().ok().filter(|n| (ctx.is_record)(*n));
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        // c-N and i-N tokens, word-bounded
        let prefix = matches!(chars[i], 'c' | 'i')
            && chars.get(i + 1) == Some(&'-')
            && (i == 0 || !word(chars[i - 1]));
        if prefix {
            let digits: String = chars[i + 2..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            let bounded =
                digits.is_empty() || chars.get(i + 2 + digits.len()).is_none_or(|c| !word(*c));
            if !digits.is_empty() && bounded {
                let end = i + 2 + digits.len();
                let letter = chars[i];
                // a token our own mark already qualified stays as it
                // stands: reported, never rewritten again
                let already_marked = out.ends_with("legacy ")
                    || (out.ends_with("legacy") && chars.get(i - 1) == Some(&' '));
                // a range: c-A–c-B or c-A..c-B
                let range = range_after(&chars, end, letter);
                if let Some((far_digits, stop)) = range {
                    let near = local(&digits);
                    let far = local(&far_digits);
                    let both_comments = near
                        .filter(|n| (ctx.is_comment)(*n))
                        .zip(far.filter(|n| (ctx.is_comment)(*n)));
                    if already_marked {
                        out.push_str(&format!("{letter}-{digits}–{letter}-{far_digits}"));
                        report.legacy.push(LegacyRef {
                            seq,
                            field: field.to_string(),
                            token: format!("{letter}-{digits}–{letter}-{far_digits}"),
                        });
                    } else if let Some((a, b)) = both_comments {
                        out.push_str(&format!(
                            "c-{}–c-{}",
                            ctx.comment_number[&a], ctx.comment_number[&b]
                        ));
                        report.comment_refs_renumbered += 2;
                    } else if let (Some(a), Some(b)) = (near, far) {
                        // local and mixed, or raw end to end: the range
                        // rides the raw spelling
                        out.push_str(&format!("seq {}–{}", a.min(b), a.max(b)));
                        report.ranges_rewritten += 1;
                    } else {
                        out.push_str(&format!("legacy {letter}-{digits}–{letter}-{far_digits}"));
                        report.legacy.push(LegacyRef {
                            seq,
                            field: field.to_string(),
                            token: format!("{letter}-{digits}–{letter}-{far_digits}"),
                        });
                    }
                    i = stop;
                    continue;
                }
                let near = local(&digits);
                if already_marked {
                    out.push_str(&format!("{letter}-{digits}"));
                    report.legacy.push(LegacyRef {
                        seq,
                        field: field.to_string(),
                        token: format!("{letter}-{digits}"),
                    });
                // the dense numbering of the kind the token names:
                // membership in that kind's own map, not mere locality
                } else if let Some(n) = near.filter(|n| match letter {
                    'c' => (ctx.is_comment)(*n),
                    _ => ctx.incarnation_number.contains_key(n),
                }) {
                    match letter {
                        'c' => {
                            out.push_str(&format!("c-{}", ctx.comment_number[&n]));
                            report.comment_refs_renumbered += 1;
                        }
                        _ => {
                            out.push_str(&format!("i-{}", ctx.incarnation_number[&n]));
                            report.incarnation_refs_renumbered += 1;
                        }
                    }
                // c-N was the all-purpose record prefix: a local
                // non-comment rides the raw spelling. i-N names
                // incarnations — local membership at a non-establishment
                // record proves nothing about its provenance, so it
                // stays legacy and reported, never remapped.
                } else if let Some(n) = near.filter(|_| letter == 'c') {
                    out.push_str(&format!("seq {n}"));
                    report.record_refs_rewritten += 1;
                } else {
                    out.push_str(&format!("legacy {letter}-{digits}"));
                    report.legacy.push(LegacyRef {
                        seq,
                        field: field.to_string(),
                        token: format!("{letter}-{digits}"),
                    });
                }
                i = end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    let _ = ctx.records;
    out
}

/// Whether `end` begins a same-letter range (`–` or `..` then the
/// same c-/i- token), returning the far number and the end position.
fn range_after(chars: &[char], end: usize, letter: char) -> Option<(String, usize)> {
    let word = |c: char| c.is_alphanumeric() || matches!(c, '-' | '#' | '_');
    let after = if chars.get(end) == Some(&'–') {
        end + 1
    } else if chars.get(end) == Some(&'.') && chars.get(end + 1) == Some(&'.') {
        end + 2
    } else {
        return None;
    };
    if chars.get(after) != Some(&letter)
        || chars.get(after + 1) != Some(&'-')
        || chars.get(after.saturating_sub(1)).is_some_and(|c| word(*c))
    {
        return None;
    }
    let digits: String = chars[after + 2..]
        .iter()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty()
        || chars
            .get(after + 2 + digits.len())
            .is_some_and(|c| word(*c))
    {
        return None;
    }
    let stop = after + 2 + digits.len();
    Some((digits, stop))
}

#[cfg(test)]
mod test {
    use super::*;

    /// A schema-3 log with one row of each shape the migration reads.
    fn v3_db(dir: &Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let conn = Connection::open(&path).unwrap();
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
             PRAGMA user_version = 3;",
        )
        .unwrap();
        let insert = |seq: i64, kind: &str, payload: &str, scope: Option<&str>| {
            conn.execute(
                "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
                 VALUES (?1, 1000+?1, 2000, 'jerry', 'human', ?2, ?3, ?4)",
                params![seq, kind, payload, scope],
            )
            .unwrap();
        };
        insert(0, "task_created", r#"{"name":"implement foo"}"#, None);
        insert(1, "task_claimed", r#"{"id":0,"incarnation":null}"#, None);
        insert(
            2,
            "commented",
            r#"{"target":{"task":0},"kind":{"note":{"body":"parked: per c-2 and t-0"}}}"#,
            None,
        );
        insert(
            3,
            "commented",
            r#"{"target":{"task":0},"kind":{"offer":{"body":"suite green, see c-2 and c-1"}}}"#,
            None,
        );
        insert(
            4,
            "commented",
            r#"{"target":{"comment":3},"kind":{"accept":{"offer":3}}}"#,
            None,
        );
        insert(
            5,
            "commented",
            r#"{"target":{"comment":2},"kind":{"request":{"body":"please resume, re c-2"}}}"#,
            None,
        );
        insert(
            6,
            "commented",
            r#"{"target":{"comment":5},"kind":{"note":{"body":"a reply in the chain, see c-99 out of range and c-0 raw"}}}"#,
            None,
        );
        insert(
            7,
            "task_created",
            r#"{"name":"consume foo, ranges c-2–c-3 and c-0..c-2 mixed"}"#,
            None,
        );
        insert(
            8,
            "commented",
            r#"{"target":{"task":1},"kind":{"offer":{"body":"the replacement"}}}"#,
            None,
        );
        insert(
            9,
            "commented",
            r#"{"target":{"task":1},"kind":{"use":{"offer":3,"reason":"builds on foo, per c-4"}}}"#,
            None,
        );
        insert(
            10,
            "comment_revised",
            r#"{"id":2,"body":"parked, corrected: c-2 and i-1 legacy"}"#,
            None,
        );
        insert(
            11,
            "workspace_established",
            r#"{"task_id":1,"base":"abc123","branch":"saccade/t-1","worktree":"/repo/wt/t-1"}"#,
            None,
        );
        insert(
            12,
            "task_claimed",
            r#"{"id":1,"incarnation":11}"#,
            Some(r#"{"task":1,"incarnation":11,"workspace":null}"#),
        );
        insert(
            13,
            "artifact_added",
            r#"{"root":1,"artifact":{"name":"sweep figure","hash":"e9a92b18a64ca761853cbb18417b3f2e0fa252949cf687a40b43e182d669c699"}}"#,
            None,
        );
        insert(
            14,
            "task_cancelled",
            r#"{"id":1,"explanation":"the i-11 worker left, see c-4"}"#,
            None,
        );
        path
    }

    #[test]
    fn the_migration_rewrites_and_folds() {
        let dir = std::env::temp_dir().join(format!("sac-migrate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = v3_db(&dir, "v3.db");
        let out = dir.join("v4.db");

        let report = migrate(&source, &out).unwrap();
        // the story's rewrites
        assert_eq!(report.records, 15);
        assert_eq!(report.requests_became_notes, 1);
        assert_eq!(report.replies_rehomed, 3);
        assert_eq!(report.uses_born_bound, 1);
        // c-2 tokens (the note's own birth) renumbered to the dense c-1
        assert!(report.comment_refs_renumbered >= 5);
        // c-1 (a claim record) and c-0 (a raw task birth) are
        // non-comment records: rewritten as seq
        assert!(report.record_refs_rewritten >= 2);
        // c-2–c-3 renumbered as a comment range; the mixed-local
        // c-0..c-2 rides the raw spelling end to end
        assert_eq!(report.ranges_rewritten, 1);
        // c-99 out of range stays legacy and reported
        assert!(
            report.legacy.iter().any(|l| l.token == "c-99"),
            "{report:?}"
        );

        // the output is a schema-4 tracker that folds and reads
        let conn = db::open_read(&out).unwrap();
        let loadout = db::load(&conn).unwrap();
        let db::LoadState::Full(world) = loadout.state else {
            panic!("the migrated log folds");
        };
        // seqs preserved: dense and identical
        assert_eq!(
            loadout.rows.iter().map(|r| r.seq).collect::<Vec<_>>(),
            (0..15).collect::<Vec<_>>()
        );
        // timestamps preserved
        assert_eq!(loadout.rows[0].event_time, 1000);
        // the request became a note on its home task, body preserved
        let note = serde_json::from_str::<serde_json::Value>(&loadout.rows[5].payload).unwrap();
        assert!(
            note["kind"].as_object().unwrap().contains_key("note"),
            "{}",
            loadout.rows[5].payload
        );
        assert!(
            note["kind"]["note"]["body"]
                .as_str()
                .unwrap()
                .contains("c-0")
        );
        // the reply became a home note with an empty mapping
        let reply = serde_json::from_str::<serde_json::Value>(&loadout.rows[6].payload).unwrap();
        assert_eq!(reply["home"], 0);
        assert_eq!(reply["references"].as_array().unwrap().len(), 0);
        // the use carries its source and its historical binding at birth
        let use_row = serde_json::from_str::<serde_json::Value>(&loadout.rows[9].payload).unwrap();
        assert_eq!(use_row["kind"]["use"]["source"], 0);
        assert_eq!(use_row["kind"]["use"]["bound"]["offer"], 3);
        // the fold agrees: the use is born bound, the claim's typed
        // incarnation reference survived untouched
        let use_ctx = world
            .comments
            .get(&crate::CommentId(crate::RecordId(9)))
            .unwrap();
        assert_eq!(
            use_ctx.comment.state.payload_offer(),
            Some(crate::CommentId(crate::RecordId(3)))
        );
        assert!(
            world
                .incarnations
                .contains_key(&crate::IncarnationId(crate::RecordId(11)))
        );
        // the accept homed on the offer's task
        assert_eq!(
            world.comments[&crate::CommentId(crate::RecordId(4))]
                .comment
                .home,
            crate::TaskId(0)
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_migration_refuses_wrong_schemas_and_used_outputs() {
        let dir = std::env::temp_dir().join(format!("sac-migrate-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // a v4 tracker is not a migration source
        let v4 = dir.join("v4.db");
        {
            let conn = db::open(&v4).unwrap();
            drop(conn);
        }
        assert!(matches!(
            migrate(&v4, &dir.join("out.db")),
            Err(MigrateFail::WrongSchema { .. })
        ));
        // a fresh output only
        let source = v3_db(&dir, "v3b.db");
        assert!(matches!(
            migrate(&source, &v4),
            Err(MigrateFail::OutputExists(_))
        ));
        // the source was never touched: still schema 3, still readable
        let conn = open_source(&source).unwrap();
        let version: i32 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, 3);
        drop(conn);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The prose rewriter's token classes, directly.
    #[test]
    fn the_rewriter_classifies_every_token() {
        let comment_number: BTreeMap<usize, usize> =
            [(2usize, 0usize), (3usize, 1usize)].into_iter().collect();
        let incarnation_number: BTreeMap<usize, usize> = [(11usize, 0usize)].into_iter().collect();
        let is_comment = |seq: usize| comment_number.contains_key(&seq);
        let is_record = |seq: usize| seq < 12;
        let ctx = Renumber {
            comment_number: &comment_number,
            incarnation_number: &incarnation_number,
            is_comment: &is_comment,
            is_record: &is_record,
            records: 12,
        };
        let mut report = Report::default();
        let rewrite =
            |text: &str, report: &mut Report| rewrite_prose(text, 0, "body", &ctx, report);

        // a comment birth renumbers onto the dense numbering
        assert_eq!(rewrite("per c-2 and c-3", &mut report), "per c-0 and c-1");
        // a raw record becomes seq; out-of-range stays legacy
        assert_eq!(
            rewrite("c-0, c-4, c-99", &mut report),
            "seq 0, seq 4, legacy c-99"
        );
        // word boundaries hold: c-20 is not c-2
        assert_eq!(rewrite("c-20 stays", &mut report), "legacy c-20 stays");
        // ranges: both comments renumber; local raw or mixed ranges
        // ride the raw spelling end to end
        assert_eq!(rewrite("c-2–c-3", &mut report), "c-0–c-1");
        assert_eq!(rewrite("c-0..c-4", &mut report), "seq 0–4");
        assert_eq!(rewrite("c-2–c-4", &mut report), "seq 2–4");
        // a range with a foreign end stays legacy, reported
        assert_eq!(rewrite("c-2–c-99", &mut report), "legacy c-2–c-99");
        assert!(
            report.legacy.iter().any(|l| l.token == "c-2–c-99"),
            "{report:?}"
        );
        // incarnations renumber; foreign ones stay legacy
        assert_eq!(rewrite("i-11 and i-3", &mut report), "i-0 and legacy i-3");
        // ordinary words and task refs pass through
        assert_eq!(
            rewrite("see t-0 and floop", &mut report),
            "see t-0 and floop"
        );
    }
}
