//! The offline migration: one older log rewritten into the current
//! schema, one step per schema generation. The source is never
//! touched; seq and timestamps carry over unchanged; typed stored
//! reference values stay birth references. The schema-3 step renumbers
//! prose onto the fold's public numbering - comment births to their
//! dense c-N, non-comment records to `seq N`, ranges to `seq N..M` -
//! and every token that cannot be identified stays visibly legacy,
//! reported for human review rather than reassigned. The schema-4
//! step backfills the authored titles the deposit variants gained:
//! each untitled Note and Offer acquires its home task's name, exactly
//! once, with no read-side inheritance and no semantic rewriting.

use std::collections::BTreeMap;
use std::path::Path;

use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};

use crate::db::{self, StoredRecord};

/// The schema generations this migration reads, verified before
/// anything else: every step from the source's generation to the
/// current one runs, in order.
const SOURCE_VERSIONS: [i32; 2] = [db::MIGRATE_FROM_VERSION, db::MIGRATE_FROM_VERSION + 1];

/// The v3 log's event vocabulary, exactly as Saccade 0.13.0 wrote it:
/// read here, never folded - the shapes live only as long as the
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
    /// a rewritten record does not fold under the new law - the
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
                "the source holds schema v{found}; this migration reads v{} or v{}",
                SOURCE_VERSIONS[0], SOURCE_VERSIONS[1]
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
    /// ranges rewritten as `seq N..M`
    pub ranges_rewritten: usize,
    /// i-N tokens renumbered onto the dense incarnation numbering
    pub incarnation_refs_renumbered: usize,
    /// Untitled notes and offers that acquired their home task's
    /// name as title
    pub titles_backfilled: usize,
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

/// Open the source read-only and verify it holds a schema this
/// migration reads, returning that generation. No write path ever
/// opens it.
fn open_source(path: &Path) -> Result<(Connection, i32), MigrateFail> {
    if !path.exists() {
        return Err(MigrateFail::Db(db::DbError::Missing(path.to_path_buf())));
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let app_id: i32 = conn.query_row("PRAGMA application_id", [], |r| r.get(0))?;
    let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if app_id != i32::from_be_bytes(*b"sacd") || !SOURCE_VERSIONS.contains(&version) {
        return Err(MigrateFail::WrongSchema { found: version });
    }
    Ok((conn, version))
}

/// Migrate one older log into a fresh tracker at the current schema:
/// read every row, run every rewrite step from the source's
/// generation onward, fold the result, and only then write. The
/// source is untouched.
pub fn migrate(source: &Path, out: &Path) -> Result<Report, MigrateFail> {
    if out.exists() {
        return Err(MigrateFail::OutputExists(out.to_path_buf()));
    }
    let (src, generation) = open_source(source)?;
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

    let mut report = Report {
        records: rows.len(),
        ..Default::default()
    };

    // step one, from schema 3 only: the renumbering onto the public
    // numbering. A schema-4 source enters past it.
    let out_rows = if generation == db::MIGRATE_FROM_VERSION {
        renumber_rows(&rows, &mut report)?
    } else {
        rows.clone()
    };
    // step two, from schema 4 and after: the authored titles arrive
    let out_rows = backfill_titles(out_rows, &mut report)?;

    // the rewritten log must fold under the new law before anything
    // is written - a migration that cannot fold writes nothing
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

/// The schema-3 step: every row rewritten onto the v4 law, prose
/// renumbered onto the fold's public numbering.
fn renumber_rows(
    rows: &[StoredRecord],
    mut report: &mut Report,
) -> Result<Vec<StoredRecord>, MigrateFail> {
    let mut walk = V3Walk {
        comments: BTreeMap::new(),
        task_births: BTreeMap::new(),
        incarnation_births: Vec::new(),
        records: rows.len(),
    };
    // dense numbering: each birth's position among births of its kind
    let mut comment_births: Vec<usize> = Vec::new();
    for row in rows {
        match row.kind.as_str() {
            "task_created" => {
                walk.task_births.insert(row.seq, walk.task_births.len());
            }
            "commented" => {
                walk.comments.insert(row.seq, parse_comment(row)?);
                comment_births.push(row.seq);
            }
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
    for row in rows {
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
    Ok(out_rows)
}

/// The schema-4 step: each untitled Note and Offer acquires its home
/// task's name as title, exactly once - the authored description the
/// deposit variants gained. Existing titles stand untouched, every
/// other family rides byte for byte, and nothing is inferred or
/// rewritten: the home task's recorded name is the whole derivation.
fn backfill_titles(
    rows: Vec<StoredRecord>,
    report: &mut Report,
) -> Result<Vec<StoredRecord>, MigrateFail> {
    // each task's name, by dense birth order: home N names the task
    // born Nth
    let mut task_names: Vec<String> = Vec::new();
    for row in &rows {
        if row.kind != "task_created" {
            continue;
        }
        #[derive(Deserialize)]
        struct Created {
            name: String,
        }
        let parsed: Created =
            serde_json::from_str(&row.payload).map_err(|e| MigrateFail::Corrupt {
                seq: row.seq,
                detail: e.to_string(),
            })?;
        task_names.push(parsed.name);
    }
    let mut out_rows = Vec::with_capacity(rows.len());
    for mut row in rows {
        if row.kind != "commented" {
            out_rows.push(row);
            continue;
        }
        let mut payload: serde_json::Value =
            serde_json::from_str(&row.payload).map_err(|e| MigrateFail::Corrupt {
                seq: row.seq,
                detail: e.to_string(),
            })?;
        #[derive(Deserialize)]
        struct Commented {
            home: usize,
            kind: serde_json::Value,
        }
        let parsed: Commented =
            serde_json::from_value(payload.clone()).map_err(|e| MigrateFail::Corrupt {
                seq: row.seq,
                detail: e.to_string(),
            })?;
        let variant = parsed
            .kind
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .unwrap_or_default();
        if variant != "note" && variant != "offer" {
            out_rows.push(row);
            continue;
        }
        let Some(inner) = parsed.kind.get(&variant) else {
            out_rows.push(row);
            continue;
        };
        // a title already standing stays as it stands
        if inner.get("title").is_some_and(|t| !t.is_null()) {
            out_rows.push(row);
            continue;
        }
        let name = task_names
            .get(parsed.home)
            .ok_or_else(|| MigrateFail::Corrupt {
                seq: row.seq,
                detail: format!(
                    "the comment homes on task {} but the log holds {} tasks",
                    parsed.home,
                    task_names.len()
                ),
            })?;
        if let Some(inner) = payload.get_mut("kind").and_then(|k| k.get_mut(&variant))
            && let Some(object) = inner.as_object_mut()
        {
            object.insert("title".to_string(), serde_json::Value::String(name.clone()));
        }
        row.payload = serde_json::to_string(&payload).map_err(|e| MigrateFail::Corrupt {
            seq: row.seq,
            detail: e.to_string(),
        })?;
        report.titles_backfilled += 1;
        out_rows.push(row);
    }
    Ok(out_rows)
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
/// binding at birth), and an empty citation mapping - old comments
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
            // body, attribution, and home preserved - no fabricated
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
            // grounds preserved - no new human judgment anywhere
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
/// ride the raw spelling end to end (`seq N..M`) unless both ends are
/// comment births; tokens this log cannot identify - and tokens
/// already carrying the legacy mark - stay visibly legacy and join
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
                // a range: same-letter tokens joined by the legacy
                // dash or by ..
                let range = range_after(&chars, end, letter);
                if let Some((far_digits, stop, sep)) = range {
                    let near = local(&digits);
                    let far = local(&far_digits);
                    let both_comments = near
                        .filter(|n| (ctx.is_comment)(*n))
                        .zip(far.filter(|n| (ctx.is_comment)(*n)));
                    if already_marked {
                        out.push_str(&format!("{letter}-{digits}{sep}{letter}-{far_digits}"));
                        report.legacy.push(LegacyRef {
                            seq,
                            field: field.to_string(),
                            token: format!("{letter}-{digits}{sep}{letter}-{far_digits}"),
                        });
                    } else if let Some((a, b)) = both_comments {
                        out.push_str(&format!(
                            "c-{}..c-{}",
                            ctx.comment_number[&a], ctx.comment_number[&b]
                        ));
                        report.comment_refs_renumbered += 2;
                    } else if let (Some(a), Some(b)) = (near, far) {
                        // local and mixed, or raw end to end: the range
                        // rides the raw spelling
                        out.push_str(&format!("seq {}..{}", a.min(b), a.max(b)));
                        report.ranges_rewritten += 1;
                    } else {
                        out.push_str(&format!(
                            "legacy {letter}-{digits}{sep}{letter}-{far_digits}"
                        ));
                        report.legacy.push(LegacyRef {
                            seq,
                            field: field.to_string(),
                            token: format!("{letter}-{digits}{sep}{letter}-{far_digits}"),
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
                // incarnations - local membership at a non-establishment
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

/// Whether `end` begins a same-letter range (the legacy dash or `..`
/// then the same c-/i- token), returning the far number, the end
/// position, and the raw separator the prose carries.
fn range_after(chars: &[char], end: usize, letter: char) -> Option<(String, usize, &'static str)> {
    let word = |c: char| c.is_alphanumeric() || matches!(c, '-' | '#' | '_');
    let (after, sep) = if chars.get(end) == Some(&'\u{2013}') {
        (end + 1, "\u{2013}")
    } else if chars.get(end) == Some(&'.') && chars.get(end + 1) == Some(&'.') {
        (end + 2, "..")
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
    Some((digits, stop, sep))
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
            &format!(
                r#"{{"name":"consume foo, ranges c-2{D}c-3 and c-0..c-2 mixed"}}"#,
                D = "\u{2013}"
            ),
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
        // the legacy-dash range renumbers as comments; the mixed c-0..c-2 rides raw
        assert_eq!(report.ranges_rewritten, 1);
        // c-99 out of range stays legacy and reported
        assert!(
            report.legacy.iter().any(|l| l.token == "c-99"),
            "{report:?}"
        );

        // the output is a current-schema tracker that folds and reads
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
        // the request became a note on its home task, body preserved,
        // carrying the backfilled title
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
        // the title step ran after the renumbering step: every untitled
        // deposit carries its home task's name - five deposits, two tasks
        assert_eq!(report.titles_backfilled, 5);
        assert!(
            note["kind"]["note"]["title"].as_str() == Some("implement foo"),
            "{}",
            loadout.rows[5].payload
        );
        let offer = serde_json::from_str::<serde_json::Value>(&loadout.rows[3].payload).unwrap();
        assert_eq!(
            offer["kind"]["offer"]["title"].as_str(),
            Some("implement foo")
        );
        let other = serde_json::from_str::<serde_json::Value>(&loadout.rows[8].payload).unwrap();
        assert!(
            other["kind"]["offer"]["title"]
                .as_str()
                .is_some_and(|t| t.contains("consume foo")),
            "{}",
            loadout.rows[8].payload
        );
        // migration's own spellings are ASCII: the range renumbers onto ..
        assert!(
            loadout.rows[7]
                .payload
                .contains("ranges c-0..c-1 and seq 0..2 mixed"),
            "{}",
            loadout.rows[7].payload
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
        // the accept homed on the offer's task, and the folded deposits
        // present their backfilled titles
        assert_eq!(
            world.comments[&crate::CommentId(crate::RecordId(4))]
                .comment
                .home,
            crate::TaskId(0)
        );
        assert_eq!(
            world.comments[&crate::CommentId(crate::RecordId(2))]
                .comment
                .kind
                .title()
                .map(|t| t.as_str()),
            Some("implement foo")
        );
        assert_eq!(
            world.comments[&crate::CommentId(crate::RecordId(4))]
                .comment
                .kind
                .title(),
            None,
            "the accept carries no title"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A schema-4 source backfills each untitled deposit with its home task's name, exactly once, preserving everything else.
    #[test]
    fn the_schema4_step_backfills_titles_once() {
        // an exclusively created unique root: a collision refuses, it
        // never deletes, and only this test's own root is cleaned
        let dir = std::env::temp_dir().join(format!(
            "sac-migrate-v4-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir(&dir).expect("the exclusive scratch root is created, never taken");
        let source = dir.join("v4.db");
        {
            let conn = Connection::open(&source).unwrap();
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
                 PRAGMA user_version = 4;",
            )
            .unwrap();
            let insert = |seq: i64, kind: &str, payload: &str| {
                conn.execute(
                    "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
                     VALUES (?1, 1000+?1, 2000, 'jerry', 'human', ?2, ?3, NULL)",
                    params![seq, kind, payload],
                )
                .unwrap();
            };
            insert(0, "task_created", r#"{"name":"implement foo"}"#);
            insert(1, "task_created", r#"{"name":"adopt the accounts"}"#);
            insert(
                2,
                "commented",
                r#"{"home":0,"kind":{"note":{"body":"parked mid-flight"}},"references":[]}"#,
            );
            insert(
                3,
                "commented",
                r#"{"home":0,"kind":{"note":{"body":"the fluke is real"}},"references":[]}"#,
            );
            insert(
                4,
                "commented",
                r#"{"home":0,"kind":{"offer":{"body":"suite green, 55 tests"}},"references":[]}"#,
            );
            insert(
                5,
                "commented",
                r#"{"home":0,"kind":{"accept":{"offer":4}},"references":[]}"#,
            );
            insert(
                6,
                "commented",
                r#"{"home":1,"kind":{"use":{"source":0,"reason":"builds on foo"}},"references":[]}"#,
            );
            insert(
                7,
                "comment_revised",
                r#"{"id":3,"body":"the fluke is real, corrected"}"#,
            );
            insert(
                8,
                "comment_withdrawn",
                r#"{"id":2,"note":"the parking was premature"}"#,
            );
            insert(
                9,
                "artifact_added",
                r#"{"root":1,"artifact":{"name":"sweep figure","hash":"e9a92b18a64ca761853cbb18417b3f2e0fa252949cf687a40b43e182d669c699"}}"#,
            );
            insert(
                10,
                "workspace_established",
                r#"{"task_id":1,"base":"abc123","branch":"saccade/t-1","worktree":"/repo/wt/t-1"}"#,
            );
            insert(
                11,
                "commented",
                r#"{"home":1,"kind":{"offer":{"title":"the standing title","body":"already titled"}},"references":[]}"#,
            );
        }

        let out = dir.join("v5.db");
        let report = migrate(&source, &out).unwrap();
        // the three untitled deposits acquired their home task's name;
        // the already-titled offer kept its own
        assert_eq!(report.titles_backfilled, 3);
        assert_eq!(report.records, 12);

        let conn = db::open_read(&out).unwrap();
        let loadout = db::load(&conn).unwrap();
        let db::LoadState::Full(world) = loadout.state else {
            panic!("the migrated log folds");
        };
        // seqs and timestamps preserved
        assert_eq!(
            loadout.rows.iter().map(|r| r.seq).collect::<Vec<_>>(),
            (0..12).collect::<Vec<_>>()
        );
        assert_eq!(loadout.rows[0].event_time, 1000);
        // the untitled note and offer carry the home task's name
        let note = serde_json::from_str::<serde_json::Value>(&loadout.rows[2].payload).unwrap();
        assert_eq!(
            note["kind"]["note"]["title"].as_str(),
            Some("implement foo")
        );
        let offer = serde_json::from_str::<serde_json::Value>(&loadout.rows[4].payload).unwrap();
        assert_eq!(
            offer["kind"]["offer"]["title"].as_str(),
            Some("implement foo")
        );
        let other = serde_json::from_str::<serde_json::Value>(&loadout.rows[11].payload).unwrap();
        assert_eq!(
            other["kind"]["offer"]["title"].as_str(),
            Some("the standing title")
        );
        // the accept gains neither a title nor a placeholder body, the
        // use and the revision ride byte for byte
        let accept = serde_json::from_str::<serde_json::Value>(&loadout.rows[5].payload).unwrap();
        assert!(accept["kind"]["accept"].get("title").is_none());
        assert!(accept["kind"]["accept"].get("body").is_none());
        assert_eq!(
            loadout.rows[6].payload,
            r#"{"home":1,"kind":{"use":{"source":0,"reason":"builds on foo"}},"references":[]}"#
        );
        assert_eq!(
            loadout.rows[7].payload,
            r#"{"id":3,"body":"the fluke is real, corrected"}"#
        );
        assert_eq!(
            loadout.rows[9].payload,
            r#"{"root":1,"artifact":{"name":"sweep figure","hash":"e9a92b18a64ca761853cbb18417b3f2e0fa252949cf687a40b43e182d669c699"}}"#
        );

        // the fold: the revised note presents its corrected body beside
        // the backfilled title - the older presentation revision never
        // displaces it
        let revised = &world.comments[&crate::CommentId(crate::RecordId(3))];
        assert_eq!(
            revised.comment.kind.prose().map(|p| p.as_str()),
            Some("the fluke is real, corrected")
        );
        assert_eq!(
            revised.comment.kind.title().map(|t| t.as_str()),
            Some("implement foo")
        );
        // the withdrawn deposit keeps its title bytes in history and
        // the read side never discloses them
        let withdrawn = &world.comments[&crate::CommentId(crate::RecordId(2))];
        assert!(matches!(
            withdrawn.comment.state,
            crate::CommentState::Withdrawn(_)
        ));
        assert_eq!(
            withdrawn.comment.kind.title().map(|t| t.as_str()),
            Some("implement foo")
        );
        let line =
            crate::views::comment_line(&world, crate::CommentId(crate::RecordId(2))).unwrap();
        assert_eq!(line.title, None);
        assert_eq!(line.body, "parked mid-flight");

        // the source stays schema 4, untouched
        let (raw, version) = open_source(&source).unwrap();
        assert_eq!(version, 4);
        drop(raw);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_migration_refuses_wrong_schemas_and_used_outputs() {
        let dir = std::env::temp_dir().join(format!("sac-migrate-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // a current-schema tracker is not a migration source
        let current = dir.join("current.db");
        {
            let conn = db::open(&current).unwrap();
            drop(conn);
        }
        assert!(matches!(
            migrate(&current, &dir.join("out.db")),
            Err(MigrateFail::WrongSchema { .. })
        ));
        // a fresh output only
        let source = v3_db(&dir, "v3b.db");
        assert!(matches!(
            migrate(&source, &current),
            Err(MigrateFail::OutputExists(_))
        ));
        // the source was never touched: still schema 3, still readable
        let (conn, version) = open_source(&source).unwrap();
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
        // ranges renumber; raw or mixed ranges ride raw; the legacy dash is escape-recognized
        assert_eq!(rewrite("c-2\u{2013}c-3", &mut report), "c-0..c-1");
        assert_eq!(rewrite("c-0..c-4", &mut report), "seq 0..4");
        assert_eq!(rewrite("c-2\u{2013}c-4", &mut report), "seq 2..4");
        // a range with a foreign end stays legacy in its raw spelling
        assert_eq!(
            rewrite("c-2\u{2013}c-99", &mut report),
            "legacy c-2\u{2013}c-99"
        );
        assert_eq!(rewrite("c-2..c-99", &mut report), "legacy c-2..c-99");
        assert!(
            report.legacy.iter().any(|l| l.token == "c-2\u{2013}c-99"),
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
