use super::launch::provision_new;
use super::*;
use std::path::Path;

use saccade::host::workspace::{ListRow, MissingPiece};
use saccade::serve::client::CleanReply;

use super::completions::collapse;

/// The creation's text face: actual Git facts and the birth record.
pub(super) fn render_birth(birth: &client::WorkspaceBirth) -> String {
    format!(
        "workspace provisioned - record {} (its identity, no numbering)\n  branch {} | base {}\n  path {}",
        birth.seq, birth.branch, birth.base, birth.path
    )
}

/// `sac workspace new`: provision and report; no task, claim, or
/// incarnation rides it.
pub(super) fn workspace_new_door(
    cli: &Cli,
    destination: &Destination,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<String, Fail> {
    let birth = provision_new(destination, label, base)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "created": true,
            "workspace": birth,
        }))
        .expect("plain data"));
    }
    Ok(render_birth(&birth))
}

pub(super) fn workspace_door(
    cli: &Cli,
    destination: &Destination,
    verb: &WorkspaceVerb,
) -> Result<String, Fail> {
    match verb {
        WorkspaceVerb::New { label, base } => {
            workspace_new_door(cli, destination, label.as_deref(), base.as_deref())
        }
        WorkspaceVerb::Clean { path, merged, yes } => {
            if *merged {
                return workspace_sweep_door(cli, destination, *yes);
            }
            if *yes {
                return Err(Fail::Usage(
                    "--yes acts only the sweep: pass --merged, never a path".into(),
                ));
            }
            let Some(path) = path else {
                return Err(Fail::Usage(
                    "workspace clean names a recorded workspace path, or --merged previews the sweep".into(),
                ));
            };
            workspace_clean_door(cli, destination, path)
        }
        WorkspaceVerb::List => workspace_list_door(cli, destination),
    }
}

/// `sac workspace list`: the standing workspaces, one row each. A
/// read like `sac list`: the fold and git only, over either
/// destination shape, no server door. A missing worktree or branch is
/// a status of its own, never a crash; any other git plumbing
/// failure refuses the read.
pub(super) fn workspace_list_door(cli: &Cli, destination: &Destination) -> Result<String, Fail> {
    let loadout = loadout_of(destination)?;
    let world = match loadout.state {
        LoadState::Full(world) => world,
        LoadState::Degraded(reason) => return Err(Fail::Degraded(reason)),
    };
    let root = resolve_repo(cli)?;
    let rows = saccade::workspace::workspace_list(&world, &root)?;
    let cursor = loadout.rows.last().map(|r| r.seq);
    if cli.json {
        let rows: Vec<serde_json::Value> = rows.iter().map(list_row_value).collect();
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "workspaces": rows,
            "cursor": cursor,
        }))
        .expect("rows are plain data"));
    }
    Ok(render_list_table(&rows))
}

/// One row's json face: the same fields the text row carries, the
/// merged rule's full answer riding `status.merged_into` (the text
/// names only the first), the missing piece named when it is there,
/// and `null` figures where the branch is gone.
fn list_row_value(row: &ListRow) -> serde_json::Value {
    let mut status = serde_json::json!({
        "merged_into": row
            .status
            .merged_into
            .iter()
            .map(|b| b.as_str())
            .collect::<Vec<_>>(),
        "dirty": row.status.dirty,
    });
    if let Some(missing) = row.status.missing {
        status["missing"] = serde_json::json!(match missing {
            MissingPiece::Worktree => "worktree",
            MissingPiece::Branch => "branch",
        });
    }
    serde_json::json!({
        "id": format!("w-{}", row.birth.0),
        "branch": row.branch.as_str(),
        "status": status,
        "change": match row.change {
            Some((files, insertions, deletions)) => serde_json::json!({
                "files": files,
                "insertions": insertions,
                "deletions": deletions,
            }),
            None => serde_json::Value::Null,
        },
        "ahead": row.ahead,
        "touched": row.touched,
    })
}

/// The listing's text face: the sketch's six columns, each padded to
/// its widest cell, two spaces between columns.
fn render_list_table(rows: &[ListRow]) -> String {
    let now = db::now_epoch();
    let header = [
        "WORKSPACE",
        "BRANCH",
        "STATUS",
        "CHANGE",
        "AHEAD",
        "TOUCHED",
    ]
    .map(str::to_string);
    let body: Vec<[String; 6]> = rows.iter().map(|row| list_cells(row, now)).collect();
    let mut widths = [0usize; 6];
    for (column, head) in header.iter().enumerate() {
        widths[column] = head.len();
    }
    for cells in &body {
        for (column, cell) in cells.iter().enumerate() {
            widths[column] = widths[column].max(cell.len());
        }
    }
    let mut lines = vec![pad_row(&header, &widths)];
    lines.extend(body.iter().map(|cells| pad_row(cells, &widths)));
    lines.join("\n")
}

/// One row's six cells: the id the fold numbers workspaces by (the
/// birth record's position), the status of its own where a piece is
/// missing, and "-" where no number exists.
fn list_cells(row: &ListRow, now: u64) -> [String; 6] {
    let status = match row.status.missing {
        Some(MissingPiece::Worktree) => "worktree missing".to_string(),
        Some(MissingPiece::Branch) => "branch missing".to_string(),
        None => {
            let merged = match row.status.merged_into.first() {
                Some(first) => format!("merged into {}", first.as_str()),
                None => "unmerged".to_string(),
            };
            if row.status.dirty {
                format!("{merged}, dirty")
            } else {
                merged
            }
        }
    };
    let change = match row.change {
        Some((files, insertions, deletions)) => format!("{files}f +{insertions} -{deletions}"),
        None => "-".to_string(),
    };
    [
        format!("w-{}", row.birth.0),
        row.branch.as_str().to_string(),
        status,
        change,
        row.ahead
            .map(|ahead| ahead.to_string())
            .unwrap_or_else(|| "-".to_string()),
        row.touched
            .map(|touched| commit_age(now, touched))
            .unwrap_or_else(|| "-".to_string()),
    ]
}

/// One padded row: every column to its width, the last bare.
fn pad_row(cells: &[String], widths: &[usize]) -> String {
    cells
        .iter()
        .enumerate()
        .map(|(column, cell)| {
            if column + 1 == cells.len() {
                cell.clone()
            } else {
                format!("{cell:<width$}  ", width = widths[column])
            }
        })
        .collect()
}

/// The commit's age in the app's units, the sketch's vocabulary:
/// minutes under an hour, hours under two days, else days.
fn commit_age(now_epoch: u64, then: u64) -> String {
    let minutes = (now_epoch.saturating_sub(then) + 30) / 60;
    if minutes < 60 {
        format!("{minutes}m")
    } else if minutes < 48 * 60 {
        format!("{}h", (minutes + 30) / 60)
    } else {
        format!("{}d", (minutes + 720) / 1440)
    }
}

/// `sac workspace clean`: the ordered cleanup through the server's
/// effects. The reply names every verified step; a failure names the
/// step that stopped, and retry is the recovery path. The branch's
/// disposition rides the reply: removed when merged (into what),
/// otherwise why it stands.
pub(super) fn workspace_clean_door(
    cli: &Cli,
    destination: &Destination,
    path: &Path,
) -> Result<String, Fail> {
    let url = server_url(destination, "workspace clean")?;
    let reply = clean_one(&url, path).map_err(Fail::Client)?;
    let record = reply.records.first().map(|r| r.seq);
    let fact = if reply.cleaned {
        format!(
            "cleaned - record {}",
            record.map(|r| r.to_string()).unwrap_or_default()
        )
    } else if reply.already_cleaned {
        "already cleaned - every resource verified absent, no new record".to_string()
    } else {
        "completed".to_string()
    };
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "cleaned": reply.cleaned,
            "already_cleaned": reply.already_cleaned,
            "workspace": reply.workspace,
            "branch": reply.branch,
            "verified": reply.verified,
            "records": reply.records,
        }))
        .expect("plain data"));
    }
    let branch_line = reply.branch.map(|outcome| {
        let name = &reply.workspace.branch;
        match (outcome.deleted, outcome.kept) {
            (true, _) => format!(
                "branch {} removed (merged into {})",
                name,
                outcome
                    .merged_into
                    .first()
                    .map(String::as_str)
                    .unwrap_or("?")
            ),
            (false, Some(client::BranchKept::Unmerged)) => {
                format!("branch {name} kept (unmerged)")
            }
            (false, Some(client::BranchKept::Missing)) => format!("branch {name} already absent"),
            (false, None) => format!("branch {name} kept"),
        }
    });
    Ok(format!(
        "workspace {} ({}, branch {})\n  {}\n{}{}",
        reply.workspace.path,
        reply.workspace.seq,
        reply.workspace.branch,
        fact,
        branch_line
            .map(|line| format!("  {line}\n"))
            .unwrap_or_default(),
        reply
            .verified
            .iter()
            .map(|step| format!("  - {step}"))
            .collect::<Vec<_>>()
            .join("\n")
    ))
}

/// The one clean ask the named door and the sweep both make: the same
/// handshake and the same wire call, so the sweep acts through
/// exactly the door `sac workspace clean <path>` acts through.
fn clean_one(url: &str, path: &Path) -> Result<CleanReply, client::ClientFail> {
    client::handshake(url);
    client::clean_workspace(url, path)
}

// -- the sweep ------------------------------------------------------------
// `sac workspace clean --merged`: the list's own read decides, and
// each qualifying workspace is cleaned through the existing clean
// door. No new server route, no new event: the sweep is a client-side
// loop over the standing rows, one clean per workspace, a refusal
// reported and the round continued.

/// One line's outcome word and the reason in its parens. The preview
/// reads them off the row's own faces; the act reads the cleaned and
/// refused rows off the door's replies, and the left rows off the
/// same faces the preview used.
struct SweepLine {
    word: &'static str,
    id: String,
    branch: String,
    reason: String,
    /// The Cleaned record's position, when this call's clean landed one
    record: Option<usize>,
    /// The refusal's code, when the door refused this workspace
    code: Option<String>,
}

impl SweepLine {
    /// A row the sweep leaves: the missing piece named, or the
    /// unmerged truth, or the dirt beside the merge - always the
    /// operator's reason, never an act.
    fn left(row: &ListRow) -> Self {
        SweepLine {
            word: "left",
            id: format!("w-{}", row.birth.0),
            branch: row.branch.as_str().to_string(),
            reason: left_reason(row),
            record: None,
            code: None,
        }
    }

    /// A row the sweep would clean: merged per the rule and no dirt
    /// guard. The reason is the merge named with what a clean removes.
    fn planned(row: &ListRow) -> Self {
        SweepLine {
            word: "cleaned",
            id: format!("w-{}", row.birth.0),
            branch: row.branch.as_str().to_string(),
            reason: format!(
                "merged into {}; worktree and branch removed",
                first_into(row),
            ),
            record: None,
            code: None,
        }
    }

    /// A workspace the door refused mid-round: the refusal's own words
    /// in the line, its code in the json face. The sweep goes on.
    fn refused(row: &ListRow, code: &str, detail: &str) -> Self {
        SweepLine {
            word: "refused",
            id: format!("w-{}", row.birth.0),
            branch: row.branch.as_str().to_string(),
            reason: refusal_words(detail),
            record: None,
            code: Some(code.to_string()),
        }
    }

    /// A clean that acted: the word is the sketch's, and the reason is
    /// the reply's own truth - the branch disposition when this call
    /// cleaned, the already-recorded fact when the record stood.
    fn cleaned(row: &ListRow, reply: &CleanReply) -> Self {
        SweepLine {
            word: "cleaned",
            id: format!("w-{}", row.birth.0),
            branch: row.branch.as_str().to_string(),
            reason: cleaned_reason(reply),
            record: reply.records.first().map(|r| r.seq),
            code: None,
        }
    }

    /// The row's json face: the same words the text line carries, the
    /// record and the refusal code riding where they exist.
    fn value(&self) -> serde_json::Value {
        let mut value = serde_json::json!({
            "id": self.id,
            "branch": self.branch,
            "outcome": self.word,
            "reason": self.reason,
        });
        if let Some(record) = self.record {
            value["record"] = serde_json::json!(record);
        }
        if let Some(code) = &self.code {
            value["code"] = serde_json::json!(code);
        }
        value
    }
}

/// The first branch the merged rule named: every standing row without
/// a missing piece carries one.
fn first_into(row: &ListRow) -> &str {
    row.status
        .merged_into
        .first()
        .map(|b| b.as_str())
        .unwrap_or("?")
}

/// The refusal's own words for one outcome line: the client carries
/// the detail JSON-encoded, flattened and bounded. A detail that
/// survived whole parses back to its string; a truncated one is no
/// longer valid JSON, so its opening quote drops and the string
/// escapes the encoding added undo. Either way the words collapse
/// into the one line.
fn refusal_words(detail: &str) -> String {
    if let Ok(text) = serde_json::from_str::<String>(detail) {
        return collapse(&text);
    }
    let bare = detail.strip_prefix('"').unwrap_or(detail);
    let unescaped = bare
        .replace("\\n", " ")
        .replace("\\t", " ")
        .replace("\\\"", "\"");
    collapse(&unescaped)
}

/// Why the sweep leaves a row alone, read off the row's own faces: a
/// missing piece is the human's inspection (the door's answer cannot
/// be read off it - a lost worktree may still be git's stale
/// registration, and the sweep never guesses), the dirt names itself
/// beside the merge, and an unmerged branch says so plainly.
fn left_reason(row: &ListRow) -> String {
    match row.status.missing {
        Some(MissingPiece::Worktree) => "worktree missing".into(),
        Some(MissingPiece::Branch) => "branch missing".into(),
        None => {
            let merged = if row.status.merged_into.is_empty() {
                "unmerged".to_string()
            } else {
                format!("merged into {}", first_into(row))
            };
            if row.status.dirty {
                format!("{merged}; dirty")
            } else {
                merged
            }
        }
    }
}

/// The sweep's qualifying rule: merged per the rule and no preserved
/// work. The dirty refusal would fire at the door anyway, so the row
/// shows as left with its reason before any --yes; a row with a
/// missing piece never qualifies, for its face cannot promise what
/// the door will answer.
fn qualifies(row: &ListRow) -> bool {
    row.status.missing.is_none() && !row.status.dirty && !row.status.merged_into.is_empty()
}

/// The reason a reply names for a cleaned workspace: the branch's
/// disposition when this call acted, the already-recorded fact when
/// the record stood before it (a concurrent Cleaned landing still
/// discloses the branch act this call ran).
fn cleaned_reason(reply: &CleanReply) -> String {
    if reply.cleaned {
        let name = &reply.workspace.branch;
        match reply.branch.as_ref() {
            Some(outcome) if outcome.deleted => format!(
                "merged into {}; worktree and branch removed",
                outcome
                    .merged_into
                    .first()
                    .map(String::as_str)
                    .unwrap_or("?")
            ),
            Some(outcome) if outcome.kept == Some(client::BranchKept::Unmerged) => {
                format!("worktree removed; branch {name} kept (unmerged)")
            }
            Some(outcome) if outcome.kept == Some(client::BranchKept::Missing) => {
                format!("worktree removed; branch {name} already absent")
            }
            Some(_) => format!("worktree removed; branch {name} kept"),
            None => "worktree removed".into(),
        }
    } else {
        match reply.branch.as_ref() {
            Some(outcome) if outcome.deleted => format!(
                "already recorded; branch {} removed (merged into {})",
                reply.workspace.branch,
                outcome
                    .merged_into
                    .first()
                    .map(String::as_str)
                    .unwrap_or("?")
            ),
            _ => "already recorded; resources verified absent".into(),
        }
    }
}

/// The sweep's text face, the sketch's shape: the outcome word, the
/// workspace, the branch, the reason in parens. The outcome word sits
/// in its own fixed column - the vocabulary's widest, "cleaned" - so
/// the lines align whatever the round's mix; the other columns pad to
/// their widest cell, one space after the word, two between the rest.
fn render_sweep(lines: &[SweepLine]) -> String {
    let word_width = "cleaned".len();
    let id_width = lines.iter().map(|l| l.id.len()).max().unwrap_or(0);
    let branch_width = lines.iter().map(|l| l.branch.len()).max().unwrap_or(0);
    lines
        .iter()
        .map(|line| {
            format!(
                "{:<ww$} {:<iw$}  {:<bw$}  ({})",
                line.word,
                line.id,
                line.branch,
                line.reason,
                ww = word_width,
                iw = id_width,
                bw = branch_width,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `sac workspace clean --merged`: the standing rows decide. Without
/// --yes the round is a preview - the sketch's cleaned/left lines, no
/// clean asked, nothing records. --yes acts: one clean per qualifying
/// workspace through the existing clean door, in birth order, a
/// refused clean printed as its outcome line with the refusal's own
/// words and the round continued; a transport failure (the server
/// gone, a broken wire) stops the round, for no further clean can
/// answer.
pub(super) fn workspace_sweep_door(
    cli: &Cli,
    destination: &Destination,
    yes: bool,
) -> Result<String, Fail> {
    // the list's rig: the same fold and the same git read the list
    // door makes, so the sweep never decides from a different world
    let loadout = loadout_of(destination)?;
    let world = match loadout.state {
        LoadState::Full(world) => world,
        LoadState::Degraded(reason) => return Err(Fail::Degraded(reason)),
    };
    let root = resolve_repo(cli)?;
    let rows = saccade::workspace::workspace_list(&world, &root)?;
    let cursor = loadout.rows.last().map(|r| r.seq);

    let mut lines: Vec<SweepLine> = Vec::new();
    if !yes {
        lines.extend(rows.iter().map(|row| {
            if qualifies(row) {
                SweepLine::planned(row)
            } else {
                SweepLine::left(row)
            }
        }));
    } else {
        let url = server_url(destination, "workspace clean --merged")?;
        for row in &rows {
            if !qualifies(row) {
                lines.push(SweepLine::left(row));
                continue;
            }
            match clean_one(&url, row.worktree.as_path()) {
                Ok(reply) => lines.push(SweepLine::cleaned(row, &reply)),
                Err(client::ClientFail::Refused { code, detail, .. }) => {
                    lines.push(SweepLine::refused(row, &code, &detail));
                }
                Err(e) => return Err(Fail::Client(e)),
            }
        }
    }

    if cli.json {
        let rows: Vec<serde_json::Value> = lines.iter().map(|line| line.value()).collect();
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "acted": yes,
            "workspaces": rows,
            "cursor": cursor,
        }))
        .expect("lines are plain data"));
    }
    Ok(render_sweep(&lines))
}

#[cfg(test)]
mod test {
    use super::*;
    use saccade::host::workspace::RowStatus;
    use saccade::record::types::pointers::{GitBranch, GitCommit, WorktreePath};

    /// One row as the facts would carry it, no git behind it.
    fn row(
        birth: usize,
        branch: &str,
        status: RowStatus,
        change: Option<(u64, u64, u64)>,
        ahead: Option<u64>,
        touched: Option<u64>,
    ) -> ListRow {
        ListRow {
            birth: RecordId(birth),
            base: GitCommit::new("aaaa1111".to_string()).unwrap(),
            branch: GitBranch::new(branch.to_string()).unwrap(),
            worktree: WorktreePath::new("/wt/x".into()).unwrap(),
            status,
            change,
            ahead,
            touched,
        }
    }

    fn status(missing: Option<MissingPiece>, merged_into: &[&str], dirty: bool) -> RowStatus {
        RowStatus {
            missing,
            merged_into: merged_into
                .iter()
                .map(|n| GitBranch::new(n.to_string()).unwrap())
                .collect(),
            dirty,
        }
    }

    /// The age carries the app's units: minutes under an hour, hours
    /// under two days, else days - the sketch's 20m, 2h, 1d.
    #[test]
    fn the_age_renders_minutes_hours_days() {
        let now = 1_800_000_000;
        assert_eq!(commit_age(now, now - 10), "0m");
        assert_eq!(commit_age(now, now - 40), "1m");
        assert_eq!(commit_age(now, now - 20 * 60), "20m");
        assert_eq!(commit_age(now, now - 59 * 60), "59m");
        assert_eq!(commit_age(now, now - 2 * 3600), "2h");
        assert_eq!(commit_age(now, now - 25 * 3600), "25h");
        assert_eq!(commit_age(now, now - 47 * 3600), "47h");
        assert_eq!(commit_age(now, now - 49 * 3600), "2d");
    }

    /// The row faces: the merged status names the first candidate and
    /// carries the dirty word, a missing piece is the status of its
    /// own, the numbers it cannot have render as "-", and the json
    /// carries the same fields with the rule's full answer.
    #[test]
    fn the_rows_render_the_sketch_s_columns_and_their_missing_faces() {
        let now = 1_800_000_000;
        let merged_dirty = row(
            41,
            "t60-phase3b",
            status(None, &["main", "backup"], true),
            Some((16, 33, 256)),
            Some(4),
            Some(now - 2 * 3600),
        );
        let cells = list_cells(&merged_dirty, now);
        assert_eq!(cells[0], "w-41");
        assert_eq!(cells[1], "t60-phase3b");
        assert_eq!(cells[2], "merged into main, dirty");
        assert_eq!(cells[3], "16f +33 -256");
        assert_eq!(cells[4], "4");
        assert_eq!(cells[5], "2h");

        // the text names the first candidate; the json carries all
        let value = list_row_value(&merged_dirty);
        assert_eq!(value["id"], "w-41");
        assert_eq!(value["branch"], "t60-phase3b");
        assert_eq!(
            value["status"]["merged_into"],
            serde_json::json!(["main", "backup"])
        );
        assert_eq!(value["status"]["dirty"], true);
        assert_eq!(
            value["change"],
            serde_json::json!({"files": 16, "insertions": 33, "deletions": 256})
        );
        assert_eq!(value["ahead"], 4);
        assert_eq!(value["touched"], now - 2 * 3600);
        assert_eq!(
            list_row_value(&row(
                42,
                "plain",
                status(None, &[], false),
                Some((0, 0, 0)),
                Some(0),
                Some(now)
            ))["status"],
            serde_json::json!({"merged_into": [], "dirty": false})
        );

        // a missing worktree: the status of its own, numbers alive
        let gone = row(
            43,
            "gone-line",
            status(Some(MissingPiece::Worktree), &["main"], false),
            Some((1, 2, 0)),
            Some(1),
            Some(now - 5 * 60),
        );
        let cells = list_cells(&gone, now);
        assert_eq!(cells[2], "worktree missing");
        assert_eq!(cells[3], "1f +2 -0");
        let value = list_row_value(&gone);
        assert_eq!(value["status"]["missing"], "worktree");
        assert_eq!(value["status"]["merged_into"], serde_json::json!(["main"]));

        // a missing branch: no numbers anywhere, "-" across the face
        let lost = row(
            44,
            "lost-line",
            status(Some(MissingPiece::Branch), &[], false),
            None,
            None,
            None,
        );
        let cells = list_cells(&lost, now);
        assert_eq!(cells[2], "branch missing");
        assert_eq!(cells[3], "-");
        assert_eq!(cells[4], "-");
        assert_eq!(cells[5], "-");
        let value = list_row_value(&lost);
        assert_eq!(value["status"]["missing"], "branch");
        assert_eq!(value["change"], serde_json::Value::Null);
        assert_eq!(value["ahead"], serde_json::Value::Null);
        assert_eq!(value["touched"], serde_json::Value::Null);
    }

    /// The table: the header row, the columns padded to their widest
    /// cell, two spaces between, the last column bare.
    #[test]
    fn the_table_pads_its_columns_and_carries_the_header() {
        let now = 1_800_000_000;
        let rows = vec![
            row(
                41,
                "t60-phase3b",
                status(None, &["main"], false),
                Some((16, 33, 256)),
                Some(4),
                Some(now - 2 * 3600),
            ),
            row(
                42,
                "rewrite-lexer-errors",
                status(None, &[], true),
                Some((5, 1904, 12)),
                Some(6),
                Some(now - 5 * 60),
            ),
        ];
        let table = render_list_table(&rows);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 3, "{table}");
        assert!(lines[0].starts_with("WORKSPACE  BRANCH"), "{table}");
        assert!(lines[0].ends_with("TOUCHED"), "{table}");
        // every column lands at the same offset on every row, the
        // last column bare
        let status_at = lines[0].find("STATUS").unwrap();
        assert!(
            lines[1][status_at..].starts_with("merged into main"),
            "{table}"
        );
        assert!(
            lines[2][status_at..].starts_with("unmerged, dirty"),
            "{table}"
        );
        let change_at = lines[0].find("CHANGE").unwrap();
        assert_eq!(lines[1].find("16f +33 -256").unwrap(), change_at, "{table}");
        assert_eq!(lines[2].find("5f +1904 -12").unwrap(), change_at, "{table}");
        let ahead_at = lines[0].find("AHEAD").unwrap();
        assert!(lines[1][ahead_at..].starts_with("4"), "{table}");
        assert!(lines[2][ahead_at..].starts_with("6"), "{table}");
        assert_eq!(lines[1].len(), lines[2].len(), "{table}");
    }

    /// The sweep's verdict reads the row's own faces: merged and clean
    /// qualifies, the dirt names itself beside the merge, an unmerged
    /// branch says so, and a missing piece is the human's inspection -
    /// the door's answer cannot be read off that face, so the sweep
    /// never guesses it.
    #[test]
    fn the_sweep_qualifies_merged_and_clean_and_leaves_the_rest_with_reasons() {
        let merged = row(
            41,
            "t60-phase3b",
            status(None, &["main"], false),
            None,
            None,
            None,
        );
        assert!(qualifies(&merged));
        assert_eq!(SweepLine::planned(&merged).word, "cleaned");
        assert_eq!(
            SweepLine::planned(&merged).reason,
            "merged into main; worktree and branch removed"
        );

        // the dirty refusal would fire anyway: left, with the dirt
        // named beside the merge or beside the unmerged truth
        let merged_dirty = row(
            48,
            "errors",
            status(None, &["main"], true),
            None,
            None,
            None,
        );
        assert!(!qualifies(&merged_dirty));
        assert_eq!(
            SweepLine::left(&merged_dirty).reason,
            "merged into main; dirty"
        );
        let unmerged_dirty = row(48, "errors", status(None, &[], true), None, None, None);
        assert_eq!(SweepLine::left(&unmerged_dirty).reason, "unmerged; dirty");

        // an unmerged branch is left plainly
        let unmerged = row(
            47,
            "rewrite-lexer",
            status(None, &[], false),
            None,
            None,
            None,
        );
        assert!(!qualifies(&unmerged));
        assert_eq!(SweepLine::left(&unmerged).reason, "unmerged");

        // a missing piece never qualifies: the sweep cannot promise
        // what the door will answer, so the human inspects first
        let gone_wt = row(
            43,
            "gone",
            status(Some(MissingPiece::Worktree), &["main"], false),
            None,
            None,
            None,
        );
        assert!(!qualifies(&gone_wt));
        assert_eq!(SweepLine::left(&gone_wt).reason, "worktree missing");
        let lost_branch = row(
            44,
            "lost",
            status(Some(MissingPiece::Branch), &[], false),
            None,
            None,
            None,
        );
        assert!(!qualifies(&lost_branch));
        assert_eq!(SweepLine::left(&lost_branch).reason, "branch missing");
    }

    /// The sweep's text face is the sketch's shape, byte for byte: the
    /// outcome word, the workspace, the branch, the reason in parens,
    /// each column padded to its widest cell. A refused clean joins
    /// the same grid with the refusal's own words.
    #[test]
    fn the_sweep_lines_render_the_sketch_s_shape() {
        let lines = vec![
            SweepLine::planned(&row(
                41,
                "t60-phase3b",
                status(None, &["main"], false),
                None,
                None,
                None,
            )),
            SweepLine::planned(&row(
                44,
                "rewrite-parser-split",
                status(None, &["rewrite"], false),
                None,
                None,
                None,
            )),
            SweepLine::left(&row(
                47,
                "rewrite-lexer",
                status(None, &[], false),
                None,
                None,
                None,
            )),
            SweepLine::left(&row(
                48,
                "rewrite-lexer-errors",
                status(None, &[], true),
                None,
                None,
                None,
            )),
        ];
        assert_eq!(
            render_sweep(&lines),
            "cleaned w-41  t60-phase3b           (merged into main; worktree and branch removed)\n\
             cleaned w-44  rewrite-parser-split  (merged into rewrite; worktree and branch removed)\n\
             left    w-47  rewrite-lexer         (unmerged)\n\
             left    w-48  rewrite-lexer-errors  (unmerged; dirty)"
        );

        // a refused clean prints as its outcome line with the
        // refusal's words, whitespace collapsed into the one line
        let refused = SweepLine::refused(
            &row(
                52,
                "t60-phase3b",
                status(None, &["main"], false),
                None,
                None,
                None,
            ),
            "workspace_effect",
            "git branch -d t60-phase3b refused: error: the branch 't60-phase3b' is not\nfully merged.",
        );
        assert_eq!(refused.word, "refused");
        assert_eq!(refused.code.as_deref(), Some("workspace_effect"));
        assert_eq!(
            refused.value(),
            serde_json::json!({
                "id": "w-52",
                "branch": "t60-phase3b",
                "outcome": "refused",
                "reason": "git branch -d t60-phase3b refused: error: the branch 't60-phase3b' is not fully merged.",
                "code": "workspace_effect",
            })
        );
        let mut grid = render_sweep(&lines);
        grid.push('\n');
        grid.push_str(&render_sweep(&[refused]));
        let refused_line = grid.lines().last().unwrap();
        assert!(refused_line.starts_with("refused w-52  "), "{refused_line}");
        assert!(refused_line.ends_with(
            "(git branch -d t60-phase3b refused: error: the branch 't60-phase3b' is not fully merged.)"
        ), "{refused_line}");

        // the wire carries the detail JSON-encoded: a string detail
        // loses its quoting before it renders
        let unwrapped = SweepLine::refused(
            &row(53, "gone", status(None, &["main"], false), None, None, None),
            "workspace_checkout_mismatch",
            serde_json::to_string(&serde_json::json!(
                "/wt/x holds a directory git does not register as this repo's worktree"
            ))
            .unwrap()
            .as_str(),
        );
        assert_eq!(
            unwrapped.reason,
            "/wt/x holds a directory git does not register as this repo's worktree"
        );
    }

    /// The act's reason is the reply's own truth: the disposition when
    /// the clean acted, the already-recorded fact (and any branch act
    /// the call still ran) when the record stood before it.
    #[test]
    fn the_act_line_reads_the_reply_s_own_truth() {
        let record = |seq: usize| StoredRecord {
            seq,
            event_time: 1,
            logged_time: 1,
            actor: "pi".into(),
            tier: "agent".into(),
            kind: "event".into(),
            payload: "{}".into(),
            scope: None,
        };
        let reply = |branch: Option<client::BranchOutcome>, cleaned: bool| CleanReply {
            cleaned,
            already_cleaned: !cleaned,
            workspace: client::WorkspaceBirth {
                seq: 7,
                base: "aaaa1111".into(),
                branch: "t60-phase3b".into(),
                path: "/wt/x".into(),
            },
            branch,
            verified: Vec::new(),
            records: if cleaned {
                vec![record(19)]
            } else {
                Vec::new()
            },
        };
        let row = row(
            41,
            "t60-phase3b",
            status(None, &["main"], false),
            None,
            None,
            None,
        );

        // the ordinary act: the merge named, the record's position
        // riding the line
        let merged_delete = reply(
            Some(client::BranchOutcome {
                deleted: true,
                merged_into: vec!["main".into()],
                kept: None,
            }),
            true,
        );
        let line = SweepLine::cleaned(&row, &merged_delete);
        assert_eq!(line.word, "cleaned");
        assert_eq!(line.record, Some(19));
        assert_eq!(line.reason, "merged into main; worktree and branch removed");
        assert_eq!(
            line.value(),
            serde_json::json!({
                "id": "w-41",
                "branch": "t60-phase3b",
                "outcome": "cleaned",
                "reason": "merged into main; worktree and branch removed",
                "record": 19,
            })
        );

        // drift: the rule named no candidate at the door - the words
        // follow the reply, never the plan
        let kept = reply(
            Some(client::BranchOutcome {
                deleted: false,
                merged_into: Vec::new(),
                kept: Some(client::BranchKept::Unmerged),
            }),
            true,
        );
        assert_eq!(
            SweepLine::cleaned(&row, &kept).reason,
            "worktree removed; branch t60-phase3b kept (unmerged)"
        );

        // the concurrent Cleaned: no duplicate record, the branch act
        // this call ran still disclosed
        let concurrent = reply(
            Some(client::BranchOutcome {
                deleted: true,
                merged_into: vec!["main".into()],
                kept: None,
            }),
            false,
        );
        let line = SweepLine::cleaned(&row, &concurrent);
        assert_eq!(line.record, None);
        assert_eq!(
            line.reason,
            "already recorded; branch t60-phase3b removed (merged into main)"
        );
        let concurrent_quiet = reply(None, false);
        assert_eq!(
            SweepLine::cleaned(&row, &concurrent_quiet).reason,
            "already recorded; resources verified absent"
        );
    }
}
