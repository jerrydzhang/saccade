use super::launch::provision_new;
use super::*;
use std::path::Path;

use saccade::host::workspace::{ListRow, MissingPiece};

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
        WorkspaceVerb::Clean { path } => workspace_clean_door(cli, destination, path),
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
/// step that stopped, and retry is the recovery path.
pub(super) fn workspace_clean_door(
    cli: &Cli,
    destination: &Destination,
    path: &Path,
) -> Result<String, Fail> {
    let url = server_url(destination, "workspace clean")?;
    client::handshake(&url);
    let reply = client::clean_workspace(&url, path).map_err(Fail::Client)?;
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
            "verified": reply.verified,
            "records": reply.records,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "workspace {} ({}, branch {})\n  {}\n{}",
        reply.workspace.path,
        reply.workspace.seq,
        reply.workspace.branch,
        fact,
        reply
            .verified
            .iter()
            .map(|step| format!("  - {step}"))
            .collect::<Vec<_>>()
            .join("\n")
    ))
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
}
