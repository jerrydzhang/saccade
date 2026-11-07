use super::launch::{
    launch_failed, launch_harness_name, placement_lines, provision_new, submitted_launch,
};
use super::wait::resolve_incarnation;
use super::*;
use std::path::Path;

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

/// `sac workspace switch`: the stopped-session change through the
/// server, then an optional resume through the ordinary door. The
/// change record is the whole effect; the resume is a separate launch
/// whose failure leaves that fact standing.
pub(super) fn workspace_switch_door(
    cli: &Cli,
    destination: &Destination,
    token: &str,
    to: &Path,
    resume: bool,
    prompt: &[String],
) -> Result<String, Fail> {
    let url = server_url(destination, "workspace switch")?;
    client::handshake(&url);
    // resolve the incarnation token against the fold first, so the
    // refusal names the fold's extent; the wire's law is unchanged
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "switch needs the world; the log will not fold".into(),
        ));
    };
    let id = resolve_incarnation(&world, token)?;
    let num = world
        .incarnation_num(id)
        .ok_or(Fail::Reject(Reject::InvalidIncarnationId))?;
    let reply = client::change_workspace(&url, num, to).map_err(Fail::Client)?;
    let record = reply.records.first().map(|r| r.seq);
    let changed = format!(
        "i-{num} workspace changed - {from} -> {to}{record}",
        num = reply.num,
        from = reply.from,
        to = reply.to,
        record = record.map(|r| format!(" (record {r})")).unwrap_or_default(),
    );
    if resume {
        // the change fact already stands; a failed launch says so
        let launch = match submitted_launch(cli, destination, None, Some(token), None, prompt) {
            Ok(launch) => launch,
            Err(e) => return Err(launch_failed(&changed, e)),
        };
        let resumed = serde_json::json!({
            "workspace": launch.workspace,
            "terminal": launch.terminal,
            "space": launch.space,
            "tab": launch.tab,
            "place_refusal": launch.place_refusal,
        });
        if cli.json {
            return Ok(serde_json::to_string_pretty(&serde_json::json!({
                "changed": true,
                "incarnation": reply.incarnation,
                "num": reply.num,
                "from": reply.from,
                "to": reply.to,
                "records": reply.records,
                "resumed": resumed,
            }))
            .expect("plain data"));
        }
        return Ok(format!(
            "{changed}\nintegrated {h} launching at {ws} - terminal {t}\n{place}\n  the resumed session carries the same native id and history",
            h = launch_harness_name(&launch),
            ws = launch.workspace,
            t = launch.terminal,
            place = placement_lines(&launch.terminal, &launch.space, &launch.place_refusal),
        ));
    }
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "changed": true,
            "incarnation": reply.incarnation,
            "num": reply.num,
            "from": reply.from,
            "to": reply.to,
            "records": reply.records,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "{changed}\n  the session was stopped; resume it there with `sac launch --resume i-{}`",
        reply.num
    ))
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
        WorkspaceVerb::Switch {
            incarnation,
            to,
            resume,
            prompt,
        } => workspace_switch_door(cli, destination, incarnation, to, *resume, prompt),
        WorkspaceVerb::Clean { path } => workspace_clean_door(cli, destination, path),
        WorkspaceVerb::Restore { path } => workspace_restore_door(cli, destination, path),
        WorkspaceVerb::Reconcile { apply } => workspace_reconcile_door(cli, destination, *apply),
        WorkspaceVerb::Root { verb } => match verb {
            RootVerb::Bind => root_bind_door(cli, destination),
            RootVerb::Show => root_show_door(cli, destination),
            RootVerb::Clear => root_clear_door(cli, destination),
        },
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

/// `sac workspace restore`: checkout-only restoration; execution stays
/// a separate explicit launch or manual resume.
pub(super) fn workspace_restore_door(
    cli: &Cli,
    destination: &Destination,
    path: &Path,
) -> Result<String, Fail> {
    let url = server_url(destination, "workspace restore")?;
    client::handshake(&url);
    let reply = client::restore_workspace(&url, path).map_err(Fail::Client)?;
    let record = reply.records.first().map(|r| r.seq);
    let fact = if reply.already_restored {
        "already restored - the matching worktree stands Available, no new record".to_string()
    } else if reply.reused {
        "restored - the matching worktree stood and was reused as-is, local changes preserved"
            .to_string()
    } else {
        format!(
            "restored - checkout established at branch {}'s current local tip - record {}",
            reply.workspace.branch,
            record.map(|r| r.to_string()).unwrap_or_default()
        )
    };
    let tip = reply
        .tip
        .as_deref()
        .map(|t| format!("\n  branch tip {t}"))
        .unwrap_or_default();
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "restored": reply.restored,
            "already_restored": reply.already_restored,
            "reused": reply.reused,
            "tip": reply.tip,
            "workspace": reply.workspace,
            "records": reply.records,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "workspace {} ({}, branch {})\n  {}{}",
        reply.workspace.path, reply.workspace.seq, reply.workspace.branch, fact, tip
    ))
}

/// `sac workspace reconcile`: the one-time cutover verification.
/// Dry run by default; --apply records the verified Cleaned facts.
pub(super) fn workspace_reconcile_door(
    cli: &Cli,
    destination: &Destination,
    apply: bool,
) -> Result<String, Fail> {
    let url = server_url(destination, "workspace reconcile")?;
    client::handshake(&url);
    let reply = client::reconcile_workspaces(&url, apply).map_err(Fail::Client)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "applied": reply.applied,
            "cleaned": reply.cleaned,
            "preserved": reply.preserved,
            "conflicts": reply.conflicts,
            "records": reply.records,
        }))
        .expect("plain data"));
    }
    // the header says only what actually landed: an applied run over
    // conflicts records no facts and says so
    let mode = if !apply {
        "dry run - nothing recorded (pass --apply to record)".to_string()
    } else if !reply.records.is_empty() {
        format!(
            "applied - {} verified Cleaned fact(s) recorded",
            reply.records.len()
        )
    } else {
        "applied - no new facts recorded".to_string()
    };
    let section = |name: &str, rows: &[client::ReconcileEntry]| -> String {
        if rows.is_empty() {
            return String::new();
        }
        format!(
            "\n{}:\n{}",
            name,
            rows.iter()
                .map(|r| format!("  - record {} {} ({})", r.seq, r.path, r.reason))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    Ok(format!(
        "workspace reconciliation over every recorded birth\n  {mode}{}{}{}",
        section("cleaned (verified absent)", &reply.cleaned),
        section("preserved", &reply.preserved),
        section("conflicts (surfaced, never classified)", &reply.conflicts),
    ))
}
