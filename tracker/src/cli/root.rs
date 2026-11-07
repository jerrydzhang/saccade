use super::*;

/// `sac workspace root bind`: the human's explicit association of the
/// primary checkout with the manual root space's terminal. One host
/// ask on this pane's terminal, the exact live pair recorded; no
/// sweep, no cwd inference. The store enforces the human tier.
pub(super) fn root_bind_door(cli: &Cli, destination: &Destination) -> Result<String, Fail> {
    let url = server_url(destination, "workspace root bind")?;
    let carried = std::env::var("HC_TERMINAL_ID")
        .ok()
        .filter(|c| !c.is_empty())
        .ok_or_else(|| {
            Fail::Usage(
                "this pane declares no host terminal (HC_TERMINAL_ID absent): run the bind inside the terminal standing in the chosen manual root space".into(),
            )
        })?;
    let host = saccade::herdr::HerdrCore::resolved().map_err(|e| Fail::Usage(e.to_string()))?;
    let placed = host
        .placement_of(&saccade::host_contract::HostAddress(carried))
        .ok_or_else(|| {
            Fail::Usage(
                "the current terminal carries no live standing placement: select the chosen manual root space in the herdr app first, then bind".into(),
            )
        })?;
    let tab = placed.tab.ok_or_else(|| {
        Fail::Usage(
            "the current terminal names a space but no tab: the bind records the exact pair".into(),
        )
    })?;
    let context = context_of(cli)?;
    let record_seq = client::send(
        &url,
        &context,
        saccade::Command::BindRootSpace {
            space: placed.space.clone(),
            tab: tab.clone(),
        },
        None,
    )
    .map_err(Fail::Client)?;
    let record = record_seq.records.first().map(|r| r.seq);
    let line = format!(
        "root space bound - {space} / {tab}{record}",
        space = placed.space,
        record = record.map(|r| format!(" (record {r})")).unwrap_or_default(),
    );
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "bound": true,
            "space": placed.space,
            "tab": tab,
            "records": record_seq.records,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "{line}\n  launches at the primary checkout declare this pair; bind again in another space to re-point, or clear: sac workspace root clear"
    ))
}

/// `sac workspace root show`: the standing binding as the record holds
/// it - a read, no host ask, never an existence claim.
pub(super) fn root_show_door(cli: &Cli, destination: &Destination) -> Result<String, Fail> {
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "root show needs the world; the log will not fold".into(),
        ));
    };
    let binding = world.root_space.as_ref();
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "bound": binding.is_some(),
            "space": binding.map(|b| b.space.clone()),
            "tab": binding.map(|b| b.tab.clone()),
        }))
        .expect("plain data"));
    }
    Ok(match binding {
        Some(b) => format!("root space: {} / {}", b.space, b.tab),
        None => "no root space is bound: root launches run unplaced".into(),
    })
}

/// `sac workspace root clear`: remove the binding; root launches run
/// unplaced until a later bind. Human-only at the store; history
/// stays in the log.
pub(super) fn root_clear_door(cli: &Cli, destination: &Destination) -> Result<String, Fail> {
    let url = server_url(destination, "workspace root clear")?;
    let context = context_of(cli)?;
    let sent = client::send(&url, &context, saccade::Command::ClearRootSpace {}, None)
        .map_err(Fail::Client)?;
    let record = sent.records.first().map(|r| r.seq);
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "cleared": true,
            "records": sent.records,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "root space cleared{record} - root launches run unplaced until a later bind",
        record = record.map(|r| format!(" (record {r})")).unwrap_or_default(),
    ))
}
