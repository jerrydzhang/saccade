use super::*;

/// The naming door: set, change, or clear what an incarnation is
/// called. The record lands first; then, when the record carries the
/// incarnation's terminal address, one name-label ask on it - a
/// refused label is disclosed, never a failed naming.
pub(super) fn name_door(
    cli: &Cli,
    destination: &Destination,
    name: Option<&str>,
    incarnation: Option<&str>,
    clear: bool,
) -> Result<String, Fail> {
    let name = match (name, clear) {
        (_, true) => None,
        (Some(name), false) => Some(saccade::record::objects::incarnation::IncarnationName::new(
            name.to_string(),
        )?),
        (None, false) => {
            return Err(Fail::Usage(
                "name what? pass the name, or --clear to remove it".into(),
            ));
        }
    };
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "the naming door needs the world; the log will not fold".into(),
        ));
    };
    let context = context_of(cli)?;
    let target = match incarnation {
        Some(token) => resolve_incarnation(&world, token)?,
        None => context
            .scope
            .as_ref()
            .and_then(|s| s.incarnation)
            .ok_or_else(|| {
                Fail::Usage(
                    "name names this session's own incarnation (SACCADE_INCARNATION_SEQ) \
                     or the one --incarnation i-<n> names"
                        .into(),
                )
            })?,
    };
    identity_fallback_guard(cli, &context, attached_terminal())?;
    let command = Command::NameIncarnation {
        incarnation: target,
        name: name.clone(),
    };
    let stored = match destination {
        Destination::Local { db } => {
            let now = cli.at.unwrap_or_else(db::now_epoch);
            let mut conn = db::open(db).map_err(Fail::Db)?;
            match db::record(&mut conn, &context, command.clone(), now) {
                Ok((stored, _)) => stored,
                Err(ExecuteFail::Reject(reject)) => return Err(refused(&conn, &command, reject)),
                Err(other) => return Err(Fail::from(other)),
            }
        }
        Destination::Server { url } => {
            client::handshake(url);
            client::send(url, &context, command, cli.at)
                .map_err(Fail::Client)?
                .records
        }
    };
    let number = world
        .incarnation_num(target)
        .map(|n| format!("i-{n}"))
        .unwrap_or_else(|| format!("seq {}", target.0.0));
    // a superseded holder's recorded address is another session's
    // terminal now: no label is asked there
    let address = world
        .incarnations
        .get(&target)
        .and_then(|ctx| ctx.incarnation.address())
        .filter(|_| !world.superseded.contains_key(&target));
    let refusal = address.and_then(|address| {
        saccade::launch::label_name(address, name.as_ref().map(|n| n.as_str()))
    });
    let shown = name.as_ref().map(|n| n.as_str().to_string());
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "records": stored,
            "incarnation": number,
            "name": shown,
            "label": address.map(|address| serde_json::json!({
                "address": address,
                "refusal": refusal,
            })),
        }))
        .expect("plain data"));
    }
    let mut lines: Vec<String> = stored.iter().map(super::task::record_line).collect();
    lines.push(match &shown {
        Some(name) => format!("{number} is named {name}"),
        None => format!("{number}'s name is cleared; it shows as {number}"),
    });
    lines.push(match (address, refusal) {
        (None, _) => "  no live terminal address is recorded: no Herdr label was asked".into(),
        (Some(address), None) if shown.is_none() => {
            format!("  the Herdr name label is removed from {address}")
        }
        (Some(address), None) => format!("  the Herdr name label is set on {address}"),
        (Some(address), Some(evidence)) => format!(
            "  the Herdr name label on {address} was refused: {evidence} - the name is recorded"
        ),
    });
    Ok(lines.join("\n"))
}
