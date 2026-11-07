use super::*;
use std::io::IsTerminal;

pub(super) fn read_only(cli: &Cli, loadout: &db::Loadout) -> Result<String, Fail> {
    let cursor = loadout.rows.last().map(|r| r.seq);
    match &cli.command {
        Cmd::Log => {
            if let LoadState::Degraded(reason) = &loadout.state {
                eprintln!("warning: world projection unavailable: {reason}");
            }
            Ok(render_log(cli, &loadout.rows, cursor))
        }
        Cmd::List { .. } => match &loadout.state {
            LoadState::Full(world) => Ok(render_tasks(cli, world, cursor)),
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason.clone())),
        },
        Cmd::Show { ids, stdin } => match &loadout.state {
            LoadState::Full(world) => {
                if *stdin {
                    return show_stdin(cli, world, &loadout.rows, ids, cursor);
                }
                if cli.json {
                    let blocks: Vec<serde_json::Value> = ids
                        .iter()
                        .map(|token| show_value(world, &loadout.rows, token))
                        .collect::<Result<_, _>>()?;
                    return Ok(serde_json::to_string_pretty(&serde_json::json!({
                        "blocks": blocks,
                        "cursor": cursor,
                    }))
                    .expect("blocks are plain data"));
                }
                let mut blocks = Vec::new();
                for token in ids {
                    blocks.push(render_id(world, &loadout.rows, token)?);
                }
                Ok(format!("{}\n{}", blocks.join("\n\n"), cursor_line(cursor)))
            }
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason.clone())),
        },
        _ => unreachable!("read_only reached from a mutating command"),
    }
}

pub(super) fn parse_task_id(token: &str) -> Result<TaskId, Fail> {
    let n = token.strip_prefix("t-").ok_or_else(|| {
        Fail::Usage(format!(
            "'{token}' is not a task id (expected t-<n>; only tasks exist){}",
            hashed_task_note(token)
        ))
    })?;
    let n: usize = n
        .parse()
        .map_err(|_| Fail::Usage(format!("'{token}' is not a task id")))?;
    Ok(TaskId(n))
}

/// The '#t-N' face taught at the parse door: '#' addresses records, a
/// task is addressed bare.
pub(super) fn hashed_task_note(token: &str) -> String {
    match token
        .strip_prefix("#t-")
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
    {
        Some(n) => format!("; drop the '#': the task is addressed as t-{n}"),
        None => String::new(),
    }
}

/// The reference a create reply names: the task whose birth record just
/// landed, resolved against the post-write fold.
pub(super) fn born_of(stored: &[StoredRecord], world: &World) -> Option<String> {
    let birth = stored.iter().find(|r| r.kind == "task_created")?;
    world
        .task_born_at(RecordId(birth.seq))
        .map(|id| format!("t-{}", id.0))
}

pub(super) fn render_records(cli: &Cli, stored: &[StoredRecord], born: Option<&str>) -> String {
    if cli.json {
        // the wire's reply shape: records always, the born task's token
        // only on a create
        let mut reply = serde_json::json!({"records": stored});
        if let Some(id) = born {
            reply["id"] = serde_json::json!(id);
        }
        return serde_json::to_string_pretty(&reply).expect("records are plain data");
    }
    let mut lines = Vec::with_capacity(stored.len() + 2);
    if let Some(id) = born {
        lines.push(id.to_string());
    }
    lines.extend(stored.iter().map(record_line));
    lines.join("\n")
}

pub(super) fn render_log(cli: &Cli, rows: &[StoredRecord], cursor: Option<usize>) -> String {
    if cli.json {
        let out: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| serde_json::to_value(r).expect("records are plain data"))
            .collect();
        return serde_json::to_string_pretty(&serde_json::json!({
            "records": out,
            "cursor": cursor,
        }))
        .expect("records are plain data");
    }
    let mut lines: Vec<String> = rows.iter().map(record_line).collect();
    lines.push(cursor_line(cursor));
    lines.join("\n")
}

/// One log row: the record's own position as its provenance - a seq
/// is never dressed as the c-N only a comment birth earns.
pub(super) fn record_line(r: &StoredRecord) -> String {
    format!(
        "seq {} {} {}/{} et={} lt={} {}",
        r.seq, r.kind, r.actor, r.tier, r.event_time, r.logged_time, r.payload
    )
}

pub(super) fn render_tasks(cli: &Cli, world: &World, cursor: Option<usize>) -> String {
    let views: Vec<TaskView> = (0..world.tasks.len())
        .map(|i| task_view(world, TaskId(i)))
        .collect::<Option<_>>()
        .expect("indices come from the vec itself");
    // the digest question is standing work; history lives in log and show
    let views: Vec<_> = views
        .into_iter()
        .filter(|v| matches!(v.state, "open" | "offered"))
        .collect();

    if cli.json {
        let rows: Vec<serde_json::Value> = views
            .iter()
            .map(|v| {
                serde_json::json!({
                    "id": v.id,
                    "state": v.state,
                    "name": v.name,
                    "owner": v.owner,
                })
            })
            .collect();
        return serde_json::to_string_pretty(&serde_json::json!({
            "tasks": rows,
            "cursor": cursor,
        }))
        .expect("views are plain data");
    }

    views
        .iter()
        .map(|v| {
            format!(
                "{}\t{}\t{}\t{}",
                v.id,
                v.state,
                v.name,
                v.owner.clone().unwrap_or_else(|| "-".into()),
            )
        })
        .chain(std::iter::once(cursor_line(cursor)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The show door as text: everything about the one task.
pub(super) fn render_show(world: &World, task_id: TaskId) -> Result<String, Fail> {
    let view =
        show_view(world, task_id).ok_or_else(|| Fail::Usage(format!("no task t-{}", task_id.0)))?;

    let mut out = vec![wrap(
        &view.name,
        WIDTH,
        &format!("{}  {}  ", view.id, view.state),
        "  ",
    )];
    if let Some(owner) = &view.owner {
        out.push(format!("owner  {owner}"));
    }
    if let Some(workspace) = &view.workspace {
        out.push(format!("workspace  {workspace}"));
    }
    if let Some(explanation) = &view.explanation {
        out.push("explanation".to_string());
        out.push(wrap(explanation, WIDTH, "  ", "  "));
    }
    Ok(out.join("\n"))
}

/// One message's block: the addressed endpoints, the delivery state -
/// bookkeeping, never a claim of reading or agreement - and the body.
pub(super) fn message_block(line: &MessageLine) -> Vec<String> {
    let mut head = format!(
        "  m-{}  i-{} -> i-{} | {}",
        line.num, line.from, line.to, line.state
    );
    if let Some(reply) = line.reply_to {
        head.push_str(&format!(" | replies to m-{reply}"));
    }
    if let Some(task) = line.task {
        head.push_str(&format!(" | context t-{task}"));
    }
    let mut out = vec![head, format!("    by {}", line.actor)];
    if !line.body.is_empty() {
        out.push(wrap(&line.body, WIDTH, "  ", "  "));
    }
    out
}

/// One id as show sees it: a whole task, a message's folded form,
/// or the raw event at a bare position.
#[derive(Clone, Copy, Debug)]
pub(super) enum ShowId {
    Thread(TaskId),
    /// m-<n>: the message's presentation, by public number
    Message(usize),
    /// bare <seq>: the raw event at that position
    Position(RecordId),
}

pub(super) fn parse_show_id(token: &str) -> Result<ShowId, Fail> {
    let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    let not_an_id = || {
        Fail::Usage(format!(
            "'{token}' is not an id (expected t-<n>, m-<n>, or a bare <seq>)"
        ))
    };
    if numeric(token.strip_prefix('#').unwrap_or_default()) {
        // the retired face teaches the door it may have meant: the raw
        // record at a position is the bare <seq>
        return Err(Fail::Usage(format!(
            "'{token}' is not an id (expected t-<n>, m-<n>, or a bare <seq>); \
             drop the '#': the raw record at a position is the bare <seq>"
        )));
    }
    if let Some(rest) = token.strip_prefix("m-").filter(|n| numeric(n)) {
        return Ok(ShowId::Message(rest.parse().map_err(|_| not_an_id())?));
    }
    if token.starts_with("t-") {
        return Ok(ShowId::Thread(parse_task_id(token)?));
    }
    if numeric(token) {
        return Ok(ShowId::Position(RecordId(
            token.parse().map_err(|_| not_an_id())?,
        )));
    }
    Err(not_an_id())
}

/// Render one id: a task whole, a message as its block, a task's
/// birth as the literal event, and every other record as the log
/// renders it, header and payload.
pub(super) fn render_id(
    world: &World,
    rows: &[db::StoredRecord],
    token: &str,
) -> Result<String, Fail> {
    let no_record = |typed: &str| {
        Fail::Usage(format!(
            "no record {typed}; the log holds {} records, 0 through {}",
            rows.len(),
            rows.len().saturating_sub(1)
        ))
    };
    match parse_show_id(token)? {
        ShowId::Thread(id) => render_show(world, id),
        ShowId::Message(num) => {
            let id = world.message_at_num(num).ok_or_else(|| {
                Fail::Usage(format!(
                    "m-{num} names no message; this fold holds {} messages",
                    world.messages.len()
                ))
            })?;
            let line = message_line(world, id)
                .ok_or_else(|| Fail::Usage(format!("m-{num} folds into no message")))?;
            Ok(message_block(&line).join("\n"))
        }
        ShowId::Position(id) => match rows.get(id.0) {
            Some(row) => Ok(raw_event(row)),
            None => Err(no_record(token)),
        },
    }
}

/// One id as the json face shows it: the text door's same walk with
/// bodies byte-exact - no wrap, no elision - and the fold's facts as
/// fields. One requested id, one array element; the caller assembles.
pub(super) fn show_value(
    world: &World,
    rows: &[db::StoredRecord],
    token: &str,
) -> Result<serde_json::Value, Fail> {
    match parse_show_id(token)? {
        ShowId::Thread(id) => {
            let view =
                show_view(world, id).ok_or_else(|| Fail::Usage(format!("no task t-{}", id.0)))?;
            Ok(serde_json::json!({
                "id": view.id,
                "state": view.state,
                "name": view.name,
                "owner": view.owner,
                "explanation": view.explanation,
            }))
        }
        ShowId::Message(num) => {
            let id = world.message_at_num(num).ok_or_else(|| {
                Fail::Usage(format!(
                    "m-{num} names no message; this fold holds {} messages",
                    world.messages.len()
                ))
            })?;
            let line = message_line(world, id)
                .ok_or_else(|| Fail::Usage(format!("m-{num} folds into no message")))?;
            Ok(serde_json::to_value(&line).expect("the message line is plain data"))
        }
        ShowId::Position(id) => match rows.get(id.0) {
            Some(row) => Ok(serde_json::to_value(row).expect("records are plain data")),
            None => Err(Fail::Usage(format!(
                "no record at {id:?}; the log holds {} records, 0 through {}",
                rows.len(),
                rows.len().saturating_sub(1)
            ))),
        },
    }
}

/// The raw event as itself: header and payload, no fold.
pub(super) fn raw_event(row: &db::StoredRecord) -> String {
    format!(
        "seq {}  {}  {}\n{}",
        row.seq, row.kind, row.actor, row.payload
    )
}

/// The piped face: one id per line, every failure a skip with a note,
/// never a broken chain. Empty input is silence. The json face skips
/// as an {"id", "error"} row, so the array holds every line's outcome.
pub(super) fn show_stdin(
    cli: &Cli,
    world: &World,
    rows: &[db::StoredRecord],
    ids: &[String],
    cursor: Option<usize>,
) -> Result<String, Fail> {
    if !ids.is_empty() {
        return Err(Fail::Usage("ids as arguments or --stdin, not both".into()));
    }
    if std::io::stdin().is_terminal() {
        return Err(Fail::Usage(
            "nothing piped: --stdin reads ids one per line".into(),
        ));
    }
    if cli.json {
        let mut blocks: Vec<serde_json::Value> = Vec::new();
        for line in std::io::stdin().lines() {
            let line = line.map_err(|e| Fail::Usage(format!("stdin: {e}")))?;
            let token = line.trim();
            if token.is_empty() {
                continue;
            }
            let block = show_value(world, rows, token)
                .unwrap_or_else(|f| serde_json::json!({"id": token, "error": f.to_string()}));
            blocks.push(block);
        }
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "blocks": blocks,
            "cursor": cursor,
        }))
        .expect("blocks are plain data"));
    }
    let mut blocks = Vec::new();
    for line in std::io::stdin().lines() {
        let line = line.map_err(|e| Fail::Usage(format!("stdin: {e}")))?;
        let token = line.trim();
        if token.is_empty() {
            continue;
        }
        match render_id(world, rows, token) {
            Ok(block) => blocks.push(block),
            Err(f) => blocks.push(format!("skipped '{token}': {f}")),
        }
    }
    Ok(format!("{}\n{}", blocks.join("\n\n"), cursor_line(cursor)))
}

/// The board search door: current boards by default, one line per
/// task; --history finds the revisions where terms appeared or
/// disappeared; a t-N or t-N@rK term is a reference search. The
/// boards live in the server's Git, so the search rides the server.
pub(super) fn search_door(
    cli: &Cli,
    destination: &Destination,
    terms: &[String],
    history: bool,
    within: &Option<String>,
) -> Result<String, Fail> {
    if terms.is_empty() {
        return Err(Fail::Usage(
            "search names its terms: sac search <terms...> [--history] [--in t-N@rA..rB]".into(),
        ));
    }
    let url = server_url(destination, "search")?;
    let reply =
        client::board_search(&url, terms, history, within.as_deref()).map_err(Fail::Client)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&reply).expect("results are plain data"));
    }
    if reply.lines.is_empty() {
        return Ok(format!("no boards match ({})", terms.join(" ")));
    }
    let lines: Vec<String> = reply
        .lines
        .iter()
        .map(|line| match line {
            saccade::serve::board_wire::SearchLine::Board {
                task,
                name,
                revision,
                excerpt,
            } => {
                format!("t-{task}@r{revision}  {name}  {excerpt}")
            }
            saccade::serve::board_wire::SearchLine::History {
                task,
                revision,
                event,
                term,
                detail,
            } => {
                let anchor = revision
                    .map(|r| format!("t-{task}@r{r}"))
                    .unwrap_or_else(|| format!("t-{task}"));
                let term = term.as_deref().unwrap_or("");
                let detail = detail.as_deref().unwrap_or("");
                if detail.is_empty() {
                    format!("{anchor} {event} {term}")
                } else {
                    format!("{anchor} {event} {term}: {detail}")
                }
            }
            saccade::serve::board_wire::SearchLine::Reference { line } => line.clone(),
        })
        .collect();
    Ok(lines.join("\n"))
}
