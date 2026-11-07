use super::task::{parse_task_id, record_line};
use super::*;
use super::{IncarnationId, MessageId};
use saccade::host_contract::HostContract;

pub(super) const WAIT_POLL: std::time::Duration = std::time::Duration::from_millis(250);
/// One arrival-wait iteration at the host: the door re-enters while
/// the overall bound allows.
pub(super) const ARRIVAL_ITERATION: std::time::Duration = std::time::Duration::from_secs(10);
/// The brief pause between host waits on a quiet answer.
pub(super) const ARRIVAL_PAUSE: std::time::Duration = std::time::Duration::from_millis(500);

pub(super) fn wait_door(
    cli: &Cli,
    destination: &Destination,
    target: &str,
    after: Option<u64>,
    until: Option<saccade::views::WaitUntil>,
    timeout: Option<u64>,
) -> Result<String, Fail> {
    // the incarnation form is the arrival wait: its own door, its own
    // flags - the t form keeps its required pair
    if target.trim_start().starts_with("i-") {
        if after.is_some() {
            return Err(Fail::Usage(
                "--after is the t-<n> wait's cursor; an i-<n> arrival wait takes --timeout, not --after".into(),
            ));
        }
        if until.is_some() {
            return Err(Fail::Usage(
                "--until is the t-<n> wait's mode; an i-<n> arrival wait takes --timeout, not --until".into(),
            ));
        }
        return wait_arrival_door(cli, destination, target, timeout);
    }
    let Some(after) = after else {
        return Err(Fail::Usage(
            "the t-<n> wait names its cursor: --after <seq> is required".into(),
        ));
    };
    let Some(until) = until else {
        return Err(Fail::Usage(
            "the t-<n> wait names its mode: --until activity or offer is required".into(),
        ));
    };
    let raw = wait_target(target)?;
    loop {
        let loadout = loadout_of(destination)?;
        let world = match loadout.state {
            LoadState::Full(world) => world,
            LoadState::Degraded(reason) => return Err(Fail::Degraded(reason)),
        };
        let target = Target::Task(raw);
        wait_target_exists(&world, &target)?;
        let matched: Vec<StoredRecord> = loadout
            .rows
            .iter()
            .filter(|r| r.seq > after as usize)
            .filter(|r| {
                let event =
                    wire::assemble(&r.kind, &r.payload).expect("a full fold assembled every row");
                saccade::views::wait_matches(&world, RecordId(r.seq), &event, &target, until)
            })
            .cloned()
            .collect();
        if !matched.is_empty() {
            return Ok(render_wait(
                cli,
                &matched,
                loadout.rows.last().map(|r| r.seq),
            ));
        }
        std::thread::sleep(WAIT_POLL);
    }
}

/// The arrival wait: block until a message addressed to the named
/// incarnation lands after this wait's entry. The host's wake is
/// advisory - every return (wake or quiet) re-reads the mailbox, so
/// a spurious wake costs one read and a real message is never
/// consumed unseen - and the entry cursor never advances through the
/// wait's own wakes, only through the mail it reported. The address
/// cross-check re-folds each pass: an incarnation whose recorded
/// address moved ends the wait with a disclosure, never a silent
/// watch of a stale terminal. Nothing is recorded; interruption ends
/// it.
pub(super) fn wait_arrival_door(
    cli: &Cli,
    destination: &Destination,
    token: &str,
    timeout: Option<u64>,
) -> Result<String, Fail> {
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "wait needs the world; the log will not fold".into(),
        ));
    };
    let id = resolve_incarnation(&world, token.trim())?;
    let num = world.incarnation_num(id).unwrap_or(id.0.0);
    let entry_address = world.incarnations[&id]
        .incarnation
        .address()
        .map(str::to_string);
    let Some(address) = entry_address else {
        return Err(Fail::Usage(format!(
            "i-{num} carries no host address: pre-B sessions take no arrival wait; the sanctioned migration is exit + resume"
        )));
    };
    // the cursor names the mail that already exists at entry; only
    // mail born after it releases the wait
    let entry_cursor = world
        .messages
        .iter()
        .filter(|(_, mctx)| mctx.message.to == id)
        .map(|(mid, _)| mid.0.0)
        .max()
        .unwrap_or(0);
    let host = saccade::herdr::HerdrCore::resolved().map_err(|e| Fail::Usage(e.to_string()))?;
    let deadline =
        timeout.map(|secs| std::time::Instant::now() + std::time::Duration::from_secs(secs));
    loop {
        // the iteration never overshoots the overall bound: a short
        // --timeout ends quiet at its own edge, not at the iteration's
        let iteration = match deadline {
            Some(at) => at
                .saturating_duration_since(std::time::Instant::now())
                .min(ARRIVAL_ITERATION),
            None => ARRIVAL_ITERATION,
        };
        if iteration.is_zero() {
            return Ok(render_arrival_quiet(cli, num));
        }
        let wake = host.wait(
            &saccade::host_contract::HostAddress(address.clone()),
            iteration,
        );
        let loadout = loadout_of(destination)?;
        let world = match loadout.state {
            LoadState::Full(world) => world,
            LoadState::Degraded(reason) => return Err(Fail::Degraded(reason)),
        };
        // the address cross-check: a moved association means this
        // watch names a terminal the incarnation no longer holds
        let current = world.incarnations[&id].incarnation.address();
        if current != Some(address.as_str()) {
            let _ = wake;
            return Err(Fail::Usage(format!(
                "i-{num}'s recorded address moved while waiting; this watch ended without reading new mail - wait again"
            )));
        }
        let arrived: Vec<MessageId> = world
            .messages
            .iter()
            .filter(|(mid, mctx)| mctx.message.to == id && mid.0.0 > entry_cursor)
            .map(|(mid, _)| *mid)
            .collect();
        if !arrived.is_empty() {
            let lines: Vec<MessageLine> = arrived
                .iter()
                .map(|m| message_line(&world, *m).expect("a resident message renders"))
                .collect();
            let cursor = loadout.rows.last().map(|r| r.seq);
            return Ok(render_arrival(cli, &lines, cursor));
        }
        if let Some(deadline) = deadline
            && std::time::Instant::now() >= deadline
        {
            return Ok(render_arrival_quiet(cli, num));
        }
        if matches!(wake, saccade::host_contract::WaitWake::Quiet) {
            std::thread::sleep(ARRIVAL_PAUSE);
        }
    }
}

/// The wait door's raw token: t-N as itself.
pub(super) fn wait_target(token: &str) -> Result<TaskId, Fail> {
    if token.starts_with('#') {
        return Err(Fail::Usage(format!(
            "'{token}' is not a wait target; drop the '#': tasks are t-<n>"
        )));
    }
    parse_task_id(token)
}

pub(super) fn wait_target_exists(world: &saccade::World, target: &Target) -> Result<(), Fail> {
    match target {
        Target::Task(task) => {
            if world.tasks.get(task.0).is_none() {
                return Err(Fail::Usage(format!(
                    "no task t-{}; the fold holds {} tasks",
                    task.0,
                    world.tasks.len()
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn render_wait(cli: &Cli, matched: &[StoredRecord], cursor: Option<usize>) -> String {
    if cli.json {
        return serde_json::to_string_pretty(&serde_json::json!({
            "records": matched,
            "cursor": cursor,
        }))
        .expect("records are plain data");
    }
    let mut lines: Vec<String> = matched.iter().map(record_line).collect();
    lines.push(cursor_line(cursor));
    lines.join("\n")
}

/// The arrival wait's answer: the mail that landed, each line naming
/// its state - pending is a recorded attempt without a confirmed
/// submission, evidence the caller interprets.
pub(super) fn render_arrival(cli: &Cli, lines: &[MessageLine], cursor: Option<usize>) -> String {
    if cli.json {
        return serde_json::to_string_pretty(&serde_json::json!({
            "messages": lines,
            "cursor": cursor,
        }))
        .expect("message lines are plain data");
    }
    let mut out: Vec<String> = lines
        .iter()
        .map(|line| {
            let mut head = format!(
                "m-{}  i-{} -> i-{} | {}",
                line.num, line.from, line.to, line.state
            );
            if let Some(reply) = line.reply_to {
                head.push_str(&format!(" | replies to m-{reply}"));
            }
            if let Some(task) = line.task {
                head.push_str(&format!(" | context t-{task}"));
            }
            head.push('\n');
            head.push_str(&format!("      {}", line.body));
            head
        })
        .collect();
    out.push(cursor_line(cursor));
    out.join("\n")
}

/// The overall bound's quiet answer: no mail landed, nothing consumed.
pub(super) fn render_arrival_quiet(cli: &Cli, num: usize) -> String {
    if cli.json {
        return serde_json::to_string_pretty(&serde_json::json!({
            "quiet": true,
            "incarnation": num,
        }))
        .expect("plain data");
    }
    format!(
        "i-{num}: quiet - no message addressed to it landed within the bound; nothing was consumed"
    )
}

/// The incarnation a public i-N names: dense birth order against the fold.
pub(super) fn resolve_incarnation(world: &World, token: &str) -> Result<IncarnationId, Fail> {
    let n = token.strip_prefix("i-").ok_or_else(|| {
        Fail::Usage(format!(
            "'{token}' is not an incarnation reference (expected i-<n>)"
        ))
    })?;
    let n: usize = n
        .parse()
        .map_err(|_| Fail::Usage(format!("'{token}' is not an incarnation reference")))?;
    world.incarnation_at_num(n).ok_or_else(|| {
        Fail::Usage(format!(
            "i-{n} names no incarnation; this fold holds {} incarnations",
            world.incarnations.len()
        ))
    })
}
