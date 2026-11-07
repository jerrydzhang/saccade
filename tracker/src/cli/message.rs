use super::task::parse_task_id;
use super::wait::resolve_incarnation;
use super::*;
use saccade::client;

/// The send door: a durable addressed message from this session's
/// registered incarnation. The record lands first; the submission
/// attempt is bounded and never waits on the recipient's work - the
/// sender chooses its own further waiting, never this verb.
pub(super) fn send_door(
    cli: &Cli,
    destination: &Destination,
    to: &str,
    reply: Option<&str>,
    task: Option<&str>,
    body: Option<String>,
    stdin: bool,
) -> Result<String, Fail> {
    let url = server_url(destination, "send")?;
    // the public tokens ride as typed: the server resolves them
    // against its own fold - one resolver, taught refusals
    let to_num = dense_token(to, "i-")?;
    let reply_num = match reply {
        Some(token) => Some(dense_token(token, "m-")?),
        None => None,
    };
    let task_num = match task {
        Some(token) => Some(parse_task_id(token)?.0),
        None => None,
    };
    let context = context_of(cli)?;
    identity_fallback_guard(cli, &context, attached_terminal())?;
    let body = Prose::new(body_source(body, stdin)?)?;
    client::handshake(&url);
    // success carries the host-accepted submission; a refused attempt
    // comes back naming the recorded m-N beside the evidence - the
    // record stands, the refusal explains it
    let sent = match client::send_message(&url, &context, to_num, reply_num, task_num, &body) {
        Ok(sent) => sent,
        Err(client::ClientFail::Refused {
            code,
            detail,
            message,
        }) => {
            let named = match message {
                Some(facts) => facts
                    .num
                    .map(|n| format!("m-{n}"))
                    .unwrap_or_else(|| format!("seq {}", facts.seq)),
                None => "its record".to_string(),
            };
            return Err(Fail::Usage(format!(
                "not sent: {named} was recorded and stays pending ({code}): {detail}"
            )));
        }
        Err(other) => return Err(Fail::Client(other)),
    };
    if cli.json {
        return Ok(serde_json::to_string_pretty(
            &serde_json::to_value(&sent).expect("the send reply is plain data"),
        )
        .expect("plain data"));
    }
    Ok(format!(
        "sent m-{} to i-{to_num} - accepted into their session's input queue",
        sent.message
            .num
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("(seq {})", sent.message.seq)),
    ))
}

/// The mailbox door: the messages an incarnation participates in, in
/// record order, with their delivery state - a descriptive read.
pub(super) fn messages_door(
    cli: &Cli,
    destination: &Destination,
    mailbox_token: Option<&str>,
) -> Result<String, Fail> {
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "messages needs the world; the log will not fold".into(),
        ));
    };
    let id = match mailbox_token {
        Some(token) => resolve_incarnation(&world, token)?,
        None => context_scope()?
            .and_then(|s| s.incarnation)
            .ok_or_else(|| {
                Fail::Usage(
                    "no mailbox declared: pass i-<n> or declare SACCADE_INCARNATION_SEQ".into(),
                )
            })?,
    };
    if !world.incarnations.contains_key(&id) {
        return Err(Fail::Reject(Reject::InvalidIncarnationId));
    }
    let lines: Vec<MessageLine> = mailbox(&world, id)
        .iter()
        .map(|m| message_line(&world, *m).expect("a resident message renders"))
        .collect();
    let num = world.incarnation_num(id);
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "incarnation": {"seq": id.0.0, "num": num},
            "messages": lines,
        }))
        .expect("plain data"));
    }
    let mut out = vec![match num {
        Some(n) => format!("i-{n} (birth seq {}) mailbox", id.0.0),
        None => format!("incarnation birth seq {} mailbox", id.0.0),
    }];
    out.push(
        "  submitted = accepted into the recipient's input queue; pending = recorded without a confirmed submission (evidence, never a delivery promise)".into(),
    );
    if lines.is_empty() {
        out.push("  no messages".into());
    }
    for line in lines {
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
        out.push(head);
        out.push(format!("      {}", line.body));
    }
    out.push(cursor_line(loadout.rows.last().map(|r| r.seq)));
    Ok(out.join("\n"))
}
