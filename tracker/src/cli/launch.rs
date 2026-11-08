use super::wait::{WAIT_POLL, resolve_incarnation};
use super::workspace::render_birth;
use super::*;
use saccade::IncarnationId;
use saccade::client;
use saccade::host;
use saccade::host_contract::HostContract;
use std::io::IsTerminal;
use std::path::Path;
use std::process::ExitCode;

/// The selected launch harness: the two compiled-in kinds, spoken as
/// the wire's own tokens; anything else refuses with the values named.
pub(super) fn parse_harness(value: &str) -> Result<saccade::Harness, Fail> {
    match value {
        "pi" => Ok(saccade::Harness::Pi),
        "claude-code" => Ok(saccade::Harness::ClaudeCode),
        other => Err(Fail::Usage(format!(
            "'{other}' is not a harness (expected pi or claude-code)"
        ))),
    }
}

/// The typed harness a door carries: None means no explicit choice
/// (fresh launches compose Pi, resumes run the recorded kind).
pub(super) fn harness_choice(value: &Option<String>) -> Result<Option<saccade::Harness>, Fail> {
    value.as_deref().map(parse_harness).transpose()
}

/// The reply's placement line: how to look, descriptive never pushy;
/// a refused declaration discloses instead of hinting.
pub(super) fn placement_lines(
    terminal: &str,
    space: &Option<String>,
    place_refusal: &Option<String>,
) -> String {
    match (space, place_refusal) {
        (Some(_), Some(evidence)) => {
            format!("  placement was refused: {evidence} - the session runs unlabeled")
        }
        (Some(_), None) => format!(
            "  look for it in the herdr app's sidebar, or attach from a terminal: hc attach {terminal}"
        ),
        (None, _) => format!(
            "  unplaced: no Herdr space was declared, attach from a terminal: hc attach {terminal}"
        ),
    }
}

/// The launch door: the composed native session in an explicit
/// existing workspace, task-free. A fresh launch registers a new
/// incarnation at session start; --resume continues a saved one,
/// validated against its recorded native session before anything
/// launches.
pub(super) fn launch_door(
    cli: &Cli,
    destination: &Destination,
    workspace: Option<&std::path::Path>,
    resume: Option<&str>,
    harness: Option<saccade::Harness>,
    prompt: &[String],
) -> Result<String, Fail> {
    let submitted = submitted_launch(cli, destination, workspace, resume, harness, prompt)?;
    let launch_face = submitted_face(&submitted);
    if cli.json {
        return Ok(serde_json::to_string_pretty(&launch_face).expect("plain data"));
    }
    // the resume's association moves when the session attaches; a
    // fresh birth carries its address from registration itself
    let association = if submitted.resume.is_some() {
        "  the association moves when the session attaches; `sac messages` reads the session's mailbox once it has one"
    } else {
        "  the launch is recorded with its host address; `sac messages` reads the session's mailbox once it has one"
    };
    Ok(format!(
        "integrated {h} launching at {ws} - address {t}\n{place}\n{association}",
        h = launch_harness_name(&submitted),
        ws = submitted.workspace,
        t = submitted.terminal,
        place = placement_lines(
            &submitted.terminal,
            &submitted.space,
            &submitted.place_refusal
        ),
    ))
}

/// The launch reply's json face: the submitted facts, the fresh
/// birth when this launch carries one - the exact (harness, native
/// id) pair, never an address rediscovery - and the resume target's
/// birth record when the arm resumed.
pub(super) fn submitted_face(submitted: &saccade::launch::Submitted) -> serde_json::Value {
    let fresh = submitted.fresh.as_ref();
    serde_json::json!({
        "workspace": submitted.workspace,
        "terminal": submitted.terminal,
        "space": submitted.space,
        "tab": submitted.tab,
        "place_refusal": submitted.place_refusal,
        "harness": launch_harness_name(submitted),
        "native_id": fresh.map(|f| f.native_id.clone()),
        "resumed": submitted.resume.as_ref().map(|r| r.id.0.0),
    })
}

/// The harness name a launch face names: the fresh selection, or the
/// resume target's recorded kind.
pub(super) fn launch_harness_name(submitted: &saccade::launch::Submitted) -> &str {
    submitted
        .fresh
        .as_ref()
        .map(|f| f.harness.as_str())
        .or_else(|| submitted.resume.as_ref().map(|r| r.harness.as_str()))
        .unwrap_or("pi")
}

/// One shared host submission: the folded world, the resolved host,
/// and the composition through the one lib module, then the calling
/// door's own registration. Every CLI arm - host, here, ask - rides
/// this head, so no arm grows its own launch dialect.
pub(super) fn submitted_launch(
    cli: &Cli,
    destination: &Destination,
    workspace: Option<&std::path::Path>,
    resume: Option<&str>,
    harness: Option<saccade::Harness>,
    prompt: &[String],
) -> Result<saccade::launch::Submitted, Fail> {
    let url = server_url(destination, "launch")?;
    let repo = resolve_repo(cli)?;
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "the launch needs the world; the log will not fold".into(),
        ));
    };
    let arm = launch_arm(&world, workspace, resume, harness, prompt)?;
    let composed = saccade::launch::compose(saccade::launch::LaunchRequest {
        world: &world,
        repo: &repo,
        url: url.clone(),
        exe: std::env::current_exe().ok(),
        arm,
        // the host and ask doors declare placement; only --here stays
        // unplaced
        declare_placement: true,
    })?;
    let submitted = saccade::launch::submit(composed).map_err(Fail::from)?;
    // the door births the fresh identity itself, before the harness's
    // first save; a resume's association moves when the session
    // attaches
    if let Some(fresh) = &submitted.fresh {
        client::register_incarnation(
            &url,
            &fresh.native_id,
            &submitted.terminal,
            false,
            fresh.harness,
        )
        .map_err(Fail::Client)?;
    }
    Ok(submitted)
}

/// The typed arm a door selected: the resume token resolves against
/// the fold here, so the composition speaks incarnations, never
/// token strings.
pub(super) fn launch_arm<'a>(
    world: &saccade::World,
    workspace: Option<&'a std::path::Path>,
    resume: Option<&str>,
    harness: Option<saccade::Harness>,
    prompt: &'a [String],
) -> Result<saccade::launch::LaunchArm<'a>, Fail> {
    match (workspace, resume) {
        (Some(dir), None) => Ok(saccade::launch::LaunchArm::Fresh {
            workspace: dir,
            prompt,
            harness: harness.unwrap_or(saccade::Harness::Pi),
        }),
        (None, Some(token)) => {
            let id = resolve_incarnation(world, token)?;
            Ok(saccade::launch::LaunchArm::Resume {
                id,
                prompt,
                expected: harness,
            })
        }
        _ => Err(Fail::Usage(
            "launch needs an explicit choice: --workspace <dir> for a fresh session, \
             --resume i-<n> for a saved one"
                .into(),
        )),
    }
}

/// The same-terminal launch door: an ordinary host launch - the same
/// composition and submission the host door makes, address recorded,
/// declared unplaced - then the caller's terminal attaches to it as a
/// viewer. No address is adopted: the session carries its own launched
/// terminal, and detaching (ctrl+\) ends the attach and leaves the
/// session running. The attach's own exit is the command's; the
/// session's own ending is never invented from it.
pub(super) fn launch_here_door(cli: &Cli) -> ExitCode {
    let Cmd::Launch {
        workspace,
        resume,
        new,
        ask,
        harness,
        prompt,
        ..
    } = &cli.command
    else {
        unreachable!("the here door dispatches only for launches");
    };
    let harness = match harness_choice(harness) {
        Ok(harness) => harness,
        Err(fail) => return fail_exit(cli, &fail),
    };
    // --ask stays a host-terminal launch: peer creation wants its own
    // terminal; --new is not one of the same-terminal compositions
    if ask.is_some() {
        return fail_exit(
            cli,
            &Fail::Usage(
                "launch --here runs the composed session in the caller's own terminal; \
                 --ask stays a host-terminal launch because peer creation wants its own terminal"
                    .into(),
            ),
        );
    }
    if new.is_some() {
        return fail_exit(
            cli,
            &Fail::Usage(
                "launch --here composes with --workspace <dir> or --resume i-<n>; \
                 --new provisions through the server and launches through a host terminal"
                    .into(),
            ),
        );
    }
    match launch_here_run(
        cli,
        workspace.as_deref(),
        resume.as_deref(),
        harness,
        prompt,
    ) {
        Ok(outcome) => {
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "workspace": outcome.workspace,
                        "terminal": outcome.terminal,
                        "harness": outcome.harness,
                        "resumed": outcome.resumed_num,
                        "native_id": outcome.native_id,
                        "exit": outcome.attach.status(),
                    })
                );
            } else {
                println!(
                    "the {h} session runs at {ws} - address {t}\n  the attach ended (ctrl+\\ detaches); the session keeps running\n  reattach from any terminal: hc attach {t}",
                    h = outcome.harness,
                    ws = outcome.workspace,
                    t = outcome.terminal,
                );
            }
            match outcome.attach.status() {
                0 => ExitCode::SUCCESS,
                code => ExitCode::from(code as u8),
            }
        }
        Err(fail) => fail_exit(cli, &fail),
    }
}

/// What a here run ended with: the submitted launch's facts and the
/// attach's own exit - the session's status is never one of them.
pub(super) struct HereRun {
    workspace: String,
    terminal: String,
    harness: &'static str,
    resumed_num: Option<usize>,
    native_id: Option<String>,
    attach: host::ChildExit,
}

/// The same-terminal mechanics: the shared composition guards run
/// first (a refused launch never needs a terminal), then the real
/// stdin gate - the attach inherits this terminal - then the ordinary
/// host submission, the fresh birth or address move, and the
/// foreground attach. The caller's own host declaration never becomes
/// the session's address; it only names the pane the no-nested guard
/// refuses for.
pub(super) fn launch_here_run(
    cli: &Cli,
    workspace: Option<&std::path::Path>,
    resume: Option<&str>,
    harness: Option<saccade::Harness>,
    prompt: &[String],
) -> Result<HereRun, Fail> {
    let destination = resolve_destination(cli)?;
    let url = server_url(&destination, "launch")?;
    let repo = resolve_repo(cli)?;
    let loadout = loadout_of(&destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "the here door needs the world; the log will not fold".into(),
        ));
    };
    let resume_target = match resume {
        Some(token) => Some(resolve_incarnation(&world, token)?),
        None => None,
    };
    // no-nested safety: a foreground attach would route this pane's
    // input into the attached session, so a pane that already carries
    // a recorded incarnation never attaches here - the suspended
    // incarnation's input would drive the child. One exception, per
    // the same-pane ruling: the resume target returning to its own
    // recorded pane is not two.
    if let Ok(carried) = std::env::var("HC_TERMINAL_ID")
        && !carried.is_empty()
        && let Some((holder, _)) = world.incarnations.iter().find(|(holder, ctx)| {
            ctx.incarnation.address() == Some(carried.as_str()) && resume_target != Some(**holder)
        })
    {
        let holder_num = world
            .incarnation_num(*holder)
            .map(|n| format!("i-{n}"))
            .unwrap_or_else(|| format!("seq {}", holder.0.0));
        return Err(Fail::Usage(format!(
            "this terminal is already the address of {holder_num}: a foreground attach would route {holder_num}'s input into the attached session, and peers go through the host door"
        )));
    }
    let arm = launch_arm(&world, workspace, resume, harness, prompt)?;
    let composed = saccade::launch::compose(saccade::launch::LaunchRequest {
        world: &world,
        repo: &repo,
        url: url.clone(),
        exe: std::env::current_exe().ok(),
        arm,
        // the invoking viewer's launch declares no placement: the new
        // terminal stays unplaced, reachable at its own host address
        declare_placement: false,
    })?;
    // the attach inherits this terminal's stdin: a pipe or file
    // carries no attach, and the door refuses before any launch
    if !std::io::stdin().is_terminal() {
        return Err(Fail::Usage(
            "launch --here attaches the session in this terminal: stdin must be a real terminal, and this one is not; run the launch without --here for a host terminal".into(),
        ));
    }
    let submitted = saccade::launch::submit(composed).map_err(Fail::from)?;
    // the door births the fresh identity itself, before the harness's
    // first save; a resume's association moves when the session
    // attaches - a launched terminal either way, never this pane
    if let Some(fresh) = &submitted.fresh {
        client::register_incarnation(
            &url,
            &fresh.native_id,
            &submitted.terminal,
            false,
            fresh.harness,
        )
        .map_err(Fail::Client)?;
    }
    // the foreground attach: this terminal views the launched session
    // until detach; the launch stands regardless of the attach's fate
    let host = saccade::herdr::HerdrCore::resolved().map_err(|e| Fail::Usage(e.to_string()))?;
    let attach = host
        .attach(&saccade::host_contract::HostAddress(submitted.terminal.clone()))
        .map_err(|e| {
            Fail::Usage(format!(
                "the launch stands at address {t} in {ws}; the attach did not carry: {e} - attach by hand: hc attach {t}",
                t = submitted.terminal,
                ws = submitted.workspace,
            ))
        })?;
    // the fold numbers are dense birth order: the snapshot's number
    // for the resume target is the number the move keeps
    let resumed_num = resume_target.and_then(|id| world.incarnation_num(id));
    Ok(HereRun {
        workspace: submitted.workspace,
        terminal: submitted.terminal,
        harness: match submitted
            .fresh
            .as_ref()
            .map(|f| f.harness)
            .or_else(|| submitted.resume.as_ref().map(|r| r.harness))
        {
            Some(saccade::Harness::ClaudeCode) => "claude-code",
            _ => "pi",
        },
        resumed_num,
        native_id: submitted.fresh.map(|f| f.native_id),
        attach,
    })
}

/// What an ask launch did: the newborn's address and the question's,
/// with the submission attempt's truth - a refused or unacknowledged
/// submission leaves the question recorded but undisclosed to the
/// newborn, and the reply says so.
pub(super) struct AskOutcome {
    workspace: String,
    terminal: String,
    space: Option<String>,
    tab: Option<String>,
    place_refusal: Option<String>,
    birth: usize,
    num: usize,
    /// the question's birth position when its record landed; a send
    /// whose transport refused carries the reference in the disclosure
    message_seq: Option<usize>,
    message_num: Option<usize>,
    creator: String,
    /// None means the queue confirmed the question's acceptance
    submission: Option<String>,
}

/// The ask door: cheap peer creation with an initial question. The
/// creator is a registered incarnation; the question is durable
/// addressed communication from its own scope - machine-supplied
/// origin on the record - never argv shaped into native human input.
/// The newborn is the exact birth the launch composed - the (harness,
/// native id) pair - never an address rediscovery.
// Native startup binds after taking raw input; terminal creation alone does not.
pub(super) const ASK_BIND_BOUND: std::time::Duration = std::time::Duration::from_secs(30);

/// The binding bound is 30 seconds in production and that is the only
/// production value. The test harness alone shortens it through
/// SACCADE_ASK_BIND_BOUND_SECS so the timeout path costs seconds, not
/// the production wait; nothing else reads that variable.
fn ask_bind_bound() -> std::time::Duration {
    match std::env::var("SACCADE_ASK_BIND_BOUND_SECS") {
        Ok(secs) => std::time::Duration::from_secs(secs.parse().unwrap_or(30)),
        Err(_) => ASK_BIND_BOUND,
    }
}

pub(super) fn wait_for_ask_binding(
    destination: &Destination,
    incarnation: IncarnationId,
) -> Result<(), Fail> {
    let bound = ask_bind_bound();
    let deadline = std::time::Instant::now() + bound;
    loop {
        let loadout = loadout_of(destination)?;
        let LoadState::Full(world) = loadout.state else {
            return Err(Fail::Degraded(
                "launch --ask needs the world; the log will not fold".into(),
            ));
        };
        let ctx = world
            .incarnations
            .get(&incarnation)
            .ok_or(Fail::Reject(Reject::InvalidIncarnationId))?;
        if ctx.incarnation.saved_session().is_some() {
            return Ok(());
        }
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return Err(Fail::Usage(format!(
                "native startup did not bind its saved-session locator within {} seconds",
                bound.as_secs()
            )));
        }
        std::thread::sleep(left.min(WAIT_POLL));
    }
}

pub(super) fn ask_run(
    cli: &Cli,
    destination: &Destination,
    workspace: &std::path::Path,
    harness: Option<saccade::Harness>,
    question: &str,
) -> Result<AskOutcome, Fail> {
    let url = server_url(destination, "launch")?;
    let context = context_of(cli)?;
    let scope = context.scope.clone().ok_or_else(|| {
        Fail::Usage(
            "launch --ask sends as a registered incarnation: declare SACCADE_INCARNATION_SEQ (a human launch needs no question - pass the prompt positionally instead)".into(),
        )
    })?;
    let creator = scope.incarnation.ok_or_else(|| {
        Fail::Usage(
            "launch --ask sends as a registered incarnation: declare SACCADE_INCARNATION_SEQ"
                .into(),
        )
    })?;
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "launch --ask needs the world; the log will not fold".into(),
        ));
    };
    if !world.incarnations.contains_key(&creator) {
        return Err(Fail::Reject(Reject::InvalidIncarnationId));
    }
    // preflight before anything launches: the creator is a registered
    // NATIVE session (a historical attribution identity cannot create),
    // and the question is prose that can exist
    match &world.incarnations[&creator].incarnation {
        saccade::Incarnation::Native { .. } => {}
        saccade::Incarnation::Established { .. } => {
            return Err(Fail::Usage(
                "launch --ask creates from a live native session; this attribution identity carries no saved native session to create from".into(),
            ));
        }
    }
    let body = Prose::new(question.to_string())?;
    let creator_num = world
        .incarnation_num(creator)
        .map(|n| format!("i-{n}"))
        .unwrap_or_else(|| format!("seq {}", creator.0.0));
    // the launch births the newborn itself: identity and address exist
    // at launch time, and the question's recipient is never polled for
    let submitted = submitted_launch(cli, destination, Some(workspace), None, harness, &[])?;
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "launch --ask needs the world; the log will not fold".into(),
        ));
    };
    // the newborn is the exact birth this launch composed - the
    // (harness, native id) pair - never a first address match
    let Some(fresh) = &submitted.fresh else {
        return Err(Fail::Usage(
            "the ask launch submitted no fresh birth; nothing was sent".into(),
        ));
    };
    let newborn = world
        .incarnations
        .iter()
        .find(|(_, ctx)| match &ctx.incarnation {
            saccade::Incarnation::Native {
                native_id,
                harness,
                ..
            } => native_id.as_str() == fresh.native_id && *harness == fresh.harness,
            _ => false,
        })
        .map(|(id, _)| *id)
        .ok_or_else(|| {
            Fail::Usage(format!(
                "the launched session at {ws} carries no birth record ({n} under {h}, address {t}); nothing was sent - a later `sac send --to` reaches it by number",
                ws = submitted.workspace,
                n = fresh.native_id,
                h = fresh.harness.as_str(),
                t = submitted.terminal,
            ))
        })?;
    let num = world
        .incarnation_num(newborn)
        .expect("the fold just resolved this incarnation");
    let birth = newborn.0.0;
    wait_for_ask_binding(destination, newborn).map_err(|error| {
        Fail::Usage(format!(
            "i-{num} was launched at address {}; the initial question was not recorded or sent: {error}",
            submitted.terminal,
        ))
    })?;
    // the question records first, then rides the same attempt as any
    // send: a refused or unacknowledged submission is disclosed here,
    // never retried behind the door
    let (sent_message_seq, sent_message_num, submission) = match client::send_message(
        &url,
        &context,
        num,
        None,
        scope.task.map(|t| t.0),
        &body,
    ) {
        Ok(sent) => {
            let state = sent.submission.as_ref().map(|s| s.state.as_str());
            let disclosure = match state {
                Some("submitted") | None => None,
                Some(other) => Some(format!(
                    "its submission reads {other}: the newborn may not have received it - read the record with sac messages or send again"
                )),
            };
            (Some(sent.message.seq), sent.message.num, disclosure)
        }
        Err(client::ClientFail::Refused {
            code,
            detail,
            message,
        }) => {
            // the record landed and the refusal names it: the ask
            // face keeps the reference beside the disclosure
            (
                message.as_ref().map(|m| m.seq),
                message.as_ref().and_then(|m| m.num),
                Some(format!("its submission was refused ({code}): {detail}")),
            )
        }
        Err(other) => return Err(Fail::Client(other)),
    };
    Ok(AskOutcome {
        workspace: submitted.workspace,
        terminal: submitted.terminal,
        space: submitted.space,
        tab: submitted.tab,
        place_refusal: submitted.place_refusal,
        birth,
        num,
        message_seq: sent_message_seq,
        message_num: sent_message_num,
        creator: creator_num,
        submission,
    })
}

pub(super) fn ask_door(
    cli: &Cli,
    destination: &Destination,
    workspace: &std::path::Path,
    harness: Option<saccade::Harness>,
    question: &str,
) -> Result<String, Fail> {
    let outcome = ask_run(cli, destination, workspace, harness, question)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "workspace": outcome.workspace,
            "terminal": outcome.terminal,
            "space": outcome.space,
            "tab": outcome.tab,
            "place_refusal": outcome.place_refusal,
            "incarnation": {"seq": outcome.birth, "num": outcome.num},
            "question": {"num": outcome.message_num, "seq": outcome.message_seq},
            "submission": outcome.submission,
        }))
        .expect("plain data"));
    }
    let answer = outcome
        .message_num
        .map(|n| format!("m-{n}"))
        .unwrap_or_else(|| match outcome.message_seq {
            Some(seq) => format!("seq {seq}"),
            None => "(its record reference rides the submission disclosure)".into(),
        });
    let disclosure = outcome
        .submission
        .map(|line| format!("; {line}"))
        .unwrap_or_default();
    Ok(format!(
        "i-{num} registered at {workspace} (terminal {terminal}); your question went as {answer} from {creator}; they receive it through their session's input queue{disclosure}\n{place}",
        num = outcome.num,
        workspace = outcome.workspace,
        terminal = outcome.terminal,
        creator = outcome.creator,
        place = placement_lines(&outcome.terminal, &outcome.space, &outcome.place_refusal),
    ))
}

// -- the task-free workspace controller's CLI face -----------------------

/// Provision a NEW task-free workspace through the server's Git
/// effects; the birth record is the workspace's identity.
pub(super) fn provision_new(
    destination: &Destination,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<client::WorkspaceBirth, Fail> {
    let url = server_url(destination, "workspace new")?;
    client::handshake(&url);
    let reply = client::create_workspace(&url, label, base).map_err(Fail::Client)?;
    Ok(reply.workspace)
}

pub(super) fn launch_failed(standing: &str, error: Fail) -> Fail {
    Fail::Usage(format!("{standing}\nthe launch failed: {error}"))
}

/// The explicit-new launch: provision through the server's effects,
/// then hand the fresh directory to the ordinary doors - fresh launch
/// or ask, exactly the existing composition, under the selected
/// harness. A launch failure leaves the creation fact true.
pub(super) fn launch_new_door(
    cli: &Cli,
    destination: &Destination,
    label: Option<&str>,
    base: Option<&str>,
    ask: Option<&str>,
    harness: Option<saccade::Harness>,
    prompt: &[String],
) -> Result<String, Fail> {
    let birth = provision_new(destination, label, base)?;
    let path = birth.path.clone();
    let provisioned = render_birth(&birth);
    if cli.json {
        let launch = match ask {
            Some(question) => {
                match ask_run(cli, destination, Path::new(&path), harness, question) {
                    Ok(outcome) => serde_json::json!({
                        "workspace": outcome.workspace,
                        "terminal": outcome.terminal,
                        "space": outcome.space,
                        "tab": outcome.tab,
                        "place_refusal": outcome.place_refusal,
                        "incarnation": {"seq": outcome.birth, "num": outcome.num},
                        "question": {"num": outcome.message_num, "seq": outcome.message_seq},
                    }),
                    Err(e) => return Err(launch_failed(&provisioned, e)),
                }
            }
            None => match submitted_launch(
                cli,
                destination,
                Some(Path::new(&path)),
                None,
                harness,
                prompt,
            ) {
                Ok(launch) => submitted_face(&launch),
                Err(e) => return Err(launch_failed(&provisioned, e)),
            },
        };
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "provisioned": birth,
            "launch": launch,
        }))
        .expect("plain data"));
    }
    let door = match ask {
        Some(question) => match ask_door(cli, destination, Path::new(&path), harness, question) {
            Ok(door) => door,
            Err(e) => return Err(launch_failed(&provisioned, e)),
        },
        None => match submitted_launch(
            cli,
            destination,
            Some(Path::new(&path)),
            None,
            harness,
            prompt,
        ) {
            Ok(launch) => format!(
                "integrated {h} launching at {ws} - address {t}\n{place}\n  the launch is recorded with its host address; `sac messages` reads the session's mailbox once it has one",
                h = launch_harness_name(&launch),
                ws = launch.workspace,
                t = launch.terminal,
                place = placement_lines(&launch.terminal, &launch.space, &launch.place_refusal),
            ),
            Err(e) => return Err(launch_failed(&provisioned, e)),
        },
    };
    Ok(format!("{}\n{door}", render_birth(&birth)))
}
