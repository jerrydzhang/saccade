use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use saccade::World;
use saccade::client;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::objects::task::TaskId;
use saccade::views::{ProposalView, TaskView, comment_thread, proposal_view, show_view, task_view};
use saccade::{
    ActorName, Command, CommentId, CommentKind, Context, GitCommit, ProposalAction, ProposalId,
    Prose, RecordId, Reject, Target, Tier,
};

#[derive(Parser)]
#[command(name = "sac", version, about = "Saccade: awesome issue tracker")]
struct Cli {
    /// Path to the event database (defaults to the repo's state root)
    #[arg(long, global = true, env = "SACCADE_DB")]
    db: Option<PathBuf>,

    /// The tracker's repo, when the working directory is not inside it
    #[arg(long, global = true, env = "SACCADE_REPO")]
    repo: Option<PathBuf>,

    /// Actor name recorded on events (default: SACCADE_ACTOR, or the account name)
    #[arg(long, global = true, env = "SACCADE_ACTOR")]
    actor: Option<String>,

    /// Machine-readable output
    #[arg(long, global = true)]
    json: bool,

    /// Event time override
    #[arg(long, global = true)]
    at: Option<u64>,

    /// Write directly to the db, bypassing the server (break-glass)
    #[arg(long, global = true)]
    offline: bool,

    /// Base URL of the running server for writes
    #[arg(
        long,
        global = true,
        env = "SACCADE_SERVER",
        default_value = "http://127.0.0.1:8811"
    )]
    server: String,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a new object
    Create {
        #[arg(value_enum)]
        kind: ObjKind,
        name: String,
        #[arg(long)]
        parent: Option<String>,
    },
    /// Claim an open task
    Claim { id: String },
    /// Deliver a claimed task's run, depositing the receipt accept reviews
    Done {
        id: String,
        #[arg(long)]
        receipt: String,
    },
    /// Human-only: drop a task (from open or done)
    Drop {
        id: String,
        #[arg(long)]
        note: String,
    },
    /// Human-only: release a claimed task back to open
    Release {
        id: String,
        #[arg(long)]
        note: String,
    },
    /// Propose a gated act for human acceptance
    Propose {
        #[arg(value_enum)]
        action: ProposeVerb,
        id: String,
        #[arg(long)]
        name: String,
    },
    /// Accept a proposal (bare seq, human) or a delivered task (t-N, the
    /// birth attribution or a human)
    Accept { id: String },
    /// Human-only: reject a proposal with a ruling note
    Reject {
        id: String,
        #[arg(long)]
        note: String,
    },
    /// Withdraw a proposal with a note
    Withdraw {
        id: String,
        #[arg(long)]
        note: String,
    },
    /// List objects (world projection)
    List {
        #[arg(value_enum, default_value_t = ObjKind::Task)]
        kind: ObjKind,
    },
    /// Print raw event records
    Log,
    /// List proposals (the ruling queue)
    Proposals,
    /// Attach a comment to a task (t-<n>) or reply to a comment (#<seq>)
    Comment {
        target: String,
        body: String,
        /// Fire a run on the task when it is free, queue while busy
        #[arg(long)]
        demand: bool,
    },
    /// Steer a task's live run at its next turn boundary; with no run
    /// living, the steer stands on the thread as intent
    Steer { id: String, body: String },
    /// Everything about one task: state, receipt, comment thread
    Show { id: String },
    /// Serve the read-only canvas over HTTP (127.0.0.1 by default)
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
        #[arg(long, default_value_t = 8811)]
        port: u16,
        /// The executor sessions run as (recorded on the bind, named in the reply door)
        #[arg(long, default_value = "pi")]
        executor: String,
    },
    /// Stop a task's active run: the event is the kill request
    Cancel { id: String },
    /// Record a task's current branch tip as its checkpoint
    Checkpoint { id: String },
    /// Block until the demand's run asks something of the waiter: settle,
    /// cancel, refusal, an answer with no run behind it, or a prompt
    /// awaiting an answer — never replies
    Wait {
        /// The demand to watch (c-<n>)
        id: String,
        /// Give up after this many seconds (default: wait forever)
        #[arg(long)]
        timeout: Option<u64>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ObjKind {
    Task,
}

#[derive(Clone, Copy, ValueEnum)]
enum ProposeVerb {
    Drop,
    Release,
}

fn main() -> ExitCode {
    // Piping a read into head/grep must end the process quietly; Rust ignores
    // SIGPIPE by default and println! panics when the reader closes.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let cli = Cli::parse();

    match run(&cli) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(fail) => {
            if cli.json {
                eprintln!(
                    "{}",
                    serde_json::json!({ "error": fail.code(), "detail": fail.to_string() })
                );
            } else {
                eprintln!("error: {fail}");
            }
            ExitCode::FAILURE
        }
    }
}

enum Fail {
    Db(db::DbError),
    Degraded(String),
    Reject(Reject),
    /// A refusal the world taught: the code plus what was probably meant.
    Taught {
        code: &'static str,
        text: String,
    },
    Usage(String),
    Client(client::ClientFail),
}

impl From<Reject> for Fail {
    fn from(r: Reject) -> Self {
        Fail::Reject(r)
    }
}

impl From<ExecuteFail> for Fail {
    fn from(e: ExecuteFail) -> Self {
        match e {
            ExecuteFail::Db(e) => Fail::Db(e),
            ExecuteFail::Degraded(r) => Fail::Degraded(r),
            ExecuteFail::Reject(r) => Fail::Reject(r),
        }
    }
}

impl Fail {
    fn code(&self) -> &'static str {
        match self {
            Fail::Db(_) => "database_error",
            Fail::Degraded(_) => "degraded",
            Fail::Reject(r) => reject_code(r),
            Fail::Taught { code, .. } => code,
            Fail::Usage(_) => "usage",
            Fail::Client(c) => c.code(),
        }
    }
}

impl std::fmt::Display for Fail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fail::Db(e) => write!(f, "database: {e}"),
            Fail::Degraded(r) => write!(
                f,
                "world projection unavailable: {r}\nraw records via 'sac log'; repair the record or upgrade this binary to resume"
            ),
            Fail::Reject(r) => write!(f, "rejected: {}", reject_code(r)),
            Fail::Taught { code, text } => write!(f, "rejected: {code} — {text}"),
            Fail::Usage(m) => write!(f, "{m}"),
            Fail::Client(c) => write!(f, "{c}"),
        }
    }
}

fn reject_code(reject: &Reject) -> &'static str {
    match reject {
        Reject::InvalidTaskId => "invalid_task_id",
        Reject::InvalidParentTaskId => "invalid_parent_task_id",
        Reject::InvalidProposalId => "invalid_proposal_id",
        Reject::ProposalAlreadyOpen => "proposal_already_open",
        Reject::InvalidCommentId => "invalid_comment_id",
        Reject::NotBirthAttribution => "not_birth_attribution",
        Reject::InvalidIncarnationId => "invalid_incarnation_id",
        Reject::IncarnationAlreadyActive => "incarnation_already_active",
        Reject::DemandNotOnTask => "demand_not_on_task",
        Reject::WorkspaceAlreadyExists => "workspace_already_exists",
        Reject::WorkspaceMissing => "workspace_missing",
        Reject::WorktreeAlreadyPresent => "worktree_already_present",
        Reject::CheckpointRewind => "checkpoint_rewind",
        Reject::SteerNotStanding => "steer_not_standing",
        Reject::NoActiveIncarnation => "no_active_incarnation",
        Reject::SteerNotStanding => "steer_not_standing",
        Reject::NoActiveIncarnation => "no_active_incarnation",
        Reject::InvalidStateTransition => "invalid_state_transition",
        Reject::HumanOnly => "human_only",
        Reject::NotClaimHolder => "not_claim_holder",
        Reject::InvalidActor => "invalid_actor",
        Reject::ReasonRequired => "reason_required",
    }
}

fn run(cli: &Cli) -> Result<String, Fail> {
    // the db default is the repo's state root, not the working tree
    let db_path = resolve_db(cli)?;
    let command = match &cli.command {
        Cmd::Create {
            kind: ObjKind::Task,
            name,
            parent,
        } => Command::CreateTask {
            name: Prose::new(name.clone())?,
            parent_id: parent.as_deref().map(parse_task_id).transpose()?,
        },
        Cmd::Claim { id } => Command::ClaimTask {
            id: parse_task_id(id)?,
        },
        Cmd::Done { id, receipt } => Command::CompleteTask {
            id: parse_task_id(id)?,
            receipt: Prose::new(receipt.clone())?,
        },
        Cmd::Drop { id, note } => Command::DropTask {
            id: parse_task_id(id)?,
            note: Prose::new(note.clone())?,
        },
        Cmd::Release { id, note } => Command::ReleaseTask {
            id: parse_task_id(id)?,
            note: Prose::new(note.clone())?,
        },
        Cmd::Propose { action, id, name } => Command::CreateProposal {
            name: Prose::new(name.clone())?,
            action: match action {
                ProposeVerb::Drop => ProposalAction::Drop {
                    task_id: parse_task_id(id)?,
                },
                ProposeVerb::Release => ProposalAction::Release {
                    task_id: parse_task_id(id)?,
                },
            },
        },
        Cmd::Accept { id } => {
            // one verb, two objects: t-N accepts a task, a bare seq a proposal
            if id.starts_with("t-") {
                Command::AcceptTask {
                    id: parse_task_id(id)?,
                }
            } else {
                Command::AcceptProposal {
                    id: parse_proposal_id(id)?,
                }
            }
        }
        Cmd::Reject { id, note } => Command::RejectProposal {
            id: parse_proposal_id(id)?,
            note: Prose::new(note.clone())?,
        },
        Cmd::Withdraw { id, note } => Command::WithdrawProposal {
            id: parse_proposal_id(id)?,
            note: Prose::new(note.clone())?,
        },
        Cmd::Comment {
            target,
            body,
            demand,
        } => Command::Comment {
            target: parse_target(target)?,
            body: Prose::new(body.clone())?,
            kind: if *demand {
                CommentKind::Demand
            } else {
                CommentKind::Note
            },
        },
        Cmd::Steer { id, body } => Command::Comment {
            target: Target::Task(parse_task_id(id)?),
            body: Prose::new(body.clone())?,
            kind: CommentKind::Steer,
        },
        Cmd::List { .. } | Cmd::Log | Cmd::Proposals | Cmd::Show { .. } => {
            return read_only(cli, &db_path);
        }
        Cmd::Serve {
            bind,
            port,
            executor,
        } => {
            let _ = tracing_subscriber::fmt()
                .json()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into()),
                )
                .with_writer(std::io::stderr)
                .try_init();
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| Fail::Usage(format!("runtime: {e}")))?;
            return match runtime.block_on(saccade::serve::run(&db_path, bind, *port, executor)) {
                Err(e) => Err(Fail::Usage(e)),
                Ok(infallible) => match infallible {},
            };
        }
        Cmd::Wait { id, timeout } => {
            let comment = parse_comment_id(id)?;
            return saccade::runner::wait(&db_path, comment, *timeout).map_err(runner_fail);
        }
        Cmd::Cancel { id } => {
            let task = parse_task_id(id)?;
            let conn = db::open_read(&db_path).map_err(Fail::Db)?;
            let loadout = db::load(&conn).map_err(Fail::Db)?;
            let LoadState::Full(world) = loadout.state else {
                return Err(Fail::Degraded(
                    "cancel needs the world; the log will not fold".into(),
                ));
            };
            let Some(incarnation) = world.tasks.get(task.0).and_then(|c| c.active_incarnation)
            else {
                return Err(Fail::Usage(format!(
                    "t-{} has no active run to cancel",
                    task.0
                )));
            };
            Command::CancelIncarnation { id: incarnation }
        }
        Cmd::Checkpoint { id } => {
            let task = parse_task_id(id)?;
            let repo_root = match &cli.repo {
                Some(repo) => repo
                    .canonicalize()
                    .map_err(|e| Fail::Usage(format!("--repo {}: {e}", repo.display())))?,
                None => saccade::paths::repo_root(std::path::Path::new(".")).map_err(|_| {
                    Fail::Usage(
                        "the working directory is not inside a git repository; pass --repo so checkpoint can resolve the branch".into(),
                    )
                })?,
            };
            let conn = db::open_read(&db_path).map_err(Fail::Db)?;
            let loadout = db::load(&conn).map_err(Fail::Db)?;
            let LoadState::Full(world) = loadout.state else {
                return Err(Fail::Degraded(
                    "checkpoint needs the world; the log will not fold".into(),
                ));
            };
            let ctx = world.tasks.get(task.0).ok_or(Reject::InvalidTaskId)?;
            let workspace = ctx.workspace.as_ref().ok_or(Reject::WorkspaceMissing)?;
            let branch: String = workspace.branch.clone().into();
            let tip = sh_git(
                &repo_root,
                &["rev-parse", "--verify", &format!("refs/heads/{branch}")],
            )
            .map_err(Fail::Usage)?;
            let recorded = workspace.checkpoint.as_str();
            if tip == recorded {
                return Ok(format!("t-{} checkpoint unchanged at {tip}", task.0));
            }
            // a tip strictly behind the recorded checkpoint rewinds it; a
            // descended or diverged tip is an advance or a new lineage
            if is_ancestor(&repo_root, &tip, recorded) {
                return Err(Fail::Reject(Reject::CheckpointRewind));
            }
            Command::CheckpointWorkspace {
                task_id: task,
                checkpoint: GitCommit::new(tip)
                    .map_err(|e| Fail::Usage(format!("git gave no commit: {e:?}")))?,
            }
        }
    };

    // Identity is required only where it is recorded: mutating commands.
    let context = context_of(cli)?;
    let creates = matches!(command, Command::CreateTask { .. });
    let mut born: Option<String> = None;
    let stored = if cli.offline {
        let now = cli.at.unwrap_or_else(db::now_epoch);
        let mut conn = db::open(&db_path).map_err(Fail::Db)?;
        match db::record(&mut conn, &context, command.clone(), now) {
            Ok((stored, world)) => {
                if creates {
                    born = born_of(&stored, &world);
                }
                stored
            }
            Err(ExecuteFail::Reject(reject)) => return Err(refused(&conn, &command, reject)),
            Err(other) => return Err(Fail::from(other)),
        }
    } else {
        client::handshake(&cli.server);
        let reply = client::send(&cli.server, &context, command, cli.at).map_err(Fail::Client)?;
        born = reply.id;
        reply.records
    };
    Ok(render_records(cli, &stored, born.as_deref()))
}

/// A refusal the world can teach: the expected format and, where the
/// world knows it, the likely intended target.
fn refused(conn: &rusqlite::Connection, command: &Command, reject: Reject) -> Fail {
    let taught = db::load(conn).ok().and_then(|loadout| match loadout.state {
        db::LoadState::Full(world) => saccade::refusals::teach(&world, command, &reject),
        db::LoadState::Degraded(_) => None,
    });
    match taught {
        Some(text) => Fail::Taught {
            code: reject_code(&reject),
            text,
        },
        None => Fail::Reject(reject),
    }
}

fn runner_fail(e: saccade::runner::RunnerFail) -> Fail {
    match e {
        saccade::runner::RunnerFail::Db(e) => Fail::from(e),
        saccade::runner::RunnerFail::Usage(m)
        | saccade::runner::RunnerFail::Git(m)
        | saccade::runner::RunnerFail::Refused { reason: m } => Fail::Usage(m),
    }
}

fn sh_git(cwd: &std::path::Path, args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .map_err(|e| format!("git {}: {e}", args.first().unwrap_or(&"")))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// True when `ancestor` is an ancestor of `descendant`, equal included.
fn is_ancestor(root: &std::path::Path, ancestor: &str, descendant: &str) -> bool {
    std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["merge-base", "--is-ancestor", ancestor, descendant])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The db default: explicit path, else the named repo's state root,
/// else the repo containing the working directory.
fn resolve_db(cli: &Cli) -> Result<PathBuf, Fail> {
    if let Some(path) = &cli.db {
        return Ok(path.clone());
    }
    let root = match &cli.repo {
        Some(repo) => repo
            .canonicalize()
            .map_err(|e| Fail::Usage(format!("--repo {}: {e}", repo.display())))?,
        None => saccade::paths::repo_root(std::path::Path::new(".")).map_err(Fail::Usage)?,
    };
    Ok(saccade::paths::db_at(&root))
}

fn parse_comment_id(token: &str) -> Result<CommentId, Fail> {
    let n = token.strip_prefix("c-").ok_or_else(|| {
        let note = token
            .strip_prefix('#')
            .filter(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
            .map(|d| format!("; drop the '#': the demand is c-{d}"))
            .unwrap_or_default();
        Fail::Usage(format!(
            "'{token}' is not a comment id (expected c-<n>){note}"
        ))
    })?;
    let n: usize = n
        .parse()
        .map_err(|_| Fail::Usage(format!("'{token}' is not a comment id")))?;
    Ok(CommentId(RecordId(n)))
}

/// Possession fixes the tier: an agent harness sets SACCADE_ACTOR, and
/// that presence is agent tier, unclaimable-away; its absence is human
/// tier. --actor names at either tier and never re-tiers.
fn context_of(cli: &Cli) -> Result<Context, Fail> {
    let tier = if std::env::var_os("SACCADE_ACTOR").is_some() {
        Tier::Agent
    } else {
        Tier::Human
    };
    let actor = cli
        .actor
        .clone()
        .filter(|a| !a.is_empty())
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "human".into());
    Ok(Context {
        actor: ActorName::new(actor)?,
        tier,
    })
}
fn read_only(cli: &Cli, db_path: &std::path::Path) -> Result<String, Fail> {
    let conn = db::open_read(db_path).map_err(Fail::Db)?;
    let loadout = db::load(&conn).map_err(Fail::Db)?;
    match &cli.command {
        Cmd::Log => {
            if let LoadState::Degraded(reason) = &loadout.state {
                eprintln!("warning: world projection unavailable: {reason}");
            }
            Ok(render_log(cli, &loadout.rows))
        }
        Cmd::List { .. } => match loadout.state {
            LoadState::Full(world) => Ok(render_tasks(cli, &world)),
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
        },
        Cmd::Proposals => match loadout.state {
            LoadState::Full(world) => Ok(render_proposals(cli, &world)),
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
        },
        Cmd::Show { id } => match loadout.state {
            LoadState::Full(world) => {
                let task_id = parse_task_id(id)?;
                Ok(render_show(&world, task_id)?)
            }
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
        },
        _ => unreachable!("read_only reached from a mutating command"),
    }
}

fn parse_task_id(token: &str) -> Result<TaskId, Fail> {
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
fn hashed_task_note(token: &str) -> String {
    match token
        .strip_prefix("#t-")
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
    {
        Some(n) => format!("; drop the '#': the task is addressed as t-{n}"),
        None => String::new(),
    }
}

fn parse_target(token: &str) -> Result<Target, Fail> {
    if let Some(rest) = token.strip_prefix('#') {
        let n: usize = rest.parse().map_err(|_| {
            Fail::Usage(format!(
                "'{token}' is not a comment id (expected #<seq>){}",
                hashed_record_note(rest)
            ))
        })?;
        return Ok(Target::Comment(CommentId(RecordId(n))));
    }
    Ok(Target::Task(parse_task_id(token)?))
}

/// The hashed-token faces taught at the comment door: a hashed task id
/// wants the bare thread, a hashed c-N wants the bare record.
fn hashed_record_note(rest: &str) -> String {
    let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if let Some(n) = rest.strip_prefix("t-").filter(|n| numeric(n)) {
        return format!("; drop the '#': the thread is addressed as t-{n}");
    }
    if let Some(n) = rest.strip_prefix("c-").filter(|n| numeric(n)) {
        return format!("; drop the 'c-': the comment is addressed as #{n}");
    }
    String::new()
}

fn parse_proposal_id(token: &str) -> Result<ProposalId, Fail> {
    let n: usize = token.parse().map_err(|_| {
        Fail::Usage(format!(
            "'{token}' is not a proposal id (expected a bare log position, e.g. 614)"
        ))
    })?;
    Ok(ProposalId(RecordId(n)))
}

/// The reference a create reply names: the task whose birth record just
/// landed, resolved against the post-write fold.
fn born_of(stored: &[StoredRecord], world: &World) -> Option<String> {
    let birth = stored.iter().find(|r| r.kind == "task_created")?;
    world
        .task_born_at(RecordId(birth.seq))
        .map(|id| format!("t-{}", id.0))
}

fn render_records(cli: &Cli, stored: &[StoredRecord], born: Option<&str>) -> String {
    if cli.json {
        // the wire's reply shape: records always, the born task's token
        // only on a create
        let mut reply = serde_json::json!({"records": stored});
        if let Some(id) = born {
            reply["id"] = serde_json::json!(id);
        }
        return serde_json::to_string_pretty(&reply).expect("records are plain data");
    }
    let mut lines = Vec::with_capacity(stored.len() + 1);
    if let Some(id) = born {
        lines.push(id.to_string());
    }
    lines.extend(stored.iter().map(record_line));
    lines.join("\n")
}

fn render_log(cli: &Cli, rows: &[StoredRecord]) -> String {
    if cli.json {
        return serde_json::to_string_pretty(rows).expect("records are plain data");
    }
    rows.iter().map(record_line).collect::<Vec<_>>().join("\n")
}

fn record_line(r: &StoredRecord) -> String {
    format!(
        "#{} {} {}/{} et={} lt={} {}",
        r.seq, r.kind, r.actor, r.tier, r.event_time, r.logged_time, r.payload
    )
}

fn render_tasks(cli: &Cli, world: &World) -> String {
    let views: Vec<TaskView> = (0..world.tasks.len())
        .map(|i| task_view(world, TaskId(i)))
        .collect::<Option<_>>()
        .expect("indices come from the vec itself");
    // the digest question is standing work; history lives in log and show
    let views: Vec<_> = views
        .into_iter()
        .filter(|v| matches!(v.state, "open" | "claimed" | "delivered"))
        .collect();

    if cli.json {
        let rows: Vec<serde_json::Value> = views
            .iter()
            .map(|v| {
                serde_json::json!({
                    "id": v.id,
                    "state": v.state,
                    "parent": v.parent,
                    "name": v.name,
                    "comments": v.n_comments,
                    "proposal": v.proposal.as_ref().map(|m| serde_json::json!({
                        "seq": m.seq,
                        "verb": m.verb,
                    })),
                })
            })
            .collect();
        return serde_json::to_string_pretty(&rows).expect("views are plain data");
    }

    views
        .iter()
        .map(|v| {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                v.id,
                v.state,
                v.parent.clone().unwrap_or_else(|| "-".into()),
                v.name,
                v.proposal
                    .as_ref()
                    .map(|m| format!("{}#{}", m.verb, m.seq))
                    .unwrap_or_else(|| "-".into()),
                if v.n_comments > 0 { "#" } else { "-" }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const WIDTH: usize = 80;

/// The inspector dock as text: everything about the one thing.
fn render_show(world: &World, task_id: TaskId) -> Result<String, Fail> {
    let view =
        show_view(world, task_id).ok_or_else(|| Fail::Usage(format!("no task t-{}", task_id.0)))?;

    let mut out = vec![wrap(
        &view.name,
        WIDTH,
        &format!("{}  {}  ", view.id, view.state),
        "  ",
    )];
    if let Some(parent) = &view.parent {
        out.push(format!("parent  {parent}"));
    }
    if let Some(receipt) = &view.receipt {
        out.push("receipt".to_string());
        out.push(wrap(receipt, WIDTH, "  ", "  "));
    }
    let ctx = &world.tasks[task_id.0];
    for line in comment_thread(&world.comments, ctx) {
        let indent = "  ".repeat(line.depth.saturating_sub(1));
        out.push(String::new());
        let state = line.state.map(|s| format!("  ({s})")).unwrap_or_default();
        out.push(format!("{indent}#{}  {}{state}", line.seq, line.actor));
        out.push(wrap(
            &line.body,
            WIDTH,
            &format!("{indent}  "),
            &format!("{indent}  "),
        ));
        if let Some(refusal) = &line.refusal {
            out.push(wrap(
                &format!("refused {}: {}", fmt_when(refusal.at), refusal.reason),
                WIDTH,
                &format!("{indent}  "),
                &format!("{indent}  "),
            ));
        }
    }
    Ok(out.join("\n"))
}

/// The refusal's moment, as the thread renders it.
fn fmt_when(ts: u64) -> String {
    let dt = time::OffsetDateTime::from_unix_timestamp(ts as i64)
        .unwrap_or(time::OffsetDateTime::UNIX_EPOCH)
        .to_offset(time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC));
    dt.format(&time::macros::format_description!(
        "[year]-[month repr:numerical]-[day] [hour repr:24]:[minute]"
    ))
    .unwrap_or_default()
}

/// Greedy word wrap; a word longer than a line is hard-broken so a single
/// token cannot re-create the wall.
fn wrap(text: &str, width: usize, first: &str, rest: &str) -> String {
    let body = width.saturating_sub(rest.chars().count()).max(1);
    let mut pieces: Vec<Vec<char>> = Vec::new();
    for word in text.split_whitespace() {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > body {
            pieces.extend(chars.chunks(body).map(|c| c.to_vec()));
        } else {
            pieces.push(chars);
        }
    }
    let mut out = String::new();
    let mut budget = width.saturating_sub(first.chars().count());
    for (i, piece) in pieces.into_iter().enumerate() {
        if i == 0 {
            out.push_str(first);
        } else if piece.len() < budget {
            out.push(' ');
            budget -= 1;
        } else {
            out.push('\n');
            out.push_str(rest);
            budget = body;
        }
        budget = budget.saturating_sub(piece.len());
        out.extend(piece);
    }
    out
}

fn render_proposals(cli: &Cli, world: &World) -> String {
    let views: Vec<ProposalView> = world
        .proposals
        .keys()
        .map(|id| proposal_view(world, *id))
        .map(|v| v.expect("ids come from the map itself"))
        .collect();

    if cli.json {
        let rows: Vec<serde_json::Value> = views
            .iter()
            .map(|v| {
                serde_json::json!({
                    "id": v.id,
                    "state": v.state,
                    "action": v.action,
                    "task": v.task,
                    "name": v.name,
                })
            })
            .collect();
        return serde_json::to_string_pretty(&rows).expect("views are plain data");
    }

    views
        .iter()
        .map(|v| format!("{}\t{}\t{} {}\t{}", v.id, v.state, v.action, v.task, v.name))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod test {
    use super::*;

    /// Possession is the only tier door: SACCADE_ACTOR's presence records
    /// agent, its absence human, and --actor names at either tier without
    /// re-tiering.
    #[test]
    fn possession_fixes_the_tier_and_actor_never_re_tiers() {
        // env is process-global; this is the only test in this binary
        // that touches SACCADE_ACTOR
        unsafe { std::env::remove_var("SACCADE_ACTOR") };
        let bare = Cli::try_parse_from(["sac", "list"]).unwrap();
        assert_eq!(
            context_of(&bare).unwrap_or_else(|f| panic!("{f}")).tier,
            Tier::Human
        );

        unsafe { std::env::set_var("SACCADE_ACTOR", "pi") };
        let possessed = Cli::try_parse_from(["sac", "list"]).unwrap();
        assert_eq!(
            context_of(&possessed)
                .unwrap_or_else(|f| panic!("{f}"))
                .tier,
            Tier::Agent
        );

        let named = Cli::try_parse_from(["sac", "--actor", "jerry", "list"]).unwrap();
        let ctx = context_of(&named).unwrap_or_else(|f| panic!("{f}"));
        assert_eq!((ctx.tier, ctx.actor.as_str()), (Tier::Agent, "jerry"));

        unsafe { std::env::remove_var("SACCADE_ACTOR") };
        let named = Cli::try_parse_from(["sac", "--actor", "saccade bot", "list"]).unwrap();
        let ctx = context_of(&named).unwrap_or_else(|f| panic!("{f}"));
        assert_eq!((ctx.tier, ctx.actor.as_str()), (Tier::Human, "saccade bot"));
        unsafe { std::env::remove_var("SACCADE_ACTOR") };
    }

    /// Hashed tokens learn their bare doors at the parse door: a hashed
    /// task id wants the bare thread, a hashed c-N wants the bare record,
    /// a hashed demand wants the c- door.
    #[test]
    fn hashed_tokens_learn_their_bare_doors() {
        let refused = parse_task_id("#t-3")
            .map_err(|f| f.to_string())
            .unwrap_err();
        assert_eq!(
            refused,
            "'#t-3' is not a task id (expected t-<n>; only tasks exist); \
             drop the '#': the task is addressed as t-3"
        );

        let refused = parse_target("#t-3").map_err(|f| f.to_string()).unwrap_err();
        assert_eq!(
            refused,
            "'#t-3' is not a comment id (expected #<seq>); \
             drop the '#': the thread is addressed as t-3"
        );

        let refused = parse_target("#c-7").map_err(|f| f.to_string()).unwrap_err();
        assert_eq!(
            refused,
            "'#c-7' is not a comment id (expected #<seq>); \
             drop the 'c-': the comment is addressed as #7"
        );

        let refused = parse_comment_id("#7")
            .map_err(|f| f.to_string())
            .unwrap_err();
        assert_eq!(
            refused,
            "'#7' is not a comment id (expected c-<n>); \
             drop the '#': the demand is c-7"
        );

        assert_eq!(
            parse_task_id("t-3").unwrap_or_else(|f| panic!("{f}")),
            TaskId(3)
        );
        assert_eq!(
            parse_target("#7").unwrap_or_else(|f| panic!("{f}")),
            Target::Comment(CommentId(RecordId(7)))
        );
        assert_eq!(
            parse_comment_id("c-7").unwrap_or_else(|f| panic!("{f}")),
            CommentId(RecordId(7))
        );
    }
}
