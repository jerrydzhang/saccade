use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use saccade::World;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::objects::task::{Receipt, TaskId};
use saccade::wire::ProposalView;
use saccade::{Command, Context, ProposalAction, ProposalId, RecordId, Reject, Tier};

#[derive(Parser)]
#[command(name = "sac", about = "Saccade: awesome issue tracker")]
struct Cli {
    /// Path to the event database
    #[arg(long, global = true, env = "SACCADE_DB", default_value = "saccade.db")]
    db: PathBuf,

    /// Actor name recorded on events (Required for mutating commands)
    #[arg(long, global = true, env = "SACCADE_ACTOR")]
    actor: Option<String>,

    /// Tier controls the authority of the actor (Required for mutating commands)
    #[arg(long, global = true, env = "SACCADE_TIER", value_enum)]
    tier: Option<TierArg>,

    /// Machine-readable output
    #[arg(long, global = true)]
    json: bool,

    /// Event time override
    #[arg(long, global = true)]
    at: Option<u64>,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum TierArg {
    Human,
    Agent,
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
    /// Complete a claimed task, depositing a receipt
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
    /// Human-only: accept a proposal, executing its act
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
    Usage(String),
}

impl Fail {
    fn code(&self) -> &'static str {
        match self {
            Fail::Db(_) => "database_error",
            Fail::Degraded(_) => "degraded",
            Fail::Reject(r) => reject_code(r),
            Fail::Usage(_) => "usage",
        }
    }
}

impl std::fmt::Display for Fail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fail::Db(e) => write!(f, "database: {e}"),
            Fail::Degraded(r) => write!(
                f,
                "world projection unavailable: {r}\nraw records via 'sac log'; upgrade this binary to resume"
            ),
            Fail::Reject(r) => write!(f, "rejected: {}", reject_code(r)),
            Fail::Usage(m) => write!(f, "{m}"),
        }
    }
}

fn reject_code(reject: &Reject) -> &'static str {
    match reject {
        Reject::InvalidTaskId => "invalid_task_id",
        Reject::InvalidParentTaskId => "invalid_parent_task_id",
        Reject::InvalidProposalId => "invalid_proposal_id",
        Reject::InvalidStateTransition => "invalid_state_transition",
        Reject::HumanOnly => "human_only",
        Reject::ReasonRequired => "reason_required",
    }
}

fn run(cli: &Cli) -> Result<String, Fail> {
    let command = match &cli.command {
        Cmd::Create {
            kind: ObjKind::Task,
            name,
            parent,
        } => Command::CreateTask {
            name: name.clone(),
            parent_id: parent.as_deref().map(parse_task_id).transpose()?,
        },
        Cmd::Claim { id } => Command::ClaimTask {
            id: parse_task_id(id)?,
        },
        Cmd::Done { id, receipt } => Command::CompleteTask {
            id: parse_task_id(id)?,
            receipt: Receipt(receipt.clone()),
        },
        Cmd::Drop { id, note } => Command::DropTask {
            id: parse_task_id(id)?,
            note: note.clone(),
        },
        Cmd::Release { id, note } => Command::ReleaseTask {
            id: parse_task_id(id)?,
            note: note.clone(),
        },
        Cmd::Propose { action, id, name } => Command::CreateProposal {
            name: name.clone(),
            action: match action {
                ProposeVerb::Drop => ProposalAction::Drop {
                    task_id: parse_task_id(id)?,
                },
                ProposeVerb::Release => ProposalAction::Release {
                    task_id: parse_task_id(id)?,
                },
            },
        },
        Cmd::Accept { id } => Command::AcceptProposal {
            id: parse_proposal_id(id)?,
        },
        Cmd::Reject { id, note } => Command::RejectProposal {
            id: parse_proposal_id(id)?,
            note: note.clone(),
        },
        Cmd::Withdraw { id, note } => Command::WithdrawProposal {
            id: parse_proposal_id(id)?,
            note: note.clone(),
        },
        Cmd::List { .. } | Cmd::Log | Cmd::Proposals => return read_only(cli),
    };

    // Identity is required only where it is recorded: mutating commands.
    let context = context_of(cli)?;
    let now = cli.at.unwrap_or_else(db::now_epoch);
    let mut conn = db::open(&cli.db).map_err(Fail::Db)?;
    let stored = db::execute(&mut conn, &context, command, now).map_err(|e| match e {
        ExecuteFail::Db(e) => Fail::Db(e),
        ExecuteFail::Degraded(r) => Fail::Degraded(r),
        ExecuteFail::Reject(r) => Fail::Reject(r),
    })?;
    Ok(render_records(cli, &stored))
}

/// clap forbids required+global, so presence is enforced here for commands that record events.
fn context_of(cli: &Cli) -> Result<Context, Fail> {
    let tier = cli.tier.ok_or_else(|| {
        Fail::Usage(
            "--tier <human|agent> (or SACCADE_TIER) is required by commands that record events"
                .into(),
        )
    })?;
    let actor = cli.actor.clone().filter(|a| !a.is_empty()).ok_or_else(|| {
        Fail::Usage(
            "--actor <name> (or SACCADDE_ACTOR) is required by commands that record events".into(),
        )
    })?;
    Ok(Context {
        actor,
        tier: match tier {
            TierArg::Human => Tier::Human,
            TierArg::Agent => Tier::Agent,
        },
    })
}
fn read_only(cli: &Cli) -> Result<String, Fail> {
    let conn = db::open_read(&cli.db).map_err(Fail::Db)?;
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
        _ => unreachable!("read_only reached from a mutating command"),
    }
}

fn parse_task_id(token: &str) -> Result<TaskId, Fail> {
    let n = token.strip_prefix("t-").ok_or_else(|| {
        Fail::Usage(format!(
            "'{token}' is not a task id (expected t-<n>; only tasks exist)"
        ))
    })?;
    let n: usize = n
        .parse()
        .map_err(|_| Fail::Usage(format!("'{token}' is not a task id")))?;
    Ok(TaskId(n))
}

fn parse_proposal_id(token: &str) -> Result<ProposalId, Fail> {
    let n: usize = token.parse().map_err(|_| {
        Fail::Usage(format!(
            "'{token}' is not a proposal id (expected a bare log position, e.g. 614)"
        ))
    })?;
    Ok(ProposalId(RecordId(n)))
}

fn render_records(cli: &Cli, stored: &[StoredRecord]) -> String {
    if cli.json {
        return serde_json::to_string_pretty(&records_json(stored))
            .expect("records are plain data");
    }
    stored
        .iter()
        .map(record_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_log(cli: &Cli, rows: &[StoredRecord]) -> String {
    if cli.json {
        return serde_json::to_string_pretty(&records_json(rows)).expect("records are plain data");
    }
    rows.iter().map(record_line).collect::<Vec<_>>().join("\n")
}

fn records_json(rows: &[StoredRecord]) -> serde_json::Value {
    serde_json::Value::Array(rows.iter().map(record_json).collect())
}

fn record_json(r: &StoredRecord) -> serde_json::Value {
    serde_json::json!({
        "seq": r.seq,
        "event_time": r.event_time,
        "logged_time": r.logged_time,
        "actor": r.actor,
        "tier": r.tier,
        "kind": r.kind,
        "payload": serde_json::from_str::<serde_json::Value>(&r.payload)
            .unwrap_or_else(|_| serde_json::Value::String(r.payload.clone())),
    })
}

fn record_line(r: &StoredRecord) -> String {
    format!(
        "#{} {} {}/{} et={} lt={} {}",
        r.seq, r.kind, r.actor, r.tier, r.event_time, r.logged_time, r.payload
    )
}

fn render_tasks(cli: &Cli, world: &World) -> String {
    let views: Vec<saccade::wire::TaskView> = world
        .tasks
        .iter()
        .map(|t| saccade::wire::view_of(t, world))
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
                "{}\t{}\t{}\t{}\t{}",
                v.id,
                v.state,
                v.parent.clone().unwrap_or_else(|| "-".into()),
                v.name,
                v.proposal
                    .as_ref()
                    .map(|m| format!("{}#{}", m.verb, m.seq))
                    .unwrap_or_else(|| "-".into())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_proposals(cli: &Cli, world: &World) -> String {
    let views: Vec<ProposalView> = world
        .proposals
        .values()
        .map(|p| saccade::wire::view_of_proposal(p, world))
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
