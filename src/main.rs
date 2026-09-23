use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use saccade::World;
use saccade::client;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::objects::task::TaskId;
use saccade::views::{
    CommentLine, ProposalView, SearchGroup, SearchQuery, TaskView, Term, comment_line,
    comment_thread, matched_line, proposal_view, search, show_view, task_view,
};
use saccade::{
    ActorName, Command, CommentId, CommentKind, Context, GitCommit, ProposalAction, ProposalId,
    Prose, RecordId, Reject, Target, Tier,
};

mod skill;

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
    /// Everything about tasks and records, many at once; raw ids show
    /// the event, t-<n> shows the task plus its thread
    Show {
        /// Ids in any mix: #<seq> or c-<seq> a comment or a task's birth,
        /// t-<n> a whole thread
        #[arg(required_unless_present = "stdin")]
        ids: Vec<String>,
        /// Read one id per line from stdin instead of arguments; blank
        /// and invalid lines are skipped with a note
        #[arg(long)]
        stdin: bool,
    },
    /// Search the folded record: exact terms over task titles, comment
    /// bodies, and receipts; an id term is a reference search
    Search {
        /// Terms to match (a record must match them all) and
        /// only-show-me facets:
        ///   in:t-N     one thread's records only
        ///   by:NAME    author, matched within names (by:pi finds pi/t-90-1)
        ///   kind:K     task, note, demand, steer, ask, receipt
        ///   under:t-N  the task's thread and its descendants' threads
        ///
        /// An id term — '#907' or 't-49' — is a reference search: every
        /// record citing it or addressing it; quote the hash in shells
        #[arg(verbatim_doc_comment)]
        terms: Vec<String>,
        /// With one id term: that record plus N before and after
        #[arg(short = 'C', long)]
        context: Option<usize>,
    },
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
    /// Clone the log's prefix through a storage seq into a fresh
    /// tracker: the reproduction recipe's cursor half
    Clone {
        /// The storage seq to copy through, inclusive
        #[arg(long)]
        at: u64,
        /// The fresh tracker to write; must not exist
        #[arg(long)]
        out: PathBuf,
    },
    /// Block until the demand's run asks something of the waiter: settle,
    /// cancel, prompt rejection carrying the cause, refusal, an answer
    /// with no run behind it, or a blocking ask — never replies
    Wait {
        /// The demand to watch (c-<n>)
        id: String,
        /// Give up after this many seconds (default: wait forever)
        #[arg(long)]
        timeout: Option<u64>,
    },
    /// Deploy this binary's embedded skill, or verify the copy in this repo
    Skill {
        #[command(subcommand)]
        verb: SkillVerb,
    },
}

#[derive(Clone, Copy, Subcommand)]
enum SkillVerb {
    /// Write this binary's skill into ./.agents/skills/saccade; a
    /// differing copy refuses until the directory is deleted
    Install,
    /// Compare ./.agents/skills/saccade with this binary's embedded copy
    Check,
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

    // the skill door carries no db: the binary deploys its own embed,
    // in any directory, repo or not
    if let Cmd::Skill { verb } = &cli.command {
        return skill_door(cli.json, *verb);
    }

    match run(&cli) {
        Ok(output) => {
            if !output.is_empty() {
                println!("{output}");
            }
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
            Fail::Reject(r) => r.code(),
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
            Fail::Reject(r) => write!(f, "rejected: {}", r.code()),
            Fail::Taught { code, text } => write!(f, "rejected: {code} — {text}"),
            Fail::Usage(m) => write!(f, "{m}"),
            Fail::Client(c) => write!(f, "{c}"),
        }
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
        Cmd::List { .. } | Cmd::Log | Cmd::Proposals | Cmd::Show { .. } | Cmd::Search { .. } => {
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
                .with_writer(saccade::attempts::WarnsTee::beside(&db_path))
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
        Cmd::Clone { at, out } => {
            // the reproduction recipe's cursor half: the clone names the
            // world a refused request died against
            let cloned = db::clone(&db_path, *at as usize, out).map_err(Fail::Db)?;
            return Ok(if cli.json {
                serde_json::json!({
                    "cloned": cloned,
                    "through": at,
                    "out": out.display().to_string(),
                })
                .to_string()
            } else {
                format!(
                    "cloned {cloned} records through seq {at} to {}",
                    out.display()
                )
            });
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
        Cmd::Skill { .. } => unreachable!("the skill door dispatches before the db doors"),
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
            code: reject.code(),
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

/// The skill verbs' output door: the verdict lands on stdout — drift
/// and absence are findings, not failures — and the exit code carries
/// it for scripts.
fn skill_door(json: bool, verb: SkillVerb) -> ExitCode {
    let cwd = std::path::Path::new(".");
    let version = env!("CARGO_PKG_VERSION");
    let report = |verdict: &str, carries: Option<&str>| {
        if json {
            println!(
                "{}",
                serde_json::json!({"verdict": verdict, "carries": carries, "binary": version})
            );
        } else {
            let carries = carries.unwrap_or("no version");
            println!(
                "skill {verdict}: {} carries {carries}, binary is {version}",
                skill::HOME
            );
        }
    };
    match verb {
        SkillVerb::Install => match skill::install(cwd, version) {
            Ok(skill::Deployed::Wrote) => {
                report("installed", Some(version));
                ExitCode::SUCCESS
            }
            Ok(skill::Deployed::Untouched) => {
                report("in-sync", Some(version));
                ExitCode::SUCCESS
            }
            Err(detail) => {
                if json {
                    eprintln!(
                        "{}",
                        serde_json::json!({"error": "usage", "detail": detail})
                    );
                } else {
                    eprintln!("error: {detail}");
                }
                ExitCode::FAILURE
            }
        },
        SkillVerb::Check => match skill::check(cwd) {
            skill::Verdict::InSync(carries) => {
                report("in-sync", carries.as_deref());
                ExitCode::SUCCESS
            }
            skill::Verdict::Drifted(carries) => {
                report("drifted", carries.as_deref());
                ExitCode::FAILURE
            }
            skill::Verdict::Absent => {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({"verdict": "absent", "carries": null, "binary": version})
                    );
                } else {
                    println!(
                        "skill absent: no {} in this directory — sac skill install deploys this binary's copy",
                        skill::HOME
                    );
                }
                ExitCode::FAILURE
            }
        },
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
        Cmd::Show { ids, stdin } => match loadout.state {
            LoadState::Full(world) => {
                if *stdin {
                    return show_stdin(&world, ids);
                }
                let mut blocks = Vec::new();
                for token in ids {
                    blocks.push(render_id(&world, token)?);
                }
                Ok(blocks.join("\n\n"))
            }
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
        },
        Cmd::Search { terms, context } => {
            let query = SearchQuery::parse(terms).map_err(|f| Fail::Usage(f.to_string()))?;
            if let Some(around) = context {
                let anchor = anchor_seq(&loadout, &query)?;
                return Ok(render_anchor(cli, &loadout.rows, anchor, *around));
            }
            match loadout.state {
                LoadState::Full(world) => {
                    let groups = search(&world, &query).map_err(|f| Fail::Usage(f.to_string()))?;
                    Ok(render_search(cli, &groups, terms, &query))
                }
                LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
            }
        }
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
        out.push(String::new());
        out.extend(comment_block(&line));
    }
    Ok(out.join("\n"))
}

/// A thread view's body format for one comment: its header line, the
/// wrapped body, and the machinery's refusal when it refused — the
/// same block whether it rides a thread or a pointer opened it.
fn comment_block(line: &CommentLine) -> Vec<String> {
    let indent = "  ".repeat(line.depth.saturating_sub(1));
    let state = line
        .state
        .as_ref()
        .map(|s| format!("  ({s})"))
        .unwrap_or_default();
    let mut out = vec![format!("{indent}#{}  {}{state}", line.seq, line.actor)];
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
    out
}

/// One id as show sees it: a whole thread or a single record.
#[derive(Clone, Copy, Debug)]
enum ShowId {
    Thread(TaskId),
    Record(CommentId),
}

fn parse_show_id(token: &str) -> Result<ShowId, Fail> {
    let not_an_id = || {
        Fail::Usage(format!(
            "'{token}' is not an id (expected t-<n>, #<seq>, or c-<seq>)"
        ))
    };
    if let Some(rest) = token
        .strip_prefix('#')
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
    {
        return Ok(ShowId::Record(CommentId(RecordId(
            rest.parse().expect("digits checked"),
        ))));
    }
    if let Some(rest) = token
        .strip_prefix("c-")
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
    {
        return Ok(ShowId::Record(CommentId(RecordId(
            rest.parse().expect("digits checked"),
        ))));
    }
    if token.starts_with("t-") {
        return Ok(ShowId::Thread(parse_task_id(token)?));
    }
    Err(not_an_id())
}

/// Render one id: a thread whole, or a record as itself — a comment
/// as its block, a task's birth as the literal event (header and
/// relation, no thread substitution: the relation names t-N, and the
/// taught law does the rest).
fn render_id(world: &World, token: &str) -> Result<String, Fail> {
    match parse_show_id(token)? {
        ShowId::Thread(id) => render_show(world, id),
        ShowId::Record(id) => {
            if let Some(line) = comment_line(world, id) {
                return Ok(comment_block(&line).join("\n"));
            }
            if let Some(task) = world.task_born_at(id.0) {
                let ctx = &world.tasks[task.0];
                let parent = task_view(world, task)
                    .and_then(|v| v.parent)
                    .map(|p| format!(" (parent {p})"))
                    .unwrap_or_default();
                return Ok(format!(
                    "#{}  task  {}\nbirth of t-{}{parent}",
                    ctx.birth.0,
                    ctx.birth_actor.as_str(),
                    task.0
                ));
            }
            Err(Fail::Usage(format!(
                "#{} is not a comment or a task birth",
                id.0.0
            )))
        }
    }
}

/// The piped face: one id per line, every failure a skip with a note,
/// never a broken chain. Empty input is silence.
fn show_stdin(world: &World, ids: &[String]) -> Result<String, Fail> {
    if !ids.is_empty() {
        return Err(Fail::Usage("ids as arguments or --stdin, not both".into()));
    }
    if std::io::stdin().is_terminal() {
        return Err(Fail::Usage(
            "nothing piped: --stdin reads ids one per line".into(),
        ));
    }
    let mut blocks = Vec::new();
    for line in std::io::stdin().lines() {
        let line = line.map_err(|e| Fail::Usage(format!("stdin: {e}")))?;
        let token = line.trim();
        if token.is_empty() {
            continue;
        }
        match render_id(world, token) {
            Ok(block) => blocks.push(block),
            Err(f) => blocks.push(format!("skipped '{token}': {f}")),
        }
    }
    Ok(blocks.join("\n\n"))
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

/// The search door as text: thread groups of pointers, one matched
/// line each. Counts ride the headers only when the query narrows —
/// narrowing's visibility, never an aggregate badge — and a query
/// that matched nothing says so in one line.
fn render_search(
    cli: &Cli,
    groups: &[SearchGroup],
    tokens: &[String],
    query: &SearchQuery,
) -> String {
    let counts = !query.facets.is_empty();
    if cli.json {
        let rows: Vec<serde_json::Value> = groups
            .iter()
            .map(|g| {
                let mut row = serde_json::json!({
                    "task": format!("t-{}", g.task),
                    "title": g.title,
                    "records": g.records.iter().map(|r| serde_json::json!({
                        "pointer": r.pointer,
                        "kind": r.kind,
                        "actor": r.actor,
                        "body": r.body,
                    })).collect::<Vec<_>>(),
                });
                if counts {
                    row["shown"] = serde_json::json!(g.records.len());
                    row["total"] = serde_json::json!(g.total);
                }
                row
            })
            .collect();
        return serde_json::to_string_pretty(&serde_json::json!({ "groups": rows }))
            .expect("results are plain data");
    }
    if groups.is_empty() {
        return format!("no matches ({})", tokens.join(" "));
    }
    let mut out = Vec::new();
    for g in groups {
        let mark = if counts {
            format!(" ({} of {})", g.records.len(), g.total)
        } else {
            String::new()
        };
        out.push(format!("t-{}  {}{}", g.task, g.title, mark));
        for r in &g.records {
            let actor = r
                .actor
                .as_deref()
                .map(|a| format!("{a}  "))
                .unwrap_or_default();
            let prefix = format!("  {}  {actor}", r.pointer);
            let budget = WIDTH.saturating_sub(prefix.chars().count());
            let line = matched_line(&query.terms, &r.body);
            out.push(format!("{prefix}{}", glimpse(&line, budget)));
        }
    }
    out.join("\n")
}

/// The anchor query names one record: #N itself, or t-N's birth.
fn anchor_seq(loadout: &db::Loadout, query: &SearchQuery) -> Result<usize, Fail> {
    match query.terms.as_slice() {
        [Term::Comment(id)] => {
            if id.0 >= loadout.rows.len() {
                return Err(Fail::Usage(format!(
                    "no record #{}; the log holds {} records, #0 through #{}",
                    id.0,
                    loadout.rows.len(),
                    loadout.rows.len().saturating_sub(1)
                )));
            }
            Ok(id.0)
        }
        [Term::Task(id)] => match &loadout.state {
            LoadState::Full(world) => {
                world.tasks.get(id.0).map(|ctx| ctx.birth.0).ok_or_else(|| {
                    Fail::Usage(format!(
                        "no task t-{}; the fold holds {} tasks",
                        id.0,
                        world.tasks.len()
                    ))
                })
            }
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason.clone())),
        },
        _ => Err(Fail::Usage(
            "-C anchors on one record: give a single id term (#seq or t-N) and no facets".into(),
        )),
    }
}

/// The anchor's window: the record, N before, N after — one line each,
/// straight off the log's own rows.
fn render_anchor(cli: &Cli, rows: &[StoredRecord], anchor: usize, around: usize) -> String {
    let from = anchor.saturating_sub(around);
    let to = (anchor + around).min(rows.len().saturating_sub(1));
    let window = &rows[from..=to];
    if cli.json {
        let records: Vec<serde_json::Value> = window
            .iter()
            .map(|row| {
                serde_json::json!({
                    "seq": row.seq,
                    "kind": row.kind,
                    "actor": format!("{}/{}", row.actor, row.tier),
                    "line": row_line(row),
                })
            })
            .collect();
        return serde_json::to_string_pretty(&serde_json::json!({
            "anchor": anchor,
            "records": records
        }))
        .expect("rows are plain data");
    }
    window
        .iter()
        .map(|row| {
            let prefix = format!("#{}  {}  {}/{}", row.seq, row.kind, row.actor, row.tier);
            let budget = WIDTH.saturating_sub(prefix.chars().count() + 2);
            format!("{prefix}  {}", glimpse(&row_line(row), budget))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The record's own first text line — the same fields the log prints.
fn row_line(row: &StoredRecord) -> String {
    let payload: serde_json::Value = match serde_json::from_str(&row.payload) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    ["body", "name", "receipt", "note"]
        .iter()
        .find_map(|k| payload.get(k).and_then(|v| v.as_str()))
        .map(|t| t.lines().next().unwrap_or_default().to_string())
        .unwrap_or_default()
}

/// One line cut to its budget on a char boundary; the ellipsis is the
/// only mark a cut earns.
fn glimpse(line: &str, budget: usize) -> String {
    if line.chars().count() <= budget {
        line.to_string()
    } else {
        let mut cut: String = line.chars().take(budget.saturating_sub(1)).collect();
        cut.push('…');
        cut
    }
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
