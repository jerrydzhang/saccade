use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use saccade::World;
use saccade::client;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::objects::task::TaskId;
use saccade::views::{
    ArtifactLine, CommentLine, RequestFacts, SearchGroup, SearchQuery, TaskView, Term, ThreadItem,
    artifact_line, comment_line, matched_line, search, show_view, task_view, thread_index,
    thread_view,
};
use saccade::{
    ActorName, Artifact, Command, CommentId, CommentKind, ContentHash, Context, GitCommit,
    IncarnationId, Prose, RecordId, Reject, Scope, Target, Tier, WorktreePath, wire,
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
    /// Create a new task
    Create {
        #[arg(value_enum)]
        kind: ObjKind,
        name: String,
    },
    /// Take whole-task responsibility; a transfer, never a state change
    Claim { id: String },
    /// Human-only: stop an unfinished undertaking, explaining why
    Cancel {
        id: String,
        #[arg(long)]
        explanation: String,
    },
    /// Human-only: resume a cancelled undertaking, explaining why
    Reopen {
        id: String,
        #[arg(long)]
        explanation: String,
    },
    /// Attach a note to a task (t-<n>) or reply to a comment (c-<seq>)
    Comment { target: String, body: String },
    /// Attach a request — prose seeking response, recording why work
    /// happened — to a task (t-<n>) or a comment (c-<seq>)
    Request { target: String, body: String },
    /// Offer the whole-task closing account (t-<n> only: a broadcast)
    Offer { id: String, body: String },
    /// Accept an offered conclusion: the offer it accepts (c-<seq>)
    Accept { offer: String },
    /// Reject an offered conclusion with grounds, asking reconsideration
    Reject {
        offer: String,
        #[arg(long)]
        grounds: String,
    },
    /// Record a consuming task's reliance on a source offer
    Use {
        /// The consuming task (t-<n>)
        task: String,
        /// The source offer (c-<seq>)
        offer: String,
        #[arg(long)]
        reason: String,
    },
    /// Revise a comment's prose: the fold presents the latest and names
    /// a reviser who differs from the birth author, the log keeps the
    /// original
    Revise {
        id: String,
        /// The revised prose, as an argument
        #[arg(required_unless_present_any = ["stdin", "body_file"])]
        body: Option<String>,
        /// Read the revised prose from stdin instead of an argument
        #[arg(long, conflicts_with_all = ["body", "body_file"])]
        stdin: bool,
        /// Read the revised prose from a file instead of an argument
        #[arg(long, value_name = "FILE", conflicts_with_all = ["body", "stdin"])]
        body_file: Option<PathBuf>,
    },
    /// Withdraw a note comment: the body leaves the fold, a tombstone
    /// with the note stays
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
    /// Everything about tasks and records, many at once; bare seqs show
    /// the raw event, t-<n> shows the task plus its thread
    Show {
        /// Ids in any mix: c-<seq> a record — comments and artifacts as
        /// their blocks, every other event as itself — bare <seq> the
        /// raw event at that position — t-<n> a whole thread
        #[arg(required_unless_present = "stdin")]
        ids: Vec<String>,
        /// Read one id per line from stdin instead of arguments; blank
        /// and invalid lines are skipped with a note
        #[arg(long)]
        stdin: bool,
        /// Show a withdrawn comment's body beside its tombstone
        #[arg(long)]
        withdrawn: bool,
    },
    /// Search the folded record: exact terms over task titles, comment
    /// prose, and explanations; an id term is a reference search
    Search {
        /// Terms to match (a record must match them all) and
        /// only-show-me facets:
        ///   in:t-N     one thread's records only
        ///   by:NAME    author, matched within names (by:pi finds pi/t-90-1)
        ///   kind:K     task, note, request, offer, accept, reject, use,
        ///              explanation, artifact
        ///
        /// An id term — 'c-907' or 't-49' — is a reference search: every
        /// record citing it or addressing it
        #[arg(verbatim_doc_comment)]
        terms: Vec<String>,
        /// With one id term: that record plus N before and after
        #[arg(short = 'C', long)]
        context: Option<usize>,
        /// Match withdrawn comment bodies too; they are omitted by default
        #[arg(long)]
        withdrawn: bool,
    },
    /// Serve the console and /api/v1 over HTTP (127.0.0.1 by default)
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
        #[arg(long, default_value_t = 8811)]
        port: u16,
    },
    /// Record a task's current branch tip as its checkpoint
    Checkpoint { id: String },
    /// Establish a task's workspace and launch the pinned harness with
    /// the initial request; a task already holding a workspace is
    /// reported and opened, never duplicated
    Start {
        id: String,
        /// The initial request, quoted
        #[arg(required_unless_present = "stdin", conflicts_with = "stdin")]
        body: Option<String>,
        /// Read the initial request from stdin instead of an argument
        #[arg(long)]
        stdin: bool,
        /// The Git base for the new branch: a ref, defaulting to the
        /// primary checkout's current HEAD
        #[arg(long)]
        base: Option<String>,
    },
    /// Open a task's workspace through the host: ensure the recorded
    /// environment exists (a missing checkout is recreated from its
    /// branch tip), launch no harness, land no records
    Open { id: String },
    /// Run a command inside a task's workspace: resolve the recorded
    /// workspace/incarnation, enter the checkout, inject the identity
    /// and routing environment, and exec the argv as given
    Exec {
        id: String,
        /// The command and its arguments, after --
        #[arg(last = true)]
        argv: Vec<String>,
    },
    /// Report this session's environment-declared identity and routing:
    /// disclosure, never a liveness claim or proof of ownership
    Identity,
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
    /// Migrate a Saccade 0.12.0 log backup (schema 2) to this binary's
    /// schema, stripping the retired task parent links; the input is
    /// read-only, the output path must be fresh
    Migrate {
        /// The schema-2 backup to read; never written
        #[arg(long = "in")]
        source: PathBuf,
        /// The fresh tracker to write; must not exist
        #[arg(long)]
        out: PathBuf,
    },
    /// Deploy this binary's embedded skill, or verify the copy in this repo
    Skill {
        #[command(subcommand)]
        verb: SkillVerb,
    },
    /// Park an artifact on a task's thread: hash the file into the
    /// store, record the pointer — the bytes never ride the wire
    Artifact {
        /// The task whose thread holds it (t-<n>)
        task: String,
        /// The file whose bytes become the artifact
        path: PathBuf,
        /// The artifact's name; defaults to the file's basename
        #[arg(long)]
        name: Option<String>,
    },
    /// Block until matching activity lands on the target: a task
    /// wait matches records folding into the task, a comment wait
    /// matches its reply chain (revisions and decisions included).
    /// Matches are strictly after the given global position; nothing
    /// is recorded
    Wait {
        /// The task (t-<n>) or comment (c-<seq>) to wait on
        target: String,
        /// The global record position matched strictly after — the
        /// cursor a read disclosed, never defaulted
        #[arg(long)]
        after: u64,
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

#[derive(Debug)]
enum Fail {
    Db(db::DbError),
    Degraded(String),
    Reject(Reject),
    /// A refusal the world taught: the code plus what was probably meant.
    Taught {
        code: &'static str,
        text: String,
    },
    /// The account-name identity fallback refused a non-tty write:
    /// identity is possessed or named, never inherited from the shell.
    IdentityFallback {
        actor: String,
        tier: &'static str,
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
            Fail::IdentityFallback { .. } => "identity_fallback",
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
            Fail::IdentityFallback { actor, tier } => write!(
                f,
                "this write would record {actor}/{tier} with no SACCADE_ACTOR and no terminal \
                 attached (--actor rides the terminal arm); export SACCADE_ACTOR=<name>"
            ),
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
        } => Command::CreateTask {
            name: Prose::new(name.clone())?,
        },
        Cmd::Claim { id } => {
            let id = parse_task_id(id)?;
            // an incarnation claim names its reference explicitly: the
            // wrapper's env, never the task scope alone
            let incarnation = context_scope()?.and_then(|s| s.incarnation);
            Command::ClaimTask { id, incarnation }
        }
        Cmd::Cancel { id, explanation } => Command::CancelTask {
            id: parse_task_id(id)?,
            explanation: Prose::new(explanation.clone())?,
        },
        Cmd::Reopen { id, explanation } => Command::ReopenTask {
            id: parse_task_id(id)?,
            explanation: Prose::new(explanation.clone())?,
        },
        Cmd::Comment { target, body } => Command::Comment {
            target: parse_target(target)?,
            kind: CommentKind::Note {
                body: Prose::new(body.clone())?,
            },
        },
        Cmd::Request { target, body } => Command::Comment {
            target: parse_target(target)?,
            kind: CommentKind::Request {
                body: Prose::new(body.clone())?,
            },
        },
        Cmd::Offer { id, body } => Command::Comment {
            target: Target::Task(parse_task_id(id)?),
            kind: CommentKind::Offer {
                body: Prose::new(body.clone())?,
            },
        },
        Cmd::Accept { offer } => {
            let offer = parse_comment_id(offer)?;
            Command::Comment {
                target: Target::Comment(offer),
                kind: CommentKind::Accept { offer },
            }
        }
        Cmd::Reject { offer, grounds } => {
            let offer = parse_comment_id(offer)?;
            Command::Comment {
                target: Target::Comment(offer),
                kind: CommentKind::Reject {
                    offer,
                    grounds: Prose::new(grounds.clone())?,
                },
            }
        }
        Cmd::Use {
            task,
            offer,
            reason,
        } => Command::Comment {
            target: Target::Task(parse_task_id(task)?),
            kind: CommentKind::Use {
                offer: parse_comment_id(offer)?,
                reason: Prose::new(reason.clone())?,
            },
        },
        Cmd::Revise {
            id,
            body,
            stdin,
            body_file,
        } => Command::ReviseComment {
            id: parse_revision_id(id)?,
            body: Prose::new(revise_body(body.clone(), *stdin, body_file.clone())?)?,
        },
        Cmd::Withdraw { id, note } => Command::WithdrawComment {
            id: parse_comment_id(id)?,
            note: Prose::new(note.clone())?,
        },
        Cmd::Artifact { task, path, name } => {
            let id = parse_task_id(task)?;
            let repo_root = match &cli.repo {
                Some(repo) => repo
                    .canonicalize()
                    .map_err(|e| Fail::Usage(format!("--repo {}: {e}", repo.display())))?,
                None => saccade::paths::repo_root(std::path::Path::new(".")).map_err(|_| {
                    Fail::Usage(
                        "the working directory is not inside a git repository; pass --repo so artifact can resolve the store".into(),
                    )
                })?,
            };
            let bytes = std::fs::read(path)
                .map_err(|e| Fail::Usage(format!("cannot read {}: {e}", path.display())))?;
            let artifact = Artifact {
                name: Prose::new(match name {
                    Some(given) => given.clone(),
                    None => path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "artifact".into()),
                })?,
                hash: ContentHash::of(&bytes),
            };
            // bytes land in the store before the record names them: a
            // crash can orphan content, never point at its absence
            let store = saccade::paths::artifacts_at(&repo_root);
            let home = store.join(artifact.hash.as_str());
            if !home.exists() {
                std::fs::create_dir_all(&store)
                    .map_err(|e| Fail::Usage(format!("artifact store {}: {e}", store.display())))?;
                std::fs::write(&home, &bytes)
                    .map_err(|e| Fail::Usage(format!("artifact store {}: {e}", home.display())))?;
            }
            Command::Artifact { root: id, artifact }
        }
        Cmd::List { .. } | Cmd::Log | Cmd::Show { .. } | Cmd::Search { .. } => {
            return read_only(cli, &db_path);
        }
        Cmd::Wait { target, after } => return wait_door(cli, &db_path, target, *after),
        Cmd::Start {
            id,
            body,
            stdin,
            base,
        } => return start_door(cli, &db_path, id, body.clone(), *stdin, base.clone()),
        Cmd::Open { id } => return open_door(cli, &db_path, id),
        Cmd::Exec { id, argv } => return exec_door(cli, &db_path, id, argv),
        Cmd::Identity => return identity_door(cli),
        Cmd::Serve { bind, port } => {
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
            return match runtime.block_on(saccade::serve::run(&db_path, bind, *port)) {
                Err(e) => Err(Fail::Usage(e)),
                Ok(infallible) => match infallible {},
            };
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
        Cmd::Migrate { source, out } => {
            let report = db::migrate(source, out).map_err(Fail::Db)?;
            return Ok(if cli.json {
                serde_json::json!({
                    "migrated": report.records,
                    "schema": {"from": db::MIGRATED_FROM, "to": db::SCHEMA_VERSION},
                    "stripped": report.stripped.iter().map(|l| serde_json::json!({
                        "seq": l.seq,
                        "task": l.task,
                        "parent": l.parent,
                    })).collect::<Vec<_>>(),
                    "out": out.display().to_string(),
                })
                .to_string()
            } else {
                let mut lines = vec![format!(
                    "migrated {} records to {} (schema {} → {}); {} parent links stripped",
                    report.records,
                    out.display(),
                    db::MIGRATED_FROM,
                    db::SCHEMA_VERSION,
                    report.stripped.len()
                )];
                lines.extend(report.stripped.iter().map(|l| {
                    format!(
                        "  c-{} birth of t-{}, parented t-{}",
                        l.seq, l.task, l.parent
                    )
                }));
                lines.join("\n")
            });
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
            let branch: String = workspace.workspace.branch.clone().into();
            let tip = sh_git(
                &repo_root,
                &["rev-parse", "--verify", &format!("refs/heads/{branch}")],
            )
            .map_err(Fail::Usage)?;
            let recorded = workspace.workspace.checkpoint.as_str();
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
    identity_fallback_guard(cli, &context, attached_terminal())?;
    let creates = matches!(command, Command::CreateTask { .. });
    let mut born: Option<String> = None;
    let mut request: Option<RequestFacts> = None;
    let stored = if cli.offline {
        let now = cli.at.unwrap_or_else(db::now_epoch);
        let mut conn = db::open(&db_path).map_err(Fail::Db)?;
        match db::record(&mut conn, &context, command.clone(), now) {
            Ok((stored, world)) => {
                if creates {
                    born = born_of(&stored, &world);
                }
                request = request_of(&command, &stored, |seq| {
                    saccade::views::request_facts(&world, seq)
                });
                stored
            }
            Err(ExecuteFail::Reject(reject)) => return Err(refused(&conn, &command, reject)),
            Err(other) => return Err(Fail::from(other)),
        }
    } else {
        client::handshake(&cli.server);
        let reply = client::send(&cli.server, &context, command, cli.at).map_err(Fail::Client)?;
        born = reply.id;
        if let Some(seq) = reply.seq {
            request = Some(RequestFacts {
                seq,
                task: reply.task.expect("a request reply names its task"),
                workspace: reply.workspace,
            });
        }
        reply.records
    };
    Ok(render_records(
        cli,
        &stored,
        born.as_deref(),
        request.as_ref(),
    ))
}

fn request_of(
    command: &Command,
    stored: &[db::StoredRecord],
    facts: impl FnOnce(RecordId) -> Option<RequestFacts>,
) -> Option<RequestFacts> {
    if !matches!(
        command,
        Command::Comment {
            kind: CommentKind::Request { .. },
            ..
        }
    ) {
        return None;
    }
    let record = stored.first()?;
    facts(RecordId(record.seq))
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
            .map(|d| format!("; drop the '#': the comment is addressed as c-{d}"))
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
        scope: context_scope()?,
    })
}

/// The environment-declared scope: SACCADE_TASK anchors it
/// ("t-<n>"), SACCADE_INCARNATION names a record-derived worker
/// ("i-<seq>", the establishment birth's position), SACCADE_WORKSPACE
/// names the workspace address (an absolute path). Disclosure riding
/// every record, never proof; the incarnation and workspace keys ride
/// only with the task anchor.
fn context_scope() -> Result<Option<Scope>, Fail> {
    let Some(task) = std::env::var("SACCADE_TASK").ok().filter(|t| !t.is_empty()) else {
        return Ok(None);
    };
    let task = parse_task_id(&task)?;
    let incarnation = match std::env::var("SACCADE_INCARNATION") {
        Ok(value) if !value.is_empty() => {
            let id = parse_incarnation_ref(&value).map_err(|_| {
                Fail::Usage(format!(
                    "SACCADE_INCARNATION='{value}' is not an i-<seq> reference"
                ))
            })?;
            Some(id)
        }
        _ => None,
    };
    let workspace = match std::env::var("SACCADE_WORKSPACE") {
        Ok(value) if !value.is_empty() => {
            Some(WorktreePath::new(value.clone().into()).map_err(|_| {
                Fail::Usage(format!(
                    "SACCADE_WORKSPACE='{value}' is not an absolute path"
                ))
            })?)
        }
        _ => None,
    };
    Ok(Some(Scope {
        task,
        incarnation,
        workspace,
    }))
}

fn attached_terminal() -> bool {
    std::io::stdin().is_terminal()
        || std::io::stdout().is_terminal()
        || std::io::stderr().is_terminal()
}

/// The write door's guard: possession passes any context; without it,
/// a terminal attached anywhere or nothing. A terminal attached to
/// any of stdin, stdout, or stderr earns the write — the unnamed
/// fallback on one notice line, a named --actor silently — and a
/// caller with no terminal and no SACCADE_ACTOR (scripts, spawned
/// agents that forgot their identity) refuses before any write,
/// taught the fix. Possession is read from the context's tier, never
/// the --actor flag: clap binds SACCADE_ACTOR into that flag, so it
/// cannot tell possession from a passed name.
fn identity_fallback_guard(cli: &Cli, context: &Context, interactive: bool) -> Result<(), Fail> {
    if matches!(context.tier, Tier::Agent) {
        return Ok(());
    }
    if interactive {
        if cli.actor.is_none() {
            eprintln!("{}", identity_fallback_notice(context));
        }
        return Ok(());
    }
    Err(Fail::IdentityFallback {
        actor: context.actor.as_str().to_owned(),
        tier: saccade::wire::tier_of(&context.tier),
    })
}

/// The keyboard's one notice line: the identity this write records,
/// and how an agent under this account names itself instead.
fn identity_fallback_notice(context: &Context) -> String {
    format!(
        "note: this write records {}/{} from the identity fallback \
         (no SACCADE_ACTOR, no --actor); agents export SACCADE_ACTOR=<name>",
        context.actor.as_str(),
        saccade::wire::tier_of(&context.tier)
    )
}
/// The primary repo root: --repo names it, else the repo containing
/// the working directory (a worktree resolves to its main checkout).
fn resolve_repo(cli: &Cli) -> Result<PathBuf, Fail> {
    match &cli.repo {
        Some(repo) => repo
            .canonicalize()
            .map_err(|e| Fail::Usage(format!("--repo {}: {e}", repo.display()))),
        None => saccade::paths::repo_root(std::path::Path::new(".")).map_err(Fail::Usage),
    }
}

/// The db anchored absolute: a relative --db must still name the same
/// tracker once the session sits inside the checkout.
fn canonical_db(db_path: &std::path::Path) -> PathBuf {
    std::fs::canonicalize(db_path).unwrap_or_else(|_| db_path.to_path_buf())
}

/// The start door: the server establishes through its own Git effects
/// and lands the caller's request with the incarnation ownership; the
/// host then carries one terminal carrying the pinned harness. A task
/// already holding a workspace is reported — nothing duplicated.
fn start_door(
    cli: &Cli,
    db_path: &std::path::Path,
    id: &str,
    body: Option<String>,
    stdin: bool,
    base: Option<String>,
) -> Result<String, Fail> {
    if cli.offline {
        return Err(Fail::Usage(
            "start runs through the server: the sole writer establishes workspaces; --offline cannot".into(),
        ));
    }
    let task = parse_task_id(id)?;
    let request = Prose::new(start_body(body, stdin)?)?;
    let context = context_of(cli)?;
    identity_fallback_guard(cli, &context, attached_terminal())?;
    client::handshake(&cli.server);
    let reply = client::start_workspace(&cli.server, &context, task.0, base.as_deref(), &request)
        .map_err(Fail::Client)?;
    let db_path = canonical_db(db_path);
    let host = if reply.established {
        launch_harness(cli, &db_path, task, &reply, &request)?
    } else {
        open_through_host(&reply.workspace.path, false).map_err(|e| {
            Fail::Usage(format!(
                "workspace stands at {}; the host would not open it: {e}",
                reply.workspace.path
            ))
        })?
    };
    render_workspace_reply(cli, &reply, &host)
}

/// The open door: ensure the recorded environment exists (a missing
/// checkout is recreated from its branch tip, identity unchanged); an
/// unestablished task births the association alone. No harness launch,
/// no request, no ownership change.
fn open_door(cli: &Cli, db_path: &std::path::Path, id: &str) -> Result<String, Fail> {
    let _ = db_path;
    if cli.offline {
        return Err(Fail::Usage(
            "open runs through the server: the sole writer establishes workspaces; --offline cannot".into(),
        ));
    }
    let task = parse_task_id(id)?;
    client::handshake(&cli.server);
    let reply = client::open_workspace(&cli.server, task.0).map_err(Fail::Client)?;
    let host = open_through_host(&reply.workspace.path, reply.established).map_err(|e| {
        Fail::Usage(format!(
            "workspace stands at {}; the host would not open it: {e}",
            reply.workspace.path
        ))
    })?;
    render_workspace_reply(cli, &reply, &host)
}

/// The exec door: enter the checkout, inject identity and routing over
/// any stale inherited scope, and exec the argv with its boundaries
/// intact. The actor is always supplied — the caller's declared
/// attribution preserved, else the command's own label — so no session
/// of the wrapper writes as an unpossessed human.
fn exec_door(
    cli: &Cli,
    db_path: &std::path::Path,
    id: &str,
    argv: &[String],
) -> Result<String, Fail> {
    let [program, rest @ ..] = argv else {
        return Err(Fail::Usage("exec needs a command after --".into()));
    };
    let task = parse_task_id(id)?;
    let conn = db::open_read(db_path).map_err(Fail::Db)?;
    let loadout = db::load(&conn).map_err(Fail::Db)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "exec needs the world; the log will not fold".into(),
        ));
    };
    let ctx = world.tasks.get(task.0).ok_or(Reject::InvalidTaskId)?;
    let workspace = ctx.workspace.as_ref().ok_or(Reject::WorkspaceMissing)?;
    let repo = resolve_repo(cli)?;
    let db_path = canonical_db(db_path);
    let actor = std::env::var("SACCADE_ACTOR")
        .ok()
        .filter(|a| !a.is_empty())
        .unwrap_or_else(|| command_label(program));
    let env = injected_environment(task, workspace, &actor, &repo, &db_path, &cli.server);
    let checkout = workspace.workspace.worktree.as_path();
    std::env::set_current_dir(checkout).map_err(|e| {
        Fail::Usage(format!(
            "the checkout {} is missing: 'sac open t-{}' recreates it from the branch — {e}",
            checkout.display(),
            task.0
        ))
    })?;
    let err = std::os::unix::process::CommandExt::exec(
        std::process::Command::new(program).args(rest).envs(env),
    );
    Err(Fail::Usage(format!("cannot exec {program}: {err}")))
}

/// A launched command's attribution label: its own file name.
fn command_label(program: &str) -> String {
    std::path::Path::new(program)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "saccade".into())
}

/// The identity door: the environment's declaration plus routing,
/// read-only; disclosure, never a liveness claim.
fn identity_door(cli: &Cli) -> Result<String, Fail> {
    let context = context_of(cli)?;
    let scope = context.scope.clone();
    let repo = resolve_repo(cli).ok();
    let db_path = resolve_db(cli).ok().map(|p| canonical_db(&p));
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "actor": context.actor.as_str(),
            "tier": saccade::wire::tier_of(&context.tier),
            "task": scope.as_ref().map(|s| format!("t-{}", s.task.0)),
            "incarnation": scope.as_ref().and_then(|s| s.incarnation.map(|i| format!("i-{}", i.0 .0))),
            "workspace": scope.as_ref().and_then(|s| s.workspace.as_ref().map(|w| w.as_path().display().to_string())),
            "repo": repo.map(|p| p.display().to_string()),
            "db": db_path.map(|p| p.display().to_string()),
            "server": cli.server,
        }))
        .expect("identity is plain data"));
    }
    let mut out = vec![format!(
        "actor  {} ({})",
        context.actor.as_str(),
        saccade::wire::tier_of(&context.tier)
    )];
    match &scope {
        Some(scope) => {
            out.push(format!("task  t-{}", scope.task.0));
            if let Some(incarnation) = scope.incarnation {
                out.push(format!("incarnation  i-{}", incarnation.0.0));
            }
            if let Some(workspace) = &scope.workspace {
                out.push(format!("workspace  {}", workspace.as_path().display()));
            }
        }
        None => out.push("task  none declared".into()),
    }
    if let Some(repo) = repo {
        out.push(format!("repo  {}", repo.display()));
    }
    if let Some(db) = db_path {
        out.push(format!("db  {}", db.display()));
    }
    out.push(format!("server  {}", cli.server));
    Ok(out.join("\n"))
}

/// The environment a task's sessions run under, overwritten never
/// merged: actor attribution, scope from the record, tracker routing.
fn injected_environment(
    task: TaskId,
    workspace: &saccade::objects::workspace::WorkspaceContext,
    actor: &str,
    repo: &std::path::Path,
    db_path: &std::path::Path,
    server: &str,
) -> Vec<(String, String)> {
    vec![
        ("SACCADE_ACTOR".into(), actor.to_string()),
        ("SACCADE_TASK".into(), format!("t-{}", task.0)),
        (
            "SACCADE_INCARNATION".into(),
            format!("i-{}", workspace.birth.0),
        ),
        (
            "SACCADE_WORKSPACE".into(),
            workspace.workspace.worktree.as_path().display().to_string(),
        ),
        ("SACCADE_REPO".into(), repo.display().to_string()),
        ("SACCADE_DB".into(), db_path.display().to_string()),
        ("SACCADE_SERVER".into(), server.to_string()),
    ]
}

/// The start door's body source: the quoted argument or the pipe.
fn start_body(body: Option<String>, stdin: bool) -> Result<String, Fail> {
    if stdin {
        if std::io::stdin().is_terminal() {
            return Err(Fail::Usage(
                "nothing piped: --stdin reads the request".into(),
            ));
        }
        return std::io::read_to_string(std::io::stdin())
            .map_err(|e| Fail::Usage(format!("stdin: {e}")));
    }
    body.ok_or_else(|| Fail::Usage("the request comes quoted or through --stdin".into()))
}

/// The pinned native Pi: the development override first, then the
/// flake pin this binary was built with. Never an arbitrary PATH pi.
fn pinned_pi() -> Result<PathBuf, Fail> {
    if let Some(path) = std::env::var_os("SACCADE_PI").filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = option_env!("SACCADE_PI_BIN") {
        return Ok(PathBuf::from(path));
    }
    Err(Fail::Usage(
        "no pinned pi: this binary carries no flake pin; export SACCADE_PI=<pi> to name one".into(),
    ))
}

/// One POSIX word, quoted so any bytes survive the shell that types it.
fn shell_quote(word: &str) -> String {
    if !word.is_empty()
        && word.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'=' | b'/' | b'.' | b'+')
        })
    {
        return word.to_string();
    }
    let mut out = String::with_capacity(word.len() + 2);
    out.push('\'');
    for c in word.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// What the host did with the workspace.
#[derive(serde::Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum HostOpen {
    /// First start: the harness command was submitted into a new
    /// terminal — acceptance is the evidence, execution is unobserved.
    Launched { terminal: String, command: String },
    /// A new shell terminal in the worktree (a birth's open).
    Created { terminal: String },
    /// A live terminal the host confirmed it brought to the front.
    Revealed { terminal: String },
    /// The workspace's address stands as the report; `live` names a
    /// terminal the host holds but would not bring forward.
    Reported { path: String, live: Option<String> },
}

fn orca(args: &[&str]) -> Result<std::process::Output, String> {
    std::process::Command::new("orca")
        .args(args)
        .output()
        .map_err(|e| format!("orca {}: {e}", args.first().unwrap_or(&"")))
}

/// Wait for the host's cached discovery to see the worktree: bounded,
/// exact-path, never a second checkout.
fn orca_await_worktree(path: &str) -> Result<(), String> {
    let selector = format!("path:{path}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
    loop {
        let out = orca(&["worktree", "show", "--worktree", &selector])?;
        if out.status.success() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "orca has not discovered {path} yet (bounded at 90s); 'sac open t-<n>' opens it once it has",
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }
}

/// The first live terminal the host holds in the worktree, if any.
fn live_terminal(path: &str) -> Result<Option<String>, String> {
    let out = orca(&["terminal", "list", "--json"])?;
    if !out.status.success() {
        return Err(format!(
            "orca terminal list failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let parsed: Option<String> =
        serde_json::from_str::<serde_json::Value>(&String::from_utf8_lossy(&out.stdout))
            .ok()
            .and_then(|v| {
                v.get("result")
                    .and_then(|r| r.get("terminals"))
                    .and_then(|ts| ts.as_array())
                    .and_then(|ts| {
                        ts.iter()
                            .filter(|t| {
                                t.get("worktreePath").and_then(|p| p.as_str()) == Some(path)
                            })
                            .filter(|t| {
                                !t.get("orphaned").and_then(|o| o.as_bool()).unwrap_or(false)
                            })
                            .find_map(|t| {
                                t.get("handle").and_then(|h| h.as_str()).map(str::to_string)
                            })
                    })
            });
    Ok(parsed)
}

/// Whether the host confirmed it brought the terminal to the front:
/// its pinned answer is result.focus.navigated.
fn switch_navigated(out: &std::process::Output) -> bool {
    serde_json::from_str::<serde_json::Value>(&String::from_utf8_lossy(&out.stdout))
        .ok()
        .and_then(|v| {
            v.get("result")
                .and_then(|r| r.get("focus"))
                .and_then(|f| f.get("navigated"))
                .and_then(|n| n.as_bool())
        })
        .unwrap_or(false)
}

/// The created terminal's handle from the host's pinned response shape.
fn terminal_handle(stdout: &str) -> Result<String, String> {
    serde_json::from_str::<serde_json::Value>(stdout)
        .ok()
        .and_then(|v| {
            v.get("result")
                .and_then(|r| r.get("terminal"))
                .and_then(|t| t.get("handle"))
                .and_then(|h| h.as_str())
                .map(str::to_string)
        })
        .ok_or_else(|| "the terminal was created but the host named no handle".into())
}

/// Open a workspace through the host: a live terminal is brought
/// forward only on the host's confirmed navigation, and only a birth's
/// open creates one — a standing workspace is reported, never grown
/// another tab.
fn open_through_host(path: &str, create_when_none: bool) -> Result<HostOpen, String> {
    if let Some(handle) = live_terminal(path)? {
        let out = orca(&["terminal", "switch", "--terminal", &handle, "--json"])?;
        if out.status.success() && switch_navigated(&out) {
            return Ok(HostOpen::Revealed { terminal: handle });
        }
        return Ok(HostOpen::Reported {
            path: path.to_string(),
            live: Some(handle),
        });
    }
    if !create_when_none {
        return Ok(HostOpen::Reported {
            path: path.to_string(),
            live: None,
        });
    }
    orca_await_worktree(path)?;
    let out = orca(&[
        "terminal",
        "create",
        "--worktree",
        &format!("path:{path}"),
        "--json",
    ])?;
    if !out.status.success() {
        return Err(format!(
            "orca terminal create failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let terminal = terminal_handle(&String::from_utf8_lossy(&out.stdout))?;
    Ok(HostOpen::Created { terminal })
}

/// Submit the pinned harness into a new host terminal at the workspace,
/// the request as its literal prompt: a self-contained sac exec
/// invocation, dev environment activated when the worktree carries an
/// .envrc, the harness's label supplied as the session's actor.
fn launch_harness(
    cli: &Cli,
    db_path: &std::path::Path,
    task: TaskId,
    reply: &client::WorkspaceReply,
    request: &Prose,
) -> Result<HostOpen, Fail> {
    let pi = pinned_pi()?;
    let label = command_label(&pi.display().to_string());
    let repo = resolve_repo(cli)?;
    let sac = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("sac"));
    // an assignment is not a word: its value is quoted here, and it
    // never passes through the word quoting below
    let assignment = format!("SACCADE_ACTOR={}", shell_quote(&label));
    let mut argv = vec![sac.display().to_string()];
    argv.extend([
        "--server".to_string(),
        cli.server.clone(),
        "--db".to_string(),
        db_path.display().to_string(),
        "--repo".to_string(),
        repo.display().to_string(),
        "exec".to_string(),
        format!("t-{}", task.0),
        "--".to_string(),
        pi.display().to_string(),
        request.as_str().to_string(),
    ]);
    let worktree = &reply.workspace.path;
    let direnv = std::path::Path::new(worktree).join(".envrc");
    if direnv.is_file() {
        // the assignment leads the typed line and the dev environment
        // wraps the command: SACCADE_ACTOR=… direnv exec <wt> <sac> …
        argv.splice(
            0..0,
            ["direnv".to_string(), "exec".to_string(), worktree.clone()],
        );
    }
    let command = std::iter::once(assignment)
        .chain(argv.iter().map(|w| shell_quote(w)))
        .collect::<Vec<_>>()
        .join(" ");
    orca_await_worktree(worktree).map_err(Fail::Usage)?;
    let out = orca(&[
        "terminal",
        "create",
        "--worktree",
        &format!("path:{worktree}"),
        "--title",
        &format!("sac t-{}", task.0),
        "--command",
        &command,
        "--json",
    ])
    .map_err(Fail::Usage)?;
    if !out.status.success() {
        return Err(Fail::Usage(format!(
            "the workspace stands at {worktree} and the records landed, but orca terminal create failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let terminal = terminal_handle(&String::from_utf8_lossy(&out.stdout)).map_err(Fail::Usage)?;
    Ok(HostOpen::Launched { terminal, command })
}

/// The workspace doors' report: every claim read off the records that
/// actually landed, and what the host did.
fn render_workspace_reply(
    cli: &Cli,
    reply: &client::WorkspaceReply,
    host: &HostOpen,
) -> Result<String, Fail> {
    if cli.json {
        let mut value = serde_json::to_value(reply).expect("the reply is plain data");
        value["host"] = serde_json::to_value(host).expect("the host report is plain data");
        return Ok(serde_json::to_string_pretty(&value).expect("plain data"));
    }
    let mut out = Vec::new();
    let requested = reply.records.iter().any(|r| r.kind == "commented");
    let claimed = reply.records.iter().any(|r| r.kind == "task_claimed");
    if reply.established && requested && claimed {
        out.push(format!(
            "t-{} workspace established — request recorded, ownership names i-{}",
            reply.workspace.task, reply.incarnation
        ));
    } else if reply.established {
        out.push(format!(
            "t-{} workspace association born — no request or ownership recorded",
            reply.workspace.task
        ));
    } else if reply.recreated {
        out.push(format!(
            "t-{} workspace stands — checkout recreated from the branch tip, no duplicate records",
            reply.workspace.task
        ));
    } else {
        out.push(format!(
            "t-{} workspace stands — no duplicate launch, request, or ownership change",
            reply.workspace.task
        ));
    }
    out.push(format!(
        "  branch {} · base {}",
        reply.workspace.branch, reply.workspace.base
    ));
    out.push(format!("  path {}", reply.workspace.path));
    match host {
        HostOpen::Launched { terminal, .. } => out.push(format!(
            "  host terminal {terminal} · harness command submitted (execution unobserved)"
        )),
        HostOpen::Created { terminal } => out.push(format!("  host terminal {terminal} · shell")),
        HostOpen::Revealed { terminal } => {
            out.push(format!("  host terminal {terminal} · revealed"))
        }
        HostOpen::Reported { path, live: None } => out.push(format!(
            "  host · no live terminal; the workspace stands at {path}"
        )),
        HostOpen::Reported {
            path,
            live: Some(handle),
        } => out.push(format!(
            "  host · terminal {handle} stands at {path}; the host would not bring it forward"
        )),
    }
    out.extend(reply.records.iter().map(record_line));
    Ok(out.join("\n"))
}

fn read_only(cli: &Cli, db_path: &std::path::Path) -> Result<String, Fail> {
    let conn = db::open_read(db_path).map_err(Fail::Db)?;
    let loadout = db::load(&conn).map_err(Fail::Db)?;
    let cursor = loadout.rows.last().map(|r| r.seq);
    match &cli.command {
        Cmd::Log => {
            if let LoadState::Degraded(reason) = &loadout.state {
                eprintln!("warning: world projection unavailable: {reason}");
            }
            let threads = match &loadout.state {
                LoadState::Full(world) => thread_index(world),
                LoadState::Degraded(_) => BTreeMap::new(),
            };
            Ok(render_log(cli, &loadout.rows, &threads, cursor))
        }
        Cmd::List { .. } => match loadout.state {
            LoadState::Full(world) => Ok(render_tasks(cli, &world, cursor)),
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
        },
        Cmd::Show {
            ids,
            stdin,
            withdrawn,
        } => match loadout.state {
            LoadState::Full(world) => {
                if *stdin {
                    return show_stdin(cli, &world, &loadout.rows, ids, *withdrawn, cursor);
                }
                if cli.json {
                    let blocks: Vec<serde_json::Value> = ids
                        .iter()
                        .map(|token| show_value(&world, &loadout.rows, token, *withdrawn))
                        .collect::<Result<_, _>>()?;
                    return Ok(serde_json::to_string_pretty(&serde_json::json!({
                        "blocks": blocks,
                        "cursor": cursor,
                    }))
                    .expect("blocks are plain data"));
                }
                let mut blocks = Vec::new();
                for token in ids {
                    blocks.push(render_id(&world, &loadout.rows, token, *withdrawn)?);
                }
                Ok(format!("{}\n{}", blocks.join("\n\n"), cursor_line(cursor)))
            }
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
        },
        Cmd::Search {
            terms,
            context,
            withdrawn,
        } => {
            let mut query = SearchQuery::parse(terms).map_err(|f| Fail::Usage(f.to_string()))?;
            query.withdrawn = *withdrawn;
            if let Some(around) = context {
                let anchor = anchor_seq(&loadout, &query)?;
                return Ok(render_anchor(cli, &loadout.rows, anchor, *around, cursor));
            }
            match loadout.state {
                LoadState::Full(world) => {
                    let groups = search(&world, &query).map_err(|f| Fail::Usage(f.to_string()))?;
                    Ok(render_search(cli, &groups, terms, &query, cursor))
                }
                LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
            }
        }
        _ => unreachable!("read_only reached from a mutating command"),
    }
}

const WAIT_POLL: std::time::Duration = std::time::Duration::from_millis(250);

fn wait_door(
    cli: &Cli,
    db_path: &std::path::Path,
    target: &str,
    after: u64,
) -> Result<String, Fail> {
    let target = parse_target(target)?;
    let conn = db::open_read(db_path).map_err(Fail::Db)?;
    loop {
        let loadout = db::load(&conn).map_err(Fail::Db)?;
        let world = match loadout.state {
            LoadState::Full(world) => world,
            LoadState::Degraded(reason) => return Err(Fail::Degraded(reason)),
        };
        wait_target_exists(&world, &target, loadout.rows.len())?;
        let matched: Vec<StoredRecord> = loadout
            .rows
            .iter()
            .filter(|r| r.seq > after as usize)
            .filter(|r| {
                let event =
                    wire::assemble(&r.kind, &r.payload).expect("a full fold assembled every row");
                saccade::views::wait_matches(&world, RecordId(r.seq), &event, &target)
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

fn wait_target_exists(world: &saccade::World, target: &Target, records: usize) -> Result<(), Fail> {
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
        Target::Comment(id) => {
            if !world.comments.contains_key(id) {
                return Err(Fail::Usage(format!(
                    "no record c-{}; the log holds {} records, 0 through {}",
                    id.0.0,
                    records,
                    records.saturating_sub(1)
                )));
            }
        }
    }
    Ok(())
}

fn render_wait(cli: &Cli, matched: &[StoredRecord], cursor: Option<usize>) -> String {
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

/// Every text read ends with this line, feedable to the next `wait --after`.
fn cursor_line(cursor: Option<usize>) -> String {
    match cursor {
        Some(c) => format!("cursor {c}"),
        None => "cursor none".into(),
    }
}

/// The 'i-<seq>' face the incarnation reference carries.
fn parse_incarnation_ref(token: &str) -> Result<IncarnationId, Fail> {
    let n = token.strip_prefix("i-").ok_or_else(|| {
        Fail::Usage(format!(
            "'{token}' is not an incarnation reference (expected i-<seq>)"
        ))
    })?;
    let n: usize = n
        .parse()
        .map_err(|_| Fail::Usage(format!("'{token}' is not an incarnation reference")))?;
    Ok(IncarnationId(RecordId(n)))
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
    let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if let Some(n) = token.strip_prefix("c-").filter(|n| numeric(n)) {
        return Ok(Target::Comment(CommentId(RecordId(
            n.parse().expect("digits checked"),
        ))));
    }
    if let Some(rest) = token.strip_prefix('#') {
        let note = if numeric(rest) {
            format!("; drop the '#': the comment is addressed as c-{rest}")
        } else {
            hashed_record_note(rest)
        };
        return Err(Fail::Usage(format!(
            "'{token}' is not a comment id (expected c-<seq>){note}"
        )));
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
        return format!("; drop the '#': the comment is addressed as c-{n}");
    }
    String::new()
}

/// The revise door's body sources: the argument, the pipe, or the file
/// — one of the three, the bytes landing verbatim.
fn revise_body(body: Option<String>, stdin: bool, file: Option<PathBuf>) -> Result<String, Fail> {
    if stdin {
        if std::io::stdin().is_terminal() {
            return Err(Fail::Usage("nothing piped: --stdin reads the body".into()));
        }
        return std::io::read_to_string(std::io::stdin())
            .map_err(|e| Fail::Usage(format!("stdin: {e}")));
    }
    if let Some(path) = file {
        return std::fs::read_to_string(&path)
            .map_err(|e| Fail::Usage(format!("cannot read {}: {e}", path.display())));
    }
    body.ok_or_else(|| Fail::Usage("the body comes by argument, --stdin, or --body-file".into()))
}

/// The 'c-<seq>' face the revise door takes; the retired '#' prefix
/// learns its replacement.
fn parse_revision_id(token: &str) -> Result<CommentId, Fail> {
    let rest = token.strip_prefix("c-").ok_or_else(|| {
        let note = token
            .strip_prefix('#')
            .filter(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
            .map(|d| format!("; drop the '#': the comment is addressed as c-{d}"))
            .unwrap_or_default();
        Fail::Usage(format!(
            "'{token}' is not a comment id (expected c-<seq>){note}"
        ))
    })?;
    let n: usize = rest
        .parse()
        .map_err(|_| Fail::Usage(format!("'{token}' is not a comment id")))?;
    Ok(CommentId(RecordId(n)))
}

/// The reference a create reply names: the task whose birth record just
/// landed, resolved against the post-write fold.
fn born_of(stored: &[StoredRecord], world: &World) -> Option<String> {
    let birth = stored.iter().find(|r| r.kind == "task_created")?;
    world
        .task_born_at(RecordId(birth.seq))
        .map(|id| format!("t-{}", id.0))
}

fn render_records(
    cli: &Cli,
    stored: &[StoredRecord],
    born: Option<&str>,
    request: Option<&RequestFacts>,
) -> String {
    if cli.json {
        // the wire's reply shape: records always, the born task's token
        // only on a create
        let mut reply = serde_json::json!({"records": stored});
        if let Some(id) = born {
            reply["id"] = serde_json::json!(id);
        }
        if let Some(request) = request {
            reply["seq"] = serde_json::json!(request.seq);
            reply["task"] = serde_json::json!(request.task);
            reply["workspace"] = serde_json::json!(request.workspace);
        }
        return serde_json::to_string_pretty(&reply).expect("records are plain data");
    }
    let mut lines = Vec::with_capacity(stored.len() + 2);
    if let Some(id) = born {
        lines.push(id.to_string());
    }
    lines.extend(stored.iter().map(record_line));
    if let Some(workspace) = request.and_then(|r| r.workspace.as_ref()) {
        lines.push(format!(
            "  workspace t-{} · {} · {}",
            workspace.task, workspace.branch, workspace.path
        ));
    }
    lines.join("\n")
}

fn render_log(
    cli: &Cli,
    rows: &[StoredRecord],
    threads: &BTreeMap<usize, TaskId>,
    cursor: Option<usize>,
) -> String {
    if cli.json {
        // the log row's own grammar, plus the fold's thread membership:
        // a row with no home thread carries null, never absence
        let out: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                let mut row = serde_json::to_value(r).expect("records are plain data");
                row["thread"] = match threads.get(&r.seq) {
                    Some(task) => serde_json::json!(format!("t-{}", task.0)),
                    None => serde_json::Value::Null,
                };
                row
            })
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

fn record_line(r: &StoredRecord) -> String {
    format!(
        "c-{} {} {}/{} et={} lt={} {}",
        r.seq, r.kind, r.actor, r.tier, r.event_time, r.logged_time, r.payload
    )
}

fn render_tasks(cli: &Cli, world: &World, cursor: Option<usize>) -> String {
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
                    "offer": v.offer,
                    "comments": v.n_comments,
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
                "{}\t{}\t{}\t{}\t{}",
                v.id,
                v.state,
                v.name,
                v.owner.clone().unwrap_or_else(|| "-".into()),
                if v.n_comments > 0 { "#" } else { "-" }
            )
        })
        .chain(std::iter::once(cursor_line(cursor)))
        .collect::<Vec<_>>()
        .join("\n")
}

const WIDTH: usize = 80;
/// The thread header's indent ceiling: deep chains stop marching right.
const HEADER_INDENT_CAP: usize = 4;

/// The inspector dock as text: everything about the one thing. The
/// thread renders the addressed forest — items in first-record order,
/// replies carrying their target chip, prose at one constant margin.
fn render_show(world: &World, task_id: TaskId) -> Result<String, Fail> {
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
    if let Some(offer) = view.offer {
        out.push(format!("offer  c-{offer}"));
    }
    if let Some(workspace) = &view.workspace {
        out.push(format!("workspace  {workspace}"));
    }
    if let Some(explanation) = &view.explanation {
        out.push("explanation".to_string());
        out.push(wrap(explanation, WIDTH, "  ", "  "));
    }
    let thread = thread_view(world, task_id).expect("a task with a show view holds its thread");
    for item in &thread.items {
        out.push(String::new());
        match item {
            ThreadItem::Note(line) => out.extend(comment_block(line, false)),
            ThreadItem::Group { root, replies } => {
                out.extend(comment_block(root, false));
                for line in replies {
                    out.extend(comment_block(line, false));
                }
            }
            ThreadItem::Artifacts(lines) => {
                for line in lines {
                    out.extend(artifact_block(line));
                }
            }
        }
    }
    Ok(out.join("\n"))
}

/// An artifact's body format: the pointer line — name and short hash.
fn artifact_block(line: &ArtifactLine) -> Vec<String> {
    vec![
        format!("c-{}  artifact", line.seq),
        format!("  {} · {}", line.name, line.short_hash()),
    ]
}

/// A thread view's body format for one comment: its header line —
/// indent capped by depth, a reply's target chip naming the comment
/// it addresses, a use's chip naming the offer it relies on — the
/// wrapped prose at a constant margin. A withdrawn note renders its
/// tombstone, the body only when revealed.
fn comment_block(line: &CommentLine, reveal: bool) -> Vec<String> {
    let indent = "  ".repeat(line.depth.saturating_sub(1).min(HEADER_INDENT_CAP));
    let chip = match line.target {
        Target::Comment(parent) => format!("→c-{}  ", parent.0.0),
        Target::Task(_) => String::new(),
    };
    let uses = line
        .offer
        .filter(|_| line.kind == "use")
        .map(|o| format!("  uses c-{o}"))
        .unwrap_or_default();
    // the withdrawal is terminal, so its mark replaces the revised one:
    // the body is gone, its provenance mark would narrate absence —
    // and a withdrawer who differs from the birth author is named
    let mark = if line.withdrawn {
        match line.withdrawer.as_deref() {
            Some(who) => format!("withdrawn by {who}"),
            None => "withdrawn".to_string(),
        }
    } else {
        match (line.revised, line.reviser.as_deref()) {
            (true, Some(who)) => format!("revised by {who}"),
            (true, None) => "revised".to_string(),
            _ => String::new(),
        }
    };
    let state = if mark.is_empty() {
        String::new()
    } else {
        format!("  ({mark})")
    };
    let mut out = vec![format!(
        "{indent}c-{}  {chip}{} [{}]{}{state}",
        line.seq, line.actor, line.kind, uses
    )];
    if line.withdrawn {
        // the tombstone: one line, the note
        let note = line.withdrawal_note.as_deref().unwrap_or_default();
        out.push(wrap(&format!("withdrawn: {note}"), WIDTH, "  ", "  "));
        if reveal {
            out.push(wrap(&format!("body: {}", line.body), WIDTH, "  ", "  "));
        }
        return out;
    }
    if !line.body.is_empty() {
        out.push(wrap(&line.body, WIDTH, "  ", "  "));
    }
    out
}

/// One id as show sees it: a whole thread, a record's folded form,
/// or the raw event at a bare position.
#[derive(Clone, Copy, Debug)]
enum ShowId {
    Thread(TaskId),
    /// c-<seq>: the presentation the record's kind earns
    Record(CommentId),
    /// bare <seq>: the raw event at that position
    Position(RecordId),
}

fn parse_show_id(token: &str) -> Result<ShowId, Fail> {
    let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if let Some(rest) = token.strip_prefix('#').filter(|n| numeric(n)) {
        return Err(Fail::Usage(format!(
            "'{token}' is not an id (expected c-<seq>, t-<n>, or a bare <seq>); \
             drop the '#': the record is addressed as c-{rest}"
        )));
    }
    if let Some(rest) = token.strip_prefix("c-").filter(|n| numeric(n)) {
        return Ok(ShowId::Record(CommentId(RecordId(
            rest.parse().expect("digits checked"),
        ))));
    }
    if token.starts_with("t-") {
        return Ok(ShowId::Thread(parse_task_id(token)?));
    }
    if numeric(token) {
        return Ok(ShowId::Position(RecordId(
            token.parse().expect("digits checked"),
        )));
    }
    Err(Fail::Usage(format!(
        "'{token}' is not an id (expected c-<seq>, t-<n>, or a bare <seq>)"
    )))
}

/// Render one id: a thread whole, or a record as itself — a comment
/// as its block, an artifact as its pointer line, a task's birth as
/// the literal event (header and relation, no thread substitution:
/// the relation names t-N, and the taught law does the rest), and
/// every other record as the log renders it, header and payload.
fn render_id(
    world: &World,
    rows: &[db::StoredRecord],
    token: &str,
    withdrawn_bodies: bool,
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
        ShowId::Record(id) => {
            if let Some(line) = comment_line(world, id) {
                return Ok(comment_block(&line, withdrawn_bodies).join("\n"));
            }
            if let Some(line) = artifact_line(world, id.0) {
                return Ok(artifact_block(&line).join("\n"));
            }
            if let Some(task) = world.task_born_at(id.0) {
                let ctx = &world.tasks[task.0];
                return Ok(format!(
                    "c-{}  task  {}\nbirth of t-{}",
                    ctx.birth.0,
                    ctx.birth_actor.as_str(),
                    task.0
                ));
            }
            // the typed door still opens any record the log holds:
            // the event itself, header and payload
            if let Some(row) = rows.get(id.0.0) {
                return Ok(raw_event(row));
            }
            Err(no_record(token))
        }
        ShowId::Position(id) => match rows.get(id.0) {
            Some(row) => Ok(raw_event(row)),
            None => Err(no_record(token)),
        },
    }
}

/// One id as the json face shows it: the text door's same walk with
/// bodies byte-exact — no wrap, no elision — and the fold's facts as
/// fields. One requested id, one array element; the caller assembles.
fn show_value(
    world: &World,
    rows: &[db::StoredRecord],
    token: &str,
    withdrawn_bodies: bool,
) -> Result<serde_json::Value, Fail> {
    match parse_show_id(token)? {
        ShowId::Thread(id) => {
            let view =
                show_view(world, id).ok_or_else(|| Fail::Usage(format!("no task t-{}", id.0)))?;
            let thread = thread_view(world, id).expect("a task with a show view holds its thread");
            let entries: Vec<serde_json::Value> = thread
                .items
                .iter()
                .flat_map(|item| match item {
                    ThreadItem::Note(line) => vec![comment_value(line, false)],
                    ThreadItem::Group { root, replies } => std::iter::once(root)
                        .chain(replies.iter())
                        .map(|line| comment_value(line, false))
                        .collect::<Vec<_>>(),
                    ThreadItem::Artifacts(lines) => {
                        lines.iter().map(artifact_value).collect::<Vec<_>>()
                    }
                })
                .collect();
            Ok(serde_json::json!({
                "id": view.id,
                "state": view.state,
                "name": view.name,
                "owner": view.owner,
                "offer": view.offer,
                "explanation": view.explanation,
                "entries": entries,
            }))
        }
        ShowId::Record(id) => {
            if let Some(line) = comment_line(world, id) {
                return Ok(comment_value(&line, withdrawn_bodies));
            }
            if let Some(line) = artifact_line(world, id.0) {
                return Ok(artifact_value(&line));
            }
            if let Some(task) = world.task_born_at(id.0) {
                let ctx = &world.tasks[task.0];
                return Ok(serde_json::json!({
                    "kind": "task",
                    "seq": id.0.0,
                    "actor": ctx.birth_actor.as_str(),
                    "task": format!("t-{}", task.0),
                }));
            }
            // the raw-id door's machine face: the log row's own grammar
            if let Some(row) = rows.get(id.0.0) {
                return Ok(serde_json::to_value(row).expect("records are plain data"));
            }
            Err(Fail::Usage(format!(
                "no record c-{}; the log holds {} records, c-0 through c-{}",
                id.0.0,
                rows.len(),
                rows.len().saturating_sub(1)
            )))
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

/// A comment as the json face renders it: the thread line's facts with
/// the prose whole — the folded bytes, never the wrapped ones. A
/// withdrawn note's body is null unless revealed; the tombstone's
/// facts ride beside it.
fn comment_value(line: &CommentLine, reveal: bool) -> serde_json::Value {
    let hidden = line.body.is_empty() || (line.withdrawn && !reveal);
    let body = if hidden {
        serde_json::Value::Null
    } else {
        serde_json::json!(line.body)
    };
    serde_json::json!({
        "kind": line.kind,
        "seq": line.seq,
        "target": serde_json::to_value(line.target).expect("targets are plain data"),
        "offer": line.offer,
        "depth": line.depth,
        "actor": line.actor,
        "tier": line.tier,
        "body": body,
        "revised": line.revised,
        "reviser": line.reviser,
        "withdrawn": line.withdrawn,
        "withdrawer": line.withdrawer,
        "withdrawal_note": line.withdrawal_note,
        "born_at": line.born_at,
    })
}

/// An artifact as the json face renders it: the pointer's facts, the
/// hash whole.
fn artifact_value(line: &ArtifactLine) -> serde_json::Value {
    serde_json::json!({
        "kind": "artifact",
        "seq": line.seq,
        "name": line.name,
        "hash": line.hash,
    })
}

/// The raw event as itself: header and payload, no fold.
fn raw_event(row: &db::StoredRecord) -> String {
    format!(
        "c-{}  {}  {}\n{}",
        row.seq, row.kind, row.actor, row.payload
    )
}

/// The piped face: one id per line, every failure a skip with a note,
/// never a broken chain. Empty input is silence. The json face skips
/// as an {"id", "error"} row, so the array holds every line's outcome.
fn show_stdin(
    cli: &Cli,
    world: &World,
    rows: &[db::StoredRecord],
    ids: &[String],
    withdrawn_bodies: bool,
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
            let block = show_value(world, rows, token, withdrawn_bodies)
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
        match render_id(world, rows, token, withdrawn_bodies) {
            Ok(block) => blocks.push(block),
            Err(f) => blocks.push(format!("skipped '{token}': {f}")),
        }
    }
    Ok(format!("{}\n{}", blocks.join("\n\n"), cursor_line(cursor)))
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

/// The search door as text: thread groups of pointers, one matched
/// line each. Counts ride the headers only when the query narrows —
/// narrowing's visibility, never an aggregate badge — and a query
/// that matched nothing says so in one line.
fn render_search(
    cli: &Cli,
    groups: &[SearchGroup],
    tokens: &[String],
    query: &SearchQuery,
    cursor: Option<usize>,
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
        return serde_json::to_string_pretty(&serde_json::json!({
            "groups": rows,
            "cursor": cursor,
        }))
        .expect("results are plain data");
    }
    if groups.is_empty() {
        return format!("no matches ({})\n{}", tokens.join(" "), cursor_line(cursor));
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
    out.push(cursor_line(cursor));
    out.join("\n")
}

/// The anchor query names one record: c-N itself, or t-N's birth.
fn anchor_seq(loadout: &db::Loadout, query: &SearchQuery) -> Result<usize, Fail> {
    match query.terms.as_slice() {
        [Term::Comment(id)] => {
            if id.0 >= loadout.rows.len() {
                return Err(Fail::Usage(format!(
                    "no record c-{}; the log holds {} records, 0 through {}",
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
            "-C anchors on one record: give a single id term (c-<seq> or t-N) and no facets".into(),
        )),
    }
}

/// The anchor's window: the record, N before, N after — one line each,
/// straight off the log's own rows.
fn render_anchor(
    cli: &Cli,
    rows: &[StoredRecord],
    anchor: usize,
    around: usize,
    cursor: Option<usize>,
) -> String {
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
            "records": records,
            "cursor": cursor,
        }))
        .expect("rows are plain data");
    }
    let mut lines: Vec<String> = window
        .iter()
        .map(|row| {
            let prefix = format!("c-{}  {}  {}/{}", row.seq, row.kind, row.actor, row.tier);
            let budget = WIDTH.saturating_sub(prefix.chars().count() + 2);
            format!("{prefix}  {}", glimpse(&row_line(row), budget))
        })
        .collect();
    lines.push(cursor_line(cursor));
    lines.join("\n")
}

/// The record's own first text line — the same fields the log prints.
fn row_line(row: &StoredRecord) -> String {
    let payload: serde_json::Value = match serde_json::from_str(&row.payload) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    ["body", "name", "explanation", "note"]
        .iter()
        .find_map(|k| payload.get(k).and_then(|v| v.as_str()))
        .or_else(|| {
            // the comment payloads nest their prose inside the variant
            let kind = payload.get("kind")?;
            let mut found = None;
            for variant in kind.as_object().into_iter().flatten() {
                for key in ["body", "grounds", "reason"] {
                    if let Some(text) = variant.1.get(key).and_then(|v| v.as_str()) {
                        found = Some(text);
                        break;
                    }
                }
            }
            found
        })
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

    /// Possession, --actor, and the declared scope, one test owning every env mutation.
    #[test]
    fn possession_fixes_the_tier_and_the_scope_reads_the_environment() {
        let saved_actor = std::env::var("SACCADE_ACTOR").ok();
        let saved_task = std::env::var("SACCADE_TASK").ok();
        let saved_incarnation = std::env::var("SACCADE_INCARNATION").ok();
        let saved_workspace = std::env::var("SACCADE_WORKSPACE").ok();
        let restore = || {
            for (key, saved) in [
                ("SACCADE_ACTOR", &saved_actor),
                ("SACCADE_TASK", &saved_task),
                ("SACCADE_INCARNATION", &saved_incarnation),
                ("SACCADE_WORKSPACE", &saved_workspace),
            ] {
                match saved {
                    Some(value) => unsafe { std::env::set_var(key, value) },
                    None => unsafe { std::env::remove_var(key) },
                }
            }
        };

        unsafe {
            std::env::remove_var("SACCADE_ACTOR");
            std::env::remove_var("SACCADE_TASK");
            std::env::remove_var("SACCADE_INCARNATION");
            std::env::remove_var("SACCADE_WORKSPACE");
        }
        let bare = Cli::try_parse_from(["sac", "list"]).unwrap();
        let ctx = context_of(&bare).unwrap_or_else(|f| panic!("{f}"));
        assert_eq!((ctx.tier, ctx.scope.is_none()), (Tier::Human, true));

        unsafe { std::env::set_var("SACCADE_ACTOR", "pi") };
        let possessed = Cli::try_parse_from(["sac", "list"]).unwrap();
        let ctx = context_of(&possessed).unwrap_or_else(|f| panic!("{f}"));
        assert_eq!(ctx.tier, Tier::Agent);

        let named = Cli::try_parse_from(["sac", "--actor", "jerry", "list"]).unwrap();
        let ctx = context_of(&named).unwrap_or_else(|f| panic!("{f}"));
        assert_eq!((ctx.tier, ctx.actor.as_str()), (Tier::Agent, "jerry"));

        unsafe { std::env::remove_var("SACCADE_ACTOR") };
        let named = Cli::try_parse_from(["sac", "--actor", "saccade bot", "list"]).unwrap();
        let ctx = context_of(&named).unwrap_or_else(|f| panic!("{f}"));
        assert_eq!((ctx.tier, ctx.actor.as_str()), (Tier::Human, "saccade bot"));

        // the scope: no anchor yields none, and the riders ride only
        // with the anchor
        unsafe { std::env::set_var("SACCADE_INCARNATION", "i-9") };
        assert!(context_scope().unwrap().is_none());
        unsafe { std::env::remove_var("SACCADE_INCARNATION") };

        unsafe { std::env::set_var("SACCADE_TASK", "t-125") };
        let scope = context_scope().unwrap().expect("the anchor yields a scope");
        assert_eq!(
            (scope.task, scope.incarnation, scope.workspace),
            (TaskId(125), None, None)
        );

        unsafe {
            std::env::set_var("SACCADE_INCARNATION", "i-9");
            std::env::set_var("SACCADE_WORKSPACE", "/repo/wt/t-125");
        }
        let scope = context_scope().unwrap().expect("the anchor yields a scope");
        assert_eq!(scope.incarnation, Some(IncarnationId(RecordId(9))));
        assert_eq!(
            scope.workspace,
            Some(WorktreePath::new("/repo/wt/t-125".into()).unwrap())
        );

        // malformed values are loud, never silently dropped
        unsafe { std::env::set_var("SACCADE_TASK", "t-one") };
        assert!(context_scope().is_err());

        restore();
    }

    /// Possession passes; the unpossessed human write needs a terminal or refuses.
    #[test]
    fn the_identity_fallback_guards_the_write_door() {
        let jerry = Context {
            actor: ActorName::new("jerry".into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        };
        let pi = Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope: None,
        };
        let cli_of = |actor: Option<&str>| Cli {
            db: None,
            repo: None,
            actor: actor.map(str::to_owned),
            json: false,
            at: None,
            offline: true,
            server: String::new(),
            command: Cmd::Log,
        };
        let unnamed = cli_of(None);
        let named = cli_of(Some("saccade bot"));

        // possession passes every cell, flag or none, terminal or none
        for terminal in [false, true] {
            assert!(identity_fallback_guard(&unnamed, &pi, terminal).is_ok());
            assert!(identity_fallback_guard(&named, &pi, terminal).is_ok());
        }

        // a terminal earns the unpossessed human write, named or unnamed
        assert!(identity_fallback_guard(&unnamed, &jerry, true).is_ok());
        assert!(identity_fallback_guard(&named, &jerry, true).is_ok());

        // no terminal: the unpossessed human write refuses, --actor notwithstanding
        let refused = identity_fallback_guard(&unnamed, &jerry, false).unwrap_err();
        assert_eq!(refused.code(), "identity_fallback");
        assert_eq!(
            refused.to_string(),
            "this write would record jerry/human with no SACCADE_ACTOR and no terminal \
             attached (--actor rides the terminal arm); export SACCADE_ACTOR=<name>"
        );
        assert!(identity_fallback_guard(&named, &jerry, false).is_err());

        // the terminal arm's one line: identity named, agent fix taught
        assert_eq!(
            identity_fallback_notice(&jerry),
            "note: this write records jerry/human from the identity fallback \
             (no SACCADE_ACTOR, no --actor); agents export SACCADE_ACTOR=<name>"
        );
    }

    /// The quoting keeps argument boundaries across arbitrary prose.
    #[test]
    fn shell_quoting_keeps_word_boundaries() {
        assert_eq!(shell_quote("sac"), "sac");
        assert_eq!(shell_quote("/nix/store/x-bin/pi"), "/nix/store/x-bin/pi");
        assert_eq!(
            shell_quote("it's a 'quoted' $WORD"),
            "'it'\\''s a '\\''quoted'\\'' $WORD'"
        );
        assert_eq!(shell_quote(""), "''");
        let words = ["fix the 'flake' and $PATH", "--flag", ""];
        let line = words
            .iter()
            .map(|w| shell_quote(w))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(line, "'fix the '\\''flake'\\'' and $PATH' --flag ''");
    }

    /// Hashed tokens learn their bare doors at the parse door: a hashed
    /// task id wants the bare thread, a hashed record id wants the
    /// bare c-N.
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
            "'#t-3' is not a comment id (expected c-<seq>); \
             drop the '#': the thread is addressed as t-3"
        );

        let refused = parse_target("#c-7").map_err(|f| f.to_string()).unwrap_err();
        assert_eq!(
            refused,
            "'#c-7' is not a comment id (expected c-<seq>); \
             drop the '#': the comment is addressed as c-7"
        );

        let refused = parse_target("#7").map_err(|f| f.to_string()).unwrap_err();
        assert_eq!(
            refused,
            "'#7' is not a comment id (expected c-<seq>); \
             drop the '#': the comment is addressed as c-7"
        );

        assert_eq!(
            parse_task_id("t-3").unwrap_or_else(|f| panic!("{f}")),
            TaskId(3)
        );
        assert_eq!(
            parse_target("c-7").unwrap_or_else(|f| panic!("{f}")),
            Target::Comment(CommentId(RecordId(7)))
        );
        assert_eq!(
            parse_comment_id("c-7").unwrap_or_else(|f| panic!("{f}")),
            CommentId(RecordId(7))
        );

        // the revise door takes the c- face; the retired '#' learns it
        let refused = parse_revision_id("#7")
            .map_err(|f| f.to_string())
            .unwrap_err();
        assert_eq!(
            refused,
            "'#7' is not a comment id (expected c-<seq>); \
             drop the '#': the comment is addressed as c-7"
        );
        assert_eq!(
            parse_revision_id("c-7").unwrap_or_else(|f| panic!("{f}")),
            CommentId(RecordId(7))
        );

        // the show door: typed ids, the bare position, and the retired
        // '#' learns the c- form
        assert!(matches!(
            parse_show_id("c-7").unwrap_or_else(|f| panic!("{f}")),
            ShowId::Record(CommentId(RecordId(7)))
        ));
        assert!(matches!(
            parse_show_id("7").unwrap_or_else(|f| panic!("{f}")),
            ShowId::Position(RecordId(7))
        ));
        assert!(matches!(
            parse_show_id("t-3").unwrap_or_else(|f| panic!("{f}")),
            ShowId::Thread(TaskId(3))
        ));
        let refused = parse_show_id("#7").map_err(|f| f.to_string()).unwrap_err();
        assert_eq!(
            refused,
            "'#7' is not an id (expected c-<seq>, t-<n>, or a bare <seq>); \
             drop the '#': the record is addressed as c-7"
        );
    }

    /// The revise door's body lands verbatim: the argument as given, the
    /// file's bytes exact — quotes, backslashes, newlines and all.
    #[test]
    fn the_revise_body_lands_verbatim_from_its_sources() {
        assert_eq!(
            revise_body(Some("plain argument".into()), false, None)
                .map_err(|f| f.to_string())
                .unwrap(),
            "plain argument"
        );
        let dir = std::env::temp_dir().join(format!("sac-revise-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("body.md");
        std::fs::write(
            &file,
            "a body with \"quotes\", \\backslashes\\, and\nlines\n",
        )
        .unwrap();
        assert_eq!(
            revise_body(None, false, Some(file.clone()))
                .map_err(|f| f.to_string())
                .unwrap(),
            "a body with \"quotes\", \\backslashes\\, and\nlines\n"
        );
        assert!(
            revise_body(None, false, None).is_err(),
            "no source, no body"
        );
        let missing = dir.join("absent.md");
        assert!(revise_body(None, false, Some(missing)).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
