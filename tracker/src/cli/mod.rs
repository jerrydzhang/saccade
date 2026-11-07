//! The CLI verb table and dispatch. Verb families live in sibling modules.

pub mod completions;
pub mod launch;
pub mod message;
pub mod root;
pub mod serve;
pub mod skill;
pub mod task;
pub mod wait;
pub mod workspace;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use saccade::World;
use saccade::views::{
    MessageLine, SearchGroup, SearchQuery, TaskView, Term, mailbox, matched_line, message_line,
    search, show_view, task_view,
};

use self::completions::{FISH, Kind, complete_candidates};
use self::launch::{ask_door, harness_choice, launch_door, launch_here_door, launch_new_door};
use self::message::{messages_door, send_door};
use self::root::{root_bind_door, root_clear_door, root_show_door};
use self::serve::serve_door;
use self::skill::skill_door;
use self::task::{born_of, parse_task_id, read_only, render_records};
use self::wait::wait_door;
use self::workspace::workspace_door;

use saccade::client;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::objects::task::TaskId;
use saccade::{
    ActorName, Command, Context, IncarnationId, MessageId, Prose, RecordId, Reject, Scope, Target,
    Tier, WorktreePath, wire,
};

#[derive(Parser)]
#[command(name = "sac", version, about = "Saccade: awesome issue tracker")]
struct Cli {
    /// Select local operations on this database file, ignoring every
    /// discovered or inherited server
    #[arg(long, global = true)]
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

    /// Select this running server for reads, writes, and waits alike
    #[arg(long, global = true)]
    server: Option<String>,

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
    /// Send a durable addressed message from this session's registered
    /// incarnation: the sender rides the environment (never authored);
    /// the message records first, then the attempt submits it to the
    /// recipient's native input queue - success only on confirmed
    /// submission, a refusal discloses the record's reference
    Send {
        /// The recipient (i-<n>)
        #[arg(long)]
        to: String,
        /// The message this one answers (m-<n>)
        #[arg(long)]
        reply: Option<String>,
        /// Task context riding the exchange, without publication (t-<n>)
        #[arg(long)]
        task: Option<String>,
        /// The body, quoted
        #[arg(required_unless_present = "stdin", conflicts_with = "stdin")]
        body: Option<String>,
        /// Read the body from stdin instead of an argument
        #[arg(long)]
        stdin: bool,
    },
    /// Show the messages an incarnation participates in, in record
    /// order, with their delivery state; the mailbox defaults to this
    /// session's declared incarnation
    Messages {
        /// The incarnation (i-<n>); defaults to the declared scope
        mailbox: Option<String>,
    },
    /// List objects (world projection)
    List {
        #[arg(value_enum, default_value_t = ObjKind::Task)]
        kind: ObjKind,
    },
    /// Print raw event records
    Log,
    /// Everything about tasks and records, many at once; bare seqs
    /// show the raw event, t-<n> the task
    Show {
        /// Ids in any mix: m-<n> a message, bare <seq> the raw event
        /// at that position, t-<n> the task
        #[arg(required_unless_present = "stdin")]
        ids: Vec<String>,
        /// Read one id per line from stdin instead of arguments; blank
        /// and invalid lines are skipped with a note
        #[arg(long)]
        stdin: bool,
    },
    /// Search the folded record: exact terms over task names and
    /// cancellation explanations; a t-<n> term is a reference search
    Search {
        /// Terms to match (a record must match them all) and
        /// only-show-me facets:
        ///   in:t-N     one task's records only
        ///   by:NAME    author, matched within names (by:pi finds pi/t-90-1)
        ///   kind:K     task, explanation
        ///
        /// A t-<n> term is a reference search: the records of that task
        #[arg(verbatim_doc_comment)]
        terms: Vec<String>,
        /// With one id term: that record plus N before and after
        #[arg(short = 'C', long)]
        context: Option<usize>,
    },
    /// Serve the console and /api/v1 over HTTP (the repo's tracker by
    /// default; --db names a different file)
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
        #[arg(long, default_value_t = 8811)]
        port: u16,
    },
    /// Launch the composed native Pi in an explicit existing
    /// workspace, task-free: a fresh session registers itself as a new
    /// incarnation; --resume continues a saved one from its recorded
    /// locator; --new provisions a fresh workspace first. No task,
    /// claim, or request is provisioned
    Launch {
        /// The workspace directory to work in; it must already exist
        #[arg(long, conflicts_with = "resume")]
        workspace: Option<PathBuf>,
        /// Continue this saved incarnation (i-<n>) from its recorded
        /// native session, validated against the saved header
        #[arg(long, conflicts_with = "workspace")]
        resume: Option<String>,
        /// Provision a NEW workspace through the server's Git effects
        /// and launch there: an optional descriptive label (the name
        /// falls back to workspace-<token>); conflicts with
        /// --workspace and --resume
        #[arg(
            long,
            num_args = 0..=1,
            default_missing_value = "",
            conflicts_with_all = ["workspace", "resume"]
        )]
        new: Option<String>,
        /// The Git base for --new: a ref, defaulting to the primary
        /// checkout's current HEAD
        #[arg(long, requires = "new")]
        base: Option<String>,
        /// The initial question: the launching incarnation sends it as
        /// a durable addressed message to the newborn peer. The creator
        /// must be registered (SACCADE_INCARNATION_SEQ); conflicts with
        /// a positional prompt and with --resume
        #[arg(long, conflicts_with = "resume")]
        ask: Option<String>,
        /// Ask the host for one independent terminal running the
        /// composed session, unplaced (no Herdr space), then attach
        /// this terminal as its viewer. Detaching (ctrl+\) ends the
        /// attach and leaves the session running at its recorded host
        /// address. Composes with --workspace and --resume
        #[arg(long)]
        here: bool,
        /// The harness a fresh launch runs: pi (the default, existing
        /// callers unchanged) or claude-code. A resume always starts
        /// the recorded harness; an explicit harness naming another
        /// kind refuses
        #[arg(long)]
        harness: Option<String>,
        /// The initial prompt, quoted; native human input when omitted
        #[arg(last = true)]
        prompt: Vec<String>,
    },
    /// The task-free workspace controller: provision a new workspace,
    /// or move a stopped native session's saved directory. Creation
    /// births its own record - no task, claim, or incarnation; a
    /// change refuses a live session and lands only after the saved
    /// header actually moved
    Workspace {
        #[command(subcommand)]
        verb: WorkspaceVerb,
    },
    /// Report this session's environment-declared identity and routing:
    /// disclosure, never a liveness claim or proof of ownership
    Identity,
    /// Emit a shell completion script from the binary itself: the
    /// static command and flag skeleton plus the dynamic value hooks,
    /// no external glue files
    Completions {
        #[arg(value_enum)]
        shell: Shell,
    },
    /// The hidden candidate read doors the completion scripts call:
    /// one `value TAB description` line per record fact, empty on any
    /// failure, nothing recorded
    #[command(hide = true)]
    Complete {
        /// The candidate kind the completion position asks for
        #[arg(value_enum)]
        kind: CompleteKind,
    },
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
    /// Deploy this binary's embedded skill, or verify the copy in this repo
    Skill {
        #[command(subcommand)]
        verb: SkillVerb,
    },
    /// Block until a match lands: a task (t-<n>) target matches
    /// record activity, the cursor named by --after and the mode by
    /// --until; an incarnation (i-<n>) target waits for messages
    /// addressed to it - waking on arrivals at its host terminal,
    /// reading its mailbox for truth, bounded overall by --timeout
    /// seconds. Nothing is recorded
    Wait {
        /// The target: t-<n> for record activity, i-<n> for arrival
        /// waiting
        target: String,
        /// The global record position matched strictly after - the
        /// cursor a read disclosed, never defaulted; required for
        /// t-<n>, refused for i-<n>
        #[arg(long)]
        after: Option<u64>,
        /// What wakes a t-<n> wait - required there, refused for i-<n>
        #[arg(long)]
        until: Option<UntilMode>,
        /// The overall bound in seconds for an i-<n> arrival wait;
        /// without it the wait runs until a message lands or the call
        /// is interrupted
        #[arg(long)]
        timeout: Option<u64>,
    },
}

/// The task-free controller's verbs.
#[derive(Clone, Subcommand)]
enum WorkspaceVerb {
    /// Provision a NEW workspace: a fresh branch named from the
    /// optional descriptive label plus a generated uniqueness suffix
    /// (workspace-<token> without a label), cut at the named base or
    /// the primary HEAD, checked out under the tracker's home
    New {
        /// The descriptive label; omit for workspace-<token>
        label: Option<String>,
        /// The Git base: a ref, defaulting to the primary checkout's
        /// current HEAD
        #[arg(long)]
        base: Option<String>,
    },
    /// Change where a stopped native session lives: an explicit
    /// existing directory; a live session refuses (exit it first), and
    /// the record is the whole effect
    Switch {
        /// The incarnation (i-<n>)
        incarnation: String,
        /// The selected directory: it must already exist
        #[arg(long)]
        to: PathBuf,
        /// Resume the session in the new directory after the change
        #[arg(long)]
        resume: bool,
        /// The prompt for the resumed session; native human input when
        /// omitted (with --resume)
        #[arg(last = true)]
        prompt: Vec<String>,
    },
    /// Clean an exact recorded workspace: terminal close, terminal
    /// closure verified, git worktree removal with the branch
    /// retained, checkout and host visible absence verified, then the
    /// Cleaned record. Staged, unstaged, and untracked work refuses;
    /// a recorded-cleaned workspace with resources present is a
    /// mismatch, never an automatic deletion.
    Clean {
        /// The workspace's exact recorded path
        path: PathBuf,
    },
    /// Restore an exact recorded cleaned workspace: establish or
    /// validate the matching git checkout at the branch's current
    /// local tip, then record Restored. No session is launched, no
    /// terminal is created - execution stays an explicit launch.
    Restore {
        /// The workspace's exact recorded path
        path: PathBuf,
    },
    /// The primary checkout's Herdr destination: bind the chosen
    /// manual root space once, show the standing choice, or clear it.
    /// Root launches declare the bound pair; an unbound root runs
    /// unplaced.
    Root {
        #[command(subcommand)]
        verb: RootVerb,
    },
    /// The one-time cutover verification over every recorded
    /// workspace: absent resources verified against git and the host,
    /// matching present ones preserved, conflicts surfaced. Dry run
    /// unless --apply records the verified Cleaned facts.
    Reconcile {
        /// Record the verified Cleaned facts; without this the door
        /// only verifies and reports
        #[arg(long)]
        apply: bool,
    },
}

#[derive(Clone, Subcommand)]
enum RootVerb {
    /// Bind the chosen manual root space: run inside the terminal
    /// standing in that space, in this repository's context. Records
    /// the terminal's exact live space/tab pair; missing or unlabeled
    /// terminal refuses. Human-only; a rebind replaces the latest
    /// choice, the log keeps the history.
    Bind,
    /// Show the bound root destination, if any: a read, no host ask
    Show,
    /// Clear the bound root destination: root launches run unplaced
    /// until a later bind. Human-only; the history stays in the log.
    Clear,
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

/// The shells whose scripts this binary emits. Fish is the primary;
/// bash and zsh stay a later option.
#[derive(Clone, Copy, ValueEnum)]
enum Shell {
    Fish,
}

/// The hidden candidate doors the completion scripts call: the names
/// ride the value enum, the derivation lives in saccade::completions.
#[derive(Clone, Copy, ValueEnum)]
enum CompleteKind {
    Tasks,
    Incarnations,
    Workspaces,
    WorkspacesClean,
    WorkspacesRestore,
}

impl From<CompleteKind> for Kind {
    fn from(kind: CompleteKind) -> Self {
        match kind {
            CompleteKind::Tasks => Kind::Tasks,
            CompleteKind::Incarnations => Kind::Incarnations,
            CompleteKind::Workspaces => Kind::Workspaces,
            CompleteKind::WorkspacesClean => Kind::WorkspacesClean,
            CompleteKind::WorkspacesRestore => Kind::WorkspacesRestore,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum UntilMode {
    Activity,
}

impl UntilMode {
    fn until(self) -> saccade::views::WaitUntil {
        match self {
            UntilMode::Activity => saccade::views::WaitUntil::Activity,
        }
    }
}

pub(crate) fn entry() -> ExitCode {
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

    // the same-terminal launch owns the process exit - the composed
    // session's own status is this command's - so it never rides the
    // ordinary success/failure face
    if let Cmd::Launch { here: true, .. } = &cli.command {
        return launch_here_door(&cli);
    }

    match run(&cli) {
        Ok(output) => {
            if !output.is_empty() {
                println!("{output}");
            }
            ExitCode::SUCCESS
        }
        Err(fail) => fail_exit(&cli, &fail),
    }
}

/// The shared refusal face: the json envelope or the error line, then
/// the failed exit.
fn fail_exit(cli: &Cli, fail: &Fail) -> ExitCode {
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
    /// A workspace effect refused or failed client-side; its codes
    /// match the wire's.
    Workspace(saccade::workspace::WorkspaceFail),
    Usage(String),
    Client(client::ClientFail),
}

impl From<Reject> for Fail {
    fn from(r: Reject) -> Self {
        Fail::Reject(r)
    }
}

impl From<saccade::workspace::WorkspaceFail> for Fail {
    fn from(f: saccade::workspace::WorkspaceFail) -> Self {
        Fail::Workspace(f)
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
            Fail::Workspace(f) => f.code(),
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
            Fail::Taught { code, text } => write!(f, "rejected: {code} - {text}"),
            Fail::IdentityFallback { actor, tier } => write!(
                f,
                "this write would record {actor}/{tier} with no SACCADE_ACTOR and no terminal \
                 attached (--actor rides the terminal arm); export SACCADE_ACTOR=<name>"
            ),
            Fail::Usage(m) => write!(f, "{m}"),
            Fail::Workspace(w) => write!(f, "{}", w.detail()),
            Fail::Client(c) => write!(f, "{c}"),
        }
    }
}

fn run(cli: &Cli) -> Result<String, Fail> {
    // identity is the lenient read: it discloses the destination
    // outcome, refusal included, so diagnosis survives broken routing
    if let Cmd::Identity = &cli.command {
        return identity_door(cli);
    }
    // the completions door emits its script from the binary itself:
    // no destination resolves, no fold reads
    if let Cmd::Completions { shell } = &cli.command {
        return Ok(match shell {
            Shell::Fish => FISH.to_string(),
        });
    }
    // the candidate doors are reads like every other, and completion
    // never errs aloud: any failure is no candidates, exit 0
    if let Cmd::Complete { kind } = &cli.command {
        return Ok(complete_candidates(cli, *kind));
    }
    // serve starts a server, so it never resolves a client destination
    if let Cmd::Serve { bind, port } = &cli.command {
        return serve_door(cli, bind, port);
    }
    let destination = resolve_destination(cli)?;
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
        Cmd::List { .. } | Cmd::Log | Cmd::Show { .. } | Cmd::Search { .. } => {
            return read_only(cli, &loadout_of(&destination)?);
        }
        Cmd::Wait {
            target,
            after,
            until,
            timeout,
        } => {
            return wait_door(
                cli,
                &destination,
                target,
                *after,
                until.map(UntilMode::until),
                *timeout,
            );
        }
        Cmd::Send {
            to,
            reply,
            task,
            body,
            stdin,
        } => {
            return send_door(
                cli,
                &destination,
                to,
                reply.as_deref(),
                task.as_deref(),
                body.clone(),
                *stdin,
            );
        }
        Cmd::Messages { mailbox } => return messages_door(cli, &destination, mailbox.as_deref()),
        Cmd::Launch {
            workspace,
            resume,
            new,
            base,
            ask,
            harness,
            prompt,
            // --here dispatches in main before run: the child's own
            // status is the command's, which this face cannot express
            here: _,
        } => {
            let harness = harness_choice(harness)?;
            if let Some(labeled) = new.as_deref().map(str::to_owned) {
                let label = (!labeled.is_empty()).then_some(labeled.as_str());
                return launch_new_door(
                    cli,
                    &destination,
                    label,
                    base.as_deref(),
                    ask.as_deref(),
                    harness,
                    prompt,
                );
            }
            if let Some(question) = ask.as_deref() {
                let Some(workspace) = workspace.as_deref() else {
                    return Err(Fail::Usage(
                        "launch --ask launches a fresh peer: pass --workspace <dir> or --new [label]".into(),
                    ));
                };
                return ask_door(cli, &destination, workspace, harness, question);
            }
            return launch_door(
                cli,
                &destination,
                workspace.as_deref(),
                resume.as_deref(),
                harness,
                prompt,
            );
        }
        Cmd::Workspace { verb } => return workspace_door(cli, &destination, verb),
        Cmd::Identity => unreachable!("identity dispatches before the destination resolves"),
        Cmd::Serve { .. } => unreachable!("serve dispatches before the destination resolves"),
        Cmd::Completions { .. } | Cmd::Complete { .. } => {
            unreachable!("the completion doors dispatch before the destination resolves")
        }
        Cmd::Clone { at, out } => {
            // the reproduction recipe's cursor half: the clone names the
            // world a refused request died against
            let cloned = match &destination {
                Destination::Local { db } => db::clone(db, *at as usize, out).map_err(Fail::Db)?,
                Destination::Server { url } => {
                    let rows = client::records(url).map_err(Fail::Client)?;
                    db::clone_rows(&rows, *at as usize, out).map_err(Fail::Db)?
                }
            };
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
        Cmd::Skill { .. } => unreachable!("the skill door dispatches before the db doors"),
    };

    // Identity is required only where it is recorded: mutating commands.
    let context = context_of(cli)?;
    identity_fallback_guard(cli, &context, attached_terminal())?;
    let creates = matches!(command, Command::CreateTask { .. });
    let mut born: Option<String> = None;
    let stored = match &destination {
        Destination::Local { db } => {
            let now = cli.at.unwrap_or_else(db::now_epoch);
            let mut conn = db::open(db).map_err(Fail::Db)?;
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
        }
        Destination::Server { url } => {
            client::handshake(url);
            let reply = client::send(url, &context, command, cli.at).map_err(Fail::Client)?;
            born = reply.id;
            reply.records
        }
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

enum Destination {
    Local { db: PathBuf },
    Server { url: String },
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

fn resolve_destination(cli: &Cli) -> Result<Destination, Fail> {
    match (&cli.db, &cli.server) {
        (Some(_), Some(_)) => Err(Fail::Usage(
            "--db and --server are both explicit destinations; pass one".into(),
        )),
        (Some(db), None) => Ok(Destination::Local { db: db.clone() }),
        (None, Some(url)) => Ok(Destination::Server { url: url.clone() }),
        (None, None) => {
            let db = saccade::paths::db_at(&resolve_repo(cli)?);
            let url = client::discover(&db).map_err(Fail::Client)?;
            Ok(Destination::Server { url })
        }
    }
}

fn loadout_of(destination: &Destination) -> Result<saccade::db::Loadout, Fail> {
    match destination {
        Destination::Local { db } => {
            let conn = db::open_read(db).map_err(Fail::Db)?;
            db::load(&conn).map_err(Fail::Db)
        }
        Destination::Server { url } => {
            let rows = client::records(url).map_err(Fail::Client)?;
            db::fold_rows(rows).map_err(Fail::Db)
        }
    }
}

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

/// The environment-declared scope: SACCADE_TASK ("t-<n>"),
/// SACCADE_INCARNATION_SEQ (a bare birth-record position - the
/// machine identity channel; the public i-N stays the fold's dense
/// numbering), SACCADE_WORKSPACE (an absolute path). Any one present
/// declares a scope: native participation precedes any task, so the
/// task binding is optional. Disclosure riding every record, never
/// proof.
fn context_scope() -> Result<Option<Scope>, Fail> {
    let task = match std::env::var("SACCADE_TASK") {
        Ok(value) if !value.is_empty() => Some(parse_task_id(&value)?),
        _ => None,
    };
    let incarnation = match std::env::var("SACCADE_INCARNATION_SEQ") {
        Ok(value) if !value.is_empty() => {
            let seq: usize = value.parse().map_err(|_| {
                Fail::Usage(format!(
                    "SACCADE_INCARNATION_SEQ='{value}' is not a bare birth-record position"
                ))
            })?;
            Some(IncarnationId(RecordId(seq)))
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
    if task.is_none() && incarnation.is_none() && workspace.is_none() {
        return Ok(None);
    }
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
/// any of stdin, stdout, or stderr earns the write - the unnamed
/// fallback on one notice line, a named --actor silently - and a
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

/// The db anchored absolute: a relative --db must still name the same
/// tracker once the session sits inside the checkout.
fn canonical_db(db_path: &std::path::Path) -> PathBuf {
    std::fs::canonicalize(db_path).unwrap_or_else(|_| db_path.to_path_buf())
}

fn server_url(destination: &Destination, verb: &str) -> Result<String, Fail> {
    match destination {
        Destination::Server { url } => Ok(url.clone()),
        Destination::Local { .. } => Err(Fail::Usage(format!(
            "{verb} runs through the server: the sole writer establishes workspaces; --db cannot"
        ))),
    }
}

/// The identity door: the environment's declaration plus routing,
/// read-only; disclosure, never a liveness claim.
fn identity_door(cli: &Cli) -> Result<String, Fail> {
    let context = context_of(cli)?;
    let scope = context.scope.clone();
    let repo = resolve_repo(cli).ok();
    let db_path = cli
        .db
        .clone()
        .or_else(|| repo.as_ref().map(|r| saccade::paths::db_at(r)))
        .map(|p| canonical_db(&p));
    // the destination's outcome is the disclosure: a broken discovery
    // is reported, never silently replaced with a local default
    let (mode, server, detail) = if cli.db.is_some() && cli.server.is_some() {
        (
            "conflict",
            None,
            Some("--db and --server are both explicit destinations; pass one".to_string()),
        )
    } else if let Some(url) = &cli.server {
        ("server", Some(url.clone()), None)
    } else if cli.db.is_some() {
        ("local", None, None)
    } else {
        match repo
            .as_ref()
            .map(|r| saccade::paths::db_at(r))
            .map(|db| client::discover(&db))
        {
            Some(Ok(url)) => ("discovered-server", Some(url), None),
            Some(Err(e)) => ("unresolved", None, Some(e.to_string())),
            None => (
                "unresolved",
                None,
                Some("no repo resolved, so no tracker to discover".to_string()),
            ),
        }
    };
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "actor": context.actor.as_str(),
            "tier": saccade::wire::tier_of(&context.tier),
            "task": scope.as_ref().and_then(|s| s.task).map(|t| format!("t-{}", t.0)),
            "incarnation_seq": scope.as_ref().and_then(|s| s.incarnation).map(|i| i.0 .0),
            "workspace": scope.as_ref().and_then(|s| s.workspace.as_ref().map(|w| w.as_path().display().to_string())),
            "repo": repo.map(|p| p.display().to_string()),
            "db": db_path.map(|p| p.display().to_string()),
            "destination": mode,
            "server": server,
            "server_detail": detail,
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
            out.push(match scope.task {
                Some(task) => format!("task  t-{}", task.0),
                None => "task  none declared".into(),
            });
            if let Some(incarnation) = scope.incarnation {
                out.push(format!("incarnation seq {}", incarnation.0.0));
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
    match server {
        Some(url) => out.push(format!("destination  {mode} {url}")),
        None => match detail {
            Some(detail) => out.push(format!("destination  {mode}: {detail}")),
            None => out.push(format!("destination  {mode}")),
        },
    }
    Ok(out.join("\n"))
}

/// The send door's body source: the quoted argument or the pipe.
fn body_source(body: Option<String>, stdin: bool) -> Result<String, Fail> {
    if stdin {
        if std::io::stdin().is_terminal() {
            return Err(Fail::Usage("nothing piped: --stdin reads the body".into()));
        }
        return std::io::read_to_string(std::io::stdin())
            .map_err(|e| Fail::Usage(format!("stdin: {e}")));
    }
    body.ok_or_else(|| Fail::Usage("the body comes quoted or through --stdin".into()))
}

/// Every text read ends with this line, feedable to the next `wait --after`.
fn cursor_line(cursor: Option<usize>) -> String {
    match cursor {
        Some(c) => format!("cursor {c}"),
        None => "cursor none".into(),
    }
}

/// A public dense token's number: the wire carries what the human
/// typed, and the server resolves it against its own fold.
fn dense_token(token: &str, prefix: &str) -> Result<usize, Fail> {
    let n = token.strip_prefix(prefix).ok_or_else(|| {
        Fail::Usage(format!(
            "'{token}' is not an {}<n> reference",
            prefix.strip_suffix('-').unwrap_or(prefix)
        ))
    })?;
    n.parse()
        .map_err(|_| Fail::Usage(format!("'{token}' is not an {prefix}<n> reference")))
}

impl From<saccade::launch::LaunchFail> for Fail {
    fn from(f: saccade::launch::LaunchFail) -> Self {
        match f {
            // the wire code is the serving edge's vocabulary; the CLI
            // face is the prose itself
            saccade::launch::LaunchFail::Refused { detail, .. } => Fail::Usage(detail),
            saccade::launch::LaunchFail::Usage(detail) => Fail::Usage(detail),
            saccade::launch::LaunchFail::Degraded(reason) => Fail::Degraded(reason),
        }
    }
}

const WIDTH: usize = 80;
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

/// One line cut to its budget on a char boundary; the ellipsis is the
/// only mark a cut earns.
fn glimpse(line: &str, budget: usize) -> String {
    if line.chars().count() <= budget {
        line.to_string()
    } else {
        let mut cut: String = line.chars().take(budget.saturating_sub(1)).collect();
        cut.push_str("...");
        cut
    }
}

#[cfg(test)]
mod test {
    use self::task::{ShowId, parse_show_id};
    use self::wait::wait_target;
    use super::*;

    /// Possession, --actor, and the declared scope, one test owning every env mutation.
    #[test]
    fn possession_fixes_the_tier_and_the_scope_reads_the_environment() {
        let saved_actor = std::env::var("SACCADE_ACTOR").ok();
        let saved_task = std::env::var("SACCADE_TASK").ok();
        let saved_incarnation = std::env::var("SACCADE_INCARNATION_SEQ").ok();
        let saved_workspace = std::env::var("SACCADE_WORKSPACE").ok();
        let restore = || {
            for (key, saved) in [
                ("SACCADE_ACTOR", &saved_actor),
                ("SACCADE_TASK", &saved_task),
                ("SACCADE_INCARNATION_SEQ", &saved_incarnation),
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
            std::env::remove_var("SACCADE_INCARNATION_SEQ");
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

        // the scope: any declared key anchors it; the machine identity
        // channel is a bare birth-record position
        unsafe { std::env::set_var("SACCADE_INCARNATION_SEQ", "9") };
        let scope = context_scope()
            .unwrap()
            .expect("a task-free session declares a scope");
        assert_eq!(
            (scope.task, scope.incarnation, scope.workspace),
            (None, Some(IncarnationId(RecordId(9))), None)
        );
        unsafe { std::env::remove_var("SACCADE_INCARNATION_SEQ") };

        unsafe { std::env::set_var("SACCADE_TASK", "t-125") };
        let scope = context_scope().unwrap().expect("the anchor yields a scope");
        assert_eq!(
            (scope.task, scope.incarnation, scope.workspace),
            (Some(TaskId(125)), None, None)
        );

        unsafe {
            std::env::set_var("SACCADE_INCARNATION_SEQ", "9");
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
        unsafe { std::env::set_var("SACCADE_TASK", "t-125") };
        unsafe { std::env::set_var("SACCADE_INCARNATION_SEQ", "i-9") };
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
            server: None,
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

    #[test]
    fn hashed_tokens_name_their_bare_doors() {
        let refused = parse_task_id("#t-3")
            .map_err(|f| f.to_string())
            .unwrap_err();
        assert_eq!(
            refused,
            "'#t-3' is not a task id (expected t-<n>; only tasks exist); \
             drop the '#': the task is addressed as t-3"
        );

        let refused = wait_target("#t-3").map_err(|f| f.to_string()).unwrap_err();
        assert_eq!(
            refused,
            "'#t-3' is not a wait target; drop the '#': tasks are t-<n>"
        );

        assert!(matches!(
            wait_target("t-3").unwrap_or_else(|f| panic!("{f}")),
            TaskId(3)
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
            "'#7' is not an id (expected t-<n>, m-<n>, or a bare <seq>); \
             drop the '#': the raw record at a position is the bare <seq>"
        );
        assert!(parse_show_id("184467440737095516160").is_err());
        assert!(wait_target("#184467440737095516160").is_err());
        assert!(parse_show_id("#184467440737095516160").is_err());
    }
}
