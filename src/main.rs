use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use saccade::World;
use saccade::client;
use saccade::db::{self, ExecuteFail, LoadState, StoredRecord};
use saccade::host;
use saccade::objects::task::TaskId;
use saccade::views::{
    ArtifactLine, CommentLine, MessageLine, SearchGroup, SearchQuery, TaskView, Term, ThreadItem,
    comment_line, mailbox, matched_line, message_line, search, show_view, task_view, thread_index,
    thread_view,
};
use saccade::{
    ActorName, Artifact, Command, CommentId, CommentKind, ContentHash, Context, IncarnationId,
    Prose, RecordId, Reject, Scope, Target, Tier, WorktreePath, wire,
};

mod skill;

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
    /// Attach a note to a task (t-<n>), or reply to a comment
    /// (c-<n>: the note homes on that comment's task and cites it)
    Comment {
        target: String,
        /// The authored title: the concise whole-contribution
        /// description, required at birth
        #[arg(long)]
        title: String,
        body: String,
        /// A citation the mapping carries: c-<n>, t-<n>, or a bare
        /// <seq>; repeatable, each earning the next [^k] marker
        #[arg(long = "ref", value_name = "REF")]
        refs: Vec<String>,
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
    /// Offer the whole-task closing account (t-<n> only: a broadcast)
    Offer {
        id: String,
        /// The authored title: the concise whole-contribution
        /// description, required at birth
        #[arg(long)]
        title: String,
        body: String,
    },
    /// Accept an offered conclusion: the offer it accepts (c-<n>)
    Accept { offer: String },
    /// Reject an offered conclusion with grounds, asking reconsideration
    Reject {
        offer: String,
        #[arg(long)]
        grounds: String,
    },
    /// Declare a consuming task's prospective reliance on a source
    /// task; bind the conclusion it rests on with `sac bind`
    Use {
        /// The consuming task (t-<n>)
        task: String,
        /// The source task (t-<n>): fixed at birth, another task
        #[arg(long)]
        source: String,
        #[arg(long)]
        reason: String,
    },
    /// Bind a use's selection: the conclusion chosen from its source
    /// task, with the grounds for that choice. Rebinding is the same
    /// verb; earlier selections stay in the log
    Bind {
        /// The use (c-<n>)
        id: String,
        /// The selected offer (c-<n>)
        offer: String,
        #[arg(long)]
        grounds: String,
    },
    /// Revise a comment's presentation: a note or an offer repairs its
    /// title and/or body independently, the other prose variants their
    /// body - the fold presents the latest and names a reviser who
    /// differs from the birth author, the log keeps the original
    Revise {
        id: String,
        /// The revised title, as an argument (notes and offers alone)
        #[arg(long)]
        title: Option<String>,
        /// The revised prose, as an argument
        #[arg(required_unless_present_any = ["title", "stdin", "body_file"])]
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
        /// Ids in any mix: c-<n> a comment as its block, m-<n> a
        /// message - bare <seq> the raw event at that position - t-<n>
        /// a whole thread
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
    /// Search the folded record: exact terms over task titles, note
    /// and offer titles, comment prose, and explanations; an id term
    /// is a reference search
    Search {
        /// Terms to match (a record must match them all) and
        /// only-show-me facets:
        ///   in:t-N     one thread's records only
        ///   by:NAME    author, matched within names (by:pi finds pi/t-90-1)
        ///   kind:K     task, note, offer, accept, reject, use,
        ///              explanation, artifact
        ///
        /// An id term - 'c-907' or 't-49' - is a reference search: every
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
    /// Migrate a schema-3 or schema-4 tracker into a fresh one at
    /// this binary's schema: the source stays untouched, seq and
    /// timestamps carry over, prose renumbers onto the fold's public
    /// numbering, untitled notes and offers acquire their home task's
    /// name as title, and anything that cannot be identified stays
    /// visibly legacy and reported
    Migrate {
        /// The fresh tracker to write; must not exist
        #[arg(long)]
        out: PathBuf,
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
    /// Park an artifact on a task's thread: hash the file into the
    /// store, record the pointer - the bytes never ride the wire
    Artifact {
        /// The task whose thread holds it (t-<n>)
        task: String,
        /// The file whose bytes become the artifact
        path: PathBuf,
        /// The artifact's name; defaults to the file's basename
        #[arg(long)]
        name: Option<String>,
    },
    /// Block until a matching record lands on the target: activity
    /// mode matches records folding into a task, or the records
    /// structurally tied to a comment - its citations, repairs, and
    /// decisions; offer mode matches an Offer born on the task.
    /// Matches are strictly after the given global position; nothing
    /// is recorded
    Wait {
        /// The task (t-<n>) or comment (c-<n>) to wait on
        target: String,
        /// The global record position matched strictly after - the
        /// cursor a read disclosed, never defaulted
        #[arg(long)]
        after: u64,
        /// What wakes the wait - required, never chosen for you;
        /// offer addresses a task only
        #[arg(long)]
        until: UntilMode,
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
    /// Move a stopped native session's saved working directory to an
    /// explicit existing one: a live session refuses (exit it first),
    /// and the change record lands only after the header moved
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
enum UntilMode {
    Activity,
    Offer,
}

impl UntilMode {
    fn until(self) -> saccade::views::WaitUntil {
        match self {
            UntilMode::Activity => saccade::views::WaitUntil::Activity,
            UntilMode::Offer => saccade::views::WaitUntil::Offer,
        }
    }
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
        Cmd::Comment {
            target,
            title,
            body,
            refs,
        } => {
            let world = world_of(&destination)?;
            let (home, mut references) = comment_address(&world, target)?;
            for token in refs {
                references.push(resolve_ref(&world, token)?);
            }
            let body = Prose::new(body.clone())?;
            let title = Prose::new(title.clone())?;
            Command::Comment {
                home,
                kind: CommentKind::Note { title, body },
                references,
            }
        }
        Cmd::Offer { id, title, body } => {
            let body = Prose::new(body.clone())?;
            let title = Prose::new(title.clone())?;
            Command::Comment {
                home: parse_task_id(id)?,
                kind: CommentKind::Offer { title, body },
                references: Vec::new(),
            }
        }
        Cmd::Accept { offer } => {
            let world = world_of(&destination)?;
            let offer = resolve_comment(&world, offer)?;
            let home = world.comments[&offer].comment.home;
            Command::Comment {
                home,
                kind: CommentKind::Accept { offer },
                references: Vec::new(),
            }
        }
        Cmd::Reject { offer, grounds } => {
            let world = world_of(&destination)?;
            let offer = resolve_comment(&world, offer)?;
            let home = world.comments[&offer].comment.home;
            Command::Comment {
                home,
                kind: CommentKind::Reject {
                    offer,
                    grounds: Prose::new(grounds.clone())?,
                },
                references: Vec::new(),
            }
        }
        Cmd::Use {
            task,
            source,
            reason,
        } => {
            let reason = Prose::new(reason.clone())?;
            Command::Comment {
                home: parse_task_id(task)?,
                kind: CommentKind::Use {
                    source: parse_task_id(source)?,
                    reason,
                    bound: None,
                },
                references: Vec::new(),
            }
        }
        Cmd::Bind { id, offer, grounds } => {
            let world = world_of(&destination)?;
            Command::BindUse {
                id: resolve_comment(&world, id)?,
                offer: resolve_comment(&world, offer)?,
                grounds: Prose::new(grounds.clone())?,
            }
        }
        Cmd::Revise {
            id,
            title,
            body,
            stdin,
            body_file,
        } => {
            let world = world_of(&destination)?;
            let title = match title {
                Some(given) => Some(Prose::new(given.clone())?),
                None => None,
            };
            let body = match (body.clone(), *stdin, body_file.clone()) {
                (None, false, None) => None,
                (body, stdin, file) => Some(Prose::new(revise_body(body, stdin, file)?)?),
            };
            Command::ReviseComment {
                id: resolve_comment(&world, id)?,
                title,
                body,
            }
        }
        Cmd::Withdraw { id, note } => {
            let world = world_of(&destination)?;
            Command::WithdrawComment {
                id: resolve_comment(&world, id)?,
                note: Prose::new(note.clone())?,
            }
        }
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
            return read_only(cli, &loadout_of(&destination)?);
        }
        Cmd::Wait {
            target,
            after,
            until,
        } => return wait_door(cli, &destination, target, *after, until.until()),
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
            prompt,
        } => {
            if let Some(labeled) = new.as_deref().map(str::to_owned) {
                let label = (!labeled.is_empty()).then_some(labeled.as_str());
                return launch_new_door(
                    cli,
                    &destination,
                    label,
                    base.as_deref(),
                    ask.as_deref(),
                    prompt,
                );
            }
            if let Some(question) = ask.as_deref() {
                let Some(workspace) = workspace.as_deref() else {
                    return Err(Fail::Usage(
                        "launch --ask launches a fresh peer: pass --workspace <dir> or --new [label]".into(),
                    ));
                };
                return ask_door(cli, &destination, workspace, question);
            }
            return launch_door(
                cli,
                &destination,
                workspace.as_deref(),
                resume.as_deref(),
                prompt,
            );
        }
        Cmd::Workspace { verb } => return workspace_door(cli, &destination, verb),
        Cmd::Identity => unreachable!("identity dispatches before the destination resolves"),
        Cmd::Serve { .. } => unreachable!("serve dispatches before the destination resolves"),
        Cmd::Migrate { out } => {
            let source = match &destination {
                Destination::Local { db } => db.clone(),
                Destination::Server { .. } => {
                    return Err(Fail::Usage(
                        "migrate reads the source file directly; pass --db".into(),
                    ));
                }
            };
            return migrate_door(cli, &source, out);
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

/// The skill verbs' output door: the verdict lands on stdout - drift
/// and absence are findings, not failures - and the exit code carries
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
                        "skill absent: no {} in this directory - sac skill install deploys this binary's copy",
                        skill::HOME
                    );
                }
                ExitCode::FAILURE
            }
        },
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

/// The folded world a public-id-bearing command resolves against:
/// one read before the write; the fold re-validates under its lock.
fn world_of(destination: &Destination) -> Result<World, Fail> {
    let loadout = loadout_of(destination)?;
    match loadout.state {
        LoadState::Full(world) => Ok(world),
        LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
    }
}

/// The comment a public c-N names: dense birth order against the fold.
fn resolve_comment(world: &World, token: &str) -> Result<CommentId, Fail> {
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
    world.comment_at_num(n).ok_or_else(|| {
        Fail::Usage(format!(
            "c-{n} names no comment; this fold holds {} comments, c-0 through c-{}",
            world.comments.len(),
            world.comments.len().saturating_sub(1)
        ))
    })
}

/// A comment door's address: t-N homes there bare, c-N homes on that
/// comment's task and cites it - the reply as an explicit reference.
fn comment_address(world: &World, token: &str) -> Result<(TaskId, Vec<RecordId>), Fail> {
    if token.starts_with("c-") {
        let id = resolve_comment(world, token)?;
        let home = world.comments[&id].comment.home;
        return Ok((home, vec![id.0]));
    }
    Ok((parse_task_id(token)?, Vec::new()))
}

/// A citation token to the record it names: c-N a comment by its
/// public number, t-N a task's birth, a bare N any record position -
/// the fold refuses positions this log does not hold.
fn resolve_ref(world: &World, token: &str) -> Result<RecordId, Fail> {
    let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if token.starts_with("c-") {
        let id = resolve_comment(world, token)?;
        return Ok(id.0);
    }
    if token.starts_with("t-") {
        let id = parse_task_id(token)?;
        let ctx = world.tasks.get(id.0).ok_or_else(|| {
            Fail::Usage(format!(
                "no task t-{}; the fold holds {}",
                id.0,
                world.tasks.len()
            ))
        })?;
        return Ok(ctx.birth);
    }
    if numeric(token) {
        let Ok(seq) = token.parse::<usize>() else {
            return Err(Fail::Usage(format!(
                "'{token}' names no record position this log holds"
            )));
        };
        return Ok(RecordId(seq));
    }
    Err(Fail::Usage(format!(
        "'{token}' is not a citation (expected c-<n>, t-<n>, or a bare <seq>)"
    )))
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

fn serve_door(cli: &Cli, bind: &str, port: &u16) -> Result<String, Fail> {
    if cli.server.is_some() {
        return Err(Fail::Usage(
            "serve starts a server; --server selects one for clients (--db names the file to serve)".into(),
        ));
    }
    // the serving repo is the one resolved here - effects and bytes
    // belong to the declared tracker, never the caller's cwd
    let repo_root = resolve_repo(cli).ok();
    let db_path = match &cli.db {
        Some(db) => db.clone(),
        None => {
            let root = repo_root.as_ref().ok_or_else(|| {
                Fail::Usage(
                    "no repo resolved: --repo names the tracker, --db names the file to serve"
                        .into(),
                )
            })?;
            saccade::paths::db_at(root)
        }
    };
    let _ = tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(saccade::attempts::WarnsTee::beside(&db_path))
        .try_init();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| Fail::Usage(format!("runtime: {e}")))?;
    match runtime.block_on(saccade::serve::run(&db_path, repo_root, bind, *port)) {
        Err(e) => Err(Fail::Usage(e)),
        Ok(infallible) => match infallible {},
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

/// The migration door: one schema-3 source into a fresh schema-4
/// tracker, the report naming every renumbering class and every
/// legacy token left for human review.
fn migrate_door(
    cli: &Cli,
    source: &std::path::Path,
    out: &std::path::Path,
) -> Result<String, Fail> {
    let report = saccade::migrate::migrate(source, out).map_err(|e| Fail::Usage(e.to_string()))?;
    if cli.json {
        let legacy: Vec<serde_json::Value> = report
            .legacy
            .iter()
            .map(|l| serde_json::json!({"seq": l.seq, "field": l.field, "token": l.token}))
            .collect();
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "records": report.records,
            "requests_became_notes": report.requests_became_notes,
            "replies_rehomed": report.replies_rehomed,
            "uses_born_bound": report.uses_born_bound,
            "comment_refs_renumbered": report.comment_refs_renumbered,
            "record_refs_rewritten": report.record_refs_rewritten,
            "ranges_rewritten": report.ranges_rewritten,
            "incarnation_refs_renumbered": report.incarnation_refs_renumbered,
            "titles_backfilled": report.titles_backfilled,
            "legacy": legacy,
            "out": out.display().to_string(),
        }))
        .expect("the report is plain data"));
    }
    let mut lines = vec![format!(
        "migrated {} records into {}",
        report.records,
        out.display()
    )];
    lines.push(format!(
        "  {} requests became notes | {} replies rehomed | {} uses born bound",
        report.requests_became_notes, report.replies_rehomed, report.uses_born_bound
    ));
    lines.push(format!(
        "  {} comment refs renumbered | {} record refs as seq | {} ranges | {} incarnation refs",
        report.comment_refs_renumbered,
        report.record_refs_rewritten,
        report.ranges_rewritten,
        report.incarnation_refs_renumbered
    ));
    lines.push(format!(
        "  {} notes and offers titled with their home task's name",
        report.titles_backfilled
    ));
    if report.legacy.is_empty() {
        lines.push("  no unresolved references".into());
    } else {
        lines.push(format!(
            "  {} unresolved references stay legacy - review each:",
            report.legacy.len()
        ));
        for l in &report.legacy {
            lines.push(format!("    seq {} {}: {}", l.seq, l.field, l.token));
        }
    }
    Ok(lines.join("\n"))
}

fn read_only(cli: &Cli, loadout: &db::Loadout) -> Result<String, Fail> {
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
        Cmd::List { .. } => match &loadout.state {
            LoadState::Full(world) => Ok(render_tasks(cli, world, cursor)),
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason.clone())),
        },
        Cmd::Show {
            ids,
            stdin,
            withdrawn,
        } => match &loadout.state {
            LoadState::Full(world) => {
                if *stdin {
                    return show_stdin(cli, world, &loadout.rows, ids, *withdrawn, cursor);
                }
                if cli.json {
                    let blocks: Vec<serde_json::Value> = ids
                        .iter()
                        .map(|token| show_value(world, &loadout.rows, token, *withdrawn))
                        .collect::<Result<_, _>>()?;
                    return Ok(serde_json::to_string_pretty(&serde_json::json!({
                        "blocks": blocks,
                        "cursor": cursor,
                    }))
                    .expect("blocks are plain data"));
                }
                let mut blocks = Vec::new();
                for token in ids {
                    blocks.push(render_id(world, &loadout.rows, token, *withdrawn)?);
                }
                Ok(format!("{}\n{}", blocks.join("\n\n"), cursor_line(cursor)))
            }
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason.clone())),
        },
        Cmd::Search {
            terms,
            context,
            withdrawn,
        } => {
            let mut query = SearchQuery::parse(terms).map_err(|f| Fail::Usage(f.to_string()))?;
            query.withdrawn = *withdrawn;
            if let Some(around) = context {
                let anchor = anchor_seq(loadout, &query)?;
                return Ok(render_anchor(cli, &loadout.rows, anchor, *around, cursor));
            }
            match &loadout.state {
                LoadState::Full(world) => {
                    let groups = search(world, &query).map_err(|f| Fail::Usage(f.to_string()))?;
                    Ok(render_search(cli, &groups, terms, &query, cursor))
                }
                LoadState::Degraded(reason) => Err(Fail::Degraded(reason.clone())),
            }
        }
        _ => unreachable!("read_only reached from a mutating command"),
    }
}

const WAIT_POLL: std::time::Duration = std::time::Duration::from_millis(250);

fn wait_door(
    cli: &Cli,
    destination: &Destination,
    target: &str,
    after: u64,
    until: saccade::views::WaitUntil,
) -> Result<String, Fail> {
    let raw = wait_target(target)?;
    if matches!(raw, WaitToken::Comment(_)) && until == saccade::views::WaitUntil::Offer {
        return Err(Fail::Usage(
            "--until offer waits on a task (t-<n>); a comment wait is --until activity only".into(),
        ));
    }
    loop {
        let loadout = loadout_of(destination)?;
        let world = match loadout.state {
            LoadState::Full(world) => world,
            LoadState::Degraded(reason) => return Err(Fail::Degraded(reason)),
        };
        // the public number resolves against the fold being polled
        let target = match &raw {
            WaitToken::Task(id) => Target::Task(*id),
            WaitToken::Comment(n) => {
                Target::Comment(world.comment_at_num(*n).ok_or_else(|| {
                    Fail::Usage(format!(
                        "c-{n} names no comment; this fold holds {} comments",
                        world.comments.len()
                    ))
                })?)
            }
        };
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

#[derive(Debug)]
enum WaitToken {
    Task(TaskId),
    Comment(usize),
}

/// The wait door's raw token: t-N as itself, c-N as the public number
/// the fold resolves per poll.
fn wait_target(token: &str) -> Result<WaitToken, Fail> {
    let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if let Some(rest) = token.strip_prefix("c-").filter(|n| numeric(n)) {
        let n = rest.parse().map_err(|_| {
            Fail::Usage(format!(
                "'{token}' is not a wait target (expected t-<n> or c-<n>)"
            ))
        })?;
        return Ok(WaitToken::Comment(n));
    }
    if token.starts_with('#') {
        return Err(Fail::Usage(format!(
            "'{token}' is not a wait target; drop the '#': tasks are t-<n>, comments c-<n>"
        )));
    }
    Ok(WaitToken::Task(parse_task_id(token)?))
}

fn wait_target_exists(world: &saccade::World, target: &Target) -> Result<(), Fail> {
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
                    "the comment's birth record seq {} is not resident; the fold holds {} comments",
                    id.0.0,
                    world.comments.len()
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

/// The incarnation a public i-N names: dense birth order against the fold.
fn resolve_incarnation(world: &World, token: &str) -> Result<IncarnationId, Fail> {
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

/// The send door: a durable addressed message from this session's
/// registered incarnation. The record lands first; the submission
/// attempt is bounded and never waits on the recipient's work - the
/// sender chooses its own further waiting, never this verb.
fn send_door(
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
    let sent = client::send_message(&url, &context, to_num, reply_num, task_num, &body)
        .map_err(Fail::Client)?;
    // delivery activation: a running peer's bridge delivers; a stopped
    // one resumes from its exact saved session; every doubt stays
    // pending and says so - the send itself never un-lands
    // activation is the wire's own act after this landed write - one
    // mechanism for every entrance - so the door's face is the plain
    // pending line and nothing else
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
fn messages_door(
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

/// The launch door: the composed native Pi in an explicit existing
/// workspace, task-free. A fresh launch registers a new incarnation at
/// session start; --resume continues a saved one, validated against
/// its recorded native session before anything launches. A launch
/// token, when given, rides the routing so the launching door can
/// learn which incarnation registered under its launch.
fn launch_door(
    cli: &Cli,
    destination: &Destination,
    workspace: Option<&std::path::Path>,
    resume: Option<&str>,
    prompt: &[String],
) -> Result<String, Fail> {
    let (dir_str, terminal) = launch_native(cli, destination, workspace, resume, prompt, None)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "workspace": dir_str,
            "terminal": terminal,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "integrated pi launching at {dir_str} - terminal {terminal}\n  the session registers itself at start; `sac messages` reads its mailbox once it has"
    ))
}

/// One compose launch through the host: the exact workspace, the
/// terminal handle, the command line. Mechanics only - rendering and
/// any follow-on act belong to the calling door.
fn launch_native(
    cli: &Cli,
    destination: &Destination,
    workspace: Option<&std::path::Path>,
    resume: Option<&str>,
    prompt: &[String],
    launch_token: Option<&str>,
) -> Result<(String, String), Fail> {
    let url = server_url(destination, "launch")?;
    let pi = host::pinned_pi().map_err(|e| Fail::Usage(e.to_string()))?;
    let repo = resolve_repo(cli)?;
    let label = host::command_label(&pi.display().to_string());
    // routing rides the launch line: the integration reaches the
    // tracker's server directly, the repo names the tracker besides,
    // and the CLI path lets the session's own handoff helper call the
    // controller door by name instead of guessing at a PATH
    let mut routing = format!(
        "SACCADE_ACTOR={} SACCADE_SERVER={} SACCADE_REPO={}",
        host::shell_quote(&label),
        host::shell_quote(&url),
        host::shell_quote(&repo.display().to_string()),
    );
    if let Ok(exe) = std::env::current_exe() {
        routing.push_str(&format!(
            " SACCADE_CLI={}",
            host::shell_quote(&exe.display().to_string())
        ));
    }
    if let Some(token) = launch_token {
        routing.push_str(&format!(
            " SACCADE_LAUNCH_TOKEN={}",
            host::shell_quote(token)
        ));
    }
    let prompt_text = prompt.join(" ");
    let (dir, argv, title, handoff) = match (workspace, resume) {
        (Some(dir), None) => {
            let dir = dir
                .canonicalize()
                .map_err(|e| Fail::Usage(format!("workspace {}: {e}", dir.display())))?;
            if !dir.is_dir() {
                return Err(Fail::Usage(format!(
                    "{} is not an existing directory; launch selects an explicit existing workspace, it never creates one",
                    dir.display()
                )));
            }
            let mut argv = vec![pi.display().to_string()];
            if !prompt_text.is_empty() {
                argv.push(prompt_text);
            }
            (dir, argv, "sac launch".to_string(), None)
        }
        (None, Some(token)) => {
            let loadout = loadout_of(destination)?;
            let LoadState::Full(world) = loadout.state else {
                return Err(Fail::Degraded(
                    "resume needs the world; the log will not fold".into(),
                ));
            };
            let id = resolve_incarnation(&world, token)?;
            // the title and error faces render the fold's canonical
            // public number, never the typed spelling (i-i-N came from
            // concatenating both)
            let num = world
                .incarnation_num(id)
                .expect("the fold just resolved this incarnation");
            let token = &format!("i-{num}");
            let ctx = &world.incarnations[&id];
            let saccade::Incarnation::Native {
                native_id,
                saved_session,
            } = &ctx.incarnation
            else {
                return Err(Fail::Usage(format!(
                    "{token} carries no saved native session to resume; historical attribution identities have none",
                )));
            };
            // strict resume: the saved file must exist and carry the
            // recorded native id - a missing file refuses, it never
            // fabricates fresh state under the same identity
            let header = saccade::pi_session::read_header(saved_session.as_path())
                .map_err(|e| Fail::Usage(format!("{token}'s saved session cannot resume: {e}")))?;
            if header.id != native_id.as_str() {
                return Err(Fail::Usage(format!(
                    "{token}'s saved session carries id {}, not {}",
                    header.id,
                    native_id.as_str()
                )));
            }
            let dir = std::path::PathBuf::from(&header.cwd);
            if !dir.is_dir() {
                return Err(Fail::Usage(format!(
                    "the saved session's workspace {} is missing; resume starts from the header cwd, never a substitute",
                    dir.display()
                )));
            }
            let locator = saved_session
                .as_path()
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            let mut argv = vec![
                pi.display().to_string(),
                "--session".to_string(),
                native_id.as_str().to_string(),
                "--session-dir".to_string(),
                locator,
            ];
            if !prompt_text.is_empty() {
                argv.push(prompt_text);
            }
            // busy and unknown preflight, before anything is submitted:
            // a preexisting live holder is a live session, never proof
            // this unsubmitted child has resumed, and an unprobeable
            // guard leaves liveness unknown - both refuse here rather
            // than duplicate the session. The handoff observation below
            // rides out as the door's own evidence, after the host has
            // the submission.
            let guard = saccade::activate::guard_path(&repo, native_id.as_str());
            if let Some(base) = guard.parent() {
                let _ = std::fs::create_dir_all(base);
            }
            match saccade::activate::guard_held(&guard) {
                Some(false) => {}
                Some(true) => {
                    return Err(Fail::Usage(format!(
                        "{token}'s session is live: its guard is already held, so no launch is submitted; exit that session first"
                    )));
                }
                None => {
                    return Err(Fail::Usage(
                        "the session's guard cannot be probed, so liveness is unknown; no launch is submitted".into(),
                    ));
                }
            }
            (dir, argv, format!("sac resume {token}"), Some(guard))
        }
        _ => {
            return Err(Fail::Usage(
                "launch needs an explicit choice: --workspace <dir> for a fresh session, \
                 --resume i-<n> for a saved one"
                    .into(),
            ));
        }
    };
    let dir_str = dir.display().to_string();
    // the shared construction: the directory is re-validated (a launch
    // selects, it never creates), direnv resolves before anything is
    // submitted, and the host is asked only for a terminal the door can
    // equip. The reply is the host's acceptance of a submission -
    // attach and execution belong to the session itself.
    let (dir_str, command) = host::launch_command(Path::new(&dir_str), &routing, &argv)
        .map_err(|e| Fail::Usage(e.to_string()))?;
    let terminal = host::submit_terminal(&dir_str, &title, &command)
        .map_err(|e| Fail::Usage(e.to_string()))?;
    // the rigorous handoff, after the submission: this door's success
    // is the launched session's guard being held - observed, bounded,
    // never assumed, and terminal-create success alone proves neither
    // attach nor execution. A late boot holds the door until it lands
    // or the bound passes; a bound pass reports failure and the mail
    // stays relaunchable, with a late attacher refused at its own
    // attach by the live holder's guard.
    if let Some(guard) = handoff {
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs(saccade::activate::GUARD_HANDOFF_SECS);
        loop {
            match saccade::activate::guard_held(&guard) {
                Some(true) => break,
                Some(false) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(400));
                }
                _ => {
                    return Err(Fail::Usage(
                        "the resumed session's guard was not taken within its bound; the launch stays unobserved".into(),
                    ));
                }
            }
        }
    }
    Ok((dir_str, terminal))
}

/// A fresh random launch token: correlation only - it names the
/// launcher's interest, rides the launch environment where the session
/// can see it, and confers nothing on its holder. Never a record.
fn launch_token() -> String {
    let mut bytes = [0u8; 16];
    // exactly sixteen bytes: /dev/urandom never ends, so anything
    // read-to-EOF here would allocate forever
    let filled = std::fs::File::open("/dev/urandom")
        .is_ok_and(|mut file| std::io::Read::read_exact(&mut file, &mut bytes).is_ok());
    if !filled {
        // no urandom: nanos and pid distinguish concurrent doors well
        // enough for an in-memory cache, and the door refuses on a
        // collision anyway
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mix = nanos ^ ((std::process::id() as u128) << 64);
        for (at, piece) in bytes.chunks_mut(8).enumerate() {
            let shift = 8 * at;
            for (i, slot) in piece.iter_mut().enumerate() {
                *slot = (mix >> (shift + 8 * i)) as u8;
            }
        }
    }
    let mut token = String::with_capacity(32);
    for b in bytes {
        token.push_str(&format!("{b:02x}"));
    }
    token
}

/// What an ask launch did: the newborn's address and the question's,
/// with the submission attempt's truth - a refused or unacknowledged
/// submission leaves the question recorded but undisclosed to the
/// newborn, and the reply says so.
struct AskOutcome {
    workspace: String,
    terminal: String,
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
/// The newborn's address arrives by launch-token correlation; knowing
/// it confers nothing beyond the reply surface.
fn ask_run(
    cli: &Cli,
    destination: &Destination,
    workspace: &std::path::Path,
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
    let token = launch_token();
    let (dir_str, terminal) =
        launch_native(cli, destination, Some(workspace), None, &[], Some(&token))?;
    // the newborn announces itself through the launch cache; the poll
    // is transport with a hard stop, not a subscription
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let (birth, num) = loop {
        match client::launch_of(&url, &token).map_err(Fail::Client)? {
            Some(found) => break found,
            None if std::time::Instant::now() >= deadline => {
                return Err(Fail::Usage(format!(
                    "the composed session at {dir_str} did not register within 20s (terminal {terminal}); nothing was sent - it may still register, and a later `sac send --to` reaches it"
                )));
            }
            None => std::thread::sleep(std::time::Duration::from_millis(400)),
        }
    };
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
        workspace: dir_str,
        terminal,
        birth,
        num,
        message_seq: sent_message_seq,
        message_num: sent_message_num,
        creator: creator_num,
        submission,
    })
}

fn ask_door(
    cli: &Cli,
    destination: &Destination,
    workspace: &std::path::Path,
    question: &str,
) -> Result<String, Fail> {
    let outcome = ask_run(cli, destination, workspace, question)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "workspace": outcome.workspace,
            "terminal": outcome.terminal,
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
        "i-{num} registered at {workspace} (terminal {terminal}); your question went as {answer} from {creator}; they receive it through their session's input queue{disclosure}",
        num = outcome.num,
        workspace = outcome.workspace,
        terminal = outcome.terminal,
        creator = outcome.creator,
    ))
}

// -- the task-free workspace controller's CLI face -----------------------

/// Provision a NEW task-free workspace through the server's Git
/// effects; the birth record is the workspace's identity.
fn provision_new(
    destination: &Destination,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<client::WorkspaceBirth, Fail> {
    let url = server_url(destination, "workspace new")?;
    client::handshake(&url);
    let reply = client::create_workspace(&url, label, base).map_err(Fail::Client)?;
    Ok(reply.workspace)
}

/// The creation's text face: actual Git facts and the birth record.
fn render_birth(birth: &client::WorkspaceBirth) -> String {
    format!(
        "workspace provisioned - record {} (its identity, no numbering)\n  branch {} | base {}\n  path {}",
        birth.seq, birth.branch, birth.base, birth.path
    )
}

/// `sac workspace new`: provision and report; no task, claim, or
/// incarnation rides it.
fn workspace_new_door(
    cli: &Cli,
    destination: &Destination,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<String, Fail> {
    let birth = provision_new(destination, label, base)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "created": true,
            "workspace": birth,
        }))
        .expect("plain data"));
    }
    Ok(render_birth(&birth))
}

/// `sac workspace switch`: the stopped-session change through the
/// server, then an optional resume through the ordinary door. The
/// change record's truth is the header move; the resume is a separate
/// launch whose failure leaves that fact standing.
fn workspace_switch_door(
    cli: &Cli,
    destination: &Destination,
    token: &str,
    to: &Path,
    resume: bool,
    prompt: &[String],
) -> Result<String, Fail> {
    let url = server_url(destination, "workspace switch")?;
    client::handshake(&url);
    // resolve the incarnation token against the fold first, so the
    // refusal names the fold's extent; the wire's law is unchanged
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = loadout.state else {
        return Err(Fail::Degraded(
            "switch needs the world; the log will not fold".into(),
        ));
    };
    let id = resolve_incarnation(&world, token)?;
    let num = world
        .incarnation_num(id)
        .ok_or(Fail::Reject(Reject::InvalidIncarnationId))?;
    let reply = client::change_workspace(&url, num, to).map_err(Fail::Client)?;
    let record = reply.records.first().map(|r| r.seq);
    let changed = format!(
        "i-{num} workspace changed - saved cwd {from} -> {to}{record}",
        num = reply.num,
        from = reply.from,
        to = reply.to,
        record = record.map(|r| format!(" (record {r})")).unwrap_or_default(),
    );
    if resume {
        // the change fact already stands; a failed launch says so
        let (dir_str, terminal) =
            match launch_native(cli, destination, None, Some(token), prompt, None) {
                Ok(launch) => launch,
                Err(e) => return Err(launch_failed(&changed, e)),
            };
        if cli.json {
            return Ok(serde_json::to_string_pretty(&serde_json::json!({
                "changed": true,
                "incarnation": reply.incarnation,
                "num": reply.num,
                "from": reply.from,
                "to": reply.to,
                "records": reply.records,
                "resumed": {"workspace": dir_str, "terminal": terminal},
            }))
            .expect("plain data"));
        }
        return Ok(format!(
            "{changed}\nintegrated pi launching at {dir_str} - terminal {terminal}\n  the resumed session carries the same native id and history"
        ));
    }
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "changed": true,
            "incarnation": reply.incarnation,
            "num": reply.num,
            "from": reply.from,
            "to": reply.to,
            "records": reply.records,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "{changed}\n  the session was stopped; resume it there with `sac launch --resume i-{}`",
        reply.num
    ))
}

fn workspace_door(
    cli: &Cli,
    destination: &Destination,
    verb: &WorkspaceVerb,
) -> Result<String, Fail> {
    match verb {
        WorkspaceVerb::New { label, base } => {
            workspace_new_door(cli, destination, label.as_deref(), base.as_deref())
        }
        WorkspaceVerb::Switch {
            incarnation,
            to,
            resume,
            prompt,
        } => workspace_switch_door(cli, destination, incarnation, to, *resume, prompt),
    }
}

fn launch_failed(standing: &str, error: Fail) -> Fail {
    Fail::Usage(format!("{standing}\nthe launch failed: {error}"))
}

/// The explicit-new launch: provision through the server's effects,
/// then hand the fresh directory to the ordinary doors - fresh launch
/// or ask, exactly the existing composition. A launch failure leaves
/// the creation fact true.
fn launch_new_door(
    cli: &Cli,
    destination: &Destination,
    label: Option<&str>,
    base: Option<&str>,
    ask: Option<&str>,
    prompt: &[String],
) -> Result<String, Fail> {
    let birth = provision_new(destination, label, base)?;
    let path = birth.path.clone();
    let provisioned = render_birth(&birth);
    if cli.json {
        let launch = match ask {
            Some(question) => match ask_run(cli, destination, Path::new(&path), question) {
                Ok(outcome) => serde_json::json!({
                    "workspace": outcome.workspace,
                    "terminal": outcome.terminal,
                    "incarnation": {"seq": outcome.birth, "num": outcome.num},
                    "question": {"num": outcome.message_num, "seq": outcome.message_seq},
                }),
                Err(e) => return Err(launch_failed(&provisioned, e)),
            },
            None => {
                match launch_native(cli, destination, Some(Path::new(&path)), None, prompt, None) {
                    Ok((dir_str, terminal)) => {
                        serde_json::json!({"workspace": dir_str, "terminal": terminal})
                    }
                    Err(e) => return Err(launch_failed(&provisioned, e)),
                }
            }
        };
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "provisioned": birth,
            "launch": launch,
        }))
        .expect("plain data"));
    }
    let door = match ask {
        Some(question) => match ask_door(cli, destination, Path::new(&path), question) {
            Ok(door) => door,
            Err(e) => return Err(launch_failed(&provisioned, e)),
        },
        None => match launch_native(cli, destination, Some(Path::new(&path)), None, prompt, None) {
            Ok((dir_str, terminal)) => format!(
                "integrated pi launching at {dir_str} - terminal {terminal}\n  the session registers itself at start; `sac messages` reads its mailbox once it has"
            ),
            Err(e) => return Err(launch_failed(&provisioned, e)),
        },
    };
    Ok(format!("{}\n{door}", render_birth(&birth)))
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

/// The revise door's body sources: the argument, the pipe, or the file
/// - one of the three, the bytes landing verbatim.
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
    let mut lines = Vec::with_capacity(stored.len() + 2);
    if let Some(id) = born {
        lines.push(id.to_string());
    }
    lines.extend(stored.iter().map(record_line));
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

/// One log row: the record's own position as its provenance - a seq
/// is never dressed as the c-N only a comment birth earns.
fn record_line(r: &StoredRecord) -> String {
    format!(
        "seq {} {} {}/{} et={} lt={} {}",
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
/// The inspector dock as text: everything about the one thing. The
/// thread renders the addressed forest - items in first-record order,
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
    if let Some(offer) = view.offer_num {
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
            ThreadItem::Artifacts(lines) => {
                for line in lines {
                    out.extend(artifact_block(line));
                }
            }
        }
    }
    Ok(out.join("\n"))
}

/// An artifact's body format: the pointer line - name and short hash.
fn artifact_block(line: &ArtifactLine) -> Vec<String> {
    vec![
        format!("seq {}  artifact", line.seq),
        format!("  {} | {}", line.name, line.short_hash()),
    ]
}

/// A thread view's body format for one comment: its header line -
/// the public number, the citation chips naming what its mapping
/// cites, a use's chips naming its source task and bound selection -
/// the wrapped prose at a constant margin. A withdrawn note renders
/// its tombstone, the body only when revealed.
fn comment_block(line: &CommentLine, reveal: bool) -> Vec<String> {
    let chips: String = line
        .references
        .iter()
        .map(|target| format!("->{}  ", ref_str(target)))
        .collect();
    let uses = line
        .source
        .map(|t| format!("  uses t-{}", t.0))
        .unwrap_or_default();
    let bound = line
        .binding
        .as_ref()
        .map(|b| format!("  bound c-{}", b.offer))
        .unwrap_or_default();
    // the withdrawal is terminal, so its mark replaces the revised one:
    // the body is gone, its provenance mark would narrate absence -
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
        "  c-{}  {chips}{} [{}]{}{bound}{state}",
        line.num, line.actor, line.kind, uses
    )];
    if let Some(binding) = &line.binding {
        out.push(wrap(
            &format!("bound c-{} | {}", binding.offer, binding.grounds),
            WIDTH,
            "    ",
            "    ",
        ));
    }
    if line.withdrawn {
        // the tombstone: one line, the note - the title left with the
        // body and stays undisclosed even beside a revealed body
        let note = line.withdrawal_note.as_deref().unwrap_or_default();
        out.push(wrap(&format!("withdrawn: {note}"), WIDTH, "  ", "  "));
        if reveal {
            out.push(wrap(&format!("body: {}", line.body), WIDTH, "  ", "  "));
        }
        return out;
    }
    if let Some(title) = &line.title {
        out.push(wrap(&format!("title  {title}"), WIDTH, "  ", "  "));
    }
    if !line.body.is_empty() {
        out.push(wrap(&line.body, WIDTH, "  ", "  "));
    }
    out
}

/// One message's block: the addressed endpoints, the delivery state -
/// bookkeeping, never a claim of reading or agreement - and the body.
fn message_block(line: &MessageLine) -> Vec<String> {
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

/// A citation target's public name.
fn ref_str(target: &saccade::views::RefTarget) -> String {
    match target {
        saccade::views::RefTarget::Comment { num, .. } => format!("c-{num}"),
        saccade::views::RefTarget::Task { task } => format!("t-{task}"),
        saccade::views::RefTarget::Incarnation { num, .. } => format!("i-{num}"),
        saccade::views::RefTarget::Message { num } => format!("m-{num}"),
        saccade::views::RefTarget::Artifact { seq, .. } => format!("seq {seq}"),
        saccade::views::RefTarget::Record { seq, .. } => format!("seq {seq}"),
    }
}

/// One id as show sees it: a whole thread, a comment's folded form,
/// or the raw event at a bare position.
#[derive(Clone, Copy, Debug)]
enum ShowId {
    Thread(TaskId),
    /// c-<n>: the presentation the comment earns, by public number
    Comment(usize),
    /// m-<n>: the message's presentation, by public number
    Message(usize),
    /// bare <seq>: the raw event at that position
    Position(RecordId),
}

fn parse_show_id(token: &str) -> Result<ShowId, Fail> {
    let numeric = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    let not_an_id = || {
        Fail::Usage(format!(
            "'{token}' is not an id (expected c-<n>, t-<n>, m-<n>, or a bare <seq>)"
        ))
    };
    if numeric(token.strip_prefix('#').unwrap_or_default()) {
        // the retired face teaches the two doors it may have meant: a
        // comment is c-<n>, the raw record at a position the bare <seq>
        return Err(Fail::Usage(format!(
            "'{token}' is not an id (expected c-<n>, t-<n>, m-<n>, or a bare <seq>); \
             drop the '#': a comment is c-<n>, the raw record at a position is the bare <seq>"
        )));
    }
    if let Some(rest) = token.strip_prefix("c-").filter(|n| numeric(n)) {
        return Ok(ShowId::Comment(rest.parse().map_err(|_| not_an_id())?));
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

/// Render one id: a thread whole, or a record as itself - a comment
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
        ShowId::Comment(num) => {
            let id = world.comment_at_num(num).ok_or_else(|| {
                Fail::Usage(format!(
                    "c-{num} names no comment; this fold holds {} comments, c-0 through c-{}",
                    world.comments.len(),
                    world.comments.len().saturating_sub(1)
                ))
            })?;
            let line = comment_line(world, id)
                .ok_or_else(|| Fail::Usage(format!("c-{num} folds into no comment")))?;
            Ok(comment_block(&line, withdrawn_bodies).join("\n"))
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
                "offer_num": view.offer_num,
                "explanation": view.explanation,
                "entries": entries,
            }))
        }
        ShowId::Comment(num) => {
            let id = world.comment_at_num(num).ok_or_else(|| {
                Fail::Usage(format!(
                    "c-{num} names no comment; this fold holds {} comments, c-0 through c-{}",
                    world.comments.len(),
                    world.comments.len().saturating_sub(1)
                ))
            })?;
            let line = comment_line(world, id)
                .ok_or_else(|| Fail::Usage(format!("c-{num} folds into no comment")))?;
            Ok(comment_value(&line, withdrawn_bodies))
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

/// A comment as the json face renders it: the thread line's facts with
/// the prose whole - the folded bytes, never the wrapped ones. A
/// withdrawn note's body is null unless revealed, its title never
/// disclosed; the tombstone's facts ride beside it.
fn comment_value(line: &CommentLine, reveal: bool) -> serde_json::Value {
    let hidden = line.body.is_empty() || (line.withdrawn && !reveal);
    let body = if hidden {
        serde_json::Value::Null
    } else {
        serde_json::json!(line.body)
    };
    serde_json::json!({
        "kind": line.kind,
        "num": line.num,
        "seq": line.seq,
        "home": format!("t-{}", line.home.0),
        "offer": line.offer,
        "source": line.source.map(|t| format!("t-{}", t.0)),
        "title": line.title,
        "binding": line.binding.as_ref().map(|b| serde_json::json!({
            "offer": b.offer,
            "grounds": b.grounds,
        })),
        "references": line.references.iter().map(ref_str).collect::<Vec<_>>(),
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
        "seq {}  {}  {}\n{}",
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
/// line each. Counts ride the headers only when the query narrows -
/// narrowing's visibility, never an aggregate badge - and a query
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
                        "title": r.title,
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
            let line = matched_line(&query.terms, &r.text());
            out.push(format!("{prefix}{}", glimpse(&line, budget)));
        }
    }
    out.push(cursor_line(cursor));
    out.join("\n")
}

/// The anchor query names one record: c-N itself, or t-N's birth.
fn anchor_seq(loadout: &db::Loadout, query: &SearchQuery) -> Result<usize, Fail> {
    match query.terms.as_slice() {
        [Term::Comment(num)] => match &loadout.state {
            LoadState::Full(world) => {
                world.comment_at_num(*num).map(|id| id.0.0).ok_or_else(|| {
                    Fail::Usage(format!(
                        "c-{num} names no comment; this fold holds {} comments",
                        world.comments.len()
                    ))
                })
            }
            LoadState::Degraded(reason) => Err(Fail::Degraded(reason.clone())),
        },
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
            "-C anchors on one record: give a single id term (c-<n> or t-N) and no facets".into(),
        )),
    }
}

/// The anchor's window: the record, N before, N after - one line each,
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
            let prefix = format!("seq {}  {}  {}/{}", row.seq, row.kind, row.actor, row.tier);
            let budget = WIDTH.saturating_sub(prefix.chars().count() + 2);
            format!("{prefix}  {}", glimpse(&row_line(row), budget))
        })
        .collect();
    lines.push(cursor_line(cursor));
    lines.join("\n")
}

/// The record's own first text line - the same fields the log prints.
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
        cut.push_str("...");
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
            "'#t-3' is not a wait target; drop the '#': tasks are t-<n>, comments c-<n>"
        );

        let refused = wait_target("t-3").unwrap_or_else(|f| panic!("{f}"));
        assert!(matches!(refused, WaitToken::Task(TaskId(3))));
        assert!(matches!(wait_target("c-7").unwrap(), WaitToken::Comment(7)));

        // the comment door teaches against a fold: c-7 with three
        // comments names none of them
        let world = World::new();
        let refused = resolve_comment(&world, "c-7")
            .map_err(|f| f.to_string())
            .unwrap_err();
        assert_eq!(
            refused,
            "c-7 names no comment; this fold holds 0 comments, c-0 through c-0"
        );
        assert!(resolve_comment(&world, "c-x").is_err());

        assert!(matches!(
            parse_show_id("c-7").unwrap_or_else(|f| panic!("{f}")),
            ShowId::Comment(7)
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
            "'#7' is not an id (expected c-<n>, t-<n>, m-<n>, or a bare <seq>); \
             drop the '#': a comment is c-<n>, the raw record at a position is the bare <seq>"
        );

        let refused = parse_show_id("c-184467440737095516160")
            .map_err(|f| f.to_string())
            .unwrap_err();
        assert_eq!(
            refused,
            "'c-184467440737095516160' is not an id (expected c-<n>, t-<n>, m-<n>, or a bare <seq>)"
        );
        assert!(parse_show_id("m-184467440737095516160").is_err());
        assert!(parse_show_id("184467440737095516160").is_err());
        assert!(wait_target("c-184467440737095516160").is_err());
        assert!(parse_show_id("#184467440737095516160").is_err());
    }

    /// The revise door's body lands verbatim: the argument as given, the
    /// file's bytes exact - quotes, backslashes, newlines and all.
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
