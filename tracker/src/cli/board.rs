//! The board controller's doors: author, checkpoint, commit, and
//! read board revisions. Working directories are local to this
//! machine; every Git effect runs through the server's wire routes.
//! Door replies teach the next step, for both readers.

use super::*;
use std::path::Path;

/// The board controller's verbs (D4): Git names only where the
/// meaning matches - commit, diff, log - and no staging, branches,
/// or push anywhere.
#[derive(Clone, Subcommand)]
pub(crate) enum BoardVerb {
    /// Download the current revision, or your checkpoint, into a
    /// local working directory and print its absolute path. With a
    /// terminal attached and no directory, opens board.yaml in
    /// $EDITOR, then offers checkpoint or commit
    Edit {
        /// The task whose board to author (t-<n>)
        task: String,
        /// The working directory; defaults to a temp directory named
        /// for the task
        dir: Option<PathBuf>,
    },
    /// Validate a working directory locally: the board parses and
    /// its figures exist within the size bound
    Check {
        /// The working directory; defaults to the task's temp
        /// directory
        dir: Option<PathBuf>,
    },
    /// Upload a snapshot onto your checkpoint ref; nothing
    /// validates - a checkpoint is mid-edit, never a published
    /// revision
    Checkpoint { task: String, dir: Option<PathBuf> },
    /// Validate, write, and record a published revision; the parent
    /// is the commit the working directory started from
    Commit {
        task: String,
        dir: Option<PathBuf>,
        /// The commit message; omitted opens $EDITOR with a
        /// terminal, refused without one
        #[arg(short = 'm', long)]
        message: Option<String>,
    },
    /// Three-way merge your working directory onto the current
    /// commit; conflicts are marked in place
    Update { task: String, dir: Option<PathBuf> },
    /// Drop your checkpoint chain: the explicit set-aside; the
    /// objects stay in the repository
    Discard {
        /// The task whose checkpoint to drop (t-<n>)
        task: String,
    },
    /// Print one component of a revision (t-N@rK:/content/0), print
    /// one file (--file path), or write the whole revision out
    Get {
        /// The revision or component spec
        spec: String,
        /// A file in the board tree to print (or write with --out)
        #[arg(long)]
        file: Option<String>,
        /// Where to write; a whole revision defaults to a temp
        /// directory (printed), a single file to stdout
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// The board diff between two revisions: t-N@rA..rB
    Diff {
        /// The revision range
        spec: String,
    },
    /// The task's history: commits with their messages interleaved
    /// with the decisions, anchored by revision
    Log {
        /// The task (t-<n>)
        task: String,
    },
    /// The materialized current board's path on the server's
    /// machine: the viewer's open door
    Path {
        /// The task (t-<n>)
        task: String,
    },
}

/// The default working directory for a task's board: deterministic,
/// so checkpoint and commit find the directory edit created.
fn default_board_dir(task: usize) -> PathBuf {
    std::env::temp_dir().join(format!("saccade-board-t-{task}"))
}

/// Parse `t-N`, `t-N@rK`, or `t-N@rK:/path`.
struct BoardSpec {
    task: usize,
    revision: Option<usize>,
    /// The positional path naming one component
    select: Option<String>,
}

fn parse_board_spec(token: &str) -> Result<BoardSpec, Fail> {
    let (head, select) = match token.split_once(':') {
        Some((head, select)) => (head, Some(select.to_string())),
        None => (token, None),
    };
    let (task_part, revision) = match head.split_once('@') {
        Some((task_part, revision)) => {
            let revision = revision
                .strip_prefix('r')
                .and_then(|r| r.parse::<usize>().ok())
                .ok_or_else(|| {
                    Fail::Usage(format!("'{revision}' is not a revision (r<number>)"))
                })?;
            (task_part, Some(revision))
        }
        None => (head, None),
    };
    Ok(BoardSpec {
        task: parse_task_id(task_part)?.0,
        revision,
        select,
    })
}

/// Parse `t-N@rA..rB`.
fn parse_diff_spec(token: &str) -> Result<(usize, usize, usize), Fail> {
    let (task_part, range) = token
        .split_once('@')
        .ok_or_else(|| Fail::Usage(format!("'{token}' names no revision range (t-N@rA..rB)")))?;
    let (from, to) = range
        .split_once("..")
        .ok_or_else(|| Fail::Usage(format!("'{range}' is not a revision range (rA..rB)")))?;
    let parse_rev = |value: &str| {
        value
            .strip_prefix('r')
            .and_then(|r| r.parse::<usize>().ok())
            .ok_or_else(|| Fail::Usage(format!("'{value}' is not a revision (r<number>)")))
    };
    Ok((
        parse_task_id(task_part)?.0,
        parse_rev(from)?,
        parse_rev(to)?,
    ))
}

fn working_dir(dir: &Option<PathBuf>, task: usize) -> PathBuf {
    dir.clone().unwrap_or_else(|| default_board_dir(task))
}

fn recorded_base_or_teach(dir: &Path, task: usize) -> Result<String, Fail> {
    saccade::boards::recorded_base(dir).map_err(|_| {
        Fail::Taught {
            code: "boards_base_unrecorded",
            text: format!(
                "{} records no base commit; run `sac board edit t-{task} {}` first - checkpoint and update need the commit the directory started from",
                dir.display(),
                dir.display()
            ),
        }
    })
}

fn collect_files(dir: &Path) -> Result<saccade::serve::board_wire::BoardFiles, Fail> {
    saccade::serve::board_wire::BoardFiles::from_workdir(dir)
        .map_err(|detail| Fail::Usage(format!("{}: {detail}", dir.display())))
}

fn write_files(files: &saccade::serve::board_wire::BoardFiles, dir: &Path) -> Result<(), Fail> {
    files
        .write_to(dir)
        .map_err(|detail| Fail::Usage(format!("{}: {detail}", dir.display())))
}

pub(super) fn board_door(
    cli: &Cli,
    destination: &Destination,
    verb: &BoardVerb,
) -> Result<String, Fail> {
    match verb {
        BoardVerb::Check { dir } => check_door(cli, dir.as_deref()),
        BoardVerb::Edit { task, dir } => edit_door(cli, destination, task, dir.as_deref()),
        BoardVerb::Checkpoint { task, dir } => {
            checkpoint_door(cli, destination, task, dir.as_deref())
        }
        BoardVerb::Commit { task, dir, message } => {
            commit_door(cli, destination, task, dir.as_deref(), message.as_deref())
        }
        BoardVerb::Update { task, dir } => update_door(cli, destination, task, dir.as_deref()),
        BoardVerb::Discard { task } => discard_door(cli, destination, task),
        BoardVerb::Get { spec, file, out } => {
            get_door(cli, destination, spec, file.as_deref(), out.as_deref())
        }
        BoardVerb::Diff { spec } => diff_door(cli, destination, spec),
        BoardVerb::Log { task } => log_door(cli, destination, task),
        BoardVerb::Path { task } => path_door(cli, destination, task),
    }
}

fn check_door(cli: &Cli, dir: Option<&Path>) -> Result<String, Fail> {
    let dir = dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_board_dir(0));
    let board = saccade::boards::Boards::validate_workdir(&dir).map_err(|fail| Fail::Taught {
        code: fail.code(),
        text: fail.detail(),
    })?;
    let figures = board_grammar::figures(&board).len();
    if cli.json {
        return Ok(serde_json::json!({ "ok": true, "figures": figures }).to_string());
    }
    Ok(format!(
        "board.yaml parses; {figures} figures within bounds - commit with `sac board commit`"
    ))
}

fn edit_door(
    cli: &Cli,
    destination: &Destination,
    task: &str,
    dir: Option<&Path>,
) -> Result<String, Fail> {
    let url = server_url(destination, "board edit")?;
    let task_n = parse_task_id(task)?.0;
    let actor = cli
        .actor
        .clone()
        .or_else(|| std::env::var("SACCADE_ACTOR").ok());
    let reply = client::board_edit(&url, task_n, actor.as_deref()).map_err(Fail::Client)?;
    let named_dir = dir.map(Path::to_path_buf);
    let dir = named_dir
        .clone()
        .unwrap_or_else(|| default_board_dir(task_n));
    write_files(&reply.files, &dir)?;
    if let Some(base) = &reply.base {
        saccade::boards::record_base(&dir, base).map_err(|fail| Fail::Taught {
            code: fail.code(),
            text: fail.detail(),
        })?;
    }
    let interactive = named_dir.is_none() && std::io::stdin().is_terminal();
    if interactive {
        open_in_editor(&dir.join("board.yaml"))?;
        return offer_after_edit(cli, destination, task_n, &dir);
    }
    if cli.json {
        return Ok(serde_json::json!({
            "task": format!("t-{task_n}"),
            "dir": dir.display().to_string(),
            "base": reply.base,
            "commit": reply.commit,
        })
        .to_string());
    }
    let mut lines = vec![format!(
        "board {} staged at {}",
        reply
            .commit
            .as_ref()
            .map(|c| format!("t-{task_n}@{}", &c[..8.min(c.len())]))
            .unwrap_or_else(|| format!("t-{task_n} (new)")),
        dir.display()
    )];
    lines.push(format!(
        "edit {} and run: sac board commit t-{task_n} {} -m <msg>",
        dir.join("board.yaml").display(),
        dir.display()
    ));
    Ok(lines.join("\n"))
}

fn open_in_editor(path: &Path) -> Result<(), Fail> {
    if std::env::var_os("EDITOR").is_none() {
        return Err(Fail::Usage(
            "$EDITOR is unset; set it or pass an explicit directory".into(),
        ));
    }
    let status = ProcessCommand::new("sh")
        .arg("-c")
        .arg("exec $EDITOR \"$@\"")
        .arg("sac-board-edit")
        .arg(path)
        .status()
        .map_err(|error| Fail::Usage(format!("could not run $EDITOR: {error}")))?;
    if !status.success() {
        return Err(Fail::Usage(format!(
            "$EDITOR exited with {status}; the working directory stands at {}",
            path.parent()
                .and_then(Path::parent)
                .map(Path::display)
                .map(|p| p.to_string())
                .unwrap_or_default()
        )));
    }
    Ok(())
}

/// After the editor closes: the checkpoint-or-commit offer, read from
/// the terminal. Anything else leaves the working directory standing.
fn offer_after_edit(
    cli: &Cli,
    destination: &Destination,
    task: usize,
    dir: &Path,
) -> Result<String, Fail> {
    println!("board staged at {}", dir.display());
    print!("checkpoint or commit? [c/m/q] ");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|error| Fail::Usage(format!("could not read the answer: {error}")))?;
    match answer.trim() {
        "c" => checkpoint_door(cli, destination, &format!("t-{task}"), Some(dir)),
        "m" => commit_door(cli, destination, &format!("t-{task}"), Some(dir), None),
        _ => Ok(format!(
            "left as-is; commit with `sac board commit t-{task} {} -m <msg>`",
            dir.display()
        )),
    }
}

fn checkpoint_door(
    cli: &Cli,
    destination: &Destination,
    task: &str,
    dir: Option<&Path>,
) -> Result<String, Fail> {
    let url = server_url(destination, "board checkpoint")?;
    let task_n = parse_task_id(task)?.0;
    let dir = working_dir(&dir.map(Path::to_path_buf), task_n);
    let base = recorded_base_or_teach(&dir, task_n)?;
    let files = collect_files(&dir)?;
    let context = context_of(cli)?;
    let reply = client::board_checkpoint(&url, &context, task_n, &files, &base, "checkpoint")
        .map_err(Fail::Client)?;
    if cli.json {
        return Ok(serde_json::json!({
            "task": format!("t-{task_n}"),
            "checkpoint": reply.commit,
            "dir": dir.display().to_string(),
        })
        .to_string());
    }
    Ok(format!(
        "checkpoint {} saved; publish with `sac board commit t-{task_n} {} -m <msg>`",
        &reply.commit[..8.min(reply.commit.len())],
        dir.display()
    ))
}

fn commit_door(
    cli: &Cli,
    destination: &Destination,
    task: &str,
    dir: Option<&Path>,
    message: Option<&str>,
) -> Result<String, Fail> {
    let url = server_url(destination, "board commit")?;
    let task_n = parse_task_id(task)?.0;
    let dir = working_dir(&dir.map(Path::to_path_buf), task_n);
    let message = editor_prose(message.map(str::to_string), "commit message")?;
    let files = collect_files(&dir)?;
    let context = context_of(cli)?;
    let reply =
        client::board_commit(&url, &context, task_n, &files, &message).map_err(Fail::Client)?;
    saccade::boards::record_base(&dir, &reply.commit).map_err(|fail| Fail::Taught {
        code: fail.code(),
        text: fail.detail(),
    })?;
    let name = format!("t-{task_n}@r{}", reply.revision);
    if cli.json {
        return Ok(serde_json::to_string(&serde_json::json!({
            "records": reply.records,
            "commit": reply.commit,
            "revision": reply.revision,
            "name": name,
        }))
        .expect("the reply is plain data"));
    }
    let mut lines = vec![format!(
        "published {name} ({})",
        &reply.commit[..8.min(reply.commit.len())]
    )];
    lines.push(format!("offer with: sac offer t-{task_n}"));
    Ok(lines.join("\n"))
}

fn update_door(
    cli: &Cli,
    destination: &Destination,
    task: &str,
    dir: Option<&Path>,
) -> Result<String, Fail> {
    let url = server_url(destination, "board update")?;
    let task_n = parse_task_id(task)?.0;
    let dir = working_dir(&dir.map(Path::to_path_buf), task_n);
    let base = recorded_base_or_teach(&dir, task_n)?;
    let files = collect_files(&dir)?;
    let reply = client::board_update(&url, task_n, &base, &files).map_err(Fail::Client)?;
    if reply.up_to_date {
        return Ok(format!(
            "{} already sits on the current commit {}",
            dir.display(),
            &reply.base[..8.min(reply.base.len())]
        ));
    }
    write_files(&reply.files, &dir)?;
    saccade::boards::record_base(&dir, &reply.base).map_err(|fail| Fail::Taught {
        code: fail.code(),
        text: fail.detail(),
    })?;
    if cli.json {
        return Ok(serde_json::to_string(&serde_json::json!({
            "task": format!("t-{task_n}"),
            "board_conflict": reply.board_conflict,
            "figure_conflicts": reply.figure_conflicts,
            "base": reply.base,
        }))
        .expect("the reply is plain data"));
    }
    if reply.board_conflict || !reply.figure_conflicts.is_empty() {
        let mut lines = vec![format!(
            "merged onto {} with conflicts marked in {}",
            &reply.base[..8.min(reply.base.len())],
            dir.join("board.yaml").display()
        )];
        for path in &reply.figure_conflicts {
            lines.push(format!("figure conflict (kept yours): {path}"));
        }
        lines.push("resolve the conflicts, then commit".into());
        return Ok(lines.join("\n"));
    }
    Ok(format!(
        "merged onto {} cleanly; commit with `sac board commit t-{task_n} {} -m <msg>`",
        &reply.base[..8.min(reply.base.len())],
        dir.display()
    ))
}

fn discard_door(cli: &Cli, destination: &Destination, task: &str) -> Result<String, Fail> {
    let url = server_url(destination, "board discard")?;
    let task_n = parse_task_id(task)?.0;
    let context = context_of(cli)?;
    let reply = client::board_discard(&url, &context, task_n).map_err(Fail::Client)?;
    if cli.json {
        return Ok(serde_json::to_string(&reply).expect("the reply is plain data"));
    }
    Ok(format!(
        "checkpoint chain dropped for {} by {}; the objects stay in the repository",
        reply.task, reply.actor
    ))
}

fn get_door(
    cli: &Cli,
    destination: &Destination,
    spec: &str,
    file: Option<&str>,
    out: Option<&Path>,
) -> Result<String, Fail> {
    let url = server_url(destination, "board get")?;
    let parsed = parse_board_spec(spec)?;
    let revision_name = |revision: Option<usize>, latest: usize| {
        revision
            .map(|r| format!("r{r}"))
            .unwrap_or_else(|| format!("r{latest}"))
    };
    if let Some(select) = &parsed.select {
        if file.is_some() {
            return Err(Fail::Usage(
                "name the component in the spec or a file with --file, not both".into(),
            ));
        }
        if out.is_some() {
            return Err(Fail::Usage(
                "the component form prints; --out writes a file read".into(),
            ));
        }
        let reply = client::board_get_component(&url, parsed.task, parsed.revision, select)
            .map_err(Fail::Client)?;
        let bytes = decode_content(&reply.content)?;
        return String::from_utf8(bytes)
            .map_err(|_| Fail::Usage("the component is not text".into()));
    }
    if let Some(path) = file {
        let reply = client::board_get_file(&url, parsed.task, parsed.revision, path)
            .map_err(Fail::Client)?;
        let bytes = decode_content(&reply.content)?;
        match out {
            Some(out) => {
                std::fs::write(out, &bytes)
                    .map_err(|error| Fail::Usage(format!("{}: {error}", out.display())))?;
                if cli.json {
                    return Ok(
                        serde_json::json!({"file": path, "out": out.display().to_string()})
                            .to_string(),
                    );
                }
                Ok(format!("{path} written to {}", out.display()))
            }
            None => {
                let text = String::from_utf8(bytes).map_err(|_| {
                    Fail::Usage(format!("{path} is not text; pass --out to write it"))
                })?;
                Ok(text)
            }
        }
    } else {
        let reply =
            client::board_get_tree(&url, parsed.task, parsed.revision).map_err(Fail::Client)?;
        let dest = out.map(Path::to_path_buf).unwrap_or_else(|| {
            default_board_dir(parsed.task).with_extension(format!("r{}", reply.revision))
        });
        write_files(&reply.files, &dest)?;
        if cli.json {
            return Ok(serde_json::to_string(&serde_json::json!({
                "task": format!("t-{}", parsed.task),
                "revision": reply.revision,
                "commit": reply.commit,
                "out": dest.display().to_string(),
            }))
            .expect("the reply is plain data"));
        }
        Ok(format!(
            "t-{}@{} ({}) written to {}",
            parsed.task,
            revision_name(parsed.revision, reply.revision),
            &reply.commit[..8.min(reply.commit.len())],
            dest.display()
        ))
    }
}

fn decode_content(content: &str) -> Result<Vec<u8>, Fail> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(content)
        .map_err(|error| Fail::Usage(format!("the reply is not base64: {error}")))
}

fn diff_door(cli: &Cli, destination: &Destination, spec: &str) -> Result<String, Fail> {
    let url = server_url(destination, "board diff")?;
    let (task, from, to) = parse_diff_spec(spec)?;
    let reply = client::board_diff(&url, task, from, to).map_err(Fail::Client)?;
    if cli.json {
        return Ok(serde_json::to_string(&reply).expect("the reply is plain data"));
    }
    Ok(reply.diff)
}

fn log_door(cli: &Cli, destination: &Destination, task: &str) -> Result<String, Fail> {
    let url = server_url(destination, "board log")?;
    let task_n = parse_task_id(task)?.0;
    let reply = client::board_log(&url, task_n).map_err(Fail::Client)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&reply).expect("history is plain data"));
    }
    let entries = reply["history"].as_array().cloned().unwrap_or_default();
    let mut lines = Vec::with_capacity(entries.len());
    for entry in entries {
        let revision = entry["revision"].as_u64();
        let anchor = revision
            .map(|r| format!("t-{task_n}@r{r}"))
            .unwrap_or_else(|| format!("t-{task_n}"));
        let name = entry["event"].as_str().unwrap_or("?");
        let detail = entry["message"]
            .as_str()
            .or_else(|| entry["grounds"].as_str())
            .or_else(|| entry["explanation"].as_str())
            .unwrap_or("");
        let commit = entry["commit"]
            .as_str()
            .map(|c| format!(" {}", &c[..8.min(c.len())]))
            .unwrap_or_default();
        if detail.is_empty() {
            lines.push(format!("{anchor} {name}{commit}"));
        } else {
            lines.push(format!("{anchor} {name}{commit}: {detail}"));
        }
    }
    if lines.is_empty() {
        return Ok(format!("t-{task_n} has no board history yet"));
    }
    Ok(lines.join("\n"))
}

fn path_door(cli: &Cli, destination: &Destination, task: &str) -> Result<String, Fail> {
    let url = server_url(destination, "board path")?;
    let task_n = parse_task_id(task)?.0;
    let reply = client::board_path(&url, task_n).map_err(Fail::Client)?;
    let path = reply["path"].as_str().unwrap_or_default().to_string();
    if cli.json {
        return Ok(serde_json::to_string(&reply).expect("the reply is plain data"));
    }
    Ok(path)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn board_specs_parse_the_three_shapes() {
        let bare = parse_board_spec("t-5").expect("a bare task parses");
        assert_eq!(bare.task, 5);
        assert_eq!(bare.revision, None);
        assert_eq!(bare.select, None);

        let revised = parse_board_spec("t-5@r3").expect("a revision parses");
        assert_eq!(revised.task, 5);
        assert_eq!(revised.revision, Some(3));

        let selected = parse_board_spec("t-5@r3:/content/1").expect("a component parses");
        assert_eq!(selected.revision, Some(3));
        assert_eq!(selected.select.as_deref(), Some("/content/1"));

        let err = parse_diff_spec("t-5").unwrap_err().to_string();
        assert!(err.contains("no revision range"));
        let range = parse_diff_spec("t-5@r2..r4").expect("a range parses");
        assert_eq!(range, (5, 2, 4));
    }
}
