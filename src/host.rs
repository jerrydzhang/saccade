//! The host launch construction: the composed native Pi command line,
//! the direnv equip, and the host terminal submission. One mechanics
//! module for every entrance - the CLI doors and the workbench launch
//! door compose and submit identically, so no face can grow its own
//! launch dialect. Nothing here touches the record; a submission's
//! success is the host's answer, never a claim that a session runs.

use std::path::PathBuf;

/// A launch construction or host-submission refusal. The Display text
/// is the CLI's usage face byte for byte; the wire maps each variant
/// to its own code at the door.
#[derive(Debug)]
pub enum HostFail {
    /// No pinned Pi: the env override and the compile pin both missed.
    NoPi,
    /// No direnv: the doors never start a session they cannot equip.
    NoDirenv,
    /// The named workspace is not a usable existing directory.
    Workspace(String),
    /// The host binary could not be asked about the worktree.
    WorktreeShow(String),
    /// The host's cached discovery never saw the worktree.
    Undiscovered(String),
    /// The host binary could not be asked for a terminal.
    CreateSpawn(String),
    /// The host was asked and refused.
    CreateRefused(String),
    /// The host created a terminal but named no handle.
    NoHandle,
    /// The host's terminal list could not be asked or its answer could
    /// not be read: terminal closure is unverified, never absent.
    TerminalList(String),
    /// The host's terminal close could not be asked. Best-effort only:
    /// the verification list is the gate, this names the failed ask.
    TerminalClose(String),
    /// The host's visible workspace answer could not be read: visible
    /// absence is unverified, never assumed.
    WorkspaceShow(String),
}

impl HostFail {
    /// The wire refusal code each variant maps to at the door.
    pub fn wire_code(&self) -> &'static str {
        match self {
            HostFail::Workspace(_) => "workspace_path_invalid",
            HostFail::NoPi => "launch_no_pi",
            HostFail::NoDirenv => "launch_no_direnv",
            HostFail::WorktreeShow(_) | HostFail::CreateSpawn(_) => "launch_host_unavailable",
            HostFail::Undiscovered(_) => "launch_host_undiscovered",
            HostFail::CreateRefused(_) => "launch_host_refused",
            HostFail::NoHandle => "launch_host_no_handle",
            HostFail::TerminalList(_) => "workspace_host_list_failed",
            HostFail::TerminalClose(_) => "workspace_host_close_failed",
            HostFail::WorkspaceShow(_) => "workspace_host_show_failed",
        }
    }
}

impl std::fmt::Display for HostFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostFail::NoPi => write!(
                f,
                "no pinned pi: this binary carries no flake pin; export SACCADE_PI=<pi> to name one"
            ),
            HostFail::NoDirenv => write!(
                f,
                "no direnv: the launch doors equip the workspace through direnv \
                 (build pin SACCADE_DIRENV or PATH), and a launch never starts unequipped"
            ),
            HostFail::Workspace(detail) => write!(f, "{detail}"),
            HostFail::WorktreeShow(detail) => write!(f, "orca worktree show: {detail}"),
            HostFail::Undiscovered(path) => {
                write!(f, "orca has not discovered {path} yet (bounded at 90s)")
            }
            HostFail::CreateSpawn(detail) => write!(f, "orca terminal create: {detail}"),
            HostFail::CreateRefused(detail) => {
                write!(f, "orca terminal create failed: {detail}")
            }
            HostFail::NoHandle => {
                write!(f, "the terminal was created but the host named no handle")
            }
            HostFail::TerminalList(detail) => write!(f, "orca terminal list failed: {detail}"),
            HostFail::TerminalClose(detail) => write!(f, "orca terminal close failed: {detail}"),
            HostFail::WorkspaceShow(detail) => write!(f, "orca worktree show failed: {detail}"),
        }
    }
}

/// The pinned native Pi: the development override first, then the
/// flake pin this binary was built with. Never an arbitrary PATH pi.
pub fn pinned_pi() -> Result<PathBuf, HostFail> {
    if let Some(path) = std::env::var_os("SACCADE_PI").filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = option_env!("SACCADE_PI") {
        return Ok(PathBuf::from(path));
    }
    Err(HostFail::NoPi)
}

/// A launched command's attribution label: its own file name.
pub fn command_label(program: &str) -> String {
    std::path::Path::new(program)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "saccade".into())
}

/// One POSIX word, quoted so any bytes survive the shell that types it.
pub fn shell_quote(word: &str) -> String {
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

/// First executable direnv under these directories, if any, returned
/// canonical absolute: the typed command changes cwd before exec, so a
/// relative PATH entry resolved against the launch workspace would eat
/// the name.
pub fn direnv_from_paths<I>(dirs: I) -> Option<PathBuf>
where
    I: IntoIterator<Item = PathBuf>,
{
    for dir in dirs {
        let candidate = dir.join("direnv");
        let Ok(meta) = std::fs::metadata(&candidate) else {
            continue;
        };
        let executable = std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o111 != 0;
        if !(meta.is_file() && executable) {
            continue;
        }
        let Ok(absolute) = std::fs::canonicalize(&candidate) else {
            continue;
        };
        return Some(absolute);
    }
    None
}

/// The launch doors equip the workspace through direnv: the committed
/// .envrc governs every launched session exactly as it governs the
/// human's own terminal, and direnv's own trust checks decide - nothing
/// here approves anything. Resolution is the build pin or the PATH,
/// never a runtime override; an unresolvable direnv refuses before any
/// terminal exists, so the door never starts a session it cannot equip.
pub fn direnv_binary() -> Result<PathBuf, HostFail> {
    if let Some(path) = option_env!("SACCADE_DIRENV") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }
    let found = std::env::var_os("PATH")
        .map(|paths| direnv_from_paths(std::env::split_paths(&paths)))
        .unwrap_or(None);
    found.ok_or(HostFail::NoDirenv)
}

/// The line the host terminal types: the workspace becomes the cwd,
/// then direnv evaluates the first .envrc up from the workspace -
/// passthrough where none exists, direnv's refusal where trust is
/// missing - and the scrub and routing ride its environment. They ride
/// the env program because direnv exec takes a program, never VAR=
/// prefixes.
pub fn workspace_launch_command(
    direnv: &std::path::Path,
    dir: &str,
    routing: &str,
    argv: &[String],
) -> String {
    let dir_quoted = shell_quote(dir);
    format!(
        "cd {dir_quoted} && {direnv_word} exec {dir_quoted} env -u SACCADE_INCARNATION \
         -u SACCADE_INCARNATION_SEQ -u SACCADE_TASK -u SACCADE_WORKSPACE -u SACCADE_DB \
         -u SACCADE_LAUNCH_TOKEN -u SACCADE_CLI {routing} {argv_list}",
        direnv_word = shell_quote(&direnv.display().to_string()),
        routing = routing,
        argv_list = argv
            .iter()
            .map(|w| shell_quote(w))
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// The composed launch line for one explicit workspace: the directory
/// must exist (a launch selects, it never creates), direnv resolves
/// before anything is submitted. Returns the canonical directory and
/// the exact command the host terminal will type.
pub fn launch_command(
    workspace: &std::path::Path,
    routing: &str,
    argv: &[String],
) -> Result<(String, String), HostFail> {
    let dir = workspace
        .canonicalize()
        .map_err(|e| HostFail::Workspace(format!("workspace {}: {e}", workspace.display())))?;
    if !dir.is_dir() {
        return Err(HostFail::Workspace(format!(
            "{} is not an existing directory; launch selects an explicit existing workspace, it never creates one",
            dir.display()
        )));
    }
    let direnv = direnv_binary()?;
    let dir_str = dir.display().to_string();
    Ok((
        dir_str.clone(),
        workspace_launch_command(&direnv, &dir_str, routing, argv),
    ))
}

fn orca(args: &[&str]) -> Result<std::process::Output, std::io::Error> {
    std::process::Command::new("orca").args(args).output()
}

/// Wait for the host's cached discovery to see the worktree: bounded,
/// exact-path, never a second checkout.
fn orca_await_worktree(path: &str) -> Result<(), HostFail> {
    let selector = format!("path:{path}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
    loop {
        let out = orca(&["worktree", "show", "--worktree", &selector])
            .map_err(|e| HostFail::WorktreeShow(e.to_string()))?;
        if out.status.success() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(HostFail::Undiscovered(path.to_string()));
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }
}

/// The created terminal's handle from the host's pinned response shape.
fn terminal_handle(stdout: &str) -> Result<String, HostFail> {
    serde_json::from_str::<serde_json::Value>(stdout)
        .ok()
        .and_then(|v| {
            v.get("result")
                .and_then(|r| r.get("terminal"))
                .and_then(|t| t.get("handle"))
                .and_then(|h| h.as_str())
                .map(str::to_string)
        })
        .ok_or(HostFail::NoHandle)
}

/// Submit one composed launch through the host: discovery wait, the
/// terminal ask, the handle. The reply is the host's acceptance of a
/// submission - attach and execution belong to the session itself.
pub fn submit_terminal(dir: &str, title: &str, command: &str) -> Result<String, HostFail> {
    orca_await_worktree(dir)?;
    let out = orca(&[
        "terminal",
        "create",
        "--worktree",
        &format!("path:{dir}"),
        "--title",
        title,
        "--command",
        command,
        "--json",
    ])
    .map_err(|e| HostFail::CreateSpawn(e.to_string()))?;
    if !out.status.success() {
        return Err(HostFail::CreateRefused(
            String::from_utf8_lossy(&out.stderr).trim().to_string(),
        ));
    }
    terminal_handle(&String::from_utf8_lossy(&out.stdout))
}

// -- the workspace lifecycle's host contract -----------------------------
// Closure and visible absence for one exact workspace. The host's
// parsed answers are the only facts here: an empty terminal list means
// no terminals, selector_not_found means the workspace names nothing
// in the host's visible surfaces - arbitrary host errors are never
// read as absence or closure. Registration removal, branch deletion,
// and metadata erasure are not ours: git's own checkout removal is,
// and the visible read contract rides it.

/// The host's terminal handles for one exact workspace, from its own
/// list door. selector_not_found is the host naming zero terminals;
/// any other failure is unverified, never empty.
pub fn workspace_terminals(dir: &str) -> Result<Vec<String>, HostFail> {
    let out = orca(&[
        "terminal",
        "list",
        "--worktree",
        &format!("path:{dir}"),
        "--json",
    ])
    .map_err(|e| HostFail::TerminalList(e.to_string()))?;
    let body = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value = serde_json::from_str(body.trim())
        .map_err(|e| HostFail::TerminalList(format!("unreadable answer: {e}")))?;
    match value.get("ok").and_then(|ok| ok.as_bool()) {
        Some(true) => {
            let handles = value
                .pointer("/result/terminals")
                .and_then(|t| t.as_array())
                .map(|rows| {
                    rows.iter()
                        .filter_map(|row| {
                            row.get("handle")
                                .and_then(|h| h.as_str())
                                .map(str::to_string)
                        })
                        .collect()
                })
                .ok_or_else(|| {
                    HostFail::TerminalList("the answer names no terminal rows".into())
                })?;
            Ok(handles)
        }
        // the host answers for the workspace by name: no such workspace
        // names no terminals, through its own read contract
        Some(false)
            if value.pointer("/error/code").and_then(|c| c.as_str())
                == Some("selector_not_found") =>
        {
            Ok(Vec::new())
        }
        _ => Err(HostFail::TerminalList(format!(
            "the host refused the list: {body}"
        ))),
    }
}

/// Ask the host to close every terminal of one exact workspace. The
/// reply is acceptance of the ask, never confirmed closure - the
/// verification list is the gate.
pub fn close_workspace_terminals(dir: &str) -> Result<(), HostFail> {
    let out = orca(&[
        "terminal",
        "close",
        "--worktree",
        &format!("path:{dir}"),
        "--all",
        "--json",
    ])
    .map_err(|e| HostFail::TerminalClose(e.to_string()))?;
    let body = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value = serde_json::from_str(body.trim())
        .map_err(|e| HostFail::TerminalClose(format!("unreadable answer: {e}")))?;
    if value.get("ok").and_then(|ok| ok.as_bool()) == Some(true) {
        return Ok(());
    }
    // a host that already names no such workspace has nothing to close:
    // its own selector_not_found is the accepted ask
    if value.pointer("/error/code").and_then(|c| c.as_str()) == Some("selector_not_found") {
        return Ok(());
    }
    Err(HostFail::TerminalClose(format!(
        "the host refused the close: {body}"
    )))
}

/// The workspace's visible presence in the host's own surfaces.
/// Listed stands when the host names it; VisiblyAbsent stands only on
/// selector_not_found - the host declaring it names nothing. Any other
/// answer is uncertainty and refuses.
#[derive(Debug, PartialEq)]
pub enum HostPresence {
    Listed,
    VisiblyAbsent,
}

pub fn workspace_visible_presence(dir: &str) -> Result<HostPresence, HostFail> {
    let out = orca(&[
        "worktree",
        "show",
        "--worktree",
        &format!("path:{dir}"),
        "--json",
    ])
    .map_err(|e| HostFail::WorkspaceShow(e.to_string()))?;
    let body = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value = serde_json::from_str(body.trim())
        .map_err(|e| HostFail::WorkspaceShow(format!("unreadable answer: {e}")))?;
    match value.get("ok").and_then(|ok| ok.as_bool()) {
        Some(true) => Ok(HostPresence::Listed),
        Some(false)
            if value.pointer("/error/code").and_then(|c| c.as_str())
                == Some("selector_not_found") =>
        {
            Ok(HostPresence::VisiblyAbsent)
        }
        _ => Err(HostFail::WorkspaceShow(format!(
            "the host gave no usable presence answer: {body}"
        ))),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn the_launch_command_routes_through_direnv_and_keeps_the_scrub() {
        let command = workspace_launch_command(
            std::path::Path::new("/tools/direnv"),
            "/some work space",
            "SACCADE_ACTOR=pi SACCADE_SERVER=http://srv",
            &[
                "/pi/bin".to_string(),
                "--session".to_string(),
                "abc".to_string(),
            ],
        );
        assert_eq!(
            command,
            "cd '/some work space' && /tools/direnv exec '/some work space' env \
             -u SACCADE_INCARNATION -u SACCADE_INCARNATION_SEQ -u SACCADE_TASK \
             -u SACCADE_WORKSPACE -u SACCADE_DB -u SACCADE_LAUNCH_TOKEN \
             -u SACCADE_CLI SACCADE_ACTOR=pi SACCADE_SERVER=http://srv /pi/bin --session abc"
        );
    }

    #[test]
    fn direnv_resolution_demands_an_executable_file_and_lands_absolute() {
        let root =
            std::env::temp_dir().join(format!("sac-direnv-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&root);
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        assert_eq!(direnv_from_paths([bin.clone()]), None);
        std::fs::write(bin.join("direnv"), b"#!").unwrap();
        assert_eq!(
            direnv_from_paths([bin.clone()]),
            None,
            "a non-executable direnv is not a direnv"
        );
        std::fs::set_permissions(
            bin.join("direnv"),
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        let expected = std::fs::canonicalize(bin.join("direnv")).unwrap();
        // redundant and symlinked entries land on the same canonical
        // absolute, never a relative candidate the workspace would eat
        let windy = bin.join("..").join("bin");
        assert_eq!(direnv_from_paths([windy]), Some(expected.clone()));
        assert!(expected.is_absolute());
        let linked = root.join("link");
        std::os::unix::fs::symlink(&bin, &linked).unwrap();
        assert_eq!(direnv_from_paths([linked]), Some(expected));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_launch_line_refuses_a_missing_or_non_directory_workspace() {
        let root =
            std::env::temp_dir().join(format!("sac-launch-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&root);
        let missing = root.join("absent");
        let err = launch_command(&missing, "SACCADE_ACTOR=pi", &[]).unwrap_err();
        assert!(matches!(err, HostFail::Workspace(_)), "{err}");
        assert!(err.to_string().contains("workspace"), "{err}");
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("a-file");
        std::fs::write(&file, b"x").unwrap();
        let err = launch_command(&file, "SACCADE_ACTOR=pi", &[]).unwrap_err();
        assert!(
            err.to_string().contains("not an existing directory"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_attribution_label_is_the_own_file_name() {
        assert_eq!(command_label("/tools/wrapped-pi"), "wrapped-pi");
        assert_eq!(command_label("pi"), "pi");
        assert_eq!(command_label(""), "saccade");
    }

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
}
