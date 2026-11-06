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

    /// The host's terminal table could not be asked or its answer
    /// could not be read: the workspace's terminal facts are
    /// unverified, never absent.
    TerminalList(String),
}

impl HostFail {
    /// The wire refusal code each variant maps to at the door.
    pub fn wire_code(&self) -> &'static str {
        match self {
            HostFail::NoPi => "launch_no_pi",
            HostFail::NoDirenv => "launch_no_direnv",

            HostFail::TerminalList(_) => "workspace_host_list_failed",
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
            HostFail::TerminalList(detail) => {
                write!(f, "the host's terminal table failed: {detail}")
            }
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

/// A foreground child's own ending: the exit code it reported, or the
/// signal that ended it. The attach verb's answer - never a session's
/// own status, which no verb here observes.
#[derive(Debug, PartialEq)]
pub enum ChildExit {
    Exited(i32),
    Signalled(i32),
}

impl ChildExit {
    /// The status a shell would report for this exit: the child's own
    /// code when it exited, 128 plus the signal number when a signal
    /// ended it. An unknown ending is never a success.
    pub fn status(&self) -> i32 {
        match self {
            ChildExit::Exited(code) => *code,
            ChildExit::Signalled(sig) => 128 + sig,
        }
    }
}

/// The composed launch argv for one exact workspace: the direnv equip
/// re-spoken for direct-exec terminals - direnv exec runs the composed
/// session itself, so the session stays the terminal foreground and a
/// live probe stays honest. The env program carries the scrub and the
/// routing words, exactly as the shell line did; the workspace is the
/// caller's canonical directory. One mechanics form for every
/// entrance: the CLI doors and the workbench launch door compose
/// identically, so no face grows its own launch dialect.
pub fn composed_launch_argv(
    workspace: &std::path::Path,
    routing: &[String],
    session_argv: &[String],
) -> Result<Vec<String>, HostFail> {
    let direnv = direnv_binary()?;
    let mut argv = vec![
        direnv.display().to_string(),
        "exec".to_string(),
        workspace.display().to_string(),
        "env".to_string(),
        "-u".to_string(),
        "SACCADE_INCARNATION".to_string(),
        "-u".to_string(),
        "SACCADE_INCARNATION_SEQ".to_string(),
        "-u".to_string(),
        "SACCADE_TASK".to_string(),
        "-u".to_string(),
        "SACCADE_WORKSPACE".to_string(),
        "-u".to_string(),
        "SACCADE_DB".to_string(),
        "-u".to_string(),
        "SACCADE_CLI".to_string(),
        "-u".to_string(),
        "SACCADE_TEST_CONTEXT".to_string(),
        "-u".to_string(),
        "SACCADE_LAUNCHED_SESSION".to_string(),
    ];
    argv.extend(routing.iter().cloned());
    argv.extend(session_argv.iter().cloned());
    Ok(argv)
}

// -- the workspace lifecycle's host contract -----------------------------
// One exact workspace's terminal facts over the adapter's terminal
// table, and the close fact through the contract's own verb. The
// host's parsed answers are the only facts here: an empty answer means
// no terminals, never an unreadable one - arbitrary host errors are
// never read as absence or closure.

/// The addresses of the host's terminals whose launch cwd names the
/// exact directory, over one terminal-table read. A stranger's
/// terminal - any other launch cwd - is invisible to this contract.
pub fn workspace_terminals(
    host: &crate::herdr::HerdrCore,
    dir: &str,
) -> Result<Vec<String>, HostFail> {
    let rows = host
        .terminals()
        .map_err(|e| HostFail::TerminalList(e.to_string()))?;
    Ok(rows
        .into_iter()
        .filter(|t| t.cwd == dir)
        .map(|t| t.address)
        .collect())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn the_attribution_label_is_the_own_file_name() {
        assert_eq!(command_label("/tools/wrapped-pi"), "wrapped-pi");
        assert_eq!(command_label("pi"), "pi");
        assert_eq!(command_label(""), "saccade");
    }

    #[test]
    fn the_terminal_table_names_only_the_exact_paths_workspace() {
        let dir = std::env::temp_dir().join(format!("sac-host-ls-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hc = dir.join("hc");
        std::fs::write(
            &hc,
            r#"#!/bin/sh
printf '{"terminals":[{"address":"a1","cwd":"/exact/ws","state":"live"},{"address":"a2","cwd":"/exact/ws/deeper","state":"live"},{"address":"a3","cwd":"/exact/ws","state":"finished"}]}
'
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let addresses =
            workspace_terminals(&crate::herdr::HerdrCore::over(&hc), "/exact/ws").unwrap();
        // the exact path's terminals only: a deeper path is a stranger
        assert_eq!(addresses, vec!["a1".to_string(), "a3".to_string()]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
