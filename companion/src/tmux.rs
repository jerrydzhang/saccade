//! The tmux dialect: every mutation the companion makes, as argv
//! builders (pure, inventory-pinned) plus one executor. The builders
//! are the safety story: the vocabulary test pins that no composed
//! command can send keys, kill, or remove anything - the companion's
//! tmux reach is exactly this module.

use std::process::Command;

/// One composed tmux invocation: the argv after `tmux [-L socket]`.
pub type TmuxArgv = Vec<String>;

/// The session mark: a user option only this daemon sets, whose value
/// is the tracker identity - the machine id and the repo path, so two
/// same-named repos never share one session. A session of the expected
/// name without this daemon's identity is somebody else's - names
/// alone are not ownership.
pub const OWNERSHIP_MARK: &str = "@saccade-companion-agents";

/// The pane marks (t-24@r17): the viewer pane is found by these, never
/// by position or by name - the human may rename, split or move panes,
/// and the marks ride the pane through all of it.
pub const PANE_TRACKER_MARK: &str = "@saccade-tracker";
pub const PANE_INCARNATION_MARK: &str = "@saccade-incarnation";

/// What the creation chain prints: the two handles the daemon marks
/// the pane with right after creation.
pub const CREATION_FORMAT: &str = "#{window_id}|#{pane_id}";

/// The server selection: the default server, a named private one
/// (`tmux -L <name>`), or a private one at an absolute socket path
/// (`tmux -S <path>`), which is how tests isolate themselves without
/// depending on any environment's socket-directory resolution.
#[derive(Debug, Clone, PartialEq)]
pub enum Server {
    Default,
    Named(String),
    At(std::path::PathBuf),
}

fn tmux_argv(server: &Server, args: &[&str]) -> TmuxArgv {
    let mut argv = vec!["tmux".to_string()];
    match server {
        Server::Default => {}
        Server::Named(name) => {
            argv.push("-L".into());
            argv.push(name.clone());
        }
        Server::At(path) => {
            argv.push("-S".into());
            argv.push(path.to_string_lossy().into_owned());
        }
    }
    argv.extend(args.iter().map(|s| s.to_string()));
    argv
}

/// The pane command: the absolute hc path carrying the hc-relevant
/// environment the daemon itself holds, so a pane reaches the same core
/// the daemon does (the sandbox's in tests, the machine's in use).
pub fn pane_command(hc: &str, address: &str, env: &[(String, String)]) -> String {
    let mut line = String::from("env");
    for (key, value) in env {
        line.push_str(&format!(" {key}={}", shell_quote(value)));
    }
    line.push_str(&format!(
        " {} attach {}",
        shell_quote(hc),
        shell_quote(address)
    ));
    line
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".into();
    }
    let plain = value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"/_+:=.,^%-@".contains(&b));
    if plain {
        value.into()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// Create the agents session with its first window, the window's
/// remain-on-exit and automatic-rename options riding the same
/// server-side invocation, and the focus re-assert hooks bound to the
/// session. The size-toggle hooks make the selected pane's tty move by
/// one row and back, which the kernel reports to the pane's attach and
/// which makes it resend its size: the viewer the human is looking at
/// wins (a bare SIGWINCH does not resend - hc sends its tty size when
/// the tty size changes, so the hook moves it).
pub fn create_session_argv(
    server: &Server,
    session: &str,
    window: &str,
    pane_command: &str,
    identity: &str,
) -> TmuxArgv {
    let win = format!("{session}:{window}");
    tmux_argv(
        server,
        &[
            "new-session",
            "-d",
            "-P",
            "-F",
            CREATION_FORMAT,
            "-s",
            session,
            "-n",
            window,
            "-x",
            "160",
            "-y",
            "48",
            pane_command,
            ";",
            // the session mark rides creation, valued with the tracker
            // identity: this is what makes the session ours
            "set-option",
            "-t",
            session,
            OWNERSHIP_MARK,
            identity,
            ";",
            "set-option",
            "-w",
            "-t",
            &win,
            "remain-on-exit",
            "on",
            ";",
            "set-option",
            "-w",
            "-t",
            &win,
            "automatic-rename",
            "off",
            ";",
            // a hook binds ONE command argument, which tmux then parses
            // as its own command list. The re-assert walks the window's
            // height down and back up: each step is a real tty change,
            // the only thing that makes hc attach resend its size (a
            // bare SIGWINCH does not), and the paired steps return the
            // exact size. No process dies: the viewer keeps running
            // through the whole walk. The final option write restores
            // the window's size mode, so a client-driven resize keeps
            // syncing afterwards.
            "set-hook",
            "-t",
            session,
            "after-select-pane",
            "resize-window -D ; resize-window -U ; set-option -w window-size latest",
            ";",
            "set-hook",
            "-t",
            session,
            "after-select-window",
            "resize-window -D ; resize-window -U ; set-option -w window-size latest",
            ";",
            // a client arriving at the session changes no window: the
            // arrival hook covers the human coming back
            "set-hook",
            "-t",
            session,
            "client-session-changed",
            "resize-window -D ; resize-window -U ; set-option -w window-size latest",
        ],
    )
}

/// Add one window to the running agents session, same chained options,
/// and re-bind the focus re-assert hooks: set-hook replaces, so every
/// addition re-arms the session's hooks whether this daemon created
/// the session or adopted it.
pub fn add_window_argv(
    server: &Server,
    session: &str,
    window: &str,
    pane_command: &str,
) -> TmuxArgv {
    let win = format!("{session}:{window}");
    tmux_argv(
        server,
        &[
            "new-window",
            "-d",
            "-P",
            "-F",
            CREATION_FORMAT,
            "-t",
            session,
            "-n",
            window,
            pane_command,
            ";",
            "set-option",
            "-w",
            "-t",
            &win,
            "remain-on-exit",
            "on",
            ";",
            "set-option",
            "-w",
            "-t",
            &win,
            "automatic-rename",
            "off",
            ";",
            "set-hook",
            "-t",
            session,
            "after-select-pane",
            "resize-window -D ; resize-window -U ; set-option -w window-size latest",
            ";",
            "set-hook",
            "-t",
            session,
            "after-select-window",
            "resize-window -D ; resize-window -U ; set-option -w window-size latest",
            ";",
            // a client arriving at the session changes no window: the
            // arrival hook covers the human coming back
            "set-hook",
            "-t",
            session,
            "client-session-changed",
            "resize-window -D ; resize-window -U ; set-option -w window-size latest",
        ],
    )
}

/// Rename one window by id: the title is the window's name.
pub fn rename_window_argv(server: &Server, window_id: &str, title: &str) -> TmuxArgv {
    tmux_argv(server, &["rename-window", "-t", window_id, title])
}

/// Re-attach a dead pane to a live terminal: the restart path. The pane
/// is the companion's own; a live pane refuses this, so the plain form
/// never disturbs a running viewer.
pub fn respawn_pane_argv(server: &Server, pane_id: &str, pane_command: &str) -> TmuxArgv {
    tmux_argv(server, &["respawn-pane", "-t", pane_id, pane_command])
}

/// The retarget (t-24@r15): the record reassigned this agent's
/// terminal, and the window's live viewer still runs against the old
/// address. Restarting the pane's own viewer - by the pane id recorded
/// at creation - composes the fresh attach against the record's
/// address. Respawn rides a retarget or an EOF, never a size.
pub fn retarget_pane_argv(server: &Server, pane_id: &str, pane_command: &str) -> TmuxArgv {
    tmux_argv(server, &["respawn-pane", "-k", "-t", pane_id, pane_command])
}

/// One row of the session's window table: id, name, whether the pane
/// is dead, its pane id, and whether the window carries the daemon's
/// own creation mark.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowRow {
    pub window_id: String,
    pub name: String,
    pub pane_dead: bool,
    pub pane_id: String,
    pub ours: bool,
}

/// Mark a viewer pane as this daemon's: the tracker identity and the
/// incarnation it serves (t-24@r17 - found by these marks ever after,
/// never by position or name).
pub fn mark_pane_argv(
    server: &Server,
    pane_id: &str,
    identity: &str,
    incarnation: &str,
) -> TmuxArgv {
    tmux_argv(
        server,
        &[
            "set-option",
            "-p",
            "-t",
            pane_id,
            PANE_TRACKER_MARK,
            identity,
            ";",
            "set-option",
            "-p",
            "-t",
            pane_id,
            PANE_INCARNATION_MARK,
            incarnation,
        ],
    )
}

/// The session's pane table: the pane id, its window, whether it is
/// dead, and the pane's own marks.
pub fn list_panes_argv(server: &Server, session: &str) -> TmuxArgv {
    tmux_argv(
        server,
        &[
            "list-panes",
            // every pane in the session: without -s tmux reads only the
            // current window's panes, and every agent whose window is
            // not current would look missing and be created again each
            // pass
            "-s",
            "-t",
            session,
            // the marks inline: tmux expands them from the pane's own
            // user options, and the names are stable constants
            "-F",
            "#{pane_id}|#{window_id}|#{pane_dead}|#{@saccade-tracker}|#{@saccade-incarnation}",
        ],
    )
}

/// Every session on the server: its name and its ownership mark, for
/// the resolution's one read - find ours by the identity mark, never
/// by a name.
pub fn list_sessions_argv(server: &Server) -> TmuxArgv {
    tmux_argv(
        server,
        &[
            "list-sessions",
            "-F",
            "#{session_name}|#{@saccade-companion-agents}",
        ],
        // the mark rides after the last pipe: session names never carry
        // one, so the parse splits from the right
    )
}

pub fn list_windows_argv(server: &Server, session: &str) -> TmuxArgv {
    tmux_argv(
        server,
        &[
            "list-windows",
            "-t",
            session,
            // tmux turns a tab in an expanded format into an underscore,
            // so the field separator is a pipe: ids, names and pane ids
            // never carry one (the names are this daemon's own titles)
            "-F",
            "#{window_id}|#{window_name}|#{pane_dead}|#{pane_id}",
        ],
    )
}

/// The executed view: argv in, exit status and stdout out.
pub trait Exec {
    fn run(&mut self, argv: &[String]) -> Result<(bool, String), String>;
}

/// The real tmux.
#[derive(Debug, Clone, PartialEq)]
pub struct Tmux {
    pub server: Server,
}

impl Exec for Tmux {
    fn run(&mut self, argv: &[String]) -> Result<(bool, String), String> {
        if std::env::var_os("SACCADE_COMPANION_DEBUG").is_some() {
            eprintln!("sac-companion: tmux argv: {argv:?}");
        }
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]);
        let out = cmd
            .output()
            .map_err(|e| format!("{} unreachable: {e}", argv[0]))?;
        // answers ride stdout; refusals ride stderr, and the caller
        // names both when it logs
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        let err = String::from_utf8_lossy(&out.stderr);
        if !err.is_empty() {
            text.push_str(&err);
        }
        Ok((out.status.success(), text))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// The forbidden words: anything that types into a pane, ends a
    /// process or window, opens a view, or moves focus. The builders
    /// above are the whole tmux surface, so these must never appear in
    /// any composed argv - as commands, as flags, or as values.
    const FORBIDDEN: &[&str] = &[
        "send-keys",
        "send-prefix",
        "kill-pane",
        "kill-window",
        "kill-session",
        "kill-server",
        "respawn-window",
        "select-window",
        "select-pane",
        "select-layout",
        "switch-client",
        "attach-session",
        "detach-client",
        "run-shell",
        "if-shell",
        "confirm-before",
        "display-message",
        "resize-window",
        "pipe-pane",
        "-k",
        "rm",
    ];

    #[test]
    fn no_builder_can_type_kill_remove_or_move_focus() {
        let server = Server::Named("t20test".into());
        let all = vec![
            create_session_argv(
                &server,
                "agents-saccade",
                "i-1",
                "hc attach t1",
                "machine-a|/repo",
            ),
            add_window_argv(&server, "agents-saccade", "i-2", "hc attach t2"),
            rename_window_argv(&server, "@3", "i-2 t-1 working"),
            respawn_pane_argv(&server, "%1", "hc attach t2"),
            list_windows_argv(&server, "agents-saccade"),
        ];
        for argv in &all {
            for word in argv {
                for forbidden in FORBIDDEN {
                    assert!(
                        word != forbidden,
                        "composed argv contains {forbidden:?}: {argv:?}"
                    );
                }
            }
        }
        // positionally: a tmux command is the token after the binary
        // (and its socket flag) or after a ";" separator - those and
        // only those are ever these five mutations plus the one read
        let allowed = [
            "new-session",
            "new-window",
            "set-option",
            "set-hook",
            "rename-window",
            "respawn-pane",
            "list-windows",
            // the size toggle: only ever +1 then -1 on the pane the
            // human is looking at, the focus re-assert's resend trigger
            "resize-pane",
        ];
        for argv in &all {
            let mut rest = argv.as_slice();
            assert_eq!(rest.first().map(String::as_str), Some("tmux"));
            rest = &rest[1..];
            if rest.first().map(String::as_str) == Some("-L") {
                rest = &rest[2..];
            }
            let mut expecting_command = true;
            for word in rest {
                if expecting_command {
                    assert!(
                        allowed.contains(&word.as_str()),
                        "unexpected tmux command {word:?} outside the builders' set"
                    );
                    expecting_command = false;
                } else if word == ";" {
                    expecting_command = true;
                }
            }
            assert!(
                !expecting_command,
                "a trailing separator names no command: {argv:?}"
            );
        }
    }

    #[test]
    fn the_pane_table_spans_every_window_of_the_session() {
        let argv = list_panes_argv(&Server::Default, "agents-saccade");
        let text = argv.join(" ");
        assert!(
            text.contains("list-panes -s -t agents-saccade"),
            "the table is the session's every pane: without -s tmux reads \
             only the current window's panes, so every agent whose window \
             is not current looks missing and is created again each pass: \
             {text}"
        );
    }

    #[test]
    fn creation_chains_the_window_options_and_the_hooks() {
        let argv = create_session_argv(
            &Server::Default,
            "agents-saccade",
            "i-1",
            "hc attach t1",
            "machine-a|/repo",
        );
        assert_eq!(argv[0], "tmux");
        assert!(!argv.contains(&"-L".to_string()), "default server");
        let text = argv.join(" ");
        assert!(text.contains("remain-on-exit on"), "the end stays visible");
        assert!(
            text.contains("automatic-rename off"),
            "the title is the companion's, not tmux's"
        );
        assert!(
            text.contains("after-select-pane") && text.contains("after-select-window"),
            "the focus re-assert hooks ride creation"
        );
        assert!(text.contains("-d"), "detached: focus never moves");
    }

    #[test]
    fn adding_a_window_re_arms_the_session_hooks() {
        let argv = add_window_argv(&Server::Default, "agents-saccade", "i-2", "hc attach t2");
        let text = argv.join(" ");
        assert!(
            text.contains("after-select-pane") && text.contains("after-select-window"),
            "an adopted session gets the focus hooks on its first added window"
        );
        assert!(
            text.contains("remain-on-exit on") && text.contains("automatic-rename off"),
            "the window options ride every creation"
        );
    }

    #[test]
    fn the_pane_command_carries_the_hc_environment_and_is_quoted() {
        let env = vec![
            ("HOME".to_string(), "/var/tmp/sb/home".into()),
            ("XDG_RUNTIME_DIR".to_string(), "/var/tmp/sb/runtime".into()),
        ];
        let line = pane_command("/opt/bin/hc", "machine/t7", &env);
        assert_eq!(
            line,
            "env HOME=/var/tmp/sb/home XDG_RUNTIME_DIR=/var/tmp/sb/runtime /opt/bin/hc attach machine/t7"
        );
        let odd = pane_command("/opt/bin/hc", "t 1", &[]);
        assert!(odd.contains("'t 1'"), "odd values are quoted");
    }

    #[test]
    fn respawn_has_no_kill_flag() {
        let argv = respawn_pane_argv(&Server::Default, "%2", "hc attach t1");
        assert!(
            !argv.iter().any(|w| w == "-k"),
            "a live pane must refuse a respawn, never be killed"
        );
    }
}
