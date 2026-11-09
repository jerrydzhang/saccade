//! The reconciler: one state machine owning every tmux mutation and
//! every hc read. Enrollment arrives from the page watcher, motion
//! from the Herdr watcher; between feeds the companion does nothing -
//! workflow fires on triggers, never clocks. Each feed lands as one
//! decide pass over owned data, then one execute pass through the
//! tmux executor; the vocabulary those passes can express is pinned
//! by the tmux module's inventory.

use crate::herdr::{self, TerminalRow};
use crate::page::AgentRow;
use crate::tmux::{self, Exec, Server, WindowRow};
use std::collections::HashMap;

/// One feed event.
#[derive(Debug, Clone)]
pub enum Feed {
    /// The root page named the tracker: the session's base name and
    /// the tracker identity (machine id and repo path) that marks and
    /// finds it. No session exists until both are known (t-24@r18).
    Tracker { base: String, identity: String },
    /// An agents page answered: the enrolled sessions as they stand.
    Snapshot(Vec<AgentRow>),
    /// A status event off the stream: the program's new word, or none
    /// when the exit nulls it.
    Word {
        terminal: String,
        word: Option<String>,
    },
    /// A finished event off the stream, with the core's reason.
    Finished { terminal: String, reason: String },
    /// The stream died: a restart signal, every time it happens.
    HerdrRestart,
    /// The subscription's own snapshot (or a missed-recovery picture):
    /// the terminals as they stand, current words included. Freshness
    /// after a gap, and the recovery path for words missed while the
    /// stream was down.
    TerminalSnapshot(Vec<TerminalRow>),
}

/// The viewer pane as the pane table carries it, with its own marks.
#[derive(Debug, Clone, PartialEq)]
struct PaneRow {
    pane_id: String,
    window_id: String,
    pane_dead: bool,
    tracker_mark: String,
    incarnation_mark: String,
}

/// One enrolled agent's life in the companion.
#[derive(Debug, Clone)]
struct Agent {
    incarnation: String,
    address: String,
    /// The address the window's viewer is composed against: the
    /// retarget signal is this diverging from `address`.
    attached_address: Option<String>,
    tasks: Vec<String>,
    word: Option<String>,
    end: Option<String>,
    /// The viewer pane as of the last pass, by its own marks: the
    /// memory that distinguishes "never created" from "seen, then the
    /// human closed it".
    pane: Option<PaneRow>,
    /// The window was seen and later vanished while the terminal was
    /// live: the human closed it. Never recreated this run.
    suppressed: bool,
}

/// One mutation decided by a pass. Nothing else is ever expressible -
/// no keys, no kills, no removes, no focus.
#[derive(Debug, Clone)]
enum Action {
    Create {
        incarnation: String,
        argv: Vec<String>,
    },
    Rename {
        incarnation: String,
        argv: Vec<String>,
    },
    Respawn {
        incarnation: String,
        argv: Vec<String>,
    },
}

/// The whole companion state plus its executors.
pub struct Companion<E: Exec> {
    tmux: E,
    hc: String,
    server: Server,
    hc_env: Vec<(String, String)>,
    session: Option<String>,
    identity: Option<String>,
    agents: HashMap<String, Agent>,
    /// The last terminal table: liveness is Herdr's word, read when
    /// the page changes and when the stream restarts, never a poll.
    terminals: Vec<TerminalRow>,
    /// Whether that table is a fresh read. A failed read clears the
    /// table - an unreadable core means unknown, never stale answers.
    terminals_fresh: bool,
    /// Set when the stream dies; consumed by the next pass.
    restart_pending: bool,
    /// What the passes did, in order (the account of one run).
    pub journal: Vec<String>,
    /// The last command's stdout (the creation capture reads its
    /// handles from here).
    last_output: String,
}

impl<E: Exec> Companion<E> {
    pub fn new(tmux: E, hc: &str, server: Server, hc_env: Vec<(String, String)>) -> Self {
        Companion {
            tmux,
            hc: hc.to_string(),
            server,
            hc_env,
            session: None,
            identity: None,
            agents: HashMap::new(),
            terminals: Vec::new(),
            terminals_fresh: false,
            restart_pending: false,
            journal: Vec::new(),
            last_output: String::new(),
        }
    }

    /// The window title: `i-<n>[ t-<m>[+k]][ <word>| ended:<reason>]`.
    fn title(agent: &Agent) -> String {
        let mut title = agent.incarnation.clone();
        if let Some(latest) = agent.tasks.first() {
            title.push(' ');
            title.push_str(latest);
            if agent.tasks.len() > 1 {
                title.push_str(&format!("+{}", agent.tasks.len() - 1));
            }
        }
        if let Some(reason) = &agent.end {
            title.push_str(&format!(" ended:{reason}"));
        } else if let Some(word) = &agent.word {
            title.push(' ');
            title.push_str(word);
        }
        title
    }

    fn pane_command(&self, address: &str) -> String {
        tmux::pane_command(&self.hc, address, &self.hc_env)
    }

    fn state_of(&self, address: &str) -> Option<&str> {
        self.terminals
            .iter()
            .find(|row| row.address == address)
            .map(|row| row.state.as_str())
    }

    /// Apply one feed and reconcile. Freshness is per pass: a table
    /// read for an earlier feed is stale for this one's decisions.
    pub fn accept(&mut self, feed: Feed) {
        self.terminals_fresh = false;
        match feed {
            Feed::Tracker { base, identity } => {
                let sanitized: String = base
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                self.session = Some(format!("agents-{sanitized}"));
                self.identity = Some(identity);
            }
            Feed::Snapshot(rows) => {
                self.enroll(rows);
            }
            Feed::Word { terminal, word } => {
                if let Some(agent) = self.by_address_mut(&terminal) {
                    agent.word = word;
                }
            }
            Feed::Finished { terminal, reason } => {
                if let Some(agent) = self.by_address_mut(&terminal) {
                    agent.end = Some(reason);
                    // the suppression dies with the terminal (t-24@r13):
                    // a kept terminal that comes back re-derives its
                    // window like any other
                    agent.suppressed = false;
                }
            }
            Feed::HerdrRestart => {
                self.restart_pending = true;
                // a fresh subscription pairs with a fresh picture: the
                // kept terminals that are live again re-attach
                self.read_terminals();
            }
            Feed::TerminalSnapshot(rows) => {
                // the snapshot is the freshest word there is: the truth
                // at subscribe time, words missed while away included
                self.terminals = rows;
                self.terminals_fresh = true;
                self.restart_pending = true;
                for agent in self.agents.values_mut() {
                    if let Some(row) = self
                        .terminals
                        .iter()
                        .find(|row| row.address == agent.address)
                    {
                        agent.word = row.word.clone();
                    }
                }
            }
        }
        self.reconcile();
    }

    fn by_address_mut(&mut self, address: &str) -> Option<&mut Agent> {
        self.agents
            .values_mut()
            .find(|agent| agent.address == address)
    }

    fn enroll(&mut self, rows: Vec<AgentRow>) {
        // the page is pure record and cannot carry liveness, so the
        // companion reads the terminal table when the page changes
        self.read_terminals();
        for AgentRow {
            incarnation,
            address,
            tasks,
            ..
        } in rows
        {
            let Some(address) = address else {
                continue; // no attach token, no window
            };
            let agent = self
                .agents
                .entry(incarnation.clone())
                .or_insert_with(|| Agent {
                    incarnation,
                    address: address.clone(),
                    attached_address: None,
                    tasks: Vec::new(),
                    word: None,
                    end: None,
                    pane: None,
                    suppressed: false,
                });
            agent.address = address;
            agent.tasks = tasks;
        }
    }

    fn read_terminals(&mut self) {
        let argv = vec![
            self.hc.clone(),
            "ls".into(),
            "--all".into(),
            "--json".into(),
        ];
        match self.tmux.run(&argv) {
            Ok((true, out)) => match herdr::parse_ls(&out) {
                Ok(rows) => {
                    self.terminals = rows;
                    self.terminals_fresh = true;
                }
                Err(fail) => {
                    // an unreadable core is unknown, never stale: the
                    // table clears so no pass acts on old liveness
                    self.terminals.clear();
                    self.terminals_fresh = false;
                    self.journal.push(format!("hc ls unparsed: {fail}"));
                }
            },
            Ok((false, err)) => {
                self.terminals.clear();
                self.terminals_fresh = false;
                self.journal.push(format!("hc ls refused: {err}"));
            }
            Err(fail) => {
                self.terminals.clear();
                self.terminals_fresh = false;
                self.journal.push(format!("hc ls failed: {fail}"));
            }
        }
    }

    fn run(&mut self, argv: &[String], what: &str) -> bool {
        match self.tmux.run(argv) {
            Ok((true, out)) => {
                self.last_output = out;
                self.journal.push(format!("{what}: ok"));
                true
            }
            Ok((false, err)) => {
                self.journal.push(format!("{what} refused: {err}"));
                false
            }
            Err(fail) => {
                self.journal.push(format!("{what} failed: {fail}"));
                false
            }
        }
    }

    fn reconcile(&mut self) {
        let (Some(base), Some(identity)) = (self.session.clone(), self.identity.clone()) else {
            return; // no identity or no base name yet: tmux stays untouched
        };
        // the session is found by its mark (whatever the human renamed
        // it to); when none carries the mark, the first free candidate
        // name births it - and when every candidate is taken by
        // others, this pass touches nothing
        let Some((session, session_exists)) = self.resolve_session(&base, &identity) else {
            self.journal
                .push(format!("no free session name for {base}; untouched"));
            return;
        };

        // --- decide: owned data only, no executor calls below ---
        let windows = self.read_windows(&session);
        let panes = self.read_panes(&session);
        let live: HashMap<String, bool> = self
            .agents
            .values()
            .map(|agent| {
                (
                    agent.address.clone(),
                    self.state_of(&agent.address) == Some("live"),
                )
            })
            .collect();
        let states: HashMap<String, String> = self
            .agents
            .values()
            .map(|agent| {
                (
                    agent.address.clone(),
                    self.state_of(&agent.address).unwrap_or("gone").to_string(),
                )
            })
            .collect();
        let panes_map: HashMap<String, String> = self
            .agents
            .values()
            .map(|agent| (agent.address.clone(), self.pane_command(&agent.address)))
            .collect();
        if std::env::var_os("SACCADE_COMPANION_DEBUG").is_some() {
            for agent in self.agents.values() {
                eprintln!(
                    "sac-companion: decide {} addr={} suppressed={} end={:?} live={} pane={:?}",
                    agent.incarnation,
                    agent.address,
                    agent.suppressed,
                    agent.end,
                    live.get(&agent.address).copied().unwrap_or(false),
                    agent.pane.as_ref().map(|p| p.pane_id.clone()),
                );
            }
            eprintln!("sac-companion: panes={panes:?}");
        }
        let mut creates = Vec::new();
        let mut respawns = Vec::new();
        // the first creation of this pass births the session; the rest
        // add windows into it, so one snapshot full of live agents
        // enrolls every one of them without waiting for another feed
        let mut session_urns_birth = session_exists;

        for agent in self.agents.values_mut() {
            // the viewer pane is found by its own marks (t-24@r17):
            // tracker identity plus incarnation, never by position or
            // by name - the human may rename, split or move panes, and
            // the marks ride the pane through all of it
            let now = panes
                .iter()
                .find(|pane| {
                    pane.tracker_mark == identity && pane.incarnation_mark == agent.incarnation
                })
                .cloned();
            let previous = agent.pane.take();
            let live_now = live.get(&agent.address).copied().unwrap_or(false);
            #[cfg(test)]
            eprintln!(
                "sac-companion: decide {} prev_pane={} now_pane={} live={live_now} end={:?} fresh={}",
                agent.incarnation,
                previous.is_some(),
                now.is_some(),
                agent.end,
                self.terminals_fresh
            );

            match (previous, now) {
                // seen, then gone, terminal still live: the human
                // closed it - never recreated this run
                (Some(_), None) if live_now && agent.end.is_none() => {
                    agent.suppressed = true;
                    continue;
                }
                (Some(_), None) => {
                    continue;
                }
                (_, Some(pane)) => {
                    // an end the stream missed (the pane died and the
                    // terminal finished while the subscription was
                    // down) still marks, from the table's own word
                    if pane.pane_dead && agent.end.is_none() && self.terminals_fresh && !live_now {
                        agent.end = Some(states.get(&agent.address).cloned().unwrap_or_default());
                        // the suppression dies with the terminal
                        agent.suppressed = false;
                    }
                    if self.terminals_fresh && live_now && agent.end.is_some() {
                        // a kept terminal came back live: the marked
                        // end is stale, the agent runs again - whatever
                        // the viewer pane's own state
                        agent.end = None;
                    }
                    agent.pane = Some(pane);
                }
                (None, None) => {
                    if agent.suppressed || !live_now {
                        continue;
                    }
                    // startup, or a fresh registration: open the window
                    let cmd = panes_map.get(&agent.address).cloned().unwrap_or_default();
                    let argv = if session_urns_birth {
                        tmux::add_window_argv(&self.server, &session, &agent.incarnation, &cmd)
                    } else {
                        session_urns_birth = true;
                        tmux::create_session_argv(
                            &self.server,
                            &session,
                            &agent.incarnation,
                            &cmd,
                            &identity,
                        )
                    };
                    creates.push(Action::Create {
                        incarnation: agent.incarnation.clone(),
                        argv,
                    });
                }
            }
        }

        // a retarget: the record reassigned this agent's terminal, and
        // the window's live viewer still runs against the old address.
        // The viewer is the companion's own, so the retarget restarts
        // it composed against the record's address (respawn-pane -k, by
        // the pane's own id, answering the record's word - t-24@r15).
        for agent in self.agents.values() {
            let Some(pane) = &agent.pane else {
                continue;
            };
            if pane.pane_dead || agent.end.is_some() {
                continue;
            }
            if agent.attached_address.as_deref() == Some(agent.address.as_str()) {
                continue;
            }
            // the record is the authority here: it reassigned the
            // terminal, so the viewer composes against the new address
            // whatever the last table said
            let cmd = panes_map.get(&agent.address).cloned().unwrap_or_default();
            respawns.push(Action::Respawn {
                incarnation: agent.incarnation.clone(),
                argv: tmux::retarget_pane_argv(&self.server, &pane.pane_id, &cmd),
            });
        }

        // dead panes on live terminals after the stream died: the kept
        // terminals are back, re-attach them. The signal holds until it
        // is fully consumed: a kept terminal that is not live yet (the
        // core still coming back) keeps it armed, so a later feed
        // finishes the re-attach without a fresh EOF.
        if self.restart_pending {
            let mut consumed = true;
            // a core that cannot be read yet holds the signal: the
            // re-attach waits for a fresh table that shows the kept
            // terminal live again
            if !self.terminals_fresh {
                consumed = false;
            }
            for agent in self.agents.values() {
                let Some(pane) = &agent.pane else {
                    continue;
                };
                if !pane.pane_dead || agent.end.is_some() {
                    continue;
                }
                if live.get(&agent.address).copied().unwrap_or(false) {
                    let cmd = panes_map.get(&agent.address).cloned().unwrap_or_default();
                    respawns.push(Action::Respawn {
                        incarnation: agent.incarnation.clone(),
                        argv: tmux::respawn_pane_argv(&self.server, &pane.pane_id, &cmd),
                    });
                } else {
                    consumed = false;
                }
            }
            if consumed {
                self.restart_pending = false;
            }
        }

        // --- execute: creations, then re-attach, then titles ---
        for action in &creates {
            let Action::Create { incarnation, argv } = action else {
                unreachable!()
            };
            let what = format!("window {incarnation}");
            if self.run(argv, &what) {
                // the creation output named this window's handles; the
                // pane is marked ours right away, so the next pass
                // finds it by mark, never by name
                // the creation output is "<window id>|<pane id>"
                let ids = self.last_output.trim().to_string();
                let pane_id = ids.split('|').nth(1).unwrap_or_default().to_string();
                if !pane_id.is_empty() {
                    let mark = tmux::mark_pane_argv(&self.server, &pane_id, &identity, incarnation);
                    self.run(&mark, &format!("mark {incarnation}"));
                }
            }
        }
        if !creates.is_empty() {
            // adopt the fresh panes from the pane table (they carry
            // their marks now)
            let table = self.read_panes(&session);
            let by_incarnation: HashMap<String, PaneRow> = table
                .iter()
                .filter(|pane| pane.tracker_mark == identity)
                .map(|pane| (pane.incarnation_mark.clone(), pane.clone()))
                .collect();
            for agent in self.agents.values_mut() {
                if agent.pane.is_none() {
                    agent.pane = by_incarnation.get(&agent.incarnation).cloned();
                }
                agent.attached_address = Some(agent.address.clone());
            }
        }
        for action in &respawns {
            let Action::Respawn { incarnation, argv } = action else {
                unreachable!()
            };
            let what = if argv.iter().any(|w| w == "-k") {
                format!("retarget {incarnation}")
            } else {
                format!("re-attach {incarnation}")
            };
            self.run(argv, &what);
            if let Some(agent) = self.agents.get_mut(incarnation) {
                agent.attached_address = Some(agent.address.clone());
            }
        }
        let renames: Vec<Action> = self
            .agents
            .values()
            .filter_map(|agent| {
                let pane = agent.pane.as_ref()?;
                let window_id = pane.window_id.as_str();
                let title = Self::title(agent);
                let name = windows
                    .iter()
                    .find(|w| w.window_id == window_id)
                    .map(|w| w.name.clone())
                    .unwrap_or_default();
                (name != title).then(|| Action::Rename {
                    incarnation: agent.incarnation.clone(),
                    argv: tmux::rename_window_argv(&self.server, window_id, &title),
                })
            })
            .collect();
        for action in &renames {
            let Action::Rename { incarnation, argv } = action else {
                unreachable!()
            };
            let what = format!("rename {incarnation}");
            self.run(argv, &what);
        }
    }

    fn read_panes(&mut self, session: &str) -> Vec<PaneRow> {
        let argv = tmux::list_panes_argv(&self.server, session);
        match self.tmux.run(&argv) {
            Ok((true, out)) => parse_panes(&out),
            Ok((false, _)) => Vec::new(),
            Err(fail) => {
                self.journal.push(format!("list-panes failed: {fail}"));
                Vec::new()
            }
        }
    }

    /// Find this tracker's session: the one whose ownership mark names
    /// our identity, whatever the human renamed it to. When none
    /// carries it, walk the candidate names (base, then the identity
    /// suffix, then further suffixes) and take the first FREE one - the
    /// creation will birth and mark it. A taken name is somebody
    /// else's; if every candidate is taken, the answer is None and the
    /// pass touches nothing.
    fn resolve_session(&mut self, base: &str, identity: &str) -> Option<(String, bool)> {
        let argv = tmux::list_sessions_argv(&self.server);
        let sessions = match self.tmux.run(&argv) {
            Ok((true, out)) => out,
            _ => String::new(),
        };
        let mut ours: Option<String> = None;
        let mut named: Vec<String> = Vec::new();
        for line in sessions.lines().filter(|line| !line.is_empty()) {
            let mut parts = line.splitn(2, '|');
            let (Some(name), Some(mark)) = (parts.next(), parts.next()) else {
                continue;
            };
            if mark == identity && ours.is_none() {
                ours = Some(name.to_string());
            }
            named.push(name.to_string());
        }
        if let Some(ours) = ours {
            return Some((ours, true));
        }
        // the collision suffix: six hex of the tracker identity
        let mut hash: u32 = 0x811c_9dc5;
        for byte in identity.as_bytes() {
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(0x0100_0193);
        }
        let suffix = format!("{hash:06x}");
        let mut candidate = base.to_string();
        let mut walk = 1;
        loop {
            if !named.contains(&candidate) {
                return Some((candidate, false));
            }
            candidate = if walk == 1 {
                format!("{base}-{suffix}")
            } else {
                format!("{base}-{suffix}-{walk}")
            };
            walk += 1;
            if walk > 8 {
                return None;
            }
        }
    }

    fn read_windows(&mut self, session: &str) -> Vec<WindowRow> {
        let argv = tmux::list_windows_argv(&self.server, session);
        match self.tmux.run(&argv) {
            Ok((true, out)) => parse_windows(&out),
            Ok((false, _)) => Vec::new(),
            Err(fail) => {
                self.journal.push(format!("list-windows failed: {fail}"));
                Vec::new()
            }
        }
    }
}

/// Parse the pane table's rows. The first three fields are fixed
/// shapes; the marks ride the rest, split from the right - the tracker
/// identity carries the repo path, pipes included.
fn parse_panes(out: &str) -> Vec<PaneRow> {
    out.lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let mut parts = line.splitn(4, '|');
            let pane_id = parts.next()?.to_string();
            let window_id = parts.next()?.to_string();
            let pane_dead = parts.next()? == "1";
            let marks = parts.next()?.to_string();
            let (tracker_mark, incarnation_mark) = marks.rsplit_once('|')?;
            Some(PaneRow {
                pane_id,
                window_id,
                pane_dead,
                tracker_mark: tracker_mark.to_string(),
                incarnation_mark: incarnation_mark.to_string(),
            })
        })
        .collect()
}

/// Parse the window table's rows (pipe-separated, per the builder).
fn parse_windows(out: &str) -> Vec<WindowRow> {
    out.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut parts = line.split('|');
            WindowRow {
                window_id: parts.next().unwrap_or_default().to_string(),
                name: parts.next().unwrap_or_default().to_string(),
                pane_dead: parts.next() == Some("1"),
                pane_id: parts.next().unwrap_or_default().to_string(),
                ours: false,
            }
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    /// The fake executor: scripted answers per invocation key
    /// (re-scripting replaces), every argv recorded.
    struct Fake {
        answers: Vec<(String, (bool, String))>,
        pub calls: Vec<Vec<String>>,
    }
    impl Fake {
        fn new() -> Self {
            Fake {
                answers: Vec::new(),
                calls: Vec::new(),
            }
        }
        /// Script the answer for the invocation whose argv contains the
        /// key as a whole word (or contains it, for nested hook
        /// strings); scripting the same key again replaces.
        fn script(&mut self, key: &str, ok: bool, out: &str) {
            let answer = (ok, out.to_string());
            match self.answers.iter().position(|(k, _)| k == key) {
                Some(at) => self.answers[at] = (key.into(), answer),
                None => self.answers.push((key.into(), answer)),
            }
        }
        fn answer_for(&self, argv: &[String]) -> (bool, String) {
            for (key, answer) in &self.answers {
                let hit = if key.contains(' ') {
                    argv.iter().any(|word| word.contains(key.as_str()))
                } else {
                    argv.iter().any(|word| word == key)
                };
                if hit {
                    return answer.clone();
                }
            }
            (true, String::new())
        }
        fn calls_matching(&self, word: &str) -> Vec<Vec<String>> {
            self.calls
                .iter()
                .filter(|argv| argv.iter().any(|w| w == word))
                .cloned()
                .collect()
        }
        fn calls_containing(&self, needle: &str) -> Vec<String> {
            self.calls
                .iter()
                .filter(|argv| argv.iter().any(|w| w.contains(needle)))
                .map(|argv| argv.join(" "))
                .collect()
        }
        fn mark(&mut self) -> usize {
            self.calls.len()
        }
    }
    impl Exec for Fake {
        fn run(&mut self, argv: &[String]) -> Result<(bool, String), String> {
            self.calls.push(argv.to_vec());
            Ok(self.answer_for(argv))
        }
    }

    fn hc_ls(rows: &[(&str, &str, Option<&str>)]) -> String {
        let list: Vec<String> = rows
            .iter()
            .map(|(a, s, w)| {
                let status = match w {
                    Some(word) => format!(r#""status":{{"state":"{word}"}}"#),
                    None => r#""status":null"#.to_string(),
                };
                format!(r#"{{"address":"{a}","state":"{s}",{status}}}"#)
            })
            .collect();
        format!(
            r#"{{"hidden_finished":0,"terminals":[{}]}}"#,
            list.join(",")
        )
    }

    fn windows(wins: &[(&str, &str, bool)]) -> String {
        wins.iter()
            .map(|(id, name, dead)| format!("{id}|{name}|{}|%0", if *dead { "1" } else { "0" }))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The pane table with marks: (pane id, window id, dead, tracker
    /// mark, incarnation mark). The tracker mark carries pipes.
    fn panes(rows: &[(&str, &str, bool, &str, &str)]) -> String {
        rows.iter()
            .map(|(p, w, dead, t, i)| format!("{p}|{w}|{}|{t}|{i}", if *dead { "1" } else { "0" }))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn row(incarnation: &str, address: &str, tasks: &[&str]) -> AgentRow {
        AgentRow {
            incarnation: incarnation.into(),
            address: Some(address.into()),
            harness: Some("pi".into()),
            tasks: tasks.iter().map(|t| t.to_string()).collect(),
            workspace: Some("/ws".into()),
        }
    }

    const IDENTITY: &str = "machine-a|/repo";

    /// A companion whose fake knows one marked session (agents-repo,
    /// mark = the identity) and no panes.
    fn companion() -> Companion<Fake> {
        let mut fake = Fake::new();
        fake.script(
            "ls",
            true,
            &hc_ls(&[("t68", "live", Some("working")), ("t66", "finished", None)]),
        );
        fake.script("list-sessions", true, "agents-repo|machine-a|/repo");
        fake.script("list-windows", true, "");
        fake.script("list-panes", true, "");
        // the creation chains print their handles (the -P output): the
        // daemon reads the pane id from here to mark it
        fake.script("new-session", true, "@1|%1");
        fake.script("new-window", true, "@2|%2");
        // show-options is NOT scripted by default: with no answer the
        // ownership probe reads None, which means the name is free and
        // the first creation births it. Tests that stage an existing
        // session script it themselves.
        Companion::new(fake, "/opt/bin/hc", Server::Default, Vec::new())
    }

    /// The fake answers the ownership probe for a session that carries
    /// this daemon's identity mark.
    fn marked_session(c: &mut Companion<Fake>) {
        c.tmux.script("show-options", true, IDENTITY);
    }

    fn tracker(c: &mut Companion<Fake>) {
        c.accept(Feed::Tracker {
            base: "saccade".into(),
            identity: IDENTITY.into(),
        });
    }

    #[test]
    fn titles_carry_incarnation_tasks_and_the_word() {
        let mut agent = Agent {
            incarnation: "i-12".into(),
            address: "t68".into(),
            attached_address: None,
            tasks: vec![],
            word: None,
            end: None,
            pane: None,
            suppressed: false,
        };
        assert_eq!(Companion::<Fake>::title(&agent), "i-12");
        agent.tasks = vec!["t-20".into()];
        assert_eq!(Companion::<Fake>::title(&agent), "i-12 t-20");
        agent.tasks = vec!["t-15".into(), "t-12".into(), "t-9".into()];
        assert_eq!(
            Companion::<Fake>::title(&agent),
            "i-12 t-15+2",
            "latest claim first, +n for the rest"
        );
        agent.word = Some("working".into());
        assert_eq!(Companion::<Fake>::title(&agent), "i-12 t-15+2 working");
        agent.end = Some("killed".into());
        assert_eq!(Companion::<Fake>::title(&agent), "i-12 t-15+2 ended:killed");
    }

    #[test]
    fn enrollment_opens_marked_windows_for_live_agents_only() {
        let mut c = companion();
        // no session exists: the live agent births it
        c.tmux.script("list-sessions", true, "");
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![
            row("i-12", "t68", &["t-20"]),
            row("i-8", "t66", &["t-11"]),
        ]));
        let created = c.tmux.calls_matching("new-session");
        assert_eq!(created.len(), 1, "the live agent births the session");
        let text = created[0].join(" ");
        assert!(
            text.contains("i-12") && text.contains("t68"),
            "the pane runs hc attach t68: {text}"
        );
        assert!(text.contains("remain-on-exit on"), "the end stays visible");
        assert!(
            text.contains("automatic-rename off"),
            "the title is the companion's"
        );
        assert!(
            text.contains("agents-saccade"),
            "the session is named for the tracker"
        );
        assert_eq!(
            c.tmux.calls_containing("t66").len(),
            0,
            "a finished terminal at startup enrolls no window"
        );
        // and the pane is marked ours right away
        assert!(
            c.tmux
                .calls_containing("@saccade-tracker")
                .iter()
                .any(|t| t.contains("@saccade-incarnation") && t.contains("i-12")),
            "the pane marks ride creation"
        );
    }

    #[test]
    fn multiple_live_agents_enroll_in_one_pass() {
        let mut c = companion();
        // two live agents, no session yet
        c.tmux.script(
            "ls",
            true,
            &hc_ls(&[("t68", "live", None), ("t66", "live", None)]),
        );
        c.tmux.script("list-sessions", true, "");
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![
            row("i-12", "t68", &[]),
            row("i-8", "t66", &[]),
        ]));
        let sessions = c.tmux.calls_matching("new-session");
        let added = c.tmux.calls_matching("new-window");
        assert_eq!(sessions.len(), 1, "one session births");
        assert_eq!(
            added.len(),
            1,
            "the second agent adds a window in the same pass"
        );
    }

    #[test]
    fn an_agent_whose_terminal_is_dead_stays_unenrolled_until_its_address_moves_to_a_live_one() {
        let mut c = companion();
        tracker(&mut c);
        // registered at a terminal that has since finished: no window,
        // and none kept opening pass after pass
        c.tmux.script(
            "ls",
            true,
            &hc_ls(&[("t19", "live", Some("working")), ("t1b", "finished", None)]),
        );
        c.accept(Feed::Snapshot(vec![
            row("i-0", "t19", &[]),
            row("i-1", "t1b", &[]),
        ]));
        let added = c.tmux.calls_matching("new-window");
        assert!(
            added
                .iter()
                .all(|argv| !argv.iter().any(|w| w.contains("t1b"))),
            "a dead terminal enrolls nothing: {added:?}"
        );
        // the record's address moves to a live terminal: the next page
        // read enrolls there. The pane table now shows the window the
        // first pass created (marked, as creation marks it).
        c.tmux.script(
            "ls",
            true,
            &hc_ls(&[
                ("t19", "live", Some("working")),
                ("t1j", "live", Some("working")),
            ]),
        );
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%0", "@0", false, IDENTITY, "i-0")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@0", "i-0", false)]));
        c.accept(Feed::Snapshot(vec![
            row("i-0", "t19", &[]),
            row("i-1", "t1j", &[]),
        ]));
        let added = c.tmux.calls_matching("new-window");
        let moved: Vec<_> = added
            .iter()
            .filter(|argv| argv.iter().any(|w| w.contains("t1j")))
            .collect();
        assert_eq!(moved.len(), 1, "the moved agent enrolls exactly once");
        assert!(
            added
                .iter()
                .all(|argv| !argv.iter().any(|w| w.contains("t1b"))),
            "the dead address is never attached: {added:?}"
        );
    }

    #[test]
    fn the_word_renames_and_the_end_marks() {
        let mut c = companion();
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &["t-20"])]));
        // the pane exists, marked ours, alive
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", false, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@1", "i-12", false)]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &["t-20"])]));
        let renames = |c: &Companion<Fake>| -> Vec<String> {
            c.tmux
                .calls_matching("rename-window")
                .iter()
                .map(|argv| argv.last().unwrap().clone())
                .collect()
        };
        c.accept(Feed::Word {
            terminal: "t68".into(),
            word: Some("working".into()),
        });
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &["t-20"])]));
        assert!(
            renames(&c).contains(&"i-12 t-20 working".to_string()),
            "{:?}",
            renames(&c)
        );
        // the terminal ends: the pane dies with it, and the finished
        // feed names the reason the title carries
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", true, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("ls", true, &hc_ls(&[("t68", "finished", None)]));
        c.accept(Feed::Finished {
            terminal: "t68".into(),
            reason: "killed".into(),
        });
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &["t-20"])]));
        assert!(
            renames(&c).contains(&"i-12 t-20 ended:killed".to_string()),
            "{:?}",
            renames(&c)
        );
    }

    #[test]
    fn a_window_the_human_closed_is_not_recreated_this_run() {
        let mut c = companion();
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", false, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@1", "i-12", false)]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        let created_before =
            c.tmux.calls_matching("new-window").len() + c.tmux.calls_matching("new-session").len();
        // the human closes the window while the terminal stays live:
        // the pane vanishes from the table
        c.tmux.script("list-panes", true, "");
        c.tmux.script("list-windows", true, "");
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        let created_after =
            c.tmux.calls_matching("new-window").len() + c.tmux.calls_matching("new-session").len();
        assert_eq!(
            created_before, created_after,
            "a suppressed window stays closed within the run"
        );
    }

    #[test]
    fn restart_re_attaches_live_kept_terminals_only() {
        let mut c = companion();
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![
            row("i-12", "t68", &[]),
            row("i-8", "t66", &[]),
        ]));
        // both panes exist and are dead; t68 kept-live, t66 finished
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[
                ("%1", "@1", true, IDENTITY, "i-12"),
                ("%2", "@2", true, IDENTITY, "i-8"),
            ]),
        );
        c.tmux.script(
            "list-windows",
            true,
            &windows(&[("@1", "i-12", true), ("@2", "i-8", true)]),
        );
        c.accept(Feed::Snapshot(vec![
            row("i-12", "t68", &[]),
            row("i-8", "t66", &[]),
        ]));
        let mark = c.tmux.mark();
        c.accept(Feed::HerdrRestart);
        let respawns: Vec<String> = c.tmux.calls[mark..]
            .iter()
            .filter(|argv| argv.iter().any(|w| w == "respawn-pane"))
            .map(|argv| argv.join(" "))
            .collect();
        assert_eq!(
            respawns.len(),
            1,
            "only the kept, live terminal re-attaches: {respawns:?}"
        );
        assert!(respawns[0].contains("t68"), "{respawns:?}");
        assert!(
            c.tmux.calls[mark..]
                .iter()
                .all(|argv| !argv.iter().any(|w| w == "-k")),
            "the plain restart never kills"
        );
    }

    #[test]
    fn an_ended_agent_gets_no_new_panes_after_its_end() {
        let mut c = companion();
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", false, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@1", "i-12", false)]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        c.accept(Feed::Finished {
            terminal: "t68".into(),
            reason: "exited".into(),
        });
        let mark = c.tmux.mark();
        c.accept(Feed::Word {
            terminal: "t68".into(),
            word: Some("working".into()),
        });
        c.accept(Feed::HerdrRestart);
        let mutations: Vec<String> = c.tmux.calls[mark..]
            .iter()
            .filter(|argv| {
                argv.iter().any(|w| {
                    matches!(
                        w.as_str(),
                        "new-window"
                            | "new-session"
                            | "respawn-pane"
                            | "send-keys"
                            | "kill-pane"
                            | "kill-window"
                            | "kill-session"
                            | "select-window"
                            | "select-pane"
                            | "switch-client"
                    )
                })
            })
            .map(|argv| argv.join(" "))
            .collect();
        assert!(
            mutations.is_empty(),
            "ended agents get no new panes and no reactions after their end: {mutations:?}"
        );
    }

    #[test]
    fn a_late_registration_joins_the_running_session() {
        let mut c = companion();
        // the session exists and is ours
        marked_session(&mut c);
        tracker(&mut c);
        // the first snapshot opens i-12's window (its terminal is live)
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        let first = c.tmux.calls_matching("new-window");
        assert_eq!(first.len(), 1, "the live agent's window opens at once");
        assert!(first[0].join(" ").contains("t68"));
        // the pane exists now, marked ours
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", false, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@1", "i-12", false)]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        // i-8 registers, and its terminal is live
        c.tmux.script(
            "ls",
            true,
            &hc_ls(&[("t68", "live", None), ("t66", "live", None)]),
        );
        c.accept(Feed::Snapshot(vec![
            row("i-12", "t68", &[]),
            row("i-8", "t66", &[]),
        ]));
        let added = c.tmux.calls_matching("new-window");
        assert_eq!(added.len(), 2, "the second agent adds a window");
        assert!(added[1].join(" ").contains("t66"), "{added:?}");
        // the pane is marked ours right away
        assert!(
            c.tmux
                .calls_containing("@saccade-incarnation")
                .iter()
                .any(|t| t.contains("i-8")),
            "the pane marks ride the late creation"
        );
    }

    #[test]
    fn the_session_is_found_by_its_mark_whatever_its_name() {
        let mut c = companion();
        // the human renamed the session; the mark still names it ours
        c.tmux
            .script("list-sessions", true, "renamed-by-human|machine-a|/repo");
        marked_session(&mut c);
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        let created = c.tmux.calls_matching("new-session");
        assert!(
            created.is_empty(),
            "the marked session is adopted, not re-created"
        );
        let added = c.tmux.calls_matching("new-window");
        assert_eq!(added.len(), 1, "the agent joins the renamed session");
        assert!(!added[0].join(" ").contains("agents-repo"), "{added:?}");
    }

    #[test]
    fn a_foreign_session_of_the_right_name_is_never_touched() {
        let mut c = companion();
        // a session of the exact name, marked with ANOTHER identity
        c.tmux
            .script("list-sessions", true, "agents-saccade|machine-b|/elsewhere");
        c.tmux.script("show-options", true, "machine-b|/elsewhere");
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        // the daemon suffixes its own name instead of touching the
        // foreign one; the created session carries OUR mark (the fake
        // answers the foreign one for the base name's probe only, so
        // the walk takes the suffix)
        let created = c.tmux.calls_matching("new-session");
        assert_eq!(created.len(), 1);
        assert!(
            created[0].join(" ").contains("agents-saccade-"),
            "the collision suffix applies: {:?}",
            created[0].join(" ")
        );
        assert!(
            !c.tmux.calls.iter().any(|argv| argv
                .iter()
                .any(|w| w == "select-window" || w == "kill-session")),
            "the foreign session is never selected or killed"
        );
    }

    #[test]
    fn no_identity_means_no_tmux_calls_at_all() {
        let mut c = companion();
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        assert!(
            c.tmux
                .calls
                .iter()
                .all(|argv| argv.first().map(|f| f.as_str()) != Some("tmux")),
            "without an identity the daemon never touches tmux"
        );
    }

    #[test]
    fn an_initial_snapshot_fills_the_missed_word() {
        let mut c = companion();
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &["t-20"])]));
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", false, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@1", "i-12", false)]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &["t-20"])]));
        // the subscription's snapshot: the word the stream missed
        c.accept(Feed::TerminalSnapshot(vec![TerminalRow {
            address: "t68".into(),
            state: "live".into(),
            word: Some("working".into()),
        }]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &["t-20"])]));
        eprintln!(
            "SNAP-DEBUG journal={:?} renames={:?}",
            c.journal,
            c.tmux
                .calls_matching("rename-window")
                .iter()
                .map(|argv| argv.last().unwrap().clone())
                .collect::<Vec<_>>()
        );
        assert!(
            c.tmux
                .calls_matching("rename-window")
                .iter()
                .any(|argv| argv.last().unwrap() == "i-12 t-20 working"),
            "the snapshot's word rides the title"
        );
    }

    #[test]
    fn a_failed_ls_clears_the_table_and_holds_the_restart() {
        let mut c = companion();
        // the session exists and is ours
        marked_session(&mut c);
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        // the pane died, and the core just stopped: ls refuses
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", true, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@1", "i-12", true)]));
        c.tmux.script("ls", false, "error connecting");
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        c.accept(Feed::HerdrRestart);
        let mark = c.tmux.mark();
        assert!(
            c.tmux.calls[mark..]
                .iter()
                .all(|argv| !argv.iter().any(|w| w == "respawn-pane")),
            "an unreadable core holds the re-attach: the stale table is gone"
        );
        // the core returns; the next pass sees t68 live and re-attaches
        c.tmux.script("ls", true, &hc_ls(&[("t68", "live", None)]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        let respawns = c.tmux.calls_matching("respawn-pane");
        assert_eq!(respawns.len(), 1, "the held signal completes: {respawns:?}");
        assert!(respawns[0].join(" ").contains("t68"));
    }

    #[test]
    fn suppression_drops_when_the_terminal_ends() {
        let mut c = companion();
        // the session exists and is ours
        marked_session(&mut c);
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        // the human closed the window: suppressed
        c.tmux.script("list-panes", true, "");
        c.tmux.script("list-windows", true, "");
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        // the terminal ends: the suppression dies with it
        c.tmux
            .script("ls", true, &hc_ls(&[("t68", "finished", None)]));
        c.accept(Feed::Finished {
            terminal: "t68".into(),
            reason: "exited".into(),
        });
        // the kept terminal returns live after a restart
        c.tmux.script("ls", true, &hc_ls(&[("t68", "live", None)]));
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%9", "@9", true, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@9", "i-12", true)]));
        c.accept(Feed::HerdrRestart);
        eprintln!(
            "SUPPRESS-DEBUG journal={:?} agents={:?} fresh={} pending={}",
            c.journal,
            c.agents
                .values()
                .map(|a| (
                    a.incarnation.clone(),
                    a.end.clone(),
                    a.suppressed,
                    a.pane.is_some()
                ))
                .collect::<Vec<_>>(),
            c.terminals_fresh,
            c.restart_pending
        );
        let respawns = c.tmux.calls_matching("respawn-pane");
        assert_eq!(respawns.len(), 1, "re-derivation recreates the window");
        assert!(respawns[0].join(" ").contains("t68"));
    }

    #[test]
    fn the_retarget_restarts_the_viewer_with_the_new_address() {
        let mut c = companion();
        // the session exists and is ours
        marked_session(&mut c);
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", false, IDENTITY, "i-12")]),
        );
        c.tmux
            .script("list-windows", true, &windows(&[("@1", "i-12", false)]));
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        // the record reassigns the terminal: the address changes
        c.accept(Feed::Snapshot(vec![row("i-12", "t77", &[])]));
        let retargets: Vec<String> = c
            .tmux
            .calls_matching("respawn-pane")
            .iter()
            .filter(|argv| argv.iter().any(|w| w == "-k"))
            .map(|argv| argv.join(" "))
            .collect();
        assert_eq!(retargets.len(), 1, "{retargets:?}");
        assert!(
            retargets[0].contains("t77") && retargets[0].contains("-k"),
            "the retarget restarts the pane's own viewer against the record's address"
        );
    }

    #[test]
    fn a_detached_pane_on_a_live_terminal_without_a_restart_is_left_alone() {
        let mut c = companion();
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        c.tmux.script(
            "list-panes",
            true,
            &panes(&[("%1", "@1", true, IDENTITY, "i-12")]),
        );
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        let mark = c.tmux.mark();
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        assert!(
            c.tmux.calls[mark..]
                .iter()
                .all(|argv| !argv.iter().any(|w| w == "respawn-pane")),
            "a detached pane is the human's doing; only a restart re-attaches"
        );
    }

    #[test]
    fn the_window_table_parses_by_its_pipe_fields() {
        let rows = parse_windows("@1|i-2|0|%1\n@2|i-1 t-15 working|1|%2\n");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].window_id, "@1");
        assert_eq!(rows[0].name, "i-2");
        assert!(!rows[0].pane_dead);
        assert_eq!(rows[0].pane_id, "%1");
        assert_eq!(rows[1].name, "i-1 t-15 working");
        assert!(rows[1].pane_dead);
        assert_eq!(rows[1].pane_id, "%2");
    }

    #[test]
    fn the_pane_table_parses_its_marks() {
        let rows = parse_panes("%1|@1|0|machine-a|/repo|i-12\n%2|@2|1||\n");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].pane_id, "%1");
        assert_eq!(rows[0].tracker_mark, "machine-a|/repo");
        assert_eq!(rows[0].incarnation_mark, "i-12");
        assert!(!rows[0].pane_dead);
        assert_eq!(rows[1].tracker_mark, "", "an unmarked pane is nobody's");
        assert_eq!(rows[1].incarnation_mark, "");
    }

    #[test]
    fn the_session_name_follows_the_tracker() {
        let mut c = companion();
        c.tmux.script("list-sessions", true, "");
        tracker(&mut c);
        c.accept(Feed::Snapshot(vec![row("i-12", "t68", &[])]));
        let created = c.tmux.calls_matching("new-session");
        assert!(created[0].contains(&"agents-saccade".to_string()));
    }
}
