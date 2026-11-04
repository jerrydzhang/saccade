//! The herdr-core dialect: one compiled-in host implementing the
//! contract over the `hc` binary. Every verb is a bounded subprocess
//! ask whose answer is translated into the contract's host-agnostic
//! outcomes; hc's own vocabulary never leaves this module. Every
//! door holds the contract's interface, and this dialect is the one
//! implementation.

use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::host_contract::{
    CloseFact, CloseRefusal, HostAddress, HostContract, LaunchRefusal, Liveness, PlaceRefusal,
    SubmitRefusal, WaitWake,
};

/// The explicit override: set and nonempty, it is the host binary,
/// always. The devshell points it at the sandboxed pinned wrapper so
/// development never joins the machine server.
pub const OVERRIDE_VAR: &str = "SACCADE_HERDR";

/// The ambient binary's name: production joins the machine's own
/// server through it.
pub const BINARY_NAME: &str = "herdr-core";

/// The submission bound: a bare send resolves in well under a second
/// against the tested host; ten seconds bounds a hung core, never a
/// working one.
pub const DEFAULT_SUBMIT_BOUND: Duration = Duration::from_secs(10);

/// The poll interval of the bounded attempt's watch.
const POLL: Duration = Duration::from_millis(25);

/// The terminal-table read's bound: one listing, the launch door's
/// placement lookup and the clean contract's ordered step, never a
/// poll.
const TERMINALS_BOUND: Duration = Duration::from_secs(10);

/// The placement declaration's bound: one label write on one address,
/// the launch's own follow-on ask, never a poll.
const PLACE_BOUND: Duration = Duration::from_secs(10);

/// The single-ask bound shared by launch, live, and close: one verb,
/// one answer - the host names its terminal in the window or the ask
/// refuses honestly, never an unbounded wait on a hung core.
const ASK_BOUND: Duration = Duration::from_secs(10);

/// The standing placement one live terminal carries, stripped of the
/// dialect's label vocabulary: the ids alone. The tab is optional -
/// a space-only label is a legal adoption shape.
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalPlacement {
    pub space: String,
    pub tab: Option<String>,
}

/// One row of the host's terminal table, fully readable or the whole
/// answer refuses: the address submissions and closes target, the
/// launch cwd the clean contract matches exactly, and the host's own
/// state word.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTerminal {
    pub address: String,
    pub cwd: String,
    pub state: String,
}

/// The slack beyond a wait's own timeout: the host self-bounds the
/// block, the adapter enforces its own answer window on top.
const WAIT_SLACK: Duration = Duration::from_secs(2);

/// The test-context marker: the test harness sets it unconditionally,
/// and the resolution refuses the ambient arm under it - a test can
/// name its host with the override or the pin, but a missing fixture
/// pin fails the test, never the production server. Scrubbed from
/// every composed launch, so it never rides a session.
pub const TEST_CONTEXT_VAR: &str = "SACCADE_TEST_CONTEXT";

/// Binary resolution's failures: the arms all missed, or the test
/// context tried to reach the ambient arm.
#[derive(Debug)]
pub enum HerdrFail {
    NoHerdr,
    AmbientInTests,
    ReadFailed(String),
}

impl std::fmt::Display for HerdrFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HerdrFail::NoHerdr => write!(
                f,
                "no herdr-core host: the override, the ambient PATH, and the built-in pin all \
                 missed; export {OVERRIDE_VAR}=<herdr-core> to name one"
            ),
            HerdrFail::AmbientInTests => write!(
                f,
                "the ambient herdr-core arm is unreachable from a test context: the harness must \
                 pin {OVERRIDE_VAR} (a fixture fails, never the production server)"
            ),
            HerdrFail::ReadFailed(detail) => {
                write!(f, "the host's terminal table is unreadable: {detail}")
            }
        }
    }
}

/// The three-arm resolution: explicit override always; the ambient
/// binary in production; the baked pin last (SACCADE_HERDR_PIN, read
/// via option_env at compile time and filtered empty), which is
/// itself the sandboxed dev wrapper so a pin-resolved host never
/// joins the machine server.
pub fn herdr_binary() -> Result<PathBuf, HerdrFail> {
    let named = std::env::var_os(OVERRIDE_VAR).filter(|v| !v.is_empty());
    let pin = option_env!("SACCADE_HERDR_PIN").filter(|p| !p.is_empty());
    let test_context = std::env::var_os(TEST_CONTEXT_VAR).filter(|v| !v.is_empty());
    resolve_herdr(
        named.as_deref(),
        &path_search_dirs(),
        pin,
        test_context.is_some(),
    )
}

/// Resolution as a pure table over injected arms, so the precedence
/// tests never touch process env. An empty arm is unset, not a path.
/// The ambient arm is unreachable under `test_context`: the fail-safe
/// that keeps the suite out of the production server.
pub fn resolve_herdr(
    named: Option<&std::ffi::OsStr>,
    dirs: &[PathBuf],
    pin: Option<&str>,
    test_context: bool,
) -> Result<PathBuf, HerdrFail> {
    let named = named.filter(|p| !p.is_empty());
    let pin = pin.filter(|p| !p.is_empty());
    if let Some(path) = named {
        return Ok(PathBuf::from(path));
    }
    if test_context {
        return Err(HerdrFail::AmbientInTests);
    }
    if let Some(found) = find_ambient(dirs) {
        return Ok(found);
    }
    if let Some(path) = pin {
        return Ok(PathBuf::from(path));
    }
    Err(HerdrFail::NoHerdr)
}

fn path_search_dirs() -> Vec<PathBuf> {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default()
}

/// First executable `herdr-core` under these directories, canonical
/// absolute: a relative resolution would break against any other cwd.
fn find_ambient(dirs: &[PathBuf]) -> Option<PathBuf> {
    for dir in dirs {
        let candidate = dir.join(BINARY_NAME);
        let Ok(meta) = std::fs::metadata(&candidate) else {
            continue;
        };
        let executable = std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o111 != 0;
        if !(meta.is_file() && executable) {
            continue;
        }
        if let Ok(absolute) = std::fs::canonicalize(&candidate) {
            return Some(absolute);
        }
    }
    None
}

/// One dialect instance: the resolved binary and the submission
/// bound. Construct through [`HerdrCore::resolved`] at a door, or
/// [`HerdrCore::over`] a known binary in tests.
pub struct HerdrCore {
    binary: PathBuf,
    submit_bound: Duration,
    ask_bound: Duration,
}

impl HerdrCore {
    pub fn resolved() -> Result<Self, HerdrFail> {
        Ok(Self::over(herdr_binary()?))
    }

    /// The host's terminal table over one read. A row the adapter
    /// cannot fully read fails the whole answer - uncertainty is
    /// never absence.
    pub fn terminals(&self) -> Result<Vec<HostTerminal>, HerdrFail> {
        let args: Vec<OsString> = ["ls", "--json"].iter().map(OsString::from).collect();
        let out = self.ask_bounded(&args, TERMINALS_BOUND).map_err(|e| {
            HerdrFail::ReadFailed(match e {
                Bound::Spawn(e) => e.to_string(),
                Bound::Expired => "the terminal read outlived its bound".into(),
            })
        })?;
        if !out.ok {
            return Err(HerdrFail::ReadFailed(out.evidence()));
        }
        let value: serde_json::Value = serde_json::from_str(out.stdout.trim())
            .map_err(|e| HerdrFail::ReadFailed(format!("unreadable answer: {e}")))?;
        let rows = value
            .get("terminals")
            .and_then(|t| t.as_array())
            .ok_or_else(|| HerdrFail::ReadFailed("the answer names no terminal rows".into()))?;
        rows.iter()
            .enumerate()
            .map(|(index, row)| {
                let field = |name: &str| -> Option<String> {
                    row.get(name)
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                };
                match (field("address"), field("cwd"), field("state")) {
                    (Some(address), Some(cwd), Some(state)) => Ok(HostTerminal {
                        address,
                        cwd,
                        state,
                    }),
                    _ => Err(HerdrFail::ReadFailed(format!(
                        "terminal row {index} names an unreadable address, cwd, or state"
                    ))),
                }
            })
            .collect()
    }

    /// The standing placement one live terminal carries: the
    /// vocabulary stays inside the dialect - the caller sees only the
    /// ids. A terminal whose record cannot be asked or read, or whose
    /// state is not live, lends nothing; the placement lookup treats
    /// every miss as no standing placement.
    pub fn placement_of(&self, address: &HostAddress) -> Option<TerminalPlacement> {
        let args: Vec<OsString> = ["get", &address.0, "--json"]
            .iter()
            .map(OsString::from)
            .collect();
        let out = self.ask_bounded(&args, TERMINALS_BOUND).ok()?;
        if !out.ok {
            return None;
        }
        let value: serde_json::Value = serde_json::from_str(out.stdout.trim()).ok()?;
        if value.get("state").and_then(|s| s.as_str()) != Some("live") {
            return None;
        }
        let field = |name: &str| -> Option<String> {
            value
                .pointer("/labels")
                .and_then(|l| l.get(name))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        Some(TerminalPlacement {
            space: field("herdr.space")?,
            tab: field("herdr.tab"),
        })
    }

    pub fn over(binary: impl Into<PathBuf>) -> Self {
        HerdrCore {
            binary: binary.into(),
            submit_bound: DEFAULT_SUBMIT_BOUND,
            ask_bound: ASK_BOUND,
        }
    }

    pub fn with_submit_bound(mut self, bound: Duration) -> Self {
        self.submit_bound = bound;
        self
    }

    pub fn with_ask_bound(mut self, bound: Duration) -> Self {
        self.ask_bound = bound;
        self
    }

    /// The bounded ask: the child self-bounds where the verb carries
    /// a timeout, and the adapter enforces its own window on top. An
    /// unresolved child is killed, never awaited.
    fn ask_bounded(&self, args: &[OsString], bound: Duration) -> Result<Answer, Bound> {
        let mut child = std::process::Command::new(&self.binary)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(Bound::Spawn)?;
        let deadline = std::time::Instant::now() + bound;
        loop {
            let status = match child.try_wait() {
                Ok(status) => status,
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Bound::Spawn(e));
                }
            };
            match status {
                Some(status) => {
                    // the child already exited: drain the pipes and reap
                    let mut stdout = Vec::new();
                    let mut stderr = Vec::new();
                    if let Some(pipe) = child.stdout.as_mut() {
                        let _ = pipe.read_to_end(&mut stdout);
                    }
                    if let Some(pipe) = child.stderr.as_mut() {
                        let _ = pipe.read_to_end(&mut stderr);
                    }
                    let _ = child.wait();
                    return Ok(Answer {
                        ok: status.success(),
                        stdout: String::from_utf8_lossy(&stdout).into_owned(),
                        stderr: String::from_utf8_lossy(&stderr).into_owned(),
                    });
                }
                None if std::time::Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Bound::Expired);
                }
                None => std::thread::sleep(POLL),
            }
        }
    }
}

/// The bounded ask's failure: the binary could not be asked, or the
/// window closed on a still-running child.
enum Bound {
    Spawn(std::io::Error),
    Expired,
}

struct Answer {
    ok: bool,
    stdout: String,
    stderr: String,
}

impl Answer {
    /// The refusal evidence: stderr when the host wrote one, else the
    /// stdout body, trimmed.
    fn evidence(&self) -> String {
        let text = if self.stderr.trim().is_empty() {
            &self.stdout
        } else {
            &self.stderr
        };
        text.trim().to_string()
    }

    /// The host's error kind from whichever body carried it: the JSON
    /// error envelope is the dialect's routing key, never its prose.
    fn error_kind(&self) -> Option<String> {
        for body in [&self.stdout, &self.stderr] {
            let parsed = serde_json::from_str::<serde_json::Value>(body.trim()).ok();
            let kind = parsed
                .as_ref()
                .and_then(|v| v.pointer("/error/kind"))
                .and_then(|k| k.as_str());
            if let Some(kind) = kind {
                return Some(kind.to_string());
            }
        }
        None
    }
}

impl HostContract for HerdrCore {
    fn launch(&self, cwd: &Path, argv: &[String]) -> Result<HostAddress, LaunchRefusal> {
        if !cwd.is_dir() {
            return Err(LaunchRefusal::UnusableCwd(format!(
                "{} is not an existing directory; launch runs one session in an existing cwd",
                cwd.display()
            )));
        }
        let mut args: Vec<OsString> = [
            "run",
            "--bg",
            // every launch passes the no-parent form: the host's
            // die-with-parent default grants saccade no lifecycle power
            "--no-parent",
            // the host authors nothing into saccade sessions: until the
            // deployment's notes=false is verified, the per-launch flag
            // holds the law
            "--no-notify",
            "--cwd",
        ]
        .iter()
        .map(OsString::from)
        .collect();
        args.push(cwd.as_os_str().to_os_string());
        args.push(OsString::from("--json"));
        args.push(OsString::from("--"));
        args.extend(argv.iter().map(OsString::from));
        let out = self.ask_bounded(&args, self.ask_bound).map_err(|e| {
            LaunchRefusal::Unavailable(match e {
                Bound::Spawn(e) => e.to_string(),
                Bound::Expired => "the launch ask outlived its bound".into(),
            })
        })?;
        if !out.ok {
            return Err(LaunchRefusal::HostRefused(out.evidence()));
        }
        let address = serde_json::from_str::<serde_json::Value>(out.stdout.trim())
            .ok()
            .and_then(|v| {
                v.get("address")
                    .and_then(|a| a.as_str())
                    .map(str::to_string)
            })
            .filter(|a| !a.is_empty());
        address
            .map(HostAddress)
            .ok_or_else(|| LaunchRefusal::NoAddress(out.evidence()))
    }

    fn live(&self, address: &HostAddress) -> Liveness {
        let args: Vec<OsString> = ["get", &address.0, "--json"]
            .iter()
            .map(OsString::from)
            .collect();
        let out = match self.ask_bounded(&args, self.ask_bound) {
            Ok(out) => out,
            Err(Bound::Spawn(e)) => return Liveness::Unknown(e.to_string()),
            Err(Bound::Expired) => {
                return Liveness::Unknown("the liveness ask outlived its bound".into());
            }
        };
        if out.ok {
            let state = serde_json::from_str::<serde_json::Value>(out.stdout.trim())
                .ok()
                .and_then(|v| v.get("state").and_then(|s| s.as_str()).map(str::to_string));
            match state.as_deref() {
                Some("live") => Liveness::Live,
                Some("finished") => Liveness::Gone,
                _ => Liveness::Unknown(out.evidence()),
            }
        } else {
            match out.error_kind().as_deref() {
                Some("not_found") => Liveness::Gone,
                _ => Liveness::Unknown(out.evidence()),
            }
        }
    }

    fn submit(&self, address: &HostAddress, text: &str) -> Result<(), SubmitRefusal> {
        let args: Vec<OsString> = ["send", &address.0, text]
            .iter()
            .map(OsString::from)
            .collect();
        match self.ask_bounded(&args, self.submit_bound) {
            Err(Bound::Expired) => Err(SubmitRefusal::Unresolved),
            Err(Bound::Spawn(e)) => Err(SubmitRefusal::Unknown(e.to_string())),
            Ok(out) if out.ok => Ok(()),
            Ok(out) => match out.error_kind().as_deref() {
                Some("not_found") | Some("finished") => {
                    Err(SubmitRefusal::AddressNotLive(out.evidence()))
                }
                _ => Err(SubmitRefusal::HostRefused(out.evidence())),
            },
        }
    }

    fn wait(&self, address: &HostAddress, bound: Duration) -> WaitWake {
        let secs = bound.as_secs().max(1).to_string();
        let args: Vec<OsString> = ["wait", &address.0, "--timeout", &secs, "--json"]
            .iter()
            .map(OsString::from)
            .collect();
        let out = match self.ask_bounded(&args, bound + WAIT_SLACK) {
            Ok(out) => out,
            Err(_) => return WaitWake::Quiet,
        };
        if !out.ok {
            return WaitWake::Quiet;
        }
        let reason = serde_json::from_str::<serde_json::Value>(out.stdout.trim())
            .ok()
            .and_then(|v| v.get("reason").and_then(|r| r.as_str()).map(str::to_string));
        match reason.as_deref() {
            Some("message") => WaitWake::Message,
            _ => WaitWake::Quiet,
        }
    }

    /// One placement declaration: the label write on the address the
    /// launch just returned, both labels in one ask. The app's own
    /// adoption path reads them; an id it does not know becomes a new
    /// space or tab.
    fn place(&self, address: &HostAddress, space: &str, tab: &str) -> Result<(), PlaceRefusal> {
        let space_label = format!("herdr.space={space}");
        let tab_label = format!("herdr.tab={tab}");
        let args: Vec<OsString> = ["label", &address.0, "--json"]
            .iter()
            .map(OsString::from)
            .chain([OsString::from(space_label), OsString::from(tab_label)])
            .collect();
        match self.ask_bounded(&args, PLACE_BOUND) {
            Err(Bound::Expired) => Err(PlaceRefusal::Unknown(
                "the placement ask outlived its bound".into(),
            )),
            Err(Bound::Spawn(e)) => Err(PlaceRefusal::Unknown(e.to_string())),
            Ok(out) if out.ok => Ok(()),
            Ok(out) => match out.error_kind().as_deref() {
                Some("not_found") | Some("finished") => {
                    Err(PlaceRefusal::AddressNotLive(out.evidence()))
                }
                _ => Err(PlaceRefusal::HostRefused(out.evidence())),
            },
        }
    }

    fn close(&self, address: &HostAddress) -> Result<CloseFact, CloseRefusal> {
        let args: Vec<OsString> = ["kill", &address.0].iter().map(OsString::from).collect();
        let out = self.ask_bounded(&args, self.ask_bound).map_err(|e| {
            CloseRefusal::Unknown(match e {
                Bound::Spawn(e) => e.to_string(),
                Bound::Expired => "the close ask outlived its bound".into(),
            })
        })?;
        if out.ok {
            // the reply is the terminal's own object; its state fact is
            // the closure evidence - no re-list names it
            let state = serde_json::from_str::<serde_json::Value>(out.stdout.trim())
                .ok()
                .and_then(|v| v.get("state").and_then(|s| s.as_str()).map(str::to_string));
            return Ok(match state.as_deref() {
                Some("finished") => CloseFact::Finished,
                _ => CloseFact::Unconfirmed(out.stdout.trim().to_string()),
            });
        }
        match out.error_kind().as_deref() {
            Some("not_found") | Some("finished") => {
                Err(CloseRefusal::AddressNotLive(out.evidence()))
            }
            _ => Err(CloseRefusal::HostRefused(out.evidence())),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// The fake-hc harness: a contract-shaped script whose answers
    /// ride the scenario file beside it. Keys name the verb and either
    /// every call (`send.x=ok`) or a call number (`wait.2=message`);
    /// each call appends its exact argv to a per-verb log the tests
    /// assert against. The canned bodies are the tested host's own
    /// observed shapes (t-43 verification table, c-707).
    const FAKE_HC: &str = r#"#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
op=${1:-}
[ -z "$op" ] && exit 9
shift
n=1
if [ -f "$here/.calls-$op" ]; then n=$(($(cat "$here/.calls-$op") + 1)); fi
echo "$n" > "$here/.calls-$op"
printf '%s\n' "$@" >> "$here/argv-$op.log"
pick() { sed -n "s/^$op\.$1=//p" "$here/scenario" | head -1; }
kind=$(pick "$n")
[ -z "$kind" ] && kind=$(pick x)
case "$kind" in
  hang) sleep 30 ;;
  run-ok)
    printf '{"address":"t7","cwd":"/tmp","state":"live","turn":0}\n'; exit 0 ;;
  run-refused)
    printf 'the core refused' >&2; exit 5 ;;
  run-noaddress)
    printf 'not json at all\n'; exit 0 ;;
  get-live)
    printf '{"state":"live"}\n'; exit 0 ;;
  get-finished)
    printf '{"state":"finished","reason":"exited","exit_code":0}\n'; exit 0 ;;
  get-notfound)
    printf '{"error":{"kind":"not_found","message":"no terminal \\"t7\\""}}\n' >&2; exit 3 ;;
  get-error)
    printf '{"error":{"kind":"needs_target","message":"no target"}}\n' >&2; exit 7 ;;
  get-garbage)
    printf 'garbage\n'; exit 0 ;;
  send-ok)
    printf '{"started":true,"status":"working","turn":2}\n'; exit 0 ;;
  send-notfound)
    printf '{"error":{"kind":"not_found","message":"no terminal \\"t7\\""}}\n' >&2; exit 3 ;;
  send-finished)
    printf '{"error":{"kind":"finished","message":"terminal finished"}}\n' >&2; exit 11 ;;
  send-refused)
    printf 'send refused' >&2; exit 5 ;;
  wait-message)
    printf '{"reason":"message","status":"working","ended":false}\n'; exit 0 ;;
  wait-timeout)
    printf '{"reason":"timeout","status":"done"}\n'; exit 0 ;;
  wait-status)
    printf '{"reason":"status","status":"done"}\n'; exit 0 ;;
  wait-garbage)
    printf 'garbage\n'; exit 0 ;;
  kill-ok)
    printf '{"address":"t7","state":"finished","reason":"killed","signal":1}\n'; exit 0 ;;
  label-ok)
    printf '{}\n'; exit 0 ;;
  label-notfound)
    printf '{"error":{"kind":"not_found","message":"no terminal \\"t7\\""}}\n' >&2; exit 3 ;;
  label-finished)
    printf '{"error":{"kind":"finished","message":"terminal finished"}}\n' >&2; exit 11 ;;
  label-refused)
    printf 'label refused' >&2; exit 5 ;;
  get-labeled)
    printf '{"state":"live","labels":{"herdr.space":"spaStanding","herdr.tab":"tbStanding"}}\n'; exit 0 ;;
  get-labeled-notab)
    printf '{"state":"live","labels":{"herdr.space":"spaStanding"}}\n'; exit 0 ;;
  get-labeled-finished)
    printf '{"state":"finished","labels":{"herdr.space":"spaStanding","herdr.tab":"tbStanding"}}\n'; exit 0 ;;
  kill-unconfirmed)
    printf '{"address":"t7","state":"running"}\n'; exit 0 ;;
  kill-notfound)
    printf '{"error":{"kind":"not_found","message":"no terminal \\"t7\\""}}\n' >&2; exit 3 ;;
  kill-refused)
    printf 'kill refused' >&2; exit 5 ;;
  *)
    printf 'unscripted kind: %s\n' "$kind" >&2; exit 9 ;;
esac
"#;

    fn fake(name: &str, lines: &[&str]) -> (HerdrCore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("sac-herdr-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("hc");
        std::fs::write(&script, FAKE_HC).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut scenario = lines.join("\n");
        scenario.push('\n');
        std::fs::write(dir.join("scenario"), scenario).unwrap();
        warmup(&script);
        (HerdrCore::over(&script), dir)
    }

    /// The just-written script can answer its first exec with ETXTBSY
    /// while a concurrent writer's descriptor is still settling; a
    /// retrying warmup spawn burns that window so the pinned verbs
    /// spawn cleanly under the parallel suite.
    fn warmup(script: &Path) {
        for _ in 0..50 {
            match std::process::Command::new(script).arg("--warmup").output() {
                Ok(_) => return,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => panic!("the fake script never became spawnable: {e}"),
            }
        }
        panic!("the fake script never became spawnable: ETXTBSY persisted");
    }

    fn calls(dir: &Path, verb: &str) -> usize {
        std::fs::read_to_string(dir.join(format!(".calls-{verb}")))
            .unwrap_or_default()
            .trim()
            .parse()
            .unwrap_or(0)
    }

    fn argv_log(dir: &Path, verb: &str) -> Vec<String> {
        std::fs::read_to_string(dir.join(format!("argv-{verb}.log")))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    // -- resolution -----------------------------------------------------

    fn ambient_dir(tag: &str, executable: bool) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sac-herdr-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let binary = dir.join(BINARY_NAME);
        std::fs::write(&binary, b"#!/bin/sh\n").unwrap();
        if executable {
            std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    }

    #[test]
    fn the_override_always_wins() {
        let dirs = ambient_dir("ovr-beats", true);
        let pin = "/nix/store/pin/bin/herdr-core";
        let found = resolve_herdr(
            Some(std::ffi::OsStr::new("/override/hc")),
            std::slice::from_ref(&dirs),
            Some(pin),
            false,
        )
        .unwrap();
        assert_eq!(found, PathBuf::from("/override/hc"));
        // an empty override is unset, not a path
        let found = resolve_herdr(
            Some(std::ffi::OsStr::new("")),
            std::slice::from_ref(&dirs),
            Some(pin),
            false,
        );
        assert_eq!(
            found.unwrap(),
            std::fs::canonicalize(dirs.join(BINARY_NAME)).unwrap()
        );
        let _ = std::fs::remove_dir_all(&dirs);
    }

    #[test]
    fn the_ambient_binary_beats_the_pin_and_demands_an_executable_file() {
        let dirs = ambient_dir("ambient", true);
        let found = resolve_herdr(
            None,
            std::slice::from_ref(&dirs),
            Some("/nix/store/pin/bin/herdr-core"),
            false,
        )
        .unwrap();
        assert_eq!(
            found,
            std::fs::canonicalize(dirs.join(BINARY_NAME)).unwrap()
        );
        let _ = std::fs::remove_dir_all(&dirs);
        // a non-executable or missing ambient is skipped, never fatal
        let dirs = ambient_dir("ambient-dud", false);
        let found = resolve_herdr(
            None,
            std::slice::from_ref(&dirs),
            Some("/nix/store/pin/bin/herdr-core"),
            false,
        )
        .unwrap();
        assert_eq!(found, PathBuf::from("/nix/store/pin/bin/herdr-core"));
        let _ = std::fs::remove_dir_all(&dirs);
    }

    #[test]
    fn the_ambient_arm_is_unreachable_under_the_test_context() {
        // the fail-safe that keeps the suite out of the production
        // server: a present ambient arm is not used when the marker is
        // set, while the override still wins
        let dirs = ambient_dir("test-context", true);
        let refused = resolve_herdr(
            None,
            std::slice::from_ref(&dirs),
            Some("/nix/store/pin/bin/herdr-core"),
            true,
        );
        assert!(
            matches!(refused, Err(HerdrFail::AmbientInTests)),
            "{refused:?}"
        );
        assert!(
            refused
                .unwrap_err()
                .to_string()
                .contains("unreachable from a test context"),
        );
        let found = resolve_herdr(
            Some(std::ffi::OsStr::new("/override/hc")),
            std::slice::from_ref(&dirs),
            None,
            true,
        )
        .unwrap();
        assert_eq!(found, PathBuf::from("/override/hc"));
        let _ = std::fs::remove_dir_all(&dirs);
    }

    #[test]
    fn the_pin_fires_last_and_no_arm_at_all_refuses() {
        let found = resolve_herdr(None, &[], Some("/nix/store/pin/bin/herdr-core"), false).unwrap();
        assert_eq!(found, PathBuf::from("/nix/store/pin/bin/herdr-core"));
        // an empty pin is unset, not a path
        let failed = resolve_herdr(None, &[], Some(""), false);
        assert!(matches!(failed, Err(HerdrFail::NoHerdr)));
        let failed = resolve_herdr(None, &[], None, false);
        assert!(matches!(failed, Err(HerdrFail::NoHerdr)));
        assert!(failed.unwrap_err().to_string().contains(OVERRIDE_VAR));
    }

    // -- launch ---------------------------------------------------------

    #[test]
    fn launch_composes_bg_no_parent_cwd_json_and_the_separator() {
        let (host, dir) = fake("launch-compose", &["run.x=run-ok"]);
        let cwd = std::env::temp_dir().join(format!("sac-herdr-cwd-{}", std::process::id()));
        std::fs::create_dir_all(&cwd).unwrap();
        let address = host
            .launch(
                &cwd,
                &["pi".to_string(), "--session".to_string(), "abc".to_string()],
            )
            .unwrap();
        assert_eq!(address, HostAddress("t7".into()));
        // --no-parent on every launch: the parentage flag is not the
        // caller's choice
        assert_eq!(
            argv_log(&dir, "run"),
            vec![
                "--bg",
                "--no-parent",
                "--no-notify",
                "--cwd",
                &cwd.display().to_string(),
                "--json",
                "--",
                "pi",
                "--session",
                "abc",
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn launch_refuses_an_unusable_cwd_without_asking_the_host() {
        let (host, dir) = fake("launch-cwd", &["run.x=run-ok"]);
        let missing = dir.join("absent");
        let err = host.launch(&missing, &[]).unwrap_err();
        assert!(matches!(err, LaunchRefusal::UnusableCwd(_)), "{err}");
        assert_eq!(calls(&dir, "run"), 0);
        let file = dir.join("scenario");
        let err = host.launch(&file, &[]).unwrap_err();
        assert!(matches!(err, LaunchRefusal::UnusableCwd(_)), "{err}");
        assert_eq!(calls(&dir, "run"), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn launch_names_a_host_refusal_with_its_evidence() {
        let (host, dir) = fake("launch-refused", &["run.x=run-refused"]);
        let err = host.launch(&std::env::temp_dir(), &[]).unwrap_err();
        assert!(matches!(err, LaunchRefusal::HostRefused(_)), "{err}");
        assert!(err.to_string().contains("the core refused"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn launch_names_an_unreadable_success_answer() {
        let (host, dir) = fake("launch-noaddress", &["run.x=run-noaddress"]);
        let err = host.launch(&std::env::temp_dir(), &[]).unwrap_err();
        assert!(matches!(err, LaunchRefusal::NoAddress(_)), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn launch_names_an_unaskable_host() {
        let host = HerdrCore::over("/nonexistent/sac-herdr-missing");
        let err = host.launch(&std::env::temp_dir(), &[]).unwrap_err();
        assert!(matches!(err, LaunchRefusal::Unavailable(_)), "{err}");
    }

    // -- live -----------------------------------------------------------

    /// The live test's per-scenario check: one answer per row, judged
    /// against the contract's Liveness.
    type LiveCheck = fn(&HerdrCore, &HostAddress) -> bool;

    #[test]
    fn live_maps_the_host_answers_honestly() {
        // (scenario, expectation): live and both gone shapes against
        // the unknown shapes - unreadable, other refusals, unaskable
        let table: [(&str, LiveCheck); 5] = [
            ("get.x=get-live", |h, a| h.live(a) == Liveness::Live),
            ("get.x=get-finished", |h, a| h.live(a) == Liveness::Gone),
            ("get.x=get-notfound", |h, a| h.live(a) == Liveness::Gone),
            ("get.x=get-garbage", |h, a| {
                matches!(h.live(a), Liveness::Unknown(_))
            }),
            ("get.x=get-error", |h, a| {
                matches!(h.live(a), Liveness::Unknown(_))
            }),
        ];
        for (line, check) in &table {
            let (host, dir) = fake("live-map", &[line]);
            let address = HostAddress("t7".into());
            assert!(check(&host, &address), "{line}");
            let _ = std::fs::remove_dir_all(&dir);
        }
        let host = HerdrCore::over("/nonexistent/sac-herdr-missing");
        assert!(matches!(
            host.live(&HostAddress("t7".into())),
            Liveness::Unknown(_)
        ));
    }

    // -- submit ---------------------------------------------------------

    #[test]
    fn submit_accepts_the_hosts_ordered_submission_answer() {
        let (host, dir) = fake("submit-ok", &["send.x=send-ok"]);
        host.submit(&HostAddress("t7".into()), "hello there")
            .unwrap();
        // queue-style: bare send, the text rides argv verbatim
        assert_eq!(argv_log(&dir, "send"), vec!["t7", "hello there"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn submit_routes_absent_and_finished_addresses_to_not_live() {
        for line in ["send.x=send-notfound", "send.x=send-finished"] {
            let (host, dir) = fake("submit-notlive", &[line]);
            let err = host.submit(&HostAddress("t7".into()), "hello").unwrap_err();
            assert!(
                matches!(err, SubmitRefusal::AddressNotLive(_)),
                "{line}: {err}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn submit_names_other_host_refusals() {
        let (host, dir) = fake("submit-refused", &["send.x=send-refused"]);
        let err = host.submit(&HostAddress("t7".into()), "hello").unwrap_err();
        assert!(matches!(err, SubmitRefusal::HostRefused(_)), "{err}");
        assert!(err.to_string().contains("send refused"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn submit_names_an_unaskable_host() {
        let host = HerdrCore::over("/nonexistent/sac-herdr-missing");
        let err = host.submit(&HostAddress("t7".into()), "hello").unwrap_err();
        assert!(matches!(err, SubmitRefusal::Unknown(_)), "{err}");
    }

    #[test]
    fn submit_expires_the_bound_into_unresolved_and_kills_the_child() {
        let (host, dir) = fake("submit-hang", &["send.x=hang"]);
        let bound = Duration::from_millis(400);
        let started = std::time::Instant::now();
        let err = host
            .with_submit_bound(bound)
            .submit(&HostAddress("t7".into()), "hello")
            .unwrap_err();
        assert!(matches!(err, SubmitRefusal::Unresolved), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the bound is enforced, not awaited"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- wait -----------------------------------------------------------

    #[test]
    fn wait_dispatches_only_the_message_reason() {
        let table: [(&str, WaitWake); 5] = [
            ("wait.x=wait-message", WaitWake::Message),
            ("wait.x=wait-timeout", WaitWake::Quiet),
            ("wait.x=wait-status", WaitWake::Quiet),
            ("wait.x=wait-garbage", WaitWake::Quiet),
            ("wait.x=get-notfound", WaitWake::Quiet),
        ];
        for (line, expected) in &table {
            let (host, dir) = fake("wait-map", &[line]);
            let bound = Duration::from_secs(2);
            assert_eq!(
                host.wait(&HostAddress("t7".into()), bound),
                *expected,
                "{line}"
            );
            // the iteration bound rides --timeout in whole seconds
            assert_eq!(argv_log(&dir, "wait")[2], "2", "{line}");
            let _ = std::fs::remove_dir_all(&dir);
        }
        let host = HerdrCore::over("/nonexistent/sac-herdr-missing");
        assert_eq!(
            host.wait(&HostAddress("t7".into()), Duration::from_secs(1)),
            WaitWake::Quiet,
            "an unaskable host is quiet, never a message"
        );
    }

    // -- close ----------------------------------------------------------

    #[test]
    fn close_carries_the_replys_finished_fact() {
        let (host, dir) = fake("close-fact", &["kill.x=kill-ok"]);
        let fact = host.close(&HostAddress("t7".into())).unwrap();
        assert!(matches!(fact, CloseFact::Finished), "{fact:?}");
        let _ = std::fs::remove_dir_all(&dir);
        let (host, dir) = fake("close-unconfirmed", &["kill.x=kill-unconfirmed"]);
        let fact = host.close(&HostAddress("t7".into())).unwrap();
        match fact {
            CloseFact::Unconfirmed(evidence) => {
                assert!(evidence.contains("running"), "{evidence}")
            }
            other => panic!("the no-fact reply read as {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn close_accepts_the_ask_and_names_the_gone_and_refused_shapes() {
        let (host, dir) = fake("close-ok", &["kill.x=kill-ok"]);
        host.close(&HostAddress("t7".into())).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let (host, dir) = fake("close-gone", &["kill.x=kill-notfound"]);
        let err = host.close(&HostAddress("t7".into())).unwrap_err();
        assert!(matches!(err, CloseRefusal::AddressNotLive(_)), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
        let (host, dir) = fake("close-refused", &["kill.x=kill-refused"]);
        let err = host.close(&HostAddress("t7".into())).unwrap_err();
        assert!(matches!(err, CloseRefusal::HostRefused(_)), "{err}");
        assert!(err.to_string().contains("kill refused"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
        let host = HerdrCore::over("/nonexistent/sac-herdr-missing");
        let err = host.close(&HostAddress("t7".into())).unwrap_err();
        assert!(matches!(err, CloseRefusal::Unknown(_)), "{err}");
    }

    #[test]
    fn launch_live_and_close_expire_their_bound_into_honest_refusals() {
        // the three single asks expire their shared window into their honest refusals
        let (host, dir) = fake("ask-hang", &["run.x=hang", "get.x=hang", "kill.x=hang"]);
        let host = host.with_ask_bound(Duration::from_millis(400));
        let started = std::time::Instant::now();
        let err = host.launch(&std::env::temp_dir(), &[]).unwrap_err();
        assert!(matches!(err, LaunchRefusal::Unavailable(_)), "{err}");
        assert!(
            matches!(host.live(&HostAddress("t7".into())), Liveness::Unknown(_)),
            "a hung liveness ask is unknown, never gone"
        );
        let err = host.close(&HostAddress("t7".into())).unwrap_err();
        assert!(matches!(err, CloseRefusal::Unknown(_)), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the bound is enforced, not awaited"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn place_writes_both_labels_in_one_ask_and_accepts_ok() {
        let (host, dir) = fake("place-ok", &["label.x=label-ok"]);
        host.place(&HostAddress("t3".into()), "spaA", "tbB")
            .unwrap();
        assert_eq!(
            argv_log(&dir, "label"),
            vec![
                "t3".to_string(),
                "--json".to_string(),
                "herdr.space=spaA".to_string(),
                "herdr.tab=tbB".to_string(),
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn place_routes_absent_and_finished_addresses_to_not_live() {
        let (host, dir) = fake(
            "place-notlive",
            &["label.x=label-notfound", "label.x=label-finished"],
        );
        let err = host
            .place(&HostAddress("t3".into()), "spaA", "tbB")
            .unwrap_err();
        assert!(matches!(err, PlaceRefusal::AddressNotLive(_)), "{err}");
        let err = host
            .place(&HostAddress("t3".into()), "spaA", "tbB")
            .unwrap_err();
        assert!(matches!(err, PlaceRefusal::AddressNotLive(_)), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn place_names_other_host_refusals_and_unaskable_hosts() {
        let (host, dir) = fake("place-refused", &["label.x=label-refused"]);
        let err = host
            .place(&HostAddress("t3".into()), "spaA", "tbB")
            .unwrap_err();
        assert!(matches!(err, PlaceRefusal::HostRefused(_)), "{err}");
        assert!(err.to_string().contains("label refused"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
        let host = HerdrCore::over("/nonexistent/sac-herdr-missing");
        let err = host
            .place(&HostAddress("t3".into()), "spaA", "tbB")
            .unwrap_err();
        assert!(matches!(err, PlaceRefusal::Unknown(_)), "{err}");
    }

    #[test]
    fn placement_of_reads_the_standing_labels_of_a_live_record() {
        let (host, dir) = fake("placement-of", &["get.x=get-labeled"]);
        let placed = host.placement_of(&HostAddress("t7".into())).unwrap();
        assert_eq!(placed.space, "spaStanding");
        assert_eq!(placed.tab.as_deref(), Some("tbStanding"));
        let _ = std::fs::remove_dir_all(&dir);
        let (host, dir) = fake("placement-of-notab", &["get.x=get-labeled-notab"]);
        let placed = host.placement_of(&HostAddress("t7".into())).unwrap();
        assert_eq!(placed.space, "spaStanding");
        assert_eq!(placed.tab, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn placement_of_lends_nothing_from_unlive_unreadable_or_unlabeled_records() {
        let (host, dir) = fake("placement-of-dead", &["get.x=get-labeled-finished"]);
        assert_eq!(host.placement_of(&HostAddress("t7".into())), None);
        let _ = std::fs::remove_dir_all(&dir);
        let (host, dir) = fake("placement-of-absent", &["get.x=get-notfound"]);
        assert_eq!(host.placement_of(&HostAddress("t7".into())), None);
        let _ = std::fs::remove_dir_all(&dir);
        let (host, dir) = fake("placement-of-bare", &["get.x=get-live"]);
        assert_eq!(host.placement_of(&HostAddress("t7".into())), None);
        let _ = std::fs::remove_dir_all(&dir);
        let host = HerdrCore::over("/nonexistent/sac-herdr-missing");
        assert_eq!(host.placement_of(&HostAddress("t7".into())), None);
    }
}
