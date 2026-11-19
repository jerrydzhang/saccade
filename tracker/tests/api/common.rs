// ---- helpers two or more sections use ----

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use saccade::api::AppState;

/// The suite's own state parent and working directory, under the temp
/// tree: every spawned sac resolves state and repo discovery here, so
/// a test run writes nothing into the host's real state root - the
/// live tracker's state included.
pub(crate) fn suite_state() -> PathBuf {
    std::env::temp_dir().join(format!("sac-suite-state-{}", std::process::id()))
}

fn suite_cwd() -> PathBuf {
    let dir = suite_state().join("cwd");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub(crate) fn scratch_db(tag: &str) -> ScratchDir {
    let dir = std::env::temp_dir().join(format!("sac-api-{tag}-{}", std::process::id()));
    // a killed run may have left this pid's dir behind; every test
    // here builds its tracker from nothing
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("saccade.db");
    ScratchDir { dir, db }
}

/// A scratch tracker dir that removes itself at test end, panic or
/// not - one run of this suite used to leave hundreds behind. Reads
/// as the tracker's db path, as before.
pub(crate) struct ScratchDir {
    pub(crate) dir: PathBuf,
    pub(crate) db: PathBuf,
}

impl std::ops::Deref for ScratchDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.db
    }
}

impl AsRef<std::ffi::OsStr> for ScratchDir {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.db.as_os_str()
    }
}

impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.db
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A serving subprocess that dies with its test, panic or not.
pub(crate) struct ServerProc(std::process::Child);

impl std::ops::Deref for ServerProc {
    type Target = std::process::Child;

    fn deref(&self) -> &std::process::Child {
        &self.0
    }
}

impl std::ops::DerefMut for ServerProc {
    fn deref_mut(&mut self) -> &mut std::process::Child {
        &mut self.0
    }
}

impl Drop for ServerProc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(crate) async fn spawn_server(db_path: &std::path::Path) -> String {
    let state = AppState::open(db_path).expect("the scratch tracker opens");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = saccade::serve::router(state);
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

pub(crate) fn post_command(base_url: &str, envelope: &Value) -> (u16, String) {
    let mut r = ureq::post(&format!("{base_url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(envelope.clone())
        .expect("the loopback server answers");
    (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
}

pub(crate) fn json_of(body: &str) -> Value {
    serde_json::from_str(body).expect("the body is json")
}

pub(crate) fn envelope(actor: &str, tier: &str, command: Value) -> Value {
    json!({"context": {"actor": actor, "tier": tier}, "command": command})
}

/// Post one JSON body to a path: the typed envelope either way.
pub(crate) fn post_json(base_url: &str, path: &str, body: &Value) -> (u16, String) {
    let mut response = ureq::post(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(body.clone())
        .expect("the server answers");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
}

/// Get one JSON path.
pub(crate) fn get_json(base_url: &str, path: &str) -> (u16, String) {
    let mut response = ureq::get(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the server answers");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
}

pub(crate) fn attempts_lines(db: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(saccade::paths::attempts_at(db))
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).expect("one json object per line"))
        .collect()
}

/// A scratch git repo with two commits; returns its root and both tips.
pub(crate) fn scratch_repo(dir: &std::path::Path) -> (std::path::PathBuf, String, String) {
    let root = dir.join("repo");
    std::fs::create_dir_all(&root).unwrap();
    sh(&root, &["init", "-q", "--initial-branch=main"]);
    std::fs::write(root.join("readme"), "the project\n").unwrap();
    std::fs::write(root.join(".envrc"), "export SACCADE_ENVRC_LAYER=envrc\n").unwrap();
    sh(&root, &["add", "."]);
    sh(&root, &["commit", "-qm", "base"]);
    let first = sh(&root, &["rev-parse", "HEAD"]);
    std::fs::write(root.join("readme"), "the project, advanced\n").unwrap();
    sh(&root, &["commit", "-qam", "advanced"]);
    let head = sh(&root, &["rev-parse", "HEAD"]);
    (root, first, head)
}

/// The declaration file a spawned serve reads its port from: the one
/// source, written into the repo the serve names.
pub(crate) fn declare_port(repo: &std::path::Path, port: u16) {
    let name = repo.file_name().unwrap().to_string_lossy();
    std::fs::write(
        repo.join("workbench.toml"),
        format!("name = \"{name}\"\n\n[tracker]\nport = {port}\n"),
    )
    .unwrap();
}

/// A fresh repository for a spawned serve, carrying its declaration:
/// every spawn names its own repo, so the scratch repo builder's
/// commits never collide across calls.
pub(crate) fn serve_suite_repo() -> std::path::PathBuf {
    static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = suite_state().join(format!("serve-repo-{n}"));
    let (repo, _, _) = scratch_repo(&dir);
    declare_port(&repo, 0);
    repo
}

pub(crate) fn sh(dir: &std::path::Path, args: &[&str]) -> String {
    let out = saccade::boards::git_command()
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A fake herdr-core: hc run accepts the composed launch and answers
/// the pinned shape (hc get answers SACCADE_FAKE_HC_STATE, default
/// live); every call logs its argv for the door-composition asserts.
pub(crate) fn fake_hc(dir: &std::path::Path) -> std::path::PathBuf {
    let bin_dir = dir.join("hc-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let hc = bin_dir.join("hc");
    std::fs::write(
        &hc,
        r#"#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s
' "$@" >> "$here/argv-$1.log"
if [ "$1" = run ] && [ -n "${FAKE_HC_RUN_WAITS:-}" ]; then
  # the attach-before-register window: the host ask holds until the
  # fixture's trigger file lands, so the newborn can attach while the
  # door's host effect is still outside the writer lock
  n=0
  while [ ! -f "$here/attach-done" ] && [ "$n" -lt 200 ]; do
    sleep 0.1
    n=$((n+1))
  done
fi
if [ "$1" = run ]; then
  printf '{"address":"t1","cwd":"x","state":"live","turn":0}
'
  exit 0
fi
if [ "$1" = get ]; then
  if [ -f "$here/state" ]; then
    printf '{"state":"%s"}
' "$(cat "$here/state")"
  else
    printf '{"state":"%s"}
' "${SACCADE_FAKE_HC_STATE:-live}"
  fi
  exit 0
fi
if [ "$1" = send ] && [ "$2" = "t9" ]; then
  # the finished-address arm: the host's own not-live evidence
  printf '{"error":{"kind":"finished"}}
'
  exit 1
fi
if [ "$1" = send ] && [ -n "${SACCADE_FAKE_HC_SEND_FAILS:-}" ]; then
  printf '{"error":{"kind":"%s"}}
' "${SACCADE_FAKE_HC_SEND_FAILS}"
  exit 1
fi
if [ "$1" = attach ]; then
  if [ -n "${SACCADE_FAKE_HC_ATTACH_FAILS:-}" ]; then
    printf 'no attachable terminal here\n' >&2
    exit 4
  fi
  printf 'attached view\n'
  exit 0
fi
printf '{}
'
exit 0
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    hc
}

/// A spawn-capable fake herdr-core for the real-door fixtures: hc run
/// starts the composed session (the argv after --) in its --cwd with
/// stdio into FIXTURE_LOG, and answers the pinned address; hc get
/// answers SACCADE_FAKE_HC_STATE.
pub(crate) fn fake_hc_spawn(dir: &std::path::Path) -> std::path::PathBuf {
    let bin_dir = dir.join("hcs-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let hc = bin_dir.join("hc");
    std::fs::write(
        &hc,
        r#"#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s
' "$@" >> "$here/argv-$1.log"
if [ "$1" = run ] && [ -n "${FAKE_HC_RUN_FAILS:-}" ]; then
  printf 'the host refused the launch\n' >&2
  exit 1
fi
if [ "$1" = run ]; then
  cwd=""; prev=""; sep=0; n=0
  for a in "$@"; do
    n=$((n+1))
    if [ "$prev" = "--cwd" ]; then cwd="$a"; fi
    if [ "$a" = "--" ]; then sep=$n; fi
    prev="$a"
  done
  (
    cd "$cwd" || exit 1
    shift "$sep"
    exec env FIXTURE_LOG="${FIXTURE_LOG:-/dev/null}" sh -c 'exec "$@"' _ "$@"
  ) >> "${FIXTURE_LOG:-/dev/null}" 2>&1 &
  printf '{"address":"t9","cwd":"x","state":"live","turn":0}
'
  exit 0
fi
if [ "$1" = get ]; then
  labels=""
  if [ -f "$here/labels-$2.json" ]; then labels=",\"labels\":$(cat "$here/labels-$2.json")"; fi
  printf '{"state":"%s"%s}
' "${SACCADE_FAKE_HC_STATE:-live}" "$labels"
  exit 0
fi
if [ "$1" = send ] && [ "$2" = "t9" ]; then
  # the finished-address arm: the host's own not-live evidence
  printf '{"error":{"kind":"finished"}}
'
  exit 1
fi
if [ "$1" = ls ]; then
  # the terminal table from the state file's rows; a missing file
  # names no terminals
  printf '{"terminals":['
  if [ -f "${FAKE_HC_TERMINALS:-/dev/null-nothing}" ]; then
    paste -sd, "${FAKE_HC_TERMINALS}" 2>/dev/null
  fi
  printf ']}
'
  exit 0
fi
if [ "$1" = kill ] && [ -n "$2" ]; then
  if [ -f "${FAKE_HC_CLOSE_FAIL:-/dev/null-nothing}" ]; then
    rm -f "${FAKE_HC_CLOSE_FAIL}"
    printf '{"error":{"kind":"close_refused","message":"the host refused the close"}}
'
    exit 1
  fi
  row=$(grep -F "\"address\":\"$2\"" "${FAKE_HC_TERMINALS:-/dev/null-nothing}" 2>/dev/null | head -1)
  if [ -n "$row" ] && [ -f "${FAKE_HC_TERMINALS:-/dev/null-nothing}" ]; then
    grep -Fv "\"address\":\"$2\"" "${FAKE_HC_TERMINALS}" > "${FAKE_HC_TERMINALS}.tmp" || true
    mv "${FAKE_HC_TERMINALS}.tmp" "${FAKE_HC_TERMINALS}"
    printf '%s
' "$row" | sed 's/"state":"[a-z]*"/"state":"finished"/'
  else
    printf '{"address":"%s","cwd":"x","state":"finished"}
' "$2"
  fi
  exit 0
fi
printf '{}
'
exit 0
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    hc
}

/// The composed argv the fake hc logged for one verb (the op line
/// skipped).
pub(crate) fn hc_argv(bin_dir: &std::path::Path, verb: &str) -> Vec<String> {
    // an unreadable log surfaces in the assert message: silence would
    // fake an empty composition
    let body = std::fs::read_to_string(bin_dir.join(format!("argv-{verb}.log")))
        .unwrap_or_else(|e| format!("<unreadable: {e}>"));
    body.lines().skip(1).map(str::to_string).collect()
}

/// A fake `orca` with pinned wire shapes; shebang /bin/sh - the sandbox has no /usr/bin/env.
/// The lifecycle doors model the host's real read contract: visible
/// presence from the registration file or an existing directory,
/// selector_not_found otherwise, and a terminal state the close door
/// actually mutates. FAKE_ORCA_CLOSE_FAIL (one-shot flag file) makes
/// exactly one close ask fail. FAKE_ORCA_SHOW_GARBAGE makes the
/// visible-presence answer unreadable for as long as it stands.
/// FAKE_ORCA_STALE (with FAKE_ORCA_STALE_COUNT and FAKE_ORCA_STALE_ASKS)
/// models a stale catalog: paths listed there answer Listed for their
/// first FAKE_ORCA_STALE_ASKS asks each, then discover the removal.
pub(crate) fn fake_host(dir: &std::path::Path) -> std::path::PathBuf {
    let bin_dir = dir.join("host-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let orca = bin_dir.join("orca");
    std::fs::write(
        &orca,
        r#"#!/bin/sh
printf '%s\n' "$@" >> "$FAKE_ORCA_LOG"
selector=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--worktree" ]; then selector="$a"; fi
  prev="$a"
done
path="${selector#path:}"
if [ "$1" = worktree ]; then
  if [ "$2" = show ]; then
    # visible presence follows the live checkout, never the
    # registration file: a gone checkout is selector_not_found
    if [ -f "${FAKE_ORCA_SHOW_GARBAGE:-/dev/null-nothing}" ]; then
      printf 'not json at all\n'
      exit 0
    fi
    # a divergent host names paths in this file listed even when the
    # checkout is gone - the model for the Listed conflict
    if [ -n "$path" ] && [ -f "${FAKE_ORCA_LISTED:-/dev/null-nothing}" ] && grep -Fxq "$path" "$FAKE_ORCA_LISTED" 2>/dev/null; then
      printf '{"ok":true,"result":{"worktree":{"path":"%s"}}}\n' "$path"
      exit 0
    fi
    # a stale catalog: the path answers Listed for its first N asks,
    # counted per path, then the discovery catches up - the model for
    # the bounded visible-absence re-ask
    if [ -n "$path" ] && [ -f "${FAKE_ORCA_STALE:-/dev/null-nothing}" ] && grep -Fxq "$path" "${FAKE_ORCA_STALE}" 2>/dev/null; then
      printf '%s\n' "$path" >> "${FAKE_ORCA_STALE_COUNT}"
      n=$(grep -Fc "$path" "${FAKE_ORCA_STALE_COUNT}" 2>/dev/null)
      if [ "$n" -le "${FAKE_ORCA_STALE_ASKS:-1}" ]; then
        printf '{"ok":true,"result":{"worktree":{"path":"%s"}}}\n' "$path"
        exit 0
      fi
    fi
    if [ -n "$path" ] && [ -d "$path" ]; then
      printf '{"ok":true,"result":{"worktree":{"path":"%s"}}}\n' "$path"
    else
      printf '{"ok":false,"error":{"code":"selector_not_found","message":"selector_not_found"}}\n'
    fi
    exit 0
  fi
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = create ]; then
  n=$(($(wc -l < "$FAKE_ORCA_STATE" 2>/dev/null || echo 0) + 1))
  handle="term_actual_shape_$n"
  if [ -n "${FAKE_ORCA_CREATE_FAILS:-}" ]; then
    printf 'the host refused the create\n' >&2
    exit 1
  fi
  if [ -n "${FAKE_ORCA_NO_HANDLE:-}" ]; then
    printf '{"ok":true,"result":{"terminal":{}}}\n'
    exit 0
  fi
  printf '{"handle":"%s","worktreePath":"%s","orphaned":false}\n' "$handle" "$path" >> "$FAKE_ORCA_STATE"
  if [ -n "${FAKE_ORCA_SPAWN:-}" ]; then
    submitted=""
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--command" ]; then submitted="$a"; fi
      prev="$a"
    done
    child_guard=""
    if [ -f "${FAKE_ORCA_GUARD_FILE:-/dev/null-nothing}" ]; then
      child_guard=$(cat "$FAKE_ORCA_GUARD_FILE")
    fi
    printf 'submitted-command-started\n' >> "$FAKE_ORCA_CHILD_LOG"
    if [ -n "$child_guard" ]; then
      SACCADE_TEST_GUARD="$child_guard" sh -c "$submitted" >> "$FAKE_ORCA_CHILD_LOG" 2>&1 &
    else
      sh -c "$submitted" >> "$FAKE_ORCA_CHILD_LOG" 2>&1 &
    fi
  fi
  printf '{"ok":true,"result":{"terminal":{"handle":"%s"}}}\n' "$handle"
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = list ]; then
  if [ -n "$path" ] && ! grep -Fxq "$path" "$FAKE_ORCA_REGISTERED" 2>/dev/null && ! [ -d "$path" ]; then
    printf '{"ok":false,"error":{"code":"selector_not_found","message":"selector_not_found"}}\n'
    exit 0
  fi
  printf '{"ok":true,"result":{"terminals":['
  if [ -n "$path" ] && [ -f "$FAKE_ORCA_STATE" ]; then
    grep -F "\"worktreePath\":\"$path\"" "$FAKE_ORCA_STATE" | paste -sd,
  else
    paste -sd, "$FAKE_ORCA_STATE" 2>/dev/null
  fi
  printf ']}}\n'
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = close ]; then
  if [ -f "${FAKE_ORCA_CLOSE_FAIL:-/dev/null-nothing}" ]; then
    rm -f "${FAKE_ORCA_CLOSE_FAIL}"
    printf '{"ok":false,"error":{"code":"close_refused","message":"the host refused the close"}}\n'
    exit 0
  fi
  if [ -n "$path" ] && ! grep -Fxq "$path" "$FAKE_ORCA_REGISTERED" 2>/dev/null && ! [ -d "$path" ]; then
    printf '{"ok":false,"error":{"code":"selector_not_found","message":"selector_not_found"}}\n'
    exit 0
  fi
  if [ -n "$path" ] && [ -f "$FAKE_ORCA_STATE" ]; then
    grep -Fv "\"worktreePath\":\"$path\"" "$FAKE_ORCA_STATE" > "$FAKE_ORCA_STATE.tmp" || true
    mv "$FAKE_ORCA_STATE.tmp" "$FAKE_ORCA_STATE"
  fi
  printf '{"ok":true,"result":{"closed":true}}\n'
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = switch ]; then
  if [ "${FAKE_ORCA_NAVIGATE:-1}" = "1" ]; then
    printf '{"ok":true,"result":{"focus":{"navigated":true}}}\n'
  else
    printf '{"ok":true,"result":{"focus":{"navigated":false,"reason":"no renderer"}}}\n'
  fi
  exit 0
fi
exit 1
"#,
    )
    .unwrap();
    std::fs::set_permissions(&orca, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    orca
}

/// A stub harness: one dump of argv, cwd, and the saccade environment.
pub(crate) fn stub_pi(dir: &std::path::Path) -> std::path::PathBuf {
    let pi = dir.join("pi dump");
    std::fs::write(
        &pi,
        r#"#!/bin/sh
{
  printf 'argv: %s\n' "$*"
  printf 'pwd: %s\n' "$PWD"
  env | grep '^SACCADE_' | sort
} > "$PI_DUMP"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&pi, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    pi
}

/// One bounded sac subprocess in its own process group; a bound pass kills the whole group and reports the timeout.
pub(crate) fn bounded_sac<S: AsRef<str>>(
    bin: &str,
    args: &[S],
    envs: &[(&str, String)],
) -> (bool, String, String, bool) {
    let mut cmd = scrubbed(std::process::Command::new(bin));
    cmd.args(args.iter().map(AsRef::as_ref))
        .process_group(0)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    // the scrub removes rather than empties: these names carry clap env fallbacks
    for key in [
        "SACCADE_REPO",
        "SACCADE_DB",
        "SACCADE_SERVER",
        "SACCADE_ACTOR",
        "SACCADE_TASK",
        "SACCADE_INCARNATION_SEQ",
        "SACCADE_WORKSPACE",
        "SACCADE_LAUNCH_TOKEN",
    ] {
        cmd.env_remove(key);
    }
    // the isolation fail-safe (see scrubbed): defaults before the
    // fixture envs, so a fixture's own pin always wins
    cmd.env("SACCADE_TEST_CONTEXT", "1");
    cmd.env("SACCADE_HERDR", "/nonexistent-sac-test-hc");
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let mut child = cmd.spawn().expect("the bounded sac spawns");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(75);
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            _ => {
                timed_out = true;
                break None;
            }
        }
    };
    if timed_out {
        // the whole group goes: a plain parent kill would leak descendants holding shared file descriptions
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = std::io::Read::read_to_string(&mut pipe, &mut stdout);
    }
    if let Some(mut pipe) = child.stderr.take() {
        let _ = std::io::Read::read_to_string(&mut pipe, &mut stderr);
    }
    let _ = child.wait();
    (
        !timed_out && status.is_some_and(|s| s.success()),
        stdout,
        stderr,
        timed_out,
    )
}

/// A serving subprocess: publishes its address file, so the probe
/// routing is the real one, and carries the fake host on PATH for its
/// own child doors.
pub(crate) fn spawn_serving_subprocess(
    db: &Path,
    repo: &Path,
    bin: &str,
    host_dir: &Path,
    state_home: Option<&Path>,
    extra_envs: &[(&str, &str)],
) -> (String, ServerProc) {
    let mut cmd = scrubbed(std::process::Command::new(bin));
    declare_port(repo, 0);
    cmd.args([
        "--db",
        db.to_str().unwrap(),
        "--repo",
        repo.to_str().unwrap(),
        "serve",
        "--bind",
        "127.0.0.1",
    ])
    .env(
        "PATH",
        format!("{}:{}", host_dir.display(), std::env::var("PATH").unwrap()),
    )
    // the marker keeps the ambient arm unreachable
    .env("SACCADE_TEST_CONTEXT", "1");
    if let Some(dir) = state_home {
        // a genuinely cold state root: the guard dirs must come into
        // being from nothing, not from warmed probe residue
        cmd.env("XDG_STATE_HOME", dir);
    }
    for (key, value) in extra_envs {
        cmd.env(key, value);
    }
    let child = cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the serving subprocess spawns");
    let _pid = child.id();
    // port 0: the bound address publishes beside the db, so parallel subprocesses never collide
    let hint = saccade::paths::address_at(db);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let url = loop {
        match std::fs::read_to_string(&hint) {
            Ok(text) if !text.trim().is_empty() => break text.trim().to_string(),
            _ if std::time::Instant::now() >= deadline => {
                panic!("the serving subprocess never published its address")
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if ureq::get(&format!("{url}/api/v1/version"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .is_ok()
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the serving subprocess never came up"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    (url, ServerProc(child))
}

pub(crate) fn bind_next_ask(url: &str) -> std::thread::JoinHandle<()> {
    let (_, body) = get_json(url, "/api/v1/records");
    let rows = json_of(&body)["records"].as_array().unwrap().clone();
    let after = rows.last().map(|r| r["seq"].as_u64().unwrap());
    let url = url.to_string();
    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let (native_id, birth) = loop {
            let (status, body) = get_json(&url, "/api/v1/records");
            assert_eq!(status, 200, "{body}");
            let rows = json_of(&body)["records"].as_array().unwrap().clone();
            if let Some(row) = rows.iter().find(|row| {
                row["kind"] == "incarnation_registered"
                    && after.is_none_or(|cursor| row["seq"].as_u64().unwrap() > cursor)
                    && row["payload"]["saved_session"].is_null()
            }) {
                break (
                    row["payload"]["native_id"].as_str().unwrap().to_string(),
                    row["seq"].as_u64().unwrap(),
                );
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no launch birth: {body}"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        std::thread::sleep(std::time::Duration::from_millis(300));
        let (status, body) = get_json(&url, "/api/v1/records");
        assert_eq!(status, 200, "{body}");
        assert!(
            !json_of(&body)["records"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| { row["kind"] == "message_sent" && row["payload"]["to"] == birth }),
            "the initial question preceded native startup: {body}"
        );
        let (status, body) = post_json(
            &url,
            "/api/v1/native/attach",
            &json!({"native_id": native_id, "session_file": format!("/sessions/{native_id}.jsonl")}),
        );
        assert_eq!(status, 200, "{body}");
        assert_eq!(json_of(&body)["incarnation"], birth);
    })
}

/// Open a pty; the child's stdout and stderr attach to its slave.
pub(crate) fn open_pty() -> (i32, std::path::PathBuf) {
    let master = unsafe { libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY) };
    assert!(master >= 0, "posix_openpt");
    assert_eq!(unsafe { libc::grantpt(master) }, 0);
    assert_eq!(unsafe { libc::unlockpt(master) }, 0);
    let mut buf = [0u8; 64];
    assert_eq!(
        unsafe { libc::ptsname_r(master, buf.as_mut_ptr().cast(), buf.len()) },
        0
    );
    let name = std::ffi::CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    (master, std::path::PathBuf::from(name))
}

/// A pty slave as spawnable stdio; the master stays open for the terminal read.
pub(crate) fn pty_stdio(masters: &mut Vec<i32>) -> std::process::Stdio {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::FromRawFd;
    let (master, slave) = open_pty();
    // libc::open reads until NUL, and Rust bytes carry none: without the
    // CString the call walks past the allocation into neighboring heap bytes
    let path = std::ffi::CString::new(slave.as_os_str().as_bytes()).unwrap();
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
    assert!(fd >= 0, "open the pty slave");
    masters.push(master);
    std::process::Stdio::from(unsafe { std::fs::File::from_raw_fd(fd) })
}

/// Run sac with piped stdin and stdout/stderr each a pipe or a pty slave.
pub(crate) fn run_with_terminals(
    bin: &str,
    args: &[String],
    stdout_tty: bool,
    stderr_tty: bool,
) -> std::process::Output {
    use std::io::Write;
    let mut masters = Vec::new();
    let mut cmd = scrubbed(std::process::Command::new(bin));
    cmd.args(args)
        .env_remove("SACCADE_ACTOR")
        .env_remove("SACCADE_TASK")
        .env_remove("SACCADE_INCARNATION_SEQ")
        .env_remove("SACCADE_WORKSPACE")
        .stdin(std::process::Stdio::piped())
        .stdout(if stdout_tty {
            pty_stdio(&mut masters)
        } else {
            std::process::Stdio::piped()
        })
        .stderr(if stderr_tty {
            pty_stdio(&mut masters)
        } else {
            std::process::Stdio::piped()
        });
    let mut child = cmd.spawn().expect("spawn sac");
    child.stdin.take().unwrap().write_all(b"").unwrap();
    let out = child.wait_with_output().expect("sac answers");
    for master in masters {
        unsafe { libc::close(master) };
    }
    out
}

/// A real `sac serve` subprocess; the reply is the URL it published.
pub(crate) fn spawn_serve(
    bin: &str,
    db: &std::path::Path,
    extra: &[&str],
    envs: &[(&str, &str)],
) -> (ServerProc, String) {
    let repo = serve_suite_repo();
    let mut cmd = scrubbed(std::process::Command::new(bin));
    cmd.args(["serve", "--db"]).arg(db);
    // a caller naming its own repository wins: the suite repo only
    // backs the port's file for spawns that name none
    if !extra.contains(&"--repo") {
        cmd.arg("--repo").arg(&repo);
    }
    let mut child = ServerProc(
        cmd.args(extra)
            .envs(envs.iter().copied())
            .env("XDG_STATE_HOME", suite_state())
            .current_dir(suite_cwd())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn serve"),
    );
    let hint = saccade::paths::address_at(db);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match std::fs::read_to_string(&hint) {
            Ok(text) if !text.trim().is_empty() => {
                return (child, text.trim().to_string());
            }
            _ if child.try_wait().unwrap().is_some() => {
                panic!("serve exited without publishing {}", hint.display());
            }
            _ if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("serve never published {}", hint.display());
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}

/// A runner free of every inherited SACCADE_* value.
pub(crate) fn scrubbed(mut cmd: std::process::Command) -> std::process::Command {
    saccade::boards::scrub_inherited_repo_env(&mut cmd);
    for key in [
        "SACCADE_ACTOR",
        "SACCADE_DB",
        "SACCADE_SERVER",
        "SACCADE_TASK",
        "SACCADE_INCARNATION",
        "SACCADE_INCARNATION_SEQ",
        "SACCADE_WORKSPACE",
        "SACCADE_REPO",
    ] {
        cmd.env_remove(key);
    }
    // the isolation fail-safe: the marker makes the ambient herdr arm
    // refuse, and the pin names a fixture path every launch-driving
    // test overrides with its own fake - a missing fixture fails the
    // test, never the production server
    cmd.env("SACCADE_TEST_CONTEXT", "1");
    cmd.env("SACCADE_HERDR", "/nonexistent-sac-test-hc");
    // the state root and the working directory are the helper's own:
    // discovery from the real tree would find the live tracker
    cmd.env("XDG_STATE_HOME", suite_state());
    cmd.current_dir(suite_cwd());
    cmd
}
