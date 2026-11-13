#!/usr/bin/env python3
"""The single documented entry point for saccade-app development.

Lanes:
  check-fast   plain tool runs: the harness's own checks, the pyflakes
               pass, the app frontend's typecheck, the core client's
               contract tests, the frontend build. Ordinary build-tool
               invocations, uncontained on purpose; they claim nothing
               about service isolation.
  preflight    GATED. The minimal containment verification: static
               reads, then one bwrap sandbox running the containment
               canary with full namespace/path baselines, bounded
               timeout, owned cleanup. No test payload, browser, core,
               compositor or SSH activity in this lane.
  session      GATED. The enforced sandbox with a private headless
               Weston around the packaged app: suite runs the app's
               vitest native project (client/port, the payload
               suite_payload.py); up / up-t36 hold one sandbox with the
               tauri-driver -> WebKitWebDriver chain and the app for an
               explicit max lifetime, and inspect/input/observe/shot and
               the window verbs act and observe over a private run-root
               socket (never a host-visible port) - the t36 Python port
               lane's driver; down tears the chain down in owned order.
  serve        GATED, not implemented in this change.

(The old app's browser, native and remote lanes, the scripted session
checks and the session tracker sequence retired with the old app,
t-36 m-554.)

Everything that spawns sandbox processes goes through
isolation.PidfdSupervisor: one bwrap supervisor per run, kernel-verified
handle, verified reaping, no pattern kills, no numeric-PID fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
VIEWER = HERE.parent

FAST_STEPS = [
    ("harness", [sys.executable, "-B", "harness/harness_checks.py"]),
    ("payload-lint", [sys.executable, "-m", "pyflakes", "harness/entry.py", "harness/isolation.py", "harness/containment_canary.py", "harness/webdriver_client.py", "harness/webdriver_session.py", "harness/harness_checks.py", "harness/t36_results.py", "harness/t36_adapters.py", "harness/t36_driver.py", "harness/t36_observation.py", "harness/t36_prototype.py", "harness/t36_windows.py", "harness/t36_seeds.py", "harness/t36_checks.py", "harness/t36_differential.py", "harness/t36_feed_proxy.py", "harness/suite_payload.py"]),
    # the app's frontend typecheck (client/port; its suites run through
    # vitest: just port-test and just port-test-browser)
    ("typecheck", ["node_modules/.bin/tsc", "--noEmit", "-p", "port/tsconfig.json"]),
    # the read-only herdr-core client is a pure crate: its contract
    # tests (request shapes, error mapping, truncation refusal, remote
    # containment) need no GUI libraries, unlike the tauri shell itself
    ("bridge-contracts", ["cargo", "test", "-q", "--manifest-path", "src-tauri/core-client/Cargo.toml"]),
    ("build", ["node_modules/.bin/vite", "build", "-c", "port/vite.config.ts"]),
]

PREFLIGHT_TIMEOUT_SECONDS = 30
SESSION_UP_TIMEOUT_SECONDS = 90
SESSION_DOWN_TIMEOUT_SECONDS = 30
SESSION_DEFAULT_LIFETIME_SECONDS = 1800
# the suite mode's whole bound: weston, then the port suite's native
# project (one fresh app chain per test), teardown
SESSION_SUITE_TIMEOUT_SECONDS = 960
# The canary file is bound read-only into the sandbox at this path: the
# interpreter inside must be able to read it without any host HOME or
# worktree visibility.
CANARY_SANDBOX_PATH = "/canary/containment_canary.py"
SESSION_PAYLOAD_SANDBOX_PATH = "/lane/webdriver_session.py"
SESSION_CLIENT_SANDBOX_PATH = "/lane/webdriver_client.py"
# the suite mode: its payload beside the session modules, and the client
# tree (node_modules, the port, the scenarios, the seed module) read-only
SUITE_PAYLOAD_SANDBOX_PATH = "/lane/suite_payload.py"
SUITE_CLIENT_SANDBOX_PATH = "/suite/client"
# the control socket and ready marker live in the run root, which is
# bind-mounted read-write at the identical path: the host reaches the
# payload over a private Unix socket inside a 0700 tree, never a port
SESSION_SOCKET_NAME = "control.sock"
SESSION_READY_NAME = "session-ready.json"
tracker_db_arg = ""


def check_fast() -> int:
    failures = 0
    for name, argv in FAST_STEPS:
        started = time.monotonic()
        result = subprocess.run(argv, cwd=str(VIEWER))
        cost = time.monotonic() - started
        print(f"[check-fast] {name}: {cost:.1f}s exit={result.returncode}", flush=True)
        if result.returncode != 0:
            failures += 1
    return 1 if failures else 0


def caller_tracker_db(*candidates: str) -> tuple[str | None, str]:
    """Pure resolution of the caller-side tracker-db baseline. The raw
    candidate must be absolute as given: a relative candidate is a
    refusal, never silently resolved against the caller's cwd."""
    for candidate in candidates:
        if not candidate:
            continue
        if not candidate.startswith("/"):
            return None, f"tracker-db baseline must be an absolute path as given: {candidate!r}"
        return str(pathlib.Path(candidate).resolve()), ""
    return None, "no absolute caller tracker-db baseline; pass --tracker-db <absolute path> or set SACCADE_DB"


def preflight() -> int:
    """Minimal containment verification, runnable exactly as reviewed.

    1. Static reads: bwrap present (absolute pinned path), user-namespace
       sysctls positive. No unshare happens here.
    2. One sandbox: build_command with the canary as the sole payload,
       fresh environment for the whole chain, absolute tool paths,
       absolute caller baselines, the recorded run root and logs. Bounded
       by PREFLIGHT_TIMEOUT_SECONDS; cleanup is pidfd-only with verified
       reaping, on all paths.
    """
    sys.path.insert(0, str(HERE))
    import isolation

    isolation.gate_or_exit("preflight")

    found = shutil.which("bwrap")
    if found is None:
        print("containment refusal: bwrap is not in PATH (pinned devshell expected)", file=sys.stderr)
        return 1
    bwrap = str(pathlib.Path(found).resolve())
    python = str(pathlib.Path(sys.executable).resolve())  # absolute pinned tool path
    canary = str((HERE / "containment_canary.py").resolve())

    # The caller-baseline path must be absolute in its raw form: a
    # relative candidate is refused as given, never silently resolved
    # against the caller's cwd and then waved through.
    tracker_db, refusal = caller_tracker_db(tracker_db_arg, os.environ.get("SACCADE_DB") or "")
    if tracker_db is None:
        print(f"containment refusal: {refusal}", file=sys.stderr)
        return 1
    if not os.path.exists(tracker_db):
        print(f"containment refusal: baseline tracker db does not exist on the caller side: {tracker_db}", file=sys.stderr)
        return 1

    try:
        isolation.static_preflight(bwrap)
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return 1

    run_root = isolation.make_run_root("preflight")
    host_ns = isolation.host_ns_ids()
    env = isolation.env_allowlist(run_root)
    env["CONTAINMENT_HOST_NS_IDS"] = host_ns
    env["CONTAINMENT_HOST_HOME"] = str(pathlib.Path.home().resolve())
    env["CONTAINMENT_HOST_UID"] = str(os.getuid())
    env["CONTAINMENT_HOST_TRACKER_DB"] = tracker_db

    # Bind only the canary file, read-only, at a fixed sandbox path: the
    # payload needs no host HOME and no worktree visibility.
    ro_binds = [(canary, CANARY_SANDBOX_PATH)]
    cmd = isolation.build_command(bwrap, run_root, [python, CANARY_SANDBOX_PATH], ro_binds)

    print("[preflight] command:", " ".join(cmd), flush=True)
    print(f"[preflight] run root: {run_root}", flush=True)
    supervisor = isolation.PidfdSupervisor(cmd, env)
    log_path = run_root / "logs" / "canary.log"
    started = time.monotonic()
    try:
        supervisor.start()
        try:
            code = supervisor.wait(timeout=PREFLIGHT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"[preflight] timeout after {PREFLIGHT_TIMEOUT_SECONDS}s: stopping via pidfd", file=sys.stderr)
            supervisor.stop()
            log_path.write_text(supervisor.read_output())
            print("containment refusal: preflight sandbox exceeded the time bound", file=sys.stderr)
            return 1
        output = supervisor.read_output()
        log_path.write_text(output)
        cost = time.monotonic() - started
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        print(f"[preflight] canary exit={code} in {cost:.1f}s; log: {log_path}", flush=True)
        if code != 0:
            print("containment refusal: the canary rejected the sandbox", file=sys.stderr)
            return 1
        return 0
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return 1
    finally:
        # owned cleanup on every path; output already captured above
        code = supervisor.stop()
        if code != 0 and code is not None:
            print(f"[preflight] supervisor exit after cleanup: {code}", file=sys.stderr)


REMOTE_LOCATION = "b:/board/real.yaml"


SESSION_TARGET_ID = re.compile(r"^[A-Za-z0-9:_-]{1,120}$")
SESSION_SHOT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def verify_fontconfig_file(path: str) -> tuple[bool, str]:
    """The pinned generated fontconfig: must be a store path as given,
    readable, and reference at least one font directory inside the
    store. Home-relative profile dirs the generator also emits simply do
    not resolve in the sandbox; fontconfig skips absent dirs, so only a
    store reference is required."""
    if not path.startswith("/nix/store/"):
        return False, f"FONTCONFIG_FILE must be a store path as given: {path!r}"
    p = pathlib.Path(path)
    if not p.is_file():
        return False, f"FONTCONFIG_FILE does not exist or is not a file: {path}"
    try:
        text = p.read_text()
    except OSError as error:
        return False, f"FONTCONFIG_FILE unreadable: {error}"
    return fontconfig_content_ok(text)


def fontconfig_content_ok(text: str) -> tuple[bool, str]:
    dirs = re.findall(r"<dir>([^<]+)</dir>", text)
    if not any(d.startswith("/nix/store/") for d in dirs):
        return False, f"fontconfig references no store font directories: {dirs}"
    return True, ""


def session_state_path() -> pathlib.Path:
    """One active session record under the runtime root; its directory
    is 0700, so the control socket it names is private to the uid."""
    sys.path.insert(0, str(HERE))
    import isolation

    base = pathlib.Path(f"{isolation.RUNTIME_ROOT}-{os.getuid()}")
    return base / "session.json"


def write_session_record(record: dict) -> None:
    path = session_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_session_record() -> dict | None:
    try:
        value = json.loads(session_state_path().read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def clear_session_record(token: str) -> bool:
    """Remove the record only when it still names our own token: a slow
    teardown never deletes a newer session's identity."""
    record = load_session_record()
    if record is None or record.get("token") != token:
        return False
    try:
        session_state_path().unlink()
    except OSError:
        return False
    return True


def process_env_value(pid: int, name: str) -> str | None:
    try:
        data = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for entry in data:
        key, sep, value = entry.partition(b"=")
        if sep and key.decode(errors="replace") == name:
            return value.decode(errors="replace")
    return None


def open_verified_pidfd(pid: int, token: str) -> int | None:
    """Pin the process FIRST - a held pidfd cannot be recycled - then
    verify the token on the pinned process and confirm it is alive
    through the same fd. There is no window in which a recycled PID
    inherits another process's verdict."""
    try:
        fd = os.pidfd_open(pid)
    except OSError:
        return None
    try:
        if process_env_value(pid, "SAC_APP_SESSION_TOKEN") != token:
            os.close(fd)
            return None
        signal.pidfd_send_signal(fd, 0)  # alive probe through the pin
        return fd
    except (OSError, ProcessLookupError):
        os.close(fd)
        return None


def signal_verified_pid(pid: int, token: str, sig: signal.Signals) -> bool:
    """One signal through a token-verified pidfd: ownership verified on
    the pinned process, no numeric-PID fallback, no pattern kills."""
    fd = open_verified_pidfd(pid, token)
    if fd is None:
        return False
    try:
        signal.pidfd_send_signal(fd, sig)
        return True
    except (OSError, ProcessLookupError):
        return False
    finally:
        os.close(fd)


def wait_pid_gone(pid: int, timeout: float) -> bool:
    """Poll the held pidfd, never the numeric /proc entry: the fd pins
    the exact process we verified, and only the kernel's release of it
    (ESRCH) counts as gone."""
    try:
        fd = os.pidfd_open(pid)
    except OSError:
        return True
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                signal.pidfd_send_signal(fd, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        return False
    finally:
        os.close(fd)


def escalate_verified_pid(pid: object, token: str) -> None:
    """TERM, bounded grace, KILL through a verified pidfd; the same
    escalation ladder the supervisor uses, outside its class because the
    daemon and bwrap outlive the process that spawned them."""
    if not isinstance(pid, int) or pid <= 0:
        return
    if signal_verified_pid(pid, token, signal.SIGTERM):
        if not wait_pid_gone(pid, 5.0):
            signal_verified_pid(pid, token, signal.SIGKILL)
            wait_pid_gone(pid, 5.0)


def session_request(socket_path: str, payload: dict, timeout: float = 60.0) -> dict:
    """One JSON request to the session payload over the private
    run-root socket; one JSON response line comes back."""
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(socket_path)
        conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        with conn.makefile("rb") as reader:
            line = reader.readline(1 << 20)
        if not line:
            raise RuntimeError("session payload closed the connection without a response")
        response = json.loads(line.decode("utf-8"))
    finally:
        conn.close()
    if not isinstance(response, dict):
        raise RuntimeError("malformed session response")
    return response


def session_resolved(board_arg: str, tracker_db_candidate: str, require_serve_bin: bool = False) -> dict | None:
    """Every pinned path the session lane consumes, with the same
    refusals the native lane makes. Returns None after printing a
    refusal."""
    sys.path.insert(0, str(HERE))
    import isolation

    found = shutil.which("bwrap")
    if found is None:
        print("containment refusal: bwrap is not in PATH (pinned devshell expected)", file=sys.stderr)
        return None
    bwrap = str(pathlib.Path(found).resolve())
    python = str(pathlib.Path(sys.executable).resolve())
    canary = str((HERE / "containment_canary.py").resolve())
    session_py = str((HERE / "webdriver_session.py").resolve())
    client_py = str((HERE / "webdriver_client.py").resolve())

    tools: dict[str, str | None] = {}
    for name, binary in (
        ("weston", "weston"),
        ("shooter", "weston-screenshooter"),
        ("tauri_driver", "tauri-driver"),
        ("webkit_driver", "WebKitWebDriver"),
    ):
        located = shutil.which(binary)
        tools[name] = str(pathlib.Path(located).resolve()) if located else None
    missing = sorted(name for name, path in tools.items() if path is None)
    if missing:
        print(
            f"containment refusal: not in PATH (pinned devshell expected): {', '.join(missing)}",
            file=sys.stderr,
        )
        return None

    app = os.environ.get("SAC_APP_BIN", "")
    # a store binary (the authoritative, reproducible build) or the
    # debug iteration pin (r88): a cargo-built port binary patched to
    # a store rpath and pinned under the lane's own working root -
    # never an arbitrary host path. The debug pin exists so iteration
    # runs the environment we develop in; Nix still builds the
    # authoritative binary once, last, per chunk
    app_accepted = app.startswith("/nix/store/") or app.startswith("/var/tmp/saccade-app-1000/debug-bin/")
    if not app_accepted or not os.access(app, os.X_OK):
        print(
            "containment refusal: SAC_APP_BIN must be an executable pinned app binary",
            file=sys.stderr,
        )
        return None

    mesa_dir = os.environ.get("SAC_APP_MESA_EGL_VENDOR_DIR", "")
    if not mesa_dir.startswith("/nix/store/") or not os.path.isdir(mesa_dir):
        print(
            "containment refusal: SAC_APP_MESA_EGL_VENDOR_DIR must be a pinned mesa egl_vendor.d directory",
            file=sys.stderr,
        )
        return None

    if not board_arg.startswith("/") or not os.path.isfile(board_arg):
        print(
            "containment refusal: pass --board <absolute board.yaml> for the session lane",
            file=sys.stderr,
        )
        return None
    board = pathlib.Path(board_arg).resolve()

    tracker_db, refusal = caller_tracker_db(os.environ.get("SACCADE_DB") or "", tracker_db_candidate or "")
    if tracker_db is None:
        print(f"containment refusal: {refusal}", file=sys.stderr)
        return None
    if not os.path.exists(tracker_db):
        print(f"containment refusal: baseline tracker db does not exist on the caller side: {tracker_db}", file=sys.stderr)
        return None

    serve_bin = ""
    git_bin = ""
    if require_serve_bin:
        serve_bin = os.environ.get("SACCADE_SERVE_BIN", "")
        if not serve_bin.startswith("/") or not os.access(serve_bin, os.X_OK):
            print(
                "containment refusal: SACCADE_SERVE_BIN must be an executable absolute sac binary for the seed server",
                file=sys.stderr,
            )
            return None
        found_git = shutil.which("git")
        if found_git is None:
            print("containment refusal: git is not in PATH (pinned devshell expected)", file=sys.stderr)
            return None
        git_bin = str(pathlib.Path(found_git).resolve())
    fontconf = os.environ.get("FONTCONFIG_FILE", "")
    usable, reason = verify_fontconfig_file(fontconf)
    if not usable:
        print(f"containment refusal: {reason}", file=sys.stderr)
        return None

    try:
        isolation.static_preflight(bwrap)
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return None

    return {
        "bwrap": bwrap,
        "python": python,
        "canary": canary,
        "session_py": session_py,
        "client_py": client_py,
        "weston": tools["weston"],
        "shooter": tools["shooter"],
        "tauri_driver": tools["tauri_driver"],
        "webkit_driver": tools["webkit_driver"],
        "app": app,
        "board_host": str(board),
        "board_sandbox": f"/board/{board.name}",
        "board_dir": str(board.parent),
        "tracker_db": tracker_db,
        "fontconf": fontconf,
        "mesa_dir": mesa_dir,
        "serve_bin": serve_bin,
        "git_bin": git_bin,
    }


def session_env(resolved: dict, run_root: pathlib.Path, token: str, lifetime: float) -> dict[str, str]:
    """The fresh allowlist plus the lane's explicit pins. The GL and
    scale facts are the native lane's; they ride the chain env, and a
    GSubprocessLauncher child (the app) inherits its parent's
    environment, so software EGL reaches the packaged webview."""
    sys.path.insert(0, str(HERE))
    import isolation

    env = isolation.env_allowlist(run_root)
    env["CONTAINMENT_HOST_NS_IDS"] = isolation.host_ns_ids()
    env["CONTAINMENT_HOST_HOME"] = str(pathlib.Path.home().resolve())
    env["CONTAINMENT_HOST_UID"] = str(os.getuid())
    env["CONTAINMENT_HOST_TRACKER_DB"] = resolved["tracker_db"]
    env["WESTON_BIN"] = resolved["weston"]
    env["SCREENSHOOTER_BIN"] = resolved["shooter"]
    env["TAURI_DRIVER_BIN"] = resolved["tauri_driver"]
    env["WEBKIT_DRIVER_BIN"] = resolved["webkit_driver"]
    env["APP_BIN"] = resolved["app"]
    env["BOARD_FILE"] = resolved["board_sandbox"]
    # the host directory this lane's /board carries (see the native
    # lane's identical pin): check-body host fixtures ride beside the
    # session board
    env["SESSION_BOARD_DIR_HOST"] = str(resolved["board_dir"])
    env["FONTCONFIG_FILE"] = resolved["fontconf"]
    env["SHOTS_DIR"] = str(run_root / "shots")
    env["__EGL_VENDOR_LIBRARY_DIRS"] = resolved["mesa_dir"]
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["SAC_APP_WEBVIEW_CONSOLE"] = "1"
    env["GDK_SCALE"] = "1"
    env["GDK_DPI_SCALE"] = "1"
    env["SESSION_CONTROL_SOCKET"] = str(run_root / SESSION_SOCKET_NAME)
    env["SESSION_READY_FILE"] = str(run_root / SESSION_READY_NAME)
    env["SESSION_LIFETIME_SECONDS"] = str(int(lifetime))
    # the ownership token the host later verifies in /proc environ
    # before any pidfd signal; never SACCADE_-prefixed (the canary
    # refuses that namespace by design)
    env["SAC_APP_SESSION_TOKEN"] = token
    (run_root / "shots").mkdir(parents=True, exist_ok=True)
    return env


def session_binds(resolved: dict, mode: str = "plain") -> list[tuple[str, str]]:
    """Read-only binds: the canary, both lane modules, the pinned
    binaries (covered by the /nix/store bind - stated explicitly for
    review) and the board's parent directory. The screenshooter stays
    bound: shot captures compositor frames beside WebDriver shots.
    A run that seeds a tracker adds its own sac binary at /lane/sac; the
    t36 mode adds the seed and feed-proxy modules beside it."""
    binds = [
        (resolved["canary"], CANARY_SANDBOX_PATH),
        (resolved["session_py"], SESSION_PAYLOAD_SANDBOX_PATH),
        (resolved["client_py"], SESSION_CLIENT_SANDBOX_PATH),
        (resolved["app"], resolved["app"]),
        (resolved["weston"], resolved["weston"]),
        (resolved["shooter"], resolved["shooter"]),
        (resolved["tauri_driver"], resolved["tauri_driver"]),
        (resolved["webkit_driver"], resolved["webkit_driver"]),
        (resolved["board_dir"], "/board"),
    ]
    if resolved.get("serve_bin"):
        binds.append((resolved["serve_bin"], "/lane/sac"))
    if resolved.get("git_bin"):
        binds.append((resolved["git_bin"], "/lane/git"))
    if mode == "t36":
        binds.append((str((HERE / "t36_feed_proxy.py").resolve()), "/lane/t36_feed_proxy.py"))
        binds.append((str((HERE / "t36_seeds.py").resolve()), "/lane/t36_seeds.py"))
    return binds


def record_alive(record: dict | None) -> bool:
    """Liveness without signaling: the daemon and the bwrap supervisor
    both carry the session token in their environment."""
    if not isinstance(record, dict):
        return False
    token = record.get("token")
    if not isinstance(token, str):
        return False
    for key in ("daemon_pid", "supervisor_pid"):
        pid = record.get(key)
        if isinstance(pid, int) and pid > 0 and process_env_value(pid, "SAC_APP_SESSION_TOKEN") == token:
            return True
    return False


def daemon_hold_supervisor(
    supervisor: object,
    run_root: pathlib.Path,
    token: str,
    lifetime: float,
    board: str,
    ready_w: int,
) -> None:
    """The daemonized holder: starts the supervisor the launcher built,
    announces readiness once the payload's control socket answers, then
    polls for the payload's exit until the session bound (plus teardown
    grace). Every exit path stops the supervisor through its pidfd and
    clears the record (token-guarded) before exiting."""
    sys.path.insert(0, str(HERE))
    import isolation

    # the holder outlives the `up` invocation: its own chatter goes to
    # the run root's holder log, never the next command's terminal
    holder_log = open(run_root / "logs" / "holder.log", "ab", buffering=0)
    os.dup2(holder_log.fileno(), 1)
    os.dup2(holder_log.fileno(), 2)

    stopping = {"now": False}

    def request_stop(_signum: object, _frame: object) -> None:
        stopping["now"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def fail(message: str) -> None:
        try:
            os.write(ready_w, f"ERROR {message}\n".encode("utf-8"))
        except OSError:
            pass

    def stop_and_clear() -> None:
        supervisor.stop()
        clear_session_record(token)

    try:
        supervisor.start()
    except isolation.ContainmentRefused as error:
        fail(f"supervisor refused: {error}")
        os.close(ready_w)
        os._exit(1)

    socket_path = str(run_root / SESSION_SOCKET_NAME)
    expires_epoch = time.time() + lifetime
    record = {
        "root": str(run_root),
        "socket": socket_path,
        "ready": str(run_root / SESSION_READY_NAME),
        "daemon_pid": os.getpid(),
        "supervisor_pid": supervisor._proc.pid,
        "token": token,
        "board": board,
        "started": time.time(),
        "expires": expires_epoch,
    }
    deadline = time.monotonic() + SESSION_UP_TIMEOUT_SECONDS
    announced = False
    while time.monotonic() < deadline:
        if stopping["now"] or supervisor._proc.poll() is not None:
            break
        try:
            probe = session_request(socket_path, {"cmd": "status"}, timeout=2.0)
        except (OSError, RuntimeError, ValueError):
            probe = None
        if isinstance(probe, dict) and probe.get("ok"):
            write_session_record(record)
            try:
                os.write(
                    ready_w,
                    ("READY " + json.dumps({k: record[k] for k in ("root", "socket", "board", "expires")}) + "\n").encode("utf-8"),
                )
                os.close(ready_w)
            except OSError:
                pass
            ready_w = -1
            announced = True
            break
        time.sleep(0.4)
    if not announced:
        output = ""
        try:
            supervisor.stop()
            output = supervisor.read_output()
        except (OSError, ValueError):
            pass
        clear_session_record(token)
        fail(f"session did not come up: {output[-400:]}")
        if ready_w != -1:
            os.close(ready_w)
        os._exit(1)

    # hold for the payload: the bound is lifetime + teardown grace
    grace = 120.0
    end = time.monotonic() + lifetime + grace
    while time.monotonic() < end:
        if stopping["now"]:
            print("[session] holder asked to stop; tearing down", flush=True)
            break
        if supervisor._proc.poll() is not None:
            break
        time.sleep(0.25)
    stop_and_clear()
    os._exit(0)


def session_up(board_arg: str, lifetime: float, mode: str = "plain", reuse_root: str = "") -> int:
    """Gated. One sandbox, one daemonized supervisor holder, one control
    socket. Idempotent while the session is alive; a stale record is
    cleared, never adopted. The t36 mode adds the feed proxy and the
    in-sandbox seed server (SAC_APP_SERVE_BIN required). reuse_root
    relaunches into an existing run root: the run's own data and cache
    dirs persist, so the app's launch-restore surface is exercisable
    (S20's persisted-surface assertions)."""
    isolation_gate("webdriver")

    resolved = session_resolved(board_arg, tracker_db_arg, require_serve_bin=(mode == "t36"))
    if resolved is None:
        return 1
    if lifetime <= 0 or lifetime > 4 * 3600:
        print("containment refusal: --max-lifetime must be within (0, 4h]", file=sys.stderr)
        return 1

    existing = load_session_record()
    if existing is not None:
        if record_alive(existing):
            print(
                json.dumps(
                    {
                        "session": "already-running",
                        "root": existing.get("root"),
                        "socket": existing.get("socket"),
                        "board": existing.get("board"),
                        "expires": existing.get("expires"),
                    },
                    indent=2,
                )
            )
            return 0
        session_state_path().unlink(missing_ok=True)

    import select
    import uuid

    sys.path.insert(0, str(HERE))
    import isolation

    token = uuid.uuid4().hex
    if reuse_root:
        reuse = pathlib.Path(reuse_root)
        if not reuse.is_dir():
            print(f"containment refusal: reuse_root does not exist: {reuse_root}", file=sys.stderr)
            return 1
        run_root = reuse
        (run_root / "logs").mkdir(exist_ok=True)
        (run_root / "shots").mkdir(exist_ok=True)
    else:
        run_root = isolation.make_run_root("session")
    env = session_env(resolved, run_root, token, lifetime)
    if mode == "t36":
        env["SESSION_MODE"] = "t36"
        # the payload's own seed server rides /lane/sac; the prelude
        # refuses without it
        env["SAC_APP_SERVE_BIN"] = "/lane/sac"
    cmd = isolation.build_command(
        resolved["bwrap"],
        run_root,
        [resolved["python"], SESSION_PAYLOAD_SANDBOX_PATH],
        session_binds(resolved, mode),
    )
    print("[session] gate command:", " ".join(cmd), flush=True)
    supervisor = isolation.PidfdSupervisor(cmd, env)

    ready_r, ready_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.setsid()
        os.close(ready_r)
        try:
            daemon_hold_supervisor(supervisor, run_root, token, lifetime, resolved["board_host"], ready_w)
        finally:
            os._exit(1)
    os.close(ready_w)

    deadline = time.monotonic() + SESSION_UP_TIMEOUT_SECONDS
    buffer = b""
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        readable, _w, _x = select.select([ready_r], [], [], min(remaining, 1.0))
        if not readable:
            continue
        chunk = os.read(ready_r, 4096)
        if not chunk:
            break
        buffer += chunk
        if b"\n" in buffer:
            break
    os.close(ready_r)
    line = buffer.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    if line.startswith("READY "):
        info = json.loads(line[len("READY "):])
        print(json.dumps({"session": "up", **info, "lifetime": int(lifetime)}, indent=2))
        return 0
    detail = line[len("ERROR "):] if line.startswith("ERROR ") else line or "holder exited without a verdict"
    print(f"[session] up failed: {detail}", file=sys.stderr)
    # reap the finished holder: it exited after writing its verdict
    try:
        os.waitpid(pid, 0)
    except (ChildProcessError, OSError):
        pass
    return 1


def session_suite(board_arg: str, tracker_db_candidate: str) -> int:
    """Gated. One bounded sandbox run of the port suite's native project
    (client/port/tests/scenarios/README.md, native): the session lane's
    resolution, env and binds unchanged, plus the suite payload, the
    client tree read-only and the pinned node. The payload runs the
    canary, starts the private Weston, then vitest; each test owns its
    tauri-driver chain. The supervisor stop is the backstop."""
    isolation_gate("webdriver")
    sys.path.insert(0, str(HERE))
    import isolation

    resolved = session_resolved(board_arg, tracker_db_candidate, require_serve_bin=True)
    if resolved is None:
        return 1
    found_node = shutil.which("node")
    if found_node is None:
        print("containment refusal: node is not in PATH (pinned devshell expected)", file=sys.stderr)
        return 1
    node = str(pathlib.Path(found_node).resolve())
    if not node.startswith("/nix/store/"):
        print(f"containment refusal: node must be a pinned store binary: {node}", file=sys.stderr)
        return 1
    if not (VIEWER / "node_modules" / "vitest").is_dir():
        print("containment refusal: client/node_modules lacks vitest; run npm ci in client", file=sys.stderr)
        return 1

    import uuid

    run_root = isolation.make_run_root("session-suite")
    token = uuid.uuid4().hex
    env = session_env(resolved, run_root, token, SESSION_DEFAULT_LIFETIME_SECONDS)
    env["SESSION_MODE"] = "suite"
    env["SUITE_NODE_BIN"] = node
    env["SUITE_CLIENT_DIR"] = SUITE_CLIENT_SANDBOX_PATH
    binds = session_binds(resolved) + [
        (str((HERE / "suite_payload.py").resolve()), SUITE_PAYLOAD_SANDBOX_PATH),
        (str(VIEWER.resolve()), SUITE_CLIENT_SANDBOX_PATH),
    ]
    cmd = isolation.build_command(resolved["bwrap"], run_root, [resolved["python"], SUITE_PAYLOAD_SANDBOX_PATH], binds)
    print("[suite] gate command:", " ".join(cmd), flush=True)
    print(f"[suite] run root: {run_root}", flush=True)
    supervisor = isolation.PidfdSupervisor(cmd, env)
    started = time.monotonic()
    try:
        supervisor.start()
        try:
            code = supervisor.wait(timeout=SESSION_SUITE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print("[suite] the suite run exceeded its bound; stopping via pidfd", file=sys.stderr)
            supervisor.stop()
            code = None
        output = supervisor.read_output()
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        print(f"[suite] done in {time.monotonic() - started:.1f}s; payload exit={code}", flush=True)
        return 0 if code == 0 else 1
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return 1
    finally:
        code = supervisor.stop()
        if code != 0 and code is not None:
            print(f"[suite] supervisor exit after cleanup: {code}", file=sys.stderr)


def session_down() -> int:
    """Gated. Graceful first: the down request makes the payload tear
    its chain down in owned order and exit, the sandbox empties, the
    holder exits and clears the record. Escalation, only if needed, is
    the same TERM -> grace -> KILL ladder through token-verified pidfds
    (daemon first - its stop path owns the supervisor - then bwrap)."""
    isolation_gate("webdriver")

    record = load_session_record()
    if record is None:
        print(json.dumps({"stopped": True, "note": "no session record"}))
        return 0
    token = record.get("token")
    if not isinstance(token, str):
        print("containment refusal: session record has no ownership token", file=sys.stderr)
        return 1
    socket_path = record.get("socket")
    if isinstance(socket_path, str) and pathlib.Path(socket_path).exists():
        try:
            session_request(socket_path, {"cmd": "down"}, timeout=20.0)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"[session] control socket did not take down ({error}); escalating", file=sys.stderr)
    deadline = time.monotonic() + SESSION_DOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not record_alive(record):
            break
        time.sleep(0.2)
    if record_alive(record):
        escalate_verified_pid(record.get("daemon_pid"), token)
        escalate_verified_pid(record.get("supervisor_pid"), token)
    stopped = not record_alive(record)
    clear_session_record(token)
    print(json.dumps({"stopped": stopped, "root": record.get("root"), "files_retained": record.get("root")}))
    return 0 if stopped else 1


def session_live_socket() -> tuple[dict, str] | None:
    record = load_session_record()
    if record is None:
        print("[session] no session record; run session up first", file=sys.stderr)
        return None
    socket_path = record.get("socket")
    if not isinstance(socket_path, str) or not pathlib.Path(socket_path).exists():
        print("[session] session control socket is gone; run session down then up", file=sys.stderr)
        return None
    return record, socket_path


def session_status() -> int:
    isolation_gate("webdriver")
    record = load_session_record()
    if record is None:
        print(json.dumps({"running": False}))
        return 0
    socket_path = record.get("socket")
    if isinstance(socket_path, str):
        try:
            response = session_request(socket_path, {"cmd": "status"}, timeout=5.0)
            print(json.dumps({"running": True, **response, "root": record.get("root")}, indent=2))
            return 0 if response.get("ok") else 1
        except (OSError, RuntimeError, ValueError):
            pass
    print(
        json.dumps(
            {
                "running": record_alive(record),
                "note": "control socket unreachable; record facts only",
                "root": record.get("root"),
                "expires": record.get("expires"),
            },
            indent=2,
        )
    )
    return 0


def parse_scalar(value: str, field: str) -> float:
    try:
        number = float(value)
    except ValueError:
        print(f"[session] {field} must be a number: {value!r}", file=sys.stderr)
        raise SystemExit(2) from None
    return number


def parse_verb_fields(args: list[str], bare_kinds: set[str]) -> tuple[dict, list[str]]:
    fields: dict[str, str] = {}
    bare: list[str] = []
    for token in args:
        key, sep, value = token.partition("=")
        if sep and re.fullmatch(r"[a-z]+", key):
            fields[key] = value
        else:
            bare.append(token)
    unknown = set(fields) - bare_kinds
    if unknown:
        print(f"[session] unknown fields: {', '.join(sorted(unknown))}", file=sys.stderr)
        raise SystemExit(2)
    return fields, bare


def build_input_request(args: list[str]) -> dict:
    if not args:
        print("[session] input needs a kind: click/type/key/wheel/drag/coords/rightclick", file=sys.stderr)
        raise SystemExit(2)
    kind = args[0]
    if kind not in ("click", "type", "key", "wheel", "drag", "coords", "rightclick"):
        print(f"[session] unknown input kind: {kind!r}", file=sys.stderr)
        raise SystemExit(2)
    fields, bare = parse_verb_fields(args[1:], {"target", "text", "dx", "dy", "x", "y", "steps"})
    request: dict = {"cmd": "input", "kind": kind}
    if kind == "key":
        keys = bare if bare else ([fields["text"]] if "text" in fields else [])
        if not keys:
            print("[session] key input needs key names", file=sys.stderr)
            raise SystemExit(2)
        request["keys"] = keys
        return request
    if kind == "coords":
        x = fields.get("x", bare[0] if bare else "")
        y = fields.get("y", bare[1] if len(bare) > 1 else "")
        if x == "" or y == "":
            print("[session] coords needs x and y", file=sys.stderr)
            raise SystemExit(2)
        request["x"] = parse_scalar(x, "x")
        request["y"] = parse_scalar(y, "y")
        for name in ("dx", "dy"):
            if name in fields:
                request[name] = parse_scalar(fields[name], name)
        return request
    target = fields.get("target", bare[0] if bare and kind != "type" else "")
    if kind == "type":
        target = fields.get("target", "")
        text = fields.get("text", " ".join(bare))
        if not target or not text:
            print("[session] type needs target=<id> and text", file=sys.stderr)
            raise SystemExit(2)
        request["text"] = text
    elif not target:
        print(f"[session] {kind} needs target=<id>", file=sys.stderr)
        raise SystemExit(2)
    request["target"] = target
    for name in ("dx", "dy", "steps", "x", "y"):
        if name in fields:
            request[name] = parse_scalar(fields[name], name)
    return request


def session_command(request: dict) -> int:
    isolation_gate("webdriver")
    live = session_live_socket()
    if live is None:
        return 1
    _record, socket_path = live
    try:
        response = session_request(socket_path, request)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[session] request failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(response, indent=2))
    return 0 if response.get("ok") else 1


def session_shot(args: list[str]) -> int:
    name = args[0] if args and not args[0].startswith("--") else "shot"
    if not SESSION_SHOT_NAME.match(name):
        print(f"[session] shot name must match {SESSION_SHOT_NAME.pattern}: {name!r}", file=sys.stderr)
        return 2
    return session_command({"cmd": "shot", "name": name})


def parse_session_verb(rest: list[str]) -> tuple[str, dict, float]:
    if not rest:
        print("[session] a verb is required: up/down/status/inspect/observe/input/shot/checks/tracker/windows/open-window/focus-window/observe-window/evaluate-window", file=sys.stderr)
        raise SystemExit(2)
    verb, args = rest[0], rest[1:]
    lifetime = SESSION_DEFAULT_LIFETIME_SECONDS
    if "--max-lifetime" in args:
        index = args.index("--max-lifetime")
        if index + 1 >= len(args):
            print("[session] --max-lifetime needs a value", file=sys.stderr)
            raise SystemExit(2)
        lifetime = parse_scalar(args[index + 1], "--max-lifetime")
        args = args[:index] + args[index + 2 :]
    if args and verb not in ("input", "observe", "shot", "open-window", "focus-window", "observe-window", "evaluate-window"):
        print(f"[session] verb {verb!r} takes no positional arguments", file=sys.stderr)
        raise SystemExit(2)
    return verb, {"args": args}, lifetime


def session_main(rest: list[str], board_arg: str) -> int:
    try:
        verb, options, lifetime = parse_session_verb(rest)
    except SystemExit as exit:
        return int(exit.code or 2)
    if verb == "up":
        return session_up(board_arg, lifetime)
    if verb == "up-t36":
        return session_up(board_arg, lifetime, mode="t36")
    if verb == "down":
        return session_down()
    if verb == "status":
        return session_status()
    if verb == "suite":
        return session_suite(board_arg, tracker_db_arg)
    if verb == "inspect":
        return session_command({"cmd": "inspect"})
    if verb == "shot":
        return session_shot(options["args"])
    if verb == "input":
        return session_command(build_input_request(options["args"]))
    if verb == "observe":
        fields, _bare = parse_verb_fields(options["args"], {"target", "props"})
        request: dict = {"cmd": "observe"}
        if "target" in fields:
            request["target"] = fields["target"]
        if "props" in fields:
            request["props"] = [p for p in fields["props"].split(",") if p]
        return session_command(request)
    if verb == "windows":
        return session_command({"cmd": "windows"})
    if verb == "open-window":
        fields, _bare = parse_verb_fields(options["args"], {"state"})
        return session_command({"cmd": "open-window", "state_id": fields.get("state")})
    if verb == "focus-window":
        fields, _bare = parse_verb_fields(options["args"], {"label"})
        if "label" not in fields:
            print("[session] focus-window needs label=<window label>", file=sys.stderr)
            return 2
        return session_command({"cmd": "focus-window", "label": fields["label"]})
    if verb == "observe-window":
        fields, _bare = parse_verb_fields(options["args"], {"window", "target", "props"})
        if "window" not in fields or "target" not in fields:
            print("[session] observe-window needs window=<label> and target=<id>", file=sys.stderr)
            return 2
        request_window: dict = {"cmd": "observe-window", "window": fields["window"], "target": fields["target"]}
        if "props" in fields:
            request_window["props"] = [p for p in fields["props"].split(",") if p]
        return session_command(request_window)
    if verb == "evaluate-window":
        fields, bare = parse_verb_fields(options["args"], {"window"})
        if "window" not in fields or not bare:
            print("[session] evaluate-window needs window=<label> and a quoted script", file=sys.stderr)
            return 2
        return session_command({"cmd": "evaluate-window", "window": fields["window"], "script": " ".join(bare)})
    print(f"[session] unknown verb: {verb!r}", file=sys.stderr)
    return 2


def gated_stub(lane: str) -> int:
    isolation_gate(lane)
    print(
        f"lane {lane!r} is not implemented in this change: preflight must pass "
        "and a further reviewed checkpoint must add the payload lane",
        file=sys.stderr,
    )
    return 2


def isolation_gate(lane: str) -> None:
    sys.path.insert(0, str(HERE))
    import isolation

    isolation.gate_or_exit(lane)


def main() -> int:
    global tracker_db_arg
    parser = argparse.ArgumentParser(description="Saccade app harness")
    parser.add_argument(
        "lane",
        choices=["check-fast", "preflight", "session", "serve"],
    )
    parser.add_argument("--tracker-db", help="absolute caller-side tracker db path for the preflight baseline")
    parser.add_argument("--board", help="absolute board file for the native and session lanes")
    parser.add_argument("session_args", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    tracker_db_arg = args.tracker_db or ""
    board_arg = args.board or ""
    if args.lane == "check-fast":
        return check_fast()
    if args.lane == "preflight":
        return preflight()
    if args.lane == "session":
        return session_main(args.session_args, board_arg)
    return gated_stub(args.lane)


if __name__ == "__main__":
    sys.exit(main())
