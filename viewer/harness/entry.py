#!/usr/bin/env python3
"""The single documented entry point for board-viewer development.

Lanes:
  check-fast   plain tool runs: svelte-check, pure contract tests, build.
               These are ordinary build-tool invocations; they are
               uncontained on purpose and claim nothing about service
               isolation.
  preflight    GATED. The minimal containment verification: static
               reads, then one bwrap sandbox running the containment
               canary with full namespace/path baselines, bounded
               timeout, owned cleanup. No test payload, browser, core,
               compositor or SSH activity in this lane.
  browser      GATED. Playwright checks inside an enforced sandbox after
               preflight passes.
  native       GATED. The Tauri/WebKitGTK application under a private
               headless Weston inside an enforced sandbox.
  session      GATED. The interactive WebDriver session tool: up holds
               one sandbox with the private compositor, the tauri-driver
               -> WebKitWebDriver chain and the packaged app for an
               explicit max lifetime; inspect/input/observe/shot act and
               observe over a private run-root socket (never a
               host-visible port); down tears the chain down in owned
               order. checks runs the scripted check sequence as a
               series of those commands under one supervisor.
  serve        GATED, not implemented in this change.

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
    ("payload-lint", [sys.executable, "-m", "pyflakes", "harness/entry.py", "harness/isolation.py", "harness/containment_canary.py", "harness/browser_lane.py", "harness/native_lane.py", "harness/webdriver_client.py", "harness/webdriver_session.py", "harness/harness_checks.py"]),
    ("typecheck", ["npm", "run", "check"]),
    ("contracts", ["npm", "run", "test"]),
    # the read-only herdr-core client is a pure crate: its contract
    # tests (request shapes, error mapping, truncation refusal, remote
    # containment) need no GUI libraries, unlike the tauri shell itself
    ("bridge-contracts", ["cargo", "test", "-q", "--manifest-path", "src-tauri/core-client/Cargo.toml"]),
    ("build", ["npm", "run", "build"]),
]

PREFLIGHT_TIMEOUT_SECONDS = 30
# the gate engine is WebKit alone (record 2926); Chromium runs only
# behind BOARD_VIEWER_BROWSER_TRIAGE_CHROMIUM and never gates
BROWSER_TIMEOUT_SECONDS = 180
NATIVE_TIMEOUT_SECONDS = 120
SESSION_UP_TIMEOUT_SECONDS = 90
SESSION_DOWN_TIMEOUT_SECONDS = 30
SESSION_DEFAULT_LIFETIME_SECONDS = 1800
# the scripted check sequence under one supervisor
SESSION_CHECKS_TIMEOUT_SECONDS = 300
# The canary file is bound read-only into the sandbox at this path: the
# interpreter inside must be able to read it without any host HOME or
# worktree visibility.
CANARY_SANDBOX_PATH = "/canary/containment_canary.py"
# Browser-lane payload and binds: the built frontend is the only host
# byte the lane reads, read-only. The stdlib loopback server removed the
# need for any node bind; the lane script rides /lane.
BROWSER_LANE_SANDBOX_PATH = "/lane/browser_lane.py"
DIST_SANDBOX = "/app/dist"
NATIVE_LANE_SANDBOX_PATH = "/lane/native_lane.py"
SESSION_PAYLOAD_SANDBOX_PATH = "/lane/webdriver_session.py"
SESSION_CLIENT_SANDBOX_PATH = "/lane/webdriver_client.py"
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


def verify_playwright_closure(python: str) -> tuple[bool, str]:
    """Pure closure check: the pinned interpreter must belong to a
    withPackages-style env whose site-packages contain the playwright
    module, so the scrubbed-env launch can import it without inheriting
    any host PYTHONPATH."""
    bin_dir = pathlib.Path(python).resolve().parent
    env_root = bin_dir.parent
    matches = sorted(env_root.glob("lib/python3.*/site-packages/playwright/__init__.py"))
    if not matches:
        return False, f"playwright module not found in the pinned closure under {env_root}"
    return True, ""


def browser_lane() -> int:
    """Gated browser lane. Same supervisor, same canary-first order:
    the payload runs the containment canary in-process BEFORE it starts
    the loopback server or the Playwright driver. WebKit gates the lane
    (the default engine); one reused headless WPE browser runs the
    check list over the built frontend, and Chromium joins only behind
    BOARD_VIEWER_BROWSER_TRIAGE_CHROMIUM as a triage pass that never
    gates; screenshots land in the run root for later inspection.
    Bounded by BROWSER_TIMEOUT_SECONDS; pidfd-only cleanup on all
    paths.

    Lane env additions beyond the fresh allowlist, all explicit and
    absolute: PLAYWRIGHT_BROWSERS_PATH (pinned browser package),
    SHOTS_DIR (the run root's own shots directory, already rw-bound),
    FONTCONFIG_FILE (the pinned generated config),
    __EGL_VENDOR_LIBRARY_DIRS (pinned mesa egl_vendor.d) and
    LIBGL_ALWAYS_SOFTWARE (the WPE browser's display-less EGL, the
    native lane's software-GL pin). The lane binds read-only: the
    canary at /canary, the lane script at /lane, viewer/dist at
    /app/dist. The payload python is the pinned withPackages
    interpreter, its closure verified to contain the playwright module
    before any launch (the scrubbed env inherits no host PYTHONPATH).
    Chromium runs headless with its own sandbox disabled - the enforced
    bwrap boundary is the containment.
    """
    sys.path.insert(0, str(HERE))
    import isolation

    isolation.gate_or_exit("browser")

    found = shutil.which("bwrap")
    if found is None:
        print("containment refusal: bwrap is not in PATH (pinned devshell expected)", file=sys.stderr)
        return 1
    bwrap = str(pathlib.Path(found).resolve())
    python = str(pathlib.Path(sys.executable).resolve())
    canary = str((HERE / "containment_canary.py").resolve())
    lane = str((HERE / "browser_lane.py").resolve())

    browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if not browsers.startswith("/"):
        print("containment refusal: PLAYWRIGHT_BROWSERS_PATH must be an absolute pinned path", file=sys.stderr)
        return 1

    # Chromium's font init is fatal without a fontconfig configuration
    # (the diagnosed first-attempt failure): pass the pinned generated
    # config through, never host /etc/fonts or font directories.
    fontconf = os.environ.get("FONTCONFIG_FILE", "")
    usable, reason = verify_fontconfig_file(fontconf)
    if not usable:
        print(f"containment refusal: {reason}", file=sys.stderr)
        return 1

    # the WPE browser's display-less EGL: same software-GL pin as the
    # native lane (pinned mesa vendor dir + forced software rendering);
    # the sandbox env carries them so the browser process inherits both
    mesa_dir = os.environ.get("BOARD_VIEWER_MESA_EGL_VENDOR_DIR", "")
    if not mesa_dir.startswith("/nix/store/") or not os.path.isdir(mesa_dir):
        print(
            "containment refusal: BOARD_VIEWER_MESA_EGL_VENDOR_DIR must be a pinned mesa egl_vendor.d directory",
            file=sys.stderr,
        )
        return 1

    dist = (VIEWER / "dist").resolve()
    if not dist.is_dir():
        print("containment refusal: viewer/dist is missing; run check-fast first", file=sys.stderr)
        return 1

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

    # The pinned launch path must carry its own playwright closure: the
    # scrubbed sandbox env inherits no host PYTHONPATH, so the module
    # must exist inside the withPackages env the interpreter belongs to.
    usable, reason = verify_playwright_closure(python)
    if not usable:
        print(f"containment refusal: {reason}", file=sys.stderr)
        return 1

    run_root = isolation.make_run_root("browser")
    env = isolation.env_allowlist(run_root)
    env["CONTAINMENT_HOST_NS_IDS"] = isolation.host_ns_ids()
    env["CONTAINMENT_HOST_HOME"] = str(pathlib.Path.home().resolve())
    env["CONTAINMENT_HOST_UID"] = str(os.getuid())
    env["CONTAINMENT_HOST_TRACKER_DB"] = tracker_db
    env["PLAYWRIGHT_BROWSERS_PATH"] = browsers
    env["SHOTS_DIR"] = str(run_root / "shots")
    env["FONTCONFIG_FILE"] = fontconf
    env["__EGL_VENDOR_LIBRARY_DIRS"] = mesa_dir
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    # cross-engine triage opt-in: chromium results never gate the lane
    env["BOARD_VIEWER_BROWSER_TRIAGE_CHROMIUM"] = os.environ.get("BOARD_VIEWER_BROWSER_TRIAGE_CHROMIUM", "")

    ro_binds = [
        (canary, CANARY_SANDBOX_PATH),
        (lane, BROWSER_LANE_SANDBOX_PATH),
        (str(dist), DIST_SANDBOX),
    ]
    cmd = isolation.build_command(bwrap, run_root, [python, BROWSER_LANE_SANDBOX_PATH], ro_binds)

    print("[browser] gate command:", " ".join(cmd), flush=True)
    print(f"[browser] run root: {run_root}", flush=True)
    supervisor = isolation.PidfdSupervisor(cmd, env)
    log_path = run_root / "logs" / "browser.log"
    started = time.monotonic()
    try:
        supervisor.start()
        try:
            code = supervisor.wait(timeout=BROWSER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"[browser] timeout after {BROWSER_TIMEOUT_SECONDS}s: stopping via pidfd", file=sys.stderr)
            supervisor.stop()
            log_path.write_text(supervisor.read_output())
            print("containment refusal: browser lane exceeded the time bound", file=sys.stderr)
            return 1
        output = supervisor.read_output()
        log_path.write_text(output)
        cost = time.monotonic() - started
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        print(f"[browser] lane exit={code} in {cost:.1f}s; log: {log_path}", flush=True)
        return code
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return 1
    finally:
        code = supervisor.stop()
        if code != 0 and code is not None:
            print(f"[browser] supervisor exit after cleanup: {code}", file=sys.stderr)


def native_lane(board_arg: str, tracker_db_arg: str) -> int:
    """Gated native/WebKitGTK lane. Same supervisor, same canary-first
    order: the payload runs the containment canary in-process BEFORE the
    private headless Weston or the app exists. One foreground app launch
    on the explicit board argument (the same foreground open a human
    client uses); bounded frame captures; owned teardown of the app and
    compositor children.

    Binds read-only: the canary, the lane script, the pinned app binary
    (a /nix/store path, covered by the store bind), the weston binary,
    the weston-screenshooter binary, and the explicit board file. Env
    additions: WESTON_BIN/APP_BIN/SCREENSHOOTER_BIN (absolute pinned
    paths), BOARD_FILE (absolute explicit board), FONTCONFIG_FILE (the
    pinned generated config), SHOTS_DIR (run-root shots). No GPU, no
    display socket beyond the fresh private XDG_RUNTIME_DIR, no host
    service contact. 120s bound; pidfd-only cleanup.
    """
    sys.path.insert(0, str(HERE))
    import isolation

    isolation.gate_or_exit("native")

    found = shutil.which("bwrap")
    if found is None:
        print("containment refusal: bwrap is not in PATH (pinned devshell expected)", file=sys.stderr)
        return 1
    bwrap = str(pathlib.Path(found).resolve())
    python = str(pathlib.Path(sys.executable).resolve())
    canary = str((HERE / "containment_canary.py").resolve())
    lane = str((HERE / "native_lane.py").resolve())

    weston = shutil.which("weston")
    shooter = shutil.which("weston-screenshooter")
    if weston is None or shooter is None:
        print(
            "containment refusal: weston/weston-screenshooter are not in PATH (pinned devshell expected)",
            file=sys.stderr,
        )
        return 1
    weston_bin = str(pathlib.Path(weston).resolve())
    shooter_bin = str(pathlib.Path(shooter).resolve())

    app = os.environ.get("BOARD_VIEWER_APP_BIN", "")
    if not app.startswith("/nix/store/") or not os.access(app, os.X_OK):
        print(
            "containment refusal: BOARD_VIEWER_APP_BIN must be an executable pinned app binary",
            file=sys.stderr,
        )
        return 1

    # the lane's software-GL path: the packaged app's glvnd libEGL finds
    # no vendor inside the sandbox (its closure carries mesa-libgbm only,
    # and the compiled-in /usr paths do not exist), so the lane pins
    # mesa's egl_vendor.d - llvmpipe/swrast, never /run/opengl-driver,
    # never /dev/dri, never a host GPU path
    mesa_dir = os.environ.get("BOARD_VIEWER_MESA_EGL_VENDOR_DIR", "")
    if not mesa_dir.startswith("/nix/store/") or not os.path.isdir(mesa_dir):
        print(
            "containment refusal: BOARD_VIEWER_MESA_EGL_VENDOR_DIR must be a pinned mesa egl_vendor.d directory",
            file=sys.stderr,
        )
        return 1

    # the app's explicit board: an absolute host path whose parent
    # DIRECTORY is bound read-only at the neutral sandbox path /board -
    # never a host /home path, so the canary's host-visibility check
    # holds and companion/relative assets resolve beside it
    if not board_arg.startswith("/") or not os.path.isfile(board_arg):
        print(
            "containment refusal: pass --board <absolute board.yaml> for the native lane",
            file=sys.stderr,
        )
        return 1
    board_dir = pathlib.Path(board_arg).resolve().parent
    board_name = pathlib.Path(board_arg).resolve().name
    board_sandbox_path = f"/board/{board_name}"

    tracker_db, refusal = caller_tracker_db(
        os.environ.get("SACCADE_DB") or "", tracker_db_arg or ""
    )
    if tracker_db is None:
        print(f"containment refusal: {refusal}", file=sys.stderr)
        return 1
    if not os.path.exists(tracker_db):
        print(f"containment refusal: baseline tracker db does not exist on the caller side: {tracker_db}", file=sys.stderr)
        return 1

    fontconf = os.environ.get("FONTCONFIG_FILE", "")
    usable, reason = verify_fontconfig_file(fontconf)
    if not usable:
        print(f"containment refusal: {reason}", file=sys.stderr)
        return 1

    try:
        isolation.static_preflight(bwrap)
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return 1

    run_root = isolation.make_run_root("native")
    env = isolation.env_allowlist(run_root)
    env["CONTAINMENT_HOST_NS_IDS"] = isolation.host_ns_ids()
    env["CONTAINMENT_HOST_HOME"] = str(pathlib.Path.home().resolve())
    env["CONTAINMENT_HOST_UID"] = str(os.getuid())
    env["CONTAINMENT_HOST_TRACKER_DB"] = tracker_db
    env["WESTON_BIN"] = weston_bin
    env["APP_BIN"] = app
    env["SCREENSHOOTER_BIN"] = shooter_bin
    env["BOARD_FILE"] = board_sandbox_path
    env["FONTCONFIG_FILE"] = fontconf
    env["SHOTS_DIR"] = str(run_root / "shots")
    env["__EGL_VENDOR_LIBRARY_DIRS"] = mesa_dir
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["BOARD_VIEWER_WEBVIEW_CONSOLE"] = "1"
    env["GDK_SCALE"] = "1"
    env["GDK_DPI_SCALE"] = "1"
    (run_root / "shots").mkdir(parents=True, exist_ok=True)

    ro_binds = [
        (canary, CANARY_SANDBOX_PATH),
        (lane, NATIVE_LANE_SANDBOX_PATH),
        (app, app),
        (weston_bin, weston_bin),
        (shooter_bin, shooter_bin),
        (str(board_dir), "/board"),
    ]
    cmd = isolation.build_command(bwrap, run_root, [python, NATIVE_LANE_SANDBOX_PATH], ro_binds)

    print("[native] gate command:", " ".join(cmd), flush=True)
    print(f"[native] run root: {run_root}", flush=True)
    supervisor = isolation.PidfdSupervisor(cmd, env)
    log_path = run_root / "logs" / "native.log"
    started = time.monotonic()
    try:
        supervisor.start()
        try:
            code = supervisor.wait(timeout=NATIVE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"[native] timeout after {NATIVE_TIMEOUT_SECONDS}s: stopping via pidfd", file=sys.stderr)
            supervisor.stop()
            log_path.write_text(supervisor.read_output())
            print("containment refusal: native lane exceeded the time bound", file=sys.stderr)
            return 1
        output = supervisor.read_output()
        log_path.write_text(output)
        cost = time.monotonic() - started
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        print(f"[native] lane exit={code} in {cost:.1f}s; log: {log_path}", flush=True)
        return code
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return 1
    finally:
        code = supervisor.stop()
        if code != 0 and code is not None:
            print(f"[native] supervisor exit after cleanup: {code}", file=sys.stderr)


# --- session lane: the interactive WebDriver tool ------------------------

SESSION_TARGET_ID = re.compile(r"^[A-Za-z0-9:_-]{1,120}$")
SESSION_SHOT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


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
        if process_env_value(pid, "BOARD_VIEWER_SESSION_TOKEN") != token:
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


def session_resolved(board_arg: str, tracker_db_candidate: str) -> dict | None:
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

    app = os.environ.get("BOARD_VIEWER_APP_BIN", "")
    if not app.startswith("/nix/store/") or not os.access(app, os.X_OK):
        print(
            "containment refusal: BOARD_VIEWER_APP_BIN must be an executable pinned app binary",
            file=sys.stderr,
        )
        return None

    mesa_dir = os.environ.get("BOARD_VIEWER_MESA_EGL_VENDOR_DIR", "")
    if not mesa_dir.startswith("/nix/store/") or not os.path.isdir(mesa_dir):
        print(
            "containment refusal: BOARD_VIEWER_MESA_EGL_VENDOR_DIR must be a pinned mesa egl_vendor.d directory",
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
    env["FONTCONFIG_FILE"] = resolved["fontconf"]
    env["SHOTS_DIR"] = str(run_root / "shots")
    env["__EGL_VENDOR_LIBRARY_DIRS"] = resolved["mesa_dir"]
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["BOARD_VIEWER_WEBVIEW_CONSOLE"] = "1"
    env["GDK_SCALE"] = "1"
    env["GDK_DPI_SCALE"] = "1"
    env["SESSION_CONTROL_SOCKET"] = str(run_root / SESSION_SOCKET_NAME)
    env["SESSION_READY_FILE"] = str(run_root / SESSION_READY_NAME)
    env["SESSION_LIFETIME_SECONDS"] = str(int(lifetime))
    # the ownership token the host later verifies in /proc environ
    # before any pidfd signal; never SACCADE_-prefixed (the canary
    # refuses that namespace by design)
    env["BOARD_VIEWER_SESSION_TOKEN"] = token
    (run_root / "shots").mkdir(parents=True, exist_ok=True)
    return env


def session_binds(resolved: dict) -> list[tuple[str, str]]:
    """Read-only binds: the canary, both lane modules, the pinned
    binaries (covered by the /nix/store bind - stated explicitly for
    review) and the board's parent directory. The screenshooter stays
    bound: shot captures compositor frames beside WebDriver shots."""
    return [
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
        if isinstance(pid, int) and pid > 0 and process_env_value(pid, "BOARD_VIEWER_SESSION_TOKEN") == token:
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


def session_up(board_arg: str, lifetime: float) -> int:
    """Gated. One sandbox, one daemonized supervisor holder, one control
    socket. Idempotent while the session is alive; a stale record is
    cleared, never adopted."""
    isolation_gate("webdriver")

    resolved = session_resolved(board_arg, tracker_db_arg)
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
    run_root = isolation.make_run_root("session")
    env = session_env(resolved, run_root, token, lifetime)
    cmd = isolation.build_command(
        resolved["bwrap"],
        run_root,
        [resolved["python"], SESSION_PAYLOAD_SANDBOX_PATH],
        session_binds(resolved),
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
        print("[session] input needs a kind: click/type/key/wheel/drag/coords", file=sys.stderr)
        raise SystemExit(2)
    kind = args[0]
    if kind not in ("click", "type", "key", "wheel", "drag", "coords"):
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
    for name in ("dx", "dy", "steps"):
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


def run_session_checks(socket_path: str, resolved: dict, query: str, log_path: str) -> int:
    """The scripted checks, expressed only through the session command
    surface: semantic addressing via the projected targets, real input
    on elements, observation of geometry, and a shot per state."""
    failures = 0
    log = open(log_path, "a", encoding="utf-8")

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        if not condition:
            failures += 1
        line = f"[webdriver] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def wait_response(payload: dict, predicate, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = session_request(socket_path, payload, timeout=20.0)
            except (OSError, RuntimeError, ValueError):
                time.sleep(0.5)
                continue
            if predicate(response):
                return response
            time.sleep(0.4)
        return None

    def inspection_of(response: dict | None) -> dict | None:
        value = (response or {}).get("inspection")
        return value if isinstance(value, dict) else None

    def target_ids(inspection: dict | None) -> list[str]:
        return [t.get("id", "") for t in (inspection or {}).get("targets", [])]

    def shot(name: str) -> None:
        try:
            response = session_request(socket_path, {"cmd": "shot", "name": name}, timeout=45.0)
        except (OSError, RuntimeError, ValueError) as error:
            check(f"shot {name}", False, str(error))
            return
        ok = bool(response.get("ok") and response.get("web"))
        check(f"shot {name} saved", ok, json.dumps(response)[:200])

    board_path = resolved["board_sandbox"]

    # 1: the board opened through the launch argument - the header shows
    # the loaded path and the projection reports a loaded resource
    opened = wait_response(
        {"cmd": "inspect"},
        lambda r: (i := inspection_of(r)) is not None
        and i.get("resource", {}).get("status") == "loaded"
        and i.get("errors") == [],
        45.0,
    )
    check("board opens through the launch board", opened is not None, board_path)
    header = None
    if opened is not None:
        try:
            header = session_request(socket_path, {"cmd": "observe", "target": "board:location"}, timeout=20.0)
        except (OSError, RuntimeError, ValueError) as error:
            check("header shows the loaded path", False, str(error))
        else:
            observed = header.get("observed") or {}
            check(
                "header shows the loaded path",
                observed.get("text") == board_path,
                repr(observed.get("text"))[:120],
            )
    shot("01-board-open")

    # 2: the projection lists the diagram as an addressable target
    diagram_id = ""
    projected = wait_response(
        {"cmd": "inspect"},
        lambda r: any(tid.startswith("diagram:") for tid in target_ids(inspection_of(r))),
        30.0,
    )
    ids = target_ids(inspection_of(projected))
    diagram_id = next((tid for tid in ids if tid.startswith("diagram:")), "")
    check("diagram targets are projected", bool(diagram_id), str(ids)[:160])
    if not diagram_id:
        return failures

    # 3: real click on the diagram frame expands the overlay
    try:
        clicked = session_request(socket_path, {"cmd": "input", "kind": "click", "target": diagram_id}, timeout=30.0)
    except (OSError, RuntimeError, ValueError) as error:
        clicked = {"ok": False, "error": str(error)}
    check("diagram click accepted", bool(clicked.get("ok")), json.dumps(clicked)[:160])
    expanded = wait_response(
        {"cmd": "inspect"},
        lambda r: "overlay:zoom-in" in target_ids(inspection_of(r)) and "diagram-stage" in target_ids(inspection_of(r)),
        15.0,
    )
    check("clicking a diagram expands it", expanded is not None, "overlay targets absent")
    shot("02-diagram-expanded")

    # 4: the + control changes the rendered geometry, not just the label
    geometry = {}
    for label in ("before", "after"):
        try:
            observed = session_request(socket_path, {"cmd": "observe", "target": "diagram-svg"}, timeout=20.0)
            geometry[label] = (observed.get("observed") or {}).get("rect", {})
        except (OSError, RuntimeError, ValueError) as error:
            geometry[label] = {"error": str(error)}
        if label == "before":
            try:
                session_request(socket_path, {"cmd": "input", "kind": "click", "target": "overlay:zoom-in"}, timeout=30.0)
            except (OSError, RuntimeError, ValueError) as error:
                check("zoom control clicked", False, str(error))
    before = geometry.get("before") or {}
    after = geometry.get("after") or {}
    grew = (
        isinstance(before.get("width"), (int, float))
        and isinstance(after.get("width"), (int, float))
        and float(after["width"]) > float(before["width"]) * 1.1
    )
    check(
        "'+' changes the expanded svg's rendered size",
        grew,
        f"width {before.get('width')} -> {after.get('width')}",
    )
    shot("03-diagram-zoomed")

    # 5: a real pointer drag on the stage moves the diagram
    panned = {}
    for label in ("before", "after"):
        try:
            observed = session_request(socket_path, {"cmd": "observe", "target": "diagram-svg"}, timeout=20.0)
            panned[label] = (observed.get("observed") or {}).get("rect", {})
        except (OSError, RuntimeError, ValueError) as error:
            panned[label] = {"error": str(error)}
        if label == "before":
            try:
                drag = session_request(
                    socket_path,
                    {"cmd": "input", "kind": "drag", "target": "diagram-stage", "dx": 140, "dy": 40},
                    timeout=45.0,
                )
                check("drag accepted", bool(drag.get("ok")), json.dumps(drag)[:160])
            except (OSError, RuntimeError, ValueError) as error:
                check("drag accepted", False, str(error))
    moved = (
        isinstance(panned.get("before"), dict)
        and isinstance(panned.get("after"), dict)
        and "error" not in panned["before"]
        and "error" not in panned["after"]
        and abs(float(panned["after"].get("x", 0)) - float(panned["before"].get("x", 0)))
        + abs(float(panned["after"].get("y", 0)) - float(panned["before"].get("y", 0)))
        > 40
    )
    check(
        "drag pan moves the diagram",
        moved,
        f"before {panned.get('before')} after {panned.get('after')}",
    )
    shot("04-diagram-panned")

    # 5a: the drag is a gesture - the diagram's label text must not be
    # selected by it
    try:
        chosen = session_request(socket_path, {"cmd": "selection"}, timeout=20.0)
        selected = chosen.get("selection")
    except (OSError, RuntimeError, ValueError) as error:
        selected = f"error: {error}"
    check("drag leaves no text selection", selected == "", repr(selected)[:120])

    # 5b: reset returns the rendered geometry: the width shrinks back
    # from the zoomed size toward the unzoomed base
    try:
        session_request(socket_path, {"cmd": "input", "kind": "click", "target": "overlay:reset"}, timeout=30.0)
    except (OSError, RuntimeError, ValueError) as error:
        check("reset clicked", False, str(error))
    try:
        observed = session_request(socket_path, {"cmd": "observe", "target": "diagram-svg"}, timeout=20.0)
        reset_rect = (observed.get("observed") or {}).get("rect", {})
    except (OSError, RuntimeError, ValueError) as error:
        reset_rect = {"error": str(error)}
    shrank = (
        isinstance(reset_rect.get("width"), (int, float))
        and isinstance(panned.get("after"), dict)
        and isinstance(panned["after"].get("width"), (int, float))
        and float(reset_rect["width"]) < float(panned["after"]["width"]) * 0.95
    )
    check(
        "reset returns the rendered geometry",
        shrank,
        f"zoomed width {panned.get('after', {}).get('width')} -> reset width {reset_rect.get('width')}",
    )
    shot("05-diagram-reset")

    # 6: close returns to the board (real click on the overlay's control)
    try:
        session_request(socket_path, {"cmd": "input", "kind": "click", "target": "overlay:close"}, timeout=30.0)
    except (OSError, RuntimeError, ValueError) as error:
        check("close clicked", False, str(error))
    closed = wait_response(
        {"cmd": "inspect"},
        lambda r: not any(tid.startswith("overlay:") for tid in target_ids(inspection_of(r))),
        15.0,
    )
    check("closing returns to the board", closed is not None, "overlay targets persisted")

    # 7: Ctrl+F opens the on-demand search, then a match is found
    try:
        hotkey = session_request(socket_path, {"cmd": "input", "kind": "hotkey", "keys": ["ctrl", "f"]}, timeout=30.0)
        check("ctrl+f accepted", bool(hotkey.get("ok")), json.dumps(hotkey)[:160])
    except (OSError, RuntimeError, ValueError) as error:
        check("ctrl+f accepted", False, str(error))
    search_open = wait_response(
        {"cmd": "inspect"},
        lambda r: "search" in target_ids(inspection_of(r)),
        15.0,
    )
    check("search opens on demand", search_open is not None, "search target absent")
    try:
        typed = session_request(socket_path, {"cmd": "input", "kind": "type", "target": "search", "text": query}, timeout=30.0)
        check("search input accepted", bool(typed.get("ok")), json.dumps(typed)[:160])
    except (OSError, RuntimeError, ValueError) as error:
        check("search input accepted", False, str(error))
    state_pattern = re.compile(r"^(\d+) / (\d+)$")
    found = wait_response(
        {"cmd": "observe", "target": "search:state"},
        lambda r: bool(state_pattern.match(((r.get("observed") or {}).get("text") or "").strip())),
        15.0,
    )
    search_text = (((found or {}).get("observed") or {}).get("text") or "").strip()
    match = state_pattern.match(search_text)
    check("search finds an occurrence", match is not None and int(match.group(2)) >= 1, repr(search_text))
    shot("06-search-match")

    # 8: next match advances when a second occurrence exists
    if match is not None and int(match.group(2)) >= 2:
        try:
            session_request(socket_path, {"cmd": "input", "kind": "click", "target": "search:next"}, timeout=30.0)
        except (OSError, RuntimeError, ValueError) as error:
            check("next match clicked", False, str(error))
        advanced = wait_response(
            {"cmd": "observe", "target": "search:state"},
            lambda r: ((r.get("observed") or {}).get("text") or "").strip().startswith("2 / "),
            15.0,
        )
        check("next match reaches the second occurrence", advanced is not None, search_text)
    else:
        check("next match reaches the second occurrence", False, f"query {query!r} had fewer than two occurrences")

    # 8: the native copy path - Plain on demand (Ctrl+Shift+C), then a
    # formula-plus-prose selection copied with Ctrl+C reaches the
    # bridge with the Plain TeX bytes (the engine's own copy command
    # never dispatches natively)
    try:
        session_request(
            socket_path,
            {"cmd": "evaluate", "script": "window.__boardViewer.setCopyMode('plain'); return true;"},
            timeout=30.0,
        )
    except (OSError, RuntimeError, ValueError):
        pass
    try:
        session_request(socket_path, {"cmd": "input", "kind": "hotkey", "keys": ["escape"]}, timeout=30.0)
        time.sleep(0.3)
    except (OSError, RuntimeError, ValueError):
        pass
    try:
        selection_result = session_request(
            socket_path,
            {
                "cmd": "evaluate",
                "script": (
                    "const math = document.querySelector('.block-text .math-inline');"
                    "if (!math) return 'no math';"
                    "const paragraph = math.closest('p');"
                    "const children = Array.from(paragraph.children);"
                    "const mathIndex = children.indexOf(math);"
                    "const range = document.createRange();"
                    "range.setStart(children[0], 0);"
                    "range.setEnd(children[mathIndex + 1].firstChild, 9);"
                    "const selection = window.getSelection();"
                    "selection.removeAllRanges();"
                    "selection.addRange(range);"
                    "return JSON.stringify(String(selection));"
                ),
            },
            timeout=30.0,
        )
        check(
            "formula-plus-prose selection is set",
            bool(selection_result.get("ok")) and "B_k" not in (selection_result.get("value") or "null"),
            json.dumps(selection_result)[:200],
        )
    except (OSError, RuntimeError, ValueError) as error:
        check("formula-plus-prose selection is set", False, str(error))
    try:
        session_request(socket_path, {"cmd": "input", "kind": "hotkey", "keys": ["ctrl", "c"]}, timeout=30.0)
        time.sleep(0.8)
        last = session_request(
            socket_path,
            {"cmd": "evaluate", "script": "return JSON.stringify(window.__lastCopy || null);"},
            timeout=30.0,
        )
        last_copy = json.loads(last.get("value") or "null")
        ok = (
            isinstance(last_copy, dict)
            and last_copy.get("mode") == "plain"
            and "B_k" in (last_copy.get("text") or "")
        )
        check("plain copy reaches the bridge with the TeX bytes", ok, json.dumps(last_copy)[:200])
    except (OSError, RuntimeError, ValueError) as error:
        check("plain copy reaches the bridge with the TeX bytes", False, str(error))

    return failures


def session_checks(board_arg: str, tracker_db_candidate: str, query: str, lifetime: float) -> int:
    """Gated. The scripted check sequence under one supervisor, then the
    owned teardown through the same command surface an agent uses."""
    isolation_gate("webdriver")
    sys.path.insert(0, str(HERE))
    import isolation

    resolved = session_resolved(board_arg, tracker_db_candidate)
    if resolved is None:
        return 1

    import uuid

    run_root = isolation.make_run_root("session")
    token = uuid.uuid4().hex
    env = session_env(resolved, run_root, token, lifetime)
    (run_root / "logs").mkdir(exist_ok=True)
    checks_log = run_root / "logs" / "session-checks.log"
    cmd = isolation.build_command(
        resolved["bwrap"],
        run_root,
        [resolved["python"], SESSION_PAYLOAD_SANDBOX_PATH],
        session_binds(resolved),
    )
    print("[webdriver] gate command:", " ".join(cmd), flush=True)
    print(f"[webdriver] run root: {run_root}", flush=True)
    socket_path = str(run_root / SESSION_SOCKET_NAME)
    supervisor = isolation.PidfdSupervisor(cmd, env)
    started = time.monotonic()
    failures = 1  # until the check sequence has run, any early exit fails
    try:
        supervisor.start()
        up_deadline = time.monotonic() + SESSION_UP_TIMEOUT_SECONDS
        up = None
        while time.monotonic() < up_deadline:
            if supervisor._proc.poll() is not None:
                break
            try:
                up = session_request(socket_path, {"cmd": "status"}, timeout=2.0)
                break
            except (OSError, RuntimeError, ValueError):
                time.sleep(0.5)
                up = None
        if not isinstance(up, dict) or not up.get("ok"):
            print("[webdriver] the session never came up", file=sys.stderr)
            if supervisor._proc.poll() is not None:
                # the payload already exited: its captured stdout is the
                # diagnosis (EOF is reached, so the read cannot block)
                output = supervisor.read_output()
                print(output, end="" if output.endswith("\n") else "\n", file=sys.stderr, flush=True)
            return 1
        failures = run_session_checks(socket_path, resolved, query, str(checks_log))
        try:
            session_request(socket_path, {"cmd": "down"}, timeout=20.0)
        except (OSError, RuntimeError, ValueError):
            pass
        remaining = started + SESSION_CHECKS_TIMEOUT_SECONDS - time.monotonic()
        try:
            code = supervisor.wait(timeout=max(remaining, 10.0))
        except subprocess.TimeoutExpired:
            print("[webdriver] checks exceeded the session bound; stopping via pidfd", file=sys.stderr)
            supervisor.stop()
            code = None
        output = supervisor.read_output()
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        cost = time.monotonic() - started
        print(f"[webdriver] checks done in {cost:.1f}s; payload exit={code}", flush=True)
        print(f"[webdriver] checks log: {checks_log}", flush=True)
        return 1 if failures or (code not in (0, None)) else 0
    except isolation.ContainmentRefused as error:
        print(f"containment refusal: {error}", file=sys.stderr)
        return 1
    finally:
        code = supervisor.stop()
        if code != 0 and code is not None:
            print(f"[webdriver] supervisor exit after cleanup: {code}", file=sys.stderr)


def parse_session_verb(rest: list[str]) -> tuple[str, dict, float]:
    if not rest:
        print("[session] a verb is required: up/down/status/inspect/observe/input/shot/checks", file=sys.stderr)
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
    query = "board"
    if "--search-query" in args:
        index = args.index("--search-query")
        if index + 1 >= len(args):
            print("[session] --search-query needs a value", file=sys.stderr)
            raise SystemExit(2)
        query = args[index + 1]
        args = args[:index] + args[index + 2 :]
    if args and verb not in ("input", "observe", "shot"):
        print(f"[session] verb {verb!r} takes no positional arguments", file=sys.stderr)
        raise SystemExit(2)
    return verb, {"args": args, "query": query}, lifetime


def session_main(rest: list[str], board_arg: str) -> int:
    try:
        verb, options, lifetime = parse_session_verb(rest)
    except SystemExit as exit:
        return int(exit.code or 2)
    if verb == "up":
        return session_up(board_arg, lifetime)
    if verb == "down":
        return session_down()
    if verb == "status":
        return session_status()
    if verb == "checks":
        return session_checks(board_arg, tracker_db_arg, options["query"], lifetime)
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
    parser = argparse.ArgumentParser(description="board viewer harness")
    parser.add_argument(
        "lane",
        choices=["check-fast", "preflight", "browser", "native", "session", "serve"],
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
    if args.lane == "browser":
        return browser_lane()
    if args.lane == "native":
        return native_lane(board_arg, tracker_db_arg)
    if args.lane == "session":
        return session_main(args.session_args, board_arg)
    return gated_stub(args.lane)


if __name__ == "__main__":
    sys.exit(main())
