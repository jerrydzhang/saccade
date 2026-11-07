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
  browser      GATED, not implemented in this change. Will run Playwright
               checks inside an enforced sandbox after preflight passes.
  native       GATED, not implemented in this change. Will run the
               Tauri/WebKitGTK application under a private headless
               Weston inside an enforced sandbox.
  serve        GATED, not implemented in this change.

Everything that spawns sandbox processes goes through
isolation.PidfdSupervisor: one bwrap supervisor per run, kernel-verified
handle, verified reaping, no pattern kills, no numeric-PID fallback.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
VIEWER = HERE.parent

FAST_STEPS = [
    ("harness", [sys.executable, "-B", "harness/harness_checks.py"]),
    ("payload-lint", [sys.executable, "-m", "pyflakes", "harness/entry.py", "harness/isolation.py", "harness/containment_canary.py", "harness/browser_lane.py", "harness/native_lane.py", "harness/harness_checks.py"]),
    ("typecheck", ["npm", "run", "check"]),
    ("contracts", ["npm", "run", "test"]),
    ("build", ["npm", "run", "build"]),
]

PREFLIGHT_TIMEOUT_SECONDS = 30
# two engine passes (Chromium, then Playwright WebKit) over one check
# list: the bound is wall time for the whole lane, so it doubles with
# the second engine
BROWSER_TIMEOUT_SECONDS = 360
NATIVE_TIMEOUT_SECONDS = 120
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
    the loopback server or the Playwright driver. One reused headless
    browser per engine - Chromium first, then Playwright WebKit - each
    running the same check list over the built frontend; screenshots
    land in the run root for later inspection. Bounded by
    BROWSER_TIMEOUT_SECONDS; pidfd-only cleanup on all paths.

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
    parser.add_argument("lane", choices=["check-fast", "preflight", "browser", "native", "serve"])
    parser.add_argument("--tracker-db", help="absolute caller-side tracker db path for the preflight baseline")
    parser.add_argument("--board", help="absolute board file for the native lane foreground open")
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
    return gated_stub(args.lane)


if __name__ == "__main__":
    sys.exit(main())
