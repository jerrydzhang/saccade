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
  session      GATED. session suite: the enforced sandbox with a private
               headless Weston around the packaged app, then the app's
               vitest native project (client/port, the payload
               suite_payload.py).
  serve        GATED, not implemented in this change.

(The old app's browser, native and remote lanes, the scripted session
checks and the session tracker sequence retired with the old app,
t-36 m-554; the t36 Python port lane and its up-t36 mode moved into
vitest, t-36 m-556; the interactive session tool - up, down and the
window verbs - retired, t-36 m-559.)

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
    ("payload-lint", [sys.executable, "-m", "pyflakes", "harness/entry.py", "harness/isolation.py", "harness/containment_canary.py", "harness/harness_checks.py", "harness/t36_seeds.py", "harness/suite_payload.py"]),
    # the harness's own unit tests (the seed module the scenarios' tracker
    # is seeded with)
    ("harness-tests", [sys.executable, "-B", "-m", "unittest", "discover", "-s", "harness/tests", "-t", "."]),
    # the app's frontend typecheck (client/port; its suites run through
    # vitest: just app-test and just app-test-browser)
    ("typecheck", ["node_modules/.bin/tsc", "--noEmit", "-p", "port/tsconfig.json"]),
    # the read-only herdr-core client is a pure crate: its contract
    # tests (request shapes, error mapping, truncation refusal, remote
    # containment) need no GUI libraries, unlike the tauri shell itself
    ("bridge-contracts", ["cargo", "test", "-q", "--manifest-path", "src-tauri/core-client/Cargo.toml"]),
    ("build", ["node_modules/.bin/vite", "build", "-c", "port/vite.config.ts"]),
]

PREFLIGHT_TIMEOUT_SECONDS = 30
# the suite mode's whole bound: weston, then the port suite's native
# project (one fresh app chain per test), teardown
SESSION_SUITE_TIMEOUT_SECONDS = 960
# The canary file is bound read-only into the sandbox at this path: the
# interpreter inside must be able to read it without any host HOME or
# worktree visibility.
CANARY_SANDBOX_PATH = "/canary/containment_canary.py"
# the suite mode: its payload, and the client tree (node_modules, the
# port, the scenarios, the seed module) read-only
SUITE_PAYLOAD_SANDBOX_PATH = "/lane/suite_payload.py"
SUITE_CLIENT_SANDBOX_PATH = "/suite/client"
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
    app_accepted = app.startswith("/nix/store/") or app.startswith(f"/var/tmp/saccade-app-{os.getuid()}/debug-bin/")
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


def session_env(resolved: dict, run_root: pathlib.Path, token: str) -> dict[str, str]:
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
    # the ownership token the host later verifies in /proc environ
    # before any pidfd signal; never SACCADE_-prefixed (the canary
    # refuses that namespace by design)
    env["SAC_APP_SESSION_TOKEN"] = token
    (run_root / "shots").mkdir(parents=True, exist_ok=True)
    return env


def session_binds(resolved: dict) -> list[tuple[str, str]]:
    """Read-only binds: the canary, both lane modules, the pinned
    binaries (covered by the /nix/store bind - stated explicitly for
    review) and the board's parent directory. The screenshooter stays
    bound: shot captures compositor frames beside WebDriver shots.
    A run that seeds a tracker adds its own sac binary at /lane/sac."""
    binds = [
        (resolved["canary"], CANARY_SANDBOX_PATH),
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
    return binds


def session_suite(board_arg: str, tracker_db_candidate: str, vitest_args: list[str] | None = None) -> int:
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
    env = session_env(resolved, run_root, token)
    env["SESSION_MODE"] = "suite"
    env["SUITE_NODE_BIN"] = node
    env["SUITE_CLIENT_DIR"] = SUITE_CLIENT_SANDBOX_PATH
    binds = session_binds(resolved) + [
        (str((HERE / "suite_payload.py").resolve()), SUITE_PAYLOAD_SANDBOX_PATH),
        (str(VIEWER.resolve()), SUITE_CLIENT_SANDBOX_PATH),
    ]
    # vitest's own filters (a file or a test name) pass through to the
    # payload, which appends them to the native project's run
    payload = [resolved["python"], SUITE_PAYLOAD_SANDBOX_PATH, *(vitest_args or [])]
    cmd = isolation.build_command(resolved["bwrap"], run_root, payload, binds)
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


def session_main(rest: list[str], board_arg: str) -> int:
    """The session lane's one verb: suite (the app's vitest native
    project in the sandbox), optionally followed by vitest's own
    filters (`session suite s18`)."""
    if rest[:1] != ["suite"]:
        print("[session] the one verb is suite (the native project run)", file=sys.stderr)
        return 2
    return session_suite(board_arg, tracker_db_arg, rest[1:])


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
