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
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
VIEWER = HERE.parent

FAST_STEPS = [
    ("typecheck", ["npm", "run", "check"]),
    ("contracts", ["npm", "run", "test"]),
    ("build", ["npm", "run", "build"]),
]

PREFLIGHT_TIMEOUT_SECONDS = 30


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


def preflight() -> int:
    """Minimal containment verification, runnable exactly as reviewed.

    1. Static reads: bwrap present in the pinned closure, user-namespace
       sysctls positive. No unshare happens here.
    2. One sandbox: build_command with the canary as the sole payload,
       fresh environment for the whole chain, full host baselines, the
       recorded run root and logs. Bounded by PREFLIGHT_TIMEOUT_SECONDS;
       cleanup is pidfd-only with verified reaping, on all paths.
    """
    sys.path.insert(0, str(HERE))
    import isolation

    isolation.gate_or_exit("preflight")

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        print("containment refusal: bwrap is not in PATH (pinned devshell expected)", file=sys.stderr)
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
    env["CONTAINMENT_HOST_HOME"] = str(pathlib.Path.home())
    env["CONTAINMENT_HOST_UID"] = str(os.getuid())
    env["CONTAINMENT_HOST_TRACKER_DB"] = "saccade.db"

    canary = str(HERE / "containment_canary.py")
    python = sys.executable  # absolute path into the pinned closure
    cmd = isolation.build_command(bwrap, run_root, [python, canary], [])

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
    parser = argparse.ArgumentParser(description="board viewer harness")
    parser.add_argument("lane", choices=["check-fast", "preflight", "browser", "native", "serve"])
    args = parser.parse_args()
    if args.lane == "check-fast":
        return check_fast()
    if args.lane == "preflight":
        return preflight()
    return gated_stub(args.lane)


if __name__ == "__main__":
    sys.exit(main())
