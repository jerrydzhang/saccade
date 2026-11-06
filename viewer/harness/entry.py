#!/usr/bin/env python3
"""The single documented entry point for board-viewer development.

Lanes:
  check-fast   typecheck (svelte-check) + pure contract tests + build.
               No services, no display, no isolation requirement: these
               are ordinary build-tool invocations. Warm cost is seconds.
  preflight    GATED. Static containment observations plus the gated
               bwrap unshare probe and the in-sandbox canary. Runs only
               after this harness code is reviewed and authorized.
  browser      GATED. Playwright checks over the built frontend with the
               fixture adapter, inside enforced isolation (bwrap + canary).
  native       GATED. The Tauri/WebKitGTK application under a private
               headless Weston, inside enforced isolation.
  serve        GATED. The built frontend served for manual inspection.

Everything that spawns processes goes through isolation.PidfdSupervisor:
one bwrap supervisor per run, kernel-verified handle, verified reaping,
no pattern kills, no PID-list arithmetic, no weaker fallback.
"""

from __future__ import annotations

import argparse
import os
import pathlib
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


def gated(lane: str) -> int:
    sys.path.insert(0, str(HERE))
    import isolation

    isolation.gate_or_exit(lane)
    # Post-authorization flow (runs only after review):
    #   1. static_preflight(bwrap)
    #   2. gated bwrap unshare probe (bwrap --true) - authorization-specific
    #   3. canary inside the fresh sandbox, before any payload
    #   4. the lane payload, only after a passing canary
    print(
        f"lane {lane!r} authorized marker present, but this checkpoint ships "
        "the harness source for review; the launch sequence activates in the "
        "next change after that review lands",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="board viewer harness")
    parser.add_argument("lane", choices=["check-fast", "preflight", "browser", "native", "serve"])
    args = parser.parse_args()
    if args.lane == "check-fast":
        return check_fast()
    return gated(args.lane)


if __name__ == "__main__":
    sys.exit(main())
