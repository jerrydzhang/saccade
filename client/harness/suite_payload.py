#!/usr/bin/env python3
"""The suite payload (entry.py session suite): runs INSIDE the session
lane's enforced sandbox, after nothing.

Order, enforced here:
  1. the containment canary (the reviewed module, as main()) - a refusal
     ends the run before anything spawns
  2. the private headless Weston, with the session lane's own compositor
     facts (webdriver_session.start_weston, imported, never copied)
  3. the port suite's native project: vitest in the bound client tree
     (client/port/vitest.config.ts --project native), whose fixture gives
     each test its own tauri-driver -> WebKitWebDriver -> packaged app
     chain, fresh XDG dirs under this run root, and a WebdriverIO session
  4. owned teardown: Weston stopped by its own handle

The suite owns its per-test chain (spawned and stopped by its fixture by
handle); the bwrap supervisor stop is the backstop. Its throwaway tracker
is a real sac serve on the sandbox loopback (/lane/sac, git at /lane/git).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

sys.path.insert(0, "/canary")
sys.path.insert(0, "/lane")

import containment_canary  # noqa: E402
import webdriver_session  # noqa: E402

SUITE_SECONDS = 900


def main(argv: list[str]) -> int:
    if containment_canary.main() != 0:
        return 1
    weston_bin = webdriver_session.pinned("WESTON_BIN")
    node = webdriver_session.pinned("SUITE_NODE_BIN")
    client = webdriver_session.pinned("SUITE_CLIENT_DIR")
    shots = pathlib.Path(webdriver_session.pinned("SHOTS_DIR"))
    logs = shots.parent / "logs"
    logs.mkdir(exist_ok=True)
    wayland = f"wayland-{os.getpid() % 97}"
    weston = webdriver_session.start_weston(weston_bin, wayland, logs)
    if weston is None:
        return 1
    print(f"[suite] private headless weston up ({wayland})", flush=True)
    env = {
        **os.environ,
        "WAYLAND_DISPLAY": wayland,
        # the lane's sac and git, then node and this python (the seeding)
        "PATH": ":".join(["/lane", str(pathlib.Path(node).parent), str(pathlib.Path(sys.executable).parent), os.environ.get("PATH", "/usr/bin:/bin")]),
        "SACCADE_SERVE_BIN": "/lane/sac",
        # vite's cache, off the read-only client tree
        "SUITE_CACHE_DIR": str(shots.parent / "cache" / "vite"),
    }
    command = [
        node,
        str(pathlib.Path(client) / "node_modules" / "vitest" / "vitest.mjs"),
        "run",
        "-c",
        "port/vitest.config.ts",
        # the client tree is read-only here: the runner loader evaluates
        # the config in place, where the bundle loader writes a temp file
        "--configLoader",
        "runner",
        "--project",
        "native",
        "--tags-filter",
        "native || multi-window",
        *argv,
    ]
    print("[suite] " + " ".join(command), flush=True)
    # the suite's own output is kept in the run root (logs/suite.log)
    # as well as relayed to the caller
    suite_log = open(logs / "suite.log", "wb")
    try:
        proc = subprocess.Popen(command, cwd=client, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for line in proc.stdout:
            suite_log.write(line)
            sys.stdout.buffer.write(line)
        try:
            code = proc.wait(timeout=SUITE_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"[suite] the native project exceeded {SUITE_SECONDS}s", file=sys.stderr, flush=True)
            proc.kill()
            proc.wait(timeout=5)
            code = 1
    finally:
        suite_log.close()
        weston.terminate()
        try:
            weston.wait(timeout=5)
        except subprocess.TimeoutExpired:
            weston.kill()
            weston.wait(timeout=5)
    print(f"[suite] native project exit={code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
