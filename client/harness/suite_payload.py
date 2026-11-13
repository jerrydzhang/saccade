#!/usr/bin/env python3
"""The suite payload (entry.py session suite): runs INSIDE the session
lane's enforced sandbox, after nothing.

Order, enforced here:
  1. the containment canary (the reviewed module, as main()) - a refusal
     ends the run before anything spawns
  2. the private headless Weston (start_weston below: headless pixman,
     a fake seat, kiosk-shell, one socket in the private runtime)
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
import time

sys.path.insert(0, "/canary")
sys.path.insert(0, "/lane")

import containment_canary  # noqa: E402

SUITE_SECONDS = 900
# the compositor's readiness bound
READY_SECONDS = 15


def pinned(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.startswith("/"):
        print(f"containment refusal: {name} must be an absolute pinned path", file=sys.stderr)
        sys.exit(1)
    return value


def start_weston(weston_bin: str, wayland_name: str, logs: pathlib.Path) -> subprocess.Popen[bytes] | None:
    """The native lane's compositor facts verbatim: headless pixman with
    a fake seat and kiosk-shell, one socket in the private runtime."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    weston_log_path = logs / "weston.log"
    weston_log = open(weston_log_path, "wb")
    try:
        weston = subprocess.Popen(
            [
                weston_bin,
                "--backend=headless",
                f"--socket={wayland_name}",
                "--renderer=pixman",
                "--width=1024",
                "--height=1280",
                "--scale=1",
                "--fake-seat",
                "--debug",
                "--shell=kiosk-shell.so",
            ],
            env={**os.environ, "WAYLAND_DISPLAY": wayland_name},
            stdout=weston_log,
            stderr=weston_log,
        )
    finally:
        weston_log.close()
    deadline = time.monotonic() + READY_SECONDS
    wayland_socket = pathlib.Path(runtime) / wayland_name
    while time.monotonic() < deadline:
        if weston.poll() is not None:
            print(
                f"containment refusal: weston exited during startup (log: {weston_log_path})",
                file=sys.stderr,
            )
            return None
        if wayland_socket.exists():
            return weston
        time.sleep(0.2)
    print(f"containment refusal: weston socket never appeared (log: {weston_log_path})", file=sys.stderr)
    weston.terminate()
    weston.wait(timeout=5)
    return None


def main(argv: list[str]) -> int:
    if containment_canary.main() != 0:
        return 1
    weston_bin = pinned("WESTON_BIN")
    node = pinned("SUITE_NODE_BIN")
    client = pinned("SUITE_CLIENT_DIR")
    shots = pathlib.Path(pinned("SHOTS_DIR"))
    logs = shots.parent / "logs"
    logs.mkdir(exist_ok=True)
    wayland = f"wayland-{os.getpid() % 97}"
    weston = start_weston(weston_bin, wayland, logs)
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
