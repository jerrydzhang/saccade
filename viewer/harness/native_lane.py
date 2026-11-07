#!/usr/bin/env python3
"""The native/WebKitGTK lane payload: runs INSIDE the enforced sandbox
after the containment canary has passed, before the private display or
the app exists.

Order, enforced here:
  1. containment canary (the reviewed module, invoked as main())
  2. a private headless Weston compositor (pixman renderer, no GPU, no
     display socket beyond the fresh private XDG_RUNTIME_DIR); the
     screenshooter protocol is core in weston 16 - no module flags
  3. the pinned Tauri application binary in the foreground, opened on
     the explicit board passed as its launch argument, sharing the
     compositor's WAYLAND_DISPLAY (one name, computed here)
  4. bounded frame captures: weston-screenshooter 16 takes no output
     argument and writes wayland-screenshot-output-*.png to its CWD, so
     captures run with cwd=SHOTS_DIR and the lane collects new files
  5. owned teardown of the app and compositor children; app and weston
     output go to log FILES in the run root, never undrained pipes

Frames and logs land in the run root for actual inspection. This lane
is verification evidence only; it is not the human client path and
grants no acceptance.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/canary")

import containment_canary  # noqa: E402

READY_SECONDS = 15
FRAME_COUNT = 3
FRAME_INTERVAL_SECONDS = 4
APP_READY_SECONDS = 20

RESULTS: list[tuple[bool, str, str]] = []
started_at = 0.0


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    print(f"[native] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}", flush=True)


def pinned(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.startswith("/"):
        print(f"containment refusal: {name} must be an absolute pinned path", file=sys.stderr)
        sys.exit(1)
    return value


def run_canary_first() -> bool:
    return containment_canary.main() == 0


def start_weston(weston_bin: str, wayland_name: str, logs: pathlib.Path) -> subprocess.Popen[bytes] | None:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    weston_log_path = logs / "weston.log"
    weston_log = open(weston_log_path, "wb")
    # --debug exposes weston-output-capture (weston(1): only with it does
    # weston-screenshooter get capture authority); kiosk-shell makes the
    # app fill the headless output and drops the desktop-shell helper
    # clients. Both are private-sandbox facts: private compositor,
    # private XDG_RUNTIME_DIR, only this lane's own app as client.
    try:
        weston = subprocess.Popen(
            [
                weston_bin,
                "--backend=headless",
                f"--socket={wayland_name}",
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


def main() -> int:
    global started_at
    started_at = time.monotonic()
    if not run_canary_first():
        return 1

    weston_bin = pinned("WESTON_BIN")
    app_bin = pinned("APP_BIN")
    board = pinned("BOARD_FILE")
    shooter_bin = pinned("SCREENSHOOTER_BIN")
    wayland_name = f"wayland-{os.getpid() % 97}"
    shots = pathlib.Path(os.environ.get("SHOTS_DIR", ""))
    logs = pathlib.Path(os.environ.get("SHOTS_DIR", "")).parent / "logs"
    logs.mkdir(exist_ok=True)
    if not shots.is_dir():
        print("containment refusal: SHOTS_DIR must be an existing run-root directory", file=sys.stderr)
        return 1

    weston = start_weston(weston_bin, wayland_name, logs)
    if weston is None:
        return 1
    print("[native] private headless weston up", flush=True)

    # the app shares the compositor's display name; its stderr carries
    # the bridge outcome lines, written to a log FILE
    app_env = {**os.environ, "WAYLAND_DISPLAY": wayland_name}
    app_log_path = logs / "app.log"
    app_log = open(app_log_path, "wb")
    app = subprocess.Popen([app_bin, board], env=app_env, stdout=app_log, stderr=app_log)
    app_log.close()
    print(f"[native] app launched in foreground on {board}", flush=True)

    frame_paths: list[str] = []
    try:
        for index in range(1, FRAME_COUNT + 1):
            if app.poll() is not None:
                break
            remaining = (started_at + 120) - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(FRAME_INTERVAL_SECONDS, max(remaining, 0)))
            before = set(shots.glob("wayland-screenshot-*.png"))
            try:
                result = subprocess.run(
                    [shooter_bin],
                    cwd=str(shots),
                    env={**os.environ, "WAYLAND_DISPLAY": wayland_name},
                    capture_output=True,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                # a hung capture is a recorded failed check, never a lane
                # crash - teardown and the remaining evidence still run
                check(f"frame {index} captured", False, "screenshooter timed out after 10s")
                continue
            new_frames = sorted(set(shots.glob("wayland-screenshot-*.png")) - before)
            if new_frames:
                frame_paths.extend(str(f) for f in new_frames)
                check(f"frame {index} captured", True, ", ".join(f.name for f in new_frames))
            else:
                detail = result.stderr.decode(errors="replace")[:120]
                check(f"frame {index} captured", False, f"rc={result.returncode} {detail}")
    finally:
        if app.poll() is None:
            app.terminate()
            try:
                app.wait(timeout=5)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=5)
        if weston.poll() is None:
            weston.terminate()
            try:
                weston.wait(timeout=5)
            except subprocess.TimeoutExpired:
                weston.kill()
                weston.wait(timeout=5)

    check("app stayed alive across frame captures", len(frame_paths) == FRAME_COUNT, str(frame_paths))
    app_log_text = app_log_path.read_text(errors="replace")
    bridge_lines = app_log_text.count("[bridge]")
    check("bridge produced outcome lines", bridge_lines > 0, f"{bridge_lines} lines")
    for line in app_log_text.splitlines():
        if line.startswith("[bridge]"):
            print(f"[native]   {line}", flush=True)

    cost = time.monotonic() - started_at
    passed = sum(1 for ok, _name, _detail in RESULTS if ok)
    print(f"[native] {passed}/{len(RESULTS)} checks passed in {cost:.1f}s", flush=True)
    print(f"[native] frames under {shots} for inspection", flush=True)
    return 1 if any(not ok for ok, _name, _detail in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
