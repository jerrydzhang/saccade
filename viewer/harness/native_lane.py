#!/usr/bin/env python3
"""The native/WebKitGTK lane payload: runs INSIDE the enforced sandbox
after the containment canary has passed, before the private display or
the app exists.

Order, enforced here:
  1. containment canary (the reviewed module, invoked as main())
  2. a private headless Weston compositor (pixman renderer, no GPU, no
     display socket beyond the fresh private XDG_RUNTIME_DIR)
  3. the pinned Tauri application binary in the foreground, opened on
     the explicit board passed as its launch argument (the same
     foreground open operation a human client uses)
  4. bounded frame captures via weston-screenshooter at intervals, plus
     the app's own bridge stderr lines as outcome evidence
  5. owned teardown of the app and compositor children

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


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    stream = sys.stdout if condition else sys.stderr
    print(f"[native] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}", flush=True)


def pinned(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.startswith("/"):
        print(f"containment refusal: {name} must be an absolute pinned path", file=sys.stderr)
        sys.exit(1)
    return value


def run_canary_first() -> bool:
    return containment_canary.main() == 0


def start_weston(weston_bin: str, logs: pathlib.Path) -> subprocess.Popen[bytes] | None:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    wayland_name = f"wayland-{os.getpid() % 97}"
    try:
        # weston 16: the screenshooter protocol is core, no module flag;
        # output goes to a log FILE, never an undrained pipe
        weston_log = open(logs / "weston.log", "wb")
        weston = subprocess.Popen(
            [weston_bin, "--backend=headless", f"--socket={wayland_name}"],
            env={**os.environ, "WAYLAND_DISPLAY": wayland_name},
            stdout=weston_log,
            stderr=weston_log,
        )
    except OSError as error:
        print(f"containment refusal: weston could not start: {error}", file=sys.stderr)
        return None
    deadline = time.monotonic() + READY_SECONDS
    wayland_socket = pathlib.Path(runtime) / wayland_name
    while time.monotonic() < deadline:
        if weston.poll() is not None:
            print("containment refusal: weston exited during startup", file=sys.stderr)
            return None
        if wayland_socket.exists():
            return weston
        time.sleep(0.2)
    print("containment refusal: weston socket never appeared", file=sys.stderr)
    weston.terminate()
    weston.wait(timeout=5)
    return None


def main() -> int:
    started = time.monotonic()
    if not run_canary_first():
        return 1

    weston_bin = pinned("WESTON_BIN")
    app_bin = pinned("APP_BIN")
    board = pinned("BOARD_FILE")
    shooter_bin = pinned("SCREENSHOOTER_BIN")
    shots = pathlib.Path(os.environ.get("SHOTS_DIR", ""))
    logs = shots.parent / "logs"
    logs.mkdir(exist_ok=True)
    if not shots.is_dir():
        print("containment refusal: SHOTS_DIR must be an existing run-root directory", file=sys.stderr)
        return 1

    weston = start_weston(weston_bin, logs)
    if weston is None:
        return 1
    print("[native] private headless weston up", flush=True)

    # the app runs in the foreground with the compositor's display; its
    # stderr (bridge outcome lines) goes to a log FILE
    app_env = dict(os.environ)
    app_env["WAYLAND_DISPLAY"] = os.environ.get("WAYLAND_DISPLAY", "")
    app_log_handle = open(logs / "app.log", "wb")
    app = subprocess.Popen(
        [app_bin, board],
        env=app_env,
        stdout=app_log,
        stderr=app_log,
    )
    print(f"[native] app launched in foreground on {board}", flush=True)
    deadline = time.monotonic() + APP_READY_SECONDS
    frame_paths: list[str] = []
    try:
        for index in range(1, FRAME_COUNT + 1):
            if app.poll() is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(FRAME_INTERVAL_SECONDS, max(remaining, 0)))
            frame = shots / f"native-frame-{index}.png"
            result = subprocess.run(
                ["weston-screenshooter", f"output={frame}"],
                env={**os.environ},
                capture_output=True,
                timeout=10,
            )
            if frame.exists() and frame.stat().st_size > 0:
                frame_paths.append(str(frame))
                check(f"frame {index} captured", True)
            else:
                check(f"frame {index} captured", False, f"rc={result.returncode}")
    finally:
        if app.poll() is None:
            app.terminate()
            try:
                app.wait(timeout=5)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=5)
        app_log = app.stdout.read().decode(errors="replace") if app.stdout else ""
        if app_log.strip():
            print("[native] app log (bridge outcome lines):", flush=True)
            for line in app_log.splitlines():
                if line.startswith("[bridge]"):
                    print(f"[native]   {line}", flush=True)
        if weston.poll() is None:
            weston.terminate()
            try:
                weston.wait(timeout=5)
            except subprocess.TimeoutExpired:
                weston.kill()
                weston.wait(timeout=5)

    check("app stayed alive across frame captures", len(frame_paths) == FRAME_COUNT, str(frame_paths))
    bridge_lines = app_log.count("[bridge]")
    check("bridge produced outcome lines", bridge_lines > 0, f"{bridge_lines} lines")

    cost = time.monotonic() - started
    print(f"[native] {len(RESULTS) - sum(1 for ok, _, _ in RESULTS if not ok)}/{len(RESULTS)} checks passed in {cost:.1f}s", flush=True)
    print(f"[native] frames under {shots} for inspection", flush=True)
    return 1 if any(not ok for ok, _name, _detail in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
