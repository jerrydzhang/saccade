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
     the test-open channel's location (BOARD_VIEWER_TEST_OPEN: the
     lanes' test-only machinery, riding the same model.open the
     inspection surface uses, t-24@r11), sharing the compositor's
     WAYLAND_DISPLAY (one name, computed here)
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
sys.path.insert(0, "/mock")

import containment_canary  # noqa: E402
import tracker_mock  # noqa: E402

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
    # the bridge outcome lines and the webview console, written to log
    # FILES
    app_env = {**os.environ, "WAYLAND_DISPLAY": wayland_name}

    def run_app(test_open: str, log_name: str) -> tuple[str, list[str], bool, int | None]:
        """One foreground launch with bounded captures, then owned
        teardown of the app alone: the log text, the frames, the
        liveness at capture start, and the exit code read BEFORE our
        teardown."""
        log_path = logs / log_name
        log = open(log_path, "wb")
        child = subprocess.Popen(
            [app_bin],
            env={**app_env, "BOARD_VIEWER_TEST_OPEN": test_open},
            stdout=log,
            stderr=log,
        )
        log.close()
        print(f"[native] app launched in foreground on test open {test_open}", flush=True)
        frames: list[str] = []
        alive = child.poll() is None
        code: int | None = None
        try:
            for index in range(1, FRAME_COUNT + 1):
                if child.poll() is not None:
                    break
                remaining = (started_at + 240) - time.monotonic()
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
                    # a hung capture is a recorded failed check, never a
                    # lane crash - teardown and evidence still run
                    check(f"frame {index} captured ({log_name})", False, "screenshooter timed out after 10s")
                    continue
                new_frames = sorted(set(shots.glob("wayland-screenshot-*.png")) - before)
                if new_frames:
                    frames.extend(str(f) for f in new_frames)
                    check(f"frame {index} captured ({log_name})", True, ", ".join(f.name for f in new_frames))
                else:
                    detail = result.stderr.decode(errors="replace")[:120]
                    check(f"frame {index} captured ({log_name})", False, f"rc={result.returncode} {detail}")
        finally:
            code = child.poll()
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        return log_path.read_text(errors="replace"), frames, alive, code

    # launch 1: the fixture board through the host reader - the lanes'
    # test-only local-file reader (t-24@r11), through the bridge's real
    # read_declaration with its real guards. The standing contract: the
    # board renders and the bridge outcome is on the log.
    app_log_text, frame_paths, alive_at_start, exit_code = run_app(f"host:{board}", "app.log")

    # liveness is recorded separately from the frame count: poll state
    # at capture start, the exit code read BEFORE our teardown, then the
    # frames. Evidence checks demand the real markers: the webview's
    # startup line (frontend executed) and a read_declaration outcome
    # (the board actually opened) - not just any [bridge] line.
    check("app alive when captures began", alive_at_start, f"poll={alive_at_start}")
    check("app alive at teardown", exit_code is None, f"exit={exit_code}")
    check("frames captured", len(frame_paths) == FRAME_COUNT, str(frame_paths))
    check(
        "webview frontend executed",
        "[viewer] startup" in app_log_text,
        "[viewer] startup marker absent",
    )
    check(
        "board open through the bridge",
        "[bridge] read_declaration" in app_log_text,
        "no read_declaration outcome",
    )
    for line in app_log_text.splitlines():
        if line.startswith("[bridge]"):
            print(f"[native]   {line}", flush=True)

    # launch 2: a tracker location. The tracker server is the lane's own
    # contract stand-in on the sandbox's loopback. Without input
    # injection this lane cannot pre-seed the server setting, so the
    # pinned behavior is the teaching refusal plus the boundary fact the
    # location form must hold everywhere: the tracker form never touches
    # the file bridge - no read_declaration, no state-directory read.
    mock = tracker_mock.TrackerMock("smoke")
    mock.start()
    try:
        tracker_log, tracker_frames, tracker_alive, tracker_exit = run_app("smoke/t-0", "app-tracker.log")
        check("tracker app alive when captures began", tracker_alive, f"poll={tracker_alive}")
        check("tracker app alive at teardown", tracker_exit is None, f"exit={tracker_exit}")
        check("tracker frames captured", len(tracker_frames) == FRAME_COUNT, str(tracker_frames))
        check(
            "tracker webview frontend executed",
            "[viewer] startup" in tracker_log,
            "[viewer] startup marker absent",
        )
        check(
            "tracker location opens no file through the bridge",
            "[bridge] read_declaration" not in tracker_log,
            "the tracker form read a file path",
        )
        check(
            "the unset server refuses with the teaching line",
            "[viewer] test open board FAILED" in tracker_log
            and "no tracker server set" in tracker_log,
            tracker_log[-400:],
        )
        for line in tracker_log.splitlines():
            if "[viewer] test open board" in line:
                print(f"[native]   {line}", flush=True)
    finally:
        mock.stop()
        if weston.poll() is None:
            weston.terminate()
            try:
                weston.wait(timeout=5)
            except subprocess.TimeoutExpired:
                weston.kill()
                weston.wait(timeout=5)

    cost = time.monotonic() - started_at
    passed = sum(1 for ok, _name, _detail in RESULTS if ok)
    print(f"[native] {passed}/{len(RESULTS)} checks passed in {cost:.1f}s", flush=True)
    print(f"[native] frames under {shots} for inspection", flush=True)
    return 1 if any(not ok for ok, _name, _detail in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
