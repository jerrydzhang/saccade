#!/usr/bin/env python3
"""The remote mesh lane payload: runs INSIDE the enforced sandbox after
the containment canary has passed. Two private herdr-core instances
(sessions a and b) form a mesh in the same fresh runtime tmpfs; B joins
A's machine list through the via program, whose pipe to A's core is the
transport - inside the namespace, no network, no ssh, no shell. The
headless viewer then opens the machine-named location through core A
with the session lane's driver-chain facts, and the lane asserts the
render, the figure bytes, and the remote containment refusal.

Every spawn holds a Popen handle; every wait is bounded; nothing is
retried - a failed step is recorded evidence and the lane goes down in
owned order. This lane is verification evidence for the boundary, not
the human client path, and runs only under a named runtime
authorization.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

HERE = "/lane"
sys.path.insert(0, "/canary")
sys.path.insert(0, HERE)

import containment_canary  # noqa: E402
from webdriver_client import WebDriver, WebDriverError  # noqa: E402
from webdriver_session import (  # noqa: E402
    INSPECT_SCRIPT,
    capture_compositor_frame,
    wait_listening,
)

DRIVER_PORT = 4444
NATIVE_PORT = 4445
NATIVE_HOST = "127.0.0.1"
READY_SECONDS = 15
LISTEN_SECONDS = 90
APP_READY_SECONDS = 45
CORE_SOCKET_SECONDS = 15
CONNECT_WAIT_SECONDS = 20
MACHINE_CALL_SECONDS = 30
MACHINE_NAME = "b"
VIA_SCRIPT_SANDBOX_PATH = "/pin/via_b.py"
REAL_FIGURE_PATH = "assets/figure.png"
ESCAPED_FIGURE_PATH = "escaped.png"
CONTAINMENT_MESSAGE = "resolved outside the declaration directory"

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    print(f"[remote] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}", flush=True)


def pinned(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.startswith("/"):
        print(f"containment refusal: {name} must be an absolute pinned path", file=sys.stderr)
        sys.exit(1)
    return value


def run_canary_first() -> bool:
    return containment_canary.main() == 0


# --- pure contract surface (exercised by harness_checks) ----------------


def via_command(python_bin: str) -> str:
    """The via string machine.add stores: the pinned python interpreter
    and the bound program."""
    return f"{python_bin} {VIA_SCRIPT_SANDBOX_PATH}"


def machine_add_argv(core_bin: str, via: str) -> list[str]:
    """The exact add request: via command, our name, B's session."""
    return [
        core_bin,
        "machine",
        "add",
        "--via",
        via,
        "--name",
        MACHINE_NAME,
        "--session",
        MACHINE_NAME,
        "--json",
    ]


def machine_ls_argv(core_bin: str) -> list[str]:
    return [core_bin, "machine", "ls", "--json"]


def figure_paths(board_text: str) -> list[str]:
    """The authored figure paths in order - render outcomes carry the
    path as their message, so paths are the assertion's address."""
    paths: list[str] = []
    for line in board_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("path:"):
            paths.append(stripped[len("path:") :].strip())
    return paths


def figure_status(inspection: dict, path: str) -> str | None:
    """'ok' when the figure's bytes arrived, 'denied' when a refusal
    quoting the path landed, None when no outcome addresses it."""
    for outcome in inspection.get("renderOutcomes") or []:
        if outcome.get("kind") != "figure":
            continue
        message = str(outcome.get("message", ""))
        if outcome.get("status") == "ok" and message == path:
            return "ok"
        if outcome.get("status") == "denied" and message.startswith("denied:") and path in message:
            return "denied"
    return None


def figure_denial(inspection: dict, path: str) -> str | None:
    """The refusal message for a path, for the check's detail line."""
    for outcome in inspection.get("renderOutcomes") or []:
        if outcome.get("kind") != "figure" or outcome.get("status") != "denied":
            continue
        message = str(outcome.get("message", ""))
        if message.startswith("denied:") and path in message:
            return message
    return None


# --- sandbox-side process steps ------------------------------------------


def start_weston(weston_bin: str, wayland_name: str, logs: pathlib.Path) -> subprocess.Popen[bytes] | None:
    """The native and session lanes' compositor facts verbatim:
    headless pixman, fake seat, kiosk-shell, one socket in the private
    runtime."""
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


def start_core(core_bin: str, session: str, logs: pathlib.Path) -> subprocess.Popen[bytes] | None:
    """One private core in the foreground (the daemon argv production
    itself runs), serving its own session socket in the fresh runtime
    tmpfs."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    log_path = logs / f"core-{session}.log"
    core_log = open(log_path, "wb")
    try:
        core = subprocess.Popen(
            [core_bin, "serve"],
            env={**os.environ, "HC_SESSION": session},
            stdout=core_log,
            stderr=core_log,
        )
    finally:
        core_log.close()
    deadline = time.monotonic() + CORE_SOCKET_SECONDS
    socket_path = pathlib.Path(runtime) / "herdr-core" / f"{session}.sock"
    while time.monotonic() < deadline:
        if core.poll() is not None:
            check(f"core {session} stayed up", False, f"exited during startup (log: {log_path})")
            return None
        if socket_path.exists():
            check(f"core {session} socket up", True, str(socket_path))
            return core
        time.sleep(0.2)
    check(f"core {session} socket up", False, f"never appeared within {CORE_SOCKET_SECONDS}s (log: {log_path})")
    core.terminate()
    core.wait(timeout=5)
    return None


def machine_call(core_bin: str, argv: list[str]) -> tuple[int, bytes, bytes]:
    """One CLI call against session a, never interactive: stdin is
    /dev/null so any accidental prompt read is a loud EOF."""
    result = subprocess.run(
        argv,
        env={**os.environ, "HC_SESSION": "a"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=MACHINE_CALL_SECONDS,
    )
    return result.returncode, result.stdout, result.stderr


def machine_state(core_bin: str) -> tuple[str | None, str | None]:
    """B's line in A's machine list: (state, why), or (None, detail)
    when the listing itself fails, (None, None) when b is absent."""
    code, out, err = machine_call(core_bin, machine_ls_argv(core_bin))
    if code != 0:
        return None, err.decode(errors="replace")[:200]
    try:
        listing = json.loads(out.decode("utf-8"))
    except ValueError:
        return None, "machine ls spoke no JSON"
    for entry in listing.get("machines") or []:
        if entry.get("name") == MACHINE_NAME:
            return str(entry.get("state", "unknown")), entry.get("why") or None
    return None, None


class LaneState:
    def __init__(self) -> None:
        self.core_a: subprocess.Popen[bytes] | None = None
        self.core_b: subprocess.Popen[bytes] | None = None
        self.weston: subprocess.Popen[bytes] | None = None
        self.tauri_driver: subprocess.Popen[bytes] | None = None
        self.client: WebDriver | None = None
        self.torn_down = False


SURVIVOR_COMMS = {
    "WebKitWebDriver",
    "WebKitWebProces",
    "WebKitNetworkPro",
    ".saccade-board-",
    "saccade-board-v",
    "herdr-core",
}


def survivor_pids() -> list[int]:
    """Read-only scan of the sandbox's own PID namespace; nothing here
    is signaled. Any surviving chain or mesh process is recorded, never
    killed by pattern - the namespace teardown ends them."""
    found: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
        except OSError:
            continue
        if comm in SURVIVOR_COMMS:
            found.append(int(entry.name))
    return sorted(found)


def teardown(state: LaneState) -> None:
    """Owned-handle teardown in dependency order: the webdriver session
    first (the remote end ends the app), then the driver chain, then
    the mesh, then the compositor. TERM -> bounded grace -> KILL on our
    handles only; the namespace teardown is the backstop."""
    if state.torn_down:
        return
    state.torn_down = True
    if state.client and state.client.session:
        try:
            state.client.delete_session()
        except WebDriverError as error:
            check("webdriver session deleted", False, str(error))
        else:
            check("webdriver session deleted", True)
    for name, handle in (
        ("tauri-driver", state.tauri_driver),
        ("core a", state.core_a),
        ("core b", state.core_b),
        ("weston", state.weston),
    ):
        if handle is not None and handle.poll() is None:
            handle.terminate()
            try:
                handle.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.kill()
                handle.wait(timeout=5)
    survivors = survivor_pids()
    check("driver chain, mesh and compositor fully torn down", not survivors, f"survivor pids: {survivors}")


def main() -> int:
    if not run_canary_first():
        return 1

    core_bin = pinned("HERDR_CORE_BIN")
    weston_bin = pinned("WESTON_BIN")
    shooter_bin = pinned("SCREENSHOOTER_BIN")
    tauri_driver_bin = pinned("TAURI_DRIVER_BIN")
    webkit_driver_bin = pinned("WEBKIT_DRIVER_BIN")
    app_bin = pinned("APP_BIN")
    via_py = pinned("HERDR_VIA_PY")
    shots = pathlib.Path(pinned("SHOTS_DIR"))
    location = os.environ.get("BOARD_LOCATION", "")
    if not location.startswith(f"{MACHINE_NAME}:/") or "\0" in location:
        print("containment refusal: BOARD_LOCATION must be 'b:/<absolute path>'", file=sys.stderr)
        return 1
    if not shots.is_dir():
        print("containment refusal: SHOTS_DIR must be an existing run-root directory", file=sys.stderr)
        return 1
    logs = shots.parent / "logs"
    logs.mkdir(exist_ok=True)

    started = time.monotonic()
    state = LaneState()
    wayland_name = f"wayland-{os.getpid() % 97}"
    try:
        # the private mesh, before anything that needs it
        core_a = start_core(core_bin, "a", logs)
        if core_a is None:
            teardown(state)
            return 1
        state.core_a = core_a
        core_b = start_core(core_bin, "b", logs)
        if core_b is None:
            teardown(state)
            return 1
        state.core_b = core_b

        # B joins A's list through the via program; add itself dials and
        # checks that hc answers there, so a failure is already evidence
        code, out, err = machine_call(core_bin, machine_add_argv(core_bin, via_command(via_py)))
        detail = (err or out).decode(errors="replace")[:200]
        check("machine add accepted b", code == 0, detail)
        if code != 0:
            teardown(state)
            return 1

        # confirm the state line, never re-dialing: a stuck state is
        # read with its why and fails the lane
        connected = False
        last_detail = ""
        deadline = time.monotonic() + CONNECT_WAIT_SECONDS
        while time.monotonic() < deadline:
            line_state, why = machine_state(core_bin)
            if line_state == "connected":
                connected = True
                break
            last_detail = f"state={line_state!r} why={why!r}"
            time.sleep(0.5)
        check("machine b connected through the via pipe", connected, last_detail)
        if not connected:
            teardown(state)
            return 1

        # the viewer chain: the session lane's facts verbatim, opened on
        # the machine-named location instead of a path
        weston = start_weston(weston_bin, wayland_name, logs)
        if weston is None:
            teardown(state)
            return 1
        state.weston = weston
        check("private headless weston up", True)

        driver_log_path = logs / "tauri-driver.log"
        driver_log = open(driver_log_path, "wb")
        try:
            tauri_driver = subprocess.Popen(
                [
                    tauri_driver_bin,
                    "--port",
                    str(DRIVER_PORT),
                    "--native-port",
                    str(NATIVE_PORT),
                    "--native-host",
                    NATIVE_HOST,
                    "--native-driver",
                    webkit_driver_bin,
                ],
                env={**os.environ, "WAYLAND_DISPLAY": wayland_name},
                stdout=driver_log,
                stderr=driver_log,
            )
        finally:
            driver_log.close()
        state.tauri_driver = tauri_driver
        if not wait_listening(DRIVER_PORT, LISTEN_SECONDS):
            check("tauri-driver listening on loopback", False, f"log: {driver_log_path}")
            teardown(state)
            return 1
        check("tauri-driver listening on loopback", True, f"127.0.0.1:{DRIVER_PORT}")

        client = WebDriver(DRIVER_PORT, timeout=60.0)
        state.client = client
        session_id = client.new_session(app_bin, [location])
        check("webdriver session created for the packaged app", True, f"session {session_id[:8]} on {location}")

        inspection: dict = {}
        ready = False
        last_detail = "no execute result"
        ready_deadline = time.monotonic() + APP_READY_SECONDS
        while time.monotonic() < ready_deadline:
            if tauri_driver.poll() is not None:
                last_detail = f"tauri-driver exited rc={tauri_driver.returncode}"
                break
            try:
                candidate = client.execute(INSPECT_SCRIPT)
            except WebDriverError as error:
                last_detail = str(error)[:160]
                time.sleep(0.5)
                continue
            if isinstance(candidate, dict):
                inspection = candidate
                resource = inspection.get("resource") or {}
                last_detail = f"status={resource.get('status')!r} machine={resource.get('machine')!r}"
                if resource.get("status") == "loaded":
                    ready = True
                    break
            else:
                last_detail = f"execute returned {type(candidate).__name__}"
            time.sleep(0.5)
        check("app frontend up and location opened", ready, f"{location}; last: {last_detail}")
        if not ready:
            teardown(state)
            return 1

        # the three assertions, addressed by the paths the outcomes carry
        resource = inspection.get("resource") or {}
        check(
            "the location names machine b and the declaration",
            resource.get("machine") == MACHINE_NAME and resource.get("path") == "/board/real.yaml",
            str(resource)[:200],
        )
        errors = inspection.get("errors") or []
        check("no load errors", not errors, str(errors)[:200])
        arrived = figure_status(inspection, REAL_FIGURE_PATH)
        check("the real figure's bytes arrived through the mesh", arrived == "ok", f"status={arrived!r}")
        denial = figure_denial(inspection, ESCAPED_FIGURE_PATH)
        check(
            "the escape is refused by remote containment",
            denial is not None and CONTAINMENT_MESSAGE in denial,
            denial or "no denial outcome",
        )

        frame = capture_compositor_frame(shooter_bin, wayland_name, shots, "remote")
        check("compositor frame captured", frame is not None, "screenshooter produced no frame")

        passed = sum(1 for ok, _name, _detail in RESULTS if ok)
        code = 0 if passed == len(RESULTS) else 1
        cost = time.monotonic() - started
        print(f"[remote] {passed}/{len(RESULTS)} checks passed in {cost:.1f}s", flush=True)
        print(f"[remote] shots under {shots} for inspection", flush=True)
        teardown(state)
        return code
    except SystemExit:
        raise  # a pinned-path refusal must keep its own exit code
    except BaseException as error:  # noqa: BLE001 - teardown owns the chain on every path
        print(f"[remote] lane payload error: {error}", file=sys.stderr, flush=True)
        teardown(state)
        return 1


if __name__ == "__main__":
    sys.exit(main())
