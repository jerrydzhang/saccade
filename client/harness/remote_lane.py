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
import socket
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
    perform_input,
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
FIGURE_OUTCOME_SECONDS = 20
SURVIVOR_SETTLE_SECONDS = 10
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
    """The authored figure inputs in order - render outcomes carry the
    input as their message, so inputs are the assertion's address."""
    paths: list[str] = []
    for line in board_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("input:"):
            paths.append(stripped[len("input:") :].strip())
    return paths


def figure_outcomes(inspection: dict) -> list[dict]:
    """Figure outcomes in authored order: hydration resolves the reads
    sequentially, so the Nth figure outcome addresses the Nth authored
    figure path. The message under a miss is the bridge's failure line
    and may not name the path at all - position is the address that
    always exists."""
    return [o for o in inspection.get("renderOutcomes") or [] if o.get("kind") == "figure"]


def figure_outcome_at(inspection: dict, index: int) -> dict | None:
    outcomes = figure_outcomes(inspection)
    return outcomes[index] if index < len(outcomes) else None


def figure_status(inspection: dict, path: str) -> str | None:
    """'ok' when the figure's bytes arrived, 'denied' when a refusal
    quoting the path landed, None when no outcome addresses it."""
    outcome = figure_outcome(inspection, path)
    if outcome is None:
        return None
    return "ok" if outcome.get("status") == "ok" else "denied"


def figure_denial_at(inspection: dict, index: int) -> str | None:
    """The refusal message at a figure position, or None when that
    position has no denial."""
    outcome = figure_outcome_at(inspection, index)
    if outcome is not None and outcome.get("status") == "denied":
        return str(outcome.get("message", ""))
    return None


def figure_outcome(inspection: dict, path: str) -> dict | None:
    """The figure outcome for a path, in any terminal state: ok
    carries the bare path as its message; denials and misses quote it
    after their prefix. Hydration publishes the whole outcome list only
    once every read has landed, so a present-but-not-ok outcome is
    itself evidence the read was attempted and refused or failed."""
    for outcome in inspection.get("renderOutcomes") or []:
        if outcome.get("kind") != "figure":
            continue
        status = outcome.get("status")
        message = str(outcome.get("message", ""))
        if status == "ok" and message == path:
            return outcome
        if status in ("denied", "missing") and message.startswith(("denied:", "missing:")) and path in message:
            return outcome
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
    tmpfs. The core's PATH is built from the pins, because bwrap's
    --setenv overrides PATH with the bare allowlist value and the
    core's own spawns (the mesh's fixed argv) would otherwise not find
    the pinned readers."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    coreutils_bin = os.environ.get("BOARD_VIEWER_COREUTILS_BIN", "")
    path_parts = [
        os.path.dirname(core_bin),
        coreutils_bin,
        os.environ.get("PATH", ""),
    ]
    core_log_path = logs / f"core-{session}.log"
    core_log = open(core_log_path, "wb")
    try:
        core = subprocess.Popen(
            [core_bin, "serve"],
            env={
                **os.environ,
                "HC_SESSION": session,
                "PATH": ":".join(part for part in path_parts if part),
            },
            stdout=core_log,
            stderr=core_log,
        )
    finally:
        core_log.close()
    deadline = time.monotonic() + CORE_SOCKET_SECONDS
    socket_path = pathlib.Path(runtime) / "herdr-core" / f"{session}.sock"
    while time.monotonic() < deadline:
        if core.poll() is not None:
            check(f"core {session} stayed up", False, f"exited during startup (log: {core_log_path})")
            return None
        if socket_path.exists():
            check(f"core {session} socket up", True, str(socket_path))
            return core
        time.sleep(0.2)
    check(f"core {session} socket up", False, f"never appeared within {CORE_SOCKET_SECONDS}s (log: {core_log_path})")
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


def survivor_pids() -> list[tuple[int, str]]:
    """Read-only scan of the sandbox's own PID namespace; nothing here
    is signaled. Any surviving chain or mesh process is recorded with
    its comm, never killed by pattern - the namespace teardown ends
    them."""
    found: list[tuple[int, str]] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
        except OSError:
            continue
        if comm in SURVIVOR_COMMS:
            found.append((int(entry.name), comm))
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
    # subprocesses end at their own pace once their parents are gone;
    # the scan settles briefly, then records whatever remains with its
    # comm - the namespace teardown stays the backstop
    deadline = time.monotonic() + SURVIVOR_SETTLE_SECONDS
    survivors = survivor_pids()
    while survivors and time.monotonic() < deadline:
        time.sleep(0.5)
        survivors = survivor_pids()
    detail = ", ".join(f"{pid}:{comm}" for pid, comm in survivors)
    check("driver chain, mesh and compositor fully torn down", not survivors, detail)


def main() -> int:
    if not run_canary_first():
        return 1

    core_bin = pinned("BOARD_VIEWER_HERDR_CORE_BIN")
    weston_bin = pinned("WESTON_BIN")
    shooter_bin = pinned("SCREENSHOOTER_BIN")
    tauri_driver_bin = pinned("TAURI_DRIVER_BIN")
    webkit_driver_bin = pinned("WEBKIT_DRIVER_BIN")
    app_bin = pinned("APP_BIN")
    via_py = pinned("BOARD_VIEWER_VIA_PY")
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

    # the client configuration declares the tracker machine (t-24@r12):
    # the deck's address rides the same bare-path rule, so the deck
    # document lands on machine b through core a's relay. Written before
    # the app launches; the config reader follows it at start.
    config_dir = pathlib.Path("config/dev.saccade.board-viewer")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(f'tracker-machine = "{MACHINE_NAME}"\n')
    check("client configuration written (tracker-machine b)", True, str(config_dir))

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
        detail = (err or out).decode(errors="replace")[:1500]
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

        # the diagnostic the budget ordered (record 3119): how the mesh
        # actually spawns on b - a bare name against PATH, the pinned
        # absolute path, and the spawner's whole environment. Evidence,
        # recorded verbatim beside the logs; none of it gates the
        # viewer assertions that follow. The readers resolve from the
        # env pin, not from which(): bwrap's --setenv overrides PATH
        # inside the sandbox, and the pin rides the inherited environ.
        coreutils_bin = pathlib.Path(pinned("BOARD_VIEWER_COREUTILS_BIN"))
        probes = [
            ("bare realpath on b", ["realpath", "--version"]),
            ("pinned absolute realpath on b", [str(coreutils_bin / "realpath"), "--version"]),
            ("the spawner's environment on b", [str(coreutils_bin / "env")]),
        ]
        for name, argv in probes:
            result = subprocess.run(
                [core_bin, "exec", "--on", MACHINE_NAME, "--json", *argv],
                env={**os.environ, "HC_SESSION": "a"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=MACHINE_CALL_SECONDS,
            )
            detail = (result.stdout or result.stderr).decode(errors="replace")
            (logs / "env-on-b.txt").open("a").write(f"=== {name} rc={result.returncode}\n{detail}\n")
            check(f"diagnostic: {name}", result.returncode == 0, detail[:300])

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
                env={
                    **os.environ,
                    "WAYLAND_DISPLAY": wayland_name,
                    # the lanes' test-open channel (t-24@r11): the
                    # launch argument left the human surface, so the
                    # machine-named location opens through the same
                    # model.open the inspection surface uses
                    "BOARD_VIEWER_TEST_OPEN": location,
                },
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
        session_id = client.new_session(app_bin, [])
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

        # figure outcomes are the product of real asset reads that land
        # when the whole hydration resolves, in authored order; poll
        # until both positions have outcomes (a miss is itself an
        # outcome), bounded, and assert on what arrived - addressed by
        # position, because a miss's message is the bridge's failure
        # line and may not name the path at all
        outcomes_deadline = time.monotonic() + FIGURE_OUTCOME_SECONDS
        real_outcome = figure_outcome_at(inspection, 0)
        escape_outcome = figure_outcome_at(inspection, 1)
        while (
            (real_outcome is None or escape_outcome is None)
            and time.monotonic() < outcomes_deadline
        ):
            time.sleep(0.5)
            try:
                inspection = client.execute(INSPECT_SCRIPT) or {}
            except WebDriverError as error:
                last_detail = str(error)[:120]
                continue
            real_outcome = figure_outcome_at(inspection, 0)
            escape_outcome = figure_outcome_at(inspection, 1)
        arrived = real_outcome
        check(
            "the real figure's bytes arrived through the mesh",
            arrived is not None
            and arrived.get("status") == "ok"
            and arrived.get("message") == REAL_FIGURE_PATH,
            f"outcome={json.dumps(arrived)[:200]}",
        )
        denial = figure_denial_at(inspection, 1)
        check(
            "the escape is refused by remote containment",
            denial is not None and CONTAINMENT_MESSAGE in denial,
            denial or f"outcome={json.dumps(escape_outcome)[:200]}",
        )

        frame = capture_compositor_frame(shooter_bin, wayland_name, shots, "remote")
        check("compositor frame captured", frame is not None, "screenshooter produced no frame")

        # -- the deck end-to-end (t-21, t-24@r21) ------------------------
        # The app offered its deck on machine b through the relay; the
        # lane now plays the agent: append the slide to the deck
        # document on the real doc store (core a's socket, on=b), and
        # the viewer's wake-driven feed picks the landing up and shows
        # the slide - a view whose path differs from the startup open,
        # so a passing check names the cue.
        DECK_LOCATION = "b:/board/deck-slide.yaml"

        def deck_state() -> dict | None:
            try:
                candidate = client.execute(
                    "return (window.__boardViewer && window.__boardViewer.deck()) || null;"
                )
            except WebDriverError as error:
                check("deck state readable", False, str(error)[:160])
                return None
            return candidate if isinstance(candidate, dict) else None

        deck: dict | None = None
        deck_deadline = time.monotonic() + APP_READY_SECONDS
        while time.monotonic() < deck_deadline:
            deck = deck_state()
            if deck is not None:
                break
            time.sleep(0.5)
        check(
            "the app offered its deck on the declared machine",
            deck is not None and str(deck.get("name", "")).startswith("deck-"),
            str(deck)[:200],
        )

        if deck is not None:
            deck_name = str(deck["name"])

            def rpc(method: str, params: dict) -> dict:
                sock = pathlib.Path(os.environ["XDG_RUNTIME_DIR"]) / "herdr-core" / "a.sock"
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock_fd:
                    sock_fd.settimeout(10)
                    sock_fd.connect(str(sock))
                    reader = sock_fd.makefile("rb")
                    writer = sock_fd.makefile("wb")
                    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                    writer.write((json.dumps(request) + "\n").encode())
                    writer.flush()
                    return json.loads(reader.readline())

            cue_landed = False
            last_detail = "no cue attempt"
            for attempt in range(2):
                read = rpc("doc.get", {"name": deck_name, "on": MACHINE_NAME})
                doc = read.get("result") or {}
                slides = doc.get("value") or []
                if DECK_LOCATION in slides:
                    cue_landed = True
                    last_detail = f"already cued at version {doc.get('version')}"
                    break
                written = rpc(
                    "doc.set",
                    {
                        "name": deck_name,
                        "on": MACHINE_NAME,
                        "version": doc.get("version", 0),
                        "value": [*slides, DECK_LOCATION],
                    },
                )
                if "error" in written:
                    last_detail = f"doc.set refused: {json.dumps(written['error'])[:160]}"
                    continue
                # bounded wait for the feed to carry the landing into a
                # shown slide; one re-cue is the protocol's own retry
                shown_deadline = time.monotonic() + CONNECT_WAIT_SECONDS
                while time.monotonic() < shown_deadline:
                    state_now = deck_state()
                    if state_now is not None and DECK_LOCATION in (state_now.get("slides") or []):
                        cue_landed = True
                        last_detail = f"slide shown at position {state_now.get('position')}"
                        break
                    time.sleep(0.5)
                if cue_landed:
                    break
            check(
                "the agent's cue lands in the deck document and is read back",
                cue_landed,
                last_detail,
            )

            # the slide is the shown view: a fresh inspection names the
            # cued location, not the startup open
            slide_shown = False
            shown_detail = "no fresh inspection"
            shown_deadline = time.monotonic() + CONNECT_WAIT_SECONDS
            while time.monotonic() < shown_deadline:
                try:
                    fresh = client.execute(INSPECT_SCRIPT)
                except WebDriverError as error:
                    shown_detail = str(error)[:160]
                    time.sleep(0.5)
                    continue
                resource = (fresh or {}).get("resource") or {}
                if (
                    resource.get("machine") == MACHINE_NAME
                    and resource.get("path") == "/board/deck-slide.yaml"
                    and resource.get("status") == "loaded"
                ):
                    slide_shown = True
                    shown_detail = str(resource)[:160]
                    break
                shown_detail = f"status={resource.get('status')!r} path={resource.get('path')!r}"
                time.sleep(0.5)
            check(
                "the cued slide is the shown view through the mesh",
                slide_shown,
                shown_detail,
            )

            hud = ""
            hud_deadline = time.monotonic() + 10
            while time.monotonic() < hud_deadline:
                hud = str(
                    client.execute(
                        "return document.querySelector('[data-control=\"deck:state\"]')?.textContent ?? '';"
                    )
                    or ""
                )
                if "slide 1/" in hud:
                    break
                time.sleep(0.5)
            check(
                "the HUD states the shown slide",
                "slide 1/1" in hud,
                hud,
            )

        frame = capture_compositor_frame(shooter_bin, wayland_name, shots, "remote-deck")
        check("deck compositor frame captured", frame is not None, "screenshooter produced no frame")

        # -- the explainer annotation write end-to-end (t-21, t-24@r21) --
        # A live bare-path board in the run root - writable, unlike the
        # read-only fixture binds - annotated through the real flow:
        # real input, the shown text's edit plan decoded before sending,
        # the whole-file re-read, one file.edit on machine b through the
        # declared machine, and the landed annotation rendered. Then the
        # stale guard: the file changes under the window, the next
        # submit refuses, the text is kept, and the note says so.
        live_path = pathlib.Path.cwd() / "live-explainer.yaml"
        live_path.write_text(
            "grammar: 1\n"
            "task: t-42\n"
            "content:\n"
            "  - text: |\n"
            "      the first part of the live explainer\n"
            "  - text: |\n"
            "      the second part of the live explainer\n"
        )

        def type_and_open(spec: str) -> None:
            perform_input(state.client, {"kind": "type", "target": "open:host-path", "text": spec})
            perform_input(state.client, {"kind": "click", "target": "open:host"})

        def inspection_bounded(predicate, timeout: float, what: str):
            deadline = time.monotonic() + timeout
            last = "no read"
            while time.monotonic() < deadline:
                try:
                    candidate = client.execute(INSPECT_SCRIPT)
                except WebDriverError as error:
                    last = str(error)[:160]
                    candidate = None
                got = candidate if isinstance(candidate, dict) else None
                if got is not None:
                    last = json.dumps(candidate)[:200]
                    if predicate(got):
                        return got
                time.sleep(0.5)
            check(what, False, last)
            return None

        type_and_open(str(live_path))
        opened = inspection_bounded(
            lambda item: (item.get("resource") or {}).get("machine") == "tracker-machine"
            and (item.get("resource") or {}).get("path") == str(live_path)
            and (item.get("resource") or {}).get("status") == "loaded",
            CONNECT_WAIT_SECONDS,
            "the live bare-path board opens through the declared machine",
        )
        editor_up = False

        if opened is not None:
            perform_input(state.client, {"kind": "key", "keys": ["a"]})
            editor_deadline = time.monotonic() + 10
            editor_up = False
            while time.monotonic() < editor_deadline:
                editor_up = (
                    client.execute(
                        "return document.querySelector('[data-control=\"draft:input\"]') !== null;"
                    )
                    is True
                )
                if editor_up:
                    break
                time.sleep(0.5)
            check("a on the explainer opens the draft editor", editor_up)

        if editor_up:
            perform_input(
                state.client,
                {"kind": "type", "target": "draft:input", "text": "a note from the lane"},
            )
            perform_input(state.client, {"kind": "hotkey", "keys": ["ctrl", "enter"]})

            written_deadline = time.monotonic() + CONNECT_WAIT_SECONDS
            written = False
            file_now = ""
            while time.monotonic() < written_deadline:
                file_now = live_path.read_text()
                if "annotation: |" in file_now and "a note from the lane" in file_now:
                    written = True
                    break
                time.sleep(0.5)
            check(
                "the submit wrote the annotation component into the file",
                written,
                file_now[:300],
            )

            landed = inspection_bounded(
                lambda item: any(
                    part.get("kind") == "annotation" for part in item.get("outline") or []
                ),
                CONNECT_WAIT_SECONDS,
                "the landed annotation renders in the reopened board",
            )
            check("landed annotation outline carries annotation", landed is not None)

            # the stale guard: the file changes under this window
            live_path.write_text(live_path.read_text() + "  - text: |\n      a later part\n")
            perform_input(state.client, {"kind": "key", "keys": ["a"]})
            editor_up = False
            editor_deadline = time.monotonic() + 10
            while time.monotonic() < editor_deadline:
                editor_up = (
                    client.execute(
                        "return document.querySelector('[data-control=\"draft:input\"]') !== null;"
                    )
                    is True
                )
                if editor_up:
                    break
                time.sleep(0.5)
            check("a after the write opens the editor again", editor_up)
            if editor_up:
                perform_input(
                    state.client,
                    {"kind": "type", "target": "draft:input", "text": "a second note"},
                )
                perform_input(state.client, {"kind": "hotkey", "keys": ["ctrl", "enter"]})
                refused_deadline = time.monotonic() + CONNECT_WAIT_SECONDS
                refused = False
                note = ""
                while time.monotonic() < refused_deadline:
                    note = str(
                        client.execute(
                            "return document.querySelector('[data-control=\"action:note\"]')?.textContent ?? '';"
                        )
                        or ""
                    )
                    if "file changed" in note:
                        refused = True
                        break
                    time.sleep(0.5)
                check(
                    "the changed file refuses the write and the note says so",
                    refused,
                    note[:200],
                )
                file_after = live_path.read_text()
                check(
                    "the refused write touched nothing",
                    "a second note" not in file_after,
                    file_after[-200:],
                )

                # the kept text rides: reopen the freshly read file and
                # press a - the editor opens with the kept text
                type_and_open(str(live_path))
                inspection_bounded(
                    lambda item: (item.get("resource") or {}).get("status") == "loaded",
                    CONNECT_WAIT_SECONDS,
                    "the file reopens after the refusal",
                )
                perform_input(state.client, {"kind": "key", "keys": ["a"]})
                kept_deadline = time.monotonic() + 10
                kept = ""
                while time.monotonic() < kept_deadline:
                    kept = str(
                        client.execute(
                            "return document.querySelector('[data-control=\"draft:input\"]')?.value ?? '';"
                        )
                        or ""
                    )
                    if kept:
                        break
                    time.sleep(0.5)
                check(
                    "the kept annotation text pre-fills the next editor",
                    kept == "a second note",
                    kept[:100],
                )

        frame = capture_compositor_frame(shooter_bin, wayland_name, shots, "remote-explainer")
        check("explainer compositor frame captured", frame is not None, "screenshooter produced no frame")

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
