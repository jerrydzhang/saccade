#!/usr/bin/env python3
"""The WebDriver session payload: runs INSIDE the enforced sandbox after
the containment canary has passed, before the private display, the
driver chain or the app exists. It holds the whole chain up for the
declared session lifetime and serves one command surface over a private
Unix socket in the run root - never a host-visible port.

Chain (all spawned here, all loopback inside the sandbox netns):
  private headless Weston (pixman, --fake-seat, kiosk-shell) - the app's
  display, exactly the native lane's compositor facts
  tauri-driver --native-driver WebKitWebDriver (pinned webkitgtk_4_1)
  -> WebKitWebDriver --port 4445 --host 127.0.0.1 (spawns the app)
  -> the packaged wrapper binary as tauri:options.application, opened
  on the explicit board, automation enabled by the env the driver chain
  injects (TAURI_WEBVIEW_AUTOMATION), software Mesa EGL inherited.

Command surface (one JSON request per connection, one response line):
  status | inspect | observe target | input click/type/key/wheel/drag/
  coords | shot name | down. Acting is real WebDriver input on elements
  addressed by their stable data-control ids; the inspection projection
  reports those targets. Raw coordinates exist only as the explicit
  coords escape hatch.

Teardown owns every process it can hold a handle on: the WebDriver
session is deleted first (WebKitWebDriver ends the app), then
tauri-driver is terminated through its handle (its signal handler kills
WebKitWebDriver), a /proc scan records any survivor as a failed check,
and Weston is terminated last. Nothing is signaled by pattern; the bwrap
PID-namespace teardown remains the backstop.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

HERE = "/lane"
sys.path.insert(0, "/canary")
sys.path.insert(0, HERE)

import containment_canary  # noqa: E402
import webdriver_client  # noqa: E402
from webdriver_client import WebDriver, WebDriverError  # noqa: E402

DRIVER_PORT = 4444
NATIVE_PORT = 4445
NATIVE_HOST = "127.0.0.1"
READY_SECONDS = 15
LISTEN_SECONDS = 90
APP_READY_SECONDS = 45
TARGET_ID = re.compile(r"^[A-Za-z0-9:_-]{1,120}$")
PORT_TARGET_ID = re.compile(r"^port:[A-Za-z0-9:_-]{1,114}$")
SHOT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    print(f"[webdriver] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}", flush=True)


def pinned(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.startswith("/"):
        print(f"containment refusal: {name} must be an absolute pinned path", file=sys.stderr)
        sys.exit(1)
    return value


def run_canary_first() -> bool:
    return containment_canary.main() == 0


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


def wait_listening(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect((NATIVE_HOST, port))
            return True
        except OSError:
            time.sleep(0.2)
        finally:
            probe.close()
    return False


def target_selector(target: str) -> str:
    if target.startswith("port:") and PORT_TARGET_ID.match(target):
        # the port's own control namespace (data-port-control), e.g.
        # port:picker-input - the same stable-id rule, port side
        return f'[data-port-control="{target[5:]}"]'
    if not TARGET_ID.match(target):
        raise WebDriverError("invalid argument", f"target id is not a stable data-control id: {target!r}")
    return f'[data-control="{target}"]'


def element_center(client: WebDriver, target: str) -> tuple[float, float]:
    rect = client.execute(
        "const el = document.querySelector(arguments[0]);"
        " if (!el) return null; const r = el.getBoundingClientRect();"
        " return {x: r.x, y: r.y, width: r.width, height: r.height};",
        [target_selector(target)],
    )
    if not isinstance(rect, dict) or rect.get("width", 0) <= 0 and rect.get("height", 0) <= 0:
        raise WebDriverError("no such element", f"target has no viewport rect: {target}")
    return float(rect["x"]) + float(rect.get("width", 0)) / 2, float(rect["y"]) + float(rect.get("height", 0)) / 2


# W3C execute/sync evaluates the script as a function BODY: statements
# with return, never a function expression (WebKit wraps the body and
# discards a bare arrow function, which reads as undefined)
OBSERVE_SCRIPT = (
    "const el = document.querySelector(arguments[0]);"
    " if (!el) return null;"
    " const props = arguments[1];"
    " const r = el.getBoundingClientRect();"
    " const cs = getComputedStyle(el);"
    " const style = {};"
    " for (const p of props || []) style[p] = cs.getPropertyValue(p);"
    " return {tag: el.tagName.toLowerCase(), rect: {x: r.x, y: r.y, width: r.width, height: r.height},"
    " transform: cs.transform, text: (el.textContent || '').trim(), style};"
)

INSPECT_SCRIPT = "return (window.__saccadeApp && window.__saccadeApp.inspection()) || null;"
# the port's observation surface (t-36 slice 1): the ready wait accepts
# either app flavor - the oracle (the shown board loaded) is the same
PORT_READY_SCRIPT = (
    "return JSON.stringify((typeof window.__saccadePort === 'object' && window.__saccadePort)"
    " ? window.__saccadePort.readout() : null);"
)

# R5 window surface: each suite webview knows its own tauri window
# label from the runtime's own metadata (read-only), and the shell's
# suite_windows command is the existing enumeration door. No product
# file is touched; the invoke calls ride the evaluate door.
WINDOW_LABEL_SCRIPT = (
    "return (window.__TAURI_INTERNALS__ && window.__TAURI_INTERNALS__.metadata"
    " && window.__TAURI_INTERNALS__.metadata.currentWindow)"
    " ? (window.__TAURI_INTERNALS__.metadata.currentWindow.label || null) : null;"
)
SUITE_WINDOWS_SCRIPT = (
    "return (window.__TAURI_INTERNALS__ && window.__TAURI_INTERNALS__.invoke)"
    " ? window.__TAURI_INTERNALS__.invoke('suite_windows') : null;"
)
OPEN_SUITE_WINDOW_SCRIPT = (
    "return (window.__TAURI_INTERNALS__ && window.__TAURI_INTERNALS__.invoke)"
    " ? window.__TAURI_INTERNALS__.invoke('open_suite_window', { stateId: arguments[0] })"
    " : null;"
)
FOCUS_SUITE_WINDOW_SCRIPT = (
    "return (window.__TAURI_INTERNALS__ && window.__TAURI_INTERNALS__.invoke)"
    " ? window.__TAURI_INTERNALS__.invoke('focus_suite_window', { label: arguments[0] })"
    " : null;"
)


def perform_input(client: WebDriver, request: dict) -> dict:
    """Real input only. click/type target an element through its stable
    id; wheel/drag anchor pointer or wheel sequences at the element's
    live center plus optional offsets; key acts on the focused element;
    coords is the explicit raw-coordinate escape hatch."""
    kind = request.get("kind")
    if kind == "rightclick":
        # a real context (right-button) click: element-addressed (the
        # S09 alternative invocation) or viewport-addressed (elements
        # without a data-control, e.g. group blocks - S13's group act)
        target = request.get("target")
        if isinstance(target, str) and target:
            selector = target_selector(target)
            x, y = element_center(client, target)
            client.perform_actions(webdriver_client.context_click_actions(x, y))
            return {"kind": "rightclick", "target": target, "x": x, "y": y}
        x = float(request.get("x", -1))
        y = float(request.get("y", -1))
        if not (0 <= x <= 4096 and 0 <= y <= 4096):
            raise WebDriverError("invalid argument", "rightclick needs a target id or x/y in viewport range")
        client.perform_actions(webdriver_client.context_click_actions(x, y))
        return {"kind": "rightclick", "x": x, "y": y}
    if kind == "hotkey":
        keys = request.get("keys")
        if not isinstance(keys, list) or len(keys) < 2 or not all(isinstance(k, str) and k for k in keys):
            raise WebDriverError("invalid argument", "hotkey input needs keys: [modifier..., key]")
        downs = [{"type": "keyDown", "value": webdriver_client.key_codepoint(name)} for name in keys]
        ups = [{"type": "keyUp", "value": webdriver_client.key_codepoint(name)} for name in reversed(keys)]
        client.perform_actions([{"type": "key", "id": "keyboard", "actions": downs + ups}])
        return {"kind": "hotkey", "keys": keys}
    if kind == "key":
        keys = request.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys):
            raise WebDriverError("invalid argument", "key input needs a nonempty keys list")
        client.perform_actions(webdriver_client.key_press_actions(keys))
        return {"kind": "key", "keys": keys}
    if kind == "coords":
        x = float(request.get("x", -1))
        y = float(request.get("y", -1))
        if not (0 <= x <= 4096 and 0 <= y <= 4096):
            raise WebDriverError("invalid argument", f"coords out of viewport range: {x},{y}")
        dx = float(request.get("dx", 0))
        dy = float(request.get("dy", 0))
        if dx or dy:
            client.perform_actions(webdriver_client.pointer_drag_actions(x, y, dx, dy))
            return {"kind": "drag", "x": x, "y": y, "dx": dx, "dy": dy}
        client.perform_actions(webdriver_client.pointer_click_actions(x, y))
        return {"kind": "click", "x": x, "y": y}

    target = request.get("target")
    if not isinstance(target, str):
        raise WebDriverError("invalid argument", f"{kind} input needs a target id")
    selector = target_selector(target)
    if kind == "click":
        dx = float(request.get("dx", 0))
        dy = float(request.get("dy", 0))
        if dx or dy:
            x, y = element_center(client, target)
            client.perform_actions(webdriver_client.pointer_drag_actions(x, y, 0, 0))
            return {"kind": "click", "target": target, "x": x, "y": y}
        client.click_element(client.find_element(selector))
        return {"kind": "click", "target": target}
    if kind == "type":
        text = request.get("text")
        if not isinstance(text, str):
            raise WebDriverError("invalid argument", "type input needs text")
        element = client.find_element(selector)
        client.clear_element(element)
        # an empty text IS the clear: WebKitWebDriver rejects an
        # element send-keys with an empty text ("Missing text
        # parameter"), observed live on the f447de1 differential run
        if text:
            client.send_keys(element, text)
        return {"kind": "type", "target": target, "text": text}
    if kind == "wheel":
        dx = float(request.get("dx", 0))
        dy = float(request.get("dy", 0))
        if dx == 0 and dy == 0:
            raise WebDriverError("invalid argument", "wheel input needs dx or dy")
        x, y = element_center(client, target)
        client.perform_actions(webdriver_client.wheel_scroll_actions(x, y, dx, dy))
        return {"kind": "wheel", "target": target, "dx": dx, "dy": dy}
    if kind == "drag":
        dx = float(request.get("dx", 0))
        dy = float(request.get("dy", 0))
        if dx == 0 and dy == 0:
            raise WebDriverError("invalid argument", "drag input needs dx or dy")
        x, y = element_center(client, target)
        client.perform_actions(webdriver_client.pointer_drag_actions(x, y, dx, dy))
        return {"kind": "drag", "target": target, "dx": dx, "dy": dy}
    raise WebDriverError("invalid argument", f"unknown input kind: {kind!r}")


def open_via_picker(client: WebDriver, spec: str, timeout: float = 10.0) -> None:
    """Open a location the way the human does since the strip replaced
    the fields (t-34@r6 change 5): a click on the strip leaves any text
    field, space space opens the picker, the location is typed, Enter
    opens it. Real input throughout; the load is the caller's to wait
    for."""
    perform_input(client, {"kind": "click", "target": "strip"})
    perform_input(client, {"kind": "key", "keys": ["escape"]})
    perform_input(client, {"kind": "key", "keys": [" "]})
    perform_input(client, {"kind": "key", "keys": [" "]})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.execute("return document.querySelector('[data-control=\"picker:input\"]') !== null;"):
            break
        time.sleep(0.2)
    else:
        raise WebDriverError("no such element", "the picker did not open on space space")
    perform_input(client, {"kind": "type", "target": "picker:input", "text": spec})
    perform_input(client, {"kind": "key", "keys": ["enter"]})


def capture_compositor_frame(shooter_bin: str, wayland_name: str, shots: pathlib.Path, name: str) -> str | None:
    """weston-screenshooter 16 takes no output argument and writes into
    its CWD, so the capture runs in the shots dir and the lane collects
    the new files (the native lane's capture facts)."""
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
        return None
    if result.returncode != 0:
        return None
    new_frames = sorted(set(shots.glob("wayland-screenshot-*.png")) - before)
    if not new_frames:
        return None
    target = shots / f"{name}-compositor.png"
    new_frames[0].replace(target)
    for extra in new_frames[1:]:
        extra.unlink()
    return str(target)


SURVIVOR_COMMS = {
    "WebKitWebDriver",
    "WebKitWebProces",
    "WebKitNetworkPro",
    ".saccade-board-",
    "saccade-board-v",
}


def survivor_pids() -> list[int]:
    """Read-only scan of the sandbox's own PID namespace; nothing here
    is signaled. Any surviving chain process is recorded, never killed
    by pattern - the namespace teardown ends them with the sandbox."""
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


class SessionState:
    def __init__(self) -> None:
        self.weston: subprocess.Popen[bytes] | None = None
        self.tauri_driver: subprocess.Popen[bytes] | None = None
        self.client: WebDriver | None = None
        self.wayland_name = ""
        self.shooter_bin = ""
        self.shots = pathlib.Path(".")
        self.board = ""
        self.expires_epoch = 0.0
        self.torn_down = False
        # the tracker sequence's own server: spawned, fed, stopped and
        # re-checked inside one bounded run; never adopted, never
        # outliving the sandbox
        self.server: subprocess.Popen[bytes] | None = None
        self.server_port = 0
        # the t36 lane mode's own machinery: the feed proxy on the
        # core socket path (started before the app spawns) and the
        # in-sandbox seed server the tracker checks open locations
        # against
        self.feed_proxy: object | None = None
        self.seed_server: object | None = None
        self.tracker_base = ""
        self.tracker_tasks: dict = {}

    def chain_status(self) -> dict:
        return {
            "weston": bool(self.weston and self.weston.poll() is None),
            "tauri_driver": bool(self.tauri_driver and self.tauri_driver.poll() is None),
            "webdriver_session": bool(self.client and self.client.session),
            "app_alive": bool(survivor_pids()),
        }


def enumerate_windows(client: WebDriver) -> list[dict]:
    """Every webview handle with the tauri window label it carries, read
    from the page's own runtime metadata. The original handle is always
    restored; a handle whose page cannot answer labels a null."""
    original = client.current_window()
    windows: list[dict] = []
    try:
        for handle in client.window_handles():
            client.switch_to_window(handle)
            try:
                label = client.execute(WINDOW_LABEL_SCRIPT)
            except WebDriverError:
                label = None
            windows.append({"handle": handle, "label": label if isinstance(label, str) else None})
    finally:
        client.switch_to_window(original)
    return windows


def window_label_of(inventory: list[dict], label: str) -> str | None:
    for entry in inventory:
        if entry.get("label") == label:
            return str(entry.get("handle"))
    return None


def invoke_suite_window(client: WebDriver, script: str, args: list) -> object:
    """One read/act door through the shell's existing tauri commands.
    The answer is whatever the invoke resolves to; a refused command
    surfaces as the evaluate error, never a fabricated ok."""
    return client.execute(script, args)


def teardown(state: SessionState) -> int:
    """Owned-handle teardown in dependency order: session delete (the
    remote end ends the app), then the driver chain through our handles,
    then the compositor. Escalation is TERM -> bounded grace -> KILL on
    the held handles only."""
    if state.torn_down:
        return 0
    state.torn_down = True
    if state.client and state.client.session:
        try:
            state.client.delete_session()
        except WebDriverError as error:
            check("webdriver session deleted", False, str(error))
        else:
            check("webdriver session deleted", True)
    if state.tauri_driver is not None and state.tauri_driver.poll() is None:
        state.tauri_driver.terminate()
        try:
            state.tauri_driver.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.tauri_driver.kill()
            state.tauri_driver.wait(timeout=5)
    # a bounded settle wait before the survivor scan: the remote end's
    # kill of the chain races nothing we own, but the process table
    # itself needs a moment to reflect the exits
    deadline = time.monotonic() + 3.0
    survivors = survivor_pids()
    while survivors and time.monotonic() < deadline:
        time.sleep(0.2)
        survivors = survivor_pids()
    check("driver chain and app fully torn down", not survivors, f"survivor pids: {survivors}")
    if state.weston is not None and state.weston.poll() is None:
        state.weston.terminate()
        try:
            state.weston.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.weston.kill()
            state.weston.wait(timeout=5)
    if state.seed_server is not None:
        seed_stop = getattr(state.seed_server, "stop", None)
        if callable(seed_stop):
            seed_stop()
    if state.feed_proxy is not None:
        proxy_stop = getattr(state.feed_proxy, "stop", None)
        if callable(proxy_stop):
            proxy_stop()
    if state.server is not None and state.server.poll() is None:
        state.server.terminate()
        try:
            state.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.server.kill()
            state.server.wait(timeout=5)
    return 0


def handle_request(state: SessionState, request: dict) -> tuple[dict, bool]:
    """One decoded request to one response dict; the second item stops
    the session after the response is written (down)."""
    command = request.get("cmd")
    if command == "status":
        return {
            "ok": True,
            "board": state.board,
            "expires": state.expires_epoch,
            "chain": state.chain_status(),
        }, False
    if command == "inspect":
        assert state.client is not None
        inspection = state.client.execute(INSPECT_SCRIPT)
        if not isinstance(inspection, dict):
            raise WebDriverError("unknown error", "inspection projection unavailable")
        return {"ok": True, "inspection": inspection}, False
    if command == "observe":
        assert state.client is not None
        target = request.get("target")
        if not isinstance(target, str):
            raise WebDriverError("invalid argument", "observe needs a target id")
        props = request.get("props")
        if props is not None and not (isinstance(props, list) and all(isinstance(p, str) for p in props)):
            raise WebDriverError("invalid argument", "props must be a list of CSS property names")
        observed = state.client.execute(OBSERVE_SCRIPT, [target_selector(target), props or []])
        return {"ok": True, "target": target, "observed": observed}, False
    if command == "input":
        assert state.client is not None
        return {"ok": True, "input": perform_input(state.client, request)}, False
    if command == "evaluate":
        assert state.client is not None
        script = request.get("script")
        if not isinstance(script, str) or not script.strip():
            raise WebDriverError("invalid argument", "evaluate needs a script")
        args = request.get("args")
        if args is not None and not (isinstance(args, list) and all(isinstance(a, (str, int, float)) for a in args)):
            raise WebDriverError("invalid argument", "evaluate args must be scalars")
        value = state.client.execute(script, args or [])
        return {"ok": True, "value": value}, False
    if command == "selection":
        assert state.client is not None
        chosen = state.client.execute("return String(window.getSelection())")
        return {"ok": True, "selection": chosen}, False
    if command == "windows":
        assert state.client is not None
        inventory = enumerate_windows(state.client)
        suite = invoke_suite_window(state.client, SUITE_WINDOWS_SCRIPT, [])
        return {"ok": True, "windows": inventory, "suite": suite}, False
    if command == "feed":
        proxy = state.feed_proxy
        if proxy is None or not getattr(proxy, "started", False):
            raise WebDriverError("invalid argument", "no feed proxy in this session (t36 lane mode only)")
        action = request.get("action")
        if action == "status":
            return {"ok": True, "feed": proxy.recordings()}, False
        if action == "inject-status":
            delivered = proxy.inject_status(
                str(request.get("terminal") or "t-1"),
                state=str(request.get("state") or "working"),
                app=str(request.get("app") or "pi"),
                msg=str(request.get("msg") or ""),
                progress=request.get("progress"),
                title=request.get("title"),
            )
            return {"ok": True, "delivered": delivered}, False
        if action == "inject-unknown":
            delivered = proxy.inject_unknown(str(request.get("event_type") or "deck-cue"))
            return {"ok": True, "delivered": delivered}, False
        if action == "inject-finished":
            delivered = proxy.inject_finished(
                str(request.get("terminal") or "t-1"), reason=str(request.get("reason") or "exited")
            )
            return {"ok": True, "delivered": delivered}, False
        raise WebDriverError("invalid argument", f"unknown feed action: {action!r}")
    if command == "tracker":
        if not state.tracker_base:
            raise WebDriverError("invalid argument", "no seed server in this session (t36 lane mode only)")
        return {"ok": True, "base": state.tracker_base, "tasks": state.tracker_tasks}, False
    if command == "tracker-commit":
        # S15's mid-check commit through the real wire: the payload
        # posts to its own seed server (the server lives in this
        # sandbox's netns; the host cannot reach it). The commit shape
        # is the seeds module's - one revised board.yaml, real wire.
        import t36_seeds

        if not state.tracker_base or state.seed_server is None:
            raise WebDriverError("invalid argument", "no seed server in this session (t36 lane mode only)")
        task = request.get("task")
        if not isinstance(task, int):
            raise WebDriverError("invalid argument", "tracker-commit needs the task number")
        board_text = request.get("board")
        if not isinstance(board_text, str) or "grammar:" not in board_text:
            raise WebDriverError("invalid argument", "tracker-commit needs a board.yaml text")
        # a wire refusal is a per-request answer, never a payload exit:
        # an escaping SeedRefused would tear the whole session down
        try:
            reply = state.seed_server.post("/api/v1/boards/commit", {
                "context": t36_seeds.SEED_CONTEXT,
                "task": task,
                "files": {"files": [{"path": "board.yaml",
                                     "content": base64.b64encode(board_text.encode("utf-8")).decode("ascii")}]},
                "message": str(request.get("message") or "t36 mid-check commit"),
                "client": t36_seeds.SEED_CLIENT,
            })
        except t36_seeds.SeedRefused as error:
            raise WebDriverError("unknown error", str(error)[:300]) from None
        return {"ok": True, "revision": reply.get("revision"), "commit": reply.get("commit") or reply.get("hash")}, False
    if command == "open-window":
        assert state.client is not None
        state_id = request.get("state_id")
        if state_id is not None and not isinstance(state_id, str):
            raise WebDriverError("invalid argument", "open-window needs an optional string state_id")
        label = invoke_suite_window(state.client, OPEN_SUITE_WINDOW_SCRIPT, [state_id])
        if not isinstance(label, str) or not label:
            raise WebDriverError("unknown error", f"open_suite_window answered {label!r}")
        return {"ok": True, "label": label, "windows": enumerate_windows(state.client)}, False
    if command == "focus-window":
        assert state.client is not None
        label = request.get("label")
        if not isinstance(label, str) or not label:
            raise WebDriverError("invalid argument", "focus-window needs a string label")
        invoke_suite_window(state.client, FOCUS_SUITE_WINDOW_SCRIPT, [label])
        return {"ok": True, "focused": label}, False
    if command == "observe-window":
        assert state.client is not None
        label = request.get("window")
        target = request.get("target")
        if not isinstance(label, str) or not label:
            raise WebDriverError("invalid argument", "observe-window needs a string window label")
        if not isinstance(target, str):
            raise WebDriverError("invalid argument", "observe-window needs a target id")
        props = request.get("props")
        if props is not None and not (isinstance(props, list) and all(isinstance(p, str) for p in props)):
            raise WebDriverError("invalid argument", "props must be a list of CSS property names")
        inventory = enumerate_windows(state.client)
        handle = window_label_of(inventory, label)
        if handle is None:
            raise WebDriverError("no such window", f"no webview carries label {label!r}: {inventory}")
        original = state.client.current_window()
        state.client.switch_to_window(handle)
        try:
            observed = state.client.execute(OBSERVE_SCRIPT, [target_selector(target), props or []])
        finally:
            state.client.switch_to_window(original)
        return {"ok": True, "window": label, "target": target, "observed": observed}, False
    if command == "evaluate-window":
        assert state.client is not None
        label = request.get("window")
        script = request.get("script")
        if not isinstance(label, str) or not label:
            raise WebDriverError("invalid argument", "evaluate-window needs a string window label")
        if not isinstance(script, str) or not script.strip():
            raise WebDriverError("invalid argument", "evaluate-window needs a script")
        args = request.get("args")
        if args is not None and not (isinstance(args, list) and all(isinstance(a, (str, int, float)) for a in args)):
            raise WebDriverError("invalid argument", "evaluate-window args must be scalars")
        inventory = enumerate_windows(state.client)
        handle = window_label_of(inventory, label)
        if handle is None:
            raise WebDriverError("no such window", f"no webview carries label {label!r}: {inventory}")
        original = state.client.current_window()
        state.client.switch_to_window(handle)
        try:
            value = state.client.execute(script, args or [])
        finally:
            state.client.switch_to_window(original)
        return {"ok": True, "window": label, "value": value}, False
    if command == "shot":
        assert state.client is not None
        name = request.get("name")
        if not isinstance(name, str) or not SHOT_NAME.match(name):
            raise WebDriverError("invalid argument", f"shot name must match {SHOT_NAME.pattern}: {name!r}")
        png = state.client.screenshot()
        web_path = state.shots / f"{name}.png"
        web_path.write_bytes(png)
        compositor = capture_compositor_frame(state.shooter_bin, state.wayland_name, state.shots, name)
        return {"ok": True, "web": str(web_path), "compositor": compositor}, False
    if command == "down":
        return {"ok": True, "stopping": True}, True
    raise WebDriverError("invalid argument", f"unknown command: {command!r}")


def serve(state: SessionState, socket_path: str, lifetime: float, ready_file: str) -> int:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale = pathlib.Path(socket_path)
    if stale.exists():
        stale.unlink()
    server.bind(socket_path)
    server.listen(4)
    deadline = time.monotonic() + lifetime
    state.expires_epoch = time.time() + lifetime
    pathlib.Path(ready_file).write_text(
        json.dumps({"ready": True, "socket": socket_path, "expires": state.expires_epoch}) + "\n"
    )
    print(f"[webdriver] session up; control socket {socket_path}; lifetime {lifetime:.0f}s", flush=True)
    stopped = False
    while not stopped:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"[webdriver] session lifetime bound reached ({lifetime:.0f}s); tearing down", flush=True)
            break
        server.settimeout(min(remaining, 0.5))
        try:
            conn, _ = server.accept()
        except TimeoutError:
            continue
        with conn:
            conn.settimeout(30)
            try:
                chunks: list[bytes] = []
                total = 0
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    total += len(data)
                    if total > 1 << 20:
                        raise WebDriverError("invalid argument", "request exceeds 1 MiB")
                    chunks.append(data)
                    if b"\n" in data:
                        break
                line = b"".join(chunks).split(b"\n", 1)[0]
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise WebDriverError("invalid argument", "request must be a JSON object")
                response, stop = handle_request(state, request)
            except (WebDriverError, ValueError, OSError) as error:
                response = {"ok": False, "error": str(error)[:300]}
                stop = False
            payload = (json.dumps(response) + "\n").encode("utf-8")
            try:
                conn.sendall(payload)
            except OSError:
                pass
        stopped = stop
        if stopped:
            print("[webdriver] down requested; tearing down", flush=True)
    server.close()
    pathlib.Path(socket_path).unlink(missing_ok=True)
    return teardown(state)


def tracker_free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind((NATIVE_HOST, 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def tracker_post(base: str, route: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"{route} answered {error.code}: {detail}") from None


TRACKER_CITED_BOARD = """grammar: 1
task: t-0
content:
  - text: the seed account a citation points at
"""

TRACKER_CITING_BOARD = """grammar: 1
task: t-1
content:
  - text: the citing account, served by the real wire
  - incorporation:
      source:
        task: t-0
        revision: 1
        select: [/content/0]
      content:
        - text: the receiving text for the cited seed
"""


def tracker_finish(state: SessionState) -> int:
    """Teardown, then the lane's verdict: any failed check fails the run."""
    teardown(state)
    return 1 if any(not ok for ok, _name, _detail in RESULTS) else 0


def t36_prelude(state: SessionState, logs: pathlib.Path) -> None:
    """The t36 lane mode's setup, BEFORE the app spawns: the feed
    proxy binds the core socket path (the app dials it at startup),
    then a real sac serve on a throwaway db carries the group-made
    seed boards the tracker checks open. Every step is a recorded
    check, teed to logs/t36-prelude.log so the driver's gate-TAP
    merge can parse them; a failure here leaves the chain down and
    the verdict red."""
    import t36_feed_proxy
    import t36_seeds

    prelude_log_path = logs / "t36-prelude.log"

    def gate(name: str, condition: bool, detail: str = "") -> None:
        check(name, condition, detail)
        line = f"[webdriver] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}".rstrip() + "\n"
        with open(prelude_log_path, "a", encoding="utf-8") as log:
            log.write(line)

    serve_bin = os.environ.get("SAC_APP_SERVE_BIN", "")
    if not serve_bin.startswith("/"):
        print("containment refusal: SAC_APP_SERVE_BIN must be an absolute sandbox path", file=sys.stderr)
        raise SystemExit(1)

    proxy = t36_feed_proxy.FeedProxy(t36_feed_proxy.default_socket_path())
    proxy.start()
    state.feed_proxy = proxy
    gate("the feed proxy listens on the core socket path", proxy.started, t36_feed_proxy.default_socket_path())

    # the server shells out to git, which the scrubbed env would never
    # find: the pinned git rides /lane beside the server
    os.environ["PATH"] = "/lane:" + os.environ.get("PATH", "/usr/bin:/bin")
    root = state.shots.parent / "seed"
    try:
        server = t36_seeds.SeedServer(serve_bin, root)
        server.start()
    except t36_seeds.SeedRefused as error:
        gate("the t36 seed server answers", False, str(error)[:200])
        teardown(state)
        raise SystemExit(1)
    state.seed_server = server
    state.tracker_base = server.base
    gate("the t36 seed server answers", server.process is not None and server.process.poll() is None, server.base)
    try:
        commits = server.seed()
        state.tracker_tasks = {entry["task"]: entry for entry in commits}
        gate(
            "the t36 seed boards committed (groups on the real wire)",
            len(commits) == 5 and all(entry.get("revision") is not None for entry in commits),
            json.dumps([[entry["task"], entry.get("revision")] for entry in commits]),
        )
    except t36_seeds.SeedRefused as error:
        gate("the t36 seed boards committed (groups on the real wire)", False, str(error)[:200])
        teardown(state)
        raise SystemExit(1)


def run_tracker_sequence(state: SessionState) -> int:
    """The scripted tracker sequence: a real `sac serve` on a throwaway
    db inside this sandbox, two boards committed on the real wire, the
    packaged app driven by WebDriver input through the tracker UI (the
    server setting typed and saved, the location opened), the citation
    navigated, and a stopped server answered by a structured refusal in
    bounded time - never an endless load. Runs under the real CSP."""
    assert state.client is not None
    serve_bin = os.environ.get("SAC_APP_SERVE_BIN", "")
    if not serve_bin.startswith("/"):
        print("containment refusal: SAC_APP_SERVE_BIN must be an absolute sandbox path", file=sys.stderr)
        teardown(state)
        return 1
    root = state.shots.parent / "tracker"
    # the tracker's name is the repo directory's basename: the location
    # spec names it, so the dir carries the name the spec uses
    (root / "lanetracker").mkdir(parents=True, exist_ok=True)
    db = root / "lanetracker.db"
    port = tracker_free_port()
    server_log_path = state.shots.parent / "logs" / "tracker-serve.log"
    server_log = open(server_log_path, "wb")
    try:
        state.server = subprocess.Popen(
            [serve_bin, "serve", "--db", str(db), "--repo", str(root / "lanetracker"),
             "--bind", NATIVE_HOST, "--port", str(port)],
            # the server shells out to git, which the scrubbed env would
            # never find: the pinned git rides /lane beside the server
            env={**os.environ, "PATH": "/lane:" + os.environ.get("PATH", "/usr/bin:/bin")},
            stdout=server_log,
            stderr=server_log,
        )
    finally:
        server_log.close()
    state.server_port = port
    base = f"http://{NATIVE_HOST}:{port}"
    deadline = time.monotonic() + 20
    ready = False
    while time.monotonic() < deadline:
        if state.server.poll() is not None:
            break
        try:
            with urllib.request.urlopen(base + "/api/v1/version", timeout=2) as reply:
                ready = reply.status == 200
                break
        except urllib.error.HTTPError:
            # any typed answer means the wire is up; a wrong status is
            # still an answer
            ready = True
            break
        except OSError:
            time.sleep(0.3)
    check("a real sac serve answers on the sandbox loopback", ready, f"log: {server_log_path}")
    if not ready:
        return tracker_finish(state)

    # two boards on the real wire: t-0 the cited plain account, t-1
    # citing it by revision - the citation the sequence navigates
    context = {"actor": "lane", "tier": "agent", "scope": None}
    ids: list[int] = []
    try:
        for name in ("the cited seed account", "the citing account"):
            reply = tracker_post(base, "/api/v1/command", {
                "context": context,
                "command": {"create_task": {"name": name}},
                "at": None,
                "client": "lane",
            })
            ids.append(int(str(reply["id"]).removeprefix("t-")))
        for task, board_text in ((ids[0], TRACKER_CITED_BOARD), (ids[1], TRACKER_CITING_BOARD)):
            reply = tracker_post(base, "/api/v1/boards/commit", {
                "context": context,
                "task": task,
                "files": {"files": [{"path": "board.yaml",
                                     "content": base64.b64encode(board_text.encode("utf-8")).decode("ascii")}]},
                "message": "the lane's seed commit",
                "client": "lane",
            })
            check(f"t-{task}'s seed board committed at revision 1", reply.get("revision") == 1, str(reply)[:160])
    except (OSError, ValueError, KeyError) as error:
        check("two boards committed on the real wire", False, str(error)[:200])
        return tracker_finish(state)

    def inspect_bounded(predicate, timeout: float):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                last = state.client.execute(INSPECT_SCRIPT)
            except WebDriverError as error:
                last = {"error": str(error)[:160]}
            if isinstance(last, dict) and predicate(last):
                return last
            time.sleep(0.4)
        return last

    # the human path: type the server address once, then open the
    # tracker-named location; every stroke is real WebDriver input
    try:
        # the first open names the host: it reaches the tracker and
        # remembers it (t-34@r6 change 5)
        open_via_picker(state.client, f"{NATIVE_HOST}:{port}/lanetracker/t-{ids[1]}")
    except WebDriverError as error:
        check("the tracker UI takes real input", False, str(error)[:200])
        return tracker_finish(state)
    check("the tracker UI takes real input", True)
    loaded = inspect_bounded(
        lambda i: i.get("resource", {}).get("status") == "loaded"
        and i.get("identity", {}).get("task") == f"t-{ids[1]}"
        and i.get("resource", {}).get("machine") == "tracker:lanetracker",
        30.0,
    )
    ok = isinstance(loaded, dict) and loaded.get("resource", {}).get("status") == "loaded"
    check("a tracker location loads through the real server under the real CSP", bool(ok), str(loaded)[:300])
    if not ok:
        return tracker_finish(state)

    # the annotation act on a task board (t-17, re-pinned here because
    # t-21's lanes caught the act-part walk matching nothing): a opens
    # the draft editor, its cancel closes it
    try:
        perform_input(state.client, {"kind": "key", "keys": ["a"]})
    except WebDriverError as error:
        check("a on a task board opens the draft editor", False, str(error)[:200])
        return tracker_finish(state)
    editor_up = False
    editor_deadline = time.monotonic() + 10
    while time.monotonic() < editor_deadline:
        editor_up = (
            state.client.execute(
                "return document.querySelector('[data-control=\"draft:input\"]') !== null;"
            )
            is True
        )
        if editor_up:
            break
        time.sleep(0.5)
    check("a on a task board opens the draft editor", editor_up)
    if editor_up:
        try:
            perform_input(state.client, {"kind": "click", "target": "draft:cancel"})
        except WebDriverError as error:
            check("the draft editor's cancel closes it", False, str(error)[:200])
        editor_closed = True
        closed_deadline = time.monotonic() + 5
        while time.monotonic() < closed_deadline:
            editor_closed = (
                state.client.execute(
                    "return document.querySelector('[data-control=\"draft:input\"]') === null;"
                )
                is True
            )
            if editor_closed:
                break
            time.sleep(0.3)
        check("the draft editor's cancel closes it", editor_closed)

    cited_ok = False
    cited = inspect_bounded(lambda i: "board:open-citation" in [t.get("id", "") for t in i.get("targets", [])], 10.0)
    if isinstance(cited, dict) and "board:open-citation" in [t.get("id", "") for t in cited.get("targets", [])]:
        try:
            perform_input(state.client, {"kind": "click", "target": "board:open-citation"})
            navigated = inspect_bounded(
                lambda i: i.get("resource", {}).get("status") == "loaded"
                and i.get("identity", {}).get("task") == f"t-{ids[0]}"
                and i.get("resource", {}).get("path") == f"t-{ids[0]}@r1",
                15.0,
            )
            cited_ok = isinstance(navigated, dict) and navigated.get("resource", {}).get("path") == f"t-{ids[0]}@r1"
        except WebDriverError as error:
            check("the citation opens the cited revision", False, str(error)[:200])
    check("the citation opens the cited revision through the real wire", bool(cited_ok))

    # the landing-on-a-real-server pin: t-2 first holds text only; the
    # figure commit lands; a re-open must render the committed figure
    # from the new revision - never a stale tree or a missing file
    landing_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    landing_r1 = (
        "grammar: 1\ntask: t-{}\ncontent:\n  - text: the landing board, first pass"
    )
    landing_r2 = (
        "grammar: 1\ntask: t-{}\ncontent:\n  - text: the landing board, revised\n"
        "  - figure:\n      input: figures/landing.png\n      caption: the landing figure, committed\n"
    )
    try:
        reply = tracker_post(base, "/api/v1/command", {
            "context": {"actor": "lane", "tier": "agent", "scope": None},
            "command": {"create_task": {"name": "the landing board"}},
            "at": None,
            "client": "lane",
        })
        landing_task = int(str(reply["id"]).removeprefix("t-"))
        reply = tracker_post(base, "/api/v1/boards/commit", {
            "context": {"actor": "lane", "tier": "agent", "scope": None},
            "task": landing_task,
            "files": {"files": [{"path": "board.yaml",
                                 "content": base64.b64encode(landing_r1.format(landing_task).encode("utf-8")).decode("ascii")}]},
            "message": "the landing seed, text only",
            "client": "lane",
        })
        check(
            f"t-{landing_task}'s landing seed commits at revision 1 (text only)",
            reply.get("revision") == 1,
            str(reply)[:160],
        )
        open_via_picker(state.client, f"lanetracker/t-{landing_task}")
        loaded_seed = inspect_bounded(
            lambda i: i.get("resource", {}).get("status") == "loaded",
            20.0,
        )
        check("the landing seed opens at revision 1", bool(
            isinstance(loaded_seed, dict) and loaded_seed.get("resource", {}).get("status") == "loaded"
        ))
        reply = tracker_post(base, "/api/v1/boards/commit", {
            "context": {"actor": "lane", "tier": "agent", "scope": None},
            "task": landing_task,
            "files": {"files": [
                {"path": "board.yaml",
                 "content": base64.b64encode(landing_r2.format(landing_task).encode("utf-8")).decode("ascii")},
                {"path": "figures/landing.png", "content": landing_png},
            ]},
            "message": "the landing board, revised with the figure",
            "client": "lane",
        })
        check(
            f"t-{landing_task}'s figure commit lands at revision 2",
            reply.get("revision") == 2,
            str(reply)[:160],
        )
        open_via_picker(state.client, f"lanetracker/t-{landing_task}")
        reopened = inspect_bounded(
            lambda i: i.get("resource", {}).get("status") == "loaded"
            and i.get("identity", {}).get("task") == f"t-{landing_task}"
            and any(o.get("kind") == "figure" for o in i.get("renderOutcomes", [])),
            20.0,
        )
        outcomes = reopened.get("renderOutcomes", []) if isinstance(reopened, dict) else []
        missing = [o for o in outcomes if o.get("kind") == "figure" and o.get("status") == "missing"]
        ok = [o for o in outcomes if o.get("kind") == "figure" and o.get("status") == "ok"]
        check(
            "a landed figure renders on re-open: no stale tree, no missing file",
            missing == [] and len(ok) == 1,
            str(outcomes)[:200],
        )
    except WebDriverError as error:
        check("the landing-on-a-real-server pin runs", False, str(error)[:200])
    try:
        png = state.client.screenshot()
        (state.shots / "tracker-real-server.png").write_bytes(png)
    except WebDriverError:
        pass

    # the server dies: the next open is a structured refusal in bounded
    # time (the request bound plus slack), never an endless load
    if state.server.poll() is None:
        state.server.terminate()
        try:
            state.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.server.kill()
            state.server.wait(timeout=5)
    check("the tracker server stopped", state.server.poll() is not None)
    stopped_at = time.monotonic()
    try:
        open_via_picker(state.client, f"lanetracker/t-{ids[1]}")
    except WebDriverError as error:
        check("the stopped-server open takes input", False, str(error)[:200])
    refused = inspect_bounded(
        lambda i: i.get("resource", {}).get("status") == "invalid" and bool(i.get("errors")),
        20.0,
    )
    bounded = (
        isinstance(refused, dict)
        and refused.get("resource", {}).get("status") == "invalid"
        and (time.monotonic() - stopped_at) < 25.0
    )
    check("a stopped server refuses in bounded time, never endless loading", bool(bounded), str(refused)[:300])
    return tracker_finish(state)


def main() -> int:
    if not run_canary_first():
        return 1

    weston_bin = pinned("WESTON_BIN")
    shooter_bin = pinned("SCREENSHOOTER_BIN")
    tauri_driver_bin = pinned("TAURI_DRIVER_BIN")
    webkit_driver_bin = pinned("WEBKIT_DRIVER_BIN")
    app_bin = pinned("APP_BIN")
    board = pinned("BOARD_FILE")
    shots = pathlib.Path(pinned("SHOTS_DIR"))
    socket_path = pinned("SESSION_CONTROL_SOCKET")
    ready_file = pinned("SESSION_READY_FILE")
    lifetime = float(os.environ.get("SESSION_LIFETIME_SECONDS", "1800"))
    if lifetime <= 0 or lifetime > 4 * 3600:
        print("containment refusal: SESSION_LIFETIME_SECONDS must be within (0, 4h]", file=sys.stderr)
        return 1
    if not shots.is_dir():
        print("containment refusal: SHOTS_DIR must be an existing run-root directory", file=sys.stderr)
        return 1
    logs = shots.parent / "logs"
    logs.mkdir(exist_ok=True)

    started = time.monotonic()
    state = SessionState()
    state.shooter_bin = shooter_bin
    state.shots = shots
    state.board = board
    state.wayland_name = f"wayland-{os.getpid() % 97}"
    try:
        # the t36 prelude runs BEFORE anything spawns: the feed proxy
        # must hold the core socket path before the app dials it, and
        # the seed server must answer before a tracker location opens
        if os.environ.get("SESSION_MODE") == "t36":
            t36_prelude(state, logs)
        weston = start_weston(weston_bin, state.wayland_name, logs)
        if weston is None:
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
                # the app is spawned by WebKitWebDriver, not by us: its
                # display name must ride the chain env (launcher
                # inheritance), the way the native lane hands it to the
                # app directly; the test-open channel rides with it
                env={
                    **os.environ,
                    "WAYLAND_DISPLAY": state.wayland_name,
                    "SAC_APP_TEST_OPEN": f"host:{state.board}",
                },
                stdout=driver_log,
                stderr=driver_log,
            )
        finally:
            driver_log.close()
        state.tauri_driver = tauri_driver
        if not wait_listening(DRIVER_PORT, LISTEN_SECONDS):
            check("tauri-driver listening on loopback", False, f"log: {driver_log_path}")
            return teardown(state)
        check("tauri-driver listening on loopback", True, f"127.0.0.1:{DRIVER_PORT}")

        client = WebDriver(DRIVER_PORT, timeout=60.0)
        state.client = client
        # the app starts on the lanes' test-open channel (t-24@r11): the
        # launch argument left the human surface, and the host reader is
        # test-only machinery. The env rides tauri-driver to the app.
        session_id = client.new_session(app_bin, [])
        check("webdriver session created for the packaged app", True, f"session {session_id[:8]}")

        ready_deadline = time.monotonic() + APP_READY_SECONDS
        ready = False
        last_detail = "no execute result"
        while time.monotonic() < ready_deadline:
            if tauri_driver.poll() is not None:
                last_detail = f"tauri-driver exited rc={tauri_driver.returncode}"
                break
            try:
                inspection = client.execute(INSPECT_SCRIPT)
            except WebDriverError as error:
                last_detail = str(error)[:160]
                time.sleep(0.5)
                continue
            if isinstance(inspection, dict):
                resource = inspection.get("resource") or {}
                last_detail = f"status={resource.get('status')!r} errors={len(inspection.get('errors') or [])}"
                if resource.get("status") == "loaded":
                    ready = True
                    break
            else:
                last_detail = f"execute returned {type(inspection).__name__}"
                # the port flavor: no old inspection surface - the port
                # readout answers the same ready oracle
                try:
                    port_value = client.execute(PORT_READY_SCRIPT)
                    port_obs = json.loads(port_value) if isinstance(port_value, str) else None
                except (WebDriverError, ValueError):
                    port_obs = None
                if isinstance(port_obs, dict):
                    board = port_obs.get("board") or {}
                    last_detail = f"port status={board.get('status')!r} location={board.get('location')!r}"
                    if board.get("status") == "loaded":
                        ready = True
                        break
            time.sleep(0.5)
        check("app frontend up and board loaded", ready, f"board {board}; last: {last_detail}")
        if not ready:
            return teardown(state)

        if os.environ.get("SESSION_MODE") == "tracker":
            return run_tracker_sequence(state)
        code = serve(state, socket_path, lifetime, ready_file)
        cost = time.monotonic() - started
        passed = sum(1 for ok, _name, _detail in RESULTS if ok)
        print(f"[webdriver] {passed}/{len(RESULTS)} session checks passed in {cost:.1f}s", flush=True)
        print(f"[webdriver] shots under {shots} for inspection", flush=True)
        return code
    except SystemExit:
        raise  # a pinned-path refusal must keep its own exit code
    except BaseException as error:  # noqa: BLE001 - teardown owns the chain on every path
        print(f"[webdriver] session payload error: {error}", file=sys.stderr, flush=True)
        return teardown(state)


if __name__ == "__main__":
    sys.exit(main())
