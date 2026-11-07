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

import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time

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
    if not TARGET_ID.match(target):
        raise WebDriverError("invalid argument", f"target id is not a stable data-control id: {target!r}")
    return f'[data-control="{target}"]'


def element_center(client: WebDriver, target: str) -> tuple[float, float]:
    rect = client.execute(
        "(selector) => { const el = document.querySelector(selector);"
        " if (!el) return null; const r = el.getBoundingClientRect();"
        " return {x: r.x, y: r.y, width: r.width, height: r.height}; }",
        [target_selector(target)],
    )
    if not isinstance(rect, dict) or rect.get("width", 0) <= 0 and rect.get("height", 0) <= 0:
        raise WebDriverError("no such element", f"target has no viewport rect: {target}")
    return float(rect["x"]) + float(rect.get("width", 0)) / 2, float(rect["y"]) + float(rect.get("height", 0)) / 2


OBSERVE_SCRIPT = (
    "(selector, props) => { const el = document.querySelector(selector);"
    " if (!el) return null;"
    " const r = el.getBoundingClientRect();"
    " const cs = getComputedStyle(el);"
    " const style = {};"
    " for (const p of props || []) style[p] = cs.getPropertyValue(p);"
    " return {tag: el.tagName.toLowerCase(), rect: {x: r.x, y: r.y, width: r.width, height: r.height},"
    " transform: cs.transform, text: (el.textContent || '').trim(), style}; }"
)

INSPECT_SCRIPT = "() => (window.__boardViewer && window.__boardViewer.inspection()) || null"


def perform_input(client: WebDriver, request: dict) -> dict:
    """Real input only. click/type target an element through its stable
    id; wheel/drag anchor pointer or wheel sequences at the element's
    live center plus optional offsets; key acts on the focused element;
    coords is the explicit raw-coordinate escape hatch."""
    kind = request.get("kind")
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

    def chain_status(self) -> dict:
        return {
            "weston": bool(self.weston and self.weston.poll() is None),
            "tauri_driver": bool(self.tauri_driver and self.tauri_driver.poll() is None),
            "webdriver_session": bool(self.client and self.client.session),
            "app_alive": bool(survivor_pids()),
        }


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
                env={**os.environ},
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
        session_id = client.new_session(app_bin, [board])
        check("webdriver session created for the packaged app", True, f"session {session_id[:8]}")

        ready_deadline = time.monotonic() + APP_READY_SECONDS
        ready = False
        while time.monotonic() < ready_deadline:
            if tauri_driver.poll() is not None:
                break
            try:
                inspection = client.execute(INSPECT_SCRIPT)
            except WebDriverError:
                time.sleep(0.5)
                continue
            if isinstance(inspection, dict) and inspection.get("resource", {}).get("status") == "loaded":
                ready = True
                break
            time.sleep(0.5)
        check("app frontend up and board loaded", ready, f"board {board}")
        if not ready:
            return teardown(state)

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
