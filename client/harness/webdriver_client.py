#!/usr/bin/env python3
"""The minimal W3C WebDriver client for the saccade-app session: stdlib
urllib only, no framework. One remote end (tauri-driver) at 127.0.0.1.

Every command goes through request(); the W3C envelope ({"value": ...},
error/element serialization) is handled here and nowhere else, so the
payload and the pure harness checks see plain values and one exception
type. Element references use the W3C element key.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"
SESSION_KEY = "sessionId"
DEFAULT_TIMEOUT = 30.0

# W3C webdriver key codepoints for named keys; any other single
# character is sent as itself. The key actions API speaks codepoints.
KEY_CODEPOINTS = {
    "null": "\ue000",
    "cancel": "\ue001",
    "help": "\ue002",
    "backspace": "\ue003",
    "back_space": "\ue003",
    "tab": "\ue004",
    "clear": "\ue005",
    "return": "\ue006",
    "enter": "\ue007",
    "shift": "\ue008",
    "control": "\ue009",
    "ctrl": "\ue009",
    "alt": "\ue00a",
    "pause": "\ue00b",
    "escape": "\ue00c",
    "esc": "\ue00c",
    "space": "\ue00d",
    "pageup": "\ue00e",
    "page_up": "\ue00e",
    "pagedown": "\ue00f",
    "page_down": "\ue00f",
    "end": "\ue010",
    "home": "\ue011",
    "left": "\ue012",
    "arrowleft": "\ue012",
    "up": "\ue013",
    "arrowup": "\ue013",
    "right": "\ue014",
    "arrowright": "\ue014",
    "down": "\ue015",
    "arrowdown": "\ue015",
    "insert": "\ue016",
    "delete": "\ue017",
    "semicolon": "\ue018",
    "equals": "\ue019",
    "numpad0": "\ue01a",
    "numpad1": "\ue01b",
    "numpad2": "\ue01c",
    "numpad3": "\ue01d",
    "numpad4": "\ue01e",
    "numpad5": "\ue01f",
    "numpad6": "\ue020",
    "numpad7": "\ue021",
    "numpad8": "\ue022",
    "numpad9": "\ue023",
    "multiply": "\ue024",
    "add": "\ue025",
    "separator": "\ue026",
    "subtract": "\ue027",
    "decimal": "\ue028",
    "divide": "\ue029",
    **{f"f{index}": chr(0xE031 + index - 1) for index in range(1, 13)},
}


class WebDriverError(RuntimeError):
    """A remote end answered with an error value, or the transport
    failed. error/message are the W3C fields when present."""

    def __init__(self, error: str, message: str) -> None:
        super().__init__(f"{error}: {message}")
        self.error = error
        self.message = message


def key_codepoint(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in KEY_CODEPOINTS:
        return KEY_CODEPOINTS[lowered]
    if len(name) == 1:
        return name
    raise WebDriverError("invalid argument", f"unknown key name: {name!r}")


def pointer_drag_actions(x: float, y: float, dx: float, dy: float, steps: int = 8) -> list[dict]:
    """A real pointer drag from (x, y) by (dx, dy) in viewport moves with
    inter-step pauses, so the target sees down/move/up, not a jump."""
    steps = max(int(steps), 1)
    moves: list[dict] = []
    for index in range(1, steps + 1):
        moves.append(
            {
                "type": "pointerMove",
                "duration": 40,
                "x": round(x + dx * index / steps),
                "y": round(y + dy * index / steps),
                "origin": "viewport",
            }
        )
    return [
        {
            "type": "pointer",
            "id": "mouse",
            "parameters": {"pointerType": "mouse"},
            "actions": [
                {"type": "pointerMove", "duration": 80, "x": round(x), "y": round(y), "origin": "viewport"},
                {"type": "pointerDown", "button": 0},
                *moves,
                {"type": "pointerUp", "button": 0},
            ],
        }
    ]


def pointer_click_actions(x: float, y: float) -> list[dict]:
    """A real pointer press/release at viewport coordinates (the escape
    hatch when no element should be located first)."""
    return [
        {
            "type": "pointer",
            "id": "mouse",
            "parameters": {"pointerType": "mouse"},
            "actions": [
                {"type": "pointerMove", "duration": 80, "x": round(x), "y": round(y), "origin": "viewport"},
                {"type": "pointerDown", "button": 0},
                {"type": "pause", "duration": 40},
                {"type": "pointerUp", "button": 0},
            ],
        }
    ]


def wheel_scroll_actions(x: float, y: float, dx: float, dy: float) -> list[dict]:
    """Wheel/scroll actions at viewport coordinates; elements route the
    scroll only through the coordinates resolved from their rect."""
    return [
        {
            "type": "wheel",
            "id": "wheel",
            "actions": [
                {"type": "scroll", "x": round(x), "y": round(y), "deltaX": round(dx), "deltaY": round(dy), "duration": 80, "origin": "viewport"}
            ],
        }
    ]


def context_click_actions(x: float, y: float) -> list[dict]:
    """A real context (right-button) click at viewport coordinates:
    the S09 menu-key-alternative invocation rides this."""
    return [
        {
            "type": "pointer",
            "id": "mouse",
            "parameters": {"pointerType": "mouse"},
            "actions": [
                {"type": "pointerMove", "duration": 80, "x": round(x), "y": round(y), "origin": "viewport"},
                {"type": "pointerDown", "button": 2},
                {"type": "pause", "duration": 40},
                {"type": "pointerUp", "button": 2},
            ],
        }
    ]


def key_press_actions(keys: list[str]) -> list[dict]:
    """Named keys as keyDown/keyUp pairs; a pressed-modifier sequence is
    the caller's responsibility, matching real single-key presses."""
    actions: list[dict] = []
    for name in keys:
        codepoint = key_codepoint(name)
        actions.append({"type": "keyDown", "value": codepoint})
        actions.append({"type": "keyUp", "value": codepoint})
    return [{"type": "key", "id": "keyboard", "actions": actions}]


def send_keys_value(text: str) -> dict:
    """The W3C element send-keys body: text plus the per-character value
    array (codepoints ride as their own characters)."""
    return {"text": text, "value": list(text)}


def session_capabilities(application: str, args: list[str]) -> dict:
    """The tauri-driver session request: the wrapper binary is the
    application and the board path its single argument; tauri-driver
    rewrites tauri:options into webkitgtk:browserOptions."""
    return {
        "capabilities": {
            "alwaysMatch": {
                "browserName": "wry",
                "tauri:options": {"application": application, "args": list(args)},
            }
        }
    }


class WebDriver:
    """One HTTP client against one remote end; sessions are explicit."""

    def __init__(self, port: int, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.port = int(port)
        self.timeout = float(timeout)
        self.session = ""

    def request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        url = f"http://127.0.0.1:{self.port}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout if timeout is None else timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError, OSError):
                raise WebDriverError("unknown error", f"http {error.code} from {path}") from error
            value = payload.get("value") if isinstance(payload, dict) else None
            if isinstance(value, dict) and "error" in value:
                raise WebDriverError(str(value.get("error")), str(value.get("message")))
            raise WebDriverError("unknown error", f"http {error.code} from {path}") from error
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise WebDriverError("unknown error", f"{method} {path} failed: {error}") from error
        if not isinstance(payload, dict) or "value" not in payload:
            raise WebDriverError("unknown error", f"malformed response to {method} {path}")
        value = payload["value"]
        if isinstance(value, dict) and "error" in value:
            raise WebDriverError(str(value.get("error")), str(value.get("message")))
        return value

    def new_session(self, application: str, args: list[str], timeout: float = 60.0) -> str:
        value = self.request("POST", "/session", session_capabilities(application, args), timeout=timeout)
        if not isinstance(value, dict) or not value.get(SESSION_KEY):
            raise WebDriverError("unknown error", f"session creation returned no session id: {value!r}")
        self.session = str(value[SESSION_KEY])
        return self.session

    def delete_session(self) -> None:
        if self.session:
            self.request("DELETE", f"/session/{self.session}", timeout=30.0)
        self.session = ""

    def find_element(self, selector: str) -> str:
        value = self.request("POST", f"/session/{self.session}/element", {"using": "css selector", "value": selector})
        if not isinstance(value, dict) or ELEMENT_KEY not in value:
            raise WebDriverError("no such element", selector)
        return str(value[ELEMENT_KEY])

    def click_element(self, element: str) -> None:
        self.request("POST", f"/session/{self.session}/element/{element}/click", {})

    def clear_element(self, element: str) -> None:
        self.request("POST", f"/session/{self.session}/element/{element}/clear", {})

    def send_keys(self, element: str, text: str) -> None:
        self.request("POST", f"/session/{self.session}/element/{element}/value", send_keys_value(text))

    def execute(self, script: str, args: list | None = None) -> object:
        return self.request("POST", f"/session/{self.session}/execute/sync", {"script": script, "args": list(args or [])})

    def perform_actions(self, actions: list[dict]) -> None:
        self.request("POST", f"/session/{self.session}/actions", {"actions": actions})

    def window_handles(self) -> list[str]:
        """All window/webview handles of the session (R5: the session
        lane's window surface rides the WebDriver handle list)."""
        value = self.request("GET", f"/session/{self.session}/window/handles")
        if not isinstance(value, list) or not all(isinstance(h, str) for h in value):
            raise WebDriverError("unknown error", f"window handles answered {value!r}")
        return value

    def current_window(self) -> str:
        value = self.request("GET", f"/session/{self.session}/window")
        if not isinstance(value, str):
            raise WebDriverError("unknown error", f"current window answered {value!r}")
        return value

    def close_window(self) -> None:
        """Close the CURRENT window (W3C DELETE /window); the caller
        switches away first when it must survive."""
        self.request("DELETE", f"/session/{self.session}/window")

    def switch_to_window(self, handle: str) -> None:
        self.request("POST", f"/session/{self.session}/window", {"handle": handle})

    def screenshot(self) -> bytes:
        value = self.request("GET", f"/session/{self.session}/screenshot", timeout=30.0)
        if not isinstance(value, str):
            raise WebDriverError("unknown error", "screenshot returned no base64 body")
        return base64.b64decode(value)
