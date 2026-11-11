#!/usr/bin/env python3
"""T-36 multi-window surface, host side (R5).

The session lane gains open/enumerate/focus/observe-by-window over the
suite's own windows:

- enumerate: the WebDriver handle list (each suite webview is a
  WebDriver window) paired with the tauri window label each webview
  reads from its own runtime metadata, plus the shell's existing
  suite_windows command as the authoritative inventory
- open: open_suite_window through the evaluate door (the same command
  the suite's picker uses; focus is never requested by the harness)
- focus: focus_suite_window for a named label (the jump's own door)
- observe-by-window: switch to the webview whose label matches, read
  the OBSERVE_SCRIPT projection, switch back
- evaluate-by-window: the same door with the R3 readout script, so a
  differential check reads any suite window the way it reads the main
  one

No product file is touched: the tauri commands and the page-side
metadata already exist. Where the driver chain refuses handle
enumeration (tauri-driver's window passthrough is unproven until the
red run), the verbs answer ok:false with the recorded error - a stop
to report, never a silent pass and never a product change to force it.
"""

from __future__ import annotations

import pathlib
import time

from t36_observation import READOUT_JS, CheckStopped


def windows_request() -> dict:
    return {"cmd": "windows"}


def open_window_request(state_id: str | None = None) -> dict:
    return {"cmd": "open-window", "state_id": state_id}


def focus_window_request(label: str) -> dict:
    return {"cmd": "focus-window", "label": label}


def observe_window_request(label: str, target: str, props: list[str] | None = None) -> dict:
    request: dict = {"cmd": "observe-window", "window": label, "target": target}
    if props:
        request["props"] = props
    return request


def readout_window_request(label: str) -> dict:
    """The R3 readout, addressed to one suite window by label."""
    return {"cmd": "evaluate-window", "window": label, "script": READOUT_JS}


def window_readout(socket_request, label: str, timeout: float = 60.0) -> dict:
    """One window's app-side readout. socket_request is the session
    socket's request callable (entry.session_request bound with the
    socket path); the evaluate-window door runs the readout in that
    webview."""
    response = socket_request(readout_window_request(label), timeout=timeout)
    if not isinstance(response, dict) or not response.get("ok"):
        raise CheckStopped(f"the window readout refused label {label!r}: {str(response)[:300]}")
    value = response.get("value")
    if not isinstance(value, dict):
        raise CheckStopped(f"window {label!r} returned no readout object")
    return value


def wait_for_window_count(socket_request, minimum: int, timeout: float = 30.0) -> list[dict]:
    """Poll the enumeration door until at least `minimum` suite windows
    answer, returning the last inventory; raises CheckStopped when the
    bound passes - the count never became the scenario's premise."""
    deadline = time.monotonic() + timeout
    inventory: list[dict] = []
    while time.monotonic() < deadline:
        response = socket_request(windows_request(), timeout=timeout)
        if isinstance(response, dict) and response.get("ok"):
            inventory = response.get("windows") or []
            if len(inventory) >= minimum:
                return inventory
        time.sleep(0.4)
    raise CheckStopped(
        f"only {len(inventory)} windows enumerated after {timeout:.0f}s (wanted {minimum});"
        " the driver chain may not pass window handles through"
    )


def write_inventory(run_root: pathlib.Path, inventory: list[dict], suite: object) -> pathlib.Path:
    path = run_root / "windows.json"
    path.write_text(
        json_dumps({"webdriver": inventory, "suite_windows": suite}) + "\n", encoding="utf-8"
    )
    return path


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
