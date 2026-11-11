#!/usr/bin/env python3
"""T-36 app observation through the session lane's evaluate door (R3).

The readout reads ALREADY-EXISTING page state only:

- window.__saccadeApp.inspection() - the machine-readable projection of
  the same view model the human UI renders (resource, identity, errors,
  diagnostics, renderOutcomes, targets, outline, copyMode, theme)
- window.__saccadeApp.deck() - the standing deck state (the t-21 test
  surface, called read-only)
- the strip, the history log and the picker, through their stable
  data-control attributes (board:location, strip:state, deck:line,
  deck:state, link:target, history:entry, picker:input)
- instrument rows, where the data-task vs data-tasks attribute split
  (TasksWindow's single value, AgentsWindow's space-joined list) is
  absorbed HERE, shim-side: both normalize to one tasks[] list, and no
  product attribute is renamed before a red run
- window.__TAURI_INTERNALS__.metadata.currentWindow.label and the
  suite_windows command - the window inventory the R5 surface builds on
- window.getSelection()

No setter is called and no product file is edited by any of this. The
R3 rule: if the existing state is insufficient for a check, the check
STOPS - CheckStopped, result status "stop", a SKIP line in TAP - and
the stop is reported. It is never resolved by renaming product
attributes, adding test-only setters, or editing client/src or
src-tauri.

The readout contract is one shape, two sides: this module's script
produces the app side; the prototype shim (t36_shim.js) produces the
same keys from the round-16 prototype's own surfaces. Fields one side
cannot produce are null and listed in `missing`; a check that needs a
missing field stops via require_fields, it never guesses.
"""

from __future__ import annotations

import json
import pathlib

READOUT_JS = r"""
const app = window.__saccadeApp;
if (!app || typeof app.inspection !== "function") {
  return {side: "app", booted: false, missing: ["__saccadeApp.inspection"], rows: [], strip: {}, windows: null, deck: null, inspection: null, selection: "", historyEntries: [], picker: {open: false, value: null}, window: {label: null}};
}
const out = {side: "app", booted: true, missing: []};
const inspection = app.inspection();
out.inspection = inspection && typeof inspection === "object" ? inspection : null;
if (!out.inspection) out.missing.push("inspection");
out.deck = typeof app.deck === "function" ? app.deck() : null;
if (out.deck === null) out.missing.push("deck");
const text = (sel) => { const el = document.querySelector(sel); return el ? (el.textContent || "").trim() : null; };
out.strip = {
  location: text('[data-control="board:location"]'),
  state: text('[data-control="strip:state"]'),
  deckLine: text('[data-control="deck:line"]'),
  deckState: text('[data-control="deck:state"]'),
  link: text('[data-control="link:target"]'),
};
const entries = [...document.querySelectorAll('[data-control="history:entry"]')];
out.historyEntries = entries.slice(0, 50).map((el) => (el.textContent || "").trim().slice(0, 300));
const pickerInput = document.querySelector('[data-control="picker:input"]');
out.picker = pickerInput ? {open: true, value: pickerInput.value} : {open: false, value: null};
// the data-task vs data-tasks split is absorbed here, shim-side: the
// single value (tasks rows) and the space-joined list (agents rows)
// both normalize to tasks[]. No product attribute is renamed.
const rows = [];
document.querySelectorAll('[data-control="instrument:row"]').forEach((el) => {
  const single = el.getAttribute("data-task");
  const multi = (el.getAttribute("data-tasks") || "").split(/\s+/).filter(Boolean);
  rows.push({
    instrument: el.getAttribute("data-instrument") || null,
    tasks: single != null && single !== "" ? [single] : multi,
    incarnation: el.getAttribute("data-incarnation") || null,
    address: el.getAttribute("data-address") || null,
    word: el.getAttribute("data-word") || null,
    app: el.getAttribute("data-app") || null,
    msg: el.getAttribute("data-msg") || null,
    progress: el.getAttribute("data-progress") || null,
    title: el.getAttribute("data-title") || null,
    at: el.getAttribute("data-at") || null,
    finished: el.getAttribute("data-finished") || null,
    workspace: el.getAttribute("data-workspace") || null,
    project: el.getAttribute("data-project") || null,
    openPath: el.getAttribute("data-open-path") || null,
    label: (el.textContent || "").trim().slice(0, 200),
  });
});
out.rows = rows;
out.selection = String(window.getSelection ? window.getSelection() : "");
const internals = window.__TAURI_INTERNALS__;
const metadata = internals && internals.metadata ? internals.metadata : null;
const label = metadata && metadata.currentWindow ? (metadata.currentWindow.label || null) : null;
if (label === null) out.missing.push("__TAURI_INTERNALS__.metadata.currentWindow.label");
out.window = {label: label};
if (internals && typeof internals.invoke === "function") {
  return internals.invoke("suite_windows").then((windows) => {
    if (Array.isArray(windows)) { out.windows = windows; } else { out.windows = null; out.missing.push("suite_windows"); }
    return out;
  }, () => { out.windows = null; out.missing.push("suite_windows"); return out; });
}
out.missing.push("__TAURI_INTERNALS__.invoke");
out.windows = null;
return out;
"""


class CheckStopped(Exception):
    """The existing page state cannot serve the check: no verdict, a
    recorded stop, a reported reason. Never a product edit."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def stop_result(scenario_id: str, lane: str, name: str, reason: str) -> dict:
    """The stopped check's result record: status stop, reason carried."""
    import t36_results

    return t36_results.make_result(scenario_id, lane, name, t36_results.STATUS_STOP, reason)


def _lookup(record: dict, dotted: str) -> object:
    value: object = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def missing_fields(readout: dict, required: list[str]) -> list[str]:
    """Which required contract fields the readout does not carry (null
    or absent). The readout's own `missing` list rides the answer."""
    gaps = [field for field in required if _lookup(readout, field) in (None, "",)]
    return gaps


def require_fields(readout: dict, required: list[str], scenario_id: str, lane: str, name: str) -> None:
    """The STOP rule as code: a check that needs fields the app does not
    already expose raises CheckStopped with the exact gaps."""
    gaps = missing_fields(readout, required)
    if gaps:
        raise CheckStopped(
            "existing app state does not expose " + ", ".join(gaps) + "; the check stops rather than"
            " touching product code (S" + scenario_id + " " + lane + " " + name + ")"
        )


def app_readout(socket_path: str, timeout: float = 60.0) -> dict:
    """The app-side readout over the session control socket: the
    evaluate door runs READOUT_JS in the app's webview and the JSON
    answer comes back. The session must be up (entry.py session up)."""
    import entry  # the harness entry carries the socket client

    response = entry.session_request(socket_path, {"cmd": "evaluate", "script": READOUT_JS}, timeout=timeout)
    if not isinstance(response, dict) or not response.get("ok"):
        raise CheckStopped(f"the evaluate door refused the readout: {json.dumps(response)[:300]}")
    value = response.get("value")
    if not isinstance(value, dict):
        raise CheckStopped("the readout script returned no object; the app page may not be the product webview")
    return value


def write_readout(run_root: pathlib.Path, name: str, readout: dict) -> pathlib.Path:
    """Persist one readout as evidence beside the run's logs."""
    path = run_root / f"readout-{name}.json"
    path.write_text(json.dumps(readout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
