#!/usr/bin/env python3
"""T-36 phase-2b scenario bodies: S09, S10, S13, S15, S20 and S08's
sharpened stop. t36_checks imports and registers these; the grounding
notes name what was observed live at the baseline binaries."""

from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Callable

import t36_checks
import t36_port
import t36_results
from t36_checks import (
    RequestFn,
    _evaluate,
    _json_evaluate,
    _key,
    _ok,
    _premise,
    _select_by_chip,
    _selected_now,
    _sleep,
    _type,
    _wait_tracker_loaded,
)
from t36_observation import CheckStopped


def _windows(request: RequestFn) -> list[dict]:
    return _ok(request({"cmd": "windows"}, timeout=60)).get("windows") or []


_PICKER_OPEN_JS = 'return !!document.querySelector(\'[data-control="picker:input"]\');'
_PICKER_RESULTS_JS = ('return JSON.stringify([...document.querySelectorAll('
                      '\'[data-control="picker:result"]\')].map('
                      'e => (e.textContent || "").slice(0, 60)));')


# -- shared open-through-the-picker helper --------------------------------


def _open_tracker_board(request: RequestFn, pace: float, message_match: str) -> str:
    """Open a seeded tracker board through the picker (real input);
    returns the task id. A tracker board already shown means the host
    is remembered (the short spec scores the exact task match top); a
    fresh window needs the host-naming spec. The picker poll follows
    the legacy open_via_picker's discipline: the spaces, then WAIT for
    the picker before typing, then wait for the picker's own results
    to carry the task before the enter - real latency both times."""
    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    base = str(tracker.get("base") or "")
    task = None
    for task_id, entry in (tracker.get("tasks") or {}).items():
        if message_match in str(entry.get("message") or ""):
            task = task_id
    _premise(task is not None, f"no seed task matching {message_match!r}: {json.dumps(tracker.get('tasks'))[:200]}")
    current = _json_evaluate(
        request,
        'const i = window.__saccadeApp && window.__saccadeApp.inspection();'
        ' return JSON.stringify({machine: i && i.resource ? i.resource.machine : null});',
    )
    machine = str((current or {}).get("machine") or "")
    if machine.startswith("tracker:"):
        spec = f"seedtracker/{task}"
    else:
        spec = f"{base.replace('http://', '')}/seedtracker/{task}"
    row_needle = f"seedtracker/{task}"
    _key(request, "escape")
    _sleep(pace, 0.3)
    picker_open = False
    for _attempt in range(2):
        _key(request, " ")
        _sleep(pace, 0.4)
        _key(request, " ")
        deadline = time.monotonic() + 4.0 * pace
        while True:
            if _evaluate(request, _PICKER_OPEN_JS) is True:
                picker_open = True
                break
            if time.monotonic() >= deadline:
                break
            _sleep(pace, 0.3)
        if picker_open:
            break
        _key(request, "escape")
        _sleep(pace, 0.4)
    _premise(picker_open, "the picker never opened on space space")
    _type(request, "picker:input", spec)
    deadline = time.monotonic() + 8.0 * pace
    while True:
        results = _evaluate(request, _PICKER_RESULTS_JS)
        if isinstance(results, list) and any(row_needle in str(row) for row in results):
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.4)
    _key(request, "enter")
    _wait_tracker_loaded(request, task, pace)
    return task


# -- S09: one action, every invocation agrees -----------------------------


def _s09_port(request: RequestFn, pace: float) -> dict:
    """The port's four invocations of the citation's reference act."""
    import time


    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    base = str(tracker.get("base") or "").replace("http://", "")
    tasks = tracker.get("tasks") or {}
    cited = next((t for t, e in tasks.items() if "group-made" in str(e.get("message") or "")), None)
    citing = next((t for t, e in tasks.items() if "incorporation into a group child" in str(e.get("message") or "")), None)
    _premise(cited is not None and citing is not None,
             f"the S09 seeds are missing on the tracker: {json.dumps(tasks)[:200]}")

    def readout():
        return t36_port.readout(request)

    def wait_cited() -> str:
        deadline = time.monotonic() + 15.0 * pace
        while True:
            board = readout().get("board") or {}
            if board.get("status") == "loaded" and str(board.get("location") or "").endswith(f"{cited}@r1"):
                return str(board.get("location"))
            if time.monotonic() >= deadline:
                raise CheckStopped(f"an invocation never followed here: last location {board.get('location')!r}")
            _sleep(pace, 0.4)

    def menu_open() -> bool:
        return bool(_evaluate(request, 'return !!document.querySelector("[data-port-menu]");'))

    def citation_rect() -> dict:
        value = _json_evaluate(
            request,
            'const el = document.querySelector(\'[data-port-control="citation"]\');'
            " if (!el) return null; const r = el.getBoundingClientRect();"
            " return {x: r.x + r.width / 2, y: r.y + r.height / 2};",
        )
        _premise(isinstance(value, dict), "the citing board renders no citation link")
        return value

    def act_button_rect() -> dict:
        value = _json_evaluate(
            request,
            'const b = document.querySelector(\'[data-port-menu-act="f"]\');'
            " if (!b) return null; const r = b.getBoundingClientRect();"
            " return {x: r.x + r.width / 2, y: r.y + r.height / 2};",
        )
        _premise(isinstance(value, dict), "the reference menu renders no follow act button")
        return value

    def ground() -> None:
        t36_port.open_via_port_picker(request, f"{base}/seedtracker/{citing}", pace)
        deadline = time.monotonic() + 15.0 * pace
        while True:
            board = readout().get("board") or {}
            if board.get("status") == "loaded" and f"/{citing}" in str(board.get("location") or ""):
                return
            if time.monotonic() >= deadline:
                raise CheckStopped(f"the citing board never loaded: {board.get('location')!r}")
            _sleep(pace, 0.4)

    locations: list[str] = []
    evidence: list[str] = []

    def invoke(name: str, act) -> None:
        ground()
        rect = citation_rect()
        act(rect)
        _sleep(pace, 1.2)
        location = wait_cited()
        locations.append(f"{name}: {location!r}")
        if menu_open():
            evidence.append(f"{name} left the menu open while executing")

    invoke("plain click", lambda rect: _ok(request({"cmd": "input", "kind": "coords", "x": rect["x"], "y": rect["y"]}, timeout=40)))

    def rightclick_then(key: str):
        def run(rect: dict) -> None:
            _ok(request({"cmd": "input", "kind": "rightclick", "x": rect["x"], "y": rect["y"]}, timeout=40))
            _sleep(pace, 0.6)
            _premise(menu_open(), "the right-click opened no reference menu")
            _key(request, key)
        return run

    invoke("right-click Enter", rightclick_then("enter"))
    invoke("menu key f", rightclick_then("f"))

    def rightclick_item(rect: dict) -> None:
        _ok(request({"cmd": "input", "kind": "rightclick", "x": rect["x"], "y": rect["y"]}, timeout=40))
        _sleep(pace, 0.6)
        button = act_button_rect()
        _ok(request({"cmd": "input", "kind": "coords", "x": button["x"], "y": button["y"]}, timeout=40))

    invoke("act item click", rightclick_item)

    if evidence:
        return {"status": t36_results.STATUS_FAIL, "detail": "invocations disagree: " + "; ".join(evidence)}
    return {
        "status": t36_results.STATUS_PASS,
        "detail": (
            "all four invocations followed here to the cited pinned revision with the"
            " menu closed before executing: " + " | ".join(locations)
            + "; the port's menu-key invocation opens through the right-click"
              " (citations ride the citation surface, not the label layer - recorded)"
        ),
    }


def check_s09_invocations_agree(request: RequestFn, pace: float = 1.0) -> dict:
    """S09 one action, every invocation agrees - literal differential
    (no ratified difference), grounded on the citing board's citation.

    Port flavor (t36@r90): the citation's reference acts - 'f' follow
    here first, so a plain click follows; the four invocations are
    the plain click, right-click + Enter, right-click + the rendered
    act item, and right-click + the declared act key. The A2-style
    port note: the prototype's fourth path opens through label
    selection; the port's citations ride the citation surface, so
    its menu-key invocation opens through the right-click. The
    oracle is unchanged: every invocation follows here to the cited
    pinned revision, anchored at the named child, menu closed before
    executing.
    """
    if t36_port.is_port(request):
        return _s09_port(request, pace)
    evidence: list[str] = []
    locations: list[str] = []
    task = _open_tracker_board(request, pace, "the citing account, incorporation")

    def follow(what: str, act: Callable[[], None]) -> None:
        act()
        _sleep(pace, 1.5)
        readout = _json_evaluate(
            request,
            'const i = window.__saccadeApp && window.__saccadeApp.inspection();'
            ' return JSON.stringify({path: i ? i.resource.path : null,'
            ' menuOpen: !!document.querySelector(".ctx-menu") });',
        )
        path = str((readout or {}).get("path") or "")
        locations.append(f"{what}: {path!r} menuClosed={not (readout or {}).get('menuOpen')}")
        if not path.endswith("t-0@r1"):
            evidence.append(f"{what} did not follow here (path {path!r})")
        if (readout or {}).get("menuOpen"):
            evidence.append(f"{what} left the menu open while executing")
        _key(request, "escape")
        _sleep(pace, 0.3)
        _key(request, " ", " ")
        _sleep(pace, 0.8)
        _type(request, "picker:input", f"seedtracker/{task}")
        _sleep(pace, 0.5)
        _key(request, "enter")
        _wait_tracker_loaded(request, task, pace)

    follow("plain click",
           lambda: _ok(request({"cmd": "input", "kind": "click", "target": "board:open-citation"}, timeout=40)))

    _key(request, "escape")
    _sleep(pace, 0.3)
    _key(request, "s")
    _sleep(pace, 0.4)
    chips = _json_evaluate(
        request,
        "return JSON.stringify([...document.querySelectorAll('.select-label')].map("
        "c => ({label: c.textContent, id: c.getAttribute('data-target-id')})));",
    )
    chip = None
    if isinstance(chips, list):
        chip = next((c for c in chips if c.get("id") == "board:open-citation"), None)
    _premise(chip is not None, "the citation chip did not paint for the menu-key invocation")
    for letter in str(chip["label"]):
        _key(request, letter)
        _sleep(pace, 0.25)
    _sleep(pace, 0.4)
    follow("menu key f", lambda: _key(request, "f"))

    _ok(request({"cmd": "input", "kind": "rightclick", "target": "board:open-citation"}, timeout=40))
    _sleep(pace, 0.8)
    follow("right-click Enter", lambda: _key(request, "enter"))

    _ok(request({"cmd": "input", "kind": "rightclick", "target": "board:open-citation"}, timeout=40))
    _sleep(pace, 0.8)
    rect = _json_evaluate(
        request,
        'const b = document.querySelector(".ctx-menu .ctx-menu-act");'
        " if (!b) return null;"
        " const r = b.getBoundingClientRect();"
        " return JSON.stringify({x: r.x + r.width / 2, y: r.y + r.height / 2});",
    )
    _premise(isinstance(rect, dict), "the menu's first act button was not readable for the click invocation")
    follow("right-click act click",
           lambda: _ok(request({"cmd": "input", "kind": "coords", "x": rect["x"], "y": rect["y"]}, timeout=40)))

    if evidence:
        return {"status": t36_results.STATUS_FAIL, "detail": "invocations disagree: " + "; ".join(evidence)}
    return {"status": t36_results.STATUS_PASS,
            "detail": "all four invocations followed here to the cited revision with the menu closed: " + " | ".join(locations)}


# -- S10: sequences through the shipped matcher ---------------------------


def _tall_fixture_path(request: "RequestFn") -> pathlib.Path:
    """The tall-board fixture is written on the caller's host side,
    beside the session's own board: the native lane binds the board's
    PARENT DIRECTORY at /board (entry.py), so the fixture is servable
    at host:/board/<name> exactly when it sits in that directory. The
    lane's status echoes the host directory (SESSION_BOARD_DIR_HOST);
    a session without it (pure lanes, scripted tests) falls back to
    /tmp - the focused probe's own board lives there, so the same
    path serves it."""
    board_dir = ""
    try:
        answer = request({"cmd": "status"}, 30)
        if isinstance(answer, dict) and answer.get("ok"):
            board_dir = str(answer.get("board_dir_host") or "")
    except Exception:
        board_dir = ""
    parent = pathlib.Path(board_dir).resolve() if board_dir else pathlib.Path("/tmp")
    return parent / "t36-tall-board.yaml"


def check_s10_sequences(request: RequestFn, pace: float = 1.0) -> dict:
    """S10 sequences go through one matcher - the app as it SHIPS at
    4d59c35 (the keyMode state machine: gg scrolls top; unmatched
    application keys do nothing observable in the reading context).

    Script: open the citing board (a tall board), scroll down (j
    lines), press g then g, assert the scroller returned to top; press
    an unmatched key (q), assert nothing moved and no surface opened.
    The matcher's consume-vs-fallthrough internals are not separately
    observable through existing state: the check asserts the
    observables and records that limitation rather than inventing a
    deeper oracle."""
    # the port flavor: the same oracle over the port's own surfaces -
    # the open door for the tall board, the viewport readout for the
    # scroll observation, the reading context's own gg/G bindings
    if t36_port.is_port(request):
        return _s10_port(request, pace)
    # the sequence premise needs a scrollable board: the lane's
    # viewport is 1024x1280 and the seed boards fit it, so the body
    # generates its own tall host board and opens it through the
    # app's own open surface - the same test-open channel the lane
    # itself uses to put the first board on screen. The fixture is
    # written BESIDE THE SESSION'S OWN BOARD: the lane binds the
    # board's parent directory at /board, so a board in /tmp is
    # servable only when the session's board itself lives in /tmp
    # (the focused probe) - the differential lane's seeds directory
    # is what /board actually carries there
    tall = _tall_fixture_path(request)
    blocks = "".join(f"  - text: sequence premise block {n}\n" for n in range(1, 151))
    tall.write_text(f"grammar: 1\ntask: t-0\ncontent:\n{blocks}", encoding="ascii")
    _key(request, "escape")
    _sleep(pace, 0.3)
    open_answer = _evaluate(
        request,
        'return Promise.resolve(window.__saccadeApp.open("host", "/board/t36-tall-board.yaml"))'
        '.then(function () { return "opened"; },'
        ' function (e) { return "open-error: " + e; });')
    _premise(open_answer == "opened",
             f"the app's open refused the tall board: {open_answer!r}")
    deadline = time.monotonic() + 15.0 * pace
    loaded = False
    while True:
        value = _json_evaluate(
            request,
            'const i = window.__saccadeApp && window.__saccadeApp.inspection();'
            ' const parts = i && i.content && i.content.parts ? i.content.parts.length : 0;'
            ' return JSON.stringify({status: i ? i.resource.status : null,'
            ' location: i && i.resource ? i.resource.location : null, parts: parts});',
        )
        if isinstance(value, dict) and value.get("status") == "loaded" and (
            "t36-tall-board" in str(value.get("location") or "") or (value.get("parts") or 0) >= 30
        ):
            loaded = True
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.5)
    _premise(loaded, "the tall host board never loaded")
    _key(request, "escape")
    _sleep(pace, 0.3)
    scroller = (
        'const host = [...document.querySelectorAll("*")].find('
        "n => n.scrollHeight > n.clientHeight + 40 && n.clientHeight > 200);"
        " const el = host || document.scrollingElement;"
        " return JSON.stringify({top: el.scrollTop, height: el.scrollHeight});"
    )
    before = _json_evaluate(request, scroller)
    _premise(isinstance(before, dict) and (before.get("height") or 0) > (before.get("top") or 0),
             "the board is not scrollable; the sequence premise needs a taller board")
    for _ in range(10):
        _key(request, "j")
        _sleep(pace, 0.12)
    mid = _json_evaluate(request, scroller)
    _premise((mid or {}).get("top", 0) > (before or {}).get("top", 0),
             "j did not scroll the reading context (the board may fit the viewport at this window size)")
    _key(request, "g")
    _sleep(pace, 0.15)
    _key(request, "g")
    _sleep(pace, 0.6)
    after_gg = _json_evaluate(request, scroller)
    gg_ok = (after_gg or {}).get("top", 1) <= (before or {}).get("top", 0) + 2
    state_before = _json_evaluate(
        request,
        'return JSON.stringify({menu: !!document.querySelector(".ctx-menu"),'
        ' search: !!document.querySelector("[data-control=search]"),'
        ' labels: !!document.querySelector(".select-labels")});',
    )
    _key(request, "q")
    _sleep(pace, 0.5)
    state_after = _json_evaluate(
        request,
        'return JSON.stringify({menu: !!document.querySelector(".ctx-menu"),'
        ' search: !!document.querySelector("[data-control=search]"),'
        ' labels: !!document.querySelector(".select-labels")});',
    )
    top_after_q = _json_evaluate(request, scroller)
    nothing_opened = state_before == state_after
    nothing_scrolled = abs((top_after_q or {}).get("top", 0) - (after_gg or {}).get("top", 0)) <= 2
    if not gg_ok:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"gg did not return to top: {before} -> j {mid} -> gg {after_gg}"}
    if not (nothing_opened and nothing_scrolled):
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the unmatched key changed observable state: {state_before} -> {state_after}, scroll {after_gg} -> {top_after_q}"}
    return {"status": t36_results.STATUS_PASS,
            "detail": ("gg ran through the shipped matcher (scrolled back to top) and the unmatched"
                       " key changed nothing observable; the matcher's consume-vs-fallthrough"
                       " internals stay unobserved (the shipped state machine is the oracle here)")}


# -- S13: nested targets and citation -------------------------------------


def _s13_port(request: RequestFn, pace: float) -> dict:
    """The port's nested-target citation copy: the citation's copy
    names the SHOWN revision and the child's full canonical path -
    never the parent - and the group's own citation names its own
    full path."""
    import time

    import t36_port

    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    base = str(tracker.get("base") or "").replace("http://", "")
    tasks = tracker.get("tasks") or {}
    cited = next((t for t, e in tasks.items() if "group-made" in str(e.get("message") or "")), None)
    citing = next((t for t, e in tasks.items() if "other group child" in str(e.get("message") or "")), None)
    _premise(cited is not None and citing is not None,
             f"the S13 seeds are missing on the tracker: {json.dumps(tasks)[:200]}")

    def readout():
        return t36_port.readout(request)

    t36_port.open_via_port_picker(request, f"{base}/seedtracker/{citing}", pace)
    deadline = time.monotonic() + 15.0 * pace
    while True:
        board = readout().get("board") or {}
        if board.get("status") == "loaded" and f"/{citing}" in str(board.get("location") or ""):
            break
        if time.monotonic() >= deadline:
            raise CheckStopped(f"the citing board never loaded: {board.get('location')!r}")
        _sleep(pace, 0.4)

    # the citation's copy: right-click the citation, press y, read the
    # payload the ACTION produced (readout.copy - the observation
    # surface; the OS clipboard itself is an environment capability)
    link = _json_evaluate(
        request,
        'const el = document.querySelector(\'[data-port-control="citation"]\');'
        " if (!el) return null; const r = el.getBoundingClientRect();"
        " return {x: r.x + r.width / 2, y: r.y + r.height / 2,"
        ' open: el.getAttribute("data-port-open")};',
    )
    _premise(isinstance(link, dict), "the citing board renders no citation link")
    _ok(request({"cmd": "input", "kind": "rightclick", "x": link["x"], "y": link["y"]}, timeout=40))
    _sleep(pace, 0.6)
    _premise(bool(_evaluate(request, 'return !!document.querySelector("[data-port-menu]");')),
             "the right-click opened no reference menu")
    _key(request, "y")
    _sleep(pace, 0.6)
    copy_state = readout().get("copy")
    _premise(isinstance(copy_state, dict), "the copy act left no readout copy state")
    payload = str((copy_state or {}).get("payload") or "")

    # the oracle facts from the model, never the DOM: the cited task
    # at its shown revision and the child's full canonical path (the
    # group child, never the parent)
    expected_child = f"{cited}@r1:/content/0/content/0"
    _premise(payload == expected_child,
             f"the citation copy payload {payload!r} is not the shown revision + the child's"
             f" full path {expected_child!r} (never the parent)")

    # the group's own citation names ITS own full path (the prototype's
    # group kind: z Fold, y Copy citation). The group lives on the
    # CITED board: open it directly (the copy act cites the SHOWN
    # revision, whichever way the board was reached)
    t36_port.open_via_port_picker(request, f"{base}/seedtracker/{cited}", pace)
    deadline = time.monotonic() + 15.0 * pace
    while True:
        board = readout().get("board") or {}
        if board.get("status") == "loaded" and str(board.get("location") or "").endswith(f"{cited}@r1"):
            break
        if time.monotonic() >= deadline:
            raise CheckStopped(f"the cited board never loaded: {board.get('location')!r}")
        _sleep(pace, 0.4)
    obs = readout()
    parts = ((obs.get("content") or {}).get("parts") or [])
    group_part = next((str(p["part"]) for p in parts if p.get("kind") == "group"), None)
    _premise(group_part is not None, "the citing board renders no group target")
    group_rect = t36_port.part_rect(request, group_part)
    _premise(group_rect is not None, "the group target has no rendered rect")
    # the group's header strip (the container's center lands on children)
    y = group_rect["y"] + min(6.0, group_rect["height"] / 4)
    _ok(request({"cmd": "input", "kind": "rightclick", "x": group_rect["x"] + group_rect["width"] / 2, "y": y}, timeout=40))
    _sleep(pace, 0.6)
    _premise(bool(_evaluate(request, 'return !!document.querySelector(\'[data-port-menu-act="y"]\');')),
             "the group menu carries no copy-citation act")
    _key(request, "y")
    _sleep(pace, 0.6)
    group_copy = readout().get("copy")
    group_payload = str((group_copy or {}).get("payload") or "")
    expected_group = f"{cited}@r1:{group_part}"
    _premise(group_payload == expected_group,
             f"the group's own citation {group_payload!r} is not its own full path"
             f" {expected_group!r}")

    return {
        "status": t36_results.STATUS_PASS,
        "detail": (
            f"the citation copied {payload!r} (the shown revision + the child's full"
            f" canonical path, never the parent) and the group copied {group_payload!r}"
            " (its own full path); the payloads ride the readout's copy observation -"
            " the OS clipboard itself stays an environment capability"
        ),
    }


def check_s13_nested_citation_copy(request: RequestFn, pace: float = 1.0) -> dict:
    """S13 nested targets and citation, app side on group-made boards.

    Grounded surfaces at 4d59c35: block targets carry data-block-id
    (b0 group, b0.0/b0.1 children); a block's citation is computed by
    the app from its own data-block-id (partPath: b0.0 ->
    /content/0/group/content/0) plus the shown location; the menu's y
    act fires the copy (the strip shows '<what> copied') and delivers
    to the OS clipboard through write_clipboard. The clipboard text
    itself is not readable through existing state (the OS clipboard
    behind the native bridge), so the check verifies the copy act
    fires on child and group and that the app's own part-path mapping
    yields the distinct full paths - the readability gap is recorded
    here, and the ground truth rides the same mapping the app runs.

    Port flavor (t36@r90): the citation's reference copy names the
    shown revision + the child's full CANONICAL path (never the
    parent), and the group's own copy act names its own full path;
    the payload rides the readout's copy observation (the action's
    product), the OS clipboard staying an environment capability.

    Baseline finding recorded when present: the group's painted label
    may not select (typing the painted uppercase letter narrowed to
    text targets instead) - the group's acts then ride the real
    right-click instead, and the label mismatch is carried in the
    result detail."""
    if t36_port.is_port(request):
        return _s13_port(request, pace)
    task = _open_tracker_board(request, pace, "the cited account")
    _key(request, "escape")
    _sleep(pace, 0.3)
    _key(request, "s")
    _sleep(pace, 0.4)
    targets = _json_evaluate(
        request,
        "return JSON.stringify([...document.querySelectorAll('.select-label')].map(c => {"
        " const id = c.getAttribute('data-target-id');"
        " const block = document.querySelector('[data-block-id=\"' + id + '\"]')"
        "   || document.querySelector('[data-control=\"' + id + '\"]');"
        " return {label: c.textContent, id: id,"
        " blockId: block ? (block.getAttribute('data-block-id') || '') : '',"
        " inGroup: !!(block && block.closest('.block-group'))};}));",
    )
    _premise(isinstance(targets, list) and targets, "select mode painted no chips on the cited board")
    # a group chip's element IS a .block-group (closest matches self);
    # children carry the dotted block id
    groups = [c for c in targets if isinstance(c, dict) and c.get("inGroup") and "." not in str(c.get("blockId") or "")]
    children = [c for c in targets if isinstance(c, dict) and c.get("inGroup") and "." in str(c.get("blockId") or "")]
    _premise(children, f"no group-child target on the board: {json.dumps(targets)[:300]}")
    _premise(groups, f"no group target on the board: {json.dumps(targets)[:300]}")
    location = _evaluate(
        request,
        'const loc = document.querySelector("[data-control=\\"board:location\\"]");'
        " return loc ? loc.textContent : null;",
    )
    _premise(location, "the window's location is unreadable")

    def mapped(block_id: str) -> str | None:
        match = re.match(r"^b(\d+(?:\.\d+)*)$", str(block_id))
        if match is None:
            return None
        indexes = match.group(1).split(".")
        segments = [f"content/{indexes[0]}"]
        segments += [f"group/content/{index}" for index in indexes[1:]]
        return "/" + "/".join(segments)

    def note_text() -> str | None:
        value = _evaluate(
            request,
            'const n = document.querySelector("[data-control=\\"instrument:note\\"]");'
            " return n ? n.textContent : null;",
        )
        return str(value) if value else None

    def copy_act_fires(select_id: str, what: str) -> str | None:
        """Select by chip, press y, wait for the copy note; returns the
        note text ('the citation copied') or None when the act never
        fired."""
        label = _select_by_chip(request, select_id, pace)
        _premise(_selected_now(request).get("selected"), f"{what} did not select by its label {label!r}")
        _key(request, "y")
        deadline = time.monotonic() + 3.0 * pace
        while True:
            note = note_text()
            if note and "copied" in note:
                return note
            if time.monotonic() >= deadline:
                return None
            _sleep(pace, 0.25)

    evidence: list[str] = []
    child = children[0]
    child_note = copy_act_fires(str(child["id"]), "the group child")
    child_path = mapped(child.get("blockId"))
    _premise(child_path is not None, f"the child's block id {child.get('blockId')!r} maps to no part path")
    if child_note is None:
        evidence.append("y on the group child never fired the copy (no '<what> copied' note within the window)")

    group = groups[0]
    label = _select_by_chip(request, str(group["id"]), pace)
    group_selected = bool(_selected_now(request).get("selected"))
    if not group_selected:
        evidence.append(
            f"the group's painted label {label!r} does not select (typing it narrowed"
            " to text targets - the exact class of mismatch the differential exists to catch)"
        )

    # the group's acts ride the real right-click (the element carries
    # no data-control; viewport coords from its own rect)
    rect = _json_evaluate(
        request,
        'const g = document.querySelector("[data-block-id=' + "'" + str(group.get("blockId")) + "'" + ']");'
        " if (!g) return null;"
        " const r = g.getBoundingClientRect();"
        " return JSON.stringify({x: r.x + r.width / 2, y: r.y + 12});",
    )
    _premise(isinstance(rect, dict), f"the group element {group.get('blockId')!r} has no viewport rect")
    _ok(request({"cmd": "input", "kind": "rightclick", "x": rect["x"], "y": rect["y"]}, timeout=40))
    _sleep(pace, 0.8)
    _key(request, "y")
    deadline = time.monotonic() + 3.0 * pace
    group_note = None
    while True:
        note = note_text()
        if note and "copied" in note:
            group_note = note
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.25)
    group_path = mapped(group.get("blockId"))
    if group_note is None:
        evidence.append("y on the group never fired the copy (no '<what> copied' note within the window)")
    if child_path is not None and group_path is not None and child_path == group_path:
        evidence.append(f"the child and the group map to the same part path ({child_path!r})")

    # folding: the group's menu z; the child's select label then disappears
    rect = _json_evaluate(
        request,
        'const g = document.querySelector("[data-block-id=' + "'" + str(group.get("blockId")) + "'" + ']");'
        " if (!g) return null;"
        " const r = g.getBoundingClientRect();"
        " return JSON.stringify({x: r.x + r.width / 2, y: r.y + 12});",
    )
    _premise(isinstance(rect, dict), "the group element lost its viewport rect before the fold step")
    _ok(request({"cmd": "input", "kind": "rightclick", "x": rect["x"], "y": rect["y"]}, timeout=40))
    _sleep(pace, 0.8)
    _key(request, "z")
    _sleep(pace, 0.8)
    _key(request, "s")
    _sleep(pace, 0.4)
    folded_chips = _json_evaluate(
        request,
        "return JSON.stringify([...document.querySelectorAll('.select-label')].map("
        "c => c.getAttribute('data-target-id')));",
    )
    if isinstance(folded_chips, list) and str(child["id"]) in [str(x) for x in folded_chips]:
        evidence.append("the folded group's child still carries a select label")
    _key(request, "escape")

    if evidence:
        return {"status": t36_results.STATUS_FAIL, "detail": "nested citation mismatch: " + "; ".join(evidence)}
    return {"status": t36_results.STATUS_PASS,
            "detail": (f"on {task}: the copy act fired on the child (path {child_path!r}) and on the"
                       f" group (path {group_path!r}) - the app's own mapping yields the distinct full"
                       " paths against the shown revision - and folding removed the child's label."
                       " The clipboard text itself is not readable through existing state (OS"
                       " clipboard behind the native bridge); the gap is recorded, the ground truth"
                       " rides the same mapping the app runs")}


# -- S15: following current versus fixed ----------------------------------


def check_s15_follows_current(request: RequestFn, pace: float = 1.0) -> dict:
    """S15 following current versus fixed revision, app-only (real
    tracker commits observed through the tracker read surface).

    Script: open the landing board normally (current); commit a new
    revision through the REAL wire (the payload's tracker-commit, the
    seeds' own shape); the open board must update live. Then open the
    explicit revision, commit again; the shown revision stays fixed.
    No pin/follow toggle exists; the distinction is internal."""
    # the port flavor: the same oracle over the port's own surfaces -
    # the picker for opens, the shown location and canonical part-kind
    # signature for revision/content changes (RECORDED ADAPTATION:
    # the port has no rawLength; counts cannot distinguish this exact
    # fixture). The tracker-commit verb is flavor-independent.
    if t36_port.is_port(request):
        return _s15_port(request, pace)
    task = _open_tracker_board(request, pace, "the landing figure")
    task_number = int(str(task).removeprefix("t-"))
    _key(request, "escape")

    def raw_length() -> int | None:
        value = _json_evaluate(
            request,
            'const i = window.__saccadeApp && window.__saccadeApp.inspection();'
            ' return JSON.stringify({raw: i && i.identity ? i.identity.rawLength : null});',
        )
        return (value or {}).get("raw")

    before = raw_length()
    revised = (
        f"grammar: 1\ntask: {task}\ncontent:\n"
        "  - text: the landing board, revised live mid-check\n"
        "  - group:\n      title: Live update\n      content:\n"
        "        - text: the live-update group child\n"
    )
    commit = _ok(request({"cmd": "tracker-commit", "task": task_number, "board": revised,
                          "message": "t36 mid-check: the live-update revision"}, timeout=60))
    revision = commit.get("revision")
    _premise(revision is not None, f"the mid-check commit did not land: {json.dumps(commit)[:200]}")
    deadline = time.monotonic() + 20.0 * pace
    live = False
    snapshot = None
    while True:
        snapshot = raw_length()
        if snapshot is not None and before is not None and snapshot != before:
            live = True
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.8)
    if not live:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the normally opened board did not update live: rawLength stayed {before} after r{revision} committed"}
    live_length = snapshot

    _key(request, "escape")
    _sleep(pace, 0.3)
    _key(request, " ", " ")
    _sleep(pace, 0.8)
    _type(request, "picker:input", f"seedtracker/{task}@r1")
    _sleep(pace, 0.5)
    _key(request, "enter")
    deadline = time.monotonic() + 20.0 * pace
    pinned_at = None
    while True:
        pinned_at = raw_length()
        if pinned_at is not None:
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.6)
    _premise(pinned_at is not None, "the explicit revision did not open")
    revised_r3 = f"grammar: 1\ntask: {task}\ncontent:\n  - text: the landing board, third revision while pinned\n"
    commit = _ok(request({"cmd": "tracker-commit", "task": task_number, "board": revised_r3,
                          "message": "t36 mid-check: the pinned-board commit"}, timeout=60))
    _premise(commit.get("revision") == (revision or 0) + 1,
             f"the pinned-board commit did not land: {json.dumps(commit)[:160]}")
    _sleep(pace, 3.0)
    pinned_now = raw_length()
    if pinned_now != pinned_at:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the explicitly opened revision moved: rawLength {pinned_at} -> {pinned_now} while current advanced"}
    return {"status": t36_results.STATUS_PASS,
            "detail": (f"the current board updated live ({before} -> {live_length} on r{revision});"
                       f" the explicitly opened revision held ({pinned_at}) while r{commit.get('revision')} landed")}


# -- S20: relaunch restores the persisted window surface ------------------


def check_s20_relaunch(request: RequestFn, pace: float = 1.0) -> dict:
    """S20's relaunch half: the persisted surface only - which windows
    exist after a relaunch into the SAME run root (the app's own data
    dir survives; the tracker and its window store do not, and the
    scenario asserts no live content beyond that). The driver injects
    the relaunch callable (down, up-t36 into the same root); app-only
    runs without it stop with that recorded reason."""
    before = _windows(request)
    labels = sorted(w["label"] for w in before)
    _premise(len(labels) >= 1, "no windows before the relaunch")
    if t36_checks._RELAUNCHER is None:
        raise CheckStopped("no relaunch callable attached (S20 needs the driver's two-session orchestration)")
    t36_checks._RELAUNCHER()

    # the recovery act (the r73 diagnosis): the relaunched app's
    # persisted tracker host points at the dead pre-relaunch seed
    # server, and its windows bootstrap loops refused against it while
    # the launch restore finds nothing. A human re-tells the app where
    # the tracker is by opening a tracker location - real input through
    # the picker, recorded as the recovery step; the window-set oracle
    # is judged after it.
    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    base = str(tracker.get("base") or "")
    tasks = tracker.get("tasks") or {}
    seed = next(iter(tasks), None)
    _premise(seed is not None and base,
             f"no seed server on the relaunched session: {json.dumps(tracker)[:160]}")
    spec = f"{base.replace('http://', '')}/seedtracker/{seed}"
    if t36_port.is_port(request):
        t36_port.open_via_port_picker(request, spec, pace)
    else:
        _key(request, "escape")
        _sleep(pace, 0.3)
        _key(request, " ", " ")
        _sleep(pace, 0.8)
        _type(request, "picker:input", spec)
        _sleep(pace, 0.5)
        _key(request, "enter")

    deadline = time.monotonic() + 40.0 * pace
    after: list[dict] = []
    last_error: str = ""
    while True:
        try:
            after = _windows(request)
            if len(after) >= len(before):
                break
        except CheckStopped as stop:
            last_error = stop.reason[:160]
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 1.0)
    if len(after) < len(before):
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"the relaunch did not restore the window surface: {len(before)} windows before,"
                           f" {len(after)} after ({last_error or 'the window enumeration never answered'})")}
    # the port flavor: the readout rides the observation layer (the
    # app-side identity of the relaunched window); the oracle - the
    # restored window set - stays unchanged
    flavor_note = ""
    try:
        if t36_port.is_port(request):
            obs = t36_port.readout(request)
            board = obs.get("board") or {}
            flavor_note = (f"; the port readout answers (window {(obs.get('window') or {}).get('label')!r},"
                           f" board {board.get('location')!r} status {board.get('status')!r})")
    except CheckStopped:
        pass
    return {"status": t36_results.STATUS_PASS,
            "detail": (f"the relaunch restored the persisted window surface: {len(labels)} window(s) before,"
                       f" {len(after)} after; labels are fresh (the persisted surface is the window set,"
                       f" not the names); the recovery act re-opened {spec!r} through the picker"
                       " (the relaunched app's persisted tracker host points at the dead"
                       " pre-relaunch server until a location re-tells it)"
                       + flavor_note)}


def _s10_port(request: Callable, pace: float) -> dict:
    import time

    from t36_checks import _json_evaluate, _key, _premise, _sleep

    # the tall board rides /board (written beside the session's own
    # board - see the shared body); the port's own open door puts it
    # on screen
    tall = _tall_fixture_path(request)
    blocks = "".join(f"  - text: sequence premise block {n}\n" for n in range(1, 151))
    tall.write_text(f"grammar: 1\ntask: t-0\ncontent:\n{blocks}", encoding="ascii")
    _key(request, "escape")
    _sleep(pace, 0.3)
    # the open is awaited with its own error surfaced: a silent no-op
    # open was indistinguishable from a slow load on BOTH flavors (the
    # shared premise issue)
    from t36_checks import _evaluate as _ev
    open_answer = _ev(
        request,
        "return Promise.resolve(window.__suiteOpenAt(\"host:/board/t36-tall-board.yaml\"))"
        ".then(function () { return 'opened'; },"
        " function (e) { return 'open-error: ' + e; });",
    )
    _premise(open_answer == "opened",
             f"the port's open door refused the tall board: {open_answer!r}")
    deadline = time.monotonic() + 15.0 * pace
    obs: dict | None = None
    while True:
        value = t36_port.readout(request)
        board = value.get("board") or {}
        parts = ((value.get("content") or {}).get("parts") or [])
        # content-based: the tall board's observable is many parts,
        # whatever location string the flavor renders
        if board.get("status") == "loaded" and (len(parts) >= 30 or "t36-tall-board" in str(board.get("location") or "")):
            obs = value
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.4)
    _premise(obs is not None, "the tall host board never loaded on the port")
    t36_port.require_fields(obs, "viewport")

    def top() -> float:
        return float((t36_port.readout(request).get("viewport") or {}).get("top") or 0.0)

    surfaces = (
        "return JSON.stringify({menu: !!document.querySelector(\"[data-port-menu]\"),"
        " labels: !!document.querySelector(\".port-select-labels\"),"
        " picker: !!document.querySelector(\"[data-port-control=\\\"picker-input\\\"]\")});"
    )
    before_top = top()
    # the G binding is the chord shift+g: a bare WebDriver uppercase
    # codepoint arrives with shift=false (verified through a read-only
    # keydown observer) and the port's matcher compares key AND shift
    # exactly - the chord is the only correct delivery
    request({"cmd": "input", "kind": "hotkey", "keys": ["shift", "g"]}, 30)
    deadline = time.monotonic() + 5.0 * pace
    mid = before_top
    while True:
        mid = top()
        if mid > before_top + 100:
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.2)
    _premise(mid > before_top + 100,
             "G did not scroll the reading context on the port (the board may fit the viewport)")
    _key(request, "g")
    _sleep(pace, 0.15)
    _key(request, "g")
    deadline = time.monotonic() + 5.0 * pace
    after_gg = mid
    while True:
        after_gg = top()
        if after_gg <= before_top + 2:
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.2)
    gg_ok = after_gg <= before_top + 2
    state_before = _json_evaluate(request, surfaces)
    _key(request, "q")
    _sleep(pace, 0.5)
    state_after = _json_evaluate(request, surfaces)
    top_after_q = top()
    nothing_opened = state_before == state_after
    nothing_scrolled = abs(top_after_q - after_gg) <= 2.0
    if not gg_ok:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"gg did not return to top on the port: {before_top} -> G {mid} -> gg {after_gg}"}
    if not (nothing_opened and nothing_scrolled):
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"the unmatched key changed observable state on the port:"
                           f" {state_before} -> {state_after}, scroll {after_gg} -> {top_after_q}")}
    return {"status": t36_results.STATUS_PASS,
            "detail": ("gg ran through the shipped matcher on the port (G scrolled down, gg back to"
                       " top through the viewport readout) and the unmatched key changed nothing"
                       " observable; the matcher's consume-vs-fallthrough internals stay unobserved"
                       " (the shipped state machine is the oracle here)")}


def _s15_port(request: RequestFn, pace: float) -> dict:
    import json as _json
    import time

    from t36_checks import _ok, _premise, _sleep
    from t36_port import open_via_port_picker

    def readout() -> dict:
        return t36_port.readout(request)

    def parts_signature() -> tuple | None:
        # the rendered-content signature: the shown location plus every
        # part's canonical path and kind. A part count cannot tell this
        # fixture's before from after (text+group+figure and
        # text+group+text are both three parts) - the signature can.
        board = readout().get("board") or {}
        content = readout().get("content") or {}
        parts = content.get("parts")
        if not isinstance(parts, list):
            return None
        return (
            str(board.get("location") or ""),
            tuple(sorted((str(p.get("part")), str(p.get("kind"))) for p in parts)),
        )

    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    base = str(tracker.get("base") or "")
    tasks = tracker.get("tasks") or {}
    landing = next((t for t, e in tasks.items() if "landing" in str(e.get("message") or "")), None)
    _premise(landing is not None, f"no landing seed task on the tracker: {_json.dumps(tasks)[:200]}")
    host = base.replace("http://", "")

    # open the landing board normally (current)
    open_via_port_picker(request, f"{host}/seedtracker/{landing}", pace)
    deadline = time.monotonic() + 15.0 * pace
    while True:
        board = (readout().get("board") or {})
        if board.get("status") == "loaded" and f"/{landing}" in str(board.get("location") or ""):
            break
        if time.monotonic() >= deadline:
            # the picker-consumed-enter evidence (the port finding): the
            # overlay closed, the input carried the spec, and the shown
            # location never moved
            picker_gone = _json_evaluate(
                request,
                "return !document.querySelector('[data-port-control=\"picker-input\"]');",
            )
            raise CheckStopped(
                f"the landing board never loaded on the port: {_json.dumps(board)[:160]}"
                + ("; the picker closed without navigating (the port's first-open"
                   " submit is unreliable - recorded product finding)" if picker_gone is True else "")
            )
        _sleep(pace, 0.4)
    before = parts_signature()
    _premise(before is not None, "the port readout carries no content parts")

    revised = (
        f"grammar: 1\ntask: {landing}\ncontent:\n"
        "  - text: the landing board, revised live mid-check\n"
        "  - group:\n      title: Live update\n      content:\n"
        "        - text: the live-update group child\n"
    )
    task_number = int(str(landing).removeprefix("t-"))
    commit = _ok(request({"cmd": "tracker-commit", "task": task_number, "board": revised,
                          "message": "t36 mid-check: the live-update revision"}, timeout=60))
    revision = commit.get("revision")
    _premise(revision is not None, f"the mid-check commit did not land: {json.dumps(commit)[:200]}")

    deadline = time.monotonic() + 20.0 * pace
    live = False
    live_signature = None
    while True:
        signature = parts_signature()
        if signature is not None and signature != before:
            live = True
            live_signature = signature
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.8)
    if not live:
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"the normally opened board did not update live: the shown revision/content"
                           f" signature stayed {before} after r{revision} committed"
                           " - the raw-length oracle's recorded adaptation rides the"
                           " location+part-kind signature")}

    # the explicit revision: pinned
    open_via_port_picker(request, f"{host}/seedtracker/{landing}@r1", pace)
    deadline = time.monotonic() + 15.0 * pace
    pinned_at = None
    while True:
        board = (readout().get("board") or {})
        if board.get("status") == "loaded" and "r1" in str(board.get("location") or ""):
            pinned_at = parts_signature()
            if pinned_at is not None:
                break
        if time.monotonic() >= deadline:
            raise CheckStopped(f"the explicit revision never opened on the port: {_json.dumps(board)[:160]}")
        _sleep(pace, 0.4)
    revised_r3 = f"grammar: 1\ntask: {landing}\ncontent:\n  - text: the landing board, third revision while pinned\n"
    commit = _ok(request({"cmd": "tracker-commit", "task": task_number, "board": revised_r3,
                          "message": "t36 mid-check: the pinned-board commit"}, timeout=60))
    _premise(commit.get("revision") == (revision or 0) + 1,
             f"the pinned-board commit did not land: {json.dumps(commit)[:160]}")
    _sleep(pace, 3.0)
    pinned_now = parts_signature()
    if pinned_now != pinned_at:
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"the explicitly opened revision moved: signature {pinned_at} -> {pinned_now}"
                           " while current advanced")}
    return {"status": t36_results.STATUS_PASS,
            "detail": (f"the current board updated live (signature {before} -> {live_signature} on"
                       f" r{revision}); the explicitly opened r1 held ({pinned_at}) while"
                       f" r{commit.get('revision')} landed; the raw-length oracle's adaptation"
                       " rides the location+part-kind signature (the port has no rawLength)")}


def _s07_state(request) -> dict:
    """The S07 observation triple: shown location, the trail, the live
    reading position (the readout's viewport.top is the scroller's
    scrollTop)."""
    obs = t36_port.readout(request)
    t36_port.require_fields(obs, "trail", "viewport")
    board = obs.get("board") or {}
    return {
        "location": str(board.get("location") or ""),
        "status": str(board.get("status") or ""),
        "trail": obs.get("trail") or {},
        "top": float((obs.get("viewport") or {}).get("top") or 0.0),
    }


def _s07_hotkey(request, *keys: str) -> None:
    from t36_checks import _ok

    _ok(request({"cmd": "input", "kind": "hotkey", "keys": list(keys)}))


def _s07_wait_place(request, pace: float, needle: str, timeout: float = 15.0) -> dict:
    """Wait until the shown board carries the needle AND its reading
    position settles (two reads agreeing within 3px, or the timeout).
    The restore is async - the place lands after the content and the
    geometry."""
    import time

    deadline = time.monotonic() + timeout * pace
    state = _s07_state(request)
    while True:
        state = _s07_state(request)
        if needle in state["location"] and state["status"] == "loaded":
            break
        if time.monotonic() >= deadline:
            return state
        _sleep(pace, 0.4)
    deadline = time.monotonic() + 5.0 * pace
    settled = state
    while True:
        nxt = _s07_state(request)
        if needle in nxt["location"] and abs(nxt["top"] - settled["top"]) <= 3:
            return nxt
        settled = nxt
        if time.monotonic() >= deadline:
            return nxt
        _sleep(pace, 0.3)


def check_s07_return_place(request: RequestFn, pace: float = 1.0) -> dict:
    """S07 return place restores the passage, not the presentation.

    Script: read a tall board at a place (scrolled to its end), open a
    second board, press Back: the addressed place AND the reading
    position left there come back, requested without advancing the
    trail; Forward retraces. Back/Forward are the app's own pinned
    chords (Ctrl-o / Ctrl-i), delivered as explicit hotkey chords (the
    lane's surviving synthesis; bare codepoints lose their modifiers).

    The variant journeys (anchor pending geometry, anchor confirmed
    absent, failed load) need fixtures and failure-injection ports the
    native harness does not carry: the failed load follows S02's
    recorded no-injection stop; the anchor variants have no authored
    fixture. They stay recorded, not asserted here.
    """
    import time

    if not t36_port.is_port(request):
        raise CheckStopped(
            "the app's inspection() projects no return-place records (no trail"
            " entries, no back/forward state - the S14 readout gap): the journey"
            " would stop on every step; the port half carries the oracle through"
            " its own observation contract"
        )

    # the tall board beside the session's own (the S10 premise shape)
    tall = _tall_fixture_path(request)
    blocks = "".join(f"  - text: return-place block {n}\n" for n in range(1, 151))
    tall.write_text(f"grammar: 1\ntask: t-0\ncontent:\n{blocks}", encoding="ascii")
    _key(request, "escape")
    _sleep(pace, 0.3)
    open_answer = _evaluate(
        request,
        'return Promise.resolve(window.__suiteOpenAt("host:/board/t36-tall-board.yaml"))'
        '.then(function () { return "opened"; },'
        ' function (e) { return "open-error: " + e; });',
    )
    _premise(open_answer == "opened",
             f"the port's open door refused the tall board: {open_answer!r}")
    deadline = time.monotonic() + 15.0 * pace
    while True:
        state = _s07_state(request)
        if "t36-tall-board" in state["location"] and state["status"] == "loaded":
            break
        if time.monotonic() >= deadline:
            raise CheckStopped(f"the tall board never loaded: {state['location']!r} {state['status']!r}")
        _sleep(pace, 0.4)

    # read the board at a place: gg first (the retained buffer may
    # still carry an earlier reading position - retention is S05's own
    # oracle), then the G chord scrolls to the end
    _key(request, "g")
    _sleep(pace, 0.15)
    _key(request, "g")
    deadline = time.monotonic() + 5.0 * pace
    while True:
        state = _s07_state(request)
        if "t36-tall-board" in state["location"] and state["top"] <= 2:
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.2)
    before_top = state["top"]
    _s07_hotkey(request, "shift", "g")
    deadline = time.monotonic() + 5.0 * pace
    placed = before_top
    while True:
        placed = _s07_state(request)["top"]
        if placed > before_top + 100:
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.2)
    _premise(placed > before_top + 100,
             f"G did not scroll the tall board ({before_top} -> {placed}); the journey needs a nonzero departure place")
    left_top = placed

    # navigate elsewhere in the same window: the picker opens the landing board
    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    tasks = tracker.get("tasks") or {}
    landing = next((t for t, e in tasks.items() if "landing" in str(e.get("message") or "")), None)
    _premise(landing is not None, "the landing seed is missing from the tracker")
    host = str(tracker.get("base") or "").replace("http://", "")
    t36_port.open_via_port_picker(request, f"{host}/seedtracker/{landing}", pace)
    elsewhere = _s07_wait_place(request, pace, f"/{landing}")
    _premise(f"/{landing}" in elsewhere["location"],
             f"the landing board never loaded: {elsewhere['location']!r}")
    depth = elsewhere["trail"].get("depth")
    _premise(isinstance(depth, int) and depth >= 2,
             f"the trail never carried the two openings: {elsewhere['trail']}")

    # Back: the addressed place and the reading position come back,
    # without advancing the trail
    _s07_hotkey(request, "ctrl", "o")
    back = _s07_wait_place(request, pace, "t36-tall-board")
    trail_back = back["trail"]
    if "t36-tall-board" not in back["location"]:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"Back did not return to the addressed place: {back['location']!r}"}
    if trail_back.get("depth") != depth or trail_back.get("cursor") != depth - 2:
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"Back moved the trail, not just the cursor: depth"
                           f" {depth}->{trail_back.get('depth')} cursor"
                           f" {trail_back.get('cursor')} (expected depth {depth}, cursor {depth - 2})")}
    if abs(back["top"] - left_top) > 3:
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"Back restored the place but not the passage: reading"
                           f" position {left_top} was left, {back['top']} came back")}

    # Forward retraces without new entries
    _s07_hotkey(request, "ctrl", "i")
    fwd = _s07_wait_place(request, pace, f"/{landing}")
    trail_fwd = fwd["trail"]
    if f"/{landing}" not in fwd["location"]:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"Forward did not retrace to the landing board: {fwd['location']!r}"}
    if trail_fwd.get("depth") != depth or trail_fwd.get("cursor") != depth - 1:
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"Forward changed the trail: depth {trail_fwd.get('depth')}"
                           f" cursor {trail_fwd.get('cursor')} (expected {depth}, {depth - 1})")}
    return {"status": t36_results.STATUS_PASS,
            "detail": (f"Back restored the tall board at its left reading position"
                       f" ({left_top}px came back as {back['top']}px) with the trail"
                       f" unchanged (depth {depth}, cursor {trail_back.get('cursor')})"
                       f" and Forward retraced to the landing board (cursor"
                       f" {trail_fwd.get('cursor')}); the chords are the pinned"
                       f" Ctrl-o/Ctrl-i delivered as explicit hotkey synthesis. The"
                       f" pending-geometry, absent-anchor and failed-load variants"
                       f" stay recorded: they need fixtures and a failure-injection"
                       f" port the native harness does not carry")}


def check_s05_temporary_buffers(request: RequestFn, pace: float = 1.0) -> dict:
    """S05 temporary buffers die with their scope; ordinary buffers
    are retained - the native half.

    Script: open a board, scroll it to a place, open a SECOND board
    through the window picker (the temporary surface), press Back: the
    picker never appears as a trail destination, exactly one entry was
    added for the submitted opening, and Back lands on the first board
    at its departure place. The board->tasks->Back half and the
    window-close half wait for the tasks surface and the linked-window
    work (recorded, not asserted here); the figure-inspection chain
    link is a pure-lane fixture (the scenario's own note).
    """
    import time

    if not t36_port.is_port(request):
        raise CheckStopped(
            "the app's inspection() projects neither the trail nor buffer"
            " lifecycle state (the S14 readout gap): the picker-never-a-destination"
            " assertion has no observation surface; the port half carries it"
        )

    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    tasks = tracker.get("tasks") or {}
    landing = next((t for t, e in tasks.items() if "landing" in str(e.get("message") or "")), None)
    cited = next((t for t, e in tasks.items() if "group-made" in str(e.get("message") or "")), None)
    _premise(landing is not None and cited is not None, "the S05 seeds are missing from the tracker")
    host = str(tracker.get("base") or "").replace("http://", "")

    # the retained ordinary buffer: the tall fixture (S10's premise
    # shape - the journey needs a scrollable first board), opened
    # through the picker itself so BOTH openings ride the temporary
    # surface the scenario is about
    tall = _tall_fixture_path(request)
    blocks = "".join(f"  - text: temporary-buffer block {n}\n" for n in range(1, 151))
    tall.write_text(f"grammar: 1\ntask: t-0\ncontent:\n{blocks}", encoding="ascii")
    t36_port.open_via_port_picker(request, "host:/board/t36-tall-board.yaml", pace)
    first = _s07_wait_place(request, pace, "t36-tall-board")
    _premise("t36-tall-board" in first["location"],
             f"the tall board never loaded through the picker: {first['location']!r}")
    before_top = first["top"]
    _s07_hotkey(request, "shift", "g")
    deadline = time.monotonic() + 5.0 * pace
    placed = before_top
    while True:
        placed = _s07_state(request)["top"]
        if placed > before_top + 100:
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.2)
    _premise(placed > before_top + 100,
             f"G did not scroll the first board ({before_top} -> {placed}); the journey needs a departure place")
    left_top = placed
    depth_first = first["trail"].get("depth")

    # the temporary surface again, to a second board
    t36_port.open_via_port_picker(request, f"{host}/seedtracker/{cited}", pace)
    second = _s07_wait_place(request, pace, f"/{cited}")
    _premise(f"/{cited}" in second["location"],
             f"the second board never loaded: {second['location']!r}")
    trail = second["trail"]
    depth = trail.get("depth")
    entries = [str(e.get("location") or "") for e in (trail.get("entries") or [])]
    if not (isinstance(depth, int) and isinstance(depth_first, int) and depth == depth_first + 1):
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"the picker journey added {None if not isinstance(depth, int) else depth - depth_first}"
                           f" entries (expected exactly 1): {entries}")}
    if any("picker" in location.lower() for location in entries):
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the picker itself landed on the trail: {entries}"}

    # Back lands on the first board at its departure place
    _s07_hotkey(request, "ctrl", "o")
    back = _s07_wait_place(request, pace, "t36-tall-board")
    if "t36-tall-board" not in back["location"]:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"Back from the second board did not land on the first: {back['location']!r}"}
    if abs(back["top"] - left_top) > 3:
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"Back landed on the first board but not at its departure"
                           f" place: {left_top}px was left, {back['top']}px came back")}
    return {"status": t36_results.STATUS_PASS,
            "detail": (f"the picker journey added exactly one trail entry (depth"
                       f" {depth_first}->{depth}, entries {entries}) - the picker is"
                       f" never a destination - and Back from the second board landed"
                       f" on the first at its departure place ({left_top}px came back"
                       f" as {back['top']}px). The board->tasks->Back and window-close"
                       f" halves wait for the tasks surface and linked windows"
                       f" (recorded); the inspector->picker chain stays a pure-lane"
                       f" fixture by the scenario's own note")}


def check_s04_deactivation(request: RequestFn, pace: float = 1.0) -> dict:
    """S04 deactivation separates input from content - the felt
    behaviour, driven for real.

    Script: a board with a displayed figure; the figure's plain click
    (its primary act, enlarge) opens the inspection; while it is open,
    board scroll keys do NOTHING to the board beneath (no silent
    fall-through); the inspector's own + - = act (zoom steps, reset);
    Escape closes it, hands input back, and adds no trail entry.
    """
    import json as _json
    import time as _time

    if not t36_port.is_port(request):
        raise CheckStopped(
            "the old-app half is not grounded at baseline (the recorded registry"
            " reason): its inspector differs in surface and the differential never"
            " drove it; the port half carries the felt behaviour"
        )

    from t36_checks import _key, _ok, _premise, _sleep

    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    base = str(tracker.get("base") or "").replace("http://", "")
    tasks = tracker.get("tasks") or {}
    landing = next((t for t, e in tasks.items() if "landing" in str(e.get("message") or "")), None)
    _premise(landing is not None, "the landing seed (its figure) is missing from the tracker")
    t36_port.open_via_port_picker(request, f"{base}/seedtracker/{landing}@r2", pace)

    def readout() -> dict:
        return t36_port.readout(request)

    deadline = _time.monotonic() + 15.0 * pace
    while True:
        obs = readout()
        board = obs.get("board") or {}
        if board.get("status") == "loaded" and f"{landing}@r2" in str(board.get("location") or ""):
            break
        if _time.monotonic() >= deadline:
            raise CheckStopped(f"the landing board never loaded: {board.get('location')!r}")
        _sleep(pace, 0.4)

    # the displayed figure (hydrated with its asset)
    box = _json_evaluate(
        request,
        "const el = document.querySelector('.block-figure img.figure-img');"
        " if (!el) return null; const r = el.getBoundingClientRect();"
        " return JSON.stringify({x: r.x + r.width / 2, y: r.y + r.height / 2});",
    )
    _premise(isinstance(box, dict), "the landing board shows no hydrated figure (the premise needs a live asset)")

    # the primary act: a plain click enlarges
    _ok(request({"cmd": "input", "kind": "coords", "x": box["x"], "y": box["y"]}, timeout=30))
    _sleep(pace, 1.0)
    opened = readout()
    inspection = opened.get("inspection")
    if not (isinstance(inspection, dict) and inspection.get("zoom") == 1):
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the figure's plain click did not open the inspection: {inspection!r}"}

    # deactivation: board keys beneath do nothing (no scroll, no
    # activation), and the inspection stays open through them
    top_before = (opened.get("viewport") or {}).get("top")
    _key(request, "j")
    _sleep(pace, 0.3)
    _ok(request({"cmd": "input", "kind": "hotkey", "keys": ["shift", "g"]}, timeout=30))
    _sleep(pace, 0.8)
    beneath = readout()
    if (beneath.get("viewport") or {}).get("top") != top_before:
        return {"status": t36_results.STATUS_FAIL,
                "detail": (f"a board key reached the board beneath the inspection:"
                           f" viewport.top {top_before} -> {(beneath.get('viewport') or {}).get('top')}")}
    if not isinstance(beneath.get("inspection"), dict):
        return {"status": t36_results.STATUS_FAIL,
                "detail": "the inspection closed under board keys (it owns every stroke while open)"}

    # the inspector's own keys: + steps the zoom, = resets it
    _key(request, "+")
    _sleep(pace, 0.2)
    _key(request, "+")
    _sleep(pace, 0.2)
    zoomed = readout().get("inspection") or {}
    if zoomed.get("zoom") != 1.5:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the inspector's + did not step the zoom: {zoomed!r}"}
    _key(request, "=")
    _sleep(pace, 0.2)
    reset = readout().get("inspection") or {}
    if reset.get("zoom") != 1:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the inspector's = did not reset the zoom: {reset!r}"}

    # Escape closes: no trail entry, input back with the board loaded
    trail_before = _json.dumps(readout().get("trail"), sort_keys=True)
    _key(request, "escape")
    _sleep(pace, 0.6)
    closed = readout()
    if closed.get("inspection") is not None:
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"Escape did not close the inspection: {closed.get('inspection')!r}"}
    if _json.dumps(closed.get("trail"), sort_keys=True) != trail_before:
        return {"status": t36_results.STATUS_FAIL,
                "detail": "closing the inspection changed the trail (a temporary surface adds no entry)"}
    board = closed.get("board") or {}
    if board.get("status") != "loaded":
        return {"status": t36_results.STATUS_FAIL,
                "detail": f"the board beneath did not survive the inspection: {board.get('status')!r}"}

    return {"status": t36_results.STATUS_PASS,
            "detail": (f"the figure's plain click opened its inspection (part"
                       f" {inspection.get('part')!r}, zoom 1); while it was open the board's"
                       f" scroll keys moved nothing beneath (viewport.top {top_before} held)"
                       f" and the inspection owned every stroke; + stepped the zoom to 1.5,"
                       f" = reset it; Escape closed it with the trail unchanged and the board"
                       f" still loaded - deactivation separated input from content, the felt"
                       f" behaviour end to end")}
