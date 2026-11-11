#!/usr/bin/env python3
"""T-36 scenario check bodies: the executable scripts behind the
S-ids, plus the feed held-break check (F01), with the registry and
runner that turn one run into TAP-ready results.

Ground rules, all enforced here:
- Every body drives the app ONLY through the session lane's command
  surface (input/evaluate/windows/open-window/focus-window/
  observe-window/evaluate-window/feed/tracker) - real input, read-only
  observation of already-existing page state. No test-only setters,
  no product edits, no fabricated publishers.
- A premise the app cannot serve raises CheckStopped (the R3 rule):
  the check stops with the recorded gap, it never invents a verdict.
- An unexpected error inside the harness's own code is recorded as a
  stop too ("check machinery error") - it is not an app verdict and
  must not masquerade as red.
- Adapters ride the spec and are guard-checked (named-only) before a
  result records them.

The bodies were grounded against the baseline binaries at 4d59c35
(app w2jwbbx2..., sac g844avb2...) in the t36 lane mode; each body's
docstring names what was observed live and what verdict it therefore
expects at baseline.

Check ids: S01..S20 are the scenario groups; F01 is the words-feed
unknown-event check (the held break S19 expects red; A7 oracle -
judged against the agreed text, no prototype comparison).
"""

from __future__ import annotations

import json
import time
from typing import Callable

import t36_adapters
import t36_port
import t36_results
from t36_observation import CheckStopped, stop_result

RequestFn = Callable[[dict], dict]

DEFAULT_PACE = 1.0

# the driver injects the S20 relaunch callable per run (down, up-t36
# into the same run root); bodies call it only when their scenario
# names a relaunch, and app-only runs leave it unset
_RELAUNCHER: Callable[[], None] | None = None


def set_relauncher(relaunch: Callable[[], None] | None) -> None:
    global _RELAUNCHER
    _RELAUNCHER = relaunch


class _Premise(CheckStopped):
    """A premise gap: the script cannot run against what exists."""


def _ok(request: dict) -> dict:
    if not isinstance(request, dict) or not request.get("ok"):
        raise _Premise(f"the lane refused {json.dumps(request)[:200]}")
    return request


def _sleep(pace: float, seconds: float) -> None:
    time.sleep(seconds * pace)


# -- shared driving helpers (all through the lane surface) ---------------


DEFAULT_WATCH_SECONDS = 8.0


def _key(request: RequestFn, *keys: str) -> None:
    _ok(request({"cmd": "input", "kind": "key", "keys": list(keys)}))


def _type(request: RequestFn, target: str, text: str) -> None:
    _ok(request({"cmd": "input", "kind": "type", "target": target, "text": text}))


def _evaluate(request: RequestFn, script: str, timeout: float = 40.0) -> object:
    return _ok(request({"cmd": "evaluate", "script": script}, timeout=timeout)).get("value")


def _json_evaluate(request: RequestFn, script: str, timeout: float = 40.0) -> object:
    value = _evaluate(request, script, timeout)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as error:
            raise _Premise(f"the page answered off shape: {str(value)[:200]}") from error
    return value


def _windows(request: RequestFn) -> list[dict]:
    return _ok(request({"cmd": "windows"}, timeout=60)).get("windows") or []


def _receiver_readout(request: RequestFn, label: str, timeout: float = 60.0) -> dict:
    value = _ok(
        request({"cmd": "evaluate-window", "window": label, "script": READOUT_WINDOW_SCRIPT}, timeout=timeout)
    ).get("value")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise _Premise(f"window {label!r} answered off shape: {str(value)[:200]}") from error
    if not isinstance(value, dict):
        raise _Premise(f"window {label!r} answered no readout object")
    return value


READOUT_WINDOW_SCRIPT = (
    "const i = window.__saccadeApp && window.__saccadeApp.inspection();"
    " const loc = document.querySelector('[data-control=\"board:location\"]');"
    " return JSON.stringify({status: i ? i.resource.status : null,"
    " path: i ? i.resource.path : null, task: i && i.identity ? i.identity.task : null,"
    " stripLocation: loc ? loc.textContent : null,"
    " suiteOpenAt: typeof window.__suiteOpenAt,"
    " linkChip: (document.querySelector('[data-control=\"link:target\"]') || {}).textContent || null});"
)


def _wait_tracker_loaded(request: RequestFn, task: str, pace: float, timeout_s: float = 30.0) -> None:
    # pace scales the wall-clock bounds: 1 is real timing, 0 is the
    # fake-surface single pass
    deadline = time.monotonic() + timeout_s * pace
    last = None
    while True:
        last = _json_evaluate(
            request,
            "const i = window.__saccadeApp && window.__saccadeApp.inspection();"
            " return JSON.stringify({status: i ? i.resource.status : null, task: i && i.identity ? i.identity.task : null});",
        )
        if isinstance(last, dict) and last.get("status") == "loaded" and last.get("task") == task:
            return
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 1.0)
    raise _Premise(f"the tracker board {task} never loaded under the real server; last: {json.dumps(last)[:200]}")


def _select_by_chip(request: RequestFn, target_id: str, pace: float) -> str:
    """Enter select mode and type the DISPLAYED-case label of the chip
    whose data-target-id matches. Returns the label that was typed."""
    _key(request, "escape")
    _sleep(pace, 0.3)
    _key(request, "s")
    _sleep(pace, 0.4)
    chips = _json_evaluate(
        request,
        "return JSON.stringify([...document.querySelectorAll('.select-label')].map("
        "c => ({label: c.textContent, id: c.getAttribute('data-target-id'), cls: c.className})));",
    )
    if not isinstance(chips, list):
        raise _Premise("select mode painted no readable chips")
    chip = next((c for c in chips if c.get("id") == target_id), None)
    if chip is None:
        raise _Premise(f"no chip names {target_id!r}; chips: {json.dumps(chips)[:300]}")
    label = str(chip.get("label") or "")
    if not label:
        raise _Premise(f"target {target_id!r} carries an empty label")
    for letter in label:
        _key(request, letter)
        _sleep(pace, 0.25)
    _sleep(pace, 0.5)
    return label


def _premise(condition: object, message: str) -> None:
    if not condition:
        raise CheckStopped(message)


def _selected_now(request: RequestFn) -> dict:
    value = _json_evaluate(
        request,
        "const el = document.querySelector('.select-selected');"
        " return JSON.stringify({selected: !!el,"
        " id: el ? (el.getAttribute('data-control') || el.getAttribute('data-block-id')) : null});",
    )
    return value if isinstance(value, dict) else {}


# -- P1: the three required reds -----------------------------------------


def check_s18_linked_delivery(request: RequestFn, pace: float = DEFAULT_PACE) -> dict:
    """S18 linked delivery, app side, under real conditions.

    Grounded live at 4d59c35 (t36 lane, seeded tracker): selecting the
    citation (board:open-citation) and pressing o with one window
    creates the second suite window (open_suite_window works - it is
    enumerable through the R5 verbs) but the receiver's location never
    changes and window.__suiteOpenAt is 'undefined' - the shell evals
    a global no component defines, so every delivery path is a silent
    no-op. The i act's ask overlay appears and the answer is typed;
    the receiver still never receives.

    Script: open the citing board through the real seeded tracker;
    with exactly one window: select the citation, press o (a new
    window opens and o makes it the link); assert the receiver shows
    the cited revision. Then press i and answer the ask with the
    receiver's chip; assert delivery without the link being rewritten
    (the ask says so; the source's link chip is read before/after).
    Then press w; assert the new viewer shows the citation.
    Baseline verdict: FAIL - the receiver never receives on any path,
    with __suiteOpenAt undefined as the recorded evidence."""
    windows = _windows(request)
    labels = sorted(w["label"] for w in windows)
    if labels != ["main"]:
        raise _Premise(f"the linked-delivery script needs exactly one window at start; found {labels}")
    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    base = str(tracker.get("base") or "")
    citing = None
    for task_id, entry in (tracker.get("tasks") or {}).items():
        if "citing" in str(entry.get("message") or ""):
            citing = task_id
    if citing is None:
        raise _Premise(f"no citing seed task on the tracker: {json.dumps(tracker.get('tasks'))[:200]}")
    host = base.replace("http://", "")
    spec = f"{host}/seedtracker/{citing}"

    # open the citing board through the picker, real input throughout
    _key(request, "escape")
    _sleep(pace, 0.3)
    _key(request, " ", " ")
    _sleep(pace, 0.8)
    _type(request, "picker:input", spec)
    _sleep(pace, 0.5)
    _key(request, "enter")
    _wait_tracker_loaded(request, citing, pace)

    # the citation act: o. One window -> a new window opens and o
    # makes it the link (t-34@r6 change 3).
    label = _select_by_chip(request, "board:open-citation", pace)
    selected = _selected_now(request)
    if not selected.get("selected"):
        raise _Premise(f"the citation did not select by its displayed label {label!r}")
    _key(request, "o")
    _sleep(pace, 3.0)

    windows = _windows(request)
    new_labels = sorted(w["label"] for w in windows if w["label"] != "main")
    evidence: list[str] = []
    if not new_labels:
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": "after 'o' no second window existed at all (open_suite_window never ran); spec was " + spec,
        }
    receiver = new_labels[-1]
    readout = _receiver_readout(request, receiver)
    received = _delivered(readout)
    suite_open_at = readout.get("suiteOpenAt")
    if not received:
        evidence.append(
            f"o (new-window path): window {receiver!r} opened but shows"
            f" {readout.get('path')!r} (strip {readout.get('stripLocation')!r}),"
            f" never the citation {spec}; window.__suiteOpenAt is {suite_open_at!r} (written nowhere)"
        )
    # the source keeps focus and content; its location is still the citing board
    main_readout = _receiver_readout(request, "main")
    if main_readout.get("task") != citing:
        evidence.append(f"the source window moved too: main shows task {main_readout.get('task')!r}")

    # i: elsewhere once, without rewriting the link (the ask says so)
    link_before = main_readout.get("linkChip")
    label = _select_by_chip(request, "board:open-citation", pace)
    if not _selected_now(request).get("selected"):
        raise _Premise("the citation did not select for the i act")
    _key(request, "i")
    _sleep(pace, 1.5)
    ask = _json_evaluate(
        request,
        "const overlay = document.querySelector('[class*=\"jump-overlay\"], [class*=\"ask\"]');"
        " const self = document.querySelector('[class*=\"jump-chip-self\"]');"
        " return JSON.stringify({askVisible: !!overlay, askText: overlay ? (overlay.textContent || '').slice(0, 120) : null,"
        " selfChip: self ? self.textContent : null});",
    )
    if not isinstance(ask, dict) or not ask.get("askVisible"):
        evidence.append("i: the ask overlay never appeared (with two windows o/i must ask)")
    else:
        other_chip = _receiver_chip(request, receiver)
        if other_chip:
            _key(request, other_chip)
            _sleep(pace, 0.4)
            _key(request, "enter")
            _sleep(pace, 2.5)
            readout = _receiver_readout(request, receiver)
            if not _delivered(readout):
                evidence.append(
                    f"i (ask answered with the receiver's chip {other_chip!r}): the receiver still shows"
                    f" {readout.get('path')!r}; window.__suiteOpenAt is {readout.get('suiteOpenAt')!r}"
                )
            main_readout = _receiver_readout(request, "main")
            if main_readout.get("linkChip") != link_before:
                evidence.append(
                    f"i rewrote the saved link: {link_before!r} -> {main_readout.get('linkChip')!r}"
                )
        else:
            evidence.append("i: the receiver's chip was unreadable, the answer could not be typed")
    _key(request, "escape")

    # w: a new window, and the citation must arrive there
    label = _select_by_chip(request, "board:open-citation", pace)
    if not _selected_now(request).get("selected"):
        raise _Premise("the citation did not select for the w act")
    before = {w["label"] for w in _windows(request)}
    _key(request, "w")
    _sleep(pace, 3.0)
    after = {w["label"] for w in _windows(request)}
    fresh = after - before
    if not fresh:
        evidence.append("w: no new viewer window was created")
    else:
        w_label = sorted(fresh)[-1]
        readout = _receiver_readout(request, w_label)
        if not _delivered(readout):
            evidence.append(
                f"w: viewer {w_label!r} opened but shows {readout.get('path')!r}, never the citation;"
                f" window.__suiteOpenAt is {readout.get('suiteOpenAt')!r}"
            )

    if evidence:
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": "the receiver never receives (" + "; ".join(evidence) + ")",
        }
    return {
        "status": t36_results.STATUS_PASS,
        "detail": f"o, i and w all delivered {spec} to the right window; focus and links behaved",
    }


def _delivered(readout: dict) -> bool:
    """Delivery means the receiver's LOCATION moved to the cited
    revision (path or strip shows t-0@r1). identity.task is authored
    board metadata, never a location - a board declaring task: t-0
    must not false-pass this probe (caught live)."""
    return (
        str(readout.get("path") or "").endswith("t-0@r1")
        or str(readout.get("stripLocation") or "").endswith("t-0@r1")
    )


def _receiver_chip(request: RequestFn, label: str) -> str | None:
    readout = _ok(
        request({
            "cmd": "evaluate-window",
            "window": label,
            "script": "const c = document.querySelector('[class*=\"jump-chip-self\"]'); return c ? c.textContent : null;",
        }, timeout=40)
    ).get("value")
    return str(readout) if readout else None


def _words_snapshot(request: RequestFn) -> dict:
    """The feed's process-global state through the evaluate door:
    words_wait (the bounded hold) with a zero hold answers at once
    with {generation, words, connected, problem}. Grounded live: the
    disconnect after an unknown event lands here within ~0.5s at
    baseline, with the problem naming the closed stream."""
    value = _evaluate(
        request,
        "return window.__TAURI_INTERNALS__ ?"
        " window.__TAURI_INTERNALS__.invoke('words_wait', {since: 0, timeout: 0})"
        ".then(s => JSON.stringify(s), e => 'err ' + String(e).slice(0, 200)) : null;",
    )
    if not isinstance(value, str) or value.startswith("err"):
        raise _Premise(f"words_wait did not answer through the evaluate door: {value!r}")
    try:
        snapshot = json.loads(value)
    except ValueError as error:
        raise _Premise(f"words_wait answered off shape: {value[:200]}") from error
    if not isinstance(snapshot, dict):
        raise _Premise(f"words_wait answered off shape: {value[:200]}")
    return snapshot


def check_feed_unknown_event(request: RequestFn, pace: float = DEFAULT_PACE) -> dict:
    """F01: the words feed must survive an unknown event type (the
    held break S19 expects red; A7 - app-only oracle, judged against
    the agreed text, no prototype comparison).

    Grounded live at 4d59c35: the proxy holds the core socket and the
    app's feed subscribes (its verbatim subscribe line recorded). At
    4d59c35 words.rs breaks the stream on CoreEvent::Other - any
    unknown type ends the connection; words_wait answers connected:
    false with problem 'the core at ... closed the event stream'
    within ~0.5s (the table cleared, the 2s reconnect backoff running).

    Script: confirm a subscription exists; deliver a healthy status
    event and confirm it reaches the words table; deliver one unknown
    event type; watch the feed state for 8s. Baseline verdict: FAIL -
    the feed disconnects and the problem names the closed stream. A
    feed that skips unknown events stays connected and the check
    passes."""
    feed = _ok(request({"cmd": "feed", "action": "status"}, timeout=30)).get("feed") or {}
    if not feed.get("total_subscribes"):
        raise _Premise(f"the app's feed never subscribed to the proxy: {json.dumps(feed)[:200]}")
    _ok(request({"cmd": "feed", "action": "inject-status", "terminal": "t-1", "msg": "t36-feed-healthy", "state": "working"}))
    deadline = time.monotonic() + 6.0 * pace
    healthy_seen = False
    while True:
        snapshot = _words_snapshot(request)
        if "t36-feed-healthy" in json.dumps(snapshot):
            healthy_seen = True
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.5)
    if not healthy_seen:
        raise _Premise(f"the healthy status never reached the words table: {json.dumps(snapshot)[:250]}")
    generation_before = snapshot.get("generation")

    _ok(request({"cmd": "feed", "action": "inject-unknown", "event_type": "t36.unknown-probe"}, timeout=30))
    deadline = time.monotonic() + 8.0 * pace
    problem: str | None = None
    while True:
        snapshot = _words_snapshot(request)
        if snapshot.get("connected") is False and snapshot.get("problem"):
            problem = str(snapshot.get("problem"))
            break
        if time.monotonic() >= deadline:
            break
        _sleep(pace, 0.3)
    if problem:
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": (
                "the feed treated the unknown event type as the stream's end:"
                f" the feed disconnected ({problem!r}); generation"
                f" {generation_before} -> {snapshot.get('generation')}"
            ),
        }
    after = _ok(request({"cmd": "feed", "action": "status"}, timeout=30)).get("feed") or {}
    return {
        "status": t36_results.STATUS_PASS,
        "detail": (
            "the feed survived the unknown event type: still connected, no disconnect"
            f" within the watch window; connections live: {after.get('connections')},"
            f" subscriptions: {after.get('total_subscribes')}"
        ),
    }


def check_s12_label_case_fold(request: RequestFn, pace: float = DEFAULT_PACE) -> dict:
    """S12 label selection, case-fold probe, app side.

    Grounded live at 4d59c35: in select mode the uppercase chip 'B'
    (board:location) selects when typed as 'B' and does NOT select
    when typed as 'b' - the label case-fold fix is held, uncommitted
    (the 'labels need Shift' symptom). select.ts: case decides the
    namespace, so a typed lowercase letter cannot complete an
    uppercase label.

    Script: select mode; read the chips; take an uppercase (non-text)
    chip; type its label lowercased - different case than displayed.
    Baseline verdict: FAIL - nothing selects. A case-folded matcher
    selects the chip's target and the check passes."""
    _key(request, "escape")
    _sleep(pace, 0.3)
    _key(request, "s")
    _sleep(pace, 0.4)
    chips = _json_evaluate(
        request,
        "return JSON.stringify([...document.querySelectorAll('.select-label')].map("
        "c => ({label: c.textContent, id: c.getAttribute('data-target-id'), cls: c.className})));",
    )
    if not isinstance(chips, list) or not chips:
        raise _Premise("select mode painted no chips")
    upper = [c for c in chips if str(c.get("label") or "") and c.get("label") != str(c["label"]).lower()]
    if not upper:
        raise _Premise(f"no uppercase chip on this board probes the case fold: {json.dumps(chips)[:300]}")
    chip = upper[0]
    displayed = str(chip["label"])
    typed = displayed.lower()
    for letter in typed:
        _key(request, letter)
        _sleep(pace, 0.3)
    _sleep(pace, 0.5)
    selected = _selected_now(request)
    if selected.get("selected") and selected.get("id") == chip.get("id"):
        return {
            "status": t36_results.STATUS_PASS,
            "detail": f"typed {typed!r} and the target with displayed label {displayed!r} selected",
        }
    return {
        "status": t36_results.STATUS_FAIL,
        "detail": (
            f"typed {typed!r}; the displayed label is {displayed!r}"
            f" (target {chip.get('id')!r}); nothing selected"
            " - labels match only when typed in the displayed case (the case-fold fix is held)"
        ),
    }


# -- the registry and runner ----------------------------------------------


class CheckSpec:
    def __init__(
        self,
        id: str,
        name: str,
        body: Callable[[RequestFn, float], dict],
        lane: str = "session",
        adapters: tuple[str, ...] = (),
        mode: str = "t36",
    ) -> None:
        self.id = id
        self.name = name
        self.body = body
        self.lane = lane
        self.adapters = adapters
        self.mode = mode


CHECKS: dict[str, CheckSpec] = {
    spec.id: spec
    for spec in (
        CheckSpec("S18", "linked delivery delivers", check_s18_linked_delivery),
        CheckSpec("F01", "the words feed survives an unknown event type", check_feed_unknown_event),
        CheckSpec("S12", "label selection matches across case", check_s12_label_case_fold),
    )
}

# ids the checks registry may record beyond the scenario inventory
EXTRA_CHECK_IDS = ("F01",)


def run_check(request: RequestFn, spec: CheckSpec, pace: float = DEFAULT_PACE) -> dict:
    """One body, one verdict-shaped result. Stopped checks and check
    machinery errors are stops with their evidence; only the body's
    own verdict can be pass or fail."""
    try:
        # the flavor gate (the slice-2 adapter contract): on the port
        # flavor a check outside the served set stops with the recorded
        # premise instead of driving old-app selectors into the fresh
        # app; the served bodies branch their observation layer
        if t36_port.is_port(request):
            t36_port.ensure_served(spec.id)
        verdict = spec.body(request, pace)
    except CheckStopped as stop:
        return stop_result(spec.id, spec.lane, spec.name, stop.reason)
    except Exception as error:  # noqa: BLE001 - a harness bug is not an app verdict
        return stop_result(spec.id, spec.lane, spec.name, f"check machinery error (not an app verdict): {error!r:.200}")
    adapters = t36_adapters.named_for_result(spec.id, spec.adapters)
    return t36_results.make_result(
        spec.id, spec.lane, spec.name, str(verdict.get("status")), str(verdict.get("detail") or ""), adapters
    )


_REGISTERED = False


def ensure_registered() -> None:
    """Pull in the body modules once (the scenario bodies and the
    differential + stop registry register into CHECKS). Lazy: plain
    app-only runs never import them."""
    global _REGISTERED
    if _REGISTERED:
        return
    import t36_differential
    import t36_scenario_bodies

    # the scenario bodies first: the stop registry never overwrites a
    # registered body
    CHECKS.update({
        spec.id: spec for spec in (
            CheckSpec("S09", "one action, every invocation agrees", t36_scenario_bodies.check_s09_invocations_agree),
            CheckSpec("S10", "sequences go through the shipped matcher", t36_scenario_bodies.check_s10_sequences),
            CheckSpec("S13", "nested targets and citation", t36_scenario_bodies.check_s13_nested_citation_copy),
            CheckSpec("S15", "following current versus fixed revision", t36_scenario_bodies.check_s15_follows_current),
            CheckSpec("S20", "relaunch restores the persisted window surface", t36_scenario_bodies.check_s20_relaunch),
        )
    })
    t36_differential.register()
    _REGISTERED = True


def _with_transport_retry(request: RequestFn) -> RequestFn:
    """One retry on transient transport errors: real-conditions runs
    see the occasional connection reset/refusal between checks (the
    payload's per-connection loop); the retry is bounded to one and
    covers only the transport, never a refused answer."""
    def wrapped(payload: dict, timeout: float = 60.0) -> dict:
        try:
            return request(payload, timeout=timeout)
        except (ConnectionResetError, ConnectionRefusedError, RuntimeError) as error:
            if "connection" not in str(error).lower() and "closed" not in str(error).lower():
                raise
            time.sleep(1.0)
            try:
                return request(payload, timeout=timeout)
            except (ConnectionResetError, ConnectionRefusedError, RuntimeError) as second:
                raise CheckStopped(f"the session transport dropped twice ({second!r}); the lane or session is gone") from None
    return wrapped


def run_checks(request: RequestFn, check_ids: list[str], pace: float = DEFAULT_PACE) -> list[dict]:
    """The inventory runner: each named check, registry order for
    unknown ids refused, one result per check."""
    ensure_registered()
    request = _with_transport_retry(request)
    results: list[dict] = []
    for check_id in check_ids:
        spec = CHECKS.get(check_id)
        if spec is None:
            results.append(stop_result(check_id, "session", "unauthored check", "no executable body exists for this id yet (t-36 phase-2)"))
            continue
        results.append(run_check(request, spec, pace))
    return results
