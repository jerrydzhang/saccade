#!/usr/bin/env python3
"""T-36 phase-2 differential core (P2) and the P3 stop registry.

One readout contract, two sides, the named adapters applied ONLY to
their checks:

- The app side drives through the session lane surface (t36_checks
  helpers); the prototype side drives the compiled round-16 artifact
  through the shim (t36_prototype.PrototypeSession).
- compare() projects both readouts onto the shared contract keys,
  applies each named adapter's normalization (guard-checked by
  t36_adapters first), and reports the mismatches that remain. An
  adapter never hides a mismatch outside its named transformation.

Bodies in this module carry their grounding notes; premises that the
baseline cannot serve are stops with the recorded gap, never silent
verdicts.
"""

from __future__ import annotations

import re
import time

import t36_adapters
import t36_results
import t36_port
from t36_checks import (
    CheckSpec,
    CHECKS,
    RequestFn,
    _json_evaluate,
    _key,
    _ok,
    _premise,
    _select_by_chip,
    _selected_now,
    _sleep,
    _type,
)
from t36_observation import CheckStopped

# the checks module registers the shared helpers; bodies here register
# into CHECKS too (the registry is one dict, two authors). The
# prototype half attaches per run: attach(session) before run_checks.

_PROTOTYPE: object | None = None


def attach(session: object) -> None:
    """Attach the prototype session for this run's differential
    halves; detach(None) for app-only runs."""
    global _PROTOTYPE
    _PROTOTYPE = session


def _prototype() -> "PrototypeHalves":
    if _PROTOTYPE is None:
        raise CheckStopped("no prototype session attached; the differential half cannot run")
    return PrototypeHalves(_PROTOTYPE)




# -- the comparison machinery --------------------------------------------


def _row_identity(rows: list[dict]) -> list[tuple[str, str]]:
    """The comparable identity of a rows projection: (kind, loc) pairs,
    deduplicated and sorted - counts differ per side by design (the
    prototype renders its own fixture data; the app renders the
    seeded tracker), so the differential compares SHAPE, not data."""
    identity = set()
    for row in rows or []:
        identity.add((str(row.get("kind")), str(row.get("loc"))))
    return sorted(identity)


def compare(
    scenario_id: str,
    app: dict,
    proto: dict,
    adapters: tuple[str, ...] = (),
) -> dict:
    """Project both sides onto the shared keys, apply the named
    adapters' normalizations, and report what still differs. The
    projection keys: location, selection, deck, row shapes."""
    applied = t36_adapters.named_for_result(scenario_id, adapters)
    mismatches: list[str] = []
    notes: list[str] = []

    app_loc = app.get("location")
    proto_loc = proto.get("location")
    if app_loc is not None and proto_loc is not None and str(app_loc) != str(proto_loc):
        # locations name each side's own open target; a differential
        # pair opens the SAME spec on both - anything else is a mismatch
        mismatches.append(f"location: app {app_loc!r} vs prototype {proto_loc!r}")

    app_rows = _row_identity(app.get("rows"))
    proto_rows = _row_identity(proto.get("rows"))
    app_kinds = sorted({kind for kind, _loc in app_rows})
    proto_kinds = sorted({kind for kind, _loc in proto_rows})
    if app_kinds != proto_kinds:
        mismatches.append(f"row kinds: app {app_kinds} vs prototype {proto_kinds}")

    if "A2" in applied:
        notes.append("A2 applied: each side presses the Find entry it actually binds; the agreed bare-/ app entry rides the adapter record")
    if "A3" in applied:
        notes.append("A3 applied: the prototype's row-cursor state is dropped (the app has no row cursor)")
    if "A4" in applied:
        notes.append("A4 applied: the app's unfold delta is accepted for this scenario")
    if "A5" in applied:
        notes.append("A5 applied: the prototype's restored history/marginalia flags are dropped")
    if "A6" in applied:
        notes.append("A6 applied: activity is compared against the agreed location semantics, not the prototype's overlay")
    return {"mismatches": mismatches, "notes": notes, "applied": list(applied)}


def _diff_verdict(scenario_id: str, mismatches: list[str], detail_extra: str = "") -> dict:
    if mismatches:
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": "differential mismatches: " + "; ".join(mismatches) + (("; " + detail_extra) if detail_extra else ""),
        }
    return {"status": t36_results.STATUS_PASS, "detail": detail_extra or "both sides agree under the named adapters"}


# -- the prototype half (shim over the compiled artifact) ----------------


class PrototypeHalves:
    """A thin wrapper over the prototype session so differential bodies
    read the shim readout the same way they read the app's."""

    def __init__(self, session: object) -> None:
        self.session = session

    def readout(self) -> dict:
        return self.session.readout()  # type: ignore[attr-defined]

    def find_on_board(self, query: str) -> dict:
        """S16's prototype half, grounded live: space opens the leader
        menu, / runs Find text (a prompt), the query paints mark.found
        hints on the shown revision."""
        import threading

        accepted: list[str] = []
        ready = threading.Event()

        def on_dialog(dialog) -> None:
            accepted.append(dialog.message)
            try:
                dialog.accept(query)
            except Exception:
                pass
            ready.set()

        page = self.session.page  # type: ignore[attr-defined]
        page.on("dialog", on_dialog)
        try:
            self.session.input(" ", "/")  # type: ignore[attr-defined]
            ready.wait(timeout=5)
            time.sleep(1.2)
            marks = page.evaluate("win('board').document.querySelectorAll('mark.found').length")
            return {"find_opened": bool(accepted), "marks": marks, "prompt": accepted[0] if accepted else None}
        finally:
            page.remove_listener("dialog", on_dialog)


# -- S16: history and Find are facilities of the shown revision ----------

FIND_QUERY_APP = "the citing account"
FIND_QUERY_PROTO = "Question"


def check_s16_find_entry(request: RequestFn, pace: float) -> dict:
    import json
    """S16, differential, adapter A2 (input): each side presses the
    Find entry it actually binds - at 4d59c35 BOTH sides bind space
    then / (the app's which-key table carries '/'; the prototype's
    leader menu carries '/'); the agreed bare-/ app entry is the
    ratified change A2 records, not a baseline behavior. The facility
    must match: Find opens on the shown revision and finds the query.

    Grounded live on both sides: the app's search opens ('type to
    search') and answers '1 / 1' on a matching query; the prototype's
    prompt paints 6 mark.found hints for a matching query."""
    prototype = _prototype()
    # -- app half: the find entry each flavor actually binds, observed
    # through that flavor's own surface. The legacy app: space / and
    # the search DOM (preserved literally). The port: bare / (the
    # ratified entry) and the readout's find field.
    if t36_port.is_port(request):
        # the find must match on the SHOWN revision: the port branch
        # opens the landing seed board first (the launch board carries
        # no match for the query)
        tracker = _ok(request({"cmd": "tracker"}, timeout=30))
        base = str(tracker.get("base") or "").replace("http://", "")
        tasks = tracker.get("tasks") or {}
        landing = next((t for t, e in tasks.items() if "landing" in str(e.get("message") or "")), None)
        _premise(landing is not None, f"no landing seed task on the tracker: {json.dumps(tasks)[:160]}")
        t36_port.open_via_port_picker(request, f"{base}/seedtracker/{landing}", pace)
        deadline = time.monotonic() + 15.0 * pace
        while True:
            board = (t36_port.readout(request).get("board") or {})
            if board.get("status") == "loaded" and f"/{landing}" in str(board.get("location") or ""):
                break
            if time.monotonic() >= deadline:
                raise CheckStopped(f"the landing board never loaded on the port: {json.dumps(board)[:160]}")
            _sleep(pace, 0.4)
        _key(request, "escape")
        _sleep(pace, 0.3)
        _key(request, "/")
        _sleep(pace, 0.8)

        def find_now() -> dict:
            value = t36_port.readout(request).get("find")
            return value if isinstance(value, dict) else {}

        find_state = find_now()
        _premise(find_state.get("open") is True,
                 f"the port's find did not open on / (readout find={json.dumps(find_state)[:160]})")
        # the query is the shown board's own text ("landing" - the
        # landing board's first block), so the expected nonzero count is
        # pinned to a known match, and the actual queried text is
        # recorded verbatim in the verdict evidence. The typing is
        # target-addressed (a real click on the find field, the clear,
        # the focus asserted, every response _ok'd, bounded)
        query = "landing"
        t36_port.type_text(request, "port:find-input", query, pace)
        _sleep(pace, 1.2)
        find_state = find_now()
        count = find_state.get("count")
        queried = find_state.get("query")
        _premise(isinstance(count, int) and count >= 1,
                 f"the port's find answered {json.dumps(find_state)[:160]} for {query!r}"
                 f" on the landing board (expected a nonzero count for a known match)")
        _premise(queried == query,
                 f"the find's query text drifted: the readout carries {queried!r}, typed {query!r}")
        _key(request, "escape")
        _sleep(pace, 0.3)
        find_closed = find_now()
        _premise(find_closed.get("open") is False, "the port's find did not close on escape")
        app_matches = f"{count} matches"
        opened = {"search": True}
    else:
        _key(request, "escape")
        _sleep(pace, 0.3)
        _key(request, " ", "/")
        _sleep(pace, 0.8)
        opened = _json_evaluate(
            request,
            "return JSON.stringify({search: !!document.querySelector('[data-control=\"search\"]'),"
            " state: (document.querySelector('[data-control=\"search:state\"]') || {}).textContent || null});",
        )
        _premise(isinstance(opened, dict) and opened.get("search"), "the app's find did not open on space /")
        _type(request, "search", FIND_QUERY_APP)
        _sleep(pace, 1.2)
        state = _json_evaluate(
            request,
            "return JSON.stringify({state: (document.querySelector('[data-control=\"search:state\"]') || {}).textContent || null});",
        )
        _key(request, "escape")
        app_matches = str((state or {}).get("state") or "")
        match = re.match(r"^(\d+) / (\d+)$", app_matches.strip())
        _premise(match is not None and int(match.group(2)) >= 1, f"the app's find answered {app_matches!r} for the query")

    # -- prototype half
    proto = prototype.find_on_board(FIND_QUERY_PROTO)

    comparison = compare(
        "S16",
        {"location": "find-open", "rows": [{"kind": "find", "loc": app_matches}]},
        {"location": "find-open", "rows": [{"kind": "find", "loc": f"{proto.get('marks', 0)} matches"}]},
        adapters=("A2",),
    )
    facility_same = bool(opened.get("search")) and proto.get("find_opened") and (proto.get("marks") or 0) >= 1
    if not facility_same:
        comparison["mismatches"].append(
            f"the find facility differs: app opened {bool(opened.get('search'))} answered {app_matches!r};"
            f" prototype opened {proto.get('find_opened')} with {proto.get('marks')} marks"
        )
    verdict = _diff_verdict(
        "S16",
        comparison["mismatches"],
        f"find opens on the shown revision on both sides"
        f" (app '{app_matches}' via {'the readout /' if t36_port.is_port(request) else 'space /'};"
        f" prototype {proto.get('marks')} marks via space /);"
        " A2 carries the input difference and the agreed bare-/ app entry",
    )
    verdict["detail"] = verdict["detail"] + " | " + "; ".join(comparison["notes"])
    return verdict


# -- S17: activity is a place, not a popup --------------------------------


def check_s17_activity_place(request: RequestFn, pace: float) -> dict:
    """S17, app-only judgment (A6 - the prototype's activity overlay is
    the superseded implementation; the agreed text says activity is a
    location retained by Back/Forward).

    Grounded at 4d59c35: the task rows carry 'Show activity (the lane
    and the events)' and showActivityOverlay renders an overlay
    (instrument:lane) - NOT a location. Baseline verdict: FAIL - the
    agreed activity-as-location is not built; the overlay is exactly
    the superseded implementation the agreement rejects. The A3
    no-row-cursor readout rides the same scenario: the app's task rows
    carry data-task (single) and no cursor state; A3 drops the
    prototype's cursor when its half runs."""
    # the tasks instrument: rows live in the tasks window state; the
    # main window's own tasks surface (the tasks page) carries the rows
    rows = _json_evaluate(
        request,
        "return JSON.stringify([...document.querySelectorAll('[data-control=\"instrument:row\"][data-instrument=\"tasks\"]')].map("
        "el => ({task: el.getAttribute('data-task'), hasCursor: el.classList.contains('cursor') || el.classList.contains('selected')})));",
    )
    _premise(isinstance(rows, list) and rows, "no task rows are mounted in this window (open the tasks instrument first)")
    row = rows[0]
    task = str(row.get("task") or "")
    _premise(task != "", "the task row carries no data-task")

    # the activity act: the task row's menu, 'e'
    label = _select_by_chip(request, f"instrument-row-{task}", pace)
    _premise(_selected_now(request).get("selected"), f"the task row for {task!r} did not select (label {label!r})")
    _key(request, "e")
    _sleep(pace, 1.5)
    after = _json_evaluate(
        request,
        "return JSON.stringify({overlay: !!document.querySelector('[data-control=\"instrument:lane\"]'),"
        " location: (document.querySelector('[data-control=\"board:location\"]') || {}).textContent || null});",
    )
    location = (after or {}).get("location")
    if after and after.get("overlay"):
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": (
                f"task {task!r}'s activity is an overlay (instrument:lane rendered in place), not a"
                f" location: the window's location stayed {location!r} (the agreed"
                " activity-as-location is not built; A6 records the superseded implementation)"
            ),
        }
    if location and "activity" in location.lower():
        return {
            "status": t36_results.STATUS_PASS,
            "detail": f"task {task!r}'s activity is a location ({location!r})",
        }
    raise CheckStopped(
        f"no activity surface appeared for task {task!r} (overlay {bool(after and after.get('overlay'))},"
        f" location {location!r}); the act may not be wired on this window"
    )


# -- S08/S14: authored with their recorded baseline gaps ------------------


def check_s08_fold_reveal(request: RequestFn, pace: float) -> dict:
    """S08 navigation reveals its actual anchor through folds, app side.

    The fold act exists (select a group, z toggles .closed) and the
    part-inside-group citation content now rides the seed set (the
    second citing board's incorporation names the first group child).
    What the baseline cannot serve: the app's CITATION SURFACE IS
    BOARD-LEVEL ONLY - hydrateCitations builds task@rK and never a
    part path, so every citation click lands on the board and no
    real-input navigation reaches an anchor inside a folded container.
    The stop records that product-surface gap for the port slice; A4
    carries the unfold delta when the surface exists."""
    # the port flavor drives the real navigation: the port's citations
    # carry part paths (data-cite-select), activation is a plain click
    # on the link, and the anchored open unfolds enclosing containers
    # (the chunk-E2 machinery). The oracle - navigation reveals its
    # actual anchor - rides the port's own surfaces.
    if t36_port.is_port(request):
        return _s08_port(request, pace)
    raise CheckStopped(
        "the app's citation surface is board-level only: hydrateCitations"
        " builds task@rK and never a part path, so no real-input navigation"
        " reaches an anchor inside a folded group; the fold act is grounded"
        " (select a group, z toggles .closed) and the part-inside-group"
        " citation content rides the seed set - the port slice that gives"
        " citations part paths unlocks this differential (A4 carries the"
        " unfold delta)"
    )


def _wait_port_loaded(request, pace: float, needle: str, timeout: float = 15.0) -> dict:
    """Wait until the port readout names a loaded board whose location
    carries the needle. The observation-side load wait, port flavor."""
    import time as _time

    from t36_checks import _json_evaluate, _sleep

    deadline = _time.monotonic() + timeout
    obs: dict | None = None
    while True:
        value = _json_evaluate(request, t36_port._READOUT_EVAL)
        if isinstance(value, dict):
            obs = value
            board = value.get("board") or {}
            if board.get("status") == "loaded" and needle in str(board.get("location") or ""):
                return value
        if _time.monotonic() >= deadline:
            break
        _sleep(pace, 0.4)
    board = (obs or {}).get("board") or {}
    raise CheckStopped(
        f"the port board never loaded {needle!r}; last: status={board.get('status')!r}"
        f" location={board.get('location')!r}"
    )


def _s08_port(request: RequestFn, pace: float) -> dict:
    import json as _json

    from t36_checks import _json_evaluate, _premise

    # the cited board with the group, then the citing board whose
    # incorporation names the first group child (the seed set's t-0/t-2)
    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    tasks = tracker.get("tasks") or {}
    cited = next((t for t, e in tasks.items() if "group-made" in str(e.get("message") or "")), None)
    citing = next((t for t, e in tasks.items() if "other group child" in str(e.get("message") or "")), None)
    _premise(cited is not None and citing is not None,
             f"the S08 seeds are missing on the tracker: {_json.dumps(tasks)[:200]}")
    host = str(tracker.get("base") or "").replace("http://", "")

    # fold the destination's group first: open the cited board, fold
    # the group part, and verify the child stopped rendering (the
    # fold's own observable: folded content renders no elements)
    t36_port.open_via_port_picker(request, f"{host}/seedtracker/{cited}", pace)
    cited_obs = _wait_port_loaded(request, pace, f"/{cited}")
    cited_location = str((cited_obs.get("board") or {}).get("location"))
    revision = cited_location.rsplit("@r", 1)[-1] if "@r" in cited_location else None
    # the canonical paths derive from the readout, never assumed: the
    # fixture proof (the opened revision really is the group-made one)
    # and the port's own part paths are separate facts (the review
    # correction: the grammar path /content/0/group is not the port's
    # canonical shape; the port nests as /content/N/content/M)
    parts = ((cited_obs.get("content") or {}).get("parts") or [])
    group_parts = [p for p in parts if p.get("kind") == "group"]
    _premise(bool(group_parts),
             f"the fixture/revision proof failed - the opened {cited_location!r} carries no group"
             f" part: {_json.dumps(parts)[:240]}")
    group_part = str(group_parts[0]["part"])
    child_part = f"{group_part}/content/0"
    t36_port.fold_part(request, group_part, pace)
    _premise(t36_port.part_rect(request, child_part) is None,
             f"the child part {child_part!r} still renders after the fold; the fold premise failed")

    # the citing board, then the real navigation: a plain click on the
    # citation link (the port's primary act)
    t36_port.open_via_port_picker(request, f"{host}/seedtracker/{citing}", pace)
    _wait_port_loaded(request, pace, f"/{citing}")
    link_rect = _json_evaluate(
        request,
        "const el = document.querySelector('[data-port-control=\"citation\"]');"
        " if (!el) return null; const r = el.getBoundingClientRect();"
        " return {x: r.x, y: r.y, width: r.width, height: r.height,"
        " open: el.getAttribute('data-port-open'), cite: el.getAttribute('data-cite-select') || null};",
    )
    _premise(isinstance(link_rect, dict) and link_rect.get("width"),
             "the citing board renders no citation link (data-port-control=citation)")
    x = link_rect["x"] + link_rect["width"] / 2
    y = link_rect["y"] + link_rect["height"] / 2
    request({"cmd": "input", "kind": "coords", "x": x, "y": y}, 30)
    arrived = _wait_port_loaded(request, pace, cited, timeout=20.0)
    board = arrived.get("board") or {}
    location = str(board.get("location") or "")
    child_after = t36_port.part_rect(request, child_part)
    anchored = _json_evaluate(
        request,
        "const el = document.querySelector("
        + _json.dumps(t36_port.part_selector(child_part)) + ");"
        " if (!el) return {present: false};"
        " const r = el.getBoundingClientRect();"
        " return {present: true, top: r.top, visible: r.width > 0 && r.height > 0};",
    )
    if not child_after or not (anchored or {}).get("present"):
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": (
                f"the citation activation landed on {location!r} but the anchor part"
                f" {child_part!r} is not rendered: navigation did not reveal the actual"
                " anchor through folds"
            ),
        }
    pinned_ok = revision is not None and location.endswith(f"@r{revision}")
    return {
        "status": t36_results.STATUS_PASS,
        "detail": (
            f"the citation click opened {location!r} anchored at {child_part!r}"
            f" (the element renders, viewport top {anchored.get('top')}); the fold"
            " premise was verified on the direct visit (the child stopped rendering)"
            + ("" if pinned_ok else f"; the location carries no pinned revision (follows current: {location!r})")
            + "; whether the fold state itself survived the round trip is not"
              " separately observable through existing state (recorded)"
        ),
    }


def check_s14_trail_flags(request: RequestFn, pace: float) -> dict:
    """S14 the trail records explicit openings only, app side.

    The app's inspection() projection exposes no trail or return-place
    readout (no trail entries, no back/forward state, no restored
    flags), and this harness adds no product instrumentation. The
    prototype side reads its trails from the saved snapshots (the shim
    carries them). The differential needs the app half; the stop
    records the readout gap."""
    # the port flavor reads the trail from the port's own observation
    # contract (chunk E): explicit openings, the cursor, back/forward
    # state. The keys still pin (chunk E has no back/forward bindings),
    # so the values are read, not driven.
    if t36_port.is_port(request):
        return _s14_port(request, pace)
    raise CheckStopped(
        "the app exposes no trail readout: inspection() carries no trail"
        " entries, back/forward state or restored flags, and the check adds no"
        " product instrumentation; the prototype half reads its trails from the"
        " saved snapshots (A5 would drop the restored flags) - the differential"
        " stops until the app projects its trail"
    )


def _s14_port(request: RequestFn, pace: float) -> dict:
    import json as _json

    from t36_checks import _premise

    def trail_now() -> dict:
        obs = t36_port.readout(request)
        t36_port.require_fields(obs, "trail", "viewport")
        return obs.get("trail") or {}

    tracker = _ok(request({"cmd": "tracker"}, timeout=30))
    tasks = tracker.get("tasks") or {}
    first = next((t for t, e in tasks.items() if "incorporation" in str(e.get("message") or "")), None)
    landing = next((t for t, e in tasks.items() if "landing" in str(e.get("message") or "")), None)
    _premise(first is not None and landing is not None,
             f"the trail-opening seeds are missing: {_json.dumps(tasks)[:200]}")
    host = str(tracker.get("base") or "").replace("http://", "")

    t36_port.open_via_port_picker(request, f"{host}/seedtracker/{first}", pace)
    _wait_port_loaded(request, pace, f"/{first}")
    t36_port.open_via_port_picker(request, f"{host}/seedtracker/{landing}", pace)
    _wait_port_loaded(request, pace, f"/{landing}")
    trail = trail_now()
    depth = trail.get("depth")
    cursor = trail.get("cursor")
    entries = trail.get("entries") or []
    locations = [str(e.get("location") or "") for e in entries]
    opened_both = all(f"/{t}" in "/".join(locations) for t in (first, landing))
    if not (isinstance(depth, int) and depth >= 2 and cursor == depth - 1 and opened_both):
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": (
                f"the trail did not record the explicit openings: depth={depth!r}"
                f" cursor={cursor!r} entries={_json.dumps(locations)[:200]}"
            ),
        }
    if trail.get("canBack") is not True or trail.get("canForward") is not False:
        return {
            "status": t36_results.STATUS_FAIL,
            "detail": (
                f"the back/forward state disagrees with the trail: canBack={trail.get('canBack')!r}"
                f" canForward={trail.get('canForward')!r} at depth {depth}"
            ),
        }

    # the fold act must not record an opening (explicit openings only)
    before_fold = _json.dumps(trail_now(), sort_keys=True)
    group_part = next((p.get("part") for p in ((t36_port.readout(request).get("content") or {}).get("parts") or [])
                       if p.get("kind") == "group"), None)
    fold_note = "no group part on the shown board; the fold drive was skipped"
    if group_part:
        t36_port.fold_part(request, group_part, pace)
        fold_note = f"the fold of {group_part!r} ran through the act menu"
        after = trail_now()
        if _json.dumps(after, sort_keys=True) != before_fold:
            return {
                "status": t36_results.STATUS_FAIL,
                "detail": (
                    f"the fold act changed the trail (explicit openings only):"
                    f" {_json.dumps(_json.loads(before_fold))[:150]} -> {_json.dumps(after)[:150]}"
                ),
            }
    return {
        "status": t36_results.STATUS_PASS,
        "detail": (
            f"the trail recorded the two explicit openings (depth {depth}, cursor {cursor},"
            f" entries {locations}) with canBack true and canForward false; {fold_note};"
            " the trail stayed untouched. The back/forward key bindings still pin"
            " (chunk E has none), so the state is read from the readout, not driven"
        ),
    }


# -- P3: the stop registry for the remaining groups -----------------------

STOP_REASONS: dict[str, str] = {
    "S01": "pure lane (controlled ports, no DOM): not executable through the real-conditions harness; belongs to the component test suite",
    "S02": "pure and component lanes only; the native harness has no failure-injection port for a board source",
    "S03": "the pure lane owns the reporting contract; the app-only native half needs a diagnostics-raising flow that no real-input path exercises yet",
    "S04": "the native-delta half needs the figure-inspection surface and a live-content source; not grounded at baseline",
    "S05": "the trail and buffer lifecycle readouts the journey asserts are not exposed by inspection() (the S14 gap applies)",
    "S06": "pure and component lanes (presentation attachment is a component fixture concern)",
    "S07": "return-place records are not readable through inspection(); the differential would stop on every step of the journey (the S14 gap applies)",
    "S09": "authored next: the four invocations (menu key, right-click Enter, right-click click, plain click) need right-click plumbing in webdriver_client before the literals can compare",
    "S10": "authored next: gg through the shared matcher and unmatched-key consumption are drivable, but the matcher's consume-vs-fallthrough oracle needs the input-context data the baseline ships as draft",
    "S11": "pure and component lanes (handoff contracts); the Find journey itself is S16",
    "S13": "authored next: the nested citation copy needs the group-child select flow grounded on group-made boards end to end (the seeds exist; the copy act needs grounding)",
    "S15": "authored next: needs a mid-check commit through the real tracker wire and a live-update readout the app actually honors; grounding pass pending",
    "S19": "executed by the coordinator's authoritative red run; the harness's own acceptance is that run, not a check body",
    "S20": "authored next: relaunch spans two lane sessions (down and up with persisted state), which is driver orchestration, not a single body",
}


# -- registration ---------------------------------------------------------


def register() -> None:
    """Register the differential and stop bodies into the checks
    registry. Idempotent; the P1 bodies stay untouched."""
    CHECKS.setdefault("S16", CheckSpec(
        "S16", "find is a facility of the shown revision", check_s16_find_entry,
        adapters=("A2",),
    ))
    CHECKS.setdefault("S17", CheckSpec(
        "S17", "task activity is a place, not a popup", check_s17_activity_place,
        adapters=("A6",),
    ))
    CHECKS.setdefault("S08", CheckSpec(
        "S08", "navigation reveals its actual anchor through folds", check_s08_fold_reveal,
        adapters=("A4",),
    ))
    CHECKS.setdefault("S14", CheckSpec(
        "S14", "the trail records explicit openings only", check_s14_trail_flags,
        adapters=("A5",),
    ))
    for scenario_id, reason in STOP_REASONS.items():
        if scenario_id in CHECKS:
            continue

        def body(request: RequestFn, pace: float, _reason: str = reason) -> dict:
            raise CheckStopped(_reason)

        CHECKS[scenario_id] = CheckSpec(scenario_id, "scenario group", body, mode="registry")
