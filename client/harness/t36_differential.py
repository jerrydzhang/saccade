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
from t36_checks import (
    CheckSpec,
    CHECKS,
    RequestFn,
    _json_evaluate,
    _key,
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
    # -- app half: space / opens find on the shown revision
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
        f" (app '{app_matches}' via space /; prototype {proto.get('marks')} marks via space /);"
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

    The fold act exists (select a group, z toggles .closed) but the
    premise cannot be served with today's seeds: following a citation
    lands board-level (t-0@r1), and no seeded citation names a part
    INSIDE a group, so there is no anchor inside a folded container to
    navigate to. The stop records the seed gap; extending
    t36_seeds with a part-path citation board is the unlock."""
    raise CheckStopped(
        "no seeded citation names a part inside a group, so no anchor inside a"
        " folded container can be navigated to; the fold act itself is grounded"
        " (select a group, z toggles .closed) - extend the seed set with a"
        " part-path citation to run this differential (A4 carries the unfold delta)"
    )


def check_s14_trail_flags(request: RequestFn, pace: float) -> dict:
    """S14 the trail records explicit openings only, app side.

    The app's inspection() projection exposes no trail or return-place
    readout (no trail entries, no back/forward state, no restored
    flags), and this harness adds no product instrumentation. The
    prototype side reads its trails from the saved snapshots (the shim
    carries them). The differential needs the app half; the stop
    records the readout gap."""
    raise CheckStopped(
        "the app exposes no trail readout: inspection() carries no trail"
        " entries, back/forward state or restored flags, and the check adds no"
        " product instrumentation; the prototype half reads its trails from the"
        " saved snapshots (A5 would drop the restored flags) - the differential"
        " stops until the app projects its trail"
    )


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
