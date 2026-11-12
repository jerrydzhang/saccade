#!/usr/bin/env python3
"""T-36 adapter set as data (part of R1's contract; the batch review
before any differential run).

Each adapter names its scenario ids and the exact input/readout
transformation, and is applied ONLY to the named checks - the guard
below is the enforcement, and every result record carrying an adapter
id is checked against it. Blanket normalization stays forbidden; an
adapter exists only where a ratified difference from the executed
prototype was demonstrated.

Verification status: a row is "unverified" until the executed
reference behaviour backs it (the design: prototype behaviour is
recorded at the first run). The revision rides every manifest so a
diff can name the adapter set it was judged under.
"""

from __future__ import annotations

ADAPTER_SET_REVISION = "t36-adapters/6"

KIND_INPUT = "input"
KIND_READOUT = "readout"
KIND_ORACLE = "oracle"
KINDS = (KIND_INPUT, KIND_READOUT, KIND_ORACLE)

VERIFIED = "verified"
UNVERIFIED = "unverified"


class AdapterMisapplied(ValueError):
    """An adapter applied to a check it does not name."""


def _adapter(id: str, name: str, kind: str, scenarios: tuple[str, ...], transformation: str, verification: str = UNVERIFIED, evidence: str = "") -> dict:
    return {
        "id": id,
        "name": name,
        "kind": kind,
        "scenarios": list(scenarios),
        "transformation": transformation,
        "verification": verification,
        "evidence": evidence,
    }


ADAPTERS: tuple[dict, ...] = (
    _adapter(
        "A1",
        "sequence-timeout",
        KIND_INPUT,
        ("S10",),
        "The app abandons a pending key sequence at SEQUENCE_TIMEOUT_MS=1000"
        " (value ratified t-36@r49) and the received keys then match their"
        " own bindings or are consumed; the prototype has no such abandon,"
        " so the differential input script gives the app side the bounded"
        " wait and the prototype side none. No normalization value is"
        " encoded.",
    ),
    _adapter(
        "A2",
        "find-entry",
        KIND_INPUT,
        ("S16",),
        "The prototype enters Find with space then /; the app enters with"
        " the bare /. The input script presses the Find entry each side"
        " actually binds; nothing else about Find is transformed.",
    ),
    _adapter(
        "A3",
        "no-row-cursor",
        KIND_READOUT,
        ("S12", "S17"),
        "The prototype task list carries a j/k row cursor; the app scrolls"
        " purely and selects by label (ratified t-36@r51). The readout"
        " drops the prototype's cursor state from the compared projection;"
        " labels and identity-selected rows still compare literally.",
    ),
    _adapter(
        "A4",
        "fold-reveal",
        KIND_READOUT,
        ("S08",),
        "The app unfolds enclosing containers to the actual anchor; the"
        " prototype does not. The readout accepts the unfold delta on the"
        " app side for this scenario only - the anchor shown must still be"
        " the actual target, never a folded group header.",
    ),
    _adapter(
        "A5",
        "trail-flags",
        KIND_READOUT,
        ("S14",),
        "The prototype's go()/back() save and restore history and"
        " marginalia flags; the app never restores viewer state (the trail"
        " is a map, not a time machine). The readout drops those restored"
        " flag fields from the prototype's trail comparison.",
    ),
    _adapter(
        "A6",
        "activity-place",
        KIND_READOUT,
        ("S17",),
        "The prototype's task activity is an overlay; the app treats it as"
        " a location retained by Back/Forward (agreed change, judged"
        " against the ratified text). The readout compares the app side"
        " against the agreed location semantics, not the overlay.",
    ),
    _adapter(
        "A7",
        "words-feed",
        KIND_ORACLE,
        (),
        "Not an adapter: app-only oracle. The Herdr words feed has no"
        " prototype counterpart, so feed checks are judged against the"
        " agreed text alone under real conditions (real core or the"
        " lane-controlled core port). Encoded here so the empty scenario"
        " set is explicit: no differential comparison may claim it.",
        verification=VERIFIED,
        evidence="t-36@r54 adapter table: 'Not an adapter: app-only oracle'",
    ),
    _adapter(
        "P1",
        "port-readout-resolver",
        KIND_READOUT,
        ("S20",),
        "On the port flavor the checks' observation layer reads"
        " window.__saccadePort.readout() (the port's own observation"
        " contract) instead of the old inspection()/DOM scraping; the"
        " oracles stay untouched. Fields the shown build cannot produce"
        " REQUIRE_FIELDS-stop naming where they land (trail/viewport ride"
        " the Board instrument, slice 2) - the old STOP rule, port side.",
    ),
    _adapter(
        "P3",
        "port-trail-readout",
        KIND_READOUT,
        ("S05", "S07"),
        "On the port flavor the S05/S07 observation layer reads the"
        " window.__saccadePort.readout() trail, viewport and location"
        " (the port's own observation contract, chunk E) and drives"
        " Back/Forward as the app's pinned chords (Ctrl-o/Ctrl-i - the"
        " old app's own bindings) through explicit hotkey synthesis."
        " The old-app lane stops: its inspection() projects no trail"
        " entries, back/forward state or return-place records (the"
        " S14-class readout gap), so the journeys cannot observe"
        " through it; the oracles stay untouched, the flip is the"
        " recorded readout gap.",
    ),
    _adapter(
        "P4",
        "port-elsewhere",
        KIND_INPUT,
        ("S18",),
        "On the port flavor the OPEN acts ride the citation's own menu"
        " (f/o/i/w/y, the r90 KINDS order) and the ask is the window's"
        " ask field; new windows receive through their state's birth"
        " location (the receiver restores it through its own model -"
        " never an eval into a half-loaded webview), live windows"
        " through the shell's open door. The old-app half keeps its"
        " recorded FAIL evidence (the receiver never receives on any"
        " path); the oracle is untouched and the flip is expected.",
    ),
    _adapter(
        "P6",
        "port-figure-inspection",
        KIND_INPUT,
        ("S04",),
        "On the port flavor the figure inspection is the window's own"
        " temporary surface (S04): the figure's primary act enlarges"
        " (the prototype's x-first figure kind), the inspector owns"
        " every stroke while open (+ - = and Escape; anything else"
        " consumed inert - no fall-through to the board beneath, which"
        " keeps its updates), and closing adds no trail entry. The"
        " old-app half keeps its recorded registry stop (never"
        " grounded at baseline); the felt behaviour is the port's.",
    ),
    _adapter(
        "P5",
        "port-tasks-surface",
        KIND_INPUT,
        ("S17",),
        "On the port flavor the tasks surface is the port's own Tasks"
        " instrument (the i-21 slice): the pipeline page restated as"
        " rows with task identities, label chips over visible rows,"
        " the row act menu (f primary, then the window's o/i/w), and"
        " the surface-routing readout (surface, the tasks section, the"
        " per-surface trail). The old-app half keeps its recorded"
        " FAIL/stop evidence (activity as the superseded overlay); the"
        " agreed activity-as-location judgment is unchanged.",
    ),
    _adapter(
        "P2",
        "port-model-open",
        KIND_INPUT,
        ("S20",),
        "On the port flavor an open drives the port's own model open door"
        " (__suiteOpenAt - the same door the shared shell evals for"
        " delivered opens; the receiving window opens through its own"
        " model) instead of the old-app picker flow. Slice-1 serves no"
        " picker, keymap beyond the routing seam, menu, or copy surface:"
        " checks needing those stop with the recorded premise"
        " (t36_port.ensure_served), never drive old-app selectors.",
    ),
)

ADAPTER_IDS = tuple(a["id"] for a in ADAPTERS)


def adapter_of(adapter_id: str) -> dict:
    for adapter in ADAPTERS:
        if adapter["id"] == adapter_id:
            return adapter
    raise AdapterMisapplied(f"unknown adapter id: {adapter_id!r} (set {ADAPTER_SET_REVISION})")


def adapters_for(scenario_id: str) -> tuple[str, ...]:
    """The adapters whose rows name this scenario. A7 names none by
    design: it is the recorded absence of a prototype comparison."""
    return tuple(a["id"] for a in ADAPTERS if scenario_id in a["scenarios"])


def assert_named_only(scenario_id: str, adapter_ids: tuple[str, ...] | list[str]) -> None:
    """The application guard: every adapter id must name the scenario it
    is applied to. Results and check scripts call this before an
    adapter's transformation touches anything."""
    for adapter_id in adapter_ids:
        adapter = adapter_of(adapter_id)
        if adapter["kind"] == KIND_ORACLE:
            raise AdapterMisapplied(f"{adapter_id} is an oracle record, not an adapter; it may not be applied")
        if scenario_id not in adapter["scenarios"]:
            raise AdapterMisapplied(
                f"{adapter_id} names {', '.join(adapter['scenarios']) or 'no checks'};"
                f" it may not be applied to {scenario_id}"
            )


def named_for_result(scenario_id: str, adapters: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate then return the adapter ids a result record may carry:
    exactly those that name the scenario."""
    assert_named_only(scenario_id, adapters)
    return tuple(adapters)
