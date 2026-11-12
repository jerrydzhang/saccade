"""The fresh app-side adapter (t-36 slice 2, t-36@r67 contract): maps
the t36 checks to the port's observation contract - the checks' DRIVING
AND OBSERVATION layer only; the oracles stay untouched.

The port's surface is `window.__saccadePort.readout()` (the port's own
observation contract, t-36 slice 1): a machine-readable projection of
the same state the human UI renders. The port's content structure is
data-port-part / data-port-kind (the port's own scheme); its open
surface is the model open door `__suiteOpenAt` (the same door the
shared shell evals for delivered opens - the receiving window owns the
open through its own model).

Slice-1 fields live today (app/window/board/errors/render/content/
diagnostics/doors). The trail and viewport fields land with the Board
instrument (slice 2): the adapter REQUIRES them exactly like the old
STOP rule - a check that needs one stops naming the field; it never
guesses (require_fields).

Flavor discipline: the harness detects the surface at run time. The
port flavor serves a subset of the inventory today; every other check
stops with the recorded premise instead of driving old-app selectors
into a fresh app. Old-app selector names are adapter material, never
the port contract.
"""

import json

from t36_observation import CheckStopped

PORT_SURFACE = "__saccadePort"
OPEN_DOOR = "__suiteOpenAt"

# the readout fields the Board instrument's slice-2 chunks carried
# (trail and viewport live since chunk E, t-36@r70): a check needing
# one requires it, and a missing field names the stale build - the
# stop says the shown build predates the field, never guesses
SLICE2_FIELDS = ("trail", "viewport")

_FLAVOR_PROBE = (
    f"return typeof window.{PORT_SURFACE} === 'object' && window.{PORT_SURFACE} !== null"
    f" && typeof window.{PORT_SURFACE}.readout === 'function';"
)
_READOUT_EVAL = f"return JSON.stringify(window.{PORT_SURFACE}.readout());"


def is_port(request) -> bool:
    """True when the shown app exposes the port's observation surface.
    One bounded probe; the answer is a property of the binary, so the
    caller may cache it per role."""
    from t36_checks import _evaluate

    return bool(_evaluate(request, _FLAVOR_PROBE))


def readout(request) -> dict:
    """The port's readout, parsed. A non-dict answer is a premise gap,
    never a guess."""
    from t36_checks import _evaluate

    value = _evaluate(request, _READOUT_EVAL)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise CheckStopped(f"the port readout answered off shape: {str(value)[:200]}") from error
    if not isinstance(value, dict):
        raise CheckStopped(f"the port readout is not an object: {type(value).__name__}")
    return value


def require_fields(obs: dict, *fields: str) -> None:
    """The REQUIRE_FIELDS-stop rule: a field the shown build cannot
    produce (absent, or null where a value is required) stops the check
    naming the field. trail/viewport have landed (chunk E); a missing
    one means the shown build predates it."""
    for field in fields:
        if field not in obs or obs[field] is None:
            note = "the shown port build predates chunk E; rebuild at the current head" if field in SLICE2_FIELDS else "a later port slice carries it"
            raise CheckStopped(f"the port readout does not carry {field!r} ({note}); required here")


def open_spec(request, spec: str) -> None:
    """Drive an open through the port's own model open door - the same
    door the shared shell evals for delivered opens; the receiving
    window opens through its own model. The spec is the port's own
    location spec string."""
    from t36_checks import _evaluate

    _evaluate(
        request,
        f"return window.{OPEN_DOOR}({json.dumps(spec)});",
    )


def part_selector(part: str) -> str:
    """The CSS selector addressing a rendered part by the port's own
    content structure (data-port-part mirrors the model's canonical
    content path)."""
    return f'[data-port-part="{part}"]'



# the inventory the port flavor serves today: F01 and S16 are
# flavor-independent (the feed proxy verbs and the shipped-bundle
# prototype differential exercise no old-app selector); S08 rides the
# part-path citations with fold reveal (chunk E2), S10 the gg/G
# reading context (the input foundation), S12 the select session's
# labels and exact-single selection, S14 the trail readout, S20 its
# port observation branch. Everything else stops with the recorded
# premise - driving old-app selectors into the fresh app is exactly
# what the adapter must not do.
PORT_SERVED = frozenset({"F01", "S08", "S10", "S12", "S14", "S15", "S16", "S20"})

# the recorded non-runnables: their verdicts are registry stops that
# name why the scenario does not execute through this harness at all -
# flavor-independent, so the flavor gate passes them through and their
# own bodies give the recorded reason
NOT_DRIVEN = frozenset({"S01", "S03", "S04", "S05", "S06", "S07", "S11", "S19"})

# the port surfaces still missing (chunks E through G landed picker,
# select labels, the act menu with fold, part-path citations, trail
# and viewport, find, and the follow act): copy (S13), Back/Forward
# key bindings and Tasks. S15 now runs and judges the landed follow
# implementation; its verdict is not a missing-surface premise. S09
# stays a recorded stop by the coordinator's ruling (the
# four-invocation agreement is a menu-surface dependency), not
# because follow is absent.
_STILL_MISSING = (
    "copy (S13),"
    " the back/forward key bindings and the tasks surface"
)


def ensure_served(check_id: str, obs: dict | None = None) -> None:
    """The flavor-aware gate run_check consults: on the port flavor a
    check outside the served set stops with the recorded premise; the
    recorded non-runnables pass through (their bodies carry the
    recorded reason). The trail/viewport fields REQUIRE_FIELDS-stop
    through require_fields at the bodies that need them."""
    if check_id in PORT_SERVED or check_id in NOT_DRIVEN:
        return
    raise CheckStopped(
        f"the port surface does not serve this scenario yet (the adapter serves"
        f" {', '.join(sorted(PORT_SERVED))}); still missing: {_STILL_MISSING}"
        " - recorded gap, not a failure"
    )


def part_rect(request, part: str) -> dict:
    """The rendered part's viewport rect from the port's own content
    structure (data-port-part mirrors the model's canonical path).
    None when the part is not rendered now (folded content renders no
    elements - the fold's own observable)."""
    from t36_checks import _json_evaluate

    value = _json_evaluate(
        request,
        "const el = document.querySelector("
        + json.dumps(part_selector(part)) + ");"
        " if (!el) return null; const r = el.getBoundingClientRect();"
        " return {x: r.x, y: r.y, width: r.width, height: r.height};",
    )
    return value if isinstance(value, dict) else None


def type_text(request, target: str, text: str, pace: float, timeout: float = 15.0) -> None:
    """Target-addressed real typing: a real click on the known field
    (focus), the field asserted fresh (the picker and find fields open
    empty - residue stops, never typed over), then per-character keys
    with EXPLICIT SHIFT CHORDS for shifted symbols - every response
    _ok'd, the delivered value asserted, the whole operation bounded
    from BEFORE the click. The target is a port control id
    (port:<name>).

    The delivery evidence (three authoritative runs plus isolated
    matrices): element send_keys loses its modifiers after a pointer
    action on this WebKitWebDriver stack (':' as ';'- the f447de1
    run2/3 record: the first send_keys before any rightclick delivered
    verbatim, the one after the fold's rightclick diverged), and bare
    shifted codepoints through the key verb lose their shift the same
    way ('@' as '2'). EXPLICIT CHORDS - shift down, base key, shift
    up - survive pointer actions (the f447de1 run2 record: the ':'
    chord delivered verbatim after the fold's rightclick). The value
    assertions on both sides of typing are the guard: any divergence
    stops naming the field."""
    import time

    from t36_checks import _evaluate, _ok, _sleep

    # the shifted symbols ride explicit chords (US layout base keys);
    # extend only with a live record of the unshifted delivery
    CHORDS = {":": ["shift", ";"], "@": ["shift", "2"]}

    deadline = time.monotonic() + timeout
    if not target.startswith("port:"):
        raise CheckStopped(f"type_text addresses port controls only; got {target!r}")
    field_selector = f'[data-port-control="{target[5:]}"]'

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise CheckStopped(f"the typing of {text!r} into {target!r} overran its bound")
        return left

    def focused_now() -> bool:
        return bool(_evaluate(
            request,
            "const el = document.querySelector(" + json.dumps(field_selector) + ");"
            " return !!el && document.activeElement === el;",
            timeout=min(10.0, remaining()),
        ))

    def field_value():
        return _evaluate(
            request,
            "const el = document.querySelector(" + json.dumps(field_selector) + ");"
            " return el ? el.value : null;",
            timeout=min(10.0, remaining()),
        )

    # the real click on the known field (focus), response checked,
    # bounded by the remaining budget
    _ok(request({"cmd": "input", "kind": "click", "target": target}, timeout=min(30.0, remaining())))
    _sleep(pace, 0.3)
    if not focused_now():
        raise CheckStopped(
            f"the field {target!r} is not focused after the click (stale or wrong focus)"
        )

    # the field is expected EMPTY (the picker and the find field
    # open fresh): asserted, never cleared. A clear_element before
    # send_keys POISONS the delivery on this stack - the shifted keys
    # land unshifted (':' as ';', recorded live twice: the 4cfbebf
    # readiness observations and the 5a44c71 run3 stops) - and the
    # per-character key verb is no fallback either ('@' landed as '2',
    # the f447de1 run2 stop). send_keys into a fresh field carried
    # full specs verbatim in the 6e30b80 authoritative run. Residue in
    # the field stops naming the field; it is never silently typed
    # over.
    if not focused_now():
        raise CheckStopped(f"the field {target!r} lost focus before typing")
    residue = field_value()
    if residue not in ("", None):
        raise CheckStopped(
            f"the field {target!r} is not fresh (carries {residue!r}); the real-input"
            " clear path needs its own grounded design - no silent typing over residue"
        )

    # the whole text, per character: shifted symbols as explicit
    # chords, plain characters as key presses; bounded, every
    # response checked, the value asserted after
    for ch in text:
        if time.monotonic() >= deadline:
            raise CheckStopped(f"typing {text!r} into {target!r} overran its bound")
        if ch in CHORDS:
            _ok(request({"cmd": "input", "kind": "hotkey", "keys": CHORDS[ch]}, timeout=min(30.0, remaining())))
        else:
            _ok(request({"cmd": "input", "kind": "key", "keys": [ch]}, timeout=min(30.0, remaining())))
        _sleep(pace, 0.05)
    if not focused_now():
        raise CheckStopped(f"the field {target!r} lost focus during typing")
    delivered = field_value()
    if delivered != text:
        raise CheckStopped(
            f"the typing into {target!r} diverged: the field carries {delivered!r}, typed {text!r}"
        )

def open_via_port_picker(request, spec: str, pace: float, timeout: float = 15.0) -> None:
    """Open a location through the port's own picker: space-space opens
    it, the spec is typed target-addressed into
    data-port-control="picker-input" (the click, the clear, the focus
    asserted, the send_keys delivery with the value asserted, every
    response _ok'd, the whole operation bounded), Enter submits through
    the one open funnel. Real input throughout; the load is the caller's
    to wait."""
    import time

    from t36_checks import _json_evaluate, _key, _sleep

    deadline = time.monotonic() + timeout
    _key(request, "escape")
    _sleep(pace, 0.3)
    _key(request, " ", " ")
    while True:
        opened = _json_evaluate(
            request,
            "return !!document.querySelector('[data-port-control=\"picker-input\"]');",
        )
        if opened is True:
            break
        if time.monotonic() >= deadline:
            raise CheckStopped("the port picker never opened on space space")
        _sleep(pace, 0.3)
    type_text(request, "port:picker-input", spec, pace, timeout=deadline - time.monotonic())
    # the submit readiness check: the input's value must equal the
    # spec before Enter (the value is a plain string - raw evaluate)
    from t36_checks import _evaluate
    while True:
        value = _evaluate(
            request,
            "const el = document.querySelector('[data-port-control=\"picker-input\"]');"
            " return el ? el.value : null;",
        )
        if value == spec:
            break
        if time.monotonic() >= deadline:
            raise CheckStopped(
                f"the port picker never carried the spec (value {value!r}; the typing"
                " was verified focused and complete - the divergence is post-typing)"
            )
        _sleep(pace, 0.2)
    _key(request, "enter")
    _sleep(pace, 0.5)




def fold_part(request, part: str, pace: float) -> None:
    """Fold or unfold a group part through the port's own act menu: a
    right-click at the part opens the menu, z carries the fold act
    (close-before-execute is the session module's own rule)."""
    from t36_checks import _json_evaluate, _key, _sleep

    rect = part_rect(request, part)
    if not rect:
        raise CheckStopped(f"the part {part!r} is not rendered (nothing to right-click)")
    # the group's rect encloses its children: the center lands on an
    # act-less child (which opens nothing); the header strip at the top
    # is the group's own hit area
    x = rect["x"] + rect["width"] / 2
    y = rect["y"] + min(6.0, rect["height"] / 4)
    request({"cmd": "input", "kind": "rightclick", "x": x, "y": y}, 30)
    _sleep(pace, 0.6)
    menu = _json_evaluate(request, 'return !!document.querySelector("[data-port-menu]");')
    if menu is not True:
        raise CheckStopped(f"the act menu never opened on the right-click of {part!r}")
    _key(request, "z")
    _sleep(pace, 0.6)
