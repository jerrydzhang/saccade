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

# the readout fields that land with the Board instrument (slice 2);
# requiring them today is the honest adapter: the port cannot produce
# them yet, so a check needing one stops naming the field
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
    naming the field and where it lands. The port's slice-2 fields
    (trail, viewport) ride the Board instrument."""
    for field in fields:
        if field not in obs or obs[field] is None:
            landing = "the Board instrument (slice 2)" if field in SLICE2_FIELDS else "a later port slice"
            raise CheckStopped(f"the port readout does not carry {field!r} yet ({landing}); required here")


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
# prototype differential exercise no old-app selector), S20's body
# carries its port observation branch. Everything else stops with the
# recorded premise - driving old-app selectors into the fresh app is
# exactly what the adapter must not do.
PORT_SERVED = frozenset({"F01", "S16", "S20"})


def ensure_served(check_id: str, obs: dict | None = None) -> None:
    """The flavor-aware gate run_check consults: on the port flavor a
    check outside the served set stops with the recorded premise. The
    trail/viewport fields REQUIRE_FIELDS-stop through require_fields at
    the bodies that need them."""
    if check_id in PORT_SERVED:
        return
    raise CheckStopped(
        f"the port slice-1 surface does not serve this scenario yet (the adapter serves"
        f" {', '.join(sorted(PORT_SERVED))}); the port's own picker/keymap/menu surfaces"
        " land with later slices - recorded gap, not a failure"
    )
