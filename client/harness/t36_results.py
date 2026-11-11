#!/usr/bin/env python3
"""T-36 phase-1 structured results: stable check ids, TAP13 emitter, run
manifest with provenance (R1).

The harness's verification contract, as data:

- Every check carries a stable id keyed to its scenario group: S01..S20
  (the ratified 20-group acceptance inventory on t-36@r50) plus the
  lane that produced the result. The inventory below pins ids and
  titles only; the per-check scripts arrive with the scenarios
  themselves, never here.
- Lanes emit TAP version 13 alongside the existing free-text lane
  output. Nothing free-text is removed or replaced by this module.
- Every run carries a manifest: the SAC_APP_BIN store path plus the
  build ref that binary was built from, the harness commit, the seed
  db hash, the adapter-set revision, the seed commit list, and the
  prototype artifact's provenance. Provenance is caller-supplied and
  validated here; a manifest that fails validation is a refusal, not
  a warning.

A result status is pass, fail or stop. "stop" is the R3 rule: the
existing page state cannot serve the check, so the check produces no
verdict and reports why. TAP carries a stop as "not ok ... # SKIP
stopped: <reason>" - it never counts as a failure, and it never
masquerades as a pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re

SCHEMA = "t36-run-manifest/1"
INVENTORY_REVISION = "t36-inventory/2"

# The harness lanes a check result can name. The scenario inventory's
# lane kinds map onto these: [P] pure and [C] component run without the
# app; "session" is the real-conditions lane (the design's N-Delta,
# N-app and N kinds); "prototype" is the round-16 prototype side of a
# differential pair, same scenario id as its session counterpart.
LANES = ("pure", "component", "session", "prototype", "browser")

SCENARIOS: tuple[dict, ...] = (
    {"id": "S01", "title": "Latest intent wins publication", "kinds": ("pure",)},
    {"id": "S02", "title": "Failure keeps the truth", "kinds": ("pure", "component")},
    {"id": "S03", "title": "Severity centrally decides notification", "kinds": ("pure", "app-only")},
    {"id": "S04", "title": "Deactivation separates input from content", "kinds": ("pure", "component", "native-delta")},
    {"id": "S05", "title": "Temporary buffers die with their scope; ordinary buffers are retained", "kinds": ("pure", "native-delta")},
    {"id": "S06", "title": "Presentation is disposable; the buffer's work is not", "kinds": ("pure", "component")},
    {"id": "S07", "title": "Return place restores the passage, not the presentation", "kinds": ("pure", "component", "native-delta")},
    {"id": "S08", "title": "Navigation reveals its actual anchor through folds", "kinds": ("pure", "component", "native-delta")},
    {"id": "S09", "title": "One action, every invocation agrees", "kinds": ("pure", "component", "native-delta")},
    {"id": "S10", "title": "Context decides what keys mean; sequences go through one matcher", "kinds": ("pure", "component", "native")},
    {"id": "S11", "title": "Handoff and safe return", "kinds": ("pure", "component")},
    {"id": "S12", "title": "Labels describe what is visible; selection follows identity", "kinds": ("pure", "component", "native-delta")},
    {"id": "S13", "title": "Nested targets and citation", "kinds": ("pure", "component", "native-delta")},
    {"id": "S14", "title": "The trail records explicit openings only", "kinds": ("pure", "native-delta")},
    {"id": "S15", "title": "Following current versus fixed revision", "kinds": ("pure", "app-only")},
    {"id": "S16", "title": "History and Find are facilities of the shown revision", "kinds": ("pure", "component", "native-delta")},
    {"id": "S17", "title": "Task activity is a place, not a popup", "kinds": ("pure", "native-delta", "app-only")},
    {"id": "S18", "title": "Linked delivery actually delivers", "kinds": ("native-delta", "pure")},
    {"id": "S19", "title": "The suite catches the held breaks before fixes land", "kinds": ("native",)},
    {"id": "S20", "title": "Windows are independent; relaunch restores persisted window state", "kinds": ("pure", "app-only")},
)

SCENARIO_IDS = tuple(s["id"] for s in SCENARIOS)

# The legacy lane gates (t36-inventory/2): the session tracker
# sequence's own checks, pinned here so their free-text results have
# stable ids in the TAP inventory. Matching is by substring because
# several check names carry dynamic task prefixes ("t-7's seed board
# committed at revision 1"). A failed gate is a FAIL entry with its
# evidence; checks the lane never reached STOP against it. G00 is the
# synthetic lane-completion gate: a lane that exits without any
# recorded check line still fails visibly in TAP.
GATES: tuple[dict, ...] = (
    {"id": "G00", "match": None, "name": "the lane run completed (synthetic)"},
    {"id": "G01", "match": "a real sac serve answers on the sandbox loopback", "name": "the seed server answers"},
    {"id": "G02", "match": "two boards committed on the real wire", "name": "the wire takes the seed boards"},
    {"id": "G03", "match": "seed board committed at revision 1", "name": "a seed board commits at revision 1"},
    {"id": "G04", "match": "the tracker UI takes real input", "name": "the tracker UI takes real input"},
    {"id": "G05", "match": "a tracker location loads through the real server under the real CSP", "name": "a tracker location loads under the real CSP"},
    {"id": "G06", "match": "opens the draft editor", "name": "a opens the draft editor"},
    {"id": "G07", "match": "the draft editor's cancel closes it", "name": "the draft editor's cancel closes it"},
    {"id": "G08", "match": "the citation opens the cited revision through the real wire", "name": "the citation opens the cited revision"},
    {"id": "G09", "match": "landing seed commits at revision 1 (text only)", "name": "the landing seed commits text-only"},
    {"id": "G10", "match": "the landing seed opens at revision 1", "name": "the landing seed opens at revision 1"},
    {"id": "G11", "match": "landing figure lands at revision 2", "name": "the landing figure commit lands"},
    {"id": "G12", "match": "a landed figure renders on re-open", "name": "a landed figure renders on re-open"},
    {"id": "G13", "match": "the tracker server stopped", "name": "the tracker server stopped"},
    {"id": "G14", "match": "a stopped server refuses in bounded time", "name": "a stopped server refuses in bounded time"},
    {"id": "G15", "match": "the landing-on-a-real-server pin runs", "name": "the landing pin runs"},
    {"id": "G16", "match": "the stopped-server open takes input", "name": "the stopped-server open takes input"},
)

GATE_IDS = tuple(g["id"] for g in GATES)


def gate_of(gate_id: str) -> dict:
    for gate in GATES:
        if gate["id"] == gate_id:
            return gate
    raise ResultError(f"unknown gate id: {gate_id!r} (inventory {INVENTORY_REVISION})")


def gate_for(text: str) -> dict | None:
    """The gate a lane check line belongs to, longest matching needle
    first (so the more specific landing-seed matcher wins over the
    generic seed one). None means the line is not a pinned gate and
    stays free-text only."""
    best: dict | None = None
    for gate in GATES:
        needle = gate["match"]
        if needle and needle in text and (best is None or len(needle) > len(best["match"] or "")):
            best = gate
    return best

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_STOP = "stop"
STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_STOP)


class ResultError(ValueError):
    """A result or manifest record refused: wrong id, lane or shape."""


def scenario_of(scenario_id: str) -> dict:
    for scenario in SCENARIOS:
        if scenario["id"] == scenario_id:
            return scenario
    raise ResultError(f"unknown scenario id: {scenario_id!r} (inventory {INVENTORY_REVISION})")


def make_result(scenario_id: str, lane: str, name: str, status: str, detail: str = "", adapters: tuple[str, ...] = ()) -> dict:
    """One check result. ids are validated against the inventory (a
    scenario id S01..S20 or a gate id G00..); lanes are validated
    against the lane set; adapter ids ride as recorded data and are
    guard-checked by t36_adapters, not here."""
    if scenario_id in GATE_IDS:
        gate_of(scenario_id)  # registry membership, refused if absent
    else:
        scenario_of(scenario_id)
    if lane not in LANES:
        raise ResultError(f"unknown lane: {lane!r} (lanes: {', '.join(LANES)})")
    if status not in STATUSES:
        raise ResultError(f"unknown result status: {status!r}")
    if not name or len(name) > 200:
        raise ResultError("a result needs a nonempty name of at most 200 characters")
    if len(detail) > 4000:
        raise ResultError("detail is capped at 4000 characters; put evidence in the run root, not the record")
    return {
        "id": scenario_id,
        "lane": lane,
        "name": name,
        "status": status,
        "detail": detail,
        "adapters": list(adapters),
    }


def result_key(result: dict) -> tuple[str, str, str]:
    return (result["id"], result["lane"], result["name"])


def _tap_description(result: dict) -> str:
    return f"{result['id']}/{result['lane']} {result['name']}"


def _tap_yaml_block(result: dict) -> str:
    lines = ["    ---"]
    lines.append(f"    status: {result['status']}")
    detail = result.get("detail") or ""
    if detail:
        lines.append("    detail: |")
        for chunk in detail.splitlines() or [""]:
            lines.append(f"      {chunk}")
    adapters = result.get("adapters") or []
    if adapters:
        lines.append(f"    adapters: {','.join(adapters)}")
    lines.append("    ...")
    return "\n".join(lines)


def emit_tap(results: list[dict], manifest: dict | None = None, manifest_path: str | None = None) -> str:
    """TAP version 13 for one lane run. Stops are skip directives, so a
    stopped check never fails a consumer's exit count; failures do.
    The manifest rides the plan as comments before the plan line."""
    for result in results:
        make_result(result["id"], result["lane"], result["name"], result["status"], result.get("detail", ""), tuple(result.get("adapters") or ()))
    ordered = sorted(results, key=result_key)
    out: list[str] = ["TAP version 13"]
    if manifest is not None:
        for line in json.dumps(manifest, indent=2, sort_keys=True).splitlines():
            out.append(f"# manifest: {line}")
    elif manifest_path:
        out.append(f"# manifest-file: {manifest_path}")
    out.append(f"1..{len(ordered)}")
    for index, result in enumerate(ordered, start=1):
        description = _tap_description(result)
        if result["status"] == STATUS_PASS:
            out.append(f"ok {index} - {description}")
        elif result["status"] == STATUS_STOP:
            reason = (result.get("detail") or "existing state insufficient").replace("\n", " ")[:300]
            out.append(f"not ok {index} - {description} # SKIP stopped: {reason}")
        else:
            out.append(f"not ok {index} - {description}")
        if result.get("detail") and result["status"] != STATUS_STOP:
            out.append(_tap_yaml_block(result))
        elif result.get("adapters"):
            out.append(f"    # adapters: {','.join(result['adapters'])}")
    return "\n".join(out) + "\n"


def tap_exit_code(tap_text: str) -> int:
    """The TAP gate: any non-skip 'not ok' line fails the run. Stops are
    skips; they never fail."""
    for line in tap_text.splitlines():
        if line.startswith("not ok ") and "# SKIP" not in line:
            return 1
    return 0


_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REF = re.compile(r"^[\x20-\x7e]{1,64}$")


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(
    lane: str,
    harness_commit: str,
    app_bin: str,
    app_ref: str,
    adapter_set_revision: str,
    serve_bin: str = "",
    serve_ref: str = "",
    seeds: dict | None = None,
    prototype: dict | None = None,
    run_id: str = "",
    started: str = "",
) -> dict:
    """A run manifest. Provenance fields are caller-supplied facts
    recorded verbatim; validation is validate_manifest's job. The
    serve binary's build ref is provenance too (t-36@r58: a store
    path alone shape-validated a stale binary - the ref is what
    catches serve skew)."""
    if lane not in LANES:
        raise ResultError(f"unknown lane: {lane!r}")
    manifest: dict = {
        "schema": SCHEMA,
        "lane": lane,
        "harness_commit": harness_commit,
        "adapter_set_revision": adapter_set_revision,
        "checks_inventory": INVENTORY_REVISION,
        "app": {"bin": app_bin, "ref": app_ref},
        "serve": {"bin": serve_bin, "ref": serve_ref},
        "run_id": run_id,
        "started": started,
    }
    if seeds is not None:
        manifest["seeds"] = seeds
    if prototype is not None:
        manifest["prototype"] = prototype
    return manifest


def _refusal(manifest: dict, field: str, why: str, refusals: list[str]) -> None:
    refusals.append(f"{field}: {why}")


def validate_manifest(manifest: dict, require_seeds: bool = False, require_prototype: bool = False) -> list[str]:
    """Structural validation plus the provenance gate. Returns the
    refusal list; empty means the manifest may ride a run. The driver
    calls this with require_seeds=True before it lets a run start:
    no provenance, no run."""
    refusals: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest: not an object"]
    if manifest.get("schema") != SCHEMA:
        _refusal(manifest, "schema", f"expected {SCHEMA!r}, got {manifest.get('schema')!r}", refusals)
    lane = manifest.get("lane")
    if lane not in LANES:
        _refusal(manifest, "lane", f"expected one of {', '.join(LANES)}, got {lane!r}", refusals)
    commit = manifest.get("harness_commit")
    if not isinstance(commit, str) or not _GIT_SHA.match(commit):
        _refusal(manifest, "harness_commit", "expected a 7-40 char lowercase git sha", refusals)
    adapter_revision = manifest.get("adapter_set_revision")
    if not isinstance(adapter_revision, str) or not _REF.match(adapter_revision):
        _refusal(manifest, "adapter_set_revision", "expected a nonempty printable revision string", refusals)
    if manifest.get("checks_inventory") != INVENTORY_REVISION:
        _refusal(manifest, "checks_inventory", f"expected {INVENTORY_REVISION!r}", refusals)

    app = manifest.get("app")
    if not isinstance(app, dict):
        _refusal(manifest, "app", "expected an object with bin and ref", refusals)
    else:
        app_bin = app.get("bin")
        if not isinstance(app_bin, str) or not app_bin.startswith("/nix/store/") or not os.path.isfile(app_bin) or not os.access(app_bin, os.X_OK):
            _refusal(manifest, "app.bin", "expected an executable /nix/store path", refusals)
        app_ref = app.get("ref")
        if not isinstance(app_ref, str) or not _REF.match(app_ref or ""):
            _refusal(manifest, "app.ref", "expected a nonempty printable build ref (the commit the binary was built from)", refusals)

    serve = manifest.get("serve")
    if not isinstance(serve, dict) or not isinstance(serve.get("bin"), str):
        _refusal(manifest, "serve.bin", "expected a string path to the real sac binary", refusals)
    elif serve.get("bin"):
        serve_bin = serve["bin"]
        if not serve_bin.startswith("/nix/store/") or not os.path.isfile(serve_bin) or not os.access(serve_bin, os.X_OK):
            _refusal(manifest, "serve.bin", "expected an executable /nix/store path when present", refusals)
        serve_ref = serve.get("ref")
        if not isinstance(serve_ref, str) or not _REF.match(serve_ref or ""):
            _refusal(manifest, "serve.ref", "a serve binary must carry its declared build ref (t-36@r58: the ref is what catches serve skew)", refusals)

    seeds = manifest.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, dict):
            _refusal(manifest, "seeds", "expected an object", refusals)
        else:
            db_hash = seeds.get("db_hash")
            if not isinstance(db_hash, str) or not _SHA256.match(db_hash or ""):
                _refusal(manifest, "seeds.db_hash", "expected a sha256 hex digest", refusals)
            db_path = seeds.get("db_path")
            if not isinstance(db_path, str) or not db_path:
                _refusal(manifest, "seeds.db_path", "expected a path", refusals)
            commits = seeds.get("commits")
            if not isinstance(commits, list) or not all(isinstance(c, dict) and c.get("task") and c.get("revision") for c in commits or []):
                _refusal(manifest, "seeds.commits", "expected a list of {task, revision, ...} commit records", refusals)
    elif require_seeds:
        _refusal(manifest, "seeds", "required for a run (per-run seeded tracker db)", refusals)

    prototype = manifest.get("prototype")
    if prototype is not None:
        if not isinstance(prototype, dict) or not all(isinstance(prototype.get(k), str) and prototype.get(k) for k in ("source", "sha256")):
            _refusal(manifest, "prototype", "expected an object with source and sha256", refusals)
        elif prototype is not None and not _SHA256.match(prototype.get("sha256") or ""):
            _refusal(manifest, "prototype.sha256", "expected a sha256 hex digest", refusals)
    elif require_prototype:
        _refusal(manifest, "prototype", "required for a differential run", refusals)

    return refusals


def write_manifest(path: str, manifest: dict) -> None:
    refusals = validate_manifest(manifest)
    if refusals:
        raise ResultError("manifest refused: " + "; ".join(refusals))
    pathlib.Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")


def load_manifest(path: str) -> dict:
    value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResultError(f"manifest at {path} is not an object")
    return value


def harness_commit_of(repo_dir: str) -> str:
    """The harness commit: the checkout's HEAD sha, read from git."""
    import subprocess

    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, timeout=20)
    sha = out.stdout.decode("ascii", errors="replace").strip()
    if out.returncode != 0 or not _GIT_SHA.match(sha):
        raise ResultError(f"harness commit unavailable: {out.stderr.decode(errors='replace')[:200]}")
    return sha
