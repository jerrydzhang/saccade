#!/usr/bin/env python3
"""T-36 two-run driver (R2): one script, everything held constant
except the app binary.

The differential contract: a baseline run (the binary built unmodified
from the baseline commit) against a candidate run; the per-check diff
of structured results is the red/green record. The baseline binary is
built UNMODIFIED - no stamp code is injected into the app under test -
so provenance is manifest-side: the caller names the store path and
the ref each binary was built from, and the driver refuses to run
without both.

Everything else is constant by construction, verified before any run
starts: the board, the tracker db baseline, the real sac binary, the
seed script (per-run seeded dbs, same commit list), the harness code,
the adapter set, the check inventory and every lane environment
variable except SAC_APP_BIN. The plans are compared modulo the app
fields; a mismatch is a refusal, never a silent difference.

Phase 1 ships the driver's plan/provenance/diff machinery and the
dry-run surface. The default mode is --dry-run: plans and manifests
are printed and nothing runs. --execute composes real lane runs by
invoking the harness entry per plan (session lane first); its run
loop is exercised by the red run, not by unit tests.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import t36_adapters  # noqa: E402
import t36_results  # noqa: E402

ROLES = ("baseline", "candidate")


class ProvenanceRefused(Exception):
    """The driver refuses to run: provenance is missing or invalid."""


def _validate_app_binary(role: str, app_bin: str, refusals: list[str]) -> None:
    if not app_bin:
        refusals.append(f"{role} app binary missing: pass --{role}-bin")
        return
    if not os.path.isabs(app_bin):
        refusals.append(f"{role} app binary must be an absolute path as given: {app_bin!r}")
        return
    if not app_bin.startswith("/nix/store/"):
        refusals.append(f"{role} app binary must be a /nix/store path (the packaged app is store-pinned): {app_bin!r}")
        return
    if not os.path.isfile(app_bin) or not os.access(app_bin, os.X_OK):
        refusals.append(f"{role} app binary is not an executable file: {app_bin!r}")


def _validate_ref(role: str, app_ref: str, refusals: list[str]) -> None:
    if not app_ref:
        refusals.append(f"{role} build ref missing: pass --{role}-ref (the commit that binary was built from)")
        return
    if len(app_ref) > 64 or not all(0x20 <= ord(ch) <= 0x7E for ch in app_ref):
        refusals.append(f"{role} build ref must be a short printable string: {app_ref!r}")


def _validate_shared(board: str, tracker_db: str, serve_bin: str, serve_ref: str, refusals: list[str]) -> None:
    if not board or not os.path.isabs(board) or not os.path.isfile(board):
        refusals.append(f"board must be an absolute path to an existing board file: {board!r}")
    if not tracker_db or not os.path.isabs(tracker_db) or not os.path.isfile(tracker_db):
        refusals.append(f"tracker db baseline must be an absolute path to an existing db: {tracker_db!r}")
    if not serve_bin or not os.path.isabs(serve_bin) or not os.access(serve_bin, os.X_OK):
        refusals.append(f"the real sac binary must be an executable absolute /nix/store path: {serve_bin!r}")
    elif not serve_bin.startswith("/nix/store/"):
        refusals.append(f"the real sac binary must be a /nix/store path, not just any absolute path: {serve_bin!r}")
    if not serve_ref:
        refusals.append(
            "serve build ref missing: pass --serve-ref (the commit the serve binary was built from;"
            " t-36@r58 - a store path alone shape-validated a stale binary)"
        )
    elif len(serve_ref) > 64 or not all(0x20 <= ord(ch) <= 0x7E for ch in serve_ref):
        refusals.append(f"serve build ref must be a short printable string: {serve_ref!r}")


def build_plan(
    role: str,
    app_bin: str,
    app_ref: str,
    board: str,
    tracker_db: str,
    serve_bin: str,
    serve_ref: str,
    harness_commit: str,
    run_id: str,
    started: str,
    seeds: dict | None = None,
) -> dict:
    """One run's plan. The manifest carries the plan's provenance;
    validation is strict: a run never starts on an unvalidated plan."""
    refusals: list[str] = []
    _validate_app_binary(role, app_bin, refusals)
    _validate_ref(role, app_ref, refusals)
    _validate_shared(board, tracker_db, serve_bin, serve_ref, refusals)
    if refusals:
        raise ProvenanceRefused("; ".join(refusals))
    manifest = t36_results.make_manifest(
        lane="session",
        harness_commit=harness_commit,
        app_bin=app_bin,
        app_ref=app_ref,
        adapter_set_revision=t36_adapters.ADAPTER_SET_REVISION,
        serve_bin=serve_bin,
        serve_ref=serve_ref,
        seeds=seeds,
        run_id=run_id,
        started=started,
    )
    refusals = t36_results.validate_manifest(manifest, require_seeds=seeds is not None)
    if refusals:
        raise ProvenanceRefused("manifest refused: " + "; ".join(refusals))
    return {
        "role": role,
        "lane": "t36",
        "app_bin": app_bin,
        "app_ref": app_ref,
        "board": board,
        "tracker_db": tracker_db,
        "serve_bin": serve_bin,
        "manifest": manifest,
    }


def build_plans(options: argparse.Namespace, harness_commit: str, run_id: str, started: str) -> list[dict]:
    """Both run plans from one option set. Everything except the app
    binary and its ref is shared by construction: the same values feed
    both plans, and constant_of() equality below pins that."""
    baseline = build_plan(
        "baseline", options.baseline_bin, options.baseline_ref,
        options.board, options.tracker_db, options.serve_bin, options.serve_ref,
        harness_commit, run_id + "-baseline", started,
    )
    candidate = build_plan(
        "candidate", options.candidate_bin, options.candidate_ref,
        options.board, options.tracker_db, options.serve_bin, options.serve_ref,
        harness_commit, run_id + "-candidate", started,
    )
    if constant_of(baseline) != constant_of(candidate):
        raise ProvenanceRefused("the two plans differ outside the app binary; the differential contract requires one constant setup")
    return [baseline, candidate]


def constant_of(plan: dict) -> dict:
    """The plan modulo the app fields: what must be identical across
    the two runs."""
    return {key: value for key, value in plan.items() if key not in ("role", "app_bin", "app_ref", "manifest")}


def check_subset(spec: str) -> list[str]:
    """A --checks subset: S-ids from the inventory, comma-separated."""
    import t36_results as results

    ids = [chunk.strip() for chunk in spec.split(",") if chunk.strip()]
    for scenario_id in ids:
        results.scenario_of(scenario_id)
    return ids


LANE_CHECK_LINE = re.compile(r"^\[webdriver\] (ok|FAIL): (.+)$")

LANES = ("tracker", "t36")


def t36_inventory_ids() -> list[str]:
    """The t36 lane's check inventory: every scenario group plus the
    feed held-break probe, in stable order."""
    return list(t36_results.SCENARIO_IDS) + ["F01"]


def lane_results_from_log(log_text: str, lane: str = "session", declared_checks: tuple[str, ...] | list[str] = ()) -> list[dict]:
    """The lane's free-text check lines as TAP inventory entries
    (t-36@r58: TAP must always exist - a lane that dies at a legacy
    gate still records that gate's failure with its evidence, and the
    declared checks the lane never reached STOP against it, instead of
    the run ending before any TAP was emitted).

    Lines matching a pinned gate (t36_results.GATES, by substring -
    several names carry dynamic task prefixes) become results with the
    gate's stable id; unmatched lines stay free-text only. Declared
    scenario checks with no result line stop: 'not reached' against
    the first failed gate when there is one, 'never produced' when the
    lane simply ended."""
    results: list[dict] = []
    seen_gates: set[str] = set()
    first_failure: dict | None = None
    for line in log_text.splitlines():
        match = LANE_CHECK_LINE.match(line.strip())
        if match is None:
            continue
        said, rest = match.group(1), match.group(2).strip()
        gate = t36_results.gate_for(rest)
        if gate is None:
            continue
        status = t36_results.STATUS_PASS if said == "ok" else t36_results.STATUS_FAIL
        detail = rest if status == t36_results.STATUS_FAIL else ""
        result = t36_results.make_result(gate["id"], lane, rest[:200], status, detail)
        results.append(result)
        seen_gates.add(gate["id"])
        if status == t36_results.STATUS_FAIL and first_failure is None:
            first_failure = result
    for scenario_id in declared_checks:
        t36_results.scenario_of(scenario_id)  # refuse unknown ids here
        already = any(r["id"] == scenario_id for r in results)
        if already:
            continue
        if first_failure is not None:
            reason = (
                "not reached: the lane stopped at failed check "
                f"{first_failure['name']!r} ({first_failure['id']})"
            )
        else:
            reason = "the lane produced no result for this check before it ended"
        results.append(t36_results.make_result(scenario_id, lane, "scenario check", t36_results.STATUS_STOP, reason))
    return results


def lane_run_result(exit_code: int, results: list[dict], lane: str = "session") -> dict | None:
    """The synthetic lane-completion entry (G00): a lane that exited
    nonzero without any failing gate line still fails visibly in the
    TAP, with its exit code as evidence. None when the run is clean or
    a gate failure already explains the exit."""
    if exit_code == 0:
        return None
    gate_failure = any(r["status"] == t36_results.STATUS_FAIL for r in results)
    if gate_failure:
        return None
    return t36_results.make_result(
        "G00", lane, "lane run", t36_results.STATUS_FAIL,
        f"the lane exited {exit_code} before any failing check was recorded",
    )


def write_run_tap(run_root: pathlib.Path, role: str, results: list[dict], manifest: dict) -> pathlib.Path:
    """ALWAYS write the run's TAP, on every path: gate failures, lane
    crashes and clean runs all land as files beside the lane log."""
    tap = t36_results.emit_tap(results, manifest=manifest)
    path = run_root / f"tap-{role}.tap"
    path.write_text(tap, encoding="ascii")
    return path


def diff_results(baseline: list[dict], candidate: list[dict]) -> list[dict]:
    """The per-check diff: rows keyed by (scenario id, check name).
    Lane is part of the key; a differential pair runs the same lane on
    both sides. A check present on one side only is recorded as such."""
    def keyed(results: list[dict]) -> dict:
        return {(r["id"], r["name"]): r for r in results}

    base, cand = keyed(baseline), keyed(candidate)
    rows: list[dict] = []
    for key in sorted(set(base) | set(cand)):
        b, c = base.get(key), cand.get(key)
        same = b is not None and c is not None and b["status"] == c["status"]
        rows.append({
            "id": key[0],
            "name": key[1],
            "baseline": None if b is None else b["status"],
            "candidate": None if c is None else c["status"],
            "verdict": "same" if same else "differs",
            "detail": {
                "baseline": None if b is None else b.get("detail", ""),
                "candidate": None if c is None else c.get("detail", ""),
            },
        })
    return rows


def run_lane(plan: dict, run_root: pathlib.Path, options: argparse.Namespace, env_extra: dict[str, str]) -> int:
    """One lane run inside its own run root: the harness entry invoked
    with the plan's constant environment and this plan's SAC_APP_BIN.
    The environment is the caller's, scrubbed of the two variables the
    plans own, then re-pinned identically for both runs."""
    env = {k: v for k, v in os.environ.items() if k not in ("SAC_APP_BIN",)}
    env.update(env_extra)
    env["SAC_APP_BIN"] = plan["app_bin"]
    entry = HERE / "entry.py"
    argv = [
        sys.executable, "-B", str(entry), "session", "tracker",
        "--board", plan["board"],
    ]
    if options.tracker_db:
        argv += ["--tracker-db", options.tracker_db]
    log_path = run_root / "logs" / f"{plan['role']}-lane.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        completed = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
    return completed.returncode


class T36Deps:
    """The session surface the t36 lane executor drives. real() binds
    the harness entry (down, up-t36 - with up_reuse relaunching into an
    existing run root so the app's persisted data dir survives, S20) -
    and the Playwright prototype session; tests inject fakes shaped by
    the grounded answers. The fresh-session premise lives in the
    executor: down + up per role, never a shared session."""

    def __init__(self, up, up_reuse, live_socket, request, down, prototype_factory) -> None:
        self.up = up
        self.up_reuse = up_reuse
        self.live_socket = live_socket
        self.request = request
        self.down = down
        self.prototype_factory = prototype_factory


def real_deps(options: argparse.Namespace) -> T36Deps:
    import entry as entry_module
    import t36_prototype

    def up(plan: dict, reuse_root: str = "") -> int:
        # a live session from a previous run is never adopted: down
        # first, so every role starts from a fresh app
        if entry_module.load_session_record() is not None:
            entry_module.session_down()
        os.environ["SAC_APP_BIN"] = plan["app_bin"]
        os.environ["SACCADE_SERVE_BIN"] = options.serve_bin
        os.environ["SACCADE_DB"] = options.tracker_db
        return entry_module.session_up(
            options.board, entry_module.SESSION_DEFAULT_LIFETIME_SECONDS,
            mode="t36", reuse_root=reuse_root,
        )

    def live_socket():
        return entry_module.session_live_socket()

    def request(socket_path: str, payload: dict, timeout: float = 60.0) -> dict:
        return entry_module.session_request(socket_path, payload, timeout=timeout)

    def down() -> int:
        return entry_module.session_down()

    def prototype_factory():
        try:
            for key, value in t36_prototype.webkit_launch_env().items():
                os.environ.setdefault(key, value)
            cache = pathlib.Path("/tmp/t36-proto-cache")
            materialized = t36_prototype.materialize(cache)
            vendored = t36_prototype.vendor_libs(cache)
            server = t36_prototype.PrototypeServer(
                pathlib.Path(materialized["path"]).read_bytes(), vendored
            )
        except t36_prototype.PrototypeRefused as error:
            print(f"[t36-driver] prototype half unavailable: {error}", file=sys.stderr, flush=True)
            return None, None
        session = t36_prototype.PrototypeSession(server)

        def stop() -> None:
            session.stop()
            server.stop()

        session.start()
        return session, stop

    return T36Deps(up, up, live_socket, request, down, prototype_factory)


def _lane_up_failed(detail: str) -> dict:
    return t36_results.make_result(
        "G00", "session", "lane run", t36_results.STATUS_FAIL, detail
    )


def execute_t36_lane(options: argparse.Namespace, plans: list[dict], out_root: pathlib.Path, deps: T36Deps | None = None) -> int:
    """The t36 lane: per role, a FRESH up-t36 session (the S18 premise
    enforces one window; the driver never reuses a session across
    roles), the prelude gate log parsed into gate entries, the check
    inventory run over the live control socket with the prototype
    half attached, the two merged into one TAP per role, then the
    per-check diff exactly as the tracker lane lands it."""
    import isolation

    import t36_checks
    import t36_differential

    deps = deps or real_deps(options)
    exit_codes: dict[str, int] = {}
    role_results: dict[str, list[dict]] = {}
    for plan in plans:
        run_root = isolation.make_run_root(f"t36-{plan['role']}")
        manifest_path = run_root / "manifest.json"
        t36_results.write_manifest(str(manifest_path), plan["manifest"])
        results: list[dict] = []
        up_code = deps.up(plan)
        if up_code != 0:
            results.append(_lane_up_failed(f"up-t36 exited {up_code}; no session came up"))
        else:
            live = deps.live_socket()
            if live is None:
                results.append(_lane_up_failed("the session record vanished after up"))
            else:
                _record, socket_path = live
                session_request = lambda payload, timeout=60.0: deps.request(socket_path, payload, timeout)  # noqa: E731

                def relaunch() -> None:
                    # S20's orchestration: down, then up-t36 into the
                    # SAME run root - the app's persisted data dir
                    # survives, the tracker and its store do not; the
                    # socket path is the run root's own, so the live
                    # request closure keeps working
                    from t36_observation import CheckStopped

                    deps.down()
                    code = deps.up_reuse(plan, str(run_root))
                    if code != 0:
                        raise CheckStopped(f"the relaunch up-t36 exited {code}")

                prototype, prototype_stop = deps.prototype_factory()
                try:
                    t36_differential.attach(prototype)
                    t36_checks.set_relauncher(relaunch)
                    results = t36_checks.run_checks(request=session_request, check_ids=t36_inventory_ids(), pace=1.0)
                finally:
                    t36_checks.set_relauncher(None)
                    t36_differential.attach(None)
                    if prototype_stop is not None:
                        prototype_stop()
                prelude = run_root / "logs" / "t36-prelude.log"
                if prelude.exists():
                    gate_results = lane_results_from_log(prelude.read_text(errors="replace"))
                    results = gate_results + results
        tap_path = write_run_tap(run_root, plan["role"], results, plan["manifest"])
        exit_codes[plan["role"]] = t36_results.tap_exit_code(t36_results.emit_tap(results))
        role_results[plan["role"]] = results
        deps.down()
        print(
            f"[t36-driver] {plan['role']} run root: {run_root} tap={tap_path}"
            f" tap_exit={exit_codes[plan['role']]}",
            flush=True,
        )
    if "baseline" in role_results and "candidate" in role_results:
        diff_path = write_diff(out_root, diff_results(role_results["baseline"], role_results["candidate"]))
        print(f"[t36-driver] per-check diff: {diff_path}", flush=True)
    return max(exit_codes.values(), default=1)


def write_diff(out_root: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path = out_root / "diff.json"
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T-36 two-run differential driver")
    parser.add_argument("--baseline-bin", help="the baseline app binary (store path, built unmodified)")
    parser.add_argument("--baseline-ref", help="the commit the baseline binary was built from")
    parser.add_argument("--candidate-bin", help="the candidate app binary (store path)")
    parser.add_argument("--candidate-ref", help="the commit the candidate binary was built from")
    parser.add_argument("--board", help="absolute board file both runs open")
    parser.add_argument("--tracker-db", help="absolute caller-side tracker db baseline")
    parser.add_argument("--serve-bin", help="absolute path to the real sac binary used for seeding and the lane")
    parser.add_argument("--serve-ref", help="the commit the serve binary was built from (t-36@r58: catches serve skew)")
    parser.add_argument("--checks", default="", help="optional comma-separated scenario subset (S08,S18)")
    parser.add_argument("--lane", default="tracker", choices=list(LANES), help="tracker (default) or t36 (up-t36 + the check inventory with the prototype half)")
    parser.add_argument("--execute", action="store_true", help="run both lanes (default is --dry-run)")
    options = parser.parse_args(argv)

    import time as _time
    import uuid

    started = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
    run_id = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    try:
        harness_commit = t36_results.harness_commit_of(str(HERE.parent.parent))
        plans = build_plans(options, harness_commit, run_id, started)
    except ProvenanceRefused as error:
        print(f"provenance refusal: {error}", file=sys.stderr)
        return 2
    if options.checks:
        try:
            check_subset(options.checks)
        except t36_results.ResultError as error:
            print(f"provenance refusal: {error}", file=sys.stderr)
            return 2

    if options.lane not in LANES:
        print(f"provenance refusal: unknown lane {options.lane!r}", file=sys.stderr)
        return 2

    if not options.execute:
        print(json.dumps({
            "mode": "dry-run",
            "lane": options.lane,
            "run_id": run_id,
            "harness_commit": harness_commit,
            "adapter_set_revision": t36_adapters.ADAPTER_SET_REVISION,
            "constant": constant_of(plans[0]),
            "plans": [
                {
                    "role": plan["role"],
                    "lane": plan["lane"],
                    "app_bin": plan["app_bin"],
                    "app_ref": plan["app_ref"],
                    "manifest": plan["manifest"],
                }
                for plan in plans
            ],
        }, indent=2, sort_keys=True))
        return 0

    import isolation

    if options.lane == "t36":
        return execute_t36_lane(options, plans, isolation.make_run_root("t36-differential"))

    out_root = isolation.make_run_root("t36-differential")
    exit_codes: dict[str, int] = {}
    role_results: dict[str, list[dict]] = {}
    declared = check_subset(options.checks) if options.checks else []
    for plan in plans:
        run_root = isolation.make_run_root(f"t36-{plan['role']}")
        manifest_path = run_root / "manifest.json"
        t36_results.write_manifest(str(manifest_path), plan["manifest"])
        lane_exit = run_lane(plan, run_root, options, {})
        # TAP always exists, on every path: the lane's own check lines
        # become gate entries, declared checks the lane never reached
        # stop against the first failed gate, and a lane that exited
        # without any recorded line still fails visibly (G00).
        log_path = run_root / "logs" / f"{plan['role']}-lane.log"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        results = lane_results_from_log(log_text, "session", declared)
        lane_entry = lane_run_result(lane_exit, results)
        if lane_entry is not None:
            results.append(lane_entry)
        role_results[plan["role"]] = results
        tap_path = write_run_tap(run_root, plan["role"], results, plan["manifest"])
        exit_codes[plan["role"]] = t36_results.tap_exit_code(t36_results.emit_tap(results))
        print(
            f"[t36-driver] {plan['role']} run root: {run_root} lane_exit={lane_exit}"
            f" tap={tap_path} tap_exit={exit_codes[plan['role']]}",
            flush=True,
        )
    if "baseline" in role_results and "candidate" in role_results:
        diff_path = write_diff(out_root, diff_results(role_results["baseline"], role_results["candidate"]))
        print(f"[t36-driver] per-check diff: {diff_path}", flush=True)
    return max(exit_codes.values(), default=1)


if __name__ == "__main__":
    sys.exit(main())
