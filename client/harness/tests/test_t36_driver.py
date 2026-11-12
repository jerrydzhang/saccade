#!/usr/bin/env python3
"""R2 unit tests: the two-run driver's provenance gate, plan
constancy, and the per-check diff. Pure - no lane, no app, no nix."""

from __future__ import annotations

import argparse
import json
import unittest.mock
from unittest import mock
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import t36_checks  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_driver  # noqa: E402
import t36_results  # noqa: E402


def _options(**overrides) -> argparse.Namespace:
    store_bin = shutil.which("bash")
    assert store_bin is not None and store_bin.startswith("/nix/store/")
    base = {
        "baseline_bin": store_bin,
        "baseline_ref": "4d59c35",
        "candidate_bin": store_bin,
        "candidate_ref": "9504b38",
        "board": "/tmp/t36-board.yaml",
        "tracker_db": "/tmp/t36.db",
        "serve_bin": store_bin,
        "serve_ref": "4d59c35",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _existing_options(tmp: str, **overrides) -> argparse.Namespace:
    """Options whose shared paths exist, so only the app fields can
    fail provenance."""
    board = pathlib.Path(tmp) / "board.yaml"
    board.write_text("grammar: 1\n", encoding="ascii")
    db = pathlib.Path(tmp) / "baseline.db"
    db.write_bytes(b"k")
    return _options(board=str(board), tracker_db=str(db), **overrides)


class ProvenanceGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def plans(self, **overrides) -> list[dict]:
        options = _existing_options(self.tmp.name, **overrides)
        return t36_driver.build_plans(options, "0" * 40, "run", "2026-01-01T00:00:00Z")

    def test_valid_provenance_yields_two_plans(self) -> None:
        plans = self.plans()
        self.assertEqual([p["role"] for p in plans], ["baseline", "candidate"])
        for plan in plans:
            self.assertEqual(t36_results.validate_manifest(plan["manifest"]), [])

    def test_missing_ref_refused(self) -> None:
        with self.assertRaises(t36_driver.ProvenanceRefused) as caught:
            self.plans(baseline_ref="")
        self.assertIn("baseline build ref missing", str(caught.exception))

    def test_missing_candidate_bin_refused(self) -> None:
        with self.assertRaises(t36_driver.ProvenanceRefused):
            self.plans(candidate_bin="")

    def test_relative_bin_refused(self) -> None:
        with self.assertRaises(t36_driver.ProvenanceRefused) as caught:
            self.plans(candidate_bin="relative/app")
        self.assertIn("absolute path as given", str(caught.exception))

    def test_non_store_bin_refused(self) -> None:
        with self.assertRaises(t36_driver.ProvenanceRefused) as caught:
            self.plans(candidate_bin="/home/jerry/app")
        self.assertIn("/nix/store", str(caught.exception))

    def test_missing_board_refused(self) -> None:
        options = _options(board="/tmp/t36-does-not-exist.yaml")
        with self.assertRaises(t36_driver.ProvenanceRefused) as caught:
            t36_driver.build_plans(options, "0" * 40, "run", "x")
        self.assertIn("board", str(caught.exception))

    def test_missing_serve_bin_refused(self) -> None:
        with self.assertRaises(t36_driver.ProvenanceRefused) as caught:
            self.plans(serve_bin="")
        self.assertIn("real sac binary", str(caught.exception))

    def test_missing_serve_ref_refused(self) -> None:
        with self.assertRaises(t36_driver.ProvenanceRefused) as caught:
            self.plans(serve_ref="")
        self.assertIn("serve build ref missing", str(caught.exception))
        self.assertIn("--serve-ref", str(caught.exception))

    def test_non_store_serve_bin_refused(self) -> None:
        with self.assertRaises(t36_driver.ProvenanceRefused) as caught:
            self.plans(serve_bin="/usr/bin/sac")
        self.assertIn("/nix/store", str(caught.exception))

    def test_plan_carries_provenance_in_manifest(self) -> None:
        plans = self.plans()
        baseline = plans[0]["manifest"]
        self.assertEqual(baseline["app"]["ref"], "4d59c35")
        self.assertEqual(baseline["app"]["bin"], plans[0]["app_bin"])
        self.assertEqual(baseline["serve"]["ref"], "4d59c35")
        self.assertEqual(baseline["serve"]["bin"], plans[0]["serve_bin"])
        self.assertEqual(baseline["adapter_set_revision"], t36_driver.t36_adapters.ADAPTER_SET_REVISION)


class ConstancyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_plans_differ_only_in_app_fields(self) -> None:
        plans = self.plans_of_tmp()
        self.assertEqual(t36_driver.constant_of(plans[0]), t36_driver.constant_of(plans[1]))
        self.assertNotEqual(plans[0]["app_bin"] and plans[0]["manifest"]["app"], plans[1]["manifest"]["app"])

    def plans_of_tmp(self) -> list[dict]:
        return t36_driver.build_plans(
            _existing_options(self.tmp.name), "0" * 40, "run", "2026-01-01T00:00:00Z"
        )

    def test_manifests_share_everything_but_app_and_run_id(self) -> None:
        plans = self.plans_of_tmp()
        def stripped(plan: dict) -> dict:
            manifest = dict(plan["manifest"])
            manifest.pop("app")
            manifest.pop("run_id")
            return manifest
        self.assertEqual(stripped(plans[0]), stripped(plans[1]))

    def test_board_and_db_identical_across_plans(self) -> None:
        plans = self.plans_of_tmp()
        for plan in plans:
            self.assertEqual(plan["board"], plans[0]["board"])
            self.assertEqual(plan["tracker_db"], plans[0]["tracker_db"])


class DiffTest(unittest.TestCase):
    def test_same_status_is_same(self) -> None:
        base = [t36_results.make_result("S08", "session", "reveal", t36_results.STATUS_FAIL, "red")]
        cand = [t36_results.make_result("S08", "session", "reveal", t36_results.STATUS_FAIL, "still red")]
        rows = t36_driver.diff_results(base, cand)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], "same")

    def test_status_change_differs(self) -> None:
        base = [t36_results.make_result("S18", "session", "delivery", t36_results.STATUS_FAIL, "nowhere")]
        cand = [t36_results.make_result("S18", "session", "delivery", t36_results.STATUS_PASS, "delivered")]
        rows = t36_driver.diff_results(base, cand)
        self.assertEqual(rows[0]["verdict"], "differs")
        self.assertEqual(rows[0]["baseline"], "fail")
        self.assertEqual(rows[0]["candidate"], "pass")

    def test_one_sided_check_recorded(self) -> None:
        base = [t36_results.make_result("S10", "session", "matcher", t36_results.STATUS_PASS)]
        rows = t36_driver.diff_results(base, [])
        self.assertEqual(rows[0]["baseline"], "pass")
        self.assertEqual(rows[0]["candidate"], None)
        self.assertEqual(rows[0]["verdict"], "differs")

    def test_stop_is_a_status_not_a_skip(self) -> None:
        base = [t36_results.make_result("S14", "session", "flags", t36_results.STATUS_STOP, "no readout")]
        cand = [t36_results.make_result("S14", "session", "flags", t36_results.STATUS_PASS)]
        rows = t36_driver.diff_results(base, cand)
        self.assertEqual(rows[0]["verdict"], "differs")


class CheckSubsetTest(unittest.TestCase):
    def test_subset_validated_against_inventory(self) -> None:
        self.assertEqual(t36_driver.check_subset("S08,S18"), ["S08", "S18"])

    def test_unknown_subset_refused(self) -> None:
        with self.assertRaises(t36_results.ResultError):
            t36_driver.check_subset("S99")


# the shape run 1 actually produced (t-36@r58): the lane's free-text
# stream dies at the tracker-load gate before any t36 TAP existed
RUN1_LOG = """[webdriver] ok: private headless weston up
[webdriver] ok: tauri-driver listening on loopback
[webdriver] ok: webdriver session created for the packaged app
[webdriver] ok: app frontend up and board loaded
[webdriver] ok: a real sac serve answers on the sandbox loopback
[webdriver] ok: t-0's seed board committed at revision 1
[webdriver] ok: t-1's seed board committed at revision 1
[webdriver] ok: the tracker UI takes real input
[webdriver] FAIL: a tracker location loads through the real server under the real CSP status=invalid errors=1 task refused
[webdriver] ok: a real sac serve answers on the sandbox loopback
[webdriver] some lane chatter that is not a check line
"""


class LaneResultsTest(unittest.TestCase):
    def test_gate_lines_become_tap_entries(self) -> None:
        results = t36_driver.lane_results_from_log(RUN1_LOG)
        by_id = {r["id"]: r for r in results}
        self.assertEqual(by_id["G01"]["status"], "pass")
        self.assertEqual(by_id["G04"]["status"], "pass")
        self.assertEqual(by_id["G05"]["status"], "fail")
        self.assertIn("status=invalid", by_id["G05"]["detail"])

    def test_dynamic_names_keep_their_evidence(self) -> None:
        results = t36_driver.lane_results_from_log(RUN1_LOG)
        seed = next(r for r in results if r["id"] == "G03")
        self.assertIn("t-0's seed board committed at revision 1", seed["name"])

    def test_unmatched_lines_stay_free_text(self) -> None:
        results = t36_driver.lane_results_from_log(RUN1_LOG)
        self.assertFalse(any("chatter" in r["name"] for r in results))

    def test_specific_matcher_wins(self) -> None:
        results = t36_driver.lane_results_from_log('[webdriver] ok: t-2\'s landing seed commits at revision 1 (text only)')
        self.assertEqual([r["id"] for r in results], ["G09"])

    def test_declared_checks_stop_against_the_failed_gate(self) -> None:
        results = t36_driver.lane_results_from_log(RUN1_LOG, declared_checks=("S12", "S13"))
        stops = [r for r in results if r["id"] in ("S12", "S13")]
        self.assertEqual(len(stops), 2)
        for stop in stops:
            self.assertEqual(stop["status"], "stop")
            self.assertIn("not reached", stop["detail"])
            self.assertIn("a tracker location loads", stop["detail"])
            self.assertIn("G05", stop["detail"])

    def test_declared_checks_stop_without_gate_failure(self) -> None:
        results = t36_driver.lane_results_from_log("[webdriver] ok: a real sac serve answers on the sandbox loopback", declared_checks=("S12",))
        stop = next(r for r in results if r["id"] == "S12")
        self.assertIn("produced no result for this check", stop["detail"])

    def test_unknown_declared_id_refused(self) -> None:
        with self.assertRaises(t36_results.ResultError):
            t36_driver.lane_results_from_log("", declared_checks=("S99",))

    def test_gate_failure_rules_even_with_repeated_passes(self) -> None:
        # run 1's log repeats the server-up gate after the failure; the
        # repeated passes never cancel the FAIL
        results = t36_driver.lane_results_from_log(RUN1_LOG)
        server_up = [r for r in results if r["id"] == "G01"]
        self.assertEqual(len(server_up), 2)
        self.assertTrue(all(r["status"] == "pass" for r in server_up))
        self.assertTrue(any(r["id"] == "G05" and r["status"] == "fail" for r in results))
        self.assertEqual(t36_results.tap_exit_code(t36_results.emit_tap(results)), 1)

    def test_run1_shape_yields_failing_tap_with_skips(self) -> None:
        results = t36_driver.lane_results_from_log(RUN1_LOG, declared_checks=("S12", "S18"))
        tap = t36_results.emit_tap(results)
        self.assertIn("not ok", tap)
        self.assertIn("G05/session", tap)
        self.assertIn("# SKIP stopped: not reached: the lane stopped at failed check", tap)
        self.assertEqual(t36_results.tap_exit_code(tap), 1)

    def test_tap_file_always_written(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        results = t36_driver.lane_results_from_log(RUN1_LOG, declared_checks=("S12",))
        manifest = {"schema": t36_results.SCHEMA}
        path = t36_driver.write_run_tap(pathlib.Path(tmp.name), "candidate", results, manifest)
        text = path.read_text(encoding="ascii")
        self.assertTrue(text.startswith("TAP version 13"))
        self.assertIn("1..", text)
        self.assertIn("not ok", text)

    def test_lane_exit_without_lines_fails_visibly(self) -> None:
        entry = t36_driver.lane_run_result(1, [])
        self.assertIsNotNone(entry)
        self.assertEqual(entry["id"], "G00")
        self.assertEqual(entry["status"], "fail")
        self.assertIn("exited 1", entry["detail"])
        tap = t36_results.emit_tap([entry])
        self.assertEqual(t36_results.tap_exit_code(tap), 1)

    def test_gate_failure_already_explains_the_exit(self) -> None:
        results = t36_driver.lane_results_from_log(RUN1_LOG)
        self.assertIsNone(t36_driver.lane_run_result(1, results))

    def test_clean_lane_yields_no_synthetic_entry(self) -> None:
        results = t36_driver.lane_results_from_log("[webdriver] ok: a real sac serve answers on the sandbox loopback")
        self.assertIsNone(t36_driver.lane_run_result(0, results))
        self.assertEqual(t36_results.tap_exit_code(t36_results.emit_tap(results)), 0)


class ServeRefCliTest(unittest.TestCase):
    """The refusal surface: no serve ref, no run - exit 2 from the CLI
    itself, not just from build_plans."""

    def store_bash(self) -> str:
        found = shutil.which("bash")
        assert found is not None and found.startswith("/nix/store/")
        return found

    def test_cli_refuses_without_serve_ref(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        board = pathlib.Path(tmp.name) / "board.yaml"
        board.write_text("grammar: 1\n", encoding="ascii")
        db = pathlib.Path(tmp.name) / "baseline.db"
        db.write_bytes(b"k")
        code = t36_driver.main([
            "--baseline-bin", self.store_bash(), "--baseline-ref", "4d59c35",
            "--candidate-bin", self.store_bash(), "--candidate-ref", "b0c33c2",
            "--board", str(board), "--tracker-db", str(db),
            "--serve-bin", self.store_bash(),
        ])
        self.assertEqual(code, 2)

    def test_cli_accepts_serve_ref_in_dry_run(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        board = pathlib.Path(tmp.name) / "board.yaml"
        board.write_text("grammar: 1\n", encoding="ascii")
        db = pathlib.Path(tmp.name) / "baseline.db"
        db.write_bytes(b"k")
        code = t36_driver.main([
            "--baseline-bin", self.store_bash(), "--baseline-ref", "4d59c35",
            "--candidate-bin", self.store_bash(), "--candidate-ref", "b0c33c2",
            "--board", str(board), "--tracker-db", str(db),
            "--serve-bin", self.store_bash(), "--serve-ref", "4d59c35",
            "--checks", "S08,S18",
        ])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()


class T36LaneWiringTest(unittest.TestCase):
    """The t36 lane wiring: dry-run shape, the fake-inventory lane pass
    (prelude gates merged with the check inventory, per-role fresh
    sessions, the per-check diff), and the S18 fresh-session premise
    honored by construction (up + down per role, never shared)."""

    def _options(self, tmp: str, lane: str = "t36", **overrides):
        store_bin = shutil.which("bash")
        assert store_bin is not None and store_bin.startswith("/nix/store/")
        board = pathlib.Path(tmp) / "board.yaml"
        board.write_text("grammar: 1\n", encoding="ascii")
        db = pathlib.Path(tmp) / "baseline.db"
        db.write_bytes(b"k")
        base = {
            "baseline_bin": store_bin,
            "baseline_ref": "4d59c35",
            "candidate_bin": store_bin,
            "candidate_ref": "671bac4",
            "board": str(board),
            "tracker_db": str(db),
            "serve_bin": store_bin,
            "serve_ref": "4d59c35",
            "checks": "",
            "execute": True,
            "lane": lane,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_dry_run_names_the_lane(self) -> None:
        import contextlib
        import io

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        options = self._options(tmp.name, execute=False)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = t36_driver.main([
                "--baseline-bin", options.baseline_bin, "--baseline-ref", options.baseline_ref,
                "--candidate-bin", options.candidate_bin, "--candidate-ref", options.candidate_ref,
                "--board", options.board, "--tracker-db", options.tracker_db,
                "--serve-bin", options.serve_bin, "--serve-ref", options.serve_ref,
                "--lane", "t36",
            ])
        self.assertEqual(code, 0)
        plan = json.loads(buffer.getvalue())
        self.assertEqual(plan["lane"], "t36")
        self.assertEqual(plan["plans"][0]["lane"], "t36")

    def test_fake_inventory_lane_pass(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        options = self._options(tmp.name)
        ups: list[str] = []
        downs: list[int] = []

        class FakeDeps(t36_driver.T36Deps):
            # the grounded baseline shape: F01, S12, S18 fail; S16
            # passes; everything else stops
            verdicts = {
                "F01": "fail", "S12": "fail", "S18": "fail", "S16": "pass",
            }

            def __init__(self, options: argparse.Namespace) -> None:
                self.options = options
                super().__init__(self._up, self._up, self._live_socket, self._request, self._down, self._prototype_factory)

            def _up(self, plan: dict, reuse_root: str = "") -> int:
                ups.append(plan["role"])
                return 0

            def _live_socket(self):
                return {"socket": "fake"}, "fake-sock"

            def _request(self, socket_path: str, payload: dict, timeout: float = 60.0) -> dict:
                if payload.get("cmd") == "windows":
                    return {"ok": True, "windows": [{"handle": "h-0", "label": "main"}], "suite": []}
                return {"ok": True}

            def _down(self) -> int:
                downs.append(1)
                return 0

            def _prototype_factory(self):
                return None, None

        # the check inventory runs through the fake surface: replace
        # run_checks' request path by stubbing the checks module for
        # the duration of the call
        real_run_checks = t36_checks.run_checks
        captured: dict = {}
        verdicts = {"F01": "fail", "S12": "fail", "S18": "fail", "S16": "pass"}

        def fake_run_checks(request, check_ids, pace=1.0, fresh_session=None):
            captured["ids"] = list(check_ids)
            results = []
            for check_id in check_ids:
                status = verdicts.get(check_id, "stop")
                detail = {"fail": "grounded baseline red", "pass": "agrees", "stop": "recorded gap"}[status]
                results.append(t36_results.make_result(check_id, "session", "fake body", status, detail))
            return results

        t36_checks.run_checks = fake_run_checks  # type: ignore[assignment]
        import isolation

        # the executor's run roots land in the test's tmp dir: a fake
        # lane never writes the shared runtime namespace (fake roots
        # there read as run evidence - the r73 confusion)
        tmp_roots: list[str] = []

        def fake_make_run_root(prefix: str) -> pathlib.Path:
            root = pathlib.Path(tmp.name) / f"root-{prefix}-{len(tmp_roots)}"
            (root / "logs").mkdir(parents=True)
            (root / "shots").mkdir()
            tmp_roots.append(str(root))
            return root

        try:
            options = self._options(tmp.name)
            deps = FakeDeps(options)
            out_root = pathlib.Path(tmp.name) / "out"
            out_root.mkdir()
            with unittest.mock.patch.object(isolation, "make_run_root", fake_make_run_root):
                code = t36_driver.execute_t36_lane(options, [
                    t36_driver.build_plan(
                        "baseline", options.baseline_bin, options.baseline_ref, options.board,
                        options.tracker_db, options.serve_bin, options.serve_ref, "0" * 40, "t-b", "now",
                    ),
                    t36_driver.build_plan(
                        "candidate", options.candidate_bin, options.candidate_ref, options.board,
                        options.tracker_db, options.serve_bin, options.serve_ref, "0" * 40, "t-c", "now",
                    ),
                ], out_root, deps)
        finally:
            t36_checks.run_checks = real_run_checks  # type: ignore[assignment]
        self.assertEqual(len(tmp_roots), 2)  # one run root per role, in the tmp dir

        self.assertEqual(code, 1)  # failures exist in the grounded shape
        self.assertEqual(ups, ["baseline", "candidate"])  # fresh session per role, in order
        self.assertEqual(len(downs), 2)
        self.assertEqual(captured["ids"], t36_driver.t36_inventory_ids())
        self.assertEqual(captured["ids"][-1], "F01")
        self.assertEqual(captured["ids"][0], "S01")
        self.assertIn("S18", captured["ids"])

        # the merged role TAPs exist in the run roots this pass
        # created, with gate and check entries and the manifest riding
        # as comments
        created = sorted((pathlib.Path(r) for r in tmp_roots), key=lambda p: p.stat().st_mtime)
        self.assertEqual(len(created), 2)
        baseline_root = next(r for r in created if r.name.startswith("root-t36-baseline-"))
        first = (baseline_root / "tap-baseline.tap").read_text(encoding="ascii")
        self.assertIn("# manifest:", first)
        self.assertIn("F01/session", first)
        self.assertIn("S18/session", first)
        # the diff lands in the out root
        diffs = list(out_root.rglob("diff.json"))
        self.assertEqual(len(diffs), 1)
        rows = json.loads(diffs[0].read_text())
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["F01"]["verdict"], "same")  # both roles fail identically

    def test_up_failure_fails_the_role_visibly(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        options = self._options(tmp.name)

        class UpFails(t36_driver.T36Deps):
            def __init__(self) -> None:
                super().__init__(self._up, self._up, self._live_socket, self._request, self._down, self._prototype_factory)

            def _up(self, plan: dict, reuse_root: str = "") -> int:
                return 1

            def _live_socket(self):
                return None

            def _request(self, socket_path: str, payload: dict, timeout: float = 60.0) -> dict:
                return {"ok": True}

            def _down(self) -> int:
                return 0

            def _prototype_factory(self):
                return None, None

        import isolation

        tmp_roots: list[pathlib.Path] = []

        def fake_make_run_root(prefix: str) -> pathlib.Path:
            root = pathlib.Path(tmp.name) / f"root-{prefix}-{len(tmp_roots)}"
            (root / "logs").mkdir(parents=True)
            (root / "shots").mkdir()
            tmp_roots.append(root)
            return root

        out_root = pathlib.Path(tmp.name) / "out"
        out_root.mkdir()
        with unittest.mock.patch.object(isolation, "make_run_root", fake_make_run_root):
            code = t36_driver.execute_t36_lane(options, [
                t36_driver.build_plan(
                    "baseline", options.baseline_bin, options.baseline_ref, options.board,
                    options.tracker_db, options.serve_bin, options.serve_ref, "0" * 40, "t-b", "now",
                ),
            ], out_root, UpFails())
        self.assertEqual(code, 1)
        self.assertEqual(len(tmp_roots), 1)  # the failed role's root, in the tmp dir
        tap = (tmp_roots[0] / "tap-baseline.tap").read_text(encoding="ascii")
        self.assertIn("not ok 1 - G00/session lane run", tap)
        self.assertIn("up-t36 exited 1", tap)

    def test_prelude_gates_merge_into_the_role_tap(self) -> None:
        # the prelude log the payload tees, parsed through the gates
        prelude = (
            "[webdriver] ok: the feed proxy listens on the core socket path\n"
            "[webdriver] ok: the t36 seed server answers\n"
            "[webdriver] FAIL: the t36 seed boards committed (groups on the real wire) refused\n"
        )
        results = t36_driver.lane_results_from_log(prelude)
        by_id = {r["id"]: r for r in results}
        self.assertEqual(by_id["G17"]["status"], "pass")
        self.assertEqual(by_id["G18"]["status"], "pass")
        self.assertEqual(by_id["G19"]["status"], "fail")
        tap = t36_results.emit_tap(results)
        self.assertEqual(t36_results.tap_exit_code(tap), 1)

    def test_inventory_ids_shape(self) -> None:
        ids = t36_driver.t36_inventory_ids()
        self.assertEqual(ids[:5], ["S01", "S02", "S03", "S04", "S05"])
        self.assertEqual(ids[-1], "F01")
        # 20 scenario groups + the D family (2 landed so far) + F01
        self.assertEqual(len(ids), 23)

    def test_unknown_lane_refused_at_cli(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        options = self._options(tmp.name, lane="teleport")
        store_bin = shutil.which("bash")
        with self.assertRaises(SystemExit) as caught:
            t36_driver.main([
                "--baseline-bin", store_bin, "--baseline-ref", "4d59c35",
                "--candidate-bin", store_bin, "--candidate-ref", "671bac4",
                "--board", options.board, "--tracker-db", options.tracker_db,
                "--serve-bin", store_bin, "--serve-ref", "4d59c35",
                "--lane", "teleport",
            ])
        self.assertEqual(caught.exception.code, 2)


class T36RelaunchWiringTest(unittest.TestCase):
    """The S20 relaunch wiring: the role's first up boots into the
    role's own run root (artifacts and persisted state land where the
    executor owns them), the request closure follows the relaunched
    payload's socket, and the relaunch is bounded hard - an overrun is
    a recorded timeout verdict, never an indefinite wait."""

    def test_relaunch_rebinds_socket_and_bounds_the_up(self) -> None:
        import threading
        import time as time_module
        from unittest import mock

        state = {"socket": "sock-a", "ups": [], "requests": []}

        class RelaunchDeps(t36_driver.T36Deps):
            def __init__(self) -> None:
                super().__init__(self._up, self._up_reuse, self._live_socket,
                                 self._request, self._down, self._prototype_factory)

            def _up(self, plan: dict, run_root: str = "") -> int:
                state["ups"].append(("up", run_root))
                state["socket"] = f"sock-{len(state['ups'])}"
                return 0

            def _up_reuse(self, plan: dict, run_root: str = "") -> int:
                time_module.sleep(1.0)  # longer than the patched bound
                return 0

            def _live_socket(self):
                return {"socket": state["socket"]}, state["socket"]

            def _request(self, socket_path: str, payload: dict, timeout: float = 60.0) -> dict:
                state["requests"].append(socket_path)
                return {"ok": True}

            def _down(self) -> int:
                return 0

            def _prototype_factory(self):
                return None, None

        deps = RelaunchDeps()

        def session_request(payload, timeout=60.0):
            live = deps.live_socket()
            return deps.request(live[1], payload, timeout)

        def relaunch() -> None:
            deps.down()
            worker = threading.Thread(target=lambda: deps.up_reuse({}, "root"), daemon=True)
            worker.start()
            worker.join(timeout=t36_driver.RELAYOUT_TIMEOUT_SECONDS)
            if worker.is_alive():
                raise t36_checks.CheckStopped("timeout verdict")

        # the socket follows the relaunched payload
        with mock.patch.object(t36_driver, "RELAYOUT_TIMEOUT_SECONDS", 0.2):
            with self.assertRaises(t36_checks.CheckStopped) as caught:
                relaunch()
        self.assertIn("timeout verdict", str(caught.exception))
        session_request({"probe": 1})
        self.assertEqual(state["requests"][-1], "sock-a")  # unchanged before a fresh up
        # a fresh up rebinds the socket and carries the run root
        deps.up({}, "root")
        session_request({"probe": 2})
        self.assertEqual(state["requests"][-1], "sock-1")
        self.assertEqual(state["ups"][0], ("up", "root"))


class BaselineCacheTest(unittest.TestCase):
    """The baseline verdict cache (r88): the key covers everything the
    baseline lane's verdicts can depend on; entries load only when
    whole and matching; corrupt or partial entries are ignored."""

    def _plan(self, harness_commit="a" * 40):
        return {
            "role": "baseline",
            "lane": "t36",
            "app_bin": "/nix/store/0000000000000000000000000000000-saccade-app-0.1.0/bin/saccade-app",
            "app_ref": "4d59c351a63b9a6448b4b5971f50284249211c4d",
            "manifest": {"harness_commit": harness_commit, "run_id": "run-x"},
        }

    def _options(self, checks="", board="/tmp/b/board.yaml"):
        return argparse.Namespace(
            baseline_bin="x", baseline_ref="x", candidate_bin="x", candidate_ref="x",
            board=board, tracker_db="/tmp/seed.db",
            serve_bin="/nix/store/0000000000000000000000000000000-saccade/bin/sac",
            serve_ref="4d59c351a63b9a6448b4b5971f50284249211c4d",
            checks=checks, lane="t36", fresh_baseline=False,
        )

    def test_key_is_stable_and_covers_harness_checks_and_constants(self):
        base = t36_driver.baseline_cache_key(self._plan(), self._options())
        # stable for identical inputs
        self.assertEqual(base, t36_driver.baseline_cache_key(self._plan(), self._options()))
        # a different harness commit changes the key
        self.assertNotEqual(base, t36_driver.baseline_cache_key(self._plan(harness_commit="b" * 40), self._options()))
        # a different checks subset changes the key
        self.assertNotEqual(base, t36_driver.baseline_cache_key(self._plan(), self._options(checks="S08")))
        # a different board changes the key
        self.assertNotEqual(base, t36_driver.baseline_cache_key(self._plan(), self._options(board="/tmp/other.yaml")))

    def test_store_then_load_roundtrip_and_corruption_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(t36_driver, "BASELINE_CACHE_ROOT", pathlib.Path(tmp)):
                key = "k" * 24
                t36_driver.baseline_cache_store(
                    key, [{"id": "S08", "status": "pass"}], {"run_id": "origin-1", "harness_commit": "a" * 40}, 1,
                )
                entry = pathlib.Path(tmp) / key
                self.assertTrue((entry / "results.json").exists())
                loaded = t36_driver.baseline_cache_load(key)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["results"], [{"id": "S08", "status": "pass"}])
                self.assertEqual(loaded["meta"]["origin_run"], "origin-1")
                self.assertEqual(loaded["meta"]["exit_code"], 1)
                # a mismatched meta key is rejected
                (entry / "meta.json").write_text(json.dumps({"key": "other"}), encoding="utf-8")
                self.assertIsNone(t36_driver.baseline_cache_load(key))
                # a missing results file is rejected
                t36_driver.baseline_cache_store(
                    key, [{"id": "S08", "status": "pass"}], {"run_id": "origin-2"}, 1,
                )
                (entry / "results.json").unlink()
                self.assertIsNone(t36_driver.baseline_cache_load(key))
