#!/usr/bin/env python3
"""R2 unit tests: the two-run driver's provenance gate, plan
constancy, and the per-check diff. Pure - no lane, no app, no nix."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
import unittest

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

    def test_plan_carries_provenance_in_manifest(self) -> None:
        plans = self.plans()
        baseline = plans[0]["manifest"]
        self.assertEqual(baseline["app"]["ref"], "4d59c35")
        self.assertEqual(baseline["app"]["bin"], plans[0]["app_bin"])
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


if __name__ == "__main__":
    unittest.main()
