#!/usr/bin/env python3
"""R1 unit tests: check inventory, result records, TAP emitter, run
manifest validation. Pure; no binary, no browser, no app."""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_adapters  # noqa: E402
import t36_results  # noqa: E402


class InventoryTest(unittest.TestCase):
    def test_twenty_stable_ids(self) -> None:
        self.assertEqual(t36_results.SCENARIO_IDS, tuple(f"S{n:02d}" for n in range(1, 21)))

    def test_titles_are_ascii_and_nonempty(self) -> None:
        for scenario in t36_results.SCENARIOS:
            self.assertTrue(scenario["title"])
            scenario["title"].encode("ascii")

    def test_scenario_lookup_refuses_unknown(self) -> None:
        with self.assertRaises(t36_results.ResultError):
            t36_results.scenario_of("S21")

    def test_lanes_covered(self) -> None:
        for lane in t36_results.LANES:
            self.assertIn(lane, ("pure", "component", "session", "prototype", "browser"))


class ResultTest(unittest.TestCase):
    def test_make_result_roundtrip(self) -> None:
        result = t36_results.make_result("S08", "session", "fold reveal shows the anchor", t36_results.STATUS_PASS, "anchor reached")
        self.assertEqual(result["id"], "S08")
        self.assertEqual(result["lane"], "session")
        self.assertEqual(result["status"], "pass")

    def test_refuses_unknown_scenario_lane_status(self) -> None:
        for bad in (
            {"scenario_id": "S99", "lane": "session", "name": "x", "status": "pass"},
            {"scenario_id": "S01", "lane": "teleport", "name": "x", "status": "pass"},
            {"scenario_id": "S01", "lane": "session", "name": "x", "status": "meh"},
            {"scenario_id": "S01", "lane": "session", "name": "", "status": "pass"},
        ):
            with self.assertRaises(t36_results.ResultError):
                t36_results.make_result(**bad)


class TapTest(unittest.TestCase):
    def results(self) -> list[dict]:
        return [
            t36_results.make_result("S18", "session", "linked delivery", t36_results.STATUS_FAIL, "receiver wrote nowhere"),
            t36_results.make_result("S08", "session", "fold reveal", t36_results.STATUS_PASS),
            t36_results.make_result("S14", "session", "trail flags", t36_results.STATUS_STOP, "no existing readout for the app trail"),
        ]

    def test_tap_plan_counts_and_order(self) -> None:
        tap = t36_results.emit_tap(self.results())
        lines = tap.splitlines()
        self.assertEqual(lines[0], "TAP version 13")
        self.assertEqual(lines[1], "1..3")
        # ordered by scenario id: S08 before S14 before S18
        self.assertIn("ok 1 - S08/session fold reveal", lines[2])
        self.assertIn("not ok 2 - S14/session trail flags # SKIP stopped: no existing readout for the app trail", lines[3])
        self.assertIn("not ok 3 - S18/session linked delivery", lines[4])

    def test_fail_detail_rides_yaml_block(self) -> None:
        tap = t36_results.emit_tap(self.results())
        self.assertIn("    detail: |", tap)
        self.assertIn("      receiver wrote nowhere", tap)

    def test_stop_never_counts_as_failure(self) -> None:
        tap = t36_results.emit_tap([t36_results.make_result("S14", "session", "trail flags", t36_results.STATUS_STOP, "nothing to read")])
        self.assertEqual(t36_results.tap_exit_code(tap), 0)

    def test_fail_fails_the_gate(self) -> None:
        tap = t36_results.emit_tap([t36_results.make_result("S18", "session", "linked delivery", t36_results.STATUS_FAIL, "x")])
        self.assertEqual(t36_results.tap_exit_code(tap), 1)

    def test_all_pass_gate_green(self) -> None:
        results = [t36_results.make_result("S01", "pure", "latest intent", t36_results.STATUS_PASS)]
        self.assertEqual(t36_results.tap_exit_code(t36_results.emit_tap(results)), 0)

    def test_manifest_rides_as_comments(self) -> None:
        manifest = {"schema": t36_results.SCHEMA, "lane": "session"}
        tap = t36_results.emit_tap(self.results(), manifest=manifest)
        self.assertIn("# manifest: ", tap)

    def test_adapter_ids_recorded_on_result(self) -> None:
        result = t36_results.make_result("S08", "session", "fold reveal", t36_results.STATUS_PASS, adapters=("A4",))
        tap = t36_results.emit_tap([result])
        self.assertIn("adapters: A4", tap)


def _valid_manifest(app_bin: str) -> dict:
    return t36_results.make_manifest(
        lane="session",
        harness_commit="0" * 40,
        app_bin=app_bin,
        app_ref="4d59c35",
        adapter_set_revision=t36_adapters.ADAPTER_SET_REVISION,
        serve_bin="/usr/bin/env",
        seeds={"db_path": "/tmp/x.db", "db_hash": "a" * 64, "commits": [{"task": "t-5", "revision": 1, "message": "seed"}]},
        prototype={"source": "t-22@r3", "sha256": "b" * 64},
    )


class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # the manifest rule matches the lanes: SAC_APP_BIN must be an
        # executable /nix/store path, so the positive cases point at a
        # real store binary (the pinned devshell always provides one)
        found = shutil.which("bash")
        assert found is not None and found.startswith("/nix/store/")
        self.app_bin = found

    def test_valid_manifest_has_no_refusals(self) -> None:
        self.assertEqual(t36_results.validate_manifest(_valid_manifest(str(self.app_bin))), [])

    def test_missing_provenance_refused(self) -> None:
        manifest = _valid_manifest(str(self.app_bin))
        manifest["app"]["ref"] = ""
        refusals = t36_results.validate_manifest(manifest)
        self.assertTrue(any(r.startswith("app.ref") for r in refusals))

    def test_app_bin_must_be_store_path(self) -> None:
        manifest = _valid_manifest(str(self.app_bin))
        manifest["app"]["bin"] = "/home/jerry/app"
        refusals = t36_results.validate_manifest(manifest)
        self.assertTrue(any(r.startswith("app.bin") for r in refusals))

    def test_require_seeds_refuses_a_seedless_manifest(self) -> None:
        manifest = _valid_manifest(str(self.app_bin))
        del manifest["seeds"]
        self.assertEqual(t36_results.validate_manifest(manifest), [])
        refusals = t36_results.validate_manifest(manifest, require_seeds=True)
        self.assertTrue(any(r.startswith("seeds") for r in refusals))

    def test_bad_seed_hash_refused(self) -> None:
        manifest = _valid_manifest(str(self.app_bin))
        manifest["seeds"]["db_hash"] = "zz"
        refusals = t36_results.validate_manifest(manifest, require_seeds=True)
        self.assertTrue(any(r.startswith("seeds.db_hash") for r in refusals))

    def test_schema_drift_refused(self) -> None:
        manifest = _valid_manifest(str(self.app_bin))
        manifest["schema"] = "t36-run-manifest/0"
        self.assertTrue(t36_results.validate_manifest(manifest))

    def test_write_then_load_roundtrip(self) -> None:
        path = pathlib.Path(self.tmp.name) / "manifest.json"
        t36_results.write_manifest(str(path), _valid_manifest(str(self.app_bin)))
        loaded = t36_results.load_manifest(str(path))
        self.assertEqual(t36_results.validate_manifest(loaded, require_seeds=True, require_prototype=True), [])
        self.assertEqual(loaded["adapter_set_revision"], t36_adapters.ADAPTER_SET_REVISION)

    def test_write_refuses_invalid(self) -> None:
        path = pathlib.Path(self.tmp.name) / "manifest.json"
        manifest = _valid_manifest(str(self.app_bin))
        manifest["app"]["ref"] = ""
        with self.assertRaises(t36_results.ResultError):
            t36_results.write_manifest(str(path), manifest)
        self.assertFalse(path.exists())

    def test_manifest_is_ascii_json(self) -> None:
        path = pathlib.Path(self.tmp.name) / "manifest.json"
        t36_results.write_manifest(str(path), _valid_manifest(str(self.app_bin)))
        raw = path.read_bytes()
        raw.decode("ascii")


class HarnessCommitTest(unittest.TestCase):
    def test_commit_of_this_checkout(self) -> None:
        repo = str(HARNESS.parent.parent)
        sha = t36_results.harness_commit_of(repo)
        self.assertRegex(sha, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
