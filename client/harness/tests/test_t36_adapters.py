#!/usr/bin/env python3
"""Adapter-set unit tests: the registry, the named-only guard, the
oracle rule. Pure."""

from __future__ import annotations

import sys
import unittest
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_adapters  # noqa: E402


class AdapterSetTest(unittest.TestCase):
    def test_seven_rows_with_ids_a1_to_a7(self) -> None:
        # the app-side adapter set grew the port rows (P1/P2 slice-2,
        # P3 the port trail readout for S05/S07): the app-side ids stay
        # A1..A7, the port rows follow under their own prefix
        self.assertEqual(t36_adapters.ADAPTER_IDS[:7], tuple(f"A{n}" for n in range(1, 8)))
        self.assertEqual(t36_adapters.ADAPTER_IDS[7:], ("P1", "P3", "P4", "P2"))

    def test_revision_is_a_printable_string(self) -> None:
        revision = t36_adapters.ADAPTER_SET_REVISION
        self.assertTrue(revision)
        revision.encode("ascii")

    def test_every_row_has_transformation_and_kind(self) -> None:
        for adapter in t36_adapters.ADAPTERS:
            self.assertIn(adapter["kind"], t36_adapters.KINDS)
            self.assertTrue(adapter["transformation"])
            self.assertIn(adapter["verification"], (t36_adapters.VERIFIED, t36_adapters.UNVERIFIED))

    def test_named_scenarios_match_the_design_table(self) -> None:
        self.assertEqual(t36_adapters.adapter_of("A1")["scenarios"], ["S10"])
        self.assertEqual(t36_adapters.adapter_of("A2")["scenarios"], ["S16"])
        self.assertEqual(sorted(t36_adapters.adapter_of("A3")["scenarios"]), ["S12", "S17"])
        self.assertEqual(t36_adapters.adapter_of("A4")["scenarios"], ["S08"])
        self.assertEqual(t36_adapters.adapter_of("A5")["scenarios"], ["S14"])
        self.assertEqual(t36_adapters.adapter_of("A6")["scenarios"], ["S17"])
        self.assertEqual(t36_adapters.adapter_of("A7")["scenarios"], [])

    def test_rows_are_unverified_until_the_first_run(self) -> None:
        # the design: prototype behaviour is recorded at the first run;
        # only the oracle row (a board fact, not executed behaviour) is
        # verified from the start
        for adapter in t36_adapters.ADAPTERS:
            if adapter["kind"] == t36_adapters.KIND_ORACLE:
                self.assertEqual(adapter["verification"], t36_adapters.VERIFIED)
            else:
                self.assertEqual(adapter["verification"], t36_adapters.UNVERIFIED)
                self.assertEqual(adapter["evidence"], "")


class GuardTest(unittest.TestCase):
    def test_named_application_passes(self) -> None:
        t36_adapters.assert_named_only("S08", ("A4",))
        t36_adapters.assert_named_only("S17", ("A3", "A6"))
        self.assertEqual(t36_adapters.named_for_result("S17", ("A3", "A6")), ("A3", "A6"))

    def test_misapplication_refused(self) -> None:
        with self.assertRaises(t36_adapters.AdapterMisapplied):
            t36_adapters.assert_named_only("S12", ("A4",))
        with self.assertRaises(t36_adapters.AdapterMisapplied):
            t36_adapters.assert_named_only("S16", ("A5",))

    def test_unknown_adapter_refused(self) -> None:
        with self.assertRaises(t36_adapters.AdapterMisapplied):
            t36_adapters.assert_named_only("S08", ("A9",))

    def test_oracle_is_never_applicable(self) -> None:
        with self.assertRaises(t36_adapters.AdapterMisapplied):
            t36_adapters.assert_named_only("S19", ("A7",))

    def test_adapters_for_aggregates(self) -> None:
        self.assertEqual(t36_adapters.adapters_for("S17"), ("A3", "A6"))
        self.assertEqual(t36_adapters.adapters_for("S01"), ())


if __name__ == "__main__":
    unittest.main()
