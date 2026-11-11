#!/usr/bin/env python3
"""P1 check-body tests: the three required reds driven through a FAKE
session surface, shaped by what was grounded live at 4d59c35.

The fake answers the lane verbs the way the real session answered
during grounding (the run-1-era baseline): open-window creates a
window whose page never receives, evaluate reads state, the feed
proxy records subscribes and the baseline feed breaks on unknown
events. Both shapes exist: baseline (red expected) and fixed
(green expected) - the bodies must tell them apart through the same
readouts the real surface carries."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_checks  # noqa: E402
import t36_results  # noqa: E402

BOARD_SCRIPT_ID = "board:open-citation"


class FakeSession:
    """A scripted session surface. `delivered` decides the shape: at
    baseline no delivery ever lands and __suiteOpenAt is undefined."""

    def __init__(self, delivered: bool = False, feed_breaks: bool = True, windows: int = 1) -> None:
        self.delivered = delivered
        self.feed_breaks = feed_breaks
        self.labels = ["main"] + [f"w-extra-{i}" for i in range(windows - 1)]
        self.created: list[str] = []
        self.keys: list[list[str]] = []
        self.feed = {"subscribes": ['{"id":1,"jsonrpc":"2.0","method":"events.subscribe","params":{}}'],
                     "connections": 1, "total_subscribes": 1, "queued": 0}
        self.injections: list[dict] = []
        self.unreachable_banner = False
        self.generation = 1
        self.connected = True
        self.problem: str | None = None
        self.words: dict = {}
        self.tracker = {
            "base": "http://127.0.0.1:46255",
            "tasks": {
                "t-0": {"task": "t-0", "revision": 1, "message": "t36 seed: the cited account, group-made"},
                "t-1": {"task": "t-1", "revision": 1, "message": "t36 seed: the citing account, incorporation into a group child"},
            },
        }
        self.loaded = {"status": "loaded", "task": "t-1"}

    # -- the verbs ---------------------------------------------------------

    def _window_payload(self, label: str) -> dict:
        return {"path": "/board/t36-dev-board.yaml", "task": None,
                "suiteOpenAt": "undefined", "stripLocation": "/board/t36-dev-board.yaml",
                "linkChip": None}

    def _receiver_payload(self, label: str) -> dict:
        if self.delivered:
            return {"path": "t-1@r1:/content/0", "task": "t-0",
                    "suiteOpenAt": "function", "stripLocation": "seedtracker/t-0@r1", "linkChip": None}
        return self._window_payload(label)

    def __call__(self, payload: dict, timeout: float = 40.0) -> dict:
        cmd = payload.get("cmd")
        script = str(payload.get("script") or "")
        if cmd == "windows":
            return {"ok": True, "windows": [{"handle": f"h-{label}" , "label": label} for label in self.labels],
                    "suite": [{"label": label} for label in self.labels]}
        if cmd == "open-window":
            label = f"w-created-{len(self.created)}"
            self.created.append(label)
            self.labels.append(label)
            return {"ok": True, "label": label, "windows": [{"handle": label, "label": label}]}
        if cmd == "tracker":
            return {"ok": True, **self.tracker}
        if cmd == "feed":
            action = payload.get("action")
            if action == "status":
                return {"ok": True, "feed": dict(self.feed)}
            if action == "inject-status":
                self.injections.append(payload)
                self.generation += 1
                self.words[str(payload.get("terminal") or "t-1")] = {
                    "state": payload.get("state") or "working", "msg": payload.get("msg") or "",
                }
                return {"ok": True, "delivered": 1}
            if action == "inject-unknown":
                self.injections.append(payload)
                if self.feed_breaks:
                    # the grounded baseline: the stream ends, the table
                    # clears, the problem names the closed stream
                    self.generation += 1
                    self.connected = False
                    self.problem = "the core at /run/herdr-core/default.sock closed the event stream"
                    self.words = {}
                return {"ok": True, "delivered": 1}
            raise AssertionError(f"unknown feed action {action}")
        if cmd == "evaluate":
            return {"ok": True, "value": self._evaluate(script)}
        if cmd == "evaluate-window":
            label = payload.get("window")
            if label not in self.labels:
                return {"ok": False, "error": f"no webview carries label {label!r}"}
            data = self._receiver_payload(label) if label != "main" else self._main_payload()
            if "jump-chip-self" in script:
                return {"ok": True, "value": "b" if label != "main" else "a"}
            return {"ok": True, "value": json.dumps(data)}
        if cmd == "input":
            if payload.get("kind") == "key":
                keys = payload.get("keys") or []
                self.keys.append(keys)
                # the grounded baseline behavior: o with one window
                # opens a new window (and makes it the link); w opens
                # a viewer; the receiver never receives
                if keys == ["o"] and len(self.labels) == 1:
                    self.labels.append(f"w-o-{len(self.created)}")
                    self.created.append("o")
                elif keys == ["w"]:
                    self.labels.append(f"w-w-{len(self.created)}")
                    self.created.append("w")
            return {"ok": True, "input": payload}
        raise AssertionError(f"unknown command {cmd}")

    def _main_payload(self) -> dict:
        return {"path": "seedtracker/t-1@r1", "task": "t-1", "suiteOpenAt": "undefined",
                "stripLocation": "seedtracker/t-1@r1", "linkChip": None}

    def _evaluate(self, script: str) -> object:
        if ".select-label" in script and "map(" in script:
            return json.dumps([
                {"label": "A", "id": "strip", "cls": "select-label select-label-other"},
                {"label": "C", "id": BOARD_SCRIPT_ID, "cls": "select-label select-label-other"},
                {"label": "a", "id": "b0", "cls": "select-label select-label-text"},
            ])
        if ".select-selected" in script:
            # selection lands only when the last typed letter matched
            # the displayed case (the fake tracks nothing more: the
            # bodies read this once per act)
            last = [keys[-1] if keys else "" for keys in self.keys[-1:]]
            typed = last[0] if last else ""
            if typed and typed.isupper():
                return json.dumps({"selected": True, "id": BOARD_SCRIPT_ID if typed == "C" else "strip"})
            return json.dumps({"selected": False, "id": None})
        if "words_wait" in script:
            return json.dumps({
                "generation": self.generation, "words": self.words,
                "connected": self.connected, "problem": self.problem,
            })
        if "jump-overlay" in script:
            return json.dumps({"askVisible": True, "askText": "open elsewhere: a label, n new", "selfChip": "a"})
        if "inspection()" in script and "resource.status" in script:
            return json.dumps(self.loaded)
        return None


class S18Test(unittest.TestCase):
    def test_baseline_red_is_fail_with_receiver_evidence(self) -> None:
        fake = FakeSession(delivered=False)
        spec = t36_checks.CHECKS["S18"]
        result = t36_checks.run_check(fake, spec, pace=0)
        self.assertEqual(result["id"], "S18")
        self.assertEqual(result["status"], "fail")
        self.assertIn("the receiver never receives", result["detail"])
        self.assertIn("__suiteOpenAt", result["detail"])

    def test_delivered_shape_is_green(self) -> None:
        fake = FakeSession(delivered=True)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S18"], pace=0)
        self.assertEqual(result["status"], "pass", result["detail"])

    def test_multi_window_start_is_a_stop(self) -> None:
        fake = FakeSession(windows=2)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S18"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("exactly one window", result["detail"])

    def test_board_never_loads_is_a_stop(self) -> None:
        fake = FakeSession()
        fake.loaded = {"status": "loading", "task": None}
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S18"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("never loaded", result["detail"])


class FeedTest(unittest.TestCase):
    def test_baseline_red_names_the_closed_stream(self) -> None:
        fake = FakeSession(feed_breaks=True)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["F01"], pace=0)
        self.assertEqual(result["id"], "F01")
        self.assertEqual(result["status"], "fail")
        self.assertIn("treated the unknown event type as the stream's end", result["detail"])
        self.assertIn("closed the event stream", result["detail"])

    def test_surviving_feed_is_green(self) -> None:
        fake = FakeSession(feed_breaks=False)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["F01"], pace=0)
        self.assertEqual(result["status"], "pass")
        self.assertIn("survived the unknown event type", result["detail"])

    def test_no_subscription_is_a_stop(self) -> None:
        fake = FakeSession()
        fake.feed["total_subscribes"] = 0
        result = t36_checks.run_check(fake, t36_checks.CHECKS["F01"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("never subscribed", result["detail"])

    def test_unhealthy_status_never_landing_is_a_stop(self) -> None:
        # the words table must show the healthy status before the
        # probe: a feed that never renders it cannot serve the check
        class DeafSession(FakeSession):
            def _evaluate(self, script: str) -> object:
                if "words_wait" in script:
                    return json.dumps({"generation": 1, "words": {}, "connected": True, "problem": None})
                return super()._evaluate(script)

        result = t36_checks.run_check(DeafSession(feed_breaks=False), t36_checks.CHECKS["F01"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("healthy status never reached", result["detail"])


class S12Test(unittest.TestCase):
    def test_baseline_red_lowercase_never_selects(self) -> None:
        fake = FakeSession()
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S12"], pace=0)
        self.assertEqual(result["id"], "S12")
        self.assertEqual(result["status"], "fail")
        self.assertIn("typed 'a'", result["detail"])
        self.assertIn("displayed label is 'A'", result["detail"])
        self.assertIn("case-fold fix is held", result["detail"])

    def test_case_folded_match_is_green(self) -> None:
        fake = FakeSession()
        # the fixed matcher selects on the lowercase letter too: the
        # fake accepts the last typed letter when the body typed the
        # lowercased label - simulated by a selecting fake
        def selecting(script: str) -> object:
            if ".select-selected" in script:
                return json.dumps({"selected": True, "id": "strip"})
            return FakeSession._evaluate(fake, script)
        fake._evaluate = selecting  # type: ignore[method-assign]
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S12"], pace=0)
        self.assertEqual(result["status"], "pass")

    def test_no_uppercase_chip_is_a_stop(self) -> None:
        class TextOnlySession(FakeSession):
            def _evaluate(self, script: str) -> object:
                if ".select-label" in script and "map(" in script:
                    return json.dumps([{"label": "a", "id": "b0", "cls": "select-label select-label-text"}])
                return super()._evaluate(script)

        result = t36_checks.run_check(TextOnlySession(), t36_checks.CHECKS["S12"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("no uppercase chip", result["detail"])


class RegistryTest(unittest.TestCase):
    def test_p1_bodies_registered(self) -> None:
        for check_id in ("S18", "F01", "S12"):
            self.assertIn(check_id, t36_checks.CHECKS)

    def test_unknown_inventory_id_stops(self) -> None:
        fake = FakeSession()
        results = t36_checks.run_checks(fake, ["S17"])
        self.assertEqual(results[0]["status"], "stop")
        self.assertIn("no executable body", results[0]["detail"])

    def test_runner_runs_in_named_order(self) -> None:
        fake = FakeSession()
        results = t36_checks.run_checks(fake, ["S12", "F01"], pace=0)
        self.assertEqual([r["id"] for r in results], ["S12", "F01"])

    def test_results_are_tap_ready(self) -> None:
        fake = FakeSession()
        results = t36_checks.run_checks(fake, list(t36_checks.CHECKS), pace=0)
        tap = t36_results.emit_tap(results)
        self.assertIn("S18/session", tap)
        self.assertIn("F01/session", tap)

    def test_machinery_error_is_a_stop_not_a_fail(self) -> None:
        def exploding(request: dict, timeout: float = 40.0) -> dict:
            raise RuntimeError("harness bug")
        result = t36_checks.run_check(exploding, t36_checks.CHECKS["S12"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("check machinery error", result["detail"])


if __name__ == "__main__":
    unittest.main()
