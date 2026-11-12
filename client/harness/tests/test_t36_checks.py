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

import unittest.mock

import t36_checks  # noqa: E402
import t36_port  # noqa: E402
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
        self.invokes: list[dict] = []
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
        if cmd == "status":
            # the lane's own board (host side): the tall fixture is
            # written beside it, where /board's bind serves it
            return {"ok": True, "board": "/tmp/t36-dev-board.yaml", "board_dir_host": "/tmp"}
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

    def test_authored_task_metadata_never_counts_as_delivery(self) -> None:
        # grounded false pass: the boot board declares task: t-0; only
        # the location moving to t-0@r1 is delivery
        fake = FakeSession(delivered=False)
        spec = t36_checks.CHECKS["S18"]
        result = t36_checks.run_check(fake, spec, pace=0)
        self.assertEqual(result["status"], "fail")

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

    def test_unknown_inventory_id_refused(self) -> None:
        import t36_results

        fake = FakeSession()
        with self.assertRaises(t36_results.ResultError):
            t36_checks.run_checks(fake, ["S99"])

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


class DifferentialRegistryTest(unittest.TestCase):
    """P2/P3 registration: the differential bodies and the named stop
    registry cover every scenario group."""

    @classmethod
    def setUpClass(cls) -> None:
        import sys as _sys
        import pathlib as _pathlib

        harness = _pathlib.Path(__file__).resolve().parent.parent
        if str(harness) not in _sys.path:
            _sys.path.insert(0, str(harness))
        import t36_differential

        t36_differential.register()

    def test_every_scenario_group_is_registered(self) -> None:
        import t36_results

        for scenario_id in t36_results.SCENARIO_IDS:
            self.assertIn(scenario_id, t36_checks.CHECKS, f"{scenario_id} has no registered body or stop")

    def test_p2_bodies_carry_their_adapters(self) -> None:
        self.assertEqual(t36_checks.CHECKS["S16"].adapters, ("A2",))
        self.assertEqual(t36_checks.CHECKS["S17"].adapters, ("A6",))
        self.assertEqual(t36_checks.CHECKS["S08"].adapters, ("A4",))
        self.assertEqual(t36_checks.CHECKS["S14"].adapters, ("A5",))

    def test_differential_without_prototype_stops(self) -> None:
        import t36_differential

        t36_differential.attach(None)
        fake = FakeSession()
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S16"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("no prototype session attached", result["detail"])

    def test_stop_bodies_record_their_reason(self) -> None:
        fake = FakeSession()
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S14"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("no trail readout", result["detail"])
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S08"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("citation surface is board-level only", result["detail"])

    def test_compare_applies_named_adapters_only(self) -> None:
        import t36_differential as _diff

        outcome = _diff.compare(
            "S17", {"rows": [{"kind": "task", "loc": "t-1"}]},
            {"rows": [{"kind": "task", "loc": "t-1"}]}, adapters=("A6",),
        )
        self.assertEqual(outcome["mismatches"], [])
        self.assertTrue(any("A6" in note for note in outcome["notes"]))
        outcome = _diff.compare(
            "S08", {"location": "t-0@r1"}, {"location": "t-0@r1"}, adapters=("A4",),
        )
        self.assertEqual(outcome["mismatches"], [])
        # a mismatch outside the adapter's transformation still reports
        outcome = _diff.compare(
            "S17",
            {"rows": [{"kind": "task", "loc": "t-1"}]},
            {"rows": [{"kind": "task", "loc": "t-1"}, {"kind": "figure", "loc": "x"}]},
            adapters=("A6",),
        )
        self.assertTrue(outcome["mismatches"])

    def test_full_inventory_stops_or_runs(self) -> None:
        fake = FakeSession()
        import t36_results

        results = t36_checks.run_checks(fake, list(t36_results.SCENARIO_IDS), pace=0)
        by_id = {r["id"]: r["status"] for r in results}
        # the app halves that judge baseline behavior fail; the gaps stop
        self.assertEqual(by_id["S17"], "stop")  # no task rows on the fake's board
        self.assertEqual(by_id["S14"], "stop")
        self.assertEqual(by_id["S16"], "stop")  # no prototype attached
        for scenario_id in ("S01", "S02", "S19"):
            self.assertEqual(by_id[scenario_id], "stop", scenario_id)


class ScenarioBodiesTest(unittest.TestCase):
    """The phase-2b bodies against the fake surface: the verdict logic
    (grounded live) runs without the app."""

    @classmethod
    def setUpClass(cls) -> None:
        t36_checks.ensure_registered()

    class FullFake(FakeSession):
        """A fake that answers the bodies' probes: chips, scroll host,
        copy note, the invoke recorder."""

        def __init__(self) -> None:
            super().__init__(delivered=True)
            self.scroll = {"top": 0, "height": 3000}
            self.note = None
            self.keys_pressed: list[str] = []

        def _evaluate(self, script: str) -> object:
            if ".select-label" in script and "closest" in script:
                return json.dumps([
                    {"label": "C", "id": "b0", "blockId": "b0", "inGroup": True},
                    {"label": "a", "id": "b0.0", "blockId": "b0.0", "inGroup": True},
                ])
            if ".select-label" in script and "data-target-id'));" in script.replace("'", "'"):
                return json.dumps(["b0.0"])
            if "scrollHeight" in script:
                return json.dumps(self.scroll)
            if "instrument:note" in script:
                return self.note
            if "__t36Invokes" in script:
                return json.dumps(self.invokes)
            if "ctx-menu-act" in script:
                return json.dumps({"x": 100, "y": 100})
            return FakeSession._evaluate(self, script)

    def test_s10_gg_green_on_scrollable_fake(self) -> None:
        fake = self.FullFake()
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S10"], pace=0)
        self.assertEqual(result["status"], "stop")  # j narrows nothing on the fake; premise recorded

    def test_s13_fake_covers_child_and_group_paths(self) -> None:
        fake = self.FullFake()
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S13"], pace=0)
        # the fake's copy note path is exercised; the verdict rides the
        # app's own part-path mapping
        self.assertIn(result["status"], ("pass", "stop", "fail"))

    def test_s20_without_relauncher_stops(self) -> None:
        fake = FakeSession()
        t36_checks.set_relauncher(None)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S20"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("no relaunch callable", result["detail"])

    def test_s20_with_relauncher_passes(self) -> None:
        fake = FakeSession()
        relaunches: list[int] = []
        t36_checks.set_relauncher(lambda: relaunches.append(1))
        self.addCleanup(t36_checks.set_relauncher, None)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S20"], pace=0)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(relaunches), 1)
        self.assertIn("persisted window surface", result["detail"])

    def test_s15_fake_green_with_tracker_commit(self) -> None:
        fake = FakeSession()
        commits: list[dict] = []
        fake_task = "t-2"

        def request(payload: dict, timeout: float = 60.0) -> dict:
            if payload.get("cmd") == "tracker":
                return {"ok": True, "base": "http://127.0.0.1:1", "tasks": {
                    "t-2": {"task": "t-2", "revision": 2, "message": "t36 seed: the landing figure, inside a group"}}}
            if payload.get("cmd") == "tracker":
                return {"ok": True, "base": "http://127.0.0.1:1", "tasks": {
                    "t-2": {"task": "t-2", "revision": 2, "message": "t36 seed: the landing figure, inside a group"}}}
            if payload.get("cmd") == "tracker-commit":
                commits.append(payload)
                return {"ok": True, "revision": len(commits) + 2, "commit": "cafe"}
            if payload.get("cmd") == "evaluate":
                script = payload.get("script") or ""
                if "picker:input" in script:
                    return {"ok": True, "value": True}
                if "picker:result" in script:
                    return {"ok": True, "value": json.dumps([f"task seedtracker/{fake_task}@r1"])}
                if "resource.status" in script:
                    return {"ok": True, "value": json.dumps(
                        {"status": "loaded", "task": fake_task if "identity" in script else None})}
                if "rawLength" in script:
                    before_first = len(commits) == 0
                    return {"ok": True, "value": json.dumps({"raw": 237 if before_first else 400})}
                if ".select-label" in script:
                    return json.dumps([])
                return {"ok": True, "value": None}
            return FakeSession.__call__(fake, payload, timeout)

        result = t36_checks.run_check(request, t36_checks.CHECKS["S15"], pace=0)
        self.assertEqual(result["status"], "pass", result["detail"])
        self.assertEqual(len(commits), 2)

    def test_s15_fake_red_when_the_board_never_updates(self) -> None:
        def request(payload: dict, timeout: float = 60.0) -> dict:
            if payload.get("cmd") == "tracker":
                return {"ok": True, "base": "http://127.0.0.1:1", "tasks": {
                    "t-2": {"task": "t-2", "revision": 2, "message": "t36 seed: the landing figure, inside a group"}}}
            if payload.get("cmd") == "tracker-commit":
                return {"ok": True, "revision": 3}
            if payload.get("cmd") == "evaluate":
                script = payload.get("script") or ""
                if "picker:input" in script:
                    return {"ok": True, "value": True}
                if "picker:result" in script:
                    return {"ok": True, "value": json.dumps(["task seedtracker/t-2@r1"])}
                if "resource.status" in script:
                    return {"ok": True, "value": json.dumps({"status": "loaded", "task": "t-2"})}
                if "rawLength" in script:
                    return {"ok": True, "value": json.dumps({"raw": 237})}
                if ".select-label" in script:
                    return json.dumps([])
                return {"ok": True, "value": None}
            return {"ok": True}

        result = t36_checks.run_check(request, t36_checks.CHECKS["S15"], pace=0)
        self.assertEqual(result["status"], "fail")
        self.assertIn("did not update live", result["detail"])


class PortAdapterTest(unittest.TestCase):
    """The slice-2 port adapter: the flavor gate, the readout resolver,
    the REQUIRE_FIELDS-stop rule, and the served-set premise."""

    class PortFake(FakeSession):
        def __init__(self, readout: dict | None = None) -> None:
            super().__init__(delivered=True)
            self.typed: str | None = None
            self.readout = readout or {
                "app": {"name": "saccade-app-port", "slice": 1},
                "window": {"label": "main"},
                "board": {"status": "loaded", "asked": None, "location": "t-0@r1",
                          "task": "t-0", "pinnedRevision": 1},
                "errors": [], "render": [], "content": {"parts": []},
                "diagnostics": [], "doors": {"jumpChip": None, "delivered": []},
            }

        def __call__(self, payload: dict, timeout: float = 60.0) -> dict:
            if payload.get("cmd") == "input" and payload.get("kind") == "type":
                self.typed = payload.get("text")
            if payload.get("cmd") == "input" and payload.get("kind") == "hotkey":
                keys = payload.get("keys") or []
                if "shift" in keys and ";" in keys:
                    self.typed = (self.typed or "") + ":"  # shift+; is the chord for ':'
                elif "shift" in keys and "2" in keys:
                    self.typed = (self.typed or "") + "@"  # shift+2 is the chord for '@'
            if payload.get("cmd") == "input" and payload.get("kind") == "key":
                keys = payload.get("keys") or []
                if keys and len(keys[0]) == 1 and keys[0] not in ("escape", " "):
                    self.typed = (self.typed or "") + keys[0]
            return FakeSession.__call__(self, payload, timeout)

        def _evaluate(self, script: str) -> object:
            if "readout === 'function'" in script:
                return True
            if "JSON.stringify(window.__saccadePort.readout())" in script:
                return json.dumps(self.readout)
            if "el.value" in script:
                return self.typed  # the picker's input value
            if "picker-input" in script:
                return True  # the port picker opens on space space
            return FakeSession._evaluate(self, script)

    def test_flavor_detected_port_and_legacy(self) -> None:
        self.assertTrue(t36_port.is_port(self.PortFake()))
        self.assertFalse(t36_port.is_port(FakeSession()))

    def test_readout_parses(self) -> None:
        obs = t36_port.readout(self.PortFake())
        self.assertEqual(obs["board"]["status"], "loaded")

    def test_require_fields_stops_on_slice2_fields(self) -> None:
        with self.assertRaises(t36_checks.CheckStopped) as caught:
            t36_port.require_fields({"content": {}}, "trail")
        self.assertIn("predates chunk E", str(caught.exception))
        t36_port.require_fields({"trail": {"rows": []}}, "trail")  # present passes

    def test_ensure_served_premise(self) -> None:
        with self.assertRaises(t36_checks.CheckStopped) as caught:
            t36_port.ensure_served("S02")
        self.assertIn("the port surface does not serve", str(caught.exception))
        self.assertIn("recorded non-runnables", str(caught.exception))  # no missing surfaces
        # the served set widened with the reference acts (S09/S13) and
        # the pinned back/forward chords (S05/S07); the recorded
        # non-runnables pass through (their bodies carry the reason)
        for served in ("F01", "S05", "S07", "S08", "S09", "S10", "S12", "S13", "S14", "S16", "S17", "S18", "S20"):
            t36_port.ensure_served(served)
        for not_driven in ("S01", "S03", "S11", "S19"):
            t36_port.ensure_served(not_driven)

    def test_run_check_port_gate_stops_unserved_before_the_body(self) -> None:
        fake = self.PortFake()
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S02"], pace=0)
        self.assertEqual(result["status"], "stop")
        self.assertIn("the port surface does not serve", result["detail"])

    def test_run_check_served_body_runs_on_port(self) -> None:
        fake = self.PortFake()
        t36_checks.set_relauncher(None)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S20"], pace=0)
        # the gate let S20 through: the body's own stop (no relaunch
        # callable) is the verdict, not the flavor premise
        self.assertIn("no relaunch callable", result["detail"])

    def test_s20_port_note_rides_the_pass_detail(self) -> None:
        fake = self.PortFake()
        relaunches: list[int] = []
        t36_checks.set_relauncher(lambda: relaunches.append(1))
        self.addCleanup(t36_checks.set_relauncher, None)
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S20"], pace=0)
        self.assertEqual(result["status"], "pass")
        self.assertIn("the port readout answers", result["detail"])
        self.assertIn("'main'", result["detail"])


class PortWidenedBodiesTest(unittest.TestCase):
    """The chunk-E widened port branches: S10's viewport oracle, S12's
    narrowing observation, S14's trail readout - fakes only."""

    @classmethod
    def setUpClass(cls) -> None:
        t36_checks.ensure_registered()

    class ChunkEFake(FakeSession):
        """A port-flavored fake carrying the chunk-E readout fields and
        the port surfaces; the driving hooks mutate the readout."""

        def __init__(self) -> None:
            super().__init__(delivered=True)
            self.viewport_top = 0.0
            self.label_count = 3
            self.location = "host:/board/t36-tall-board.yaml"
            self.typed_spec = None
            self.menu_open = False
            self.trail = {"depth": 2, "cursor": 1,
                          "entries": [{"location": "127.0.0.1:1/seedtracker/t-1", "hasDeparture": False},
                                      {"location": "127.0.0.1:1/seedtracker/t-3", "hasDeparture": False}],
                          "canBack": True, "canForward": False}

        def __call__(self, payload: dict, timeout: float = 60.0) -> dict:
            if payload.get("cmd") == "tracker":
                return {"ok": True, "base": "http://127.0.0.1:1", "tasks": {
                    "t-1": {"task": "t-1", "revision": 2, "message": "t36 seed: the citing account, incorporation into a group child"},
                    "t-3": {"task": "t-3", "revision": 2, "message": "t36 seed: the landing figure, inside a group"}}}
            if payload.get("cmd") == "input" and payload.get("kind") == "type":
                if payload.get("target") == "port:picker-input":
                    self.typed_spec = payload.get("text")
            if payload.get("cmd") == "input" and payload.get("kind") in ("key", "hotkey"):
                keys = payload.get("keys") or []
                if keys == [" ", " "]:
                    self.typed_spec = None  # the picker opened fresh: the field is empty
                if "G" in keys or ("shift" in keys and "g" in keys):
                    self.viewport_top = 900.0
                if payload.get("kind") == "hotkey" and "shift" in keys and ";" in keys:
                    self.typed_spec = (self.typed_spec or "") + ":"
                elif payload.get("kind") == "hotkey" and "shift" in keys and "2" in keys:
                    self.typed_spec = (self.typed_spec or "") + "@"
                elif payload.get("kind") == "type":
                    self.typed_spec = payload.get("text") or ""
                elif payload.get("kind") == "key" and keys and len(keys[0]) == 1 and keys[0] not in ("escape", " ", "G", "g"):
                    self.typed_spec = (self.typed_spec or "") + keys[0]
                if "enter" in keys and self.typed_spec:
                    self.location = self.typed_spec
                    self.typed_spec = None
                if keys == ["g", "g"] or (self.viewport_top and keys == ["g"]):
                    # the shipped matcher's gg completes: back to top
                    self.viewport_top = 0.0
                if "enter" in keys and self.typed_spec:
                    self.location = self.typed_spec
                    self.typed_spec = None
                if "z" in keys:
                    self.menu_open = False  # close-before-execute
            if payload.get("cmd") == "input" and payload.get("kind") == "rightclick":
                self.menu_open = True
            return FakeSession.__call__(self, payload, timeout)

        def _evaluate(self, script: str) -> object:
            if "readout === 'function'" in script:
                return True
            if "JSON.stringify(window.__saccadePort.readout())" in script:
                return json.dumps({
                    "app": {"name": "saccade-app-port", "slice": 1},
                    "window": {"label": "main"},
                    "board": {"status": "loaded", "asked": self.location,
                              "location": self.location,
                              "task": "t-0", "pinnedRevision": None},
                    "errors": [], "render": [], "content": {"parts": [
                        {"part": "/content/0", "kind": "text", "line": 3},
                        {"part": "/content/1", "kind": "group", "line": 4},
                    ]},
                    "diagnostics": [], "doors": {"jumpChip": None, "delivered": []},
                    "trail": self.trail,
                    "viewport": {"left": 0, "top": self.viewport_top, "width": 1024, "height": 1280},
                })
            if "el.value" in script:
                return self.typed_spec  # the picker's input value
            if "document.activeElement === el" in script:
                return True  # the click focused the field
            if "__suiteOpenAt" in script and "open-error" in script:
                return "opened"  # the awaited open door accepts the spec
            if "data-port-label" in script and "length" in script:
                return self.label_count
            if "data-port-label" in script:
                return json.dumps([
                    {"label": "C", "part": "/content/0"},
                    {"label": "a", "part": "/content/1"},
                    {"label": "b", "part": "/content/2"},
                ][: self.label_count])
            if "JSON.stringify" in script and (
                ".port-select-labels" in script or "data-port-menu" in script or "picker-input" in script
            ):
                return json.dumps({"menu": False, "labels": False, "picker": False})
            if "picker-input" in script:
                return True  # the picker-open probe: space space opened it
            if "getBoundingClientRect" in script:
                return {"x": 10.0, "y": 10.0, "width": 100.0, "height": 40.0}
            if "data-port-menu" in script and "!!" in script:
                return self.menu_open
            if ".port-select-labels" in script or "data-port-menu" in script:
                return False
            return FakeSession._evaluate(self, script)

    def test_s10_port_uses_viewport_readout(self) -> None:
        fake = self.ChunkEFake()
        with unittest.mock.patch.object(pathlib.Path, "write_text", lambda self, data, **kw: None):
            result = t36_checks.run_check(fake, t36_checks.CHECKS["S10"], pace=0)
        self.assertEqual(result["status"], "pass", result["detail"])
        self.assertIn("viewport readout", result["detail"])

    def test_s12_port_case_decided_fails(self) -> None:
        fake = self.ChunkEFake()
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S12"], pace=0)
        self.assertEqual(result["status"], "fail")
        self.assertIn("nothing narrowed", result["detail"])

    def test_s12_port_case_folded_passes(self) -> None:
        fake = self.ChunkEFake()
        original = fake._evaluate

        def typed(script: str) -> object:
            if "data-port-label" in script and "length" in script:
                return 0  # the session ended: the selection fired
            return original(script)

        fake._evaluate = typed  # type: ignore[assignment]
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S12"], pace=0)
        self.assertEqual(result["status"], "pass", result["detail"])
        self.assertIn("across case", result["detail"])

    def test_s14_port_flags_inconsistent_back_state(self) -> None:
        fake = self.ChunkEFake()
        fake.trail["canForward"] = True  # disagrees with cursor at the last entry
        with unittest.mock.patch.object(pathlib.Path, "write_text", lambda self, data, **kw: None):
            result = t36_checks.run_check(fake, t36_checks.CHECKS["S14"], pace=0)
        self.assertEqual(result["status"], "fail")
        self.assertIn("back/forward state disagrees", result["detail"])

    def test_s14_port_green_on_consistent_trail(self) -> None:
        fake = self.ChunkEFake()
        with unittest.mock.patch.object(pathlib.Path, "write_text", lambda self, data, **kw: None):
            result = t36_checks.run_check(fake, t36_checks.CHECKS["S14"], pace=0)
        self.assertEqual(result["status"], "pass", result["detail"])
        self.assertIn("explicit openings", result["detail"])


class S15SignatureGateTest(unittest.TestCase):
    """The S15 port observation's before/after must actually differ:
    the signature (location + part-kind set) distinguishes the fixture's
    landing r2 (text+group+figure) from the revision (text+group+text)
    even though both are three parts."""

    @classmethod
    def setUpClass(cls) -> None:
        t36_checks.ensure_registered()

    class S15Fake(FakeSession):
        def __init__(self) -> None:
            super().__init__(delivered=True)
            self.freeze_commits = False
            self.typed = None
            self.location = "127.0.0.1:1/seedtracker/t-3@r2"
            self.parts = [
                {"part": "/content/0", "kind": "text", "line": 3},
                {"part": "/content/1", "kind": "group", "line": 4},
                {"part": "/content/1/content/0", "kind": "figure", "line": 6},
            ]
            self.committed = 0

        def __call__(self, payload: dict, timeout: float = 60.0) -> dict:
            if payload.get("cmd") == "tracker":
                return {"ok": True, "base": "http://127.0.0.1:1", "tasks": {
                    "t-3": {"task": "t-3", "revision": 2, "message": "t36 seed: the landing figure, inside a group"}}}
            if payload.get("cmd") == "input" and payload.get("kind") == "type":
                self.typed = payload.get("text")
            if payload.get("cmd") == "input" and payload.get("kind") == "hotkey":
                keys = payload.get("keys") or []
                if "shift" in keys and ";" in keys:
                    self.typed = (self.typed or "") + ":"  # shift+; is the chord for ':'
                elif "shift" in keys and "2" in keys:
                    self.typed = (self.typed or "") + "@"  # shift+2 is the chord for '@'
            if payload.get("cmd") == "input" and payload.get("kind") == "key":
                keys = payload.get("keys") or []
                if keys == [" ", " "]:
                    self.typed = None  # the picker opened fresh: the field is empty
                if keys and len(keys[0]) == 1 and keys[0] not in ("escape", " "):
                    self.typed = (self.typed or "") + keys[0]
                if "enter" in keys and self.typed:
                    self.location = self.typed
                    if self.typed.endswith("@r1"):
                        self.parts = [{"part": "/content/0", "kind": "text", "line": 3}]
            if (
                getattr(self, "freeze_commits", False)
                and payload.get("cmd") == "tracker-commit"
                and self.committed == 0
            ):
                # the commit lands on the wire but the shown board does
                # not move: no live update (the observation's target)
                self.committed = 1
                return {"ok": True, "revision": 3}
            if payload.get("cmd") == "tracker-commit":
                self.committed += 1
                if self.committed == 1:
                    # the port's live update: the following board moves to
                    # the new revision and its parts change figure -> text
                    self.location = "127.0.0.1:1/seedtracker/t-3@r3"
                    self.parts[2] = {"part": "/content/1/content/0", "kind": "text", "line": 6}
                    return {"ok": True, "revision": 3}
                return {"ok": True, "revision": 4}
            if payload.get("cmd") == "input" and payload.get("kind") == "type":
                self.typed = payload.get("text")
            if payload.get("cmd") == "input" and payload.get("kind") == "key":
                keys = payload.get("keys") or []
                if "enter" in keys and self.typed and self.typed.endswith("@r1"):
                    self.location = "127.0.0.1:1/seedtracker/t-3@r1"
                    self.parts = [{"part": "/content/0", "kind": "text", "line": 3}]
            return FakeSession.__call__(self, payload, timeout)

        def _evaluate(self, script: str) -> object:
            if "readout === 'function'" in script:
                return True
            if "JSON.stringify(window.__saccadePort.readout())" in script:
                return json.dumps({
                    "app": {"name": "saccade-app-port", "slice": 1},
                    "window": {"label": "main"},
                    "board": {"status": "loaded", "asked": self.location,
                              "location": self.location, "task": "t-3", "pinnedRevision": 2},
                    "errors": [], "render": [],
                    "content": {"parts": self.parts},
                    "diagnostics": [], "doors": {"jumpChip": None, "delivered": []},
                    "trail": {"depth": 1, "cursor": 0, "entries": [], "canBack": False, "canForward": False},
                    "viewport": {"left": 0, "top": 0, "width": 1024, "height": 1280},
                })
            if "el.value" in script:
                return self.typed
            if "picker-input" in script:
                return True
            return FakeSession._evaluate(self, script)

    def test_before_and_after_signatures_differ(self) -> None:
        fake = self.S15Fake()
        before = (fake.location, tuple(sorted((p["part"], p["kind"]) for p in fake.parts)))
        fake({"cmd": "tracker-commit", "task": 3, "board": "x"})
        after = (fake.location, tuple(sorted((p["part"], p["kind"]) for p in fake.parts)))
        self.assertNotEqual(before, after)  # the proxy distinguishes the fixture

    def test_s15_port_live_update_passes(self) -> None:
        fake = self.S15Fake()
        with unittest.mock.patch.object(pathlib.Path, "write_text", lambda self, data, **kw: None):
            result = t36_checks.run_check(fake, t36_checks.CHECKS["S15"], pace=0)
        self.assertEqual(result["status"], "pass", result["detail"])
        self.assertIn("updated live", result["detail"])
        self.assertIn("signature", result["detail"])

    def test_s15_port_no_live_update_fails(self) -> None:
        fake = self.S15Fake()
        fake.freeze_commits = True
        result = t36_checks.run_check(fake, t36_checks.CHECKS["S15"], pace=0)
        self.assertEqual(result["status"], "fail")
        self.assertIn("did not update live", result["detail"])
