#!/usr/bin/env python3
"""R3 unit tests: the app-side readout script and the STOP rule.

The readout script is exercised for real against a WebKit page that
mimics the app's existing surfaces (the __saccadeApp projection shape,
the strip's stable data-control attributes, both instrument-row
attribute shapes, a __TAURI_INTERNALS__ stand-in). The fixture is
harness-owned; nothing from client/src is loaded or edited. Skipped
cleanly when the pinned Playwright WebKit is unavailable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_observation  # noqa: E402
import t36_results  # noqa: E402


def webkit_available() -> tuple[bool, str]:
    try:
        import playwright.sync_api

        playwright.sync_api.sync_playwright  # referenced so the probe import is used
        return True, ""
    except ImportError:
        return False, "playwright unavailable"
    if not pathlib.Path("/nix/store").is_dir():
        return False, "not a nix store machine"
    return True, ""


FIXTURE = """
<div class="app">
  <header class="strip" data-control="strip">
    <span class="loc" data-control="board:location">tracker:lanetracker/t-7@r3</span>
    <span class="chip warn" data-control="strip:state">Herdr unreachable - no live states</span>
    <span class="chip on" data-control="deck:line">deck: review</span>
    <span class="chip" data-control="deck:state">slide 2/5</span>
    <span class="lnk" data-control="link:target">-&gt; tasks</span>
  </header>
  <div role="table" data-control="instrument:section">
    <div data-control="instrument:row" data-instrument="tasks" data-task="t-7" data-open-path="tracker:x/t-7">t-7 the citing account</div>
    <div data-control="instrument:row" data-instrument="agents" data-tasks="t-9 t-4 t-6" data-incarnation="i-4"
         data-address="t66" data-word="done" data-app="pi" data-msg="test-inventory" data-progress=""
         data-title="" data-at="11:31" data-workspace="wt" data-project="saccade">i-4 done</div>
  </div>
  <aside data-control="history:margin">
    <div data-control="history:entry">r3 the revision shown</div>
    <div data-control="history:entry">r2 an earlier one</div>
  </aside>
</div>
<script>
window.__saccadeApp = {
  inspection: () => ({
    resource: {machine: "tracker:lanetracker", path: "t-7@r3", status: "loaded"},
    identity: {task: "t-7", grammar: 1, supportedGrammarVersions: [1], rawLength: 42},
    errors: [], diagnostics: [], renderOutcomes: [{kind: "figure", status: "ok"}],
    copyMode: "plain", theme: "dark", targets: [{id: "diagram:x", role: "figure", label: "x", enabled: true, bounds: {x: 0, y: 0, width: 1, height: 1}}],
    outline: [], laneMessage: null, renderGeneration: 2,
  }),
  deck: () => ({name: "review", slides: ["a", "b", "c", "d", "e"], position: 1, entered: true, feedLost: false}),
};
window.__TAURI_INTERNALS__ = {
  metadata: {currentWindow: {label: "main"}},
  invoke: (cmd) => (cmd === "suite_windows"
    ? Promise.resolve([{label: "main", state_id: null}, {label: "w-17abc0", state_id: "board"}])
    : Promise.reject(new Error("no such command"))),
};
</script>
"""


def run_readout(page) -> dict:
    """The readout body as a function expression; a returned promise is
    awaited by the driver, exactly as WebDriver execute/sync does."""
    script = "(function() {" + t36_observation.READOUT_JS + "})()"
    return page.evaluate(script)


@unittest.skipUnless(webkit_available()[0], f"requires Playwright WebKit: {webkit_available()[1]}")
class ReadoutScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from playwright.sync_api import sync_playwright

        cls._playwright = sync_playwright().start()
        cls._browser = cls._playwright.webkit.launch(headless=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._browser.close()
        cls._playwright.stop()

    def setUp(self) -> None:
        self.page = self._browser.new_page()
        self.page.set_content(FIXTURE)

    def tearDown(self) -> None:
        self.page.close()

    def readout(self) -> dict:
        return run_readout(self.page)

    def test_booted_with_full_projection(self) -> None:
        readout = self.readout()
        self.assertTrue(readout["booted"])
        self.assertEqual(readout["missing"], [])
        self.assertEqual(readout["inspection"]["resource"]["status"], "loaded")
        self.assertEqual(readout["inspection"]["identity"]["task"], "t-7")
        self.assertEqual(readout["deck"]["position"], 1)

    def test_attribute_split_absorbed(self) -> None:
        rows = self.readout()["rows"]
        self.assertEqual(rows[0]["instrument"], "tasks")
        self.assertEqual(rows[0]["tasks"], ["t-7"])
        self.assertEqual(rows[1]["instrument"], "agents")
        self.assertEqual(rows[1]["tasks"], ["t-9", "t-4", "t-6"])

    def test_word_attributes_ride_rows(self) -> None:
        row = self.readout()["rows"][1]
        self.assertEqual(row["word"], "done")
        self.assertEqual(row["app"], "pi")
        self.assertEqual(row["msg"], "test-inventory")

    def test_strip_fields_read_by_control_id(self) -> None:
        strip = self.readout()["strip"]
        self.assertEqual(strip["location"], "tracker:lanetracker/t-7@r3")
        self.assertIn("Herdr unreachable", strip["state"] or "")
        self.assertEqual(strip["deckLine"], "deck: review")
        self.assertEqual(strip["deckState"], "slide 2/5")

    def test_history_and_picker(self) -> None:
        readout = self.readout()
        self.assertEqual(len(readout["historyEntries"]), 2)
        self.assertFalse(readout["picker"]["open"])

    def test_window_label_and_suite_windows(self) -> None:
        readout = self.readout()
        self.assertEqual(readout["window"]["label"], "main")
        self.assertEqual([w["label"] for w in readout["windows"]], ["main", "w-17abc0"])

    def test_unbooted_page_reports_missing_not_crash(self) -> None:
        page = self._browser.new_page()
        try:
            page.set_content("<p>not the app</p>")
            readout = run_readout(page)
        finally:
            page.close()
        self.assertFalse(readout["booted"])
        self.assertIn("__saccadeApp.inspection", readout["missing"])

    def test_readout_is_json_serializable(self) -> None:
        json.dumps(self.readout())


class StopRuleTest(unittest.TestCase):
    def test_missing_fields_detected_dotted(self) -> None:
        readout = {"strip": {"location": "x"}, "windows": None, "rows": []}
        gaps = t36_observation.missing_fields(readout, ["strip.location", "windows", "rows", "deck.position"])
        self.assertEqual(gaps, ["windows", "deck.position"])

    def test_require_fields_raises_stop_with_reason(self) -> None:
        readout = {"strip": {"location": None}}
        with self.assertRaises(t36_observation.CheckStopped) as caught:
            t36_observation.require_fields(readout, ["strip.location"], "S14", "session", "flags")
        self.assertIn("strip.location", caught.exception.reason)
        self.assertIn("stops", caught.exception.reason)

    def test_stop_result_record(self) -> None:
        result = t36_observation.stop_result("S14", "session", "flags", "no trail readout exists")
        self.assertEqual(result["status"], "stop")
        self.assertEqual(result["detail"], "no trail readout exists")
        self.assertEqual(t36_results.tap_exit_code(t36_results.emit_tap([result])), 0)

    def test_present_fields_never_stop(self) -> None:
        readout = {"strip": {"location": "t-3"}, "rows": [{"tasks": ["t-3"]}]}
        t36_observation.require_fields(readout, ["strip.location", "rows"], "S08", "session", "reveal")


if __name__ == "__main__":
    unittest.main()
