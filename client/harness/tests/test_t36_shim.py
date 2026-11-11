#!/usr/bin/env python3
"""R4 unit/mechanics tests: the prototype shim against the ACTUAL
prototype, the serving rewrite, and the artifact refusals.

Static tests always run. The end-to-end class loads the real compiled
round-16 prototype (materialized from t-22@r3 through the real sac
binary, its CDN scripts vendored once) into Playwright's WebKit and
exercises the shim plus one real input round trip; it skips cleanly
when WebKit, sac or the network fetch is unavailable."""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import unittest
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_prototype  # noqa: E402

RAW_REFERENCE_DIR = pathlib.Path("/tmp/saccade-board-t-34.r6/reference")


def webkit_available() -> bool:
    try:
        import playwright.sync_api

        playwright.sync_api.sync_playwright  # referenced so the probe import is used
        return True
    except ImportError:
        return False


class ShimSourceTest(unittest.TestCase):
    """The shim's own contract, checked without a browser."""

    def test_shim_is_ascii_and_an_iife(self) -> None:
        source = t36_prototype.shim_expression()
        source.encode("ascii")
        self.assertTrue(source.startswith("/*"))
        self.assertIn("(function () {", source)
        self.assertTrue(source.rstrip().endswith("})()"))

    def test_shim_reads_the_prototype_surfaces(self) -> None:
        source = t36_prototype.shim_expression()
        for surface in ("SUITE", "frames", "SESSIONS", "data-kind", "data-loc", "snapshot", ".picker.open input", "contentDocument"):
            self.assertIn(surface, source, f"the shim must read {surface}")

    def test_shim_writes_only_its_answer_holder(self) -> None:
        import re

        source = t36_prototype.shim_expression()
        # every assignment statement in the shim targets its own locals
        # (`out`, `readout`, `var` declarations); no prototype surface
        # is written and nothing is called that would act
        allowed = {"out", "readout", "var", "detached", "word", "classes", "attached"}
        for line in source.splitlines():
            match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.$\[\]\"']*)\s*=(?![=>])", line)
            if match is None:
                continue
            head = match.group(1).split(".")[0].split("[")[0]
            self.assertIn(head, allowed, f"the shim must write only its own locals, saw {match.group(1)!r}")
        for forbidden in ("innerHTML", "dispatchEvent", "addEventListener", ".focus()", ".click()"):
            self.assertNotIn(forbidden, source, f"the shim must never call {forbidden}")

    def test_shim_syntax_is_valid_javascript(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node unavailable")
        import subprocess

        result = subprocess.run([node, "--check", str(HERE.parent / "t36_shim.js")], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace")[:400])

    def test_raw_templates_carry_what_the_shim_reads(self) -> None:
        if not RAW_REFERENCE_DIR.is_dir():
            self.skipTest(f"{RAW_REFERENCE_DIR} not present")
        inst = (RAW_REFERENCE_DIR / "inst16.html").read_text(encoding="utf-8")
        shell = (RAW_REFERENCE_DIR / "shell16.html").read_text(encoding="utf-8")
        for attribute in ('data-kind="task"', 'data-kind="session"', "data-loc", "data-state", "data-chg"):
            self.assertIn(attribute, inst, f"inst16 must render {attribute}")
        for surface in ('window.snapshot = snapshot', "const SESSIONS", "const SUITE = {", "window.receive ="):
            self.assertIn(surface, shell + inst, f"the prototype must expose {surface}")


class ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_template_bytes_refused(self) -> None:
        # all compiled markers plus a template placeholder: the shape of
        # a build input mistakenly handed to the lane
        template = (
            "<title>Saccade Mockups R16</title>"
            '<script id="inst" type="application/json">/*INST*/</script>'
            "<script>function snapshot(){}</script>"
        ).encode("utf-8")
        with self.assertRaises(t36_prototype.PrototypeRefused) as caught:
            t36_prototype.verify_page(template)
        self.assertIn("unbuilt template", str(caught.exception))

    def test_wrong_page_refused(self) -> None:
        with self.assertRaises(t36_prototype.PrototypeRefused):
            t36_prototype.verify_page(b"<title>other</title> nothing here")

    def test_compiled_markers_present_in_fetched_artifact(self) -> None:
        if shutil.which("sac") is None:
            self.skipTest("sac unavailable")
        try:
            record = t36_prototype.materialize(pathlib.Path(self.tmp.name) / "cache")
        except t36_prototype.PrototypeRefused as error:
            self.skipTest(f"materialize refused: {error}")
        self.assertEqual(record["source"], t36_prototype.PROTOTYPE_BOARD_REF)
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
        page = pathlib.Path(record["path"]).read_text(encoding="utf-8")
        # the artifact on disk keeps its CDN URLs; the rewrite is the
        # server's, never the file's
        for lib in t36_prototype.CDN_LIBS:
            self.assertIn(lib["url"], page)
        self.assertNotIn("/*DATA*/", page)


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for lib in t36_prototype.CDN_LIBS:
            (pathlib.Path(self.tmp.name) / lib["name"]).write_bytes(b"// vendored stand-in " + lib["name"].encode("ascii"))
        self.libs = t36_prototype.vendor_libs(pathlib.Path(self.tmp.name))
        self.server = t36_prototype.PrototypeServer(
            b"<html><title>Saccade Mockups R16</title><script src='https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js'></script></html>",
            {**self.libs, "js-yaml.min.js": self.libs["js-yaml.min.js"]},
        )
        self.addCleanup(self.server.stop)

    def test_page_served_with_urls_rewritten(self) -> None:
        with urllib.request.urlopen(self.server.base_url + "/prototype.html", timeout=10) as reply:
            body = reply.read().decode("utf-8")
        self.assertIn("src='/vendor/js-yaml.min.js'", body.replace('"', "'"))
        self.assertNotIn("cdn.jsdelivr.net", body)

    def test_vendor_paths_served(self) -> None:
        with urllib.request.urlopen(self.server.base_url + "/vendor/js-yaml.min.js", timeout=10) as reply:
            body = reply.read()
        self.assertTrue(body.startswith(b"// vendored stand-in"))

    def test_unknown_path_404(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.server.base_url + "/nope", timeout=10)
        self.assertEqual(caught.exception.code, 404)


@unittest.skipUnless(webkit_available(), "requires Playwright WebKit")
class PrototypeReadoutTest(unittest.TestCase):
    """The shim readout against the ACTUAL prototype, plus one real
    input round trip. One session for the class; the prototype's own
    data (12 tasks, 6 sessions, the board at t-3) is the oracle."""

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("sac") is None:
            raise unittest.SkipTest("sac unavailable")
        cls.tmp = tempfile.TemporaryDirectory()
        cache = pathlib.Path(cls.tmp.name) / "cache"
        try:
            record = t36_prototype.materialize(cache)
            libs = t36_prototype.vendor_libs(cache)
        except t36_prototype.PrototypeRefused as error:
            cls.tmp.cleanup()
            raise unittest.SkipTest(f"prototype materialization refused: {error}")
        cls.server = t36_prototype.PrototypeServer(pathlib.Path(record["path"]).read_bytes(), libs)
        cls.record = record
        cls.session = t36_prototype.PrototypeServer and t36_prototype.PrototypeSession(cls.server)
        try:
            cls.session.start()
        except t36_prototype.PrototypeRefused:
            cls.session.stop()
            cls.server.stop()
            cls.tmp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.stop()
        cls.server.stop()
        cls.tmp.cleanup()

    @classmethod
    def readout(cls) -> dict:
        return cls.session.readout()

    def test_booted_and_complete(self) -> None:
        readout = self.readout()
        self.assertTrue(readout["booted"])
        self.assertEqual(readout["missing"], [])

    def test_window_inventory_mirrors_the_suite(self) -> None:
        readout = self.readout()
        ids = {w["id"] for w in readout["windows"]}
        self.assertLessEqual({"tasks", "agents", "board", "deck", "specimen", "readme"}, ids)
        by_id = {w["id"]: w for w in readout["windows"]}
        self.assertEqual(by_id["board"]["inst"], "viewer")
        self.assertTrue(by_id["board"]["attached"])
        self.assertTrue(by_id["board"]["focused"])
        # deck holds deck state in the pool but is not attached on desk
        self.assertFalse(by_id["deck"]["attached"])

    def test_rows_come_from_the_real_data(self) -> None:
        rows = self.readout()["rows"]
        task_rows = [r for r in rows if r["kind"] == "task"]
        session_rows = [r for r in rows if r["kind"] == "session"]
        self.assertEqual(len(task_rows), 12)
        self.assertEqual(len(session_rows), 6)
        board_rows = [r for r in rows if r["kind"] in ("block", "ref", "group", "incorp", "figure")]
        self.assertGreater(len(board_rows), 20)
        some_ref = next(r for r in rows if r["kind"] == "ref")
        self.assertTrue((some_ref.get("loc") or "").startswith("t-"))

    def test_words_mirror_sessions(self) -> None:
        words = {w["id"]: w for w in self.readout()["words"]}
        self.assertEqual(words["i-3"]["word"], "working")
        self.assertIn("t-3", words["i-3"]["claims"])
        self.assertEqual(words["i-4"]["word"], "done")

    def test_trails_and_deck_from_the_state_pool(self) -> None:
        readout = self.readout()
        board_trail = readout["trails"]["board"]
        self.assertTrue(board_trail["location"].startswith("t-3"))
        self.assertEqual(board_trail["back"], 0)
        deck_window = next(w for w in readout["windows"] if w["id"] == "deck")
        self.assertEqual(deck_window["deck"], {"on": True, "slides": 1, "position": 0, "waiting": 0})

    def test_strip_reads_the_focused_window(self) -> None:
        strip = self.readout()["strip"]
        self.assertEqual(strip["name"], "viewer")
        self.assertTrue((strip["location"] or "").startswith("saccade/t-3"))

    def test_input_round_trip_opens_and_closes_the_picker(self) -> None:
        self.session.input("Escape")
        self.session.input(" ", " ")
        readout = self.readout()
        self.assertTrue(readout["picker"]["open"])
        board = next(w for w in readout["windows"] if w["id"] == "board")
        self.assertTrue(board["picker"]["open"])
        self.session.input("Escape")
        self.assertFalse(self.readout()["picker"]["open"])

    def test_readout_is_json_serializable(self) -> None:
        json.dumps(self.readout())


if __name__ == "__main__":
    unittest.main()


class PrototypeSessionStopTest(unittest.TestCase):
    """The bg-33 bounding, revised per review: the close stays on the
    OWNING thread (the sync API is thread-affine - a cross-thread close
    throws 'Cannot switch to a different thread', verified live) and a
    watchdog kills the driver's process subtree at the deadline. The
    process-backed cases (healthy close exits the subtree; a wedged
    driver returns at the bound with everything killed) are verified
    LIVE - this suite covers the pure-python parts."""

    def test_stop_closes_a_live_browser_quickly(self) -> None:
        from t36_prototype import PrototypeSession

        session = PrototypeSession.__new__(PrototypeSession)
        session._driver_handles = []
        closed: list[str] = []

        class Browser:
            def close(self) -> None:
                closed.append("browser")

        class Playwright:
            def stop(self) -> None:
                closed.append("playwright")

        session._browser = Browser()
        session._playwright = Playwright()
        session.page = object()
        session.stop()
        self.assertEqual(sorted(closed), ["browser", "playwright"])
        self.assertIsNone(session.page)

    def test_stop_drops_references_before_closing(self) -> None:
        # the r74 race: the close must run against captured handles, not
        # the session's fields (the caller drops those at the top)
        from t36_prototype import PrototypeSession

        session = PrototypeSession.__new__(PrototypeSession)
        session._driver_handles = []
        seen: list[object] = []

        class Browser:
            def close(self) -> None:
                seen.append(session._browser)  # must already be None

        session._browser = Browser()
        session._playwright = None
        session.page = None
        session.stop()
        self.assertEqual(seen, [None])

    def test_stop_without_handles_returns_immediately(self) -> None:
        from t36_prototype import PrototypeSession

        session = PrototypeSession.__new__(PrototypeSession)
        session._browser = None
        session._playwright = None
        session._driver_handles = []
        session.page = object()
        session.stop()
        self.assertIsNone(session.page)


class ScopedTeardownTest(unittest.TestCase):
    """The watchdog is rooted at the captured Playwright driver handle:
    a wedged driver dies with its browsers at the bound while a
    non-Playwright sibling child of the orchestrator survives (the
    r73-bounded-stop review's scope ruling). Live, process-backed."""

    def test_watchdog_kills_driver_subtree_and_spares_sibling(self) -> None:
        import os
        import subprocess
        import sys
        import time

        sys.path.insert(0, "client/harness")
        import t36_prototype
        from t36_prototype import PrototypeSession, _children_of

        for key, value in t36_prototype.webkit_launch_env().items():
            os.environ.setdefault(key, value)

        # the sentinel: an unrelated child of this process that must
        # survive the teardown
        sentinel = subprocess.Popen(
            ["/proc/self/exe", "-c", "import time; time.sleep(60)"]
            if False
            else [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        try:
            self.assertIn(sentinel.pid, _children_of(os.getpid()))

            from playwright.sync_api import sync_playwright

            children_before = set(_children_of(os.getpid()))
            pw = sync_playwright().start()
            spawned = set(_children_of(os.getpid())) - children_before
            self.assertTrue(spawned, "the playwright driver spawn was not captured")
            browser = pw.webkit.launch(headless=True)
            page = browser.new_page()
            page.set_content("<h1>scoped</h1>")

            session = PrototypeSession.__new__(PrototypeSession)
            session._browser = browser
            session._playwright = pw
            session._driver_handles = [(pid, t36_prototype._starttime(pid)) for pid in sorted(spawned)]
            session.page = page
            session.TEARDOWN_TIMEOUT_SECONDS = 2.0

            # wedged driver: the close cannot complete on its own
            for pid in sorted(spawned):
                os.kill(pid, 19)  # SIGSTOP

            started = time.monotonic()
            session.stop()
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 8.0, f"the teardown overran its bound: {elapsed:.1f}s")
            time.sleep(0.5)
            driver_pids_alive = [pid for pid in spawned if os.path.exists(f"/proc/{pid}")]
            self.assertEqual(driver_pids_alive, [], f"driver pids leaked: {driver_pids_alive}")
            self.assertTrue(os.path.exists(f"/proc/{sentinel.pid}"),
                            "the sentinel was killed - the watchdog exceeded its scope")
        finally:
            sentinel.kill()
            sentinel.wait()


class WatchdogIdentityTest(unittest.TestCase):
    """Fail-closed ownership: a missing launch identity (a None start
    time) never kills - the sentinel survives even a wedged driver."""

    def test_missing_launch_identity_never_kills(self) -> None:
        import subprocess
        import sys

        sys.path.insert(0, "client/harness")
        from t36_prototype import PrototypeSession

        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        wedged = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            session = PrototypeSession.__new__(PrototypeSession)

            class HangingClose:
                def close(self) -> None:
                    wedged.wait(timeout=3)  # wedged cleanup; nothing may kill it

            session._browser = HangingClose()
            session._playwright = None
            # the launch identity is MISSING (None): not ownership proof
            session._driver_handles = [(wedged.pid, None)]
            session.page = object()
            session.TEARDOWN_TIMEOUT_SECONDS = 0.5
            session.stop()
            # the fail-closed rule: no launch identity, no kill - the
            # wedged process survives (the close ends on its own fake
            # timeout; the bound protects only proven ownership)
            self.assertIsNone(wedged.poll(),
                              "a driver with no launch identity was killed - ownership must be proven")
            self.assertIsNone(sentinel.poll())
        finally:
            wedged.kill()
            sentinel.kill()
            wedged.wait()
            sentinel.wait()

    def test_descendant_identity_rechecked_before_kill(self) -> None:
        # the subtree walk collects the root's real descendants (the
        # victim's own child - a grandchild of this process), captures
        # each start time at discovery, and re-checks before the kill
        import os
        import subprocess
        import sys
        import time

        sys.path.insert(0, "client/harness")
        from t36_prototype import _children_of, _kill_subtree, _starttime

        victim = subprocess.Popen([
            sys.executable, "-c",
            "import subprocess, sys, time;\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n",
        ], stdout=subprocess.PIPE)
        try:
            grandchild_pid = int(victim.stdout.readline().strip())
            deadline = time.monotonic() + 3.0
            while grandchild_pid not in _children_of(victim.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIn(grandchild_pid, _children_of(victim.pid))

            _kill_subtree(victim.pid, _starttime(victim.pid))
            poll_deadline = time.monotonic() + 3.0
            while victim.poll() is None and time.monotonic() < poll_deadline:
                time.sleep(0.05)
            self.assertIsNotNone(victim.poll())  # killed, reaped by poll
            self.assertFalse(os.path.exists(f"/proc/{grandchild_pid}"))  # killed, re-parented and reaped

            # a None identity on the root kills nothing
            spared = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            try:
                _kill_subtree(spared.pid, None)
                self.assertIsNone(spared.poll())
            finally:
                spared.kill()
                spared.wait()
        finally:
            victim.kill()
            victim.wait()
