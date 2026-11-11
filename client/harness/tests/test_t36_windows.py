#!/usr/bin/env python3
"""R5 unit tests: the multi-window surface - payload verbs driven
through handle_request with a fake WebDriver, and the host-side
request builders. No app, no compositor, no driver chain."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_observation  # noqa: E402
import t36_windows  # noqa: E402
import webdriver_session  # noqa: E402
from webdriver_client import WebDriverError  # noqa: E402


class FakeWebDriver:
    """Duck-typed WebDriver: handles stand for webviews; each answers
    the label script with its own label, exactly as the runtime
    metadata would."""

    def __init__(self, labels: list[str]) -> None:
        self.handles = [f"h-{index}" for index in range(len(labels))]
        self.labels = dict(zip(self.handles, labels))
        self.current = self.handles[0]
        self.executions: list[tuple[str, str]] = []
        self.fresh_counter = 0

    def current_window(self) -> str:
        return self.current

    def window_handles(self) -> list[str]:
        return list(self.handles)

    def switch_to_window(self, handle: str) -> None:
        assert handle in self.handles, handle
        self.current = handle

    def execute(self, script: str, args: list | None = None) -> object:
        self.executions.append((self.current, script))
        if script == webdriver_session.WINDOW_LABEL_SCRIPT:
            return self.labels[self.current]
        if script == webdriver_session.SUITE_WINDOWS_SCRIPT:
            return [{"label": "main", "state_id": None}] + [
                {"label": label, "state_id": "board"} for label in list(self.labels.values())[1:]
            ]
        if "open_suite_window" in script:
            self.fresh_counter += 1
            label = f"w-fresh-{self.fresh_counter}"
            handle = f"h-fresh-{self.fresh_counter}"
            self.handles.append(handle)
            self.labels[handle] = label
            return label
        if "focus_suite_window" in script:
            assert isinstance(args, list) and isinstance(args[0], str)
            self.focused_label = args[0]
            return None
        if script == webdriver_session.OBSERVE_SCRIPT:
            return {"tag": "div", "text": "observed-in-" + self.labels[self.current]}
        return {"script": script, "args": args, "window": self.labels[self.current]}


def make_state(labels: list[str]) -> tuple[webdriver_session.SessionState, FakeWebDriver]:
    state = webdriver_session.SessionState()
    client = FakeWebDriver(labels)
    state.client = client  # type: ignore[assignment]
    return state, client


class WindowsVerbTest(unittest.TestCase):
    def test_enumeration_pairs_handles_with_labels_and_restores(self) -> None:
        state, client = make_state(["main", "w-a", "w-b"])
        response, _stop = webdriver_session.handle_request(state, {"cmd": "windows"})
        self.assertTrue(response["ok"])
        self.assertEqual([w["label"] for w in response["windows"]], ["main", "w-a", "w-b"])
        self.assertEqual([w["handle"] for w in response["windows"]], ["h-0", "h-1", "h-2"])
        self.assertEqual(client.current, "h-0")
        self.assertEqual(response["suite"][0], {"label": "main", "state_id": None})

    def test_open_window_returns_label_and_refreshed_inventory(self) -> None:
        state, _client = make_state(["main"])
        response, _stop = webdriver_session.handle_request(state, {"cmd": "open-window", "state_id": "board"})
        self.assertTrue(response["ok"])
        self.assertEqual(response["label"], "w-fresh-1")
        self.assertEqual(len(response["windows"]), 2)
        self.assertIn("w-fresh-1", [w["label"] for w in response["windows"]])

    def test_open_window_without_state_id_is_allowed(self) -> None:
        state, _client = make_state(["main"])
        response, _stop = webdriver_session.handle_request(state, {"cmd": "open-window"})
        self.assertTrue(response["ok"])

    def test_open_window_refuses_non_string_state_id(self) -> None:
        state, _client = make_state(["main"])
        with self.assertRaises(WebDriverError):
            webdriver_session.handle_request(state, {"cmd": "open-window", "state_id": 7})

    def test_focus_window_asks_the_shell_door(self) -> None:
        state, client = make_state(["main"])
        response, _stop = webdriver_session.handle_request(state, {"cmd": "focus-window", "label": "w-a"})
        self.assertTrue(response["ok"])
        self.assertEqual(response["focused"], "w-a")
        self.assertEqual(client.focused_label, "w-a")

    def test_focus_window_refuses_empty_label(self) -> None:
        state, _client = make_state(["main"])
        with self.assertRaises(WebDriverError):
            webdriver_session.handle_request(state, {"cmd": "focus-window", "label": ""})

    def test_observe_by_window_reads_the_right_webview(self) -> None:
        state, client = make_state(["main", "w-a", "w-b"])
        response, _stop = webdriver_session.handle_request(
            state, {"cmd": "observe-window", "window": "w-b", "target": "board:location"}
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["observed"]["text"], "observed-in-w-b")
        self.assertEqual(client.current, "h-0")

    def test_observe_by_window_restores_handle_even_on_error(self) -> None:
        state, client = make_state(["main", "w-a"])
        bad_state = webdriver_session.SessionState()
        bad_state.client = client
        # a target id that fails the selector validation raises before
        # any switch; the more interesting path is an error mid-read,
        # simulated by a client whose execute raises after switching
        class ExplodingClient(FakeWebDriver):
            def execute(self, script: str, args: list | None = None) -> object:
                if script == webdriver_session.OBSERVE_SCRIPT:
                    raise WebDriverError("javascript error", "boom")
                return super().execute(script, args)

        exploding = ExplodingClient(["main", "w-a"])
        bad_state.client = exploding  # type: ignore[assignment]
        with self.assertRaises(WebDriverError):
            webdriver_session.handle_request(bad_state, {"cmd": "observe-window", "window": "w-a", "target": "strip"})
        self.assertEqual(exploding.current, "h-0")

    def test_observe_by_window_unknown_label_records_inventory(self) -> None:
        state, _client = make_state(["main"])
        with self.assertRaises(WebDriverError) as caught:
            webdriver_session.handle_request(state, {"cmd": "observe-window", "window": "w-ghost", "target": "strip"})
        self.assertIn("w-ghost", str(caught.exception))

    def test_evaluate_window_runs_the_readout_in_that_window(self) -> None:
        state, client = make_state(["main", "w-a"])
        response, _stop = webdriver_session.handle_request(
            state, {"cmd": "evaluate-window", "window": "w-a", "script": t36_observation.READOUT_JS}
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["window"], "w-a")
        # the fake answers object-valued scripts with the window name
        self.assertEqual(response["value"]["window"], "w-a")
        self.assertEqual(client.current, "h-0")

    def test_evaluate_window_refuses_empty_script(self) -> None:
        state, _client = make_state(["main"])
        with self.assertRaises(WebDriverError):
            webdriver_session.handle_request(state, {"cmd": "evaluate-window", "window": "main", "script": "  "})


class HostBuildersTest(unittest.TestCase):
    def test_requests_match_the_payload_surface(self) -> None:
        self.assertEqual(t36_windows.windows_request(), {"cmd": "windows"})
        self.assertEqual(t36_windows.open_window_request(), {"cmd": "open-window", "state_id": None})
        self.assertEqual(t36_windows.open_window_request("board"), {"cmd": "open-window", "state_id": "board"})
        self.assertEqual(t36_windows.focus_window_request("w-1"), {"cmd": "focus-window", "label": "w-1"})
        self.assertEqual(
            t36_windows.observe_window_request("w-1", "strip", ["color"]),
            {"cmd": "observe-window", "window": "w-1", "target": "strip", "props": ["color"]},
        )
        request = t36_windows.readout_window_request("w-1")
        self.assertEqual(request["cmd"], "evaluate-window")
        self.assertEqual(request["window"], "w-1")
        self.assertEqual(request["script"], t36_observation.READOUT_JS)

    def test_window_readout_consumes_the_payload_answer(self) -> None:
        def socket_request(payload: dict, timeout: float = 60.0) -> dict:
            self.assertEqual(payload["cmd"], "evaluate-window")
            return {"ok": True, "window": payload["window"], "value": {"booted": True}}

        readout = t36_windows.window_readout(socket_request, "w-1")
        self.assertTrue(readout["booted"])

    def test_window_readout_stops_on_refusal(self) -> None:
        def socket_request(payload: dict, timeout: float = 60.0) -> dict:
            return {"ok": False, "error": "no such window: w-ghost"}

        with self.assertRaises(t36_observation.CheckStopped) as caught:
            t36_windows.window_readout(socket_request, "w-ghost")
        self.assertIn("no such window", caught.exception.reason)

    def test_wait_for_window_count_polls_and_returns(self) -> None:
        calls = {"n": 0}

        def socket_request(payload: dict, timeout: float = 60.0) -> dict:
            calls["n"] += 1
            windows = [{"handle": f"h-{i}", "label": f"w-{i}"} for i in range(min(calls["n"], 3))]
            return {"ok": True, "windows": windows}

        inventory = t36_windows.wait_for_window_count(socket_request, 3, timeout=5)
        self.assertEqual(len(inventory), 3)

    def test_wait_for_window_count_stops_at_bound(self) -> None:
        def socket_request(payload: dict, timeout: float = 60.0) -> dict:
            return {"ok": True, "windows": [{"handle": "h-0", "label": "main"}]}

        with self.assertRaises(t36_observation.CheckStopped) as caught:
            t36_windows.wait_for_window_count(socket_request, 2, timeout=0.5)
        self.assertIn("windows enumerated", caught.exception.reason)

    def test_inventory_persists_for_the_run_root(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = t36_windows.write_inventory(pathlib.Path(tmp.name), [{"handle": "h-0", "label": "main"}], [{"label": "main"}])
        text = path.read_text(encoding="utf-8")
        self.assertIn('"webdriver"', text)
        self.assertIn('"suite_windows"', text)


class VerbParsingTest(unittest.TestCase):
    """The host verb surface: parse and shape only (dispatch needs a
    live session socket, which no unit test fakes)."""

    def test_new_verbs_parse_with_field_args(self) -> None:
        verb, options, _lifetime = __import__("entry").parse_session_verb(["windows"])
        self.assertEqual(verb, "windows")
        verb, options, _lifetime = __import__("entry").parse_session_verb(["open-window", "state=board"])
        self.assertEqual((verb, options["args"]), ("open-window", ["state=board"]))
        verb, options, _lifetime = __import__("entry").parse_session_verb(["observe-window", "window=w-1", "target=strip"])
        self.assertEqual(verb, "observe-window")

    def test_positional_args_refused_for_new_verbs(self) -> None:
        import entry

        # the verb parses; the bare token is not a field, so the label
        # is missing and session_main's field check refuses (exit 2)
        verb, options, _lifetime = entry.parse_session_verb(["focus-window", "w-1"])
        self.assertEqual(verb, "focus-window")
        fields, _bare = entry.parse_verb_fields(options["args"], {"label"})
        self.assertNotIn("label", fields)


if __name__ == "__main__":
    unittest.main()
