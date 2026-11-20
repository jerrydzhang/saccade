#!/usr/bin/env python3
"""The suite lane's core facts (harness/lane_core.py): the socket path
the app's shell bridge dials, the machine name the core answers to, and
the bounded readiness wait. Pure tests: no sandbox, no binary."""

from __future__ import annotations

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import lane_core  # noqa: E402


class SocketPathTest(unittest.TestCase):
    def test_socket_is_the_transport_default_under_the_runtime_dir(self) -> None:
        # the Rust transport's own default_path shape: the app dials
        # $XDG_RUNTIME_DIR/herdr-core/<session>.sock, so the lane's core
        # must bind exactly there
        self.assertEqual(
            lane_core.core_socket_path("/run/user/9", "suite"),
            pathlib.Path("/run/user/9/herdr-core/suite.sock"),
        )

    def test_a_session_carrying_a_separator_names_no_socket(self) -> None:
        for session in ("", "a/b", "a\0b"):
            with self.assertRaises(lane_core.CoreRefused):
                lane_core.core_socket_path("/run/user/9", session)

    def test_an_empty_runtime_dir_has_nowhere_to_bind(self) -> None:
        with self.assertRaises(lane_core.CoreRefused):
            lane_core.core_socket_path("", "suite")


class CoreFactsTest(unittest.TestCase):
    def test_the_status_json_names_the_version_and_the_machine(self) -> None:
        # the shape the pinned core's `server start` answers with
        status = (
            '{"core":{"id":"5b294d0c28992a14","name":"other"},"finished":0,'
            '"holder":null,"holders":[],"installed_by":"nix","live":0,'
            '"pid":1234,"uptime_ms":7,"version":"0.1.17"}'
        )
        self.assertEqual(lane_core.core_facts(status), ("0.1.17", "other"))

    def test_a_status_without_a_machine_name_is_a_refusal(self) -> None:
        for line in ("", "not json", '{"version":"0.1.17"}', '{"core":{}}', '{"core":{"name":""},"version":"1"}'):
            with self.assertRaises(lane_core.CoreRefused):
                lane_core.core_facts(line)


class WaitTest(unittest.TestCase):
    def test_a_socket_that_appears_in_time_answers(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            socket = pathlib.Path(tmp) / "herdr-core" / "suite.sock"
            socket.parent.mkdir()
            self.assertRaises(lane_core.CoreRefused, lane_core.wait_for_socket, socket, 0.3)
            socket.touch()
            lane_core.wait_for_socket(socket, 1.0)

    def test_a_socket_that_never_appears_is_a_refusal_naming_it(self) -> None:
        with self.assertRaises(lane_core.CoreRefused) as caught:
            lane_core.wait_for_socket(pathlib.Path("/nonexistent/suite.sock"), 0.3)
        self.assertIn("/nonexistent/suite.sock", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
