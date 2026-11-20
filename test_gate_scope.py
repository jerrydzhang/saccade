"""The merge gate's scope rule (gate_scope.py): which suites a push runs.

Run: python3 -m unittest test_gate_scope (the root justfile's test).
"""

import os
import subprocess
import tempfile
import unittest
from unittest import mock

import gate_scope
from gate_scope import APP, SACCADE


class SuitesFromPaths(unittest.TestCase):
    def test_only_app_paths_run_only_the_app_suites(self):
        self.assertEqual(gate_scope.suites(["app/port/src/main.ts", "app/README.md"]), [APP])

    def test_a_saccade_path_runs_both(self):
        self.assertEqual(gate_scope.suites(["saccade/tracker/src/main.rs"]), [SACCADE, APP])

    def test_a_grammar_path_runs_both(self):
        self.assertEqual(gate_scope.suites(["saccade/grammar/src/lib.rs", "app/port/src/main.ts"]), [SACCADE, APP])

    def test_a_root_file_runs_both(self):
        self.assertEqual(gate_scope.suites(["flake.nix"]), [SACCADE, APP])
        self.assertEqual(gate_scope.suites(["app/justfile", "prek.toml"]), [SACCADE, APP])

    def test_a_root_directory_runs_both(self):
        self.assertEqual(gate_scope.suites(["nix/flake-module.nix"]), [SACCADE, APP])

    def test_a_path_only_named_like_a_project_is_a_root_path(self):
        self.assertEqual(gate_scope.suites(["application.txt"]), [SACCADE, APP])

    def test_no_paths_run_nothing(self):
        self.assertEqual(gate_scope.suites([]), [])


class ChangedPaths(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.repo = self.scratch.name
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env = env

        def git(*args):
            return subprocess.run(
                ["git", "-C", self.repo, "-c", "user.email=gate@example", "-c", "user.name=gate", *args],
                capture_output=True, text=True, check=True, env=env,
            ).stdout.strip()

        self.git = git
        git("init", "-q")
        os.makedirs(os.path.join(self.repo, "app"))
        with open(os.path.join(self.repo, "app", "a.txt"), "w") as handle:
            handle.write("one\n")
        git("add", "-A")
        git("commit", "-q", "-m", "first")
        self.first = git("rev-parse", "HEAD")

    def commit(self, relative):
        full = os.path.join(self.repo, relative)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "a") as handle:
            handle.write("more\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", relative)
        return self.git("rev-parse", "HEAD")

    def changed(self, old, new):
        here = os.getcwd()
        os.chdir(self.repo)
        try:
            # a hook's own git environment never reaches the scratch repo
            with mock.patch.dict(os.environ, self.env, clear=True):
                return gate_scope.changed_paths(old, new)
        finally:
            os.chdir(here)

    def test_an_app_commit_names_its_app_path(self):
        second = self.commit("app/a.txt")
        self.assertEqual(self.changed(self.first, second), ["app/a.txt"])

    def test_a_root_commit_names_its_root_path(self):
        second = self.commit("flake.nix")
        self.assertEqual(self.changed(self.first, second), ["flake.nix"])

    def test_an_absent_old_side_is_unreadable(self):
        self.assertIsNone(self.changed(gate_scope.ABSENT, self.first))
        self.assertIsNone(self.changed("", self.first))

    def test_an_unknown_commit_is_unreadable(self):
        self.assertIsNone(self.changed("1" * 40, self.first))


if __name__ == "__main__":
    unittest.main()
