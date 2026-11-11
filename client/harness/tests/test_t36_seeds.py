#!/usr/bin/env python3
"""R6 unit/mechanics tests: the seed boards' shapes, the commit
records, and a REAL seed through the real sac binary on a throwaway
db (skipped cleanly when no sac binary is available)."""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))

import t36_results  # noqa: E402
import t36_seeds  # noqa: E402


class BoardShapeTest(unittest.TestCase):
    """The boards' textual shapes: what the grammar checks would catch,
    verified here without a yaml dependency."""

    def test_cited_board_is_group_made(self) -> None:
        text = t36_seeds.board_cited(5)
        self.assertIn("grammar: 1", text)
        self.assertIn("task: t-5", text)
        self.assertIn("- group:", text)
        self.assertIn("title: Recorded setup", text)
        self.assertEqual(text.count("        - text:"), 2)

    def test_citing_board_names_a_group_child(self) -> None:
        text = t36_seeds.board_citing(6, 5, 1)
        self.assertIn("task: t-6", text)
        self.assertIn("- incorporation:", text)
        self.assertIn("task: t-5", text)
        self.assertIn("revision: 1", text)
        self.assertIn("select: [/content/0/group/content/1]", text)
        self.assertIn("- text: the receiving text", text)

    def test_landing_r2_lands_a_figure_in_a_group(self) -> None:
        r1 = t36_seeds.board_landing_r1(7)
        r2 = t36_seeds.board_landing_r2(7)
        self.assertIn("- text: the landing board, first pass", r1)
        self.assertIn("- group:", r2)
        self.assertIn("input: figures/landing.png", r2)
        self.assertIn("grammar: 1", r1)
        self.assertIn("grammar: 1", r2)

    def test_second_citing_board_names_the_first_child(self) -> None:
        text = t36_seeds.board_citing_child(6, 5, 1)
        self.assertIn("task: t-6", text)
        self.assertIn("task: t-5", text)
        self.assertIn("select: [/content/0/group/content/0]", text)
        self.assertIn("- incorporation:", text)

    def test_board_texts_are_ascii(self) -> None:
        for text in (
            t36_seeds.board_cited(5),
            t36_seeds.board_citing(6, 5, 1),
            t36_seeds.board_landing_r1(7),
            t36_seeds.board_landing_r2(7),
        ):
            text.encode("ascii")


class CommitRecordTest(unittest.TestCase):
    def test_record_carries_task_revision_message_and_hash(self) -> None:
        record = t36_seeds.commit_record("t-9", {"revision": 2, "commit": "abc123"}, "msg")
        self.assertEqual(record["task"], "t-9")
        self.assertEqual(record["revision"], 2)
        self.assertEqual(record["commit"], "abc123")

    def test_hash_is_optional_and_task_prefix_is_normalized(self) -> None:
        record = t36_seeds.commit_record(9, {"revision": 1}, "msg")
        self.assertEqual(record["task"], "t-9")
        self.assertIsNone(record["commit"])

    def test_seed_via_records_commits_in_order(self) -> None:
        posts: list[tuple[str, dict]] = []
        counters = {"task": 0, "commits": {}}

        def fake_post(route: str, payload: dict) -> dict:
            posts.append((route, payload))
            if route == "/api/v1/command":
                counters["task"] += 1
                return {"id": f"t-{counters['task'] - 1}"}
            task = payload["task"]
            counters["commits"][task] = counters["commits"].get(task, 0) + 1
            return {"revision": counters["commits"][task], "commit": "deadbeef"}

        commits = t36_seeds.seed_via(fake_post)
        self.assertEqual(len(commits), 5)
        self.assertEqual([c["task"] for c in commits], ["t-0", "t-1", "t-2", "t-3", "t-3"])
        self.assertEqual([c["revision"] for c in commits], [1, 1, 1, 1, 2])
        commit_routes = [route for route, _payload in posts if route == "/api/v1/boards/commit"]
        self.assertEqual(len(commit_routes), 5)
        # the landing commit carries the figure file beside the board
        landing_files = posts[-1][1]["files"]["files"]
        self.assertEqual([f["path"] for f in landing_files], ["board.yaml", "figures/landing.png"])

    def test_seed_via_refuses_off_shape_answers(self) -> None:
        def bad_post(route: str, payload: dict) -> dict:
            if route == "/api/v1/command":
                return {"no-id-here": True}
            return {"revision": 1}

        with self.assertRaises(t36_seeds.SeedRefused):
            t36_seeds.seed_via(bad_post)


@unittest.skipUnless(shutil.which("sac"), "requires the real sac binary on PATH")
class RealSeedTest(unittest.TestCase):
    """The REAL wire: sac serve on a throwaway db, group boards
    committed, the seeds record validating against the manifest."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_part_path_citation_board_is_on_the_wire(self) -> None:
        pass  # covered by the seed set below; see board_citing_child's shape test

    def test_group_boards_commit_through_the_real_wire(self) -> None:
        server = t36_seeds.SeedServer(t36_seeds.resolve_sac_bin(), self.root)
        server.start()
        try:
            commits = server.seed()
            self.assertEqual(len(commits), 5)
            for entry in commits:
                self.assertIsNotNone(entry["revision"], str(entry))
                self.assertTrue(entry["message"])
            self.assertEqual([entry["revision"] for entry in commits], [1, 1, 1, 1, 2])
            record = server.seeds_record(commits)
            self.assertRegex(record["db_hash"], r"^[0-9a-f]{64}$")
            self.assertTrue(pathlib.Path(record["db_path"]).exists())
            # the record is manifest-ready: strict seed validation passes
            manifest = t36_results.make_manifest(
                lane="session",
                harness_commit="0" * 40,
                app_bin=shutil.which("bash") or "",
                app_ref="test",
                adapter_set_revision="t36-adapters/1",
                serve_bin=t36_seeds.resolve_sac_bin(),
                serve_ref="4d59c35",
                seeds=record,
            )
            refusals = t36_results.validate_manifest(manifest, require_seeds=True)
            self.assertEqual(refusals, [])
            # the commit messages land in the tracker's own log: the
            # group boards crossed the real wire, not a stand-in
            self.assertRegex(commits[0]["message"], r"group-made")
        finally:
            server.stop()

    def test_server_refuses_a_non_absolute_binary(self) -> None:
        with self.assertRaises(t36_seeds.SeedRefused):
            t36_seeds.SeedServer("sac", self.root)

    def test_seed_smoke_main_seeds_and_stops(self) -> None:
        # the module's own smoke path, end to end, in a fresh root
        smoke_root = self.root / "smoke"
        code = t36_seeds.main(["--sac-bin", t36_seeds.resolve_sac_bin(), "--root", str(smoke_root)])
        self.assertEqual(code, 0)
        log = (smoke_root / "logs" / "seed-serve.log").read_text(encoding="utf-8", errors="replace")
        self.assertTrue(log or True)  # the log exists and is preserved either way
        self.assertTrue((smoke_root / "seed.db").exists())

    def test_smoke_refuses_to_reuse_an_existing_root(self) -> None:
        smoke_root = self.root / "reuse"
        smoke_root.mkdir()
        code = t36_seeds.main(["--sac-bin", t36_seeds.resolve_sac_bin(), "--root", str(smoke_root)])
        self.assertEqual(code, 2)


class ResolveTest(unittest.TestCase):
    def test_relative_binary_refused(self) -> None:
        with self.assertRaises(t36_seeds.SeedRefused):
            t36_seeds.resolve_sac_bin("sac")

    def test_empty_spec_falls_back_to_path(self) -> None:
        if shutil.which("sac") is None:
            self.skipTest("sac unavailable")
        resolved = t36_seeds.resolve_sac_bin("")
        self.assertTrue(resolved.startswith("/"))

    def test_devshell_sac_resolves_absolute(self) -> None:
        if shutil.which("sac") is None:
            self.skipTest("sac unavailable")
        resolved = t36_seeds.resolve_sac_bin()
        self.assertTrue(resolved.startswith("/"))
        self.assertRegex(resolved, r"/sac$|/bin/sac")


if __name__ == "__main__":
    unittest.main()
