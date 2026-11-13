#!/usr/bin/env python3
"""T-36 per-run seeded tracker db (R6): group-made boards committed
through the REAL sac binary, the seed commit list recorded for the
run manifest.

What is new here is the content, not the wire: the session lane's
tracker sequence already proves create_task plus boards/commit through
a real `sac serve` on a throwaway db - but only with text,
incorporation-at-part and figure boards. Groups have never crossed the
real wire in a lane. The boards below are the seed set the differential
run needs: a cited board whose content sits inside a group, a citing
board whose incorporation names a child INSIDE that group (the
scenario-13 lane), and a landing board whose second revision lands a
figure inside a group (the t-35 regression lane).

Everything runs against a throwaway db and repo inside the run's own
space, spawned with an owned process handle, bounded, and stopped by
that handle on every path. The manifest's seeds block carries the db
hash and the commit list (task, revision, message, git hash when the
wire answers one), so a differential pair can prove both runs started
from the same seeded tracker.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

SEED_REPO_NAME = "seedtracker"
SEED_TIMEOUT_SECONDS = 60
SEED_CONTEXT = {"actor": "t36-seed", "tier": "agent", "scope": None}
SEED_CLIENT = "t36-seed"

# a 1x1 PNG, the same shape the tracker sequence lands
LANDING_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class SeedRefused(Exception):
    """The seed run refuses: server, wire or board problem."""


def board_cited(task: int) -> str:
    """The cited seed: a plain account whose content sits inside a
    group, so every child carries a group path. Boards name their task
    in the t-N form; the wire takes the bare number."""
    return f"""grammar: 1
task: t-{task}
content:
  - group:
      title: Recorded setup
      content:
        - text: the cited account's first child, inside a group
        - text: a second child the citation names by its group path
"""


def board_citing(task: int, source_task: int, source_revision: int) -> str:
    """The citing seed: an incorporation naming a child INSIDE the
    source's group - the nested-citation lane, group-made."""
    return f"""grammar: 1
task: t-{task}
content:
  - text: the citing account, served by the real wire
  - incorporation:
      source:
        task: t-{source_task}
        revision: {source_revision}
        select: [/content/0/group/content/1]
      content:
        - text: the receiving text for the cited group child
"""


def board_citing_child(task: int, source_task: int, source_revision: int) -> str:
    """The second citing seed: its incorporation names the OTHER group
    child of the source, so the part-inside-group citation content
    exists on the wire (S08's anchor fixture; S13's second nested
    target). The incorporation's select names the child path - the
    rendered cited content is that child, inside a group on the
    source board."""
    return f"""grammar: 1
task: t-{task}
content:
  - text: the second citing account, naming the other group child
  - incorporation:
      source:
        task: t-{source_task}
        revision: {source_revision}
        select: [/content/0/group/content/0]
      content:
        - text: the receiving text for the first group child
"""


def board_landing_r1(task: int) -> str:
    return f"""grammar: 1
task: t-{task}
content:
  - text: the landing board, first pass
"""


def board_landing_r2(task: int) -> str:
    return f"""grammar: 1
task: t-{task}
content:
  - text: the landing board, revised
  - group:
      title: Figures
      content:
        - figure:
            input: figures/landing.png
            caption: the landing figure, committed inside a group
"""


def commit_record(task: str, reply: dict, message: str) -> dict:
    """One wire commit as the manifest records it: task, revision,
    message, and the git hash when the wire answers one."""
    return {
        "task": f"t-{task}" if not str(task).startswith("t-") else str(task),
        "revision": reply.get("revision"),
        "message": message,
        "commit": reply.get("commit") or reply.get("hash"),
    }


def seed_via(post) -> list[dict]:
    """Seed the whole set through a `post(route, payload) -> dict`
    callable (the tracker wire). Returns the commit list in commit
    order; a wire failure raises SeedRefused with the route and the
    answer, never a partial success pretending to be whole."""
    commits: list[dict] = []
    tasks: dict[str, int] = {}
    try:
        for name, key in (
            ("the cited seed account (group-made)", "cited"),
            ("the citing seed account (group-child citation)", "citing"),
            ("the second citing account (the other group child)", "citing_child"),
            ("the landing seed account (figure lands in a group)", "landing"),
        ):
            reply = post("/api/v1/command", {
                "context": SEED_CONTEXT,
                "command": {"create_task": {"name": name}},
                "at": None,
                "client": SEED_CLIENT,
            })
            # the wire names tasks t-N and takes the bare number back
            tasks[key] = int(str(reply["id"]).removeprefix("t-"))
        cited = tasks["cited"]
        citing = tasks["citing"]
        citing_child = tasks["citing_child"]
        landing = tasks["landing"]

        def commit(task_id: int, files: list[dict], message: str) -> None:
            reply = post("/api/v1/boards/commit", {
                "context": SEED_CONTEXT,
                "task": task_id,
                "files": {"files": files},
                "message": message,
                "client": SEED_CLIENT,
            })
            commits.append(commit_record(task_id, reply, message))

        commit(cited, [{"path": "board.yaml", "content": _b64(board_cited(cited))}],
               "t36 seed: the cited account, group-made")
        commit(citing, [{"path": "board.yaml", "content": _b64(board_citing(citing, cited, 1))}],
               "t36 seed: the citing account, incorporation into a group child")
        commit(citing_child, [{"path": "board.yaml", "content": _b64(board_citing_child(citing_child, cited, 1))}],
               "t36 seed: the second citing account, the other group child")
        commit(landing, [{"path": "board.yaml", "content": _b64(board_landing_r1(landing))}],
               "t36 seed: the landing board, text only")
        commit(landing, [
            {"path": "board.yaml", "content": _b64(board_landing_r2(landing))},
            {"path": "figures/landing.png", "content": LANDING_PNG_B64},
        ], "t36 seed: the landing figure, inside a group")
    except (KeyError, ValueError, TypeError) as error:
        raise SeedRefused(f"the seed wire answered off shape: {error!r}") from error
    return commits


def _sha256_of_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


class SeedServer:
    """One real `sac serve` on a throwaway db and repo. Owned handle;
    bounded readiness; stop() terminates by that handle only."""

    def __init__(self, sac_bin: str, root: pathlib.Path) -> None:
        if not (sac_bin and os.path.isabs(sac_bin) and os.access(sac_bin, os.X_OK)):
            raise SeedRefused(f"the real sac binary must be an executable absolute path: {sac_bin!r}")
        self.sac_bin = sac_bin
        self.root = root
        self.repo = root / SEED_REPO_NAME
        self.repo.mkdir(parents=True, exist_ok=True)
        self.db = root / "seed.db"
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = root / "logs" / "seed-serve.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        # the server shells out to git; the caller's PATH must carry it
        if shutil.which("git") is None:
            raise SeedRefused("git must be on PATH: sac serve drives the seed repo through git")
        log = open(self.log_path, "wb")
        try:
            self.process = subprocess.Popen(
                [self.sac_bin, "serve", "--db", str(self.db), "--repo", str(self.repo),
                 "--bind", "127.0.0.1", "--port", str(self.port)],
                stdout=log, stderr=log,
            )
        finally:
            log.close()
        deadline = time.monotonic() + SEED_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise SeedRefused(f"sac serve exited during startup (log: {self.log_path})")
            try:
                with urllib.request.urlopen(self.base + "/api/v1/version", timeout=2) as reply:
                    if reply.status == 200:
                        return
            except urllib.error.HTTPError:
                return  # any typed answer means the wire is up
            except OSError:
                time.sleep(0.3)
        self.stop()
        raise SeedRefused(f"sac serve never answered within {SEED_TIMEOUT_SECONDS}s (log: {self.log_path})")

    def post(self, route: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base + route,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise SeedRefused(f"{route} answered {error.code}: {detail}") from None

    def seed(self) -> list[dict]:
        if self.process is None or self.process.poll() is not None:
            raise SeedRefused("seed called on a server that is not up")
        return seed_via(self.post)

    def seeds_record(self, commits: list[dict]) -> dict:
        """The seeds record: where the db and repo are, and the db's hash."""
        return {
            "db_path": str(self.db),
            "db_hash": _sha256_of_file(self.db),
            "repo": str(self.repo),
            "serve_bin": self.sac_bin,
            "commits": commits,
        }

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> "SeedServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def resolve_sac_bin(explicit: str = "") -> str:
    found = explicit or os.environ.get("SACCADE_SERVE_BIN") or shutil.which("sac") or ""
    if not found or not os.path.isabs(found) or not os.access(found, os.X_OK):
        raise SeedRefused(
            "the real sac binary is required for seeding; pass it explicitly, set SACCADE_SERVE_BIN, or put sac on PATH"
        )
    return str(pathlib.Path(found).resolve())


def seed_run(sac_bin: str, root: pathlib.Path) -> tuple[SeedServer, list[dict], dict]:
    """One run's seed: server up, boards committed, seeds record built.
    The caller owns the server and stops it after the lane has taken
    the db and repo it needs (they live under `root`)."""
    server = SeedServer(resolve_sac_bin(sac_bin), root)
    server.start()
    commits = server.seed()
    return server, commits, server.seeds_record(commits)


def main(argv: list[str] | None = None) -> int:
    """Mechanics smoke: seed a throwaway db through the real sac binary
    and print the seeds record. Logs stay in the run directory."""
    import argparse

    parser = argparse.ArgumentParser(description="T-36 seed mechanics smoke")
    parser.add_argument("--sac-bin", default="")
    parser.add_argument("--root", default="")
    options = parser.parse_args(argv)
    root = pathlib.Path(options.root) if options.root else pathlib.Path("/tmp/t36-seed-smoke")
    if root.exists():
        print(f"seed smoke root {root} already exists; refusing to reuse it", file=sys.stderr)
        return 2
    root.mkdir(parents=True)
    try:
        server = SeedServer(resolve_sac_bin(options.sac_bin), root)
    except SeedRefused as error:
        print(f"seed refusal: {error}", file=sys.stderr)
        return 2
    try:
        server.start()
        commits = server.seed()
        record = server.seeds_record(commits)
        for entry in commits:
            ok = entry.get("revision") is not None
            print(f"[t36-seed] {'ok' if ok else 'FAIL'}: committed {entry['task']} at r{entry.get('revision')}: {entry['message']}")
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0 if all(entry.get("revision") is not None for entry in commits) else 1
    except SeedRefused as error:
        print(f"seed refusal: {error}", file=sys.stderr)
        return 2
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
