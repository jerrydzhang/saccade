#!/usr/bin/env python3
"""The publish wire smoke: one throwaway db, one real `sac serve`, the
viewer's publish act over the wire exactly as the client performs it -
human-own-identity context, the shown board text plus annotation
components and nothing else, figures unchanged - then the served board
read back through /view/v1 and the guards exercised (a stale board, an
offered board)."""

from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAC = str(ROOT.parent / "target" / "debug" / "sac")


def wait_port(url: str, deadline: float = 30.0) -> None:
    # any HTTP answer is an up server: refusals are documents too
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            urllib.request.urlopen(f"{url}/view/v1/", timeout=1)
            return
        except urllib.error.HTTPError:
            return
        except Exception:
            time.sleep(0.2)
    raise SystemExit("the server never answered")


def post(url: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def get(url: str, binary: bool = False):
    with urllib.request.urlopen(url) as response:
        data = response.read()
        return data if binary else json.loads(data)


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> int:
    scratch = tempfile.mkdtemp(prefix="publish-smoke-")
    port = str(free_port())
    repo = pathlib.Path(scratch) / "project"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    db = pathlib.Path(scratch) / "saccade.db"
    serve = subprocess.Popen(
        [SAC, "serve", "--repo", str(repo), "--db", str(db), "--bind", "127.0.0.1", "--port", port],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        wait_port(base)

        # the task and its seed board ride the same wire the viewer reads
        context = {"actor": "smoker", "tier": "agent", "scope": None}
        status, reply = post(f"{base}/api/v1/command", {
            "context": context,
            "command": {"create_task": {"name": "the publish smoke"}},
            "at": None,
            "client": "smoke",
        })
        assert status == 200, reply
        task = int(str(reply["id"]).removeprefix("t-"))
        print(f"[smoke] task t-{task} created")

        board_text = (
            "grammar: 1\n"
            f"task: t-{task}\n"
            "content:\n"
            "  - text: the part the reader annotates\n"
            "  - text: an untouched part\n"
        )
        status, reply = post(
            f"{base}/api/v1/boards/commit",
            {
                "context": context,
                "task": task,
                "files": {"files": [{"path": "board.yaml",
                                     "content": base64.b64encode(board_text.encode()).decode()}]},
                "message": "the seed board",
                "client": "smoke",
            },
        )
        assert status == 200 and reply["revision"] == 1, reply
        print("[smoke] seed board committed at r1")

        # 1. the happy path: the viewer's publish, byte-exact insertion
        actor = "smoker"
        annotated = (
            "grammar: 1\n"
            f"task: t-{task}\n"
            "content:\n"
            "  - text: the part the reader annotates\n"
            "  - annotation: |\n"
            "      the reviewer's concern, drafted in place\n"
            "  - text: an untouched part\n"
        )
        status, reply = post(
            f"{base}/api/v1/boards/commit",
            {
                "context": {"actor": actor, "tier": "human", "scope": None},
                "task": task,
                "files": {"files": [{"path": "board.yaml",
                                     "content": base64.b64encode(annotated.encode()).decode()}]},
                "message": "annotations: the reviewer's concern, drafted in place",
                "client": "board-viewer",
            },
        )
        assert status == 200, f"publish refused: {status} {reply}"
        assert reply["revision"] == 2, reply
        print(f"[smoke] published as r2 under {actor}/human")

        # the served board carries the annotation beside its part: the
        # wrapped page rides the annotation as a group titled annotation
        page = get(f"{base}/view/v1/t-{task}")
        annotations = [
            part for part in page["content"]
            if part.get("kind") == "group" and part.get("title") == "annotation"
        ]
        assert len(annotations) == 1, page["content"]
        inner = annotations[0].get("content", [])
        assert any(
            part.get("kind") == "text" and "the reviewer's concern" in part.get("text", "")
            for part in inner
        ), annotations[0]
        # the record's identity: the commit entry names the human actor
        history = get(f"{base}/view/v1/t-{task}/history")
        commits = [entry for entry in history["entries"] if entry["event"] == "task_committed"]
        assert commits[-1]["actor"] == actor and commits[-1]["tier"] == "human", commits[-1]
        print("[smoke] the served board carries the annotation; the record names the human identity")

        # 2. the offered guard: offer, then publish refuses
        status, reply = post(f"{base}/api/v1/command", {
            "context": context,
            "command": {"offer_board": {"task": task, "commit": None}},
            "at": None,
            "client": "smoke",
        })
        # the offer names the commit; resolve it from the history first
        if status != 200:
            history = get(f"{base}/view/v1/t-{task}/history")
            commit = [entry for entry in history["entries"] if entry["event"] == "task_committed"][-1]["commit"]
            status, reply = post(f"{base}/api/v1/command", {
                "context": context,
                "command": {"offer_board": {"task": task, "commit": commit}},
                "at": None,
                "client": "smoke",
            })
        assert status == 200, reply
        status, reply = post(
            f"{base}/api/v1/boards/commit",
            {
                "context": {"actor": actor, "tier": "human", "scope": None},
                "task": task,
                "files": {"files": [{"path": "board.yaml",
                                     "content": base64.b64encode((annotated + "  - text: more\n").encode()).decode()}]},
                "message": "should refuse",
                "client": "board-viewer",
            },
        )
        assert status != 200, f"the offered board published: {reply}"
        # the door wraps the store's reject; the code names it in the detail
        assert "invalid_state_transition" in reply["error"]["code"] + reply["error"]["detail"], reply
        print("[smoke] the offered board refuses: invalid_state_transition")
        status, reply = post(f"{base}/api/v1/command", {
            "context": context,
            "command": {"reject_board": {"task": task, "grounds": "the smoke needs the task back"}},
            "at": None,
            "client": "smoke",
        })
        assert status == 200, reply

        # 3. the concurrency guard: the viewer's byte-compare catches a
        # landed revision - the client-side half, pinned here by reading
        # current and comparing with what the window showed
        current = get(f"{base}/view/v1/t-{task}/file/board.yaml", binary=True).decode()
        assert current != board_text, "the current board moved past the shown text"
        print("[smoke] the stale-viewer guard's wire half holds: the current text differs from the shown one")

        print("[smoke] PASS")
        return 0
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=5)
        except subprocess.TimeoutExpired:
            serve.kill()
            serve.wait()


if __name__ == "__main__":
    sys.exit(main())
