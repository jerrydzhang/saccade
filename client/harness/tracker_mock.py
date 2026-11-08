"""The tracker contract stand-in for the viewer lanes.

A tracker that never stores a board its own grammar refuses cannot hold
the unknown-kind or unsupported-major fixtures: authoring validation is
strict. The viewer's half of the contract - the version handshake, the
lenient unknown-kind render, the major-version refusal, the citation
labels - is what the lanes pin, so this mock speaks exactly the wire the
P4 routes define: /api/v1/version, /api/v1/boards/get, and
/api/v1/boards/log, over canned boards. The real wire is pinned on the
server side by the cargo api tests; here the viewer's behavior is the
thing under test.

Boards (the per-version fixtures):
  t-0  supported grammar 1, incorporating t-1's revision by number
       (the citation the viewer labels and links)
  t-1  grammar 1 carrying an unknown component kind (lenient render)
  t-2  grammar 2 (the unsupported major: refuses)
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SOURCE_COMMIT = "a" * 40
CITED_REVISION = 1

T0 = f"""grammar: 1
task: t-0
content:
  - text: the citing account, leaning on the source below
  - incorporation:
      source:
        task: t-1
        revision: {CITED_REVISION}
        select: [/content/0]
      content:
        - text: the receiving text for the cited source
"""

T1 = """grammar: 1
task: t-1
content:
  - changelog: a component kind this viewer does not know
"""

T2 = """grammar: 2
task: t-2
content: []
"""

HISTORY_T1 = {
    "history": [
        {"seq": 4, "event": "task_committed", "actor": "smoke", "tier": "agent",
         "at": 1, "revision": 1, "commit": SOURCE_COMMIT, "message": "the cited account",
         "grounds": None, "explanation": None},
    ]
}


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def tree(*texts: str) -> dict:
    return {
        "files": {"files": [{"path": "board.yaml", "content": b64(text)} for text in texts]},
        "commit": SOURCE_COMMIT,
        "revision": 1,
    }


BOARDS = {
    0: tree(T0),
    1: tree(T1),
    2: tree(T2),
}

# the landing task: a revision landing between the text fetch and the
# figure fetch of one open. The first boards/get answers the new tree
# (caption, figure component, the file); every later boards/get
# answers the old tree (text with the image embedded, no file) - the
# exact split behind "caption from the new revision, figure missing"
LANDING_TASK = 3
LANDING_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
LANDING_NEW_BOARD = f"""grammar: 1
task: t-{LANDING_TASK}
content:
  - text: the landing board, revised
  - figure:
      input: figures/error-render.png
      caption: the error render, committed
"""
LANDING_OLD_BOARD = f"""grammar: 1
task: t-{LANDING_TASK}
content:
  - text: the landing board, first pass ![the error render](figures/error-render.png)
"""


def landing_tree(first: bool) -> dict:
    if first:
        return tree(LANDING_NEW_BOARD, "") | {
            "files": {
                "files": [
                    {"path": "board.yaml", "content": b64(LANDING_NEW_BOARD)},
                    {"path": "figures/error-render.png", "content": LANDING_PNG},
                ]
            },
            "commit": SOURCE_COMMIT,
            "revision": 2,
        }
    return tree(LANDING_OLD_BOARD) | {"revision": 1}


class TrackerMock:
    """One mock tracker on a private loopback port. `min`/`max` are the
    grammar range the version route reports; `name` is the tracker the
    route claims to serve."""

    def __init__(self, name: str, min_version: int = 1, max_version: int = 1, allowed_origins: list[str] | None = None):
        self.name = name
        self.min_version = min_version
        self.max_version = max_version
        # how many boards/get requests the landing task has served: the
        # straddle pin's counter
        self.landing_gets = 0
        # the origin policy the routes carry: a request naming an origin
        # outside the allowlist refuses on every method; an allowlisted
        # read gets its own origin echoed, never a wildcard
        self.allowed_origins = allowed_origins or []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request noise
                pass

            def do_GET(self):
                # http.server collapses a leading // before self.path; the
                # real tracker answers 404, so the stand-in must too
                if self.requestline.split(" ")[1].startswith("//"):
                    self._reply(json.dumps({"error": {"code": "not_found", "detail": "no route"}}).encode(), 404)
                    return
                origin = self.headers.get("Origin")
                if origin is not None and origin not in mock.allowed_origins:
                    self._reply(
                        json.dumps({"error": {"code": "origin_forbidden", "detail": f"origin '{origin}' is not allowed here"}}).encode(),
                        403,
                    )
                    return
                if self.path.startswith("/api/v1/version"):
                    body = json.dumps({
                        "version": "0.0.0-mock",
                        "tracker": mock.name,
                        "board_grammar": {"min": mock.min_version, "max": mock.max_version},
                        "db": "/mock/saccade.db",
                    }).encode()
                    self._reply(body, origin=origin)
                elif self.path.startswith("/api/v1/boards/get"):
                    query = dict(pair.split("=", 1) for pair in self.path.split("?", 1)[-1].split("&") if "=" in pair)
                    task = int(query.get("task", "-1"))
                    if task == LANDING_TASK:
                        mock.landing_gets += 1
                        board = landing_tree(mock.landing_gets == 1)
                    else:
                        board = BOARDS.get(task)
                    if board is None:
                        self._reply(json.dumps({"error": {"code": "board_missing", "detail": "no such board"}}).encode(), 404, origin=origin)
                    else:
                        self._reply(json.dumps(board).encode(), origin=origin)
                elif self.path.startswith("/api/v1/boards/log"):
                    self._reply(json.dumps(HISTORY_T1).encode(), origin=origin)
                else:
                    self._reply(b"{}", 404, origin=origin)

            def _reply(self, body: bytes, status: int = 200, origin: str | None = None):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                if origin is not None:
                    self.send_header("Access-Control-Allow-Origin", origin)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
