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
  t-0  supported grammar 1, incorporating t-1's revision (the citation
       the viewer resolves and links)
  t-1  grammar 1 carrying an unknown component kind (lenient render)
  t-2  grammar 2 (the unsupported major: refuses)
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SOURCE_COMMIT = "a" * 40
CITED_HASH = "b" * 40

T0 = f"""grammar: 1
task: t-0
content:
  - text: the citing account, leaning on the source below
  - incorporation:
      source:
        task: t-1
        revision: {CITED_HASH}
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
         "at": 1, "revision": 1, "commit": CITED_HASH, "message": "the cited account",
         "grounds": None, "explanation": None},
    ]
}


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def tree(*texts: str) -> dict:
    return {
        "files": [{"path": "board.yaml", "content": b64(text)} for text in texts],
        "commit": SOURCE_COMMIT,
        "revision": 1,
    }


BOARDS = {
    0: tree(T0),
    1: tree(T1),
    2: tree(T2),
}


class TrackerMock:
    """One mock tracker on a private loopback port. `min`/`max` are the
    grammar range the version route reports; `name` is the tracker the
    route claims to serve."""

    def __init__(self, name: str, min_version: int = 1, max_version: int = 1):
        self.name = name
        self.min_version = min_version
        self.max_version = max_version
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request noise
                pass

            def do_GET(self):
                if self.path.startswith("/api/v1/version"):
                    body = json.dumps({
                        "version": "0.0.0-mock",
                        "tracker": mock.name,
                        "board_grammar": {"min": mock.min_version, "max": mock.max_version},
                        "db": "/mock/saccade.db",
                    }).encode()
                    self._reply(body)
                elif self.path.startswith("/api/v1/boards/get"):
                    query = dict(pair.split("=", 1) for pair in self.path.split("?", 1)[-1].split("&") if "=" in pair)
                    task = int(query.get("task", "-1"))
                    board = BOARDS.get(task)
                    if board is None:
                        self._reply(json.dumps({"error": {"code": "board_missing", "detail": "no such board"}}).encode(), 404)
                    else:
                        self._reply(json.dumps(board).encode())
                elif self.path.startswith("/api/v1/boards/log"):
                    self._reply(json.dumps(HISTORY_T1).encode())
                else:
                    self._reply(b"{}", 404)

            def _reply(self, body: bytes, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
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
