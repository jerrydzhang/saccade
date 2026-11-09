"""The tracker contract stand-in for the viewer lanes.

A tracker that never stores a board its own grammar refuses cannot hold
the unknown-kind or unsupported-major fixtures: authoring validation is
strict. The viewer's half of the contract - the root-page handshake,
the lenient unknown-kind render, the major-version refusal, the
citation labels, the landing-pinned file reads - is what the lanes pin,
so this mock speaks exactly the wire the pages define (t-14 "Pages",
t-24@r2): /view/v1/ (the root contract), /view/v1/t-N[@rK] (the board
page), /view/v1/t-N[@rK]/file/<path> (raw bytes), and
/view/v1/t-N/history, over canned boards. The real wire is pinned on
the server side by the cargo api tests; here the viewer's behavior is
the thing under test.

Boards (the per-version fixtures):
  t-0  supported grammar 1, incorporating t-1's revision by number
       (the citation the viewer labels and links)
  t-1  grammar 1 carrying an unknown component kind (lenient render)
  t-2  grammar 2 (the unsupported major: refuses)
"""

from __future__ import annotations

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


def board_page(task: int, revision: int, name: str, text: str, files: dict[str, str]) -> dict:
    """The board page the real server wraps (t-15): facts, refs, then
    the authored content, every part carrying its tracker-addressed
    location. An incorporation rides its collapsed shape: the source
    revision ref, the receiving text, a ref per cited component. An
    unknown component kind rides as its raw source in a text fallback,
    the degrade-instead-of-break reading."""
    content: list[dict] = [
        {"kind": "fact", "label": "task", "value": f"t-{task}"},
        {"kind": "fact", "label": "name", "value": name},
        {"kind": "fact", "label": "state", "value": "open"},
        {"kind": "fact", "label": "revision", "value": f"r{revision} of {revision}"},
        {"kind": "fact", "label": "commit", "value": SOURCE_COMMIT},
        {
            "kind": "ref",
            "location": f"t-{task}/history",
            "label": "history",
        },
        {
            "kind": "ref",
            "location": f"t-{task}@r{revision}/file/board.yaml",
            "label": "board.yaml",
        },
    ]
    parts = parse_components(text)
    for index, component in enumerate(parts):
        content.append(wrap_component(task, revision, f"/content/{index}", component))
    return {
        "view": {"major": 1},
        "kind": "document",
        "location": f"t-{task}@r{revision}",
        "seq": 1,
        "task": f"t-{task}",
        "revision": revision,
        "content": content,
        "_files": files,
    }


def parse_components(text: str) -> list[dict]:
    """The canned boards' components, parsed just enough to wrap: the
    mock's boards are fixed shapes, so a line-based reader is exact."""
    components: list[dict] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("  - text: "):
            components.append({"kind": "text", "text": line[len("  - text: "):]})
            index += 1
        elif line.startswith("  - incorporation:"):
            block: list[str] = []
            index += 1
            while index < len(lines) and lines[index].startswith("      "):
                block.append(lines[index])
                index += 1
            source_task = next(int(b.split(": ")[1][2:]) for b in block if b.strip().startswith("task:"))
            source_revision = next(int(b.split(": ")[1]) for b in block if b.strip().startswith("revision:"))
            select = next(b.split("select: ")[1] for b in block if b.strip().startswith("select:"))
            receiving = next(b.split("text: ")[1] for b in block if b.strip().startswith("- text:"))
            components.append(
                {
                    "kind": "incorporation",
                    "source": {"task": source_task, "revision": source_revision, "select": select},
                    "receiving": receiving,
                }
            )
        elif line.startswith("  - figure:"):
            block = []
            index += 1
            while index < len(lines) and lines[index].startswith("      "):
                block.append(lines[index])
                index += 1
            input_path = next(b.split("input: ")[1] for b in block if b.strip().startswith("input:"))
            caption = next(b.split("caption: ")[1] for b in block if b.strip().startswith("caption:"))
            components.append({"kind": "figure", "input": input_path, "caption": caption})
        elif line.startswith("  - ") and ":" in line:
            # an unknown component kind: the authored scalar, wrapped raw
            components.append({"kind": line[4:].split(":")[0], "raw": line})
            index += 1
        else:
            index += 1
    return components


def wrap_component(task: int, revision: int, path: str, component: dict) -> dict:
    location = f"t-{task}@r{revision}:{path}"
    kind = component["kind"]
    if kind == "text":
        return {"kind": "text", "text": component["text"], "location": location}
    if kind == "incorporation":
        source = component["source"]
        children = [
            {
                "kind": "ref",
                "location": f"t-{source['task']}@r{source['revision']}",
                "label": f"source revision t-{source['task']}@r{source['revision']}",
            },
            {"kind": "text", "text": component["receiving"], "location": f"{location}/incorporation/content/0"},
            {
                "kind": "ref",
                "location": f"t-{source['task']}@r{source['revision']}{source['select']}",
                "label": f"cites {source['select']}",
            },
        ]
        return {
            "kind": "group",
            "title": f"incorporates t-{source['task']}@r{source['revision']}",
            "content": children,
            "location": location,
        }
    if kind == "figure":
        return {
            "kind": "figure",
            "caption": component["caption"],
            "file": f"t-{task}@r{revision}/file/{component['input']}",
            "location": location,
        }
    # an unknown board kind: raw source beside a diagnostic, the
    # reader-side fallback a later minor's kinds degrade to
    return {"kind": "text", "text": component["raw"], "location": location}


T0_FILES = {"board.yaml": T0}
T1_FILES = {"board.yaml": T1}
T2_FILES = {"board.yaml": T2}

HISTORY_T1 = {
    "view": {"major": 1},
    "kind": "timeline",
    "location": "t-1/history",
    "seq": 4,
    "task": "t-1",
    "revision": 1,
    "entries": [
        {
            "seq": 4,
            "event": "task_committed",
            "actor": "smoke",
            "tier": "agent",
            "at": 1,
            "revision": 1,
            "commit": SOURCE_COMMIT,
            "message": "the cited account",
            "grounds": None,
            "explanation": None,
            "session": None,
            "refs": [
                {"kind": "ref", "location": "t-1@r1", "label": "revision r1"},
            ],
        },
    ],
}

LANDING_TASK = 3
LANDING_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# the landing task: a revision landing between one open's revision
# resolution and a re-open. r1 embeds the image in prose and carries
# the file; r2 carries the figure component and the same file. The
# board page answers r1 to the first fetch and r2 afterwards - and
# every file fetch rides the pinned revision, so one open's reads are
# one immutable tree whatever lands mid-open
LANDING_R1 = f"""grammar: 1
task: t-{LANDING_TASK}
content:
  - text: the landing board, first pass ![the error render](figures/error-render.png)
"""
LANDING_R2 = f"""grammar: 1
task: t-{LANDING_TASK}
content:
  - text: the landing board, revised
  - figure:
      input: figures/error-render.png
      caption: the error render, committed
"""


class TrackerMock:
    """One mock tracker on a private loopback port. `min`/`max` are the
    grammar range the root page reports; `name` is the tracker the root
    page claims to serve."""

    def __init__(self, name: str, min_version: int = 1, max_version: int = 1, allowed_origins: list[str] | None = None):
        self.name = name
        self.min_version = min_version
        self.max_version = max_version
        # how many board-page fetches the landing task has served: the
        # straddle pin's counter
        self.landing_gets = 0
        self.landing_served = False
        # the origin policy the routes carry: a request naming an origin
        # outside the allowlist refuses on every method; an allowlisted
        # read gets its own origin echoed, never a wildcard
        self.allowed_origins = allowed_origins or []
        mock = self

        def root_page() -> dict:
            return {
                "view": {"major": 1},
                "kind": "root",
                "location": "/",
                "seq": 1,
                "tracker": mock.name,
                "view_majors": [1],
                "board_grammar": {"min": mock.min_version, "max": mock.max_version},
                "content": [
                    {"kind": "fact", "label": "family", "value": "boards: /view/v1/t-N and /view/v1/t-N@rK"},
                ],
            }

        def board_pages() -> dict[int, dict]:
            pages = {
                0: board_page(0, 1, "the citing account", T0, T0_FILES),
                1: board_page(1, 1, "the unknown kind", T1, T1_FILES),
                2: board_page(2, 1, "the future grammar", T2, T2_FILES),
            }
            return pages

        def landing_board(revision: int) -> tuple[dict, dict[str, str]]:
            text = LANDING_R1 if revision == 1 else LANDING_R2
            files = {
                "board.yaml": text,
                "figures/error-render.png": LANDING_PNG,
            }
            return board_page(LANDING_TASK, revision, "the landing board", text, files), files

        def landing_page() -> dict:
            # the landing simulation: the bare page answers r1 once,
            # r2 afterwards - a revision lands between one open's
            # resolution and the next
            first = not mock.landing_served
            mock.landing_gets += 1
            mock.landing_served = True
            page, _files = landing_board(1 if first else 2)
            return page

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
                path = self.path.split("?", 1)[0]
                if not path.startswith("/view/v1/"):
                    self._reply(b"{}", 404, origin=origin)
                    return
                route = path[len("/view/v1/"):]
                if route == "":
                    self._reply(json.dumps(root_page()).encode(), origin=origin)
                    return
                pages = board_pages()
                if route.endswith("/history"):
                    spec = route[: -len("/history")]
                    if spec == "t-1" or spec == "t-1@r1":
                        self._reply(json.dumps(HISTORY_T1).encode(), origin=origin)
                    else:
                        self._reply(refusal(route, "unknown_location", "no history for this location"), 404, origin=origin)
                    return
                if "/file/" in route:
                    spec, _, relative = route.partition("/file/")
                    answered = page_for_spec(pages, spec)
                    if answered is None:
                        self._reply(refusal(route, "unknown_task", "no such task"), 404, origin=origin)
                        return
                    _page, files = answered
                    if relative in files:
                        kind = "text/yaml" if relative.endswith(".yaml") else "image/png"
                        self._reply_bytes(files[relative].encode(), kind, origin=origin)
                    else:
                        self._reply(refusal(route, "file_missing", f"no file '{relative}' in {spec}"), 404, origin=origin)
                    return
                if route == f"t-{LANDING_TASK}":
                    page = landing_page()
                else:
                    answered = page_for_spec(pages, route)
                    page = None if answered is None else answered[0]
                if page is None:
                    self._reply(refusal(route, "unknown_task", "no such task"), 404, origin=origin)
                else:
                    body = {k: v for k, v in page.items() if not k.startswith("_")}
                    self._reply(json.dumps(body).encode(), origin=origin)

            def _reply(self, body: bytes, status: int = 200, origin: str | None = None, kind: str = "application/json"):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                if origin is not None:
                    self.send_header("Access-Control-Allow-Origin", origin)
                self.end_headers()
                self.wfile.write(body)

            def _reply_bytes(self, body: bytes, kind: str, origin: str | None = None):
                self._reply(body, 200, origin, kind)

        def refusal(location: str, code: str, detail: str) -> dict:
            return {
                "view": {"major": 1},
                "kind": "refusal",
                "location": location,
                "seq": 1,
                "code": code,
                "content": [
                    {"kind": "fact", "label": "refusal", "value": code},
                    {"kind": "text", "text": detail},
                ],
            }

        def page_for_spec(pages: dict[int, dict], spec: str) -> tuple[dict, dict[str, str]] | None:
            """The page and its files at the revision the spec names:
            a pinned read is immutable, whatever landed after it."""
            if spec.startswith(f"t-{LANDING_TASK}@"):
                revision = int(spec.partition("@r")[2])
                return landing_board(revision)
            if spec == f"t-{LANDING_TASK}":
                return landing_board(1)
            if "@" in spec:
                head, _, _revision = spec.partition("@")
                task = int(head[2:])
            else:
                task = int(spec[2:])
            page = pages.get(task)
            if page is None:
                return None
            return page, page["_files"]

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
