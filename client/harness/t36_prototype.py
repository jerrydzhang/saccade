#!/usr/bin/env python3
"""T-36 prototype side (R4): the round-16 prototype in a WebKit
webview, the shim injected, the same driver family driving input.

The executable prototype is the COMPILED round-16 mockup: shell16.html
with the inst16 instrument assembled in and the data embedded - exactly
what reference/build16.py produces from the t-34 templates. The raw
inst16.html/shell16.html are build inputs carrying /*DATA*/-style
placeholders and do not execute; the harness therefore materializes
the compiled artifact from its provenance revision t-22@r3 through the
real `sac` binary (the same wire every other seed uses), verifies it
against the templates' markers, and records source, revision and
sha256 into the run manifest.

Three pieces:

- materialize(): fetch the compiled page via `sac board get`, gunzip,
  verify markers, hash.
- vendor_libs(): fetch the page's three pinned CDN scripts (js-yaml
  4.1.0, marked 12.0.2, mermaid 10.9.1 - the exact versions the page
  names) once, with sha256s recorded, so the page can boot without
  network (the red-run sandbox has none).
- serve(): a loopback stdlib server that serves the UNMODIFIED page
  bytes with the three CDN URLs rewritten to the vendored same-origin
  paths. The page file is never edited; the rewrite lives in the
  server, the way the browser lane serves /app/dist.
- PrototypeSession: WebKit drives the page (Playwright's WebKit for
  the mechanics tests, WebKitWebDriver through webdriver_client for
  the lane - same driver family the session lane uses), injects the
  shim, and answers readouts. Input rides the same session; the shim
  never drives anything.
"""

from __future__ import annotations

import gzip
import hashlib
import http.server
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROTOTYPE_BOARD_REF = "t-22@r3"
PROTOTYPE_BOARD_FILE = "reference/saccade-mockups-r16.html.gz"
# markers the compiled page must carry (shell16 + inst16 assembled)
MARKERS = ('<title>Saccade Mockups R16</title>', '<script id="inst" type="application/json">', "function snapshot()")
TEMPLATE_MARKERS = ("/*DATA*/", "/*INST*/", "/*TOKENS*/")

CDN_LIBS = (
    {"name": "js-yaml.min.js", "url": "https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"},
    {"name": "marked.min.js", "url": "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"},
    {"name": "mermaid.min.js", "url": "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"},
)


class PrototypeRefused(Exception):
    """The prototype lane refuses: provenance or artifact problem."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sac_bin_resolved(explicit: str = "") -> str:
    found = explicit or shutil.which("sac") or ""
    if not found or not os.access(found, os.X_OK):
        raise PrototypeRefused("the real sac binary is needed to materialize the prototype; pass --sac-bin or put sac on PATH")
    return str(pathlib.Path(found).resolve())


def materialize(cache_dir: pathlib.Path, sac_bin: str = "") -> dict:
    """Fetch the compiled prototype from its provenance revision through
    the real sac binary, gunzip it, verify the compiled markers, and
    return the provenance record. Cached by content hash; a cached file
    is re-verified, never trusted blindly."""
    sac = sac_bin_resolved(sac_bin)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "saccade-mockups-r16.html.gz"
    if not gz_path.exists() or gz_path.stat().st_size == 0:
        result = subprocess.run(
            [sac, "board", "get", PROTOTYPE_BOARD_REF, "--file", PROTOTYPE_BOARD_FILE, "--out", str(gz_path)],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0 or not gz_path.exists():
            raise PrototypeRefused(
                f"sac board get {PROTOTYPE_BOARD_REF} {PROTOTYPE_BOARD_FILE} failed: "
                f"{result.stderr.decode(errors='replace')[:300]}"
            )
    gz_bytes = gz_path.read_bytes()
    page_bytes = gzip.decompress(gz_bytes)
    page_path = cache_dir / "saccade-mockups-r16.html"
    page_path.write_bytes(page_bytes)
    verify_page(page_bytes)
    return {
        "source": PROTOTYPE_BOARD_REF,
        "board_file": PROTOTYPE_BOARD_FILE,
        "gzip_sha256": sha256_bytes(gz_bytes),
        "sha256": sha256_bytes(page_bytes),
        "path": str(page_path),
        "bytes": len(page_bytes),
    }


def verify_page(page_bytes: bytes) -> None:
    text = page_bytes.decode("utf-8", errors="replace")
    for marker in MARKERS:
        if marker not in text:
            raise PrototypeRefused(f"the artifact does not look like the compiled round-16 mockup; missing marker: {marker!r}")
    for marker in TEMPLATE_MARKERS:
        if marker in text:
            raise PrototypeRefused(
                f"unbuilt template marker {marker!r} found: this is a build input,"
                " not the executable prototype; materialize the compiled artifact from "
                + PROTOTYPE_BOARD_REF
            )


def vendor_libs(cache_dir: pathlib.Path, timeout: float = 60.0) -> dict:
    """Fetch the page's pinned CDN scripts once and record their hashes.
    The red-run sandbox has no network; the vendored copies ride the
    lane binds and the served page references them same-origin."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    vendored: dict[str, dict] = {}
    for lib in CDN_LIBS:
        target = cache_dir / lib["name"]
        if not target.exists() or target.stat().st_size == 0:
            try:
                with urllib.request.urlopen(lib["url"], timeout=timeout) as reply:
                    target.write_bytes(reply.read())
            except (OSError, urllib.error.URLError) as error:
                raise PrototypeRefused(f"vendoring {lib['url']} failed (needed once, offline sandbox depends on it): {error}") from error
        data = target.read_bytes()
        if b"<!DOCTYPE html>" in data[:200] or not data:
            raise PrototypeRefused(f"vendored {lib['name']} is not a javascript payload")
        vendored[lib["name"]] = {"url": lib["url"], "sha256": sha256_bytes(data), "path": str(target), "bytes": len(data)}
    return vendored


def rewritten_page(page_bytes: bytes) -> bytes:
    """The page bytes with the three CDN script URLs rewritten to the
    same-origin vendored paths. The artifact on disk stays untouched;
    this is a serving transformation, recorded in the manifest."""
    text = page_bytes.decode("utf-8")
    for lib in CDN_LIBS:
        text = text.replace(lib["url"], "/vendor/" + lib["name"])
    return text.encode("utf-8")


class _PrototypeHandler(http.server.BaseHTTPRequestHandler):
    page: bytes = b""
    libs: dict = {}

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # silence per-request chatter; the run root carries real logs

    def do_GET(self) -> None:
        if self.path in ("/prototype.html", "/"):
            body, ctype = self.page, "text/html; charset=utf-8"
        elif self.path.startswith("/vendor/"):
            name = self.path[len("/vendor/"):]
            record = self.libs.get(name)
            if record is None:
                self.send_error(404)
                return
            body = pathlib.Path(record["path"]).read_bytes()
            ctype = "application/javascript" if name.endswith(".js") else "application/octet-stream"
        elif self.path.startswith("/figures/"):
            # the compiled page's figure payloads are embedded; the path
            # space stays reserved so a future external figure resolves
            self.send_error(404)
            return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


class PrototypeServer:
    """One loopback server for one page. Owned handle; stop() joins it.
    The page is served from memory (rewritten once at construction),
    so the artifact on disk is never the served bytes."""

    def __init__(self, page_bytes: bytes, libs: dict) -> None:
        handler = type("BoundHandler", (_PrototypeHandler,), {"page": rewritten_page(page_bytes), "libs": libs})
        self.port = free_port()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def shim_expression() -> str:
    """The shim source as one expression (an IIFE). Playwright
    evaluates it as-is; WebDriver execute wraps it with `return `."""
    return (HERE / "t36_shim.js").read_text(encoding="utf-8")


def webkit_launch_env() -> dict[str, str]:
    """The software-GL pins the WebKit webview needs on this machine.
    The pinned devshell exports them; a bare process may not carry
    them, so mechanics tests discover the mesa vendor dir the same way
    the flake pins it (a store mesa, never a host GPU path)."""
    env: dict[str, str] = {}
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    if not os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", "").startswith("/nix/store"):
        for candidate in sorted(pathlib.Path("/nix/store").glob("*mesa*/share/glvnd/egl_vendor.d")):
            env["__EGL_VENDOR_LIBRARY_DIRS"] = str(candidate)
            break
    return env


class PrototypeSession:
    """The prototype page in a Playwright-driven WebKit webview, the
    shim injected after boot. Readout is the shim's answer; input is
    real keyboard/pointer through the same driver. The WebKitWebDriver
    lane path (below) shares shim and server; only the client differs."""

    def __init__(self, server: PrototypeServer, viewport: tuple[int, int] = (1400, 1000)) -> None:
        self.server = server
        self.viewport = viewport
        self._playwright = None
        self._browser = None
        self._driver_handles: list[tuple[int, int | None]] = []
        self.page = None

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        # the software-GL pins: the devshell exports them; a bare run
        # self-pins the same store paths (defaults only, never overrides)
        for key, value in webkit_launch_env().items():
            os.environ.setdefault(key, value)
        children_before = set(_children_of(os.getpid()))
        self._playwright = sync_playwright().start()
        # the driver process is the spawn delta: scoped-teardown root
        # (the watchdog may only ever kill THIS subtree, never the
        # orchestrator's other children - the app's detached lane
        # supervisor is a sibling and must survive)
        spawned = set(_children_of(os.getpid())) - children_before
        self._driver_handles = [(pid, _starttime(pid)) for pid in sorted(spawned)]
        self._browser = self._playwright.webkit.launch(headless=True)
        self.page = self._browser.new_page(viewport={"width": self.viewport[0], "height": self.viewport[1]})
        page_errors = []
        self.page.on("pageerror", lambda error: page_errors.append(str(error)[:200]))
        self.page.goto(self.server.base_url + "/prototype.html", wait_until="load", timeout=60_000)
        # boot: the shell script builds SUITE and the frames; wait for
        # the shim's own booted answer rather than a wall-clock sleep
        for _ in range(100):
            probe = self.page.evaluate("({suite: typeof SUITE !== 'undefined', frames: typeof frames !== 'undefined' ? Object.keys(frames).length : -1})")
            if probe.get("suite") and probe.get("frames", -1) >= 0 and self.page.evaluate("!!(SUITE && SUITE.states && Object.keys(SUITE.states).length)"):
                break
            self.page.wait_for_timeout(200)
        else:
            raise PrototypeRefused(f"the prototype never booted in WebKit; page errors: {page_errors[:4]}")
        self.page.wait_for_timeout(1500)  # first render of the instrument frames
        self._inject()

    def _inject(self) -> None:
        assert self.page is not None
        answer = self.page.evaluate(shim_expression())
        if not isinstance(answer, dict) or not answer.get("booted"):
            raise PrototypeRefused(f"the shim reported an unbooted prototype: {json.dumps(answer)[:200]}")
        # the holder is a FUNCTION: each readout re-runs the shim, it is
        # never a cached boot-time answer
        # parens keep the return operand on the return line: the shim
        # source opens with a multi-line comment, and a line terminator
        # after `return` would make ASI swallow the operand
        self.page.evaluate("window.__t36Shim = function () { return (" + shim_expression() + "); }")

    def readout(self) -> dict:
        assert self.page is not None
        answer = self.page.evaluate("window.__t36Shim ? window.__t36Shim() : (" + shim_expression() + ")")
        if not isinstance(answer, dict):
            raise PrototypeRefused("the shim returned no object")
        return answer

    def input(self, *keys: str, delay_ms: int = 120) -> None:
        """Real keyboard input through the same driver session; the
        events land on whatever the prototype's own focus logic holds."""
        assert self.page is not None
        for key in keys:
            self.page.keyboard.press(key)
            self.page.wait_for_timeout(delay_ms)

    def screenshot(self, path: pathlib.Path) -> None:
        assert self.page is not None
        self.page.screenshot(path=str(path))

    # the teardown bound (the bg-33 bounding): Playwright's close and
    # stop are cleanup RPCs with no library timeout - a wedged or dead
    # prototype browser blocks the owning thread forever. The sync API
    # is thread-affine (a close from another thread throws 'Cannot
    # switch to a different thread' - verified live), so the close
    # stays on the OWNING thread and a watchdog kills the driver's
    # process subtree at the deadline; the driver's death closes the
    # pipe, the close unblocks with a target-closed error, and the
    # teardown completes. Verified live: the wedged case returns at
    # the bound, the healthy case closes normally.
    TEARDOWN_TIMEOUT_SECONDS = 20.0

    def stop(self) -> None:
        browser, playwright = self._browser, self._playwright
        driver_handles = self._driver_handles
        self._browser = None
        self._playwright = None
        self._driver_handles = []
        self.page = None
        if browser is None and playwright is None:
            return
        fired = threading.Event()
        fired_fired = False
        started = time.monotonic()
        print(f"[t36-prototype] teardown START (bound {self.TEARDOWN_TIMEOUT_SECONDS:.0f}s)", flush=True)

        def watchdog() -> None:
            if fired.wait(timeout=self.TEARDOWN_TIMEOUT_SECONDS):
                return
            nonlocal fired_fired
            fired_fired = True
            for pid, start in driver_handles:
                # fail closed: a missing launch identity (None start)
                # is NOT ownership proof - no kill
                if start is not None and _starttime(pid) == start:
                    _kill_subtree(pid, start)

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()
        try:
            if browser is not None:
                try:
                    browser.close()
                except Exception:  # noqa: BLE001 - a watchdog-fired close ends in a target-closed error
                    pass
            if playwright is not None:
                try:
                    playwright.stop()
                except Exception:  # noqa: BLE001 - teardown never raises
                    pass
        finally:
            fired.set()
            elapsed = time.monotonic() - started
            print(f"[t36-prototype] teardown END {elapsed:.1f}s"
                  + (" (watchdog deadline reached; scoped identity checks applied)" if fired_fired else " (clean)"),
                  flush=True)


def _children_of(pid: int) -> list[int]:
    """The direct children of pid, from /proc."""
    return _proc_children().get(pid, [])


def _proc_children() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return children


def _starttime(pid: int) -> int | None:
    """The process's start time in clock ticks (stat field 22): the
    PID-reuse guard - a kill only fires when the start time still
    matches what was captured."""
    try:
        stat = (pathlib.Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _kill_subtree(root: int, root_start: int | None) -> None:
    """SIGKILL root and its whole current subtree. Called only for a
    driver whose start time still matches its launch capture; every
    collected pid is re-checked against the start time captured at
    discovery, immediately before the kill (fail closed on reuse)."""
    children = _proc_children()
    collected: list[tuple[int, int | None]] = [(root, root_start)]
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, ()):
            collected.append((child, _starttime(child)))
            frontier.append(child)
    for pid, start in collected:
        if start is not None and _starttime(pid) == start:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


class WebKitWebDriverSession:
    """The lane path: WebKitWebDriver (the session lane's native
    driver) driving a plain WebKit browser at the served prototype,
    shim injected through execute/sync. Requires the WebKitWebDriver
    binary and a WebKit browser binary the driver may spawn; both are
    provenance-refused, never guessed. Exercised by the red run under
    the lane sandbox; the mechanics tests cover the Playwright path on
    the same server and shim."""

    def __init__(self, server: PrototypeServer, webkit_driver_bin: str, browser_bin: str, port: int = 0) -> None:
        if not (webkit_driver_bin and os.path.isabs(webkit_driver_bin) and os.access(webkit_driver_bin, os.X_OK)):
            raise PrototypeRefused(f"WebKitWebDriver must be an executable absolute path: {webkit_driver_bin!r}")
        if not (browser_bin and os.path.isabs(browser_bin) and os.access(browser_bin, os.X_OK)):
            raise PrototypeRefused(f"the WebKit browser binary must be an executable absolute path: {browser_bin!r}")
        import webdriver_client

        self.port = port or free_port()
        self.client = webdriver_client.WebDriver(self.port, timeout=60.0)
        self._webkit_driver_bin = webkit_driver_bin
        self._browser_bin = browser_bin
        self.server = server

    def start(self) -> None:
        import socket as _socket
        import time as _time
        import webdriver_client

        self._driver = subprocess.Popen(
            [self._webkit_driver_bin, "--port", str(self.port), "--host", "127.0.0.1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = 15.0
        end = _time.monotonic() + deadline
        while _time.monotonic() < end:
            probe = _socket.socket()
            probe.settimeout(0.5)
            try:
                probe.connect(("127.0.0.1", self.port))
                probe.close()
                break
            except OSError:
                probe.close()
                _time.sleep(0.2)
        else:
            self.stop()
            raise PrototypeRefused("WebKitWebDriver never listened on its port")
        # the capability shape tauri-driver emits for its native driver:
        # webkitgtk:browserOptions names the browser binary and args
        value = self.client.request("POST", "/session", {
            "capabilities": {
                "alwaysMatch": {
                    "browserName": "MiniBrowser",
                    "webkitgtk:browserOptions": {
                        "binary": self._browser_bin,
                        "args": ["--headless"],
                    },
                }
            }
        })
        if not isinstance(value, dict) or not value.get(webdriver_client.SESSION_KEY):
            self.stop()
            raise PrototypeRefused(f"the WebKitWebDriver session refused the browser: {json.dumps(value)[:200]}")
        self.client.session = str(value[webdriver_client.SESSION_KEY])
        self.client.request("POST", f"/session/{self.client.session}/url", {"url": self.server.base_url + "/prototype.html"})
        for _ in range(100):
            booted = self.client.execute("return !!(window.SUITE && SUITE.states && Object.keys(SUITE.states).length);")
            if booted is True:
                break
            _time.sleep(0.2)
        else:
            self.stop()
            raise PrototypeRefused("the prototype never booted under WebKitWebDriver")
        _time.sleep(1.5)
        answer = self.client.execute("return (" + shim_expression() + ")")
        if not isinstance(answer, dict) or not answer.get("booted"):
            self.stop()
            raise PrototypeRefused("the shim reported an unbooted prototype under WebKitWebDriver")
        self.client.execute("window.__t36Shim = function () { return (" + shim_expression() + "); }")

    def readout(self) -> dict:
        answer = self.client.execute("return window.__t36Shim ? window.__t36Shim() : (" + shim_expression() + ")")
        if not isinstance(answer, dict):
            raise PrototypeRefused("the shim returned no object")
        return answer

    def stop(self) -> None:
        client = getattr(self, "client", None)
        if client is not None and client.session:
            try:
                client.delete_session()
            except Exception:  # noqa: BLE001 - teardown owns the driver on every path
                pass
        driver = getattr(self, "_driver", None)
        if driver is not None and driver.poll() is None:
            driver.terminate()
            try:
                driver.wait(timeout=5)
            except subprocess.TimeoutExpired:
                driver.kill()
                driver.wait(timeout=5)


def prototype_record(materialized: dict, vendored: dict) -> dict:
    """The manifest's prototype block: provenance revision, page hash
    and the vendored library hashes."""
    return {
        "source": materialized["source"],
        "sha256": materialized["sha256"],
        "serving": "loopback, CDN URLs rewritten to vendored same-origin copies",
        "libs": {name: lib["sha256"] for name, lib in sorted(vendored.items())},
    }


def main(argv: list[str] | None = None) -> int:
    """Mechanics smoke: materialize, vendor, serve, boot, shim, readout,
    one input round trip. Print the readout as TAP-adjacent free text
    (the structured run rides the lane)."""
    import argparse

    parser = argparse.ArgumentParser(description="T-36 prototype mechanics smoke")
    parser.add_argument("--sac-bin", default="", help="the real sac binary (default: PATH)")
    parser.add_argument("--cache-dir", default="/tmp/t36-proto-cache")
    options = parser.parse_args(argv)
    try:
        materialized = materialize(pathlib.Path(options.cache_dir), options.sac_bin)
        vendored = vendor_libs(pathlib.Path(options.cache_dir))
        server = PrototypeServer(pathlib.Path(materialized["path"]).read_bytes(), vendored)
    except PrototypeRefused as error:
        print(f"prototype refusal: {error}", file=sys.stderr)
        return 2
    session = PrototypeSession(server)
    failures = 0
    try:
        session.start()
        readout = session.readout()
        checks = [
            ("prototype boots in WebKit", readout.get("booted") is True),
            ("suite states enumerated", len(readout.get("windows", [])) >= 8),
            ("instrument rows read", len(readout.get("rows", [])) >= 50),
            ("words read from SESSIONS", len(readout.get("words", [])) >= 5),
            ("board trail carries the saved location", (readout.get("trails", {}).get("board") or {}).get("location", "").startswith("t-3")),
            ("deck state read from the pool", "deck" in readout.get("trails", {}) or any(w["id"] == "deck" for w in readout.get("windows", []))),
        ]
        session.input(" ", " ")
        after = session.readout()
        checks.append(("picker opens on real input", after.get("picker", {}).get("open") is True))
        for name, ok in checks:
            print(f"[t36-prototype] {'ok' if ok else 'FAIL'}: {name}")
            failures += 0 if ok else 1
        print(json.dumps(prototype_record(materialized, vendored), indent=2, sort_keys=True))
    finally:
        session.stop()
        server.stop()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
