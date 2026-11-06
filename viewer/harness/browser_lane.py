#!/usr/bin/env python3
"""The browser lane payload: runs INSIDE the enforced sandbox, after the
containment canary has passed in the same sandbox, and never before.

Order, enforced here:
  1. containment canary (the reviewed module, invoked as main()) - a
     refusal ends the lane before any listener or driver exists
  2. a stdlib loopback static file server for the built frontend (the
     only listener this lane ever creates; no child process, no
     framework)
  3. one reused headless Chromium session via the Nix-pinned Playwright
     driver (driver and browser from the same nixpkgs revision; nothing
     downloaded at runtime); the page navigates to the server and waits
     for the shared viewer API before any action
  4. a bounded check list over the real frontend and the real fixture
     boards: DOM and inspection outcomes with actual readiness waits,
     screenshots saved under the run root for later inspection
  5. owned teardown: the server thread shuts down before exit

Everything here is inside the sandbox: the clipboard a browser writes
is Chromium's own clipboard, not the desktop's - native-clipboard
fidelity is a real-client question and is reported as a limit, never
implied.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
# The canary rides its own read-only bind; import it from there, never
# from this lane's directory.
sys.path.insert(0, "/canary")

import containment_canary  # noqa: E402

SERVER_PORT = 1421
DIST_SANDBOX = "/app/dist"
READY_SECONDS = 20

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    stream = sys.stdout if condition else sys.stderr
    print(f"[browser] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}", flush=True)


def run_canary_first() -> bool:
    """The reviewed canary, in-process, before anything listens."""
    return containment_canary.main() == 0


def start_server() -> ThreadingHTTPServer | None:
    """Stdlib loopback static server for the built frontend. A thread in
    this process - no child, no teardown protocol beyond shutdown()."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=DIST_SANDBOX)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", SERVER_PORT), handler)
    except OSError as error:
        print(f"containment refusal: loopback server could not start: {error}", file=sys.stderr)
        return None
    server.timeout = 1
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def run_checks(page, base_url: str, shots_dir: pathlib.Path) -> int:
    viewer = "window.__boardViewer"

    def open_board(name: str) -> None:
        page.evaluate(f"() => {viewer}.open('fixture', {name!r})")
        page.wait_for_function(
            f"() => {viewer}.inspection().resource.status === 'loaded'"
            f" || {viewer}.inspection().resource.status === 'invalid'"
        )

    def wait_rendered(kinds: tuple[str, ...]) -> list[dict]:
        """Actual readiness: every requested kind has a terminal outcome
        (ok or a named failure) in this generation, or a bounded timeout
        says so - never a fixed sleep, never just the first result."""
        kind_list = json.dumps(list(kinds))
        page.wait_for_function(
            f"""() => {{
                const outcomes = {viewer}.inspection().renderOutcomes;
                return {kind_list}.every(kind => outcomes.some(o => o.kind === kind));
            }}"""
        )
        return page.evaluate(f"() => {viewer}.inspection().renderOutcomes")

    # 1. the real migration board loads through the same handler the
    #    header control uses
    open_board("saccade-and-migration.yaml")
    inspection = page.evaluate(f"() => {viewer}.inspection()")
    check(
        "real migration board loads",
        inspection["resource"]["status"] == "loaded" and inspection["errors"] == [],
        str(inspection["resource"]),
    )
    check(
        "real board task metadata is authored-only",
        inspection["identity"]["task"] == "saccade-current-system-and-migration",
        str(inspection["identity"]),
    )

    # 2. actual rendering outcomes: math and diagrams hydrated
    outcomes = wait_rendered(("math", "mermaid"))
    math_ok = [o for o in outcomes if o["kind"] == "math" and o["status"] == "ok"]
    mermaid_ok = [o for o in outcomes if o["kind"] == "mermaid" and o["status"] == "ok"]
    math_bad = [o for o in outcomes if o["kind"] == "math" and o["status"] != "ok"]
    mermaid_bad = [o for o in outcomes if o["kind"] == "mermaid" and o["status"] != "ok"]
    check("migration board math renders (actual outcomes)", len(math_ok) > 0 and len(math_bad) == 0, str(outcomes[:4]))
    check("migration board diagrams render (actual outcomes)", len(mermaid_ok) > 0 and len(mermaid_bad) == 0, str(outcomes[:4]))
    containers = page.evaluate("() => document.querySelectorAll('mjx-container').length")
    check("math is in the DOM as MathJax containers", containers > 0, str(containers))
    page.screenshot(path=f"{shots_dir}/real-board.png")

    # 3. copy semantics: exact bytes for a deterministic selection on the
    #    repeated-math fixture - the formula emitted once, authored bytes
    #    in Source mode, TeX value in Plain mode. The selection starts at
    #    the first text of the paragraph that CONTAINS the first math
    #    expression and ends nine characters into the trailing text node
    #    after it (" in this "), matching the expected span exactly.
    #    First evidence toward full copy certification, not the
    #    certification itself.
    open_board("repeated-math.yaml")
    page.wait_for_function("() => document.querySelector('.block-text .math-inline mjx-container, .block-text .math-inline') !== null")
    page.evaluate("() => window.__boardViewer.setCopyMode('source')")

    def select_first_sentence_plus_math() -> None:
        page.evaluate("""() => {
            const math = document.querySelector('.block-text .math-inline');
            const paragraph = math.closest('p');
            const children = Array.from(paragraph.children);
            const mathIndex = children.indexOf(math.closest('.md-text') ?? math);
            const startEl = children[0];
            const endEl = children[mathIndex + 1];
            const range = document.createRange();
            range.setStart(startEl, 0);
            range.setEnd(endEl.firstChild, 9);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            document.execCommand('copy');
        }""")

    def copied_text() -> str:
        return page.evaluate("async () => navigator.clipboard.readText()")

    select_first_sentence_plus_math()
    source_copy = copied_text()
    expected_source = "First occurrence: $0 < \\eta < 2/b$ in this "
    check("Source copy emits the exact authored bytes once", source_copy == expected_source, repr(source_copy))
    check("the formula appears exactly once in the Source copy", source_copy.count("$0 < \\eta < 2/b$") == 1, repr(source_copy))

    page.evaluate("() => window.__boardViewer.setCopyMode('plain')")
    select_first_sentence_plus_math()
    plain_copy = copied_text()
    expected_plain = "First occurrence: 0 < \\eta < 2/b in this "
    check("Plain copy emits the exact TeX value", plain_copy == expected_plain, repr(plain_copy))
    page.screenshot(path=f"{shots_dir}/copy-modes.png")

    # 4. duplicate-expression search: two occurrences, both reachable
    page.fill("input[type=search]", "0 < \\eta < 2/b")
    page.wait_for_function("() => (document.querySelector('.search-state')?.textContent ?? '').trim().startsWith('1 / 2')")
    search_state = page.text_content(".search-state") or ""
    check("duplicate expression search finds two occurrences", search_state.strip().startswith("1 / 2"), repr(search_state))
    page.click("button[aria-label='next match']")
    page.wait_for_function("() => (document.querySelector('.search-state')?.textContent ?? '').trim().startsWith('2 / 2')")
    search_state = page.text_content(".search-state") or ""
    check("next match reaches the second occurrence", search_state.strip().startswith("2 / 2"), repr(search_state))

    # 5. theme and narrow width
    page.evaluate("() => window.__boardViewer.setTheme('dark')")
    theme = page.evaluate("() => document.documentElement.dataset.theme")
    check("dark theme applies to the document root", theme == "dark", str(theme))
    page.set_viewport_size({"width": 520, "height": 800})
    overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
    check("narrow window has no page-level horizontal overflow", overflow <= 0, str(overflow))
    page.screenshot(path=f"{shots_dir}/dark-narrow.png")

    # 6. expanded diagram controls and return
    open_board("large-diagram.yaml")
    page.wait_for_function("() => document.querySelector('.diagram-frame svg') !== null")
    page.click(".diagram-frame")
    page.wait_for_function("() => document.querySelector('.diagram-overlay') !== null")
    check("diagram expands into the overlay", True)
    page.click("[data-action='zoom-in']")
    page.wait_for_function("() => (document.querySelector('.diagram-zoom-label')?.textContent ?? '').trim() !== '100%'")
    zoom_label = page.text_content(".diagram-zoom-label") or ""
    check("zoom control raises the scale", zoom_label.strip() != "100%", repr(zoom_label))
    page.click("[data-action='reset']")
    page.wait_for_function("() => (document.querySelector('.diagram-zoom-label')?.textContent ?? '').trim() === '100%'")
    check("reset returns scale to 100%", True)
    page.screenshot(path=f"{shots_dir}/diagram-expanded.png")
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelector('.diagram-overlay') === null")
    check("closing returns to the board", True)
    page.evaluate("() => window.__boardViewer.setTheme('system')")

    # 7. invalid and missing inputs keep source visible with outcomes
    open_board("invalid.yaml")
    inspection = page.evaluate(f"() => {viewer}.inspection()")
    check(
        "invalid board is a structured error, never a board",
        inspection["resource"]["status"] == "invalid" and any(e["code"] == "yaml-syntax" for e in inspection["errors"]),
        str(inspection["errors"])[:200],
    )
    error_visible = page.evaluate("() => document.querySelector('.errors .render-error') !== null")
    check("invalid board error is rendered visibly", error_visible is True)

    open_board("missing-asset.yaml")
    page.wait_for_function(
        f"() => {viewer}.inspection().renderOutcomes.some(o => o.kind === 'figure' && o.status === 'missing')"
    )
    outcomes = page.evaluate(f"() => {viewer}.inspection().renderOutcomes")
    missing = [o for o in outcomes if o["kind"] == "figure" and o["status"] == "missing"]
    check("missing figure is an actual reported outcome", len(missing) == 1, str(outcomes))
    missing_visible = page.evaluate("() => document.querySelector('.figure-frame.asset-missing .render-error') !== null")
    check("missing figure error is rendered visibly", missing_visible is True)
    page.screenshot(path=f"{shots_dir}/missing-asset.png")

    return sum(1 for ok, _name, _detail in RESULTS if not ok)


def main() -> int:
    started = time.monotonic()
    if not run_canary_first():
        return 1

    shots_raw = os.environ.get("SHOTS_DIR", "")
    if not shots_raw.startswith("/"):
        print("containment refusal: SHOTS_DIR must be an absolute run-root path", file=sys.stderr)
        return 1
    shots_dir = pathlib.Path(shots_raw)
    shots_dir.mkdir(parents=True, exist_ok=True)

    server = start_server()
    if server is None:
        return 1
    base_url = f"http://127.0.0.1:{SERVER_PORT}"
    print(f"[browser] serving {DIST_SANDBOX} on {base_url}", flush=True)

    failures: int
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, chromium_sandbox=False)
            context = browser.new_context(viewport={"width": 1100, "height": 800})
            # Chromium-internal clipboard only; never the desktop's.
            context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
            page = context.new_page()
            page.set_default_timeout(10000)
            page.goto(base_url)
            page.wait_for_function("() => window.__boardViewer !== undefined")
            failures = run_checks(page, base_url, shots_dir)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

    cost = time.monotonic() - started
    print(f"[browser] {len(RESULTS) - failures}/{len(RESULTS)} checks passed in {cost:.1f}s", flush=True)
    print(f"[browser] screenshots saved under {shots_dir} for inspection", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
