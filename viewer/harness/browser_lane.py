#!/usr/bin/env python3
"""The browser lane payload: runs INSIDE the enforced sandbox, after the
containment canary has passed in the same sandbox, and never before.

Order, enforced here:
  1. containment canary (the reviewed module, invoked as main()) - a
     refusal ends the lane before any listener or driver exists
  2. a stdlib loopback static file server for the built frontend (the
     only listener this lane ever creates; no child process, no
     framework)
  3. two engine passes through the SAME check list, one reused headless
     browser per engine over the Nix-pinned Playwright driver (driver
     and browsers from the same nixpkgs revision; the pinned browsers
     store path carries WebKit; nothing downloaded at runtime):
     Chromium first, then Playwright WebKit - the pinned WebKit
     launches its MiniBrowser directly because pw_run.sh needs PATH
     tools the fresh sandbox PATH deliberately does not carry; a second
     engine on identical checks so a Chromium-only pass cannot ship a
     WebKit-only defect again; the page navigates to the server and
     waits for the shared viewer API before any action
  4. per-engine outcomes over the real frontend and the real fixture
     boards: DOM and inspection outcomes with actual readiness waits,
     screenshots saved under the run root for later inspection, names
     prefixed by engine
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


def record_check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    print(f"[browser] {'ok' if condition else 'FAIL'}: {name} {detail if not condition else ''}", flush=True)


def pinned_webkit_executable() -> str:
    """The pinned driver resolves WebKit through pw_run.sh, which needs
    PATH tools (dirname, uname) the fresh sandbox PATH deliberately does
    not carry. The nix-built browsers dir ships the same MiniBrowser the
    script would exec in its zip-layout arm, so the lane launches it
    directly. Exactly one webkit-* directory under the pinned browsers
    path is the pinned contract - zero or several is a refusal, never a
    guess."""
    browsers = pathlib.Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"])
    matches = sorted(browsers.glob("webkit-*/minibrowser-wpe/bin/MiniBrowser"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one pinned webkit MiniBrowser under {browsers}, found {len(matches)}"
        )
    return str(matches[0])


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


def run_checks(page, base_url: str, shots_dir: pathlib.Path, engine: str) -> int:
    started_records = len(RESULTS)
    viewer = "window.__boardViewer"

    def check(name: str, condition: bool, detail: str = "") -> None:
        record_check(f"[{engine}] {name}", condition, detail)

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

    def screenshot_at(selector: str, name: str) -> None:
        """Capture at a reading position, not the page top."""
        try:
            page.evaluate(
                f"""() => {{
                    const el = document.querySelector({selector!r});
                    if (el) el.scrollIntoView({{block: 'center'}});
                }}"""
            )
            page.wait_for_timeout(150)
        except Exception:
            pass
        page.screenshot(path=f"{shots_dir}/{engine}-{name}")

    def phase(name: str, body) -> None:
        """One failed phase records its failure and never starves the
        remaining independent checks."""
        try:
            body()
        except Exception as error:  # playwright timeouts, navigation errors
            check(f"{name}", False, f"{type(error).__name__}: {str(error)[:160]}")

    # phase 1: the real migration board through the shared handler
    def phase_real_board() -> None:
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
        outcomes = wait_rendered(("math", "mermaid"))
        math_ok = [o for o in outcomes if o["kind"] == "math" and o["status"] == "ok"]
        mermaid_ok = [o for o in outcomes if o["kind"] == "mermaid" and o["status"] == "ok"]
        math_bad = [o for o in outcomes if o["kind"] == "math" and o["status"] != "ok"]
        mermaid_bad = [o for o in outcomes if o["kind"] == "mermaid" and o["status"] != "ok"]
        check("migration board math renders (actual outcomes)", len(math_ok) > 0 and len(math_bad) == 0, str(outcomes[:4]))
        check("migration board diagrams render (actual outcomes)", len(mermaid_ok) > 0 and len(mermaid_bad) == 0, str(outcomes[:4]))
        containers = page.evaluate("() => document.querySelectorAll('mjx-container').length")
        check("math is in the DOM as MathJax containers", containers > 0, str(containers))
        diagrams = page.locator(".mermaid-block .diagram-frame > svg").count()
        check("successful diagrams remain mounted", diagrams == len(mermaid_ok) and diagrams > 0, str(diagrams))
        screenshot_at("mjx-container", "real-board-math.png")
        screenshot_at(".diagram-frame", "real-board-diagram.png")

    phase("real board load and render", phase_real_board)

    # phases 2-4: copy semantics, each step independent and bounded - a
    # failed copy check records its failure and never starves the rest.
    def start_clipboard_read() -> None:
        page.evaluate("""() => {
            window.__copy = {state: 'pending'};
            navigator.clipboard.readText()
                .then(t => { window.__copy = {state: 'done', text: t}; })
                .catch(e => { window.__copy = {state: 'error', message: String(e)}; });
        }""")

    def await_clipboard() -> tuple[bool, str]:
        timed_out = False
        try:
            page.wait_for_function("() => window.__copy && window.__copy.state !== 'pending'", timeout=8000)
        except Exception:
            timed_out = True
        state = page.evaluate("() => window.__copy")
        if timed_out or state.get("state") != "done":
            return False, f"state={state} timed_out={timed_out}"
        return True, state.get("text") or ""

    def select_first_formula_paragraph() -> None:
        page.evaluate("""() => {
            const math = document.querySelector('.block-text .math-inline');
            const paragraph = math.closest('p');
            const children = Array.from(paragraph.children);
            const mathIndex = children.indexOf(math);
            const range = document.createRange();
            range.setStart(children[0], 0);
            range.setEnd(children[mathIndex + 1].firstChild, 9);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            const ok = document.execCommand('copy');
            window.__exec = ok;
        }""")

    def phase_copy_source() -> None:
        open_board("repeated-math.yaml")
        page.wait_for_function("() => document.querySelector('.block-text .math-inline') !== null")
        mounted = page.query_selector(".block-text")
        page.evaluate("() => window.__boardViewer.setCopyMode('source')")
        check(
            "copy mode preserves the mounted document",
            mounted is not None and mounted.evaluate("el => el.isConnected && el === document.querySelector('.block-text')"),
        )
        focus = page.evaluate("() => document.hasFocus()")
        page.evaluate("() => { window.__copy = {state: 'pending', focus: document.hasFocus()}; }")
        select_first_formula_paragraph()
        start_clipboard_read()
        exec_result = page.evaluate("() => window.__exec")
        print(f"[browser] copy phase: focus={focus} execCommand={exec_result}", flush=True)
        ok, evidence = await_clipboard()
        if not ok:
            check("clipboard read reaches a terminal state (bounded)", False, evidence)
            page.screenshot(path=f"{shots_dir}/{engine}-copy-timeout.png")
            return
        source_copy = evidence
        expected_source = "First occurrence: $0 < \\eta < 2/b$ in this "
        check("Source copy emits the exact authored bytes once", source_copy == expected_source, repr(source_copy))
        check("the formula appears exactly once in the Source copy", source_copy.count("$0 < \\eta < 2/b$") == 1, repr(source_copy))
        screenshot_at(".block-text .math-inline", "copy-modes.png")

    phase("copy source exact", phase_copy_source)

    def phase_copy_plain() -> None:
        open_board("repeated-math.yaml")
        page.wait_for_function("() => document.querySelector('.block-text .math-inline') !== null")
        page.evaluate("() => window.__boardViewer.setCopyMode('plain')")
        page.evaluate("() => { window.__copy = {state: 'pending'}; }")
        select_first_formula_paragraph()
        start_clipboard_read()
        ok, evidence = await_clipboard()
        if not ok:
            check("plain clipboard read reaches a terminal state (bounded)", False, evidence)
            return
        plain_copy = evidence
        expected_plain = "First occurrence: 0 < \\eta < 2/b in this "
        check("Plain copy emits the exact TeX value", plain_copy == expected_plain, repr(plain_copy))
        screenshot_at(".block-text .math-inline", "copy-plain.png")

    phase("copy plain exact", phase_copy_plain)

    def phase_copy_multi_and_flow() -> None:
        # multi-unit: from the first formula through the second, two math
        # units with prose between - partial evidence: each authored
        # formula exactly once and the connecting prose present
        open_board("repeated-math.yaml")
        page.wait_for_function("() => document.querySelectorAll('.block-text .math-inline').length >= 2")
        page.evaluate("() => window.__boardViewer.setCopyMode('source')")
        page.evaluate("() => { window.__copy = {state: 'pending'}; }")
        page.evaluate("""() => {
            const math = document.querySelectorAll('.block-text .math-inline');
            const range = document.createRange();
            range.setStartBefore(math[0]);
            range.setEndAfter(math[1]);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            document.execCommand('copy');
            navigator.clipboard.readText()
                .then(t => { window.__copy = {state: 'done', text: t}; })
                .catch(e => { window.__copy = {state: 'error', message: String(e)}; });
        }""")
        ok, evidence = await_clipboard()
        if ok:
            check(
                "multi-unit copy keeps each formula exactly once",
                evidence.count("$0 < \\eta < 2/b$") == 2 and "Second occurrence" in evidence,
                repr(evidence[:120]),
            )
        else:
            check("multi-unit copy keeps each formula exactly once", False, evidence)

        # flow math: whole fenced display element -> exact authored bytes
        open_board("display-scroll.yaml")
        page.wait_for_function("() => document.querySelectorAll('.math-display').length >= 1")
        page.evaluate("() => { window.__copy = {state: 'pending'}; }")
        page.evaluate("""() => {
            const display = document.querySelectorAll('.math-display')[1];
            const range = document.createRange();
            range.selectNodeContents(display);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            document.execCommand('copy');
            navigator.clipboard.readText()
                .then(t => { window.__copy = {state: 'done', text: t}; })
                .catch(e => { window.__copy = {state: 'error', message: String(e)}; });
        }""")
        ok, evidence = await_clipboard()
        check(
            "flow math Source copy is the exact authored fence",
            ok and evidence == "$$\n\\frac{a}{b}\n$$",
            repr(evidence[:120]),
        )
        screenshot_at(".math-display", "flow-math-copy.png")

    phase("copy multi-unit and flow math", phase_copy_multi_and_flow)

    # phase 3: duplicate-expression search, both occurrences reachable
    def phase_search() -> None:
        open_board("repeated-math.yaml")
        page.fill("input[type=search]", "0 < \\eta < 2/b")
        page.wait_for_function("() => (document.querySelector('.search-state')?.textContent ?? '').trim().startsWith('1 / 2')")
        search_state = page.text_content(".search-state") or ""
        check("duplicate expression search finds two occurrences", search_state.strip().startswith("1 / 2"), repr(search_state))
        page.click("button[aria-label='next match']")
        page.wait_for_function("() => (document.querySelector('.search-state')?.textContent ?? '').trim().startsWith('2 / 2')")
        search_state = page.text_content(".search-state") or ""
        check("next match reaches the second occurrence", search_state.strip().startswith("2 / 2"), repr(search_state))
        screenshot_at(".block-text .math-inline", "search-occurrences.png")

    phase("duplicate search", phase_search)

    # phase 4: theme and narrow width
    def phase_theme() -> None:
        page.evaluate("() => window.__boardViewer.setTheme('dark')")
        theme = page.evaluate("() => document.documentElement.dataset.theme")
        check("dark theme applies to the document root", theme == "dark", str(theme))
        page.set_viewport_size({"width": 520, "height": 800})
        overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        check("narrow window has no page-level horizontal overflow", overflow <= 0, str(overflow))
        screenshot_at(".block-text", "dark-narrow.png")
        page.evaluate("() => window.__boardViewer.setTheme('system')")
        page.set_viewport_size({"width": 1100, "height": 800})

    phase("theme and width", phase_theme)

    # phase 5: expanded diagram controls and return
    def phase_diagram() -> None:
        open_board("large-diagram.yaml")
        page.wait_for_function("() => document.querySelector('.diagram-frame svg') !== null")
        page.click(".diagram-frame")
        page.wait_for_function("() => document.querySelector('.diagram-overlay') !== null")
        check("diagram expands into the overlay", True)

        def rendered() -> dict:
            # the rendered truth of the expanded diagram: its painted
            # bounding box and the computed CSS transform - the zoom
            # label alone hid the WebKit-ignored outer-svg transform
            return page.evaluate(
                """() => {
                    const svg = document.querySelector('.diagram-stage svg');
                    const rect = svg.getBoundingClientRect();
                    const style = getComputedStyle(svg).transform;
                    const m = new DOMMatrixReadOnly(style === 'none' ? '' : style);
                    return {x: rect.x, y: rect.y, width: rect.width, height: rect.height, scale: m.a, tx: m.m41};
                }"""
            )

        baseline = rendered()
        page.click("[data-action='zoom-in']")
        page.wait_for_function("() => (document.querySelector('.diagram-zoom-label')?.textContent ?? '').trim() !== '100%'")
        zoom_label = page.text_content(".diagram-zoom-label") or ""
        check("zoom control raises the scale", zoom_label.strip() != "100%", repr(zoom_label))
        zoomed = rendered()
        check(
            "zoom-in changes the rendered diagram geometry, not just the label",
            abs(zoomed["width"] - baseline["width"] * 1.25) <= baseline["width"] * 0.02
            and abs(zoomed["height"] - baseline["height"] * 1.25) <= baseline["height"] * 0.02,
            f"baseline={baseline} zoomed={zoomed}",
        )
        check(
            "zoom-in computes a real CSS transform on the expanded diagram",
            zoomed["scale"] is not None and abs(zoomed["scale"] - 1.25) < 0.02,
            str(zoomed),
        )

        # a real drag must move the painted diagram; the drag distance
        # lands on the computed matrix divided by the zoom scale
        stage = page.locator(".diagram-stage").bounding_box()
        cx = stage["x"] + stage["width"] / 2
        cy = stage["y"] + stage["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + 80, cy + 60, steps=4)
        page.mouse.up()
        panned = rendered()
        check(
            "pan moves the rendered diagram",
            abs(panned["x"] - zoomed["x"]) > 20 and abs(panned["y"] - zoomed["y"]) > 20,
            f"zoomed={zoomed} panned={panned}",
        )
        check(
            "pan keeps the zoom and lands on the computed transform",
            panned["scale"] is not None
            and abs(panned["scale"] - 1.25) < 0.02
            and abs(panned["tx"] - zoomed["tx"] - 64) < 8,
            f"zoomed={zoomed} panned={panned}",
        )

        page.click("[data-action='reset']")
        page.wait_for_function("() => (document.querySelector('.diagram-zoom-label')?.textContent ?? '').trim() === '100%'")
        check("reset returns scale to 100%", True)
        reset_geo = rendered()
        check(
            "reset restores the rendered geometry",
            abs(reset_geo["width"] - baseline["width"]) <= baseline["width"] * 0.02
            and abs(reset_geo["height"] - baseline["height"]) <= baseline["height"] * 0.02,
            f"baseline={baseline} reset={reset_geo}",
        )
        page.screenshot(path=f"{shots_dir}/{engine}-diagram-expanded.png")
        page.keyboard.press("Escape")
        page.wait_for_function("() => document.querySelector('.diagram-overlay') === null")
        check("closing returns to the board", True)

    phase("diagram expansion", phase_diagram)

    # phase 6: invalid and missing inputs keep source visible with outcomes
    def phase_invalid() -> None:
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
        screenshot_at(".figure-frame", "missing-asset.png")

    phase("invalid and missing inputs", phase_invalid)

    return sum(1 for ok, _name, _detail in RESULTS[started_records:] if not ok)


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

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            # one reused browser per engine; the identical check list
            # runs under each - engine differences are the evidence
            for engine in ("chromium", "webkit"):
                try:
                    if engine == "chromium":
                        browser = playwright.chromium.launch(headless=True, chromium_sandbox=False)
                    else:
                        # env replaces the inherited environment, so the
                        # pinned complex-text flag rides a merged copy;
                        # pw_run.sh sets the same one before its exec
                        launch_env = dict(os.environ)
                        launch_env["WEBKIT_FORCE_COMPLEX_TEXT"] = "1"
                        browser = playwright.webkit.launch(
                            headless=True,
                            executable_path=pinned_webkit_executable(),
                            env=launch_env,
                        )
                except Exception as error:
                    # the browser's own stderr is the evidence a dead
                    # launch leaves; a truncated detail once hid the
                    # actual cause
                    record_check(
                        f"[{engine}] headless browser launches from the pinned driver",
                        False,
                        f"{type(error).__name__}: {error}",
                    )
                    continue
                context = browser.new_context(viewport={"width": 1100, "height": 800})
                try:
                    # Engine-internal clipboard only; never the desktop's.
                    # An engine that cannot grant these lets the copy
                    # checks measure what it actually does - masking the
                    # grant would defeat the parity pass.
                    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
                except Exception as error:
                    print(f"[browser] {engine}: clipboard permission grant unavailable: {type(error).__name__}: {str(error)[:120]}", flush=True)
                page = context.new_page()
                page.set_default_timeout(10000)
                page.goto(base_url)
                page.wait_for_function("() => window.__boardViewer !== undefined")
                run_checks(page, base_url, shots_dir, engine)
                browser.close()
    finally:
        server.shutdown()
        server.server_close()

    # every recorded failure counts, including an engine whose browser
    # never launched - the lane passes only when BOTH engines did
    failed = sum(1 for ok, _name, _detail in RESULTS if not ok)
    cost = time.monotonic() - started
    print(f"[browser] {len(RESULTS) - failed}/{len(RESULTS)} checks passed in {cost:.1f}s", flush=True)
    print(f"[browser] screenshots saved under {shots_dir} for inspection", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
