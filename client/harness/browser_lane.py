#!/usr/bin/env python3
"""The browser lane payload: runs INSIDE the enforced sandbox, after the
containment canary has passed in the same sandbox, and never before.

Order, enforced here:
  1. containment canary (the reviewed module, invoked as main()) - a
     refusal ends the lane before any listener or driver exists
  2. a stdlib loopback static file server for the built frontend (the
     only listener this lane ever creates; no child process, no
     framework)
  3. the WebKit pass - the gate and the default engine: one reused
     headless WPE browser over the Nix-pinned Playwright driver (driver
     and browsers from the same nixpkgs revision; nothing downloaded at
     runtime), launched through a tiny stderr-capturing wrapper because
     pw_run.sh needs PATH tools the fresh sandbox PATH deliberately
     does not carry; its EGL is display-less (pinned mesa vendor dir,
     forced software rendering, EGL_PLATFORM=surfaceless); a WPE grant
     refusal can take the first browser down, so the setup relaunches
     once and runs the pass ungranted; the page navigates to the server
     and waits for the shared viewer API before any action
  4. per-engine outcomes over the real frontend and the real fixture
     boards: DOM and inspection outcomes with actual readiness waits,
     screenshots saved under the run root for later inspection, names
     prefixed by engine; the copy checks observe the payload the
     product's copy handler writes into the copy event (a window-bubble
     listener reads clipboardData after the product handler ran), not a
     system-clipboard round-trip; Chromium is not in the default run -
     BOARD_VIEWER_BROWSER_TRIAGE_CHROMIUM adds it as a triage pass
     whose results never gate the exit
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
sys.path.insert(0, "/mock")

import containment_canary  # noqa: E402
import tracker_mock  # noqa: E402

SERVER_PORT = 1421
DIST_SANDBOX = "/app/dist"
# the env-gated reference page, bound by the entry point at these fixed
# sandbox paths and served by the loopback server: the page at the
# reference route, its sibling reference-assets/ beside it (the mockup
# vendors its pinned CDN libraries there; a scrubbed sandbox has no
# mime database and no network, so a file:// page renders as plain
# source and CDN scripts never arrive)
REFERENCE_SANDBOX_PATH = "/app/reference.html"
REFERENCE_ROUTE = "/reference.html"
REFERENCE_ASSETS_SANDBOX_PATH = "/app/reference-assets"
REFERENCE_ASSETS_ROUTE = "/reference-assets"
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


def webkit_stderr_capture() -> str:
    """The driver swallows WPE stderr when the browser dies mid-run, so
    the browser writes its own stderr through a tiny wrapper into the
    run root - the abort message behind a TargetClosedError must be
    readable evidence, not a silent close. stdout stays on the pipe:
    the driver reads the inspector handshake from it."""
    real = pinned_webkit_executable()
    # the pinned MiniBrowser wrapper's shebang names the pinned bash
    bash = pathlib.Path(real).read_text().splitlines()[0].removeprefix("#!").strip().split()[0]
    run_artifacts = pathlib.Path(os.environ["SHOTS_DIR"])
    err_log = run_artifacts / "webkit-browser-stderr.log"
    wrapper = run_artifacts / "webkit-minibrowser.sh"
    wrapper.write_text(
        f"#!{bash}\n"
        f'exec 2>>"{err_log}"\n'
        f'exec "{real}" "$@"\n'
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def launch_engine(playwright, engine: str):
    if engine == "chromium":
        return playwright.chromium.launch(headless=True, chromium_sandbox=False)
    # env replaces the inherited environment, so the browser-specific
    # flags ride a merged copy: WEBKIT_FORCE_COMPLEX_TEXT is what
    # pw_run.sh sets before its exec; EGL_PLATFORM=surfaceless is the
    # pinned Mesa's only display-less platform, whose ForceSoftware path
    # runs swrast without DRM (dri2_initialize_surfaceless in the pinned
    # source) - the vendor dir and LIBGL_ALWAYS_SOFTWARE arrive through
    # the sandbox env
    launch_env = dict(os.environ)
    launch_env["WEBKIT_FORCE_COMPLEX_TEXT"] = "1"
    launch_env["EGL_PLATFORM"] = "surfaceless"
    return playwright.webkit.launch(
        headless=True,
        executable_path=webkit_stderr_capture(),
        env=launch_env,
    )


def setup_engine_page(playwright, engine: str, base_url: str):
    """One reused browser per engine, with one documented exception: the
    clipboard grant is best-effort, and a WPE grant refusal surfaces
    lazily at page creation and takes the first browser down - the
    evidence prints, then the pass runs ungranted on a fresh launch.
    Every setup failure lands as a named per-engine check."""
    try:
        browser = launch_engine(playwright, engine)
    except Exception as error:
        record_check(
            f"[{engine}] headless browser launches from the pinned driver",
            False,
            f"{type(error).__name__}: {error}",
        )
        return None
    try:
        context = browser.new_context(viewport={"width": 1100, "height": 800})
        # Engine-internal clipboard only; never the desktop's. The grant
        # can be refused lazily at page creation, fatally to the browser;
        # the copy checks then measure the engine's real behavior.
        context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
        return browser, context.new_page()
    except Exception as error:
        print(f"[browser] {engine}: clipboard permission grant unavailable: {type(error).__name__}: {str(error)[:160]}", flush=True)
    try:
        browser.close()
    except Exception:
        pass
    try:
        browser = launch_engine(playwright, engine)
        context = browser.new_context(viewport={"width": 1100, "height": 800})
        return browser, context.new_page()
    except Exception as error:
        record_check(
            f"[{engine}] headless page setup from the pinned driver",
            False,
            f"{type(error).__name__}: {error}",
        )
        return None


def run_canary_first() -> bool:
    """The reviewed canary, in-process, before anything listens."""
    return containment_canary.main() == 0


def start_server() -> ThreadingHTTPServer | None:
    """Stdlib loopback static server for the built frontend, plus the
    env-gated reference route: the bound reference file is served from
    its sandbox path with an explicit text/html type, because a scrubbed
    sandbox has no mime database and WebKit renders a file:// page as
    plain source. A thread in this process - no child, no teardown
    protocol beyond shutdown()."""

    class Handler(SimpleHTTPRequestHandler):
        def translate_reference(self, path: str):
            """A reference-assets request maps into the bound reference
            directory; anything else under the reference prefix is a 404
            like any unknown path."""
            if not path.startswith(REFERENCE_ASSETS_ROUTE + "/"):
                return None
            name = path[len(REFERENCE_ASSETS_ROUTE) + 1:]
            if "/" in name or ".." in name or name.startswith("."):
                return None
            candidate = pathlib.Path(REFERENCE_ASSETS_SANDBOX_PATH) / name
            return candidate if candidate.is_file() else None

        def do_GET(self):
            route = self.path.split("?")[0]
            if route == REFERENCE_ROUTE and os.path.isfile(REFERENCE_SANDBOX_PATH):
                data = pathlib.Path(REFERENCE_SANDBOX_PATH).read_bytes()
                ctype = "text/html; charset=utf-8"
            elif (asset := self.translate_reference(route)) is not None:
                data = asset.read_bytes()
                ctype = "text/javascript" if asset.suffix == ".js" else "application/octet-stream"
            else:
                super().do_GET()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    handler = functools.partial(Handler, directory=DIST_SANDBOX)
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

    def install_copy_observer() -> None:
        """Observe the payload the product's copy handler writes into
        the copy event: a window-bubble listener runs after the
        product's document handler (bubble order), so the assertions
        measure the product's serialized bytes directly - the
        system-clipboard round-trip a WPE engine denies ungranted is
        not part of the check."""
        page.evaluate("""() => {
            window.__copy = {state: 'idle'};
            window.__copyObserver = (event) => {
                window.__copy = {
                    state: 'done',
                    text: event.clipboardData ? event.clipboardData.getData('text/plain') : '',
                };
            };
            window.addEventListener('copy', window.__copyObserver);
        }""")

    install_copy_observer()

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
            inspection["identity"]["task"] == "t-18",
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
        page.evaluate("() => { window.__copy = {state: 'pending'}; }")
        select_first_formula_paragraph()
        exec_result = page.evaluate("() => window.__exec")
        print(f"[browser] copy phase: focus={focus} execCommand={exec_result}", flush=True)
        ok, evidence = await_clipboard()
        if not ok:
            check("copy event reaches the observer (bounded)", False, evidence)
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
        ok, evidence = await_clipboard()
        if not ok:
            check("plain copy event reaches the observer (bounded)", False, evidence)
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
        }""")
        ok, evidence = await_clipboard()
        check(
            "flow math Source copy is the exact authored fence",
            ok and evidence == "$$\n\\frac{a}{b}\n$$",
            repr(evidence[:120]),
        )
        screenshot_at(".math-display", "flow-math-copy.png")

    phase("copy multi-unit and flow math", phase_copy_multi_and_flow)

    # phase 3: duplicate-expression search, both occurrences reachable;
    # the search bar is on demand now, so the phase opens it with the
    # real Ctrl+F keystroke first
    def phase_search() -> None:
        open_board("repeated-math.yaml")
        page.keyboard.press("Control+f")
        page.wait_for_function("() => document.querySelector('input[type=search]') !== null")
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

    # phase 7: the grammar version range over the real frontend - the
    # supported range is the viewer's declared claim, an unknown kind
    # renders as raw source, an unknown major version and a board with
    # no grammar field refuse under their own codes
    def phase_grammar() -> None:
        inspection = page.evaluate(f"() => {viewer}.inspection()")
        check(
            "the viewer declares exactly grammar 1",
            inspection["identity"]["supportedGrammarVersions"] == [1],
            str(inspection["identity"]),
        )

        open_board("unknown-kind.yaml")
        inspection = page.evaluate(f"() => {viewer}.inspection()")
        check(
            "an unknown kind loads the board with a diagnostic",
            inspection["resource"]["status"] == "loaded"
            and inspection["identity"]["grammar"] == 1
            and any(d["code"] == "unknown-kind" for d in inspection["diagnostics"]),
            str(inspection["diagnostics"])[:200],
        )
        raw_visible = page.evaluate("() => document.querySelector('.block-unknown pre.md-code') !== null")
        check("the unknown component renders as raw source", raw_visible is True)
        screenshot_at(".block-unknown", "unknown-kind.png")

        open_board("grammar-2.yaml")
        inspection = page.evaluate(f"() => {viewer}.inspection()")
        check(
            "an unknown major version refuses under its own code",
            inspection["resource"]["status"] == "invalid"
            and any(e["code"] == "grammar-version" for e in inspection["errors"]),
            str(inspection["errors"])[:200],
        )

        open_board("missing-grammar.yaml")
        inspection = page.evaluate(f"() => {viewer}.inspection()")
        check(
            "a board with no grammar field refuses, not legacy",
            inspection["resource"]["status"] == "invalid"
            and any(e["code"] == "missing-key" for e in inspection["errors"]),
            str(inspection["errors"])[:200],
        )

    phase("grammar version range", phase_grammar)

    # phase 8: tracker-named locations over the tracker contract stand-in
    # (harness/tracker_mock.py): the one server setting, the root-page
    # handshake, the unknown-kind lenient render, the unsupported major,
    # the wrong-tracker refusal, and the citation link navigating to the
    # cited revision. The real wire is pinned on the server side by the
    # cargo api tests; this phase pins the viewer's half.
    def phase_tracker() -> None:
        tracker = tracker_mock.TrackerMock("smoke", allowed_origins=[base_url])
        tracker.start()
        future = tracker_mock.TrackerMock("future", min_version=2, max_version=2, allowed_origins=[base_url])
        future.start()
        try:
            save_server = lambda value: (  # noqa: E731
                page.fill('[data-control="server:host"]', value),
                page.click('[data-control="server:save"]'),
            )

            def open_location(spec: str) -> dict:
                page.fill('[data-control="open:host-path"]', spec)
                page.click('[data-control="open:host"]')
                page.wait_for_function(
                    f"() => ['loaded','invalid'].includes({viewer}.inspection().resource.status)"
                )
                return page.evaluate(f"() => {viewer}.inspection()")

            save_server(f"127.0.0.1:{tracker.port}")
            inspection = open_location("smoke/t-0")
            check(
                "a tracker location loads through the server",
                inspection["resource"]["status"] == "loaded"
                and inspection["identity"]["task"] == "t-0"
                and inspection["resource"]["machine"] == "tracker:smoke",
                str(inspection["resource"]),
            )

            page.wait_for_selector('[data-control="board:open-citation"]')
            label = page.text_content('[data-control="board:open-citation"]')
            check(
                "the incorporation citation resolves to t-1@r1",
                label == "t-1@r1",
                str(label),
            )
            page.click('[data-control="board:open-citation"]')
            try:
                page.wait_for_function(
                    f"() => {viewer}.inspection().resource.path === 't-1@r1'", timeout=5000
                )
            except Exception:
                state = page.evaluate(f"() => {viewer}.inspection()")
                raise AssertionError(f"the citation click did not open the cited revision: {state}")
            cited = page.evaluate(f"() => {viewer}.inspection()")
            check(
                "the citation link opens the cited revision",
                cited["resource"]["status"] == "loaded"
                and cited["identity"]["task"] == "t-1"
                and any(d["code"] == "unknown-kind" for d in cited["diagnostics"]),
                str(cited["resource"]),
            )
            raw_visible = page.evaluate("() => document.querySelector('.block-unknown pre.md-code') !== null")
            check("the cited unknown kind renders as raw source", raw_visible is True)
            screenshot_at(".block-unknown", "tracker-cited-unknown-kind.png")

            inspection = open_location("smoke/t-2")
            check(
                "an unsupported major version refuses with its range named",
                inspection["resource"]["status"] == "invalid"
                and any(e["code"] == "grammar-version" for e in inspection["errors"]),
                str(inspection["errors"])[:200],
            )

            inspection = open_location("other/t-0")
            check(
                "a wrong tracker name refuses at the version handshake",
                inspection["resource"]["status"] == "invalid"
                and any("serves tracker 'smoke'" in e["message"] for e in inspection["errors"]),
                str(inspection["errors"])[:200],
            )

            save_server(f"127.0.0.1:{future.port}")
            inspection = open_location("future/t-0")
            check(
                "a server range excluding the viewer refuses before any board read",
                inspection["resource"]["status"] == "invalid"
                and any("move the binaries in step" in e["message"] for e in inspection["errors"]),
                str(inspection["errors"])[:200],
            )
            save_server(f"127.0.0.1:{tracker.port}")
            inspection = open_location("/some/research-file.md")
            check(
                "a bare path refuses where no Herdr core exists",
                inspection["resource"]["status"] == "invalid"
                and any("tracker's machine" in e["message"] for e in inspection["errors"]),
                str(inspection["errors"])[:200],
            )
        finally:
            tracker.stop()
            future.stop()

    phase("tracker locations", phase_tracker)

    # phase 9: the instruments (t-14 step 3) - the pipeline page and
    # the glance over the same stand-in: the sections and rows render,
    # select-first puts labels on the rows, Enter goes through the
    # link, the menu's e shows the activity (the lane and its events),
    # the agents' live join sections by the injected word (the lanes'
    # inspection setter - the browser lane has no host), and the
    # copies name what they carry. Real key events over the real
    # frontend, like the interaction phase.
    def phase_instruments() -> None:
        tracker = tracker_mock.TrackerMock("smoke", allowed_origins=[base_url])
        tracker.start()
        try:
            page.fill('[data-control="server:host"]', f"127.0.0.1:{tracker.port}")
            page.click('[data-control="server:save"]')

            def open_location(spec: str) -> dict:
                page.fill('[data-control="open:host-path"]', spec)
                page.click('[data-control="open:host"]')
                page.wait_for_function(
                    f"() => ['loaded','invalid'].includes({viewer}.inspection().resource.status)"
                )
                return page.evaluate(f"() => {viewer}.inspection()")

            inspection = open_location("smoke/tasks")
            check(
                "the pipeline page opens as the tasks instrument",
                inspection["resource"]["status"] == "loaded"
                and inspection["resource"]["path"] == "tasks"
                and inspection["resource"]["machine"] == "tracker:smoke",
                str(inspection["resource"]),
            )
            sections = page.eval_on_selector_all(
                ".instrument-label", "els => els.map(e => e.textContent)"
            )
            check(
                "the four recorded states section the view, in order",
                sections == ["waiting", "in flight", "awaiting judgment", "concluded last 24h"],
                str(sections),
            )
            rows = page.eval_on_selector_all(
                '[data-control="instrument:row"]', "els => els.map(e => e.getAttribute('data-task'))"
            )
            check(
                "the rows are task-identifiable and carry their links",
                "t-1" in rows and "t-0" in rows,
                str(rows),
            )
            heads = page.eval_on_selector_all(
                ".instrument-head", "els => els.map(e => e.textContent.trim())"
            )
            check(
                "the rows share one grid, the columns named once per section",
                all("tasktitleagentslatestcommitage" in "".join(h.split()) for h in heads),
                str(heads)[:160],
            )
            # select first: s puts labels on what can be selected,
            # rows included; a click selects a row, Enter goes through
            # the link
            page.keyboard.press("s")
            page.wait_for_selector(".select-label")
            chips = page.eval_on_selector_all(
                ".select-label",
                "els => els.map(e => ({text: e.textContent.trim(), block: e.getAttribute('data-target-block')}))",
            )
            row_chips = [c for c in chips if (c.get("block") or "").startswith("instrument-row-")]
            check(
                "s puts labels on the rows",
                len(row_chips) >= 2,
                str(chips)[:160],
            )
            # the label mode leaves with the phase, whatever the checks
            # found: a later phase never inherits my modes
            page.keyboard.press("Escape")
            page.wait_for_function("() => document.querySelector('.select-label') === null")
            # the pointer's own path: click the waiting row, then the
            # primary act - through the link
            page.click('[data-instrument="tasks"][data-task="t-1"]')
            page.keyboard.press("Enter")
            page.wait_for_function(
                f"() => ['loaded','invalid'].includes({viewer}.inspection().resource.status)"
                f" && {viewer}.inspection().identity.task === 't-1'"
            )
            check(
                "Enter opens the clicked row's board through the link",
                page.evaluate(f"() => {viewer}.inspection().identity.task") == "t-1"
                and page.evaluate(f"() => {viewer}.inspection().resource.path") == "t-1@r1",
                str(page.evaluate(f"() => {viewer}.inspection().resource")),
            )
            # back to the pipeline: typing a row's label opens its one
            # menu, its acts keyed; the menu's e shows the activity -
            # the lane on its axis and the events beside it
            open_location("smoke/tasks")
            page.keyboard.press("s")
            page.wait_for_selector(".select-label")
            chips = page.eval_on_selector_all(
                ".select-label",
                "els => els.map(e => ({text: e.textContent.trim(), block: e.getAttribute('data-target-block')}))",
            )
            t0_chip = next((c["text"] for c in chips if c.get("block") == "instrument-row-t-0"), None)
            assert t0_chip is not None, f"no label chip rides the t-0 row: {chips}"
            page.keyboard.type(t0_chip)
            page.wait_for_selector(".ctx-menu")
            page.keyboard.press("e")
            page.wait_for_selector('[data-control="instrument:overlay"]')
            events = page.eval_on_selector_all(
                '[data-control="instrument:events"] .overlay-event',
                "els => els.length",
            )
            lane_present = page.evaluate(
                """() => document.querySelector('[data-control="instrument:lane"]') !== null"""
            )
            check(
                "the task's menu e shows its activity: the lane and its events",
                events >= 2 and lane_present,
                f"events={events} lane={lane_present}",
            )
            screenshot_at('[data-control="instrument:overlay"]', "instrument-activity.png")
            page.keyboard.press("Escape")
            page.wait_for_function("""() => document.querySelector('[data-control="instrument:overlay"]') === null""")
            # help on demand names the instrument's own keys
            page.keyboard.press("?")
            page.wait_for_selector('[data-control="help:overlay"]')
            help_text = page.text_content('[data-control="help:overlay"]') or ""
            check(
                "? in the pipeline names the instrument's keys",
                "pipeline" in help_text and "the task's activity" in help_text,
                help_text[:160],
            )
            page.keyboard.press("Escape")

            # the glance: the agents instrument, its live join riding
            # the injected words (no host here - the inspection
            # setter, named as such)
            inspection = open_location("smoke/agents")
            check(
                "the agents page opens as the agents instrument",
                inspection["resource"]["status"] == "loaded"
                and inspection["resource"]["path"] == "agents",
                str(inspection["resource"]),
            )
            page.evaluate(
                f"""() => {viewer}.setWords({{
                    tA: {{ state: "working", app: "pi", msg: "editing the smoke", progress: "42", at: {int(1_000)} }},
                }})"""
            )
            page.wait_for_selector(".st.working")
            sections = page.eval_on_selector_all(
                ".instrument-label", "els => els.map(e => e.textContent)"
            )
            check(
                "the glance sections by the live word, the absent last",
                "working" in sections and sections[-1] == "absent",
                str(sections),
            )
            detail = page.eval_on_selector_all(
                ".instrument-detail", "els => els.map(e => e.textContent.trim())"
            )
            check(
                "the row shows Herdr's msg and progress exactly as sent",
                "editing the smoke" in detail and any("42" in d for d in detail),
                str(detail),
            )
            wordless = page.evaluate(
                "() => [...document.querySelectorAll('.instrument-row')].some(r => r.getAttribute('data-incarnation') === 'i-5' && r.querySelector('.st') === null)"
            )
            check(
                "a session with no word rides absent, shapeless",
                wordless is True,
                str(wordless),
            )
            screenshot_at("main.instrument", "instrument-agents.png")
            # the row menu: the copies and the details
            page.keyboard.press("s")
            page.wait_for_selector(".select-label")
            chips = page.eval_on_selector_all(
                ".select-label",
                "els => els.map(e => ({text: e.textContent.trim(), block: e.getAttribute('data-target-block')}))",
            )
            row_chip = next((c["text"] for c in chips if c.get("block") == "instrument-row-i-2"), None)
            assert row_chip is not None, f"no label chip rides the i-2 row: {chips}"
            page.keyboard.type(row_chip)
            page.wait_for_selector(".ctx-menu", timeout=3000)
            acts = page.eval_on_selector_all(
                ".ctx-menu-act", "els => els.map(e => e.textContent.trim())"
            )
            check(
                "the agents row's menu carries the copies and the details",
                any("attach command" in a for a in acts)
                and any("workspace path" in a for a in acts)
                and any("the branch" in a for a in acts),
                str(acts),
            )
            attach_key = page.evaluate("""() => {
                const rows = [...document.querySelectorAll('.ctx-menu-act')];
                for (const row of rows) {
                    if ((row.textContent ?? '').includes('attach command')) {
                        return row.querySelector('.ctx-menu-key')?.textContent?.trim() ?? null;
                    }
                }
                return null;
            }""")
            assert attach_key is not None, f"the attach act carries no key: {acts}"
            page.keyboard.press(attach_key)
            page.wait_for_selector('[data-control="instrument:note"]')
            note = page.text_content('[data-control="instrument:note"]')
            check(
                "the attach command's copy names what it carried",
                "attach command copied" in (note or ""),
                str(note),
            )
            page.keyboard.press("?")
            page.wait_for_selector('[data-control="help:overlay"]')
            help_text = page.text_content('[data-control="help:overlay"]') or ""
            check(
                "? in the glance names the agents' keys",
                "agents" in help_text and "attach command / workspace path / branch" in help_text,
                help_text[:160],
            )
            page.keyboard.press("Escape")
        finally:
            # my modes leave with my phase, whatever the checks found:
            # three honest Escapes cover overlay, menu and labels; the
            # extra ones land on nothing
            page.keyboard.press("Escape")
            page.keyboard.press("Escape")
            page.keyboard.press("Escape")
            tracker.stop()

    phase("instruments", phase_instruments)

    # the straddle the human hit, adapted to the pinned reads: a
    # revision landing between one open's revision resolution and the
    # next open. The stand-in's bare page answers r1 once (image
    # embedded in prose, the file present) and r2 afterwards (the
    # figure component). One open resolves its revision once and reads
    # text and figures at that one immutable revision, so a landing
    # mid-open cannot split the tree; the re-open answers the new
    # revision and its committed figure renders.
    def phase_landing() -> None:
        def open_landing(spec: str) -> dict:
            page.fill('[data-control="open:host-path"]', spec)
            page.click('[data-control="open:host"]')
            page.wait_for_function(
                f"() => ['loaded','invalid'].includes({viewer}.inspection().resource.status)"
            )
            return page.evaluate(f"() => {viewer}.inspection()")

        tracker = tracker_mock.TrackerMock("smoke", allowed_origins=[base_url])
        tracker.start()
        try:
            # this phase serves its own mock: the header setting must
            # point at it, not at the tracker phase's stopped one
            page.fill('[data-control="server:host"]', f"127.0.0.1:{tracker.port}")
            page.click('[data-control="server:save"]')
            tracker.landing_gets = 0
            tracker.landing_served = False
            inspection = open_landing("smoke/t-3")
            page.wait_for_function(
                f"() => {viewer}.inspection().renderOutcomes.some(o => o.kind === 'image' || o.kind === 'figure')"
            )
            inspection = page.evaluate(f"() => {viewer}.inspection()")
            outcomes = inspection["renderOutcomes"]
            missing = [o for o in outcomes if o["kind"] in ("image", "figure") and o["status"] == "missing"]
            ok = [o for o in outcomes if o["kind"] in ("image", "figure") and o["status"] == "ok"]
            check(
                "a landing mid-open never reports a committed figure as missing",
                missing == [],
                str(outcomes)[:200],
            )
            check("the first revision's image renders from its own tree", len(ok) == 1, str(outcomes)[:200])
            check(
                "one open resolves its revision once",
                tracker.landing_gets == 1,
                f"landing board-page count: {tracker.landing_gets}",
            )
            inspection = open_landing("smoke/t-3")
            page.wait_for_function(
                f"() => {viewer}.inspection().renderOutcomes.some(o => o.kind === 'figure' && o.status === 'ok')"
            )
            inspection = page.evaluate(f"() => {viewer}.inspection()")
            outcomes = inspection["renderOutcomes"]
            ok = [o for o in outcomes if o["kind"] == "figure" and o["status"] == "ok"]
            check(
                "the committed figure renders on re-open after the landing",
                len(ok) == 1 and inspection["resource"]["path"].startswith("t-3@r2"),
                f"{outcomes} {inspection['resource']}",
            )
            screenshot_at(".figure-frame", "landing-figure.png")
        finally:
            tracker.stop()

    phase("landing straddle", phase_landing)

    # phase 9: select first, the picker, help, zoom - the interaction
    # chapter over the real frontend. The acts ride real key events; the
    # inspection surface observes the DOM state.
    def phase_interaction() -> None:
        def key(key: str, modifiers: list[str] | None = None) -> None:
            for modifier in modifiers or []:
                page.keyboard.down(modifier)
            page.keyboard.press(key)
            for modifier in reversed(modifiers or []):
                page.keyboard.up(modifier)

        def inspect_now() -> dict:
            return page.evaluate(f"() => {viewer}.inspection()")

        # labels on what can be selected
        key("s")
        chips = page.evaluate("() => document.querySelectorAll('.select-label').length")
        check(
            "s puts labels on what can be selected",
            chips > 0,
            f"chips: {chips}",
        )
        key("Escape")
        chips_after = page.evaluate("() => document.querySelectorAll('.select-label').length")
        check("escape leaves the label mode", chips_after == 0, f"chips: {chips_after}")

        # right-click opens the document menu with its acts
        page.evaluate("""
          () => {
            const target = document.querySelector('.content');
            const rect = target.getBoundingClientRect();
            target.dispatchEvent(new MouseEvent('contextmenu', {
              bubbles: true, cancelable: true,
              clientX: rect.left + 40, clientY: rect.top + 40,
            }));
          }
        """)
        menu_visible = page.evaluate("() => document.querySelector('.ctx-menu') !== null")
        check("right-click opens the same menu at the pointer", menu_visible)
        acts = page.evaluate(
            "() => Array.from(document.querySelectorAll('.ctx-menu-act .ctx-menu-label')).map(e => e.textContent)"
        )
        check(
            "the menu carries the acts with their keys",
            any("zoom" in str(act) for act in acts) and any("search" in str(act) for act in acts),
            str(acts)[:120],
        )
        key("Escape")

        # the picker opens over space space and resolves a typed location
        key(" ")
        key(" ")
        picker_open = page.evaluate("() => document.querySelector('[data-control=\"picker:overlay\"]') !== null")
        check("space space opens the picker", picker_open)
        page.fill('[data-control="picker:input"]', "smoke/t-0")
        page.wait_for_selector('[data-control="picker:result"]')
        rows = page.evaluate(
            "() => Array.from(document.querySelectorAll('[data-control=\"picker:result\"] .picker-name')).map(e => e.textContent)"
        )
        check(
            "a typed location resolves in the picker even unenumerated",
            any("t-0" in str(row) for row in rows),
            str(rows)[:120],
        )
        key("Escape")

        # help on demand
        key("?")
        help_visible = page.evaluate("() => document.querySelector('[data-control=\"help:overlay\"]') !== null")
        check("? lists the keys and commands", help_visible)
        key("Escape")

        # zoom
        before = page.evaluate("() => document.querySelector('.content').style.zoom || '1'")
        key("+")
        after = page.evaluate("() => document.querySelector('.content').style.zoom")
        check("plus zooms the document", float(after) > float(before), f"{before} -> {after}")
        key("=")
        reset = page.evaluate("() => document.querySelector('.content').style.zoom")
        check("equals resets the zoom", abs(float(reset) - 1.0) < 0.001, reset)

    phase("interaction", phase_interaction)

    return sum(1 for ok, _name, _detail in RESULTS[started_records:] if not ok)


def reference_document() -> pathlib.Path | None:
    """The env-gated reference page: the entry point validated the host
    path (absolute, readable, .html) and bound it at the fixed sandbox
    path; unset means absent, a missing bind means the lane refuses
    loudly (the caller wanted a reference and did not get one)."""
    doc_raw = os.environ.get("BOARD_VIEWER_BROWSER_REFERENCE_HTML", "")
    if not doc_raw:
        return None
    bound = pathlib.Path(REFERENCE_SANDBOX_PATH)
    if not bound.is_file():
        return None
    return bound


def run_reference(page, base_url: str, shots_dir: pathlib.Path, engine: str) -> None:
    """One navigation of the reference page after the main pass: render
    it, keep a screenshot under the run root. Test plumbing within the
    existing containment - the same loopback server, the same engine,
    no new capability."""
    doc = reference_document()
    if doc is None:
        record_check(
            f"[{engine}] reference page is bound and readable",
            False,
            os.environ.get("BOARD_VIEWER_BROWSER_REFERENCE_HTML", ""),
        )
        return
    try:
        page.goto(f"{base_url}{REFERENCE_ROUTE}")
        page.wait_for_load_state("load")
        # a bounded settle: the reference may compose nested documents
        # (the round 16 mockup builds each window in an iframe) that
        # paint a beat after load
        page.wait_for_timeout(800)
        title = page.title()
        record_check(f"[{engine}] reference page renders with its own title", bool(title), str(title))
        # a viewport capture: a reference page may lay out taller than
        # the engine's screenshot bound in full-page mode
        shot = shots_dir / f"reference-{engine}.png"
        page.screenshot(path=str(shot))
        record_check(f"[{engine}] reference page screenshot retained", shot.is_file(), str(shot))
    except Exception as error:
        record_check(f"[{engine}] reference page renders", False, f"{type(error).__name__}: {str(error)[:160]}")


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
            # WebKit gates the lane (the default engine); Chromium runs
            # only behind the explicit triage flag and never gates - the
            # identical check list runs under each engine either way
            engines = ("webkit",)
            if os.environ.get("BOARD_VIEWER_BROWSER_TRIAGE_CHROMIUM", ""):
                engines = ("webkit", "chromium")
            for engine in engines:
                setup = setup_engine_page(playwright, engine, base_url)
                if setup is None:
                    continue
                browser, page = setup
                try:
                    page.set_default_timeout(10000)
                    page.goto(base_url)
                    page.wait_for_function("() => window.__boardViewer !== undefined")
                except Exception as error:
                    record_check(
                        f"[{engine}] navigates to the served frontend",
                        False,
                        f"{type(error).__name__}: {error}",
                    )
                    try:
                        browser.close()
                    except Exception:
                        pass
                    continue
                run_checks(page, base_url, shots_dir, engine)
                # the env-gated reference page rides the same sandboxed
                # engine and the same loopback server after the product pass
                if os.environ.get("BOARD_VIEWER_BROWSER_REFERENCE_HTML", ""):
                    run_reference(page, base_url, shots_dir, engine)
                browser.close()
    finally:
        server.shutdown()
        server.server_close()

    # the gate is the WebKit engine's own records - Chromium triage
    # failures are printed and saved but never decide the lane exit
    gate = [record for record in RESULTS if record[1].startswith("[webkit]")]
    triage = [record for record in RESULTS if record[1].startswith("[chromium]")]
    gate_failed = sum(1 for ok, _name, _detail in gate if not ok)
    triage_failed = sum(1 for ok, _name, _detail in triage if not ok)
    cost = time.monotonic() - started
    print(f"[browser] webkit (gate): {len(gate) - gate_failed}/{len(gate)} checks passed in {cost:.1f}s", flush=True)
    if triage:
        print(f"[browser] chromium (triage): {len(triage) - triage_failed}/{len(triage)} checks passed", flush=True)
    print(f"[browser] screenshots saved under {shots_dir} for inspection", flush=True)
    return 1 if gate_failed else 0


if __name__ == "__main__":
    sys.exit(main())
