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
    # (harness/tracker_mock.py): the one server setting, the version
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
        finally:
            tracker.stop()
            future.stop()

    phase("tracker locations", phase_tracker)

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
