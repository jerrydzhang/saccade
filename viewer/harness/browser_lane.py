#!/usr/bin/env python3
"""The browser lane payload: runs INSIDE the enforced sandbox, after the
containment canary has passed in the same sandbox, and never before.

Order, enforced here:
  1. containment canary (same module, same baselines) - a refusal ends
     the lane before any server or driver exists
  2. the built frontend is served on loopback inside the sandbox netns
     (the only listener this lane ever creates)
  3. one reused headless Chromium session via the Nix-pinned Playwright
     driver (driver and browser from the same nixpkgs revision; nothing
     downloaded at runtime)
  4. a bounded check list over the real frontend and the real fixture
     boards: DOM and state first, screenshots saved for later inspection
  5. owned teardown of the server this lane spawned

Everything here is inside the sandbox: the clipboard a browser writes is
Chromium's own clipboard, not the desktop's - native-clipboard fidelity
is a real-client question and is reported as a limit, never implied.
"""

from __future__ import annotations

import io
import os
import pathlib
import socket
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import containment_canary  # noqa: E402

LANE_TIMEOUT_SECONDS = 150
SERVER_PORT = 1421
SERVER_READY_SECONDS = 20
DIST_SANDBOX = "/app/dist"
MODULES_SANDBOX = "/app/node_modules"
SHOTS_SANDBOX = "/shots"

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    mark = "ok" if condition else "FAIL"
    stream = sys.stdout if condition else sys.stderr
    print(f"[browser] {mark}: {name} {detail if not condition else ''}", flush=True)


def run_canary_first() -> bool:
    expected, failures = containment_canary.parse_baselines()
    if failures:
        print("containment refusal: baseline contract failed inside the sandbox", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return False
    failures = containment_canary.verify_containment(expected)
    if failures:
        print("containment refusal: the sandbox is not effectively isolated", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return False
    print("[browser] containment canary passed before any server or driver", flush=True)
    return True


def start_server(node_bin: str) -> subprocess.Popen[bytes] | None:
    argv = [node_bin, f"{MODULES_SANDBOX}/vite/bin/vite.js", "preview", DIST_SANDBOX, "--port", str(SERVER_PORT), "--strictPort"]
    try:
        server = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as error:
        print(f"containment refusal: vite preview could not start: {error}", file=sys.stderr)
        return None
    deadline = time.monotonic() + SERVER_READY_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            print("containment refusal: vite preview exited during startup", file=sys.stderr)
            return None
        try:
            with socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=0.5):
                return server
        except OSError:
            time.sleep(0.2)
    print("containment refusal: vite preview never became ready", file=sys.stderr)
    server.terminate()
    server.wait(timeout=5)
    return None


def main() -> int:
    started = time.monotonic()
    if not run_canary_first():
        return 1

    node_bin = os.environ.get("NODE_BIN", "")
    if not node_bin or not node_bin.startswith("/"):
        print("containment refusal: NODE_BIN must be an absolute pinned path", file=sys.stderr)
        return 1

    server = start_server(node_bin)
    if server is None:
        return 1
    print(f"[browser] serving {DIST_SANDBOX} on 127.0.0.1:{SERVER_PORT}", flush=True)

    shots = pathlib.Path("/shots")
    shots.mkdir(exist_ok=True)

    failures = 0
    try:
        from playwright.sync_api import sync_playwright

        base_url = f"http://127.0.0.1:{SERVER_PORT}"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, chromium_sandbox=False)
            context = browser.new_context(viewport={"width": 1100, "height": 800})
            # Chromium-internal clipboard only; never the desktop's.
            context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
            page = context.new_page()
            page.set_default_timeout(10000)
            viewer = "window.__boardViewer"

            def open_board(name: str) -> None:
                page.evaluate(f"() => {viewer}.open('fixture', {name!r})")
                page.wait_for_function(f"() => {viewer}.inspection().resource.status === 'loaded' || {viewer}.inspection().resource.status === 'invalid'")

            # 1. the real migration board loads through the same handler the
            #    header control uses
            open_board("saccade-and-migration.yaml")
            inspection = page.evaluate(f"() => {viewer}.inspection()")
            check("real migration board loads", inspection["resource"]["status"] == "loaded" and inspection["errors"] == [], str(inspection["resource"]))
            check("real board task metadata is authored-only", inspection["identity"]["task"] == "saccade-current-system-and-migration", str(inspection["identity"]))

            # 2. actual rendering outcomes: math and diagrams hydrated
            page.wait_for_timeout(1500)
            outcomes = page.evaluate(f"() => {viewer}.inspection().renderOutcomes")
            math_ok = [o for o in outcomes if o["kind"] == "math" and o["status"] == "ok"]
            mermaid_ok = [o for o in outcomes if o["kind"] == "mermaid" and o["status"] == "ok"]
            math_bad = [o for o in outcomes if o["kind"] == "math" and o["status"] != "ok"]
            mermaid_bad = [o for o in outcomes if o["kind"] == "mermaid" and o["status"] != "ok"]
            check("migration board math renders (actual outcomes)", len(math_ok) > 0 and len(math_bad) == 0, str(outcomes[:4]))
            check("migration board diagrams render (actual outcomes)", len(mermaid_ok) > 0 and len(mermaid_bad) == 0, str(outcomes[:4]))
            containers = page.evaluate("() => document.querySelectorAll('mjx-container').length")
            check("math is in the DOM as MathJax containers", containers > 0, str(containers))

            # 3. copy semantics: representative prose+math selection, both
            #    modes. execCommand triggers the real document copy event
            #    our handler intercepts; Chromium routes the prevented
            #    payload into its own clipboard, which this sandbox reads.
            def select_prose_plus_math() -> None:
                page.evaluate("""() => {
                    const math = document.querySelector('.math-inline');
                    const range = document.createRange();
                    range.setStartBefore(math.parentNode.firstChild);
                    range.setEndAfter(math);
                    const selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);
                    document.execCommand('copy');
                }""")

            def copied_text() -> str:
                return page.evaluate("async () => navigator.clipboard.readText()")

            page.evaluate("() => window.__boardViewer.setCopyMode('source')")
            select_prose_plus_math()
            source_copy = copied_text()
            check("Source copy keeps authored bytes for math", "$" in source_copy, repr(source_copy[:80]))

            page.evaluate("() => window.__boardViewer.setCopyMode('plain')")
            select_prose_plus_math()
            plain_copy = copied_text()
            check("Plain copy emits the TeX value", "$" not in plain_copy and len(plain_copy) > 0, repr(plain_copy[:80]))
            page.screenshot(path=f"{SHOTS_SANDBOX}/copy-modes.png")

            # 4. duplicate-expression search: two occurrences, both reachable
            open_board("repeated-math.yaml")
            page.fill("input[type=search]", "0 < \\eta < 2/b")
            page.wait_for_timeout(300)
            search_state = page.text_content(".search-state") or ""
            check("duplicate expression search finds two occurrences", search_state.strip().startswith("1 / 2"), repr(search_state))
            page.click("button[aria-label='next match']")
            page.wait_for_timeout(200)
            search_state = page.text_content(".search-state") or ""
            check("next match reaches the second occurrence", search_state.strip().startswith("2 / 2"), repr(search_state))

            # 5. theme and narrow width
            page.evaluate("() => window.__boardViewer.setTheme('dark')")
            theme = page.evaluate("() => document.documentElement.dataset.theme")
            check("dark theme applies to the document root", theme == "dark", str(theme))
            page.set_viewport_size({"width": 520, "height": 800})
            overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
            check("narrow window has no page-level horizontal overflow", overflow <= 0, str(overflow))
            page.screenshot(path=f"{SHOTS_SANDBOX}/dark-narrow.png")

            # 6. expanded diagram controls and return
            open_board("large-diagram.yaml")
            page.wait_for_timeout(1200)
            page.click(".diagram-frame")
            overlay_visible = page.evaluate("() => document.querySelector('.diagram-overlay') !== null")
            check("diagram expands into the overlay", overlay_visible is True)
            page.click("[data-action='zoom-in']")
            page.wait_for_timeout(150)
            zoom_label = page.text_content(".diagram-zoom-label") or ""
            check("zoom control raises the scale", zoom_label.strip() not in ("100%", ""), repr(zoom_label))
            page.click("[data-action='reset']")
            page.wait_for_timeout(150)
            zoom_label = page.text_content(".diagram-zoom-label") or ""
            check("reset returns scale to 100%", zoom_label.strip() == "100%", repr(zoom_label))
            page.screenshot(path=f"{SHOTS_SANDBOX}/diagram-expanded.png")
            page.keyboard.press("Escape")
            page.wait_for_timeout(150)
            overlay_gone = page.evaluate("() => document.querySelector('.diagram-overlay') === null")
            check("closing returns to the board", overlay_gone is True)
            page.evaluate("() => window.__boardViewer.setTheme('system')")

            # 7. invalid and missing inputs keep source visible with outcomes
            open_board("invalid.yaml")
            inspection = page.evaluate(f"() => {viewer}.inspection()")
            check("invalid board is a structured error, never a board", inspection["resource"]["status"] == "invalid" and any(e["code"] == "yaml-syntax" for e in inspection["errors"]), str(inspection["errors"])[:200])
            error_visible = page.evaluate("() => document.querySelector('.errors .render-error') !== null")
            check("invalid board error is rendered visibly", error_visible is True)

            open_board("missing-asset.yaml")
            page.wait_for_timeout(800)
            outcomes = page.evaluate(f"() => {viewer}.inspection().renderOutcomes")
            missing = [o for o in outcomes if o["kind"] == "figure" and o["status"] == "missing"]
            check("missing figure is an actual reported outcome", len(missing) == 1, str(outcomes))
            missing_visible = page.evaluate("() => document.querySelector('.figure-frame.asset-missing .render-error') !== null")
            check("missing figure error is rendered visibly", missing_visible is True)
            page.screenshot(path=f"{SHOTS_SANDBOX}/missing-asset.png")

            browser.close()
    finally:
        # owned teardown of this lane's own server child
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        output = server.stdout.read().decode(errors="replace") if server.stdout else ""
        if output.strip():
            print(f"[browser] server log: {output[:500]}", flush=True)

    failures = sum(1 for ok, _name, _detail in RESULTS if not ok)
    cost = time.monotonic() - started
    print(f"[browser] {len(RESULTS) - failures}/{len(RESULTS)} checks passed in {cost:.1f}s", flush=True)
    print(f"[browser] screenshots saved under {SHOTS_SANDBOX} for inspection", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
