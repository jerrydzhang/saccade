---
name: playwright
description: Use when driving a real browser headlessly — inspecting or testing a web UI, taking screenshots, measuring layout, verifying rendering, or scraping content that needs JS. Nix-native setup (never `playwright install`). Carries the inspection loop that reads the DOM before the eyes, plus live-learned HTML and process pitfalls.
---

# Playwright browser use, nix-native

## Setup

Preferred: the project flake already wires it (saccade's does — python3 with
playwright + `PLAYWRIGHT_BROWSERS_PATH` pointing at nix store browsers).
Everything runs inside `nix develop`.

Outside a flake, build a one-off env (flake refs with parentheses cannot go
through `nix shell` — use `--expr`):

```sh
nix build --impure --expr '(import <nixpkgs> {}).python3.withPackages(ps: [ps.playwright])' -o /tmp/pwpy
export PLAYWRIGHT_BROWSERS_PATH="$(nix build nixpkgs#playwright-driver.browsers --print-out-paths)"
/tmp/pwpy/bin/python3 script.py
```

Rules the hard way:
- **Never `playwright install`** — it downloads browsers outside the store.
- `PLAYWRIGHT_BROWSERS_PATH` and the python package must come from the **same
  nixpkgs revision**; the revision check is exact (chromium-1228 vs 1229 fails).
- `nix shell nixpkgs#python3Packages.playwright` gives the CLI but **not** an
  importable module — use `python3.withPackages`.

## Core loop

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(url); pg.wait_for_load_state("load")
    ...
    b.close()
```

Click real selectors, not coordinates. `pg.click("canvas a >> nth=0")` hangs
30s if the element is invisible — see pitfalls below.

## Inspect the DOM before the eyes

Programmatic checks answer most questions better than screenshots, and they
compose into assertions:

- Geometry: `getComputedStyle(el).gridTemplateColumns`,
  `el.getBoundingClientRect()` (x/y/width), `document.documentElement.scrollWidth - window.innerWidth` (0 = no horizontal overflow).
- Counts: rows per section, events rendered, `document.querySelectorAll('script').length` (0 = no injected script from hostile text — HTML escaping held).
- Overflow: an element with `scrollWidth > clientWidth + 2`.
- Depth/nesting: measure the **content** element's rect (an `<a>` inside the
  row), not the padded container — a div's border-box left doesn't move when
  you add `padding-left`.

Then screenshots for the human and for yourself: `pg.screenshot(path=...)`,
then `read` the png — image input works in this setup; describe what you see,
don't claim you can't.

## HTML pitfalls (each one bit in real use)

- **Never use the `<canvas>` tag as a layout region.** Its children are
  fallback content and render as *nothing* — selectors work, innerText works,
  nothing is visible. Use `<div class="canvas">`.
- **Whitespace indentation collapses in HTML.** Tree depth must be CSS
  (`padding-left` per level), not leading spaces in text.
- Multi-column reading with wrapped lines: continuation lines fall back under
  a fixed-width id gutter unless you use hanging indents (grid `dl`, or
  `padding-left` + negative `text-indent`).
- Truncate at word boundaries; fixed char cuts produce "cli…" garbage.

## Pairing with a local dev server

Background it so it survives between tool calls:

```sh
setsid nohup python3 server.py 7783 0.0.0.0 >log 2>&1 < /dev/null & disown
```

- Plain `nohup … &` gets reaped with the command's process group by some
  harnesses; `setsid` survives.
- `pkill -f pattern` kills the *invoking shell too* if its own command line
  contains the literal pattern anywhere (e.g. the relaunch in the same chain).
  Use the bracket trick (`pkill -f 'serve_moc[k].py'`) **and** keep the pkill
  in its own command, separate from anything mentioning the literal name.
- `grep -c` counts matching **lines**; a one-line join makes 614 events count
  as 1. Use `grep -o pat | wc -l` for occurrences.
