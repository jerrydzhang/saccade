# Board viewer

A retained, read-only research reader: it renders an explicitly supplied
board declaration and nothing else. Loading a board never touches tracker
storage, authority, or workflow; the `task` field in a declaration is
authored metadata, not a tracker association.

## Run

```sh
# the fast development loop: typecheck, pure contract tests, build
python3 harness/entry.py check-fast

# the native client's own guard tests (Tauri shell, no UI start)
nix develop .#board-viewer -c bash -c 'cd src-tauri && cargo test'
```

Measured warm cost of the fast loop: about 9 seconds. Service lanes
(browser, native, serve, any SSH) are separate, gated, and launched only
through `harness/entry.py` under enforced OS isolation - see
`harness/README.md`.

## Reading a board

The first input is a transitional explicit selector: pick a fixture board
or give an explicit host path in the app header. Fixtures live in
`fixtures/` and include the real migration board
(`saccade-and-migration.yaml`, byte-pinned from the tracker's artifact
store by test) and its source-map companion.

Contracts this reader holds (pinned by `tests/`):

- malformed input is a structured, positional error, never a manufactured
  board; unknown keys are surfaced diagnostics, not silently dropped
- one source-positioned Markdown pipeline: math spans, mermaid fences and
  ordinary inert code all come from the same parse; raw HTML in authored
  content is reported, never rendered
- math renders at natural size in the surrounding text color; over-wide
  display math scrolls in its own area; an indivisible inline expression
  may overflow naturally rather than shrink
- an authored math expression is one logical copy and search unit:
  Source mode emits authored bytes, Plain mode emits the TeX value, and
  equal formulas at distinct source locations are distinct occurrences
- ordinary code fences never execute; mermaid renders as a figure that
  never captures gestures while embedded and expands on activation;
  rendering failure keeps the authored source visible
- incorporation provenance is displayed exactly as authored and validates
  nothing
- relative assets resolve at the declaration's location on the selected
  machine; traversal and absolute paths are refused on both the frontend
  and the host side

## Layout

- `src/lib/` - read model, decode, Markdown pipeline, projection, copy
  semantics, sources, view model, inspection projection, DOM renderer,
  hydration, diagram overlay
- `src/App.svelte` - the shell; one view model feeds the human UI, the
  machine-readable inspection (`window.__boardViewer`) and the copy
  handler
- `src/styles/board.css` - the centrally shared stylesheet
- `src-tauri/` - the narrow Tauri 2 shell: read-only declaration and
  asset commands with mirrored path guards; no shell, no eval, no
  network, no plugins
- `fixtures/` - deterministic boards and assets (ASCII bytes; Unicode
  data is exercised at runtime with escapes in tests)
- `tests/` - pure contract tests (node --test)
- `harness/` - the single entry point and the containment contract
