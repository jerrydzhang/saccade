# Board viewer

A retained, read-only research reader: it renders an explicitly supplied
board declaration and nothing else. Loading a board never touches tracker
storage, authority, or workflow; the `task` field in a declaration is
authored metadata, not a tracker association.

## Run

```sh
# the fast development loop: typecheck, pure contract tests, the
# bridge crate's contract tests, build
python3 harness/entry.py check-fast

# the native client's own guard tests (Tauri shell, no UI start)
nix develop .#board-viewer -c bash -c 'cd src-tauri && cargo test'
```

Measured warm cost of the fast loop: about 9 seconds. Service lanes
(browser, native, serve, any SSH) are separate, gated, and launched only
through `harness/entry.py` under enforced OS isolation - see
`harness/README.md`.

## Reading a board

The standing control is the header location field: give an explicit
location and open it. A location is `/absolute/path` (the machine this
viewer runs on) or `machine:/absolute/path` (a machine from the local
herdr-core's list, as `hc machine ls` shows); the same grammar is the
launch argument, `board-viewer /path/b.yaml` or `board-viewer --on
machine /path/b.yaml`. Fixtures live in `fixtures/` and remain a
dev/test path (the browser lane and the tests address them through the
inspection surface, not the human UI).

Grammar versions: the viewer declares the versions it renders - exactly
1 today, the same range the shared `board-grammar` crate parses. A board
declaring no grammar version, or an unknown major version, refuses with
a message that names the supported versions; there is no legacy
interpretation. A component kind the build does not know (a newer minor
grammar's admission) never refuses the board: it renders as its raw
authored source beside a diagnostic.

The validation split: admission is the shared crate's job on the Tauri
side (`validate_board`, a path dependency), and the native lanes refuse
on its verdict; the frontend keeps only the presentation decoder, which
mirrors the crate's contract exactly so the browser lane (no bridge)
behaves identically. Parity over the shared fixture corpus is pinned by
tests on both sides: this repository's node tests and the
`viewer-native-check` cargo suite read the same `fixtures/` files.

Keys:

- `Ctrl+F` opens the search bar; `Escape` closes it; `Enter` and
  `Shift+Enter` walk the matches
- copy is Source by default; `Ctrl+Shift+C` toggles Plain; `Ctrl+C`
  copies the selection in the current mode (a formula is copied whole
  in its current mode whatever fraction of it is selected)

A machine-named board reads through the local core's read-only face:
`file.read` for declaration text, fixed-argv `exec` (`realpath -e`, then
`cat`) for figure bytes over base64 - no shell strings anywhere. The
viewer never adds machines, logs in, updates a core or writes a file;
those are the human's own host setup. Machine states are structured,
human-actionable errors with the machine's own reason (`needs_login`,
`unreachable`, `needs_update`, unknown name) and no retries. Because the
remote host enforces no path containment, the asset's canonical prefix
check runs on the remote side: one `realpath -e -- <base> <candidate>`
answers with both canonical forms, the prefix relation is checked
there, and the read itself uses the canonical path. Bytes over the 8 MB
exec cap, or flagged `truncated`, and text reads that stopped inside
the file, are refused with an error - never partially rendered.

Contracts this reader holds (pinned by `tests/` and the bridge's cargo
suite):

- malformed input is a structured, positional error, never a
  manufactured board; unknown keys and fields refuse, exactly as the
  crate refuses them, and unknown component kinds render as source
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
  asset commands with mirrored path guards; board admission through the
  shared `board-grammar` crate (`validate_board`); no shell, no eval, no
  generic filesystem surface, no plugins. `src-tauri/core-client/` is
  the pure read-only herdr-core client (JSON-RPC over the local core's
  socket, machine-named reads, remote containment), contract-tested in
  the fast loop against fakes
- `fixtures/` - deterministic boards and assets (ASCII bytes; Unicode
  data is exercised at runtime with escapes in tests)
- `tests/` - pure contract tests (node --test)
- `harness/` - the single entry point and the containment contract
