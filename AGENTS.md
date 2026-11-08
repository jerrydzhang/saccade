# AGENTS.md - how agents work in this repo

Saccade is a tracker for agent work. The packages: `tracker` (the `sac`
binary: event log, server, CLI, boards), `grammar` (the board format),
`client` (the board viewer), `tools/herdr-ui` (the desktop-app
sandbox). Each package owns its version and its docs. `AXIOMS.md`
carries world belief - read it before proposing anything
architectural. Using the tracker is not required to develop this
project; `tracker/skill/SKILL.md` teaches the `sac` CLI for when you
do, and `tracker/README.md` carries the tracker's law.

## Decisions and ratification

- One decision at a time; the human ratifies; it lands in `AXIOMS.md`,
  a package's docs, or the sections here.
- The problem is never silently modified. Deviations are announced
  before enactment. Success claims name their evidence: tests run and
  counts, files touched, commands exercised.
- Ownership is conceptual, never file-based. The human authors
  ontology, event payload meaning, authority and transitions,
  invariants, and acceptance assertions. Agents draft codecs, storage
  plumbing, HTTP/CLI adapters, Git and process mechanics, formatting,
  harnesses, and already-ratified variants. Ambiguity returns to the
  human, never resolved by path.

## Compatibility and versioning

- Each package versions its own contract in 0.x semver: the minor
  position is breaking, PATCH covers everything else; UI surfaces and
  internal modules are never breaking alone. Tags and flake outputs
  are `<package>-v<version>`.
- Cross-package changes land as one tested commit; components declare
  the contract ranges they use.
- Breaking payload changes are fine while this repo is the sole
  consumer; the old-logs-always-load promise starts at external
  adoption or painful volume.
- A required stored-format migration is a bounded version-to-version
  step in code, run before serving, with transactional validation and
  recovery safeguards. Behavioral breaks alone never rewrite data.
- Shared code is one package per contract: shared only when a second
  consumer needs a settled shape and the life cycles match. Internal
  crates are named for what they do, never "common".

## Layout and docs

- Markdown lives in the root law files and per-package READMEs,
  written for anyone working on the repository. The skill shares
  `sac`'s life cycle: source in `tracker/skill/`, deployed copies
  generated and git-ignored (`sac skill install`).
- Authored text is ASCII, held by a repository check over every
  tracked file. Record content stays Unicode-capable; the rule never
  rewrites data.
- A phase review checks locality: no stray docs, no stale paths, no
  files outside their package's story.

## Building and testing

- Run cargo and nix through `direnv exec . bash -c '...'`.
- `just fmt`, `just lint`, `just test`; the clippy gate is
  `cargo clippy -p saccade -p board-grammar --all-targets -- -D warnings`.
- `cargo build` is the judge, not grep.
- A phase gate: the suites, fmt, every prek hook on each commit,
  `nix build .#sac .#board-viewer`, and the three viewer lanes under
  `BOARD_VIEWER_RUNTIME_AUTHORIZED=3030`:
  `nix develop .#board-viewer -c env BOARD_VIEWER_RUNTIME_AUTHORIZED=3030 SACCADE_DB=<db> python3 client/harness/entry.py browser|remote`,
  with `--board <yaml>` and the built app for native. The lane logs
  are inspected, not assumed.
- Phase reviews diff the test inventory by name against the previous
  phase: additions fine, every removal explained.
- Tracker test placement and suite contracts live in
  `tracker/README.md`; the grammar's fixtures in `grammar/README.md`.

## Sandboxes and safety

- Drive the desktop app headless only through `just herdr-ui up`;
  never reconstruct its runtime environment. The mechanical loop is
  `tools/herdr-ui/README.md`.
- Never add production machines to a sandbox, never automate the
  person's real display, never stop another session's core or
  compositor.
- The headless loop proves mechanics, not the human's actual client;
  the real-client gate stays separate.

## Branches and commits

- One commit per decision or mechanical batch; the message says what
  changed, not the history of deciding.
- Re-read the exact lines before every edit - files change live during
  review sessions. Verify each change before stacking the next:
  build, suite, then a CLI smoke on a throwaway db for anything
  touching verbs or wire.
- Every prek hook passes on every commit; the fmt hook is never
  skipped.
