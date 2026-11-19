# Saccade app

The Saccade suite's native app: a Svelte 5 frontend (`port/`) inside a
Tauri 2 shell (`port-shell/` over the shell library in
`src-tauri/src/`). Its windows hold the instruments - a board, the
task list - read from a tracker over its HTTP wire. The frame is a
harness for the instruments (t-73): an instrument publishes its
targets with its data (`ContentFrame.targets`, the contract's one
target shape in `port/src/lib/contracts/buffer.ts`), and the frame
presents them - select mode and its labels, the act menu, the click
paths, the shared reading keys (`frameReadingContext`, an instrument
adds keys and never rebinds one), the reading anchor Back rides. Both
instruments publish their targets with their data and keep only what
is about their own data - the task list: the rail's feed, the add
field, its return places; the board: find, folding, expansion,
history, following its task's revisions, figures, and how its content
is drawn. Unfolding the groups around an anchor before it can be
revealed is the board's too - finding and measuring the anchor is the
frame's, as it is for the task list.
Which instrument shows a location is each instrument's own claim
(`port/src/lib/instruments/claims.ts`), read by the window assembly
instead of a hard-coded choice. The app keeps no window
state between runs (t-71@r3): a launch opens one fresh window, and
what a run needs - which window is which and each window's link (the
receiver `o` delivers into) - lives in the shell's memory (the window
registry, `src-tauri/src/window_registry.rs`), never in any file.
Zoom and the column width (the board's reading layout,
`port/src/lib/board/reading-layout.ts`) are per window for the run and
start at their defaults. Files an earlier build left on a device
(`window-store.json`, `open-windows.json`, an old `<db>.windows`) sit
where they are: never read, never cleaned. The app is the
round 16 prototype's behaviour, built and tested scenario by scenario
(t-36).

A project is registered by where its repository is
(`port/src/lib/projects.ts`): the rail's add field (Space a, in any
window) takes `machine:/path/to/repository`, the declaration file
there (`workbench.toml`) names the project and the loopback port its
tracker answers on, and the device keeps the registered paths alone -
the file is read again through the one core module whenever a fact is
needed. The tracker is started with its own `--port`; the file names
the same port, and the app checks the tracker it reached serves the
registered repository, refusing with both paths when it does not. A
device that registered trackers by `host:port` under the
old keys (`saccade.trackers`, `saccade.trackers.home`) re-registers
its repositories by path; the old keys are never read and never
cleaned, and no special case reads them.

The rail is the frame's (`port/src/lib/frame/rail.ts`): any window
can show it (Space r, per window for the run), none starts with it, a
cramped window hides it. Its entries are ordinary targets of the
shown buffer, and its project views come from what the instruments
claim (`port/src/lib/instruments/` - the task list claims
`name/tasks`); the frame never names an instrument (the suite holds
the boundary: no frame file imports an instrument, the assembly names
none). A window with no location is the empty window: the frame's own
surface, three lines - Space Space open something, Space r show
projects, and Space a register a project while no project is
registered - and the window at launch is one.

The earlier frontend (`src/`) retired at t-36 m-554: it carried the
features first asked for, but was never usable enough for any of them to
be tried. Those features are the app's backlog - hypotheses to build
small, try, and then decide - not requirements. Their last description
is this file and `src/` at commit `0809fe7^`.

## Run

```sh
# the app, built and run (the flake's default app)
nix run .#saccade-app

# the full suite from the repository root: cargo, the shell's own tests,
# the frontend's vitest unit and browser projects
just test

# the plain uncontained checks: the harness's own checks, pyflakes, the
# frontend typecheck, the core client's contract tests, the build
just viewer-check
```

The recipes that run the frontend's tools depend on `just client-deps`,
which runs `npm ci` in `client/` whenever `package-lock.json` is newer
than the installed copy, so a merge that adds a dependency never leaves
a checkout or worktree on a stale install.

The native test project runs only through the gated session lane:
`just app-test-native` (below, and `port/tests/scenarios/README.md`).

## Layout

- `port/` - the frontend: `src/` (the window assembly `App.ts`, the
  instruments, `lib/` for the board model, rendering, find, select,
  menus, input, figures, and the frame's presentation of published
  targets in `lib/frame/`), `tests/` (vitest: unit tests in `tests/`,
  scenarios in `tests/scenarios/` with their kit), `vitest.config.ts`
  (the three projects), `vite.config.ts`
- `port-shell/` - the app's crate: its own workspace root, the binary
  `saccade-app`, and the tauri conf embedding `port/dist`
  (identifier `dev.saccade.app`)
- `src-tauri/src/` - the shell library the crate compiles: read-only
  declaration and asset commands with path guards, the one local
  request command (loopback, a bounded curl argv composed here), board
  admission through the shared `board-grammar` crate, the suite's
  window acts (`suite.rs`: windows, links, focus, the window jump),
  the window registry (the run's one memory of its windows), the look
  configuration, the
  key trace's stderr door
- `src-tauri/core-client/` - the pure read-only herdr-core client,
  contract-tested in `check-fast`
- `fixtures/` - deterministic boards and assets the frontend's fixture
  source bundles (ASCII bytes)
- `tauri-driver/` - the pinned `Cargo.lock` the flake builds tauri-driver
  from, for the native test project
- `harness/` - the gated sandbox (below)

## The harness

One entry point: `python3 client/harness/entry.py <lane>`. Every lane
that spawns runs inside one bwrap sandbox under
`isolation.PidfdSupervisor`, canary first, requiring a caller-side
tracker db (`SACCADE_DB`) the canary proves invisible:

- `check-fast` - plain, uncontained tool runs (above)
- `preflight` - the minimal containment verification
- `session suite` - a private headless Weston around the built app, then
  vitest's native project (payload `suite_payload.py`); what
  `just app-test-native` runs

Run roots are kept under `/var/tmp/saccade-app-<uid>/` for inspection.

## Packaging

`nix/saccade-app.nix` is one callPackage-able file taking every
dependency from the pkgs it is called on: the frontend dist
(`saccade-app-web`), the app (`saccade-app`, main program
`saccade-app`) and the tauri-driver for the native project
(`saccade-app-tauri-driver`). `packages.saccade-app` and
`apps.saccade-app` (also the flake's default app) build against the
`nixpkgs-client` input, the revision the human's system flake is built
from, so the build agents test is the build the client runs; the same
derivation is `overlays.default`, so a consumer builds it against its
own nixpkgs and the app shares the host's GL stack:

```nix
nixpkgs.overlays = [ inputs.saccade.overlays.default ];
```

then `pkgs.saccade-app` is the app.
