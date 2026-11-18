# Saccade app

The Saccade suite's native app: a Svelte 5 frontend (`port/`) inside a
Tauri 2 shell (`port-shell/` over the shell library in
`src-tauri/src/`). Its windows hold the instruments - a board, the task
list - read from a tracker over its HTTP wire, with each window's state
kept on the device the app runs on, in the app's own store
(`port/src/lib/window-store.ts`; the file is
`<app-data-dir>/window-store.json`, read and written through the
shell's two commands - t-71, revising t-14's placement in the
tracker's `/api/v1/windows`, which the app no longer reads): what it
shows and the part it reads (written once they settle, restored at the
next open, `port/src/lib/navigation/window-place.ts`), its link, and
the board's reading layout (the text zoom and the column width,
`port/src/lib/board/reading-layout.ts`). In the browser test project
the same store code runs over localStorage behind the mocked shell.
The app is the
round 16 prototype's behaviour, built and tested scenario by scenario
(t-36).

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
  menus, input, figures), `tests/` (vitest: unit tests in `tests/`,
  scenarios in `tests/scenarios/` with their kit), `vitest.config.ts`
  (the three projects), `vite.config.ts`
- `port-shell/` - the app's crate: its own workspace root, the binary
  `saccade-app`, and the tauri conf embedding `port/dist`
  (identifier `dev.saccade.app`)
- `src-tauri/src/` - the shell library the crate compiles: read-only
  declaration and asset commands with path guards, board admission
  through the shared `board-grammar` crate, the suite's window acts
  (`suite.rs`: windows, links, focus, the window jump), the window
  registry, the launch restore's session memory (`session.rs`) and the
  window store's file (`window_store.rs`), the look configuration, the
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
