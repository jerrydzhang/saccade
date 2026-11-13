# Saccade app

The Saccade suite's native app: a Svelte 5 frontend (`port/`) inside a
Tauri 2 shell (`port-shell/` over the shell library in
`src-tauri/src/`). Its windows hold the instruments - a board, the task
list - read from a tracker over its HTTP wire, with each window's state
kept in the tracker's window store (`/api/v1/windows`). The app is the
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

The native test project runs only through the gated session lane:
`just port-test-native` (below, and `port/tests/scenarios/README.md`).

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
  registry, the look configuration, the key trace's stderr door
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
`isolation.PidfdSupervisor`, canary first, gated by
`SAC_APP_RUNTIME_AUTHORIZED` and a caller-side tracker db
(`SACCADE_DB`) the canary proves invisible:

- `check-fast` - plain, uncontained tool runs (above)
- `preflight` - the minimal containment verification
- `session suite` - a private headless Weston around the built app, then
  vitest's native project (payload `suite_payload.py`); what
  `just port-test-native` runs
- `session up` / `up-t36` / `down` and the window verbs - the held
  WebDriver session the t36 Python port lane (`t36_*.py`) drives; that
  lane migrates into vitest next, then goes

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
