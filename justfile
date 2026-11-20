fmt:
    cargo fmt --manifest-path saccade/Cargo.toml
    cargo fmt --manifest-path app/Cargo.toml -p saccade-app-core
    nixfmt flake.nix nix/*.nix saccade/nix/*.nix app/nix/*.nix

lint:
    cargo clippy --manifest-path saccade/Cargo.toml -p saccade -p board-grammar --all-targets -- -D warnings

# the full suite: the cargo workspace, the app shell's own tests, the
# app frontend's vitest unit and browser projects, then the native
# node suites (the vitest native project is gated: just app-test-native)
test:
    cargo test -q --manifest-path saccade/Cargo.toml
    just shell-test
    just app-test
    just app-test-browser
    just native-test

# the app shell library's tests (app/src-tauri/src, compiled through
# app/port-shell): the GTK/WebKit build needs the saccade-app shell
shell-test:
    nix develop .#saccade-app -c bash -c 'cargo test -q --manifest-path app/port-shell/Cargo.toml'

# keep app/node_modules at the lockfile: npm writes its own copy of
# the lock into node_modules/.package-lock.json on every install, so a
# lockfile newer than that copy (a merge, pull or switch that changed
# it, or a fresh worktree) means the install is stale; every recipe
# that runs the frontend's node_modules depends on this
client-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    cd app
    if [ ! -e node_modules/.package-lock.json ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
        npm ci
    fi

# The port's suite: one runner (vitest), three projects
# (app/port/vitest.config.ts, app/port/tests/scenarios/README.md).
# unit: the pure and component tests, in node.
app-test: client-deps
    cd app && node_modules/.bin/vitest run -c port/vitest.config.ts --project unit

# browser: the scenarios in headless WebKit against the port served by
# vite, with a throwaway seeded tracker from the built sac; the
# saccade-app shell carries the pinned Playwright browsers and the
# software-EGL mesa pin
app-test-browser *args: client-deps
    cargo build -q --manifest-path saccade/Cargo.toml -p saccade --bin sac
    nix develop .#saccade-app -c bash -c 'cd app && SACCADE_SERVE_BIN="$PWD/../saccade/target/debug/sac" node_modules/.bin/vitest run -c port/vitest.config.ts --project browser "$@"' app-test-browser {{args}}

# pending: the tests tagged pending, written red before the feature
# they gate (skipped in every other run); unit, then browser
app-test-pending: client-deps
    cd app && SAC_PORT_SUITE_PENDING=1 node_modules/.bin/vitest run -c port/vitest.config.ts --project unit --tags-filter pending || true
    cargo build -q --manifest-path saccade/Cargo.toml -p saccade --bin sac
    nix develop .#saccade-app -c bash -c 'cd app && SAC_PORT_SUITE_PENDING=1 SACCADE_SERVE_BIN="$PWD/../saccade/target/debug/sac" node_modules/.bin/vitest run -c port/vitest.config.ts --project browser --tags-filter pending'

# native: the scenarios tagged native or multi-window, in the built
# port app, through the session lane's suite mode (entry.py session
# suite: the enforced sandbox, the canary, a private Weston, then
# vitest's native project). Like every lane that spawns: the caller
# passes SACCADE_DB (the caller-side tracker db the canary proves
# invisible); SAC_APP_BIN defaults to the nix-built port app.
app-test-native *args:
    cargo build -q --manifest-path saccade/Cargo.toml -p saccade --bin sac
    nix develop .#saccade-app -c bash -c 'app="${SAC_APP_BIN:-$(nix build --no-link --print-out-paths .#saccade-app)/bin/saccade-app}"; SAC_APP_BIN="$app" SACCADE_SERVE_BIN="$PWD/saccade/target/debug/sac" python3 app/harness/entry.py --board "$PWD/app/fixtures/display-scroll.yaml" session suite "$@"' app-test-native {{args}}

typecheck:
    saccade/tracker/native/typecheck.sh

# the plain, uncontained checks: the harness's own checks, pyflakes, the
# app frontend's typecheck, the core client's contract tests, the build
viewer-check: client-deps
    python3 app/harness/entry.py check-fast

# The port's debug binary for native-lane iteration (r88): vite builds
# the port dist, cargo builds the port crate with the embedded-dist
# feature, patchelf sets a store rpath so the binary runs inside the
# scrubbed lane sandbox, and the result is pinned at the lane's debug
# root. Nix still builds the authoritative binary once, last, per
# chunk - Nix reproduces this environment; it is not a per-iteration
# toll.
port-debug-bin: client-deps
    cd app && node_modules/.bin/vite build -c port/vite.config.ts
    nix develop .#saccade-app -c bash -c 'cargo build --manifest-path app/port-shell/Cargo.toml --features custom-protocol && bash app/harness/port_debug_pin.sh'

viewer-shell:
    nix develop .#saccade-app

native-test: typecheck
    node --experimental-strip-types --test saccade/tracker/native/saccade.test.ts
    node --test saccade/tracker/native/claude-code/claude-code.test.mjs
