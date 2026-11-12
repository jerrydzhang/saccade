fmt:
    cargo fmt

lint:
    cargo check
    cargo clippy

test:
    cargo test -q
    just port-test
    just port-test-browser

# The port's suite: one runner (vitest), three projects
# (client/port/vitest.config.ts, client/port/tests/scenarios/README.md).
# unit: the pure and component tests, in node.
port-test:
    cd client && node_modules/.bin/vitest run -c port/vitest.config.ts --project unit

# browser: the scenarios in headless WebKit against the port served by
# vite, with a throwaway seeded tracker from the built sac; the
# saccade-app shell carries the pinned Playwright browsers and the
# software-EGL mesa pin
port-test-browser *args:
    cargo build -q -p saccade --bin sac
    nix develop .#saccade-app -c bash -c 'cd client && SACCADE_SERVE_BIN="$PWD/../target/debug/sac" node_modules/.bin/vitest run -c port/vitest.config.ts --project browser "$@"' port-test-browser {{args}}

# pending: the tests tagged pending, written red before the feature
# they gate (skipped in every other run); unit, then browser
port-test-pending:
    cd client && SAC_PORT_SUITE_PENDING=1 node_modules/.bin/vitest run -c port/vitest.config.ts --project unit --tags-filter pending || true
    cargo build -q -p saccade --bin sac
    nix develop .#saccade-app -c bash -c 'cd client && SAC_PORT_SUITE_PENDING=1 SACCADE_SERVE_BIN="$PWD/../target/debug/sac" node_modules/.bin/vitest run -c port/vitest.config.ts --project browser --tags-filter pending'

# native: the scenarios tagged native or multi-window, in the built
# port app, through the session lane's suite mode (entry.py session
# suite: the enforced sandbox, the canary, a private Weston, then
# vitest's native project). Gated like every lane that spawns: the
# caller passes SAC_APP_RUNTIME_AUTHORIZED and SACCADE_DB (the
# caller-side tracker db the canary proves invisible); SAC_APP_BIN
# defaults to the nix-built port app.
port-test-native:
    cargo build -q -p saccade --bin sac
    nix develop .#saccade-app -c bash -c 'app="${SAC_APP_BIN:-$(nix build --no-link --print-out-paths .#saccade-app-port)/bin/saccade-app-port}"; SAC_APP_BIN="$app" SACCADE_SERVE_BIN="$PWD/target/debug/sac" python3 client/harness/entry.py --board "$PWD/client/fixtures/display-scroll.yaml" session suite'

typecheck:
    tracker/native/typecheck.sh

viewer-check:
    python3 client/harness/entry.py check-fast

# The port's debug binary for native-lane iteration (r88): vite builds
# the port dist, cargo builds the port crate with the embedded-dist
# feature, patchelf sets a store rpath so the binary runs inside the
# scrubbed lane sandbox, and the result is pinned at the lane's debug
# root. Nix still builds the authoritative binary once, last, per
# chunk - Nix reproduces this environment; it is not a per-iteration
# toll.
port-debug-bin:
    cd client && node_modules/.bin/vite build -c port/vite.config.ts
    nix develop .#saccade-app -c bash -c 'cargo build --manifest-path client/port-shell/Cargo.toml --features custom-protocol && bash client/harness/port_debug_pin.sh'

viewer-native-check:
    nix develop .#saccade-app -c bash -c 'cd client/src-tauri && cargo test'

viewer-shell:
    nix develop .#saccade-app

native-test: typecheck
    node --experimental-strip-types --test tracker/native/saccade.test.ts
    node --test tracker/native/claude-code/claude-code.test.mjs

[positional-arguments]
herdr-ui *args:
    nix run .#herdr-ui -- "$@"

herdr-ui-test:
    python3 -B tools/herdr-ui/herdr_ui_test.py -v
