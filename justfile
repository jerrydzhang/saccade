fmt:
    cargo fmt

lint:
    cargo check
    cargo clippy

test:
    cargo test -q

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
