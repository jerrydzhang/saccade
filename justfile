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

viewer-native-check:
    nix develop .#board-viewer -c bash -c 'cd client/src-tauri && cargo test'

viewer-shell:
    nix develop .#board-viewer

native-test: typecheck
    node --experimental-strip-types --test tracker/native/saccade.test.ts
    node --test tracker/native/claude-code/claude-code.test.mjs

[positional-arguments]
herdr-ui *args:
    nix run .#herdr-ui -- "$@"

herdr-ui-test:
    python3 -B tools/herdr-ui/herdr_ui_test.py -v
