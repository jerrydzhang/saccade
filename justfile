fmt:
    cargo fmt

lint:
    cargo check
    cargo clippy

test:
    cargo test -q

typecheck:
    native/typecheck.sh

viewer-check:
    python3 viewer/harness/entry.py check-fast

viewer-shell:
    nix develop .#board-viewer

native-test: typecheck
    node --experimental-strip-types --test native/saccade.test.ts
    node --test native/claude-code/claude-code.test.mjs

web-build:
    cd web && npm run build

web-test:
    cd web && npm run build && node --test tests/*.test.mjs

web-browser-test:
    cd web/harness && python3 run_tests.py

[positional-arguments]
herdr-ui *args:
    nix run .#herdr-ui -- "$@"

herdr-ui-test:
    python3 -B dev/herdr_ui_test.py -v
