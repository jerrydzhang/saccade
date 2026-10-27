fmt:
    cargo fmt

lint:
    cargo check
    cargo clippy

test:
    cargo test -q

typecheck:
    native/typecheck.sh

native-test: typecheck
    node --experimental-strip-types --test native/saccade.test.ts
    node --import "$SACCADE_COMPOSITION_PRELOAD/saccade-composition-shim.mjs" --experimental-strip-types --test native/runtime-acceptance.test.ts

web-build:
    cd web && npm run build

web-test:
    cd web && npm run build && node --test tests/*.test.mjs

web-browser-test:
    cd web/harness && python3 run_tests.py
