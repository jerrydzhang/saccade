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
