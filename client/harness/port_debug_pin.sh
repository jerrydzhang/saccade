#!/usr/bin/env bash
# Pin the cargo-built port debug binary for native-lane iteration
# (t-36@r88). The debug binary REPRODUCES the authoritative nix
# build's runtime - its interpreter and library set are read from the
# cached nix build (instant: --print-out-paths hits the cache) and
# patched onto the cargo binary - instead of inventing a mixed one.
# The devshells mix stdenvs (a 2.42 interpreter with 2.44-linked GUI
# libraries); only the authoritative closure is internally
# consistent. Nix still builds the authoritative binary once, last,
# per chunk; this pin is the iteration surface.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$ROOT/client/port-shell/target/debug/saccade-app
PIN_DIR=/var/tmp/saccade-app-1000/debug-bin

test -x "$BIN" || { echo "no debug binary at $BIN (run the cargo build first)" >&2; exit 1; }

NIX_APP=$(cd "$ROOT" && nix build --accept-flake-config .#saccade-app --print-out-paths --no-link 2>/dev/null | grep -o '/nix/store/[a-z0-9]*-saccade-app-0\.1\.0' | tail -1)
NIX_ELF="$NIX_APP/bin/.saccade-app-wrapped"
test -f "$NIX_ELF" || { echo "the cached authoritative build names no ELF at $NIX_ELF" >&2; exit 1; }
INTERP=$(patchelf --print-interpreter "$NIX_ELF")
NIX_RPATH=$(patchelf --print-rpath "$NIX_ELF")
GLIBC_DIR=$(dirname "$INTERP")
test -x "$INTERP" || { echo "no interpreter at $INTERP" >&2; exit 1; }
test -n "$NIX_RPATH" || { echo "the authoritative build carries no rpath" >&2; exit 1; }

# bootstrap: the authoritative interpreter AND its full library set,
# so the resolution below resolves everything (ldd omits unresolved
# libraries from its => lines, and the devshells resolve differently
# than the authoritative runtime)
patchelf --set-interpreter "$INTERP" --set-rpath "$NIX_RPATH:$GLIBC_DIR" "$BIN"
# the final rpath: the authoritative set UNION everything this
# binary additionally resolves against in that runtime
RESOLVED=$(ldd "$BIN" | awk '/=> \/nix\/store\//{print $3}' | xargs -r -n1 dirname | sort -u | paste -sd:)
LIBS=$(printf '%s:%s' "$NIX_RPATH" "$RESOLVED" | tr ':' '\n' | sort -u | paste -sd:)
test -n "$LIBS" || { echo "no rpath resolved for $BIN" >&2; exit 1; }
patchelf --set-rpath "$LIBS" "$BIN"

mkdir -p "$PIN_DIR"
install -m 0755 "$BIN" "$PIN_DIR/saccade-app.tmp"
mv "$PIN_DIR/saccade-app.tmp" "$PIN_DIR/saccade-app"
echo "debug binary pinned: $PIN_DIR/saccade-app (runtime reproduced from $NIX_APP)"
