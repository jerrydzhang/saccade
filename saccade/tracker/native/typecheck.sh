#!/usr/bin/env bash
# Typecheck the native production code and its test fake against the
# PINNED SDK's own types. Resolves the SDK through the SACCADE_PI
# chain (dev wrapper -> repo saccade-pi wrapper -> wrapped-pi -> the
# pinned pi package, whose lib/pi/node_modules carries the pinned
# @earendil-works SDK and typebox), writes the generated tsconfig
# beside this script (gitignored), links the runtime typebox under
# native/node_modules so the plain node test command resolves it, and
# runs the dev shell's typescript. No npm install, no lock changes.
# Setup only (no tsc run): pass --setup-only.
set -euo pipefail
cd "$(dirname "$0")/.."
setup_only="${1:-}"
dev="${SACCADE_PI:-}"
[ -n "$dev" ] || {
	echo "typecheck: SACCADE_PI is unset; run inside the dev shell (nix develop), which exports it" >&2
	exit 2
}
dev=$(readlink -f "$dev")
current_file="$dev"
node_modules=""
for _ in 1 2 3 4 5 6; do
	# store paths on the script's own exec line are chain hops: the
	# wrapper bin it execs, or the pinned cli bundle naming the pinned
	# dist directly - the -e extension arguments are dead ends and
	# never followed
	exec_line=$(grep '^exec ' "$current_file" 2>/dev/null | tail -1)
	all_refs=$(grep -ohE '/nix/store/[^ "[:space:]]+' <<< "$exec_line" 2>/dev/null | sort -u || true)
	[ -n "$all_refs" ] || { echo "typecheck: no store reference in $current_file" >&2; exit 2; }
	# the exec target is the chain continuation by construction; the
	# remaining store refs are ride-along arguments, checked for the
	# SDK but never followed as the chain
	primary=$(grep -ohE '/nix/store/[^ "[:space:]]+' <<< "$exec_line" | head -1)
	next_file=""
	for ref in $all_refs; do
		pkg="${ref%/bin/*}"
		if [ "$pkg" != "$ref" ] && [ -d "$pkg/lib/pi/node_modules/@earendil-works" ]; then
			node_modules="$pkg/lib/pi/node_modules"
			break
		fi
		# a store path pointing at the pinned dist names the SDK types
		# directly (the generated launch chain execs the cli bundle)
		case $ref in
			*/dist/bundle/cli.js)
				if [ -f "$(dirname "$ref")/../index.d.ts" ]; then
					node_modules="$(cd "$(dirname "$ref")/../../../.." && pwd)"
					break
				fi
				;;
		esac
	done
	if [ -n "$node_modules" ]; then
		break
	fi
	# no SDK in this hop: the exec target continues the chain when it
	# is a readable script; store binaries are skipped, never followed
	if [ -f "$primary" ] && grep -qI . "$primary" 2>/dev/null; then
		next_file="$primary"
	fi
	if [ -n "$node_modules" ] || [ -z "$next_file" ]; then
		break
	fi
	current_file="$next_file"
done
[ -n "$node_modules" ] || { echo "typecheck: the pinned pi package was never found in the wrapper chain" >&2; exit 2; }
sdk="$node_modules/@earendil-works/pi-coding-agent/dist/index.d.ts"
[ -f "$sdk" ] || { echo "typecheck: the pinned SDK types are missing at $sdk" >&2; exit 2; }
cat > native/tsconfig.local.json <<TSCONFIG
{
	"extends": "./tsconfig.json",
	"compilerOptions": {
		"paths": {
			"@earendil-works/pi-coding-agent": ["$sdk"],
			"@earendil-works/pi-tui": ["$node_modules/@earendil-works/pi-tui"],
			"typebox": ["$node_modules/typebox"]
		}
	}
}
TSCONFIG
# the runtime links: saccade.ts imports typebox and pi-tui as values,
# and the acceptance smoke drives the pinned SDK itself, so the plain
# node test command needs them resolvable
mkdir -p native/node_modules native/node_modules/@earendil-works
ln -sfn "$node_modules/typebox" native/node_modules/typebox
ln -sfn "$node_modules/@earendil-works/pi-tui" native/node_modules/@earendil-works/pi-tui
ln -sfn "$node_modules/@earendil-works/pi-coding-agent" native/node_modules/@earendil-works/pi-coding-agent
if [ "$setup_only" = "--setup-only" ]; then
	echo "typecheck: pinned SDK resolved at $node_modules; config and runtime link written"
	exit 0
fi
exec tsc -p native/tsconfig.local.json
