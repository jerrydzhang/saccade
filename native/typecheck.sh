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
for _ in 1 2 3 4; do
	refs=$(grep -ohE '/nix/store/[^ "]+/bin/(saccade-pi|wrapped-pi|pi)' "$current_file" 2>/dev/null | sort -u || true)
	[ -n "$refs" ] || { echo "typecheck: no store reference in $current_file" >&2; exit 2; }
	next_file=""
	for ref in $refs; do
		pkg=$(dirname "$(dirname "$ref")")
		if [ -d "$pkg/lib/pi/node_modules/@earendil-works" ]; then
			node_modules="$pkg/lib/pi/node_modules"
			break
		fi
		# the referenced binary is the next wrapper script in the chain
		next_file="$ref"
	done
	if [ -n "$node_modules" ] || [ ! -f "$next_file" ]; then
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
			"typebox": ["$node_modules/typebox"]
		}
	}
}
TSCONFIG
# the runtime link: saccade.ts imports typebox as a value (the schema
# builders), so the plain node test command needs it resolvable
mkdir -p native/node_modules
ln -sfn "$node_modules/typebox" native/node_modules/typebox
if [ "$setup_only" = "--setup-only" ]; then
	echo "typecheck: pinned SDK resolved at $node_modules; config and runtime link written"
	exit 0
fi
exec tsc -p native/tsconfig.local.json
