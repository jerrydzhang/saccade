#!/bin/sh
# Throwaway fake-host smoke for the root destination contract
# (t-73, commit 28805d5 onward). Every CLI invocation is fully
# scrubbed: no ambient SACCADE_* variable, no HC_TERMINAL_ID, and the
# host adapter is ALWAYS the fake - the adapter never resolves
# ambiently. This hygiene exists because the first, discarded smoke
# run under-scrubbed one case: `env -u SACCADE_ACTOR sac ... workspace
# root bind` inherited this session's HC_TERMINAL_ID and no
# SACCADE_HERDR, so the CLI resolved the ambient host adapter and one
# read-only `hc get` for this pane's own record hit the real host. No
# host mutation, label write, launch, or move occurred; the write went
# to a throwaway db since deleted. The corrected script below fixed
# every invocation and is the only version kept or run since.
set -u
SAC=${SAC_SMOKE_BIN:-target/debug/sac}
DIR=$(mktemp -d /tmp/sac-smoke-XXXXXX)
git -C "$DIR" init -q repo
cd "$DIR/repo"
git -c user.email=s@l -c user.name=s commit -q --allow-empty -m first
mkdir -p other-dir "$DIR/bin"
cat > "$DIR/bin/hc" <<'HC'
#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s\n' "$@" >> "$here/argv-$1.log"
if [ "$1" = get ]; then
  if [ "$2" = tc1 ]; then
    printf '{"state":"live","labels":{"herdr.space":"spaManual","herdr.tab":"tbManual"}}\n'
    exit 0
  fi
  printf '{"state":"live"}\n'
  exit 0
fi
if [ "$1" = label ]; then exit 0; fi
printf '{"address":"t7","cwd":"x","state":"live","turn":0}\n'
HC
chmod +x "$DIR/bin/hc"
SACCADE_HERDR="$DIR/bin/hc" "$SAC" serve --port 0 --db "$DIR/smoke.db" --repo "$DIR/repo" >/dev/null 2>&1 &
SERVE_PID=$!
sleep 1
URL=$(cat "$DIR/smoke.db.addr")
CLR="env -u SACCADE_ACTOR -u SACCADE_TASK -u SACCADE_INCARNATION_SEQ -u SACCADE_WORKSPACE -u SACCADE_REPO -u SACCADE_SERVER -u HC_TERMINAL_ID -u SACCADE_HERDR -u SACCADE_LAUNCHED_SESSION -u SACCADE_CLI -u SACCADE_PI"
H="$DIR/bin/hc"; R="$DIR/repo"
echo "== 1 show unbound"
$CLR "$SAC" --server "$URL" --repo "$R" workspace root show
echo "== 2 bind without HC_TERMINAL_ID (expect refuse)"
$CLR "$SAC" --server "$URL" --repo "$R" workspace root bind; echo "exit=$?"
echo "== 3 bind unlabeled tc2 (expect refuse)"
$CLR HC_TERMINAL_ID=tc2 "$SAC" --server "$URL" --repo "$R" workspace root bind; echo "exit=$?"
echo "== 4 agent bind tc1 (expect human_only after the get)"
env SACCADE_ACTOR=pi HC_TERMINAL_ID=tc1 SACCADE_HERDR="$H" "$SAC" --server "$URL" --repo "$R" workspace root bind; echo "exit=$?"
echo "   get argv so far:"; cat "$DIR/bin/argv-get.log"
echo "== 5 wire human bind"
curl -s -X POST "$URL/api/v1/command" -H 'content-type: application/json' -d '{"context":{"actor":"jerry","tier":"human"},"command":{"bind_root_space":{"space":"spaManual","tab":"tbManual"}}}' | head -c 200; echo
echo "== 6 show bound (text + json)"
$CLR "$SAC" --server "$URL" --repo "$R" workspace root show
$CLR "$SAC" --server "$URL" --repo "$R" --json workspace root show
echo "== 7 launch at the root declares the bound pair"
$CLR SACCADE_HERDR="$H" SACCADE_PI=/bin/true "$SAC" --server "$URL" --repo "$R" --json launch --workspace "$R" | grep -E '"space"|"tab"|"place_refusal"|"terminal"'
echo "   label argv:"; cat "$DIR/bin/argv-label.log"
echo "== 8 launch elsewhere unplaced (text face)"
$CLR SACCADE_HERDR="$H" SACCADE_PI=/bin/true "$SAC" --server "$URL" --repo "$R" launch --workspace "$DIR/repo/other-dir" | grep -E "unplaced|integrated"
echo "== 9 agent clear (expect human_only)"
env SACCADE_ACTOR=pi "$SAC" --server "$URL" --repo "$R" workspace root clear; echo "exit=$?"
echo "== 10 wire human clear; show unbound again"
curl -s -X POST "$URL/api/v1/command" -H 'content-type: application/json' -d '{"context":{"actor":"jerry","tier":"human"},"command":{"clear_root_space":{}}}' | head -c 160; echo
$CLR "$SAC" --server "$URL" --repo "$R" workspace root show
kill $SERVE_PID 2>/dev/null
sleep 0.3
cd /; rm -rf "$DIR"
echo "SMOKE-DONE"
