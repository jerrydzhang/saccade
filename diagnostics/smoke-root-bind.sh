#!/bin/sh
# Throwaway fake-host smoke for the root destination contract (t-73).
# ISOLATION LAW, learned from the under-scrubbed first runs: every CLI
# invocation takes ONE path that names the host adapter explicitly -
# the fake for host-answering cases, a denying adapter for cases that
# must never ask - and scrubs every ambient tracker routing, scope,
# and pane key. The adapter never resolves ambiently; an assertion on
# the fake's own argv log proves which adapter answered, so a fallback
# cannot masquerade as a refusal. Cleanup kills only the serve
# subprocess this script started. The run is judged by its
# assertions: SMOKE-PASS with exit 0, or SMOKE-FAIL with exit 1 - a
# final line alone is never a pass judge.
#
# History: the original run under-scrubbed two cases (an inherited
# HC_TERMINAL_ID and a removed-but-not-replaced SACCADE_HERDR), and
# two read-only gets resolved the ambient host adapter. Those exact
# bytes are published as t-73 artifacts
# smoke-root-bind-28805d5-as-ran.sh/.log; the fake's argv log showing
# only tc1 is itself the evidence of the second fallback.
set -u
SAC=${SAC_SMOKE_BIN:-target/debug/sac}
DIR=$(mktemp -d /tmp/sac-smoke-XXXXXX)
FAILS=0
fail() {
  echo "FAIL: $*"
  FAILS=$((FAILS + 1))
}
git -C "$DIR" init -q repo
cd "$DIR/repo" || exit 1
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
printf '#!/bin/sh\nexit 3\n' > "$DIR/bin/deny-hc"
chmod +x "$DIR/bin/deny-hc"
HERDR="$DIR/bin/hc"
DENY="$DIR/bin/deny-hc"
SACCADE_HERDR="$HERDR" "$SAC" serve --port 0 --db "$DIR/smoke.db" --repo "$DIR/repo" >/dev/null 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT
sleep 1
URL=$(cat "$DIR/smoke.db.addr")
# the one invocation path: the adapter is always named, everything
# ambient is always scrubbed
cli() {
  adapter="$1"
  shift
  env -u SACCADE_ACTOR -u SACCADE_TASK -u SACCADE_INCARNATION_SEQ \
    -u SACCADE_WORKSPACE -u SACCADE_REPO -u SACCADE_SERVER \
    -u HC_TERMINAL_ID -u SACCADE_LAUNCHED_SESSION -u SACCADE_CLI \
    -u SACCADE_PI SACCADE_HERDR="$adapter" "$SAC" "$@"
}
get_log() { cat "$DIR/bin/argv-get.log" 2>/dev/null; }
label_log() { cat "$DIR/bin/argv-label.log" 2>/dev/null; }

echo "== 1 show unbound (deny adapter: a host ask would fail loudly)"
OUT=$(cli "$DENY" --server "$URL" --repo "$DIR/repo" workspace root show) \
  && [ "$OUT" = "no root space is bound: root launches run unplaced" ] \
  || fail "case 1: unbound show: $OUT"

echo "== 2 bind without HC_TERMINAL_ID (expect refuse before any ask)"
BEFORE=$(get_log)
if cli "$DENY" --server "$URL" --repo "$DIR/repo" workspace root bind >/dev/null 2>&1; then
  fail "case 2: bind without a pane declared succeeded"
fi
AFTER=$(get_log)
[ "$BEFORE" = "$AFTER" ] || fail "case 2: a get ask rode the paneless refusal"

echo "== 3 bind unlabeled tc2 (expect refuse; the FAKE must answer tc2)"
OUT=$(cli "$HERDR" HC_TERMINAL_ID=tc2 --server "$URL" --repo "$DIR/repo" workspace root bind 2>&1) \
  && fail "case 3: unlabeled bind succeeded: $OUT"
printf '%s' "$OUT" | grep -q "no live standing placement" \
  || fail "case 3: wrong refusal: $OUT"
get_log | grep -q "tc2" \
  || fail "case 3: the fake's get log lacks tc2 - the adapter fell back ambient"

echo "== 4 agent bind tc1 (expect human_only after the fake's get)"
OUT=$(env SACCADE_ACTOR=pi HC_TERMINAL_ID=tc1 cli "$HERDR" --server "$URL" --repo "$DIR/repo" workspace root bind 2>&1) \
  && fail "case 4: agent bind succeeded: $OUT"
printf '%s' "$OUT" | grep -q "human_only" || fail "case 4: wrong refusal: $OUT"
get_log | grep -q "tc1" || fail "case 4: the fake's get log lacks tc1"

echo "== 5 wire human bind"
REPLY=$(curl -s -X POST "$URL/api/v1/command" -H 'content-type: application/json' \
  -d '{"context":{"actor":"jerry","tier":"human"},"command":{"bind_root_space":{"space":"spaManual","tab":"tbManual"}}}')
printf '%s' "$REPLY" | grep -q '"kind":"root_space_bound"' || fail "case 5: no bind record: $REPLY"
printf '%s' "$REPLY" | grep -q '"tier":"human"' || fail "case 5: bind tier not human: $REPLY"

echo "== 6 show bound (deny adapter; text and json name the exact pair)"
OUT=$(cli "$DENY" --server "$URL" --repo "$DIR/repo" workspace root show) \
  && [ "$OUT" = "root space: spaManual / tbManual" ] \
  || fail "case 6: bound show: $OUT"
J=$(cli "$DENY" --server "$URL" --repo "$DIR/repo" --json workspace root show)
printf '%s' "$J" | grep -q '"bound": true' || fail "case 6: json bound: $J"
printf '%s' "$J" | grep -q '"tab": "tbManual"' || fail "case 6: json tab: $J"

echo "== 7 launch at the root declares the bound pair"
OUT=$(cli "$HERDR" SACCADE_PI=/bin/true --server "$URL" --repo "$DIR/repo" --json launch --workspace "$DIR/repo")
printf '%s' "$OUT" | grep -q '"space": "spaManual"' || fail "case 7: space: $OUT"
printf '%s' "$OUT" | grep -q '"tab": "tbManual"' || fail "case 7: tab: $OUT"
printf '%s' "$OUT" | grep -q '"place_refusal": null' || fail "case 7: place_refusal: $OUT"
label_log | grep -q "herdr.space=spaManual" || fail "case 7: label space ask missing"
label_log | grep -q "herdr.tab=tbManual" || fail "case 7: label tab ask missing"

echo "== 8 launch elsewhere unplaced"
OUT=$(cli "$HERDR" SACCADE_PI=/bin/true --server "$URL" --repo "$DIR/repo" launch --workspace "$DIR/repo/other-dir")
printf '%s' "$OUT" | grep -q "unplaced: no Herdr space was declared" \
  || fail "case 8: unplaced line: $OUT"

echo "== 9 agent clear (expect human_only; deny adapter)"
if env SACCADE_ACTOR=pi cli "$DENY" --server "$URL" --repo "$DIR/repo" workspace root clear >/dev/null 2>&1; then
  fail "case 9: agent clear succeeded"
fi

echo "== 10 wire human clear; show unbound again"
REPLY=$(curl -s -X POST "$URL/api/v1/command" -H 'content-type: application/json' \
  -d '{"context":{"actor":"jerry","tier":"human"},"command":{"clear_root_space":{}}}')
printf '%s' "$REPLY" | grep -q '"kind":"root_space_cleared"' || fail "case 10: no clear record: $REPLY"
OUT=$(cli "$DENY" --server "$URL" --repo "$DIR/repo" workspace root show) \
  && [ "$OUT" = "no root space is bound: root launches run unplaced" ] \
  || fail "case 10: final show: $OUT"

cd / || exit 1
rm -rf "$DIR"
if [ "$FAILS" -eq 0 ]; then
  echo "SMOKE-PASS"
  exit 0
fi
echo "SMOKE-FAIL: $FAILS assertion(s) failed"
exit 1
