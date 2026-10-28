#!/usr/bin/env bash
# The completions smoke: a scratch tracker served for real, the
# emitted fish script loaded the way fish loads vendor completions,
# completion probed end to end. Every candidate answer must carry
# record facts; nothing may record.
set -u
fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

BIN="$(pwd)/target/debug/sac"
test -x "$BIN" || fail "build the binary first"
# fish invokes sac by PATH name; the fresh build goes first so the
# probes exercise exactly what this tree emits
export PATH="$(dirname "$BIN"):${PATH}"

TAB=$(printf "\t")
SMOKE=$(mktemp -d /tmp/sac-fish-smoke-XXXXXX)
export XDG_STATE_HOME="$SMOKE/state"
export XDG_DATA_HOME="$SMOKE/data"
PORT=18931

# the scratch repo: one commit, a real checkout, one plain directory
# so the filesystem fallback has something ordinary to offer
git -C "$SMOKE" init -q repo
git -C "$SMOKE/repo" -c user.email=smoke@example -c user.name=smoke commit -q --allow-empty -m smoke
mkdir -p "$SMOKE/repo/realdir"
mkdir -p "$XDG_DATA_HOME/fish/vendor_completions.d"

# the server serves the scratch tracker; the address hint lands where
# discovery looks
"$BIN" serve --repo "$SMOKE/repo" --bind 127.0.0.1 --port $PORT >"$SMOKE/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null; rm -rf "$SMOKE"' EXIT
for i in $(seq 1 50); do
  curl -sf "http://127.0.0.1:$PORT/api/v1/version" >/dev/null 2>&1 && break
  [ "$i" = 50 ] && fail "server never came up: $(cat "$SMOKE/serve.log")"
  sleep 0.2
done

cd "$SMOKE/repo" || fail "cd repo"
unset SACCADE_ACTOR SACCADE_TASK SACCADE_INCARNATION_SEQ SACCADE_WORKSPACE SACCADE_REPO SACCADE_CLI
export SACCADE_ACTOR=smoker

# the story through real doors: a task, a claimed task, a workspace
"$BIN" create task "smoke the completions door" >/dev/null || fail "create task"
"$BIN" claim t-0 >/dev/null || fail "claim"
"$BIN" workspace new smokecell >/dev/null || fail "workspace new"
WS=$("$BIN" complete workspaces | head -1 | cut -f1)
test -n "$WS" || fail "no recorded workspace candidate"
test -d "$WS" || fail "the recorded path is not a directory"

# a real registration through the attach door
curl -sf -X POST "http://127.0.0.1:$PORT/api/v1/native/attach" \
  -H 'content-type: application/json' \
  -d '{"native_id":"smoke-sess-1","session_file":"'"$SMOKE"'/sessions/7777.json","client":"smoke"}' \
  >/dev/null || fail "attach refused"

# clean the workspace through the real door: the restore candidates
# then lead Cleaned
"$BIN" workspace clean "$WS" >/dev/null || fail "workspace clean"

# the script loads the way fish loads vendor completions; probes run
# in an interactive fish against the discovered server
"$BIN" completions fish > "$XDG_DATA_HOME/fish/vendor_completions.d/sac.fish" || fail "emit"
fish -i -c '
  function report
    echo "### $argv[1]"
    complete -C $argv[2]
  end
  report claim "sac claim "
  report cancel_flag "sac cancel t-0 --explanation "
  report launch_workspace "sac launch --workspace "
  report launch_resume "sac launch --resume "
  report send_to "sac send --to "
  report send_task "sac send --task "
  report switch_target "sac workspace switch "
  report switch_to "sac workspace switch i-0 --to "
  report clean "sac workspace clean "
  report restore "sac workspace restore "
  report use_source "sac use t-0 --source "
  report title_value "sac comment --title "
' > "$SMOKE/probes.out" 2> "$SMOKE/probes.err" || { cat "$SMOKE/probes.err"; fail "fish probes errored"; }
[ -s "$SMOKE/probes.err" ] && { cat "$SMOKE/probes.err"; fail "fish stderr noise"; }

# one probe section each, cut between the markers
section() {
  awk -v want="$1" '$0 == "### " want { on = 1; next } /^### / { on = 0 } on' "$SMOKE/probes.out"
}

section claim | grep -Fq "t-0${TAB}smoke the completions door (open)" || {
  section claim; fail "claim carries no task candidates"
}
[ -z "$(section cancel_flag)" ] || { section cancel_flag; fail "a flag value leaked candidates"; }

# launch --workspace: the recorded workspace AND the ordinary
# filesystem fallback beside it
section launch_workspace | grep -Fq "$WS" || { section launch_workspace; fail "launch misses the recorded workspace"; }
section launch_workspace | grep -q "realdir" || { section launch_workspace; fail "launch misses the directory fallback"; }

section launch_resume | grep -Fq "i-0${TAB}saccade native session smoke-sess-1" || {
  section launch_resume; fail "resume carries no incarnation candidates"
}
section send_to | grep -Fq "i-0${TAB}saccade native session smoke-sess-1" || { section send_to; fail "send --to misses incarnations"; }
section send_task | grep -Fq "t-0${TAB}smoke the completions door" || { section send_task; fail "send --task misses tasks"; }
section switch_target | grep -Fq "i-0${TAB}saccade native session smoke-sess-1" || { section switch_target; fail "switch misses incarnations"; }

# switch --to: recorded beside the fallback
section switch_to | grep -Fq "$WS" || { section switch_to; fail "switch --to misses the recorded workspace"; }
section switch_to | grep -q "realdir" || { section switch_to; fail "switch --to misses the fallback"; }

# clean and restore: recorded paths only, no filesystem fallback
section clean | grep -Fq "$WS" || { section clean; fail "clean misses the recorded path"; }
section clean | grep -q "realdir" && fail "clean leaked the filesystem fallback"
section restore | grep -Fq "$WS" || { section restore; fail "restore misses the recorded path"; }
section restore | grep -q "realdir" && fail "restore leaked the filesystem fallback"

section use_source | grep -Fq "t-0${TAB}smoke the completions door" || { section use_source; fail "use --source misses tasks"; }
[ -z "$(section title_value)" ] || { section title_value; fail "a title value leaked candidates"; }

# completion never records: the record count is unchanged
BEFORE=$("$BIN" log | grep -c '^seq ')
section claim >/dev/null
section send_to >/dev/null
AFTER=$("$BIN" log | grep -c '^seq ')
[ "$BEFORE" = "$AFTER" ] || fail "completion recorded ($BEFORE -> $AFTER)"

echo "SMOKE OK: script emits, candidates carry record facts, fallback sits beside recorded paths only where ratified, nothing records"
