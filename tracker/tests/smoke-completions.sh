#!/usr/bin/env bash
# The completions smoke: a scratch tracker served for real, the
# emitted fish script loaded the way fish loads vendor completions,
# completion probed end to end. Every candidate answer must carry
# record facts; nothing may record.
set -u
fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# A session's routing (SACCADE_SERVER, SACCADE_REPO, SACCADE_ACTOR,
# the scope words, the HC_* terminal words) must never steer this
# smoke at the real tracker: every inherited SACCADE_* and HC_* word
# goes before the first sac invocation, and the smoke builds its own
# environment on the bare base.
for name in $(compgen -v SACCADE_; compgen -v HC_); do
  unset "$name"
done
# belt and braces: the scrub leaves nothing behind
survivor=$(compgen -v SACCADE_; compgen -v HC_)
[ -z "$survivor" ] || fail "inherited tracker environment survived the scrub: $survivor"

BIN="$(pwd)/target/debug/sac"
test -x "$BIN" || fail "build the binary first"
# fish invokes sac by PATH name; the fresh build goes first so the
# probes exercise exactly what this tree emits
PATH="$(dirname "$BIN"):${PATH}"
export PATH

TAB=$(printf "\t")
# the scratch honors the caller's TMPDIR; the tracker inside it is
# entirely its own
SMOKE=$(mktemp -d "${TMPDIR:-/tmp}/sac-fish-smoke.XXXXXX")
export XDG_STATE_HOME="$SMOKE/state"
export XDG_DATA_HOME="$SMOKE/data"
export SACCADE_ACTOR=smoker

# workspace clean's host step reads the host's terminal table over one
# `herdr-core ls --json` ask, and that ask must never join the
# machine's real herdr-core. The resolver's ambient arm walks PATH for
# an executable named herdr-core - the SACCADE_HERDR override is
# scrubbed above and never set here, and a compile-time pin sits below
# the ambient arm - so the stand-in takes that exact name and the
# scratch dir holding it goes first on PATH. The tests' fake_hc is the
# shape: argv lands in the scratch dir as evidence, the listing verb
# answers an empty terminal table so clean's closure verification
# passes, any other verb refuses nonzero with a note.
FAKE_HOST="$SMOKE/fake-host"
mkdir -p "$FAKE_HOST"
cat > "$FAKE_HOST/herdr-core" <<'EOF'
#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s\n' "$*" >> "$here/argv-$1.log"
if [ "$1" = ls ] && [ "$2" = --json ]; then
  printf '{"terminals":[]}\n'
  exit 0
fi
printf 'the smoke fake answers ls --json only\n' >&2
exit 64
EOF
chmod +x "$FAKE_HOST/herdr-core"
PATH="$FAKE_HOST:$PATH"
export PATH

# the scratch repo: one commit, a real checkout, one plain directory
# so the filesystem fallback has something ordinary to offer
git -C "$SMOKE" init -q repo
git -C "$SMOKE/repo" -c user.email=smoke@example -c user.name=smoke commit -q --allow-empty -m smoke
mkdir -p "$SMOKE/repo/realdir"
mkdir -p "$XDG_DATA_HOME/fish/vendor_completions.d"

# the server serves the scratch tracker; port 0 asks the kernel for a
# free port, and the resolved address is published in the hint beside
# the tracker's database, exactly where discovery looks
"$BIN" serve --repo "$SMOKE/repo" --bind 127.0.0.1 --port 0 >"$SMOKE/serve.log" 2>&1 &
SERVE_PID=$!
# SMOKE_KEEP=1 preserves the scratch dir for inspection
trap 'kill $SERVE_PID 2>/dev/null; if [ "${SMOKE_KEEP:-}" != 1 ]; then rm -rf "$SMOKE"; fi' EXIT
URL=""
for i in $(seq 1 100); do
  hint=$(find "$XDG_STATE_HOME" -name '*.addr' -print -quit 2>/dev/null)
  if [ -n "$hint" ] && [ -s "$hint" ]; then
    URL=$(cat "$hint")
    break
  fi
  [ "$i" = 100 ] && fail "the server never published its address: $(cat "$SMOKE/serve.log")"
  sleep 0.1
done
[ -n "$URL" ] || fail "the server published no address: $(cat "$SMOKE/serve.log")"
for i in $(seq 1 50); do
  curl -sf "$URL/api/v1/version" >/dev/null 2>&1 && break
  [ "$i" = 50 ] && fail "server never came up: $(cat "$SMOKE/serve.log")"
  sleep 0.2
done

cd "$SMOKE/repo" || fail "cd repo"

# the story through real doors: a task, a claimed task, a workspace
"$BIN" create task "smoke the completions door" >/dev/null || fail "create task"
"$BIN" claim t-0 >/dev/null || fail "claim"
"$BIN" workspace new smokecell >/dev/null || fail "workspace new"
WS=$("$BIN" complete workspaces | head -1 | cut -f1)
test -n "$WS" || fail "no recorded workspace candidate"
test -d "$WS" || fail "the recorded path is not a directory"

# a real registration through the attach door
curl -sf -X POST "$URL/api/v1/native/attach" \
  -H 'content-type: application/json' \
  -d '{"native_id":"smoke-sess-1","session_file":"'"$SMOKE"'/sessions/7777.json","client":"smoke"}' \
  >/dev/null || fail "attach refused"

# clean the workspace through the real door
"$BIN" workspace clean "$WS" >/dev/null || fail "workspace clean"
# the clean door's host read went through the stand-in: the evidence
# log names the one ask, or the resolver walked past the fake
ls "$FAKE_HOST"/argv-*.log >/dev/null 2>&1 || fail "clean asked no host at all"
grep -qx "ls --json" "$FAKE_HOST/argv-ls.log" || {
  cat "$FAKE_HOST"/argv-*.log
  fail "workspace clean never asked the fake host"
}

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
  report clean "sac workspace clean "
  report board_static "sac board "
  report offer_tasks "sac offer "
  report create_static "sac create "
  report skill_static "sac skill "
  report shell_static "sac completions "
  report global_flag "sac --ac"
  report cancel_flag_name "sac cancel t-0 --expl"
  report cancel_no_leak "sac cancel t-0 --wor"
  report launch_flag_name "sac launch --work"
  report launch_no_leak "sac launch --re"
  report reject_flag_name "sac reject t-0 --gro"
  report global_beside_verb "sac cancel --j"
  report flag_value_still_clean "sac cancel --explanation "
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

# clean: recorded paths only, no filesystem fallback
section clean | grep -Fq "$WS" || { section clean; fail "clean misses the recorded path"; }
section clean | grep -q "realdir" && fail "clean leaked the filesystem fallback"

# the board controller's verbs complete statically; the decision verbs
# name their tasks
section board_static | grep -qx "edit" || { section board_static; fail "board lost edit"; }
section board_static | grep -qx "commit" || { section board_static; fail "board lost commit"; }
section board_static | grep -qx "path" || { section board_static; fail "board lost path"; }
section offer_tasks | grep -Fq "t-0${TAB}smoke the completions door" || { section offer_tasks; fail "offer misses tasks"; }

# static skeleton: create completes; multi-candidate registrations keep
# every candidate; a static description survives in one piece
section create_static | grep -qx "task" || { section create_static; fail "create offers no task kind"; }
section skill_static | grep -qx "install" || { section skill_static; fail "skill lost install"; }
section skill_static | grep -qx "check" || { section skill_static; fail "skill lost check"; }
section shell_static | grep -Fqe "fish${TAB}the primary shell" || { section shell_static; fail "completions lost the fish description"; }
section shell_static | grep -qx "primary" && fail "the fish description split on spaces"
section shell_static | grep -qx "shell" && fail "the fish description split on spaces"

# flag names: global position-independent, per-verb names exactly as
# the CLI defines them, no cross-verb leaks, values stay clean
section global_flag | grep -Fqe "--actor${TAB}actor name recorded on events" || { section global_flag; fail "global flag names do not complete"; }
section cancel_flag_name | grep -Fqe "--explanation" || { section cancel_flag_name; fail "cancel misses --explanation"; }
[ -z "$(section cancel_no_leak)" ] || { section cancel_no_leak; fail "another verb's flag leaked into cancel"; }
section launch_flag_name | grep -Fqe "--workspace" || { section launch_flag_name; fail "launch misses --workspace"; }
section launch_no_leak | grep -Fqe "--resume" || { section launch_no_leak; fail "launch misses --resume"; }
section reject_flag_name | grep -Fqe "--grounds" || { section reject_flag_name; fail "reject misses --grounds"; }
section global_beside_verb | grep -Fqe "--json${TAB}machine-readable output" || { section global_beside_verb; fail "global flags do not ride beside the per-verb set"; }
[ -z "$(section flag_value_still_clean)" ] || { section flag_value_still_clean; fail "a flag name leaked into its own value"; }

# completion never records: the record count is unchanged
BEFORE=$("$BIN" log | grep -c '^seq ')
section claim >/dev/null
section send_to >/dev/null
AFTER=$("$BIN" log | grep -c '^seq ')
[ "$BEFORE" = "$AFTER" ] || fail "completion recorded ($BEFORE -> $AFTER)"

echo "SMOKE OK: script emits, candidates carry record facts, fallback sits beside recorded paths only where ratified, nothing records"
