#!/usr/bin/env bash
# Vetting transcript: exercises every user-visible output surface of `sac`,
# labeled and deterministic (timestamps normalized), for terminal review.
#
#   ./vet-output.sh                 # build + run against target/debug/saccade
#   ./vet-output.sh | less -R       # page through it as rendered
#   ./vet-output.sh > /tmp/t1; ... > /tmp/t2; diff /tmp/t1 /tmp/t2
#
# Colored clap help/error rendering can be vetted with:
#   CLICOLOR_FORCE=1 ./vet-output.sh | less -R

set -u
cd "$(dirname "$0")"

BIN="${1:-target/debug/sac}"
if [ $# -eq 0 ]; then
    cargo build -q || exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
DB="$WORK/vet.db"

normalize() { sed -E "s|$WORK|<WORK>|g; s/et=[0-9]+/et=<T>/g; s/lt=[0-9]+/lt=<T>/g; s/\"(event_time|logged_time)\": [0-9]+/\"\1\": <T>/g"; }

# label + command; prints output, and the exit code when nonzero
say() { local label="$1"; shift; echo; echo "== $label =="; local out; out="$("$@" 2>&1)"; SAY_OUT="$out"; local code=$?; echo "$out" | normalize; [ "$code" -ne 0 ] && echo "(exit $code)"; return 0; }

# first log position say() just printed; proposal and comment ids are birth seqs
first_seq() { printf '%s\n' "$SAY_OUT" | sed -n 's/^#\([0-9]*\).*/\1/p' | head -1; }

H=(--actor "human person" --tier human)
A=(--actor "saccade bot" --tier agent)

echo "######## help surfaces ########"
say "sac --help"            "$BIN" --help
say "sac help"              "$BIN" help
say "sac create --help"     "$BIN" create --help
say "sac done --help"       "$BIN" done --help
say "sac drop --help"       "$BIN" drop --help
say "sac release --help"    "$BIN" release --help
say "sac list --help"       "$BIN" list --help
say "sac log --help"        "$BIN" log --help
say "sac propose --help"    "$BIN" propose --help
say "sac accept --help"     "$BIN" accept --help
say "sac reject --help"     "$BIN" reject --help
say "sac withdraw --help"   "$BIN" withdraw --help
say "sac proposals --help"  "$BIN" proposals --help
say "sac comment --help"    "$BIN" comment --help
say "sac show --help"       "$BIN" show --help
say "unrecognized subcommand" "$BIN" frobnicate
say "invalid --tier value"  "$BIN" --tier robot list

echo
echo "######## usage errors (no db touched) ########"
say "no subcommand"         "$BIN"
say "anonymous mutate"      "$BIN" --db "$DB" create task x
say "id without prefix"     "$BIN" --db "$DB" --actor a --tier human claim 3
say "wrong-type prefix"     "$BIN" --db "$DB" --actor a --tier human claim h-1
say "comment without prefix" "$BIN" --db "$DB" --actor a --tier agent comment 3 body
say "proposal id not a number" "$BIN" --db "$DB" --actor a --tier human accept x
say "missing required --note" "$BIN" drop t-1

echo
echo "######## database errors ########"
say "read missing db"       "$BIN" --db "$WORK/absent.db" list
python3 - "$WORK/foreign.db" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1]); c.execute("CREATE TABLE theirs (x)"); c.commit()
EOF
say "foreign database"      "$BIN" --db "$WORK/foreign.db" list

echo
echo "######## empty database ########"
python3 - "$WORK/empty.db" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.executescript("""
CREATE TABLE events (
    seq INTEGER PRIMARY KEY, event_time INTEGER NOT NULL, logged_time INTEGER NOT NULL,
    actor TEXT NOT NULL, tier TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
PRAGMA application_id = 0x73616364;
PRAGMA user_version = 1;
""")
c.commit()
EOF
say "list on empty log"     "$BIN" --db "$WORK/empty.db" list
say "log on empty log"      "$BIN" --db "$WORK/empty.db" log

echo
echo "######## success lines ########"
say "create"                "$BIN" --db "$DB" "${H[@]}" create task "implement foo"
say "create with parent"    "$BIN" --db "$DB" "${H[@]}" create task "wire subtask" --parent t-0
say "claim"                 "$BIN" --db "$DB" "${A[@]}" claim t-0
say "release with note"     "$BIN" --db "$DB" "${H[@]}" release t-0 --note "run dead, reclaim"
say "reclaim with --at"     "$BIN" --db "$DB" "${A[@]}" --at 1000 claim t-0
say "done with receipt"     "$BIN" --db "$DB" "${A[@]}" done t-0 --receipt "22/22 green"
say "drop with note"        "$BIN" --db "$DB" "${H[@]}" drop t-0 --note "superseded"

echo
echo "######## proposal lifecycle ########"
say "create for proposal"   "$BIN" --db "$DB" "${A[@]}" create task "migrate floop"
say "propose drop (gated)"  "$BIN" --db "$DB" "${A[@]}" propose drop t-2 --name "superseded by foo"
P="$(first_seq)"
say "accept (executes act)" "$BIN" --db "$DB" "${H[@]}" accept "$P"
say "create for rulings"    "$BIN" --db "$DB" "${A[@]}" create task "wire dock"
say "claim for rulings"     "$BIN" --db "$DB" "${A[@]}" claim t-3
say "propose release"       "$BIN" --db "$DB" "${A[@]}" propose release t-3 --name "claim is stale"
P="$(first_seq)"
say "reject with note"      "$BIN" --db "$DB" "${H[@]}" reject "$P" --note "claim is live"
say "re-propose (free)"     "$BIN" --db "$DB" "${A[@]}" propose release t-3 --name "claim is stale"
P="$(first_seq)"
say "withdraw own proposal" "$BIN" --db "$DB" "${A[@]}" withdraw "$P" --note "holding for dock design"
say "re-propose (open)"     "$BIN" --db "$DB" "${A[@]}" propose release t-3 --name "stale claim, take 2"
say "proposals (queue)"     "$BIN" --db "$DB" proposals
say "proposals --json"      "$BIN" --db "$DB" --json proposals

echo
echo "######## comments and show ########"
say "create for comments"   "$BIN" --db "$DB" "${A[@]}" create task "triage inbox"
say "comment on task"       "$BIN" --db "$DB" "${A[@]}" comment t-4 "inbox swept; two corpses proposed for drop"
C="$(first_seq)"
say "reply to comment"      "$BIN" --db "$DB" "${A[@]}" comment "#$C" "correction: one was fresh work"
say "show (the dock)"       "$BIN" --db "$DB" show t-4

echo
echo "######## reads ########"
say "list"                  "$BIN" --db "$DB" list
say "list --json"           "$BIN" --db "$DB" --json list
say "log"                   "$BIN" --db "$DB" log
say "log --json"            "$BIN" --db "$DB" --json log

echo
echo "######## rejections ########"
say "invalid_task_id"       "$BIN" --db "$DB" --actor a --tier human claim t-9
say "invalid_parent_task_id" "$BIN" --db "$DB" --actor a --tier human create task x --parent t-9
say "invalid_state_transition" "$BIN" --db "$DB" --actor a --tier human claim t-0
say "invalid_proposal_id"   "$BIN" --db "$DB" "${H[@]}" accept 999
say "invalid_comment_id"    "$BIN" --db "$DB" "${A[@]}" comment "#999" orphan
say "reason_required"       "$BIN" --db "$DB" "${A[@]}" comment t-4 "   "
say "usage (show absent)"   "$BIN" --db "$DB" show t-999
say "human_only (agent drop)" "$BIN" --db "$DB" "${A[@]}" drop t-1 --note "agents propose, not rule"
say "human_only --json"     "$BIN" --db "$DB" "${A[@]}" --json drop t-1 --note "agents propose, not rule"

echo
echo "######## degraded mode ########"
NEXT="$(python3 -c 'import sqlite3,sys
print(sqlite3.connect(sys.argv[1]).execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM events").fetchone()[0])' "$DB")"
python3 - "$DB" "$NEXT" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute("INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload) VALUES (?, 100, 200, 'future binary', 'system', 'task_moved', '{\"id\":0}')", (int(sys.argv[2]),))
c.commit()
EOF
say "log under skew (warns, prints)" "$BIN" --db "$DB" log
say "list under skew (refuses)"      "$BIN" --db "$DB" list
say "mutate under skew (refuses)"    "$BIN" --db "$DB" "${H[@]}" create task x

# The append-only triggers block UPDATE/DELETE — corruption is simulated the
# only way it could really happen: a bad row appended (INSERT is legal).
say "prepare corrupt db" true
CORRUPT="$WORK/corrupt.db"
"$BIN" --db "$CORRUPT" "${H[@]}" create task pristine >/dev/null
python3 - "$CORRUPT" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute("INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload) VALUES (1, 100, 200, 'vandal', 'human', 'task_done', '{')")
c.commit()
EOF
say "corrupt payload (loud)"  "$BIN" --db "$CORRUPT" log

SCHEMA="$WORK/schema.db"
"$BIN" --db "$SCHEMA" "${H[@]}" create task pristine >/dev/null
python3 - "$SCHEMA" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute("PRAGMA user_version = 99")
c.commit()
EOF
say "newer schema (refuses)"  "$BIN" --db "$SCHEMA" list
