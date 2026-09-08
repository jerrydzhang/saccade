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

BIN="${1:-target/debug/saccade}"
if [ $# -eq 0 ]; then
    cargo build -q || exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
DB="$WORK/vet.db"

normalize() { sed -E "s|$WORK|<WORK>|g; s/et=[0-9]+/et=<T>/g; s/lt=[0-9]+/lt=<T>/g; s/\"(event_time|logged_time)\": [0-9]+/\"\1\": <T>/g"; }

# label + command; prints output, and the exit code when nonzero
say() { local label="$1"; shift; echo; echo "== $label =="; local out; out="$("$@" 2>&1)"; local code=$?; echo "$out" | normalize; [ "$code" -ne 0 ] && echo "(exit $code)"; return 0; }

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
say "unrecognized subcommand" "$BIN" frobnicate
say "invalid --tier value"  "$BIN" --tier robot list

echo
echo "######## usage errors (no db touched) ########"
say "no subcommand"         "$BIN"
say "anonymous mutate"      "$BIN" --db "$DB" create task x
say "id without prefix"     "$BIN" --db "$DB" --actor a --tier human claim 3
say "wrong-type prefix"     "$BIN" --db "$DB" --actor a --tier human claim h-1

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
say "create for note-less"  "$BIN" --db "$DB" "${H[@]}" create task "bare"
say "drop without note"     "$BIN" --db "$DB" "${H[@]}" drop t-2

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
say "human_only (agent drop)" "$BIN" --db "$DB" "${A[@]}" drop t-1
say "human_only --json"     "$BIN" --db "$DB" "${A[@]}" --json drop t-1

echo
echo "######## degraded mode ########"
python3 - "$DB" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute("INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload) VALUES (9, 100, 200, 'future binary', 'system', 'task_moved', '{\"id\":0}')")
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
