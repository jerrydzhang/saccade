# Migrating a 0.12.0 log to 0.13.0

Reader: the operator at cutover. 0.13.0 removes task parent/child
relationships; schema 2 logs (written by 0.12.0) carry a `parent_id`
member in `task_created` payloads that this binary no longer reads.
The schema stamp (`user_version`) moved 2 → 3, so the binaries guard
each other: 0.12.0 refuses a migrated log ("newer schema"), 0.13.0
refuses an unmigrated one ("predates this binary") — nothing folds
with discarded fields.

The migration strips the `parent_id` member from every task-created
payload and changes nothing else: seqs, times, actors, tiers, scopes,
and all other payloads ride byte-identical, so task, comment, and
incarnation ids never move. Historical parent links are dropped as
grouping facts; they translate to nothing.

Work on backup copies — the live file stays untouched throughout:

```
# stop the tracker's server, then take a consistent snapshot
# (.backup is SQLite's online backup API, safe on a WAL database)
sqlite3 /path/to/saccade.db ".backup '/path/to/backup.db'"

# migrate the backup into a fresh output file
sac migrate --in /path/to/backup.db --out /path/to/migrated.db

# verify with the new binary, then cut over
sac --db /path/to/migrated.db list
mv /path/to/saccade.db /path/to/saccade.db.0.12  # keep the original
mv /path/to/migrated.db /path/to/saccade.db
```

`sac migrate` prints what it did, naming every stripped link
(`c-<seq> birth of t-<n>, parented t-<m>`); `--json` emits the same
report as data. It refuses an existing output path, a source that is
not a schema-2 saccade log, and any rewrite whose folded world differs
from the source's — a refused migration writes nothing. Keep the
original backup; it is the rollback (restore it and run the 0.12.0
binary).
