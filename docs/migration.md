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

Stop the service first. A clean stop checkpoints the WAL away; the
rest of the sequence relies on no sidecars lingering beside the main
file — SQLite would apply an old `saccade.db-wal` to whatever main
file carries that name, so a stale sidecar must never meet the
migrated file. Sidecars always retire together with the main file
they belong to.

```
systemctl --user stop saccade
cd <state dir holding saccade.db>
ls saccade.db-wal saccade.db-shm 2>/dev/null    # a clean stop lists nothing

# retire the original whole — main file plus any sidecars — together
mkdir ../saccade-0.12
mv saccade.db saccade.db-wal saccade.db-shm ../saccade-0.12/ 2>/dev/null

# snapshot the retired log (.backup is SQLite's online backup API)
sqlite3 ../saccade-0.12/saccade.db ".backup 'backup.db'"

# migrate into a fresh file and verify it with the new binary
sac migrate --in backup.db --out migrated.db
sac --db migrated.db list

# swap in; migrate's connection closed cleanly, so it carries no sidecars
mv migrated.db saccade.db
```

`sac migrate` names every stripped link in its report
(`c-<seq> birth of t-<n>, parented t-<m>`); it refuses an existing
output path, a source that is not a schema-2 saccade log, a
task-created payload in a serialization the log writer never produced,
and any rewrite whose folded world differs from the source's — a
refused migration writes nothing.

Rollback is the retired directory: stop the service, remove the new
`saccade.db`, move the retired set back, run the 0.12.0 binary. Keep
the retired directory and the backup.
