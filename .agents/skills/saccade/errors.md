# Errors and degraded mode

Read this when a command fails. Failures exit 1 with a named variant on
stderr (`--json` emits `{"error": "<variant>", "detail": "…"}`). If
resolving an error requires violating any law of the skill, that is a
finding — name it on the thread before proceeding.

- `human_only` — you attempted a judgment act. Stop; escalate.
- `identity_fallback` — a non-tty write with no `SACCADE_ACTOR` and no
  `--actor` would record the account name; export
  `SACCADE_ACTOR=<name>` or pass `--actor`.
- `invalid_task_id` / `invalid_parent_task_id` — no such task; ids are exact
  `t-N` tokens, not searches.
- `invalid_proposal_id` — no proposal was born at that seq; proposal ids are
  the bare log position of the `proposal_created` event.
- `invalid_state_transition` — read the log; the task's state says otherwise
  (already claimed, already done, not delivered so there is nothing to
  accept).
- `steer_not_standing` — the forward names a steer that is not standing
  intent (wrong kind, or already consumed by its run).
- `no_active_incarnation` — the forward names a task whose run slot is
  empty: nothing consumes the steer.
- `not_birth_attribution` — you accepted a delivered task not born under
  your attribution; the asker or a human accepts.
- `degraded` — a newer binary wrote events this one can't understand.
  Writes and `list` refuse; `sac log` still works. Report it; upgrading
  fixes it.
- `database_error` — includes missing file (reads) and corruption (loud,
  named).

When the tracker misbehaves or a refusal needs reproducing, read
`diagnosing.md` beside this file: the attempts log reconstructs the
conditions.
