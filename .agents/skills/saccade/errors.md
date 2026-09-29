# Errors and degraded mode

Read this when a command fails. Failures exit 1 with a named variant on
stderr (`--json` emits `{"error": "<variant>", "detail": "…"}`). If
resolving an error requires violating any law of this skill, that is a
finding — name it on the thread before proceeding.

- `human_only` — you attempted a judgment act. Stop; escalate.
- `identity_fallback` — a non-tty write with no `SACCADE_ACTOR` refuses
  before any write lands (`--actor` rides the keyboard arm); export
  `SACCADE_ACTOR=<name>`.
- `invalid_task_id` / `invalid_parent_task_id` — no such task; ids are exact
  `t-N` tokens, not searches.
- `invalid_proposal_id` — no proposal was born at that seq; proposal ids are
  the bare log position of the `proposal_created` event.
- `proposal_already_open` — that task already holds an open judgment
  proposal; rule the open one first.
- `invalid_comment_id` — the `c-N` target names no comment; a task birth is
  addressed `t-N`, a proposal birth by its bare seq.
- `invalid_state_transition` — read the log; the task's state says otherwise
  (already claimed, already done, not delivered so there is nothing to
  accept).
- `not_claim_holder` — `done` and `release` need the task's current claim
  holder.
- `invalid_incarnation_id` — the incarnation names no run; the record's
  `incarnation_bound` events hold the ids.
- `incarnation_already_active` — the task's run slot is taken; a demand on
  a busy task queues and fires when the slot frees.
- `demand_not_on_task` — the demand lives on another task's thread; a run
  answers its own thread's demand.
- `workspace_already_exists` — the task already holds a workspace; prepare
  rebuilds only what is missing.
- `workspace_missing` — the task has no recorded workspace; one is cut when
  a demand first fires.
- `worktree_already_present` — the worktree the fold would create already
  exists on disk; move it away and prepare again.
- `checkpoint_rewind` — the branch tip sits behind the recorded checkpoint;
  a checkpoint advances, never rewinds.
- `steer_not_standing` — the forward names a steer that is not standing
  intent (wrong kind, or already consumed by its run).
- `no_active_incarnation` — the forward names a task whose run slot is
  empty: nothing consumes the steer.
- `not_birth_attribution` — you accepted a delivered task not born under
  your attribution; the asker or a human accepts.
- `invalid_actor` — the actor name is empty or carries edge whitespace.
- `reason_required` — the prose is empty or whitespace; names, notes,
  receipts, and bodies need words.
- `degraded` — a newer binary wrote events this one can't understand.
  Writes and `list` refuse; `sac log` still works. Report it; upgrading
  fixes it.
- `database_error` — includes missing file (reads) and corruption (loud,
  named).

When the tracker misbehaves or a refusal needs reproducing, read
`diagnosing.md` beside this file: the attempts log reconstructs the
conditions.
