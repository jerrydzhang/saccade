# Errors and degraded mode

Read this when a command fails. Failures exit 1 with a named variant on
stderr (`--json` emits `{"error": "<variant>", "detail": "…"}`). If
resolving an error requires violating any law of this skill, that is a
finding — name it on the thread before proceeding.

- `human_only` — you attempted a judgment act (cancel, reopen). Stop;
  escalate.
- `identity_fallback` — a non-tty write with no `SACCADE_ACTOR` refuses
  before any write lands (`--actor` rides the keyboard arm); export
  `SACCADE_ACTOR=<name>`.
- `invalid_task_id` / `invalid_parent_task_id` — no such task; ids are exact
  `t-N` tokens, not searches.
- `invalid_comment_id` — the `c-N` target names no comment; a task birth
  is addressed `t-N`.
- `not_an_offer` — the `c-N` you accepted, rejected, or used names a
  comment that is not an offer.
- `not_current_offer` — that offer is not the task's standing one (it
  was superseded or consumed). To replace an account, reject it while
  it stands, then offer anew.
- `self_acceptance` — your declared task scope covers the offered task:
  a task-bound producer cannot accept its own conclusion. Acceptance
  is an outside consumer's act.
- `invalid_state_transition` — read the log; the task's state says
  otherwise (a second offer while one stands, a decision on a concluded
  or open task, revising or withdrawing a bodyless or withdrawn
  comment, cancel or reopen from a state that does not admit it).
- `invalid_incarnation_id` — a claim names an incarnation the record
  never birthed, or one bound to another task.
- `workspace_already_exists` — the task already holds a workspace
  lineage; one per task.
- `workspace_missing` — the task has no recorded workspace. In this
  build nothing creates one; the native-workspace slice will.
- `worktree_already_present` — a second physical creation while the
  worktree is present.
- `checkpoint_rewind` — the branch tip sits behind the recorded
  checkpoint; a checkpoint advances, never rewinds.
- `invalid_actor` — the actor name is empty or carries edge whitespace.
- `reason_required` — the prose is empty or whitespace; names, bodies,
  grounds, reasons, and explanations need words.
- `degraded` — a newer binary wrote events this one can't understand.
  Writes and `list` refuse; `sac log` still works. Report it; upgrading
  fixes it.
- `database_error` — includes missing file (reads) and corruption (loud,
  named). A pre-0.10 log (old model) is refused loudly, not folded
  into nonsense — this build starts a fresh record.

When the tracker misbehaves or a refusal needs reproducing, read
`diagnosing.md` beside this file: the attempts log reconstructs the
conditions.
