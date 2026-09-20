---
name: saccade
description: Use when running the `sac` CLI in this repo — claiming or completing tracker tasks, reading `sac list` / `sac log`, writing receipts, or any question about Saccade's event log, task states, or tiers. Carries the identity discipline (tier is possessed via SACCADE_ACTOR, never passed; drop and release are human-only).
globs:
  - "saccade.db*"
  - ".agents/**"
---

# Saccade (`sac`) — how agents use this repo's tracker

Saccade is this repo's event-sourced issue tracker. One binary, `sac`
(`target/debug/sac` in a dev checkout, `sac` from PATH when installed). Every
mutation appends an event recording who acted, at which tier, and when — the
log is shared memory; treat every line you append as something a human will
read in a retro.

## The rule that cannot break

**Your tier is possessed, not passed: the CLI has no tier argument.**

- Your harness sets `SACCADE_ACTOR`; its presence *is* agent tier. Never set,
  unset, or override it — identity comes from the session, not from you.
- `--actor` names at either tier and never re-tiers. Your name is what
  `SACCADE_ACTOR` says it is; never override it — attribution is provenance,
  and retro queries count on it being true.
- `drop`, `release`, `accept`, and `reject` are human-only judgment acts. If you run one you get
  `rejected: human_only` and exit 1 — that is the system working as designed.
  Do not retry, do not hunt for a flag combination that works; none exists.
- Judgment-shaped facts (a duplicate, a supersession, a corpse) have a
  first-class home: `sac propose drop t-12 --name "the evidence"`. The
  proposal carries your evidence into the world; the human rules per act.
  Never execute the judgment — propose it.

## Commands

| Command | Who | Notes |
|---|---|---|
| `sac create task "name" [--parent t-N]` | any tier | noun-as-argument; only `task` exists today |
| `sac claim t-N` | any tier | a claim is a **reservation**, not a progress report |
| `sac done t-N --receipt "…"` | any tier | receipt required — see receipts below |
| `sac drop t-N [--note "…"]` | **human only** | judgment act; agents propose instead |
| `sac release t-N [--note "…"]` | **human only** | judgment act; agents propose instead |
| `sac propose <drop\|release> t-N --name "…"` | any tier | the gate queue: your evidence, the human's call |
| `sac accept <seq>` / `sac reject <seq> --note "…"` | **human only** | ruling acts on proposals |
| `sac withdraw <seq> --note "…"` | any tier | take your own proposal off the queue |
| `sac comment t-N "…"` / `sac comment #<seq> "…"` | any tier | how-context on a task; `#<seq>` replies to a comment |
| `sac cancel t-N` | any tier | stops the task's active run; cancel only what you started |
| `sac wait c-N [--timeout S]` | anonymous | block until a demand's reply lands |
| `sac show t-N` | anonymous | the dock: state, receipt, comment thread with voices |
| `sac proposals` | anonymous | the ruling queue; `stale` marks acts gone illegal |
| `sac list` / `sac list --json` | anonymous | reads need no identity |
| `sac log` / `sac log --json` | anonymous | raw events; always available |

Mutating commands record your identity automatically (see the rule above);
reads need none. Add `--json` for machine-readable output, including errors.

## The working loop

1. `sac list` at session start — that's the board.
2. Claim what you will actually finish this session: `sac claim t-N …`.
   Claims are reservations; do not stockpile them.
3. Do the work. Park what you learn mid-flight (deferred wrangles, review
   findings, working state) as `sac comment t-N "…"` — never a side file.
   Titles say **what**; comments say **how**. If your observation changes the
   what, that is supersession: propose the drop naming the successor.
4. `sac done t-N --receipt "…"` when it lands.
5. Task wrong? Propose the judgment (`sac propose drop t-N --name "…"`)
   and say so to the human. Never sit silently on a claim.

### Receipts

The receipt is the deposit the done-state guards; the retro reads it.
Name the outcome and the evidence: what changed, which tests ran and their
counts, which files. "done", "fixed", "implemented" are not receipts.

### Event times

`--at <epoch-seconds>` backdates an event. Use it only with a **sourced**
measurement (a timestamp from a log, an import, a CI run). Never invent or
eyeball a time; omit `--at` and the event stamps honestly as "now".

## Errors and degraded mode

Failures exit 1 with a named variant on stderr (`--json` emits
`{"error": "<variant>", "detail": "…"}`):

- `human_only` — you attempted a judgment act. Stop; escalate.
- `invalid_task_id` / `invalid_parent_task_id` — no such task; ids are exact
  `t-N` tokens, not searches.
- `invalid_proposal_id` — no proposal was born at that seq; proposal ids are
  the bare log position of the `proposal_created` event.
- `invalid_state_transition` — read the log; the task's state says otherwise
  (e.g. it is already claimed by someone else, or already done).
- `degraded` — a newer binary wrote events this one can't understand. Writes
  and `list` refuse; `sac log` still works. Report it; upgrading fixes it.
- `database_error` — includes missing file (reads) and corruption (loud, named).

## Semantics worth knowing

- Task states: `open → claimed → done`; `dropped` is the sole terminal state
  and human-only. A task can be dropped from `open` or from `done` (a void).
- "Claimed" means reserved by someone — possibly a human claiming to prevent
  agents from taking it while they think. It does not mean work is happening.
- Proposals are seq-addressed (`sac accept 614`): the id is the birth event's
  log position. They are inert while open (never block their target), show
  `stale` in the queue when the embedded act goes illegal, and re-propose
  after a rejection is free — the rejection note names the missing evidence.
- The log is append-only and ordered by seq; timestamps are claims, not order.
- One database per repo (`saccade.db`, gitignored), resolved from the repo
  containing the working directory; `--repo` or `SACCADE_REPO` names it when
  working from elsewhere.
