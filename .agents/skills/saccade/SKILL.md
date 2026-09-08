---
name: saccade
description: Use when running the `sac` CLI in this repo — claiming or completing tracker tasks, reading `sac list` / `sac log`, writing receipts, or any question about Saccade's event log, task states, or tiers. Carries the agent tier discipline (agents never pass --tier human; drop and release are human-only).
globs:
  - "saccade.db*"
  - ".agents/**"
---

# Saccade (`sac`) — how agents use this repo's tracker

Saccade is this repo's event-sourced issue tracker. One binary, `sac`
(`target/debug/saccade` or `cargo run --`). Every mutation appends an event
recording who acted, at which tier, and when — the log is shared memory; treat
every line you append as something a human will read in a retro.

## The rule that cannot break

**You are an agent. Every mutating command you run carries `--tier agent`.**

- `drop` and `release` are human-only judgment acts. If you run one you get
  `rejected: human_only` and exit 1 — that is the system working as designed.
  Do not retry, do not hunt for a flag combination that works; none exists.
- If a task deserves dropping or a claim deserves releasing: say so to the
  human, or create a task for it (`sac create task "human: drop t-12 because …"`).
  Escalate the judgment; never execute it.
- Never set or export `SACCADDE_TIER`, never alias or wrap `sac` to change
  your tier. The devshell deliberately leaves it unset.
- Always pass `--actor` explicitly with your own name, even when
  `SACCADDE_ACTOR` is set in the environment — attribution is provenance, and
  retro queries count on it being true.

## Commands

| Command | Who | Notes |
|---|---|---|
| `sac create task "name" [--parent t-N]` | any tier | noun-as-argument; only `task` exists today |
| `sac claim t-N` | any tier | a claim is a **reservation**, not a progress report |
| `sac done t-N --receipt "…"` | any tier | receipt required — see receipts below |
| `sac drop t-N [--note "…"]` | **human only** | judgment act; agents escalate instead |
| `sac release t-N [--note "…"]` | **human only** | judgment act; agents escalate instead |
| `sac list` / `sac list --json` | anonymous | reads need no identity |
| `sac log` / `sac log --json` | anonymous | raw events; always available |

Every mutating command needs `--actor <name> --tier agent`. Reads don't.
Add `--json` for machine-readable output, including errors.

## The working loop

1. `sac list` at session start — that's the board.
2. Claim what you will actually finish this session: `sac claim t-N …`.
   Claims are reservations; do not stockpile them.
3. Do the work.
4. `sac done t-N --receipt "…"` when it lands.
5. Blocked, or the task is wrong? Surface it to the human — they release or
   drop. Never sit silently on a claim.

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
- The log is append-only and ordered by seq; timestamps are claims, not order.
- One database per repo (`saccade.db`, gitignored; the devshell exports
  `SACCADDE_DB`).
