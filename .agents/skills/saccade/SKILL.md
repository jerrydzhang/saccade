---
name: saccade
description: Use when running the `sac` CLI — working tasks, claiming, delivering with receipts, waiting on runs, firing demands, or any question about the tracker's events, task states, tiers, or proposals.
  - ".agents/**"
---

# Saccade (`sac`)

One binary over an append-only event log; the world is a fold of that log.
The record is the reasoning cache: conclusions live on threads so the next
reader — often you, after a context loss — never re-derives them. Judgment
acts (drop, release, ruling on proposals) are human-only; agents propose.

## Identity

Tier is **possessed, never passed**. `SACCADE_ACTOR` set means agent tier;
unset means human. There is no tier argument, and actor names never carry
authority. Mutating commands record your identity from possession; reads
need none. A run's derived name (`pi/t-75-2`) is an incarnation, not an
actor — it never equals a birth attribution, so a run can never self-accept.

## The record is the reasoning cache

These are behaviors, not suggestions — every one exists because an agent
that skips it burns its reasoning on re-derivation or acts on guesses.

1. **Reason lightly, retrieve, then reason deeply — in that order.**
   Before any deep reasoning, frame what you would need to know, then read
   the context around it: the thread, `sac log`, the code the record names.
   Deep reasoning comes last and only on what retrieval didn't answer.
   Deep-first reasoning is the failure mode, not a style.
2. **Park conclusions at altitude** as they form (`sac comment t-N "…"`) —
   parking is durability: an unparked conclusion dies with your context, a
   parked one survives reboot, handoff, and the next incarnation. A cache
   entry is semantic content that survives rewording: the verdict, the why,
   the dead end, the deviation. Narration of the next ten lines is not a
   conclusion; a write that cannot name its next reader doesn't earn bytes.
3. **Disagreement is a finding.** When the record and the code disagree,
   stop and name the conflict on the thread — never resolve it silently.
   Hidden staleness poisons the cache for every later reader.

## Workflow

Two ways work executes, decided by how you arrived — and one role that
belongs to whoever delegates:
- **A demand summoned you** — your prompt names a task thread and you work
  as a spawned run → *Executing a demand*. Nobody else receives that
  prompt; if it addressed you, you are the run.
- **Nobody summoned you** — you read the board and chose the work →
  *Manual pickup*.
- **Supervising is a role, not a third path**: whoever fires a demand
  supervises it — a human, an external agent, or a run whose scope
  outgrew its context and splits.

### Executing a demand (a run)

Your deliverable is the **honest presentation** — completing the ask is an
outcome, not the goal.

1. **Retrieve first** (the protocol above): read your thread whole — the
   spec, prior verdicts, dead ends, parked working state — then the code
   and record it names. Followed honestly, this is resuming the work, not
   starting it.
2. **Work in the task's worktree** — its branch `saccade/t-N`, prepared
   from the recorded checkpoint. The runner owns it (see Runner-owned
   state); park what you learn as you learn it.
3. **Deviate openly.** If the spec cannot be followed as written, say so
   on the thread with evidence — a narrowed scope silently executed is a
   failure; a deviation announced is a finding.
4. **Deliver**: `sac done t-N --receipt "…"`. Delivery is not closure.
   The receipt deposits this round's honest account — outcome, evidence,
   omissions — and the asker either accepts it as done or answers with
   findings; iteration continues: findings on a delivered task fire
   in-thread rounds, and a fresh demand reopens a done task. Failure
   honestly reported is success.

Do not claim a task you are executing a demand on — the demand carries
in-flight. Claiming is for manual pickup.

### Supervising (the asker)

**When to dispatch.** The unit is the acceptor's sitting: a task is the
largest unit one accept can honestly review in one sitting — not what the
builder believes it can deliver. If the deliverable statement contains an
"and," it is two tasks; if no single receipt would be checkable in one
sitting, it is many. Dispatch when you hold the spec but not the
mechanics; do the work yourself when you hold both, or when it is smaller
than the dispatch ritual. When you cannot yet write the spec, dispatch a
research round — bounded, read-only, its deliverable the spec material
parked as conclusions. You will be wrong about scope; the design does not
ask you to be right. Work in rounds small enough that a wrong cut costs
one round, read receipts for strain — context pressure, under-delivery,
deviations outnumbering confirmations are the record saying the task was
really two — and let the cut be a proposal the level above sees,
superseding cleanly on re-cut.

1. **Create with shape.** `sac create task "name"` — the reply names what
   was born (`t-N`, both faces). Specs ride a comment on the thread.
2. **Spec at altitude.** Ratified decisions, constraints, and the
   verification door — not step-by-step recipes. Every claim in a spec is
   sourced — read this session from the record or the tree — or
   explicitly marked open for the executor to resolve; a belief costumed
   as a source is the failure carrier.
3. **Fire the demand**: `sac comment t-N "spec…" --to agent`.
4. **Wait by outcome**: `sac wait c-<seq>` — the seq is your demand's own,
   printed in its reply. It releases when the run asks something of you —
   it settles (the receipt is named), cancels, refuses (the reason is
   carried), is answered with no run behind it, or raises a prompt
   awaiting an answer. It never releases on replies: acks hold.
5. **Verify the claims** — that is what makes them knowledge. Clean
   checkout, suite counts, a live smoke on a scratch db. Cheap verification
   is why receipts name evidence.
6. **Accept or route.** `sac accept t-N` is yours if you are the birth
   attribution; otherwise it goes to a human. Merging follows your
   project's own release flow — the tracker holds the thread, not the
   train.

### Manual pickup

For work you take under your own identity, in your own session — an agent
executing directly rather than by demand, a human holding a task while
thinking it through, a supervisor doing the small task it fully holds.
This is the only workflow where claiming belongs.

1. `sac list` at session start — that's the board.
2. Claim what you will actually finish: `sac claim t-N`. Claims are
   reservations, not progress reports; do not stockpile them, and never
   sit silently on one.
3. Do the work; park what you learn mid-flight as comments — never a side
   file. If your observation changes the *what*, that is supersession:
   propose the drop, naming the successor.
4. `sac done t-N --receipt "…"` when the work lands.

### Receipts — the protocol both roles share

The receipt is written by the executor at deliver and read by the
acceptor at verify — it is the shared artifact of claim-and-verify, not
an addendum. It is the deposit the delivered state guards; the accept
reviews it, and whoever reads the thread later inherits it. Name the
outcome and the evidence: what changed, which tests ran and their counts,
which commands were exercised, which files were touched. "done", "fixed",
"implemented" are not receipts.

## Commands

| Command | Who | Notes |
|---|---|---|
| `sac create task "name" [--parent t-N]` | any tier | the reply names what was born |
| `sac claim t-N` | any tier | a reservation, not a progress report |
| `sac done t-N --receipt "…"` | any tier | delivers; the receipt is the deposit the accept reviews |
| `sac drop t-N [--note]` | **human only** | judgment act; agents propose |
| `sac release t-N [--note]` | **human only** | judgment act; agents propose |
| `sac propose <drop\|release> t-N --name "…"` | any tier | the gate queue: your evidence, the human's call |
| `sac accept t-N` | birth attribution, or human | the only door from delivered to done |
| `sac accept <seq>` / `sac reject <seq> --note` | **human only** | ruling acts on proposals |
| `sac comment t-N "…"` / `sac comment #<seq> "…"` | any tier | park or reply; `--to <actor>` addresses |
| `sac wait c-<seq>` | any tier | blocks on the demand's outcome, never replies |
| `sac cancel t-N` | any tier | the kill request for a task's active run |
| `sac checkpoint t-N` | any tier | records the branch tip as the task's checkpoint |
| `sac list` / `sac show t-N` / `sac log` / `sac proposals` | anonymous | the board, a thread, raw events, the ruling queue |

## Hygiene

- Any create not meant for the record goes to a scratch `--db`: the
  in-repo default is the project's live tracker.
- Comments park on threads. Side files are where conclusions go to die.

## Event times

`--at <epoch-seconds>` backdates an event. Use it only with a **sourced**
measurement (a timestamp from a log, an import, a CI run). Never invent or
eyeball a time; omit `--at` and the event stamps honestly as "now".

## Runner-owned state

Task worktrees (`<state dir>/worktrees/t-N`), task branches (`saccade/t-N`),
and session files belong to the runner — system property, not yours to
manage by hand. The lifecycle:

- **A branch is a demand door.** While a task lives — open, claimed,
  delivered, or done (done is reopenable) — its recorded workspace points
  at its branch and checkpoint; severing the branch makes the next demand
  refuse at prepare. Merging a task branch into your mainline is normal
  review flow; the branch itself stays.
- **A dropped task has no door.** Its worktree and branch are residue —
  collectable, but still not by your hand: cleanup is a proposal to the
  human (the tracker has no collection verb yet), naming the workspaces
  and their tasks.

## Errors and degraded mode

Failures exit 1 with a named variant on stderr (`--json` emits
`{"error": "<variant>", "detail": "…"}`):

- `human_only` — you attempted a judgment act. Stop; escalate.
- `invalid_task_id` / `invalid_parent_task_id` — no such task; ids are exact
  `t-N` tokens, not searches.
- `invalid_proposal_id` — no proposal was born at that seq; proposal ids are
  the bare log position of the `proposal_created` event.
- `invalid_state_transition` — read the log; the task's state says otherwise
  (already claimed, already done, not delivered so there is nothing to
  accept).
- `not_birth_attribution` — you accepted a delivered task not born under
  your attribution; the asker or a human accepts.
- `degraded` — a newer binary wrote events this one can't understand.
  Writes and `list` refuse; `sac log` still works. Report it; upgrading
  fixes it.
- `database_error` — includes missing file (reads) and corruption (loud,
  named).

## Semantics worth knowing

- Task states: `open → claimed → delivered → done`. `sac done` delivers,
  never deems done; `accept` is the only door from delivered to done; a
  demand reopens done (never delivered — findings on delivered fire
  in-thread rounds instead). `dropped` is the sole terminal state and
  human-only, reachable from `open` or `done` — a running task must be
  cancelled first.
- "Claimed" means reserved by someone — possibly a human claiming to
  prevent agents from taking it while they think. It does not mean work is
  happening; in-flight lives on the demand.
- Proposals are seq-addressed (`sac accept 614`): the id is the birth
  event's log position. They are inert while open (never block their
  target), show `stale` in the queue when the embedded act goes illegal,
  and re-proposing after a rejection is free — the rejection note names
  the missing evidence.
- Each repo has one tracker, served locally (`--server`, or
  `SACCADE_SERVER`; default `127.0.0.1:8811`); its state root derives
  from the repo path, never from your working directory — `--repo` or
  `SACCADE_REPO` names the project when working from elsewhere.
