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
3. **Disagreement is a finding.** When the record, the skill, and the
   code disagree — with each other or with what you observe — stop and
   name the conflict on the thread and route it to your asker — the
   asker carries it to the skill's author. Never resolve it silently.
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
2. **Size the work.** After retrieval — or after the reasoning an
   ambiguous spec demanded, whichever road produced your understanding —
   answer the sizing question explicitly: one acceptable sitting, or
   several tasks. Labels like CHANGE ONE, CHANGE TWO are task boundaries
   wearing phase names: a change that could land and be accepted on its
   own is a task. Several parks the split on the thread (`sac create
   task --parent`) with its shape named — a chain fires one child at a
   time, each demand when its upstream lands; a fan-out fires together
   when the changes share nothing. The question is never skipped, only
   answered: a spec you have already called large is the signal, and
   naming its size while executing it whole is the failure this step
   exists to stop. Crash resume is never a reason to keep a too-large
   shape.
3. **Work in the task's worktree** — its branch `saccade/t-N`, prepared
   from the recorded checkpoint. The runner owns it (see Runner-owned
   state); park what you learn as you learn it.
4. **Ask when blocked.** Your session carries an `ask` tool: a blocking
   question to the human. The question lands on the task's thread; the
   tool returns when the human answers; the answer is the tool result.
   Observations that don't block the work are notes on the thread, not
   asks.
5. **Deviate openly.** If the spec cannot be followed as written, say so
   on the thread with evidence — a narrowed scope silently executed is a
   failure; a deviation announced is a finding.
6. **Deliver**: commit your work to the task branch, `sac checkpoint
   t-N`, claim, then `sac done t-N --receipt "…"`. The claim is the door
   key `done` requires — it is not an in-flight signal (the demand
   carries in-flight); take it at delivery, never before. Delivery is not
   closure. The receipt deposits this round's honest account — outcome,
   evidence, omissions, deviations — and the asker either accepts it as
   done or answers with findings; iteration continues: findings on a
   delivered task fire in-thread rounds, and a fresh demand reopens a
   done task. Park dead ends and deviations on the thread before you
   deliver. Failure honestly reported is success.

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
3. **Fire the demand**: `sac comment t-N "spec…" --demand`.
4. **Wait by outcome**: `sac wait c-<seq>` — the seq is your demand's own,
   printed in its reply. It releases when the run asks something of you —
   it settles (the receipt is named), cancels, rejects its prompt before
   accepting (the cause is carried), refuses (the reason is carried), is
   answered with no run behind it, or asks a blocking question (the
   release carries it; answer on the thread, then re-arm the wait). It
   never releases on replies: acks hold, and a run still being born
   holds the wait too.
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

One idea per comment, at every door. A multi-idea answer forks into
one comment per idea; the formal reply or receipt indexes them, never
contains them. This is practice, not machinery: the reply door stays
single, and the day a reviewed blob survives this law anyway is the day
the door itself changes.

## Searching

`sac search` finds records by exact terms over the folded text fields —
task titles, comment bodies, receipts. No ranking, no relevance, no
fuzzy: a term matches as a whole, word-bounded, case-folded, or it
doesn't match — inflections do not match, so try each form. Results are
thread-grouped pointers with one matched line each, in record order.

- **Find where something was decided**: `sac search telemetry`. The
  groups are the threads that hold matches.
- **Follow a reference**: an id term is a reference search — `'#907'`
  finds every record citing that comment, `t-49` everything naming that
  task. Quote the hash.
- **Narrow**: `in:` `by:` `kind:` `under:` — only-show-me filters;
  every narrowing prints visible counts, and a query that matches
  nothing prints one honest line. `sac search --help` owns the grammar.
- **Read the neighborhood**: `sac search '#907' -C 3` anchors on that
  record and shows its window.
- **Read whole records**: raw ids show the event, `t-N` shows the task
  plus thread; several ids in one call, or one per line through
  `--stdin`. The pipe reads end to end:
  `sac search telemetry --json | jq -r '.groups[].records[].pointer' | sac show --stdin`.

Pointers stay pointers: a receipt's address is its task (`t-90
receipt` — the fold keeps no delivery seq), and a receipt carries no
author in the fold, so `by:` narrows receipts out; the counts say so.
The power tail is unchanged: `sac log | grep` reads the raw record.

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
| `sac comment t-N "…"` / `sac comment #<seq> "…"` | any tier | park or reply — a note; `--demand` fires a run |
| `sac steer t-N "…"` | any tier | reach the task's live run at its next turn boundary; with no run it stands as intent on the thread |
| `sac wait c-<seq>` | any tier | blocks on the demand's outcome, never replies |
| `sac cancel t-N` | any tier | the kill request for a task's active run |
| `sac checkpoint t-N` | any tier | records the branch tip as the task's checkpoint |
| `sac list` / `sac show <ids…>` / `sac log` / `sac proposals` | anonymous | the board, threads and records, raw events, the ruling queue |
| `sac search <terms…> [in:t-N by:NAME kind:K under:t-N] [-C N]` | anonymous | exact terms over titles, comment bodies, receipts; id terms are reference searches; facets only-show-me with visible counts; `-C N` anchors on one record |
| `sac skill install` / `sac skill check` | any tier | deploy this binary's copy of this skill into ./.agents/skills/saccade, or verify the copy against it; a differing copy refuses until deleted |

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
manage by hand. Task workspaces are best-effort and ephemeral: the
record governs, the workspace is a cache. The lifecycle:

- **Prepare rebuilds what is missing.** A workspace expected but missing is
  re-cut at the recorded checkpoint — or the task's base when no
  checkpoint exists — so absence never refuses. Recreation covers
  absence only, never divergence: a branch that disagrees with the
  recorded checkpoint still refuses, and a checkpoint nothing retains is
  lost work; prepare refuses, naming it. That state is human-made —
  deletion stays manual, there is no collector, no sweep.
- **The record is the demand door.** While a task lives — open, claimed,
  delivered, or done (done is reopenable) — its recorded workspace names
  the branch and checkpoint the next demand rebuilds from. Merging a
  task branch into your mainline is normal review flow; the branch
  itself stays.
- **Deletion is manual hygiene**, under one rule: never delete the sole
  holder of recorded work. The record names the checkpoint, so

  sole-holder is checkable before deleting — a branch merged into the
  mainline is not sole holder, an unmerged branch usually is, and a
  worktree's uncommitted files are never recorded work.
- **A dropped task has no door.** Its worktree and branch are residue —
  collectable, but still not by your hand: cleanup is a proposal to the
  human (the tracker has no collection verb yet), naming the workspaces
  and their tasks.

## Errors and degraded mode

Failures exit 1 with a named variant on stderr (`--json` emits
`{"error": "<variant>", "detail": "…"}`). If resolving an error requires
violating any law of this skill, that is a finding — name it on the
thread before proceeding.

The variant table and degraded mode live in `errors.md` beside this
skill — read it when a command names a variant. When the tracker
misbehaves or a refusal needs reproducing, `diagnosing.md` reconstructs
the conditions from the attempts log.

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
