# Saccade — Design

An opinionated, agent-native issue tracker that is the shared propositional memory between
a human, their notes, and their agents. Homelab-hosted. Rust. Single artifact.

Status: design complete, pre-implementation. This document consolidates every settled
decision from the design phase and is the seed for build step 1.

---

## 1. What Saccade is

**Saccade is a tracker where work, claims, and evidence share one graph, under one
invariant: nothing is here unless it demands something.**

It is not an "issue tracker that stores memories" and not a "tracker + memory system."
It is the middle layer of a three-layer epistemic stack:

```
┌─────────────────────────────────────────────┐
│  Zettelkasten (yours, agent-free)           │
│  fleeting notes → permanent notes           │
│  theory · load-bearing understanding        │
│  permanence by design · authority = you     │
└──────────┬──────────────────────▲───────────┘
           │ operationalize       │ distill
           │ (claim becomes       │ (verified results
           │  testable)           │  become understanding)
┌──────────▼──────────────────────┴───────────┐
│  SACCADE (shared, provenance-backed)        │
│  hypotheses ← findings ← tasks/experiments  │
│  claims-under-test · authority = criteria   │
└──────────┬──────────────────────▲───────────┘
           │ operationalize       │ evidence
           │ launch agents        │ runs, receipts
┌──────────▼──────────────────────┴───────────┐
│  Executors (jernerics, shells, CI, humans)  │
└─────────────────────────────────────────────┘
```

Memory types, precisely:
- **Journal/Zettelkasten** — formation memory. Episodic/reflective. Agent-free by
  constitution: an agent writing your notes removes the cognitive act that produces the
  insight. Permanently out of scope. Interface = graduation only.
- **Saccade** — shared propositional memory. What we currently believe, why, and what is
  being done about it — a consensus record with provenance.
- **Executors** — evidence production. Saccade never executes; it renders commands.

## 2. Stance: opinionated, Linear-not-Jira

Core semantics are fixed prescriptions in code, never configuration. There is no plugin
surface for invariants; adapters extend external systems only. The prescriptions, stated
as rules with the failure mode each prevents:

1. **Evidence never carries verdicts.** A finding supports, refutes, or abstains; the
   schema has no place for a conclusion. *Prevents: agents resolving under their own
   private standard.*
2. **Findings anchor to hypotheses.** Unanchored capture is allowed but flagged and must
   be anchored or discarded at triage. *Prevents: floating facts degrading the evidence
   layer into a changelog.*
3. **Agent verdicts are provisional. Always.** No auto-approval timer, no trust
   escalation. *Prevents: "solved" under an agent's definition.*
4. **Criteria are declared before execution; amendments are justified events.**
   *Prevents: post-hoc goalpost moves, including by the human.*
5. **References are inert.** Commit text never transitions state — no `fixes #N`.
   *Prevents: evidence-free closure by keyword.*
6. **Nothing closes without a deposit** — a finding (claim-shaped knowledge) or a receipt
   (outcome record). *Prevents: "done" meaning "we felt finished."*
7. **Claims are leases; nobody preempts but you.** *Prevents: agents judging each
   other's priorities.*
8. **State is derived, never stored twice.** The event log is the only truth; views are
   projections. *Prevents: stale-state rot.*
9. **The system never judges**: no learning pedagogy, no theory storage, no productivity
   surfaces, no auto-synthesis of your notes. *Prevents: the system acquiring opinions
   about its owner.*
10. **What happened is visible at a glance and answerable from the log.** Board and
    log tell the story without archaeology; retrieval answers structurally, not by
    prose-reading. *Prevents: history only participants can reconstruct.*

Configurability was explicitly rejected: Trac/Redmine-style configurable workflows were
assessed as mechanisms to steal — we steal the enforcement machinery, not the
configurability. The workflow is the product.

## 3. Core model — three objects, four lifecycles' worth of difference

"Issue" was decomposed into the four incompatible jobs one object was silently doing.

| Object | The job | Lifecycle |
|---|---|---|
| **Task** | claimable work | `open → claimed → done`; drop from `open` **or** `done` (human-only); reopen (human, justified) |
| **Hypothesis** | a claim under test | `open → resolved(kind) → verified`; resolutions locked |
| **Finding** | an observation | `valid → superseded \| invalidated` (tombstones, never deletion) |

Key semantics:

- **Task `done` guard:** requires a deposit. Investigation tasks (serving a hypothesis):
  ≥1 finding linked via that hypothesis. Chores: execution receipt on the done event, or
  human close. Root/arc tasks: children all terminal **plus a wrap finding** — closing an
  effort and recording what it taught are the same event.
- **Task done ≠ hypothesis resolved.** Tasks conclude work and deposit the outcome;
  hypotheses resolve when versioned criteria are satisfied. Concluding and succeeding are
  different events; if the goal survives the conclusion, a *new* task spawns.
- **Supersession over silent re-scope:** a task's identity is its done-deposit — what
  receipt would honestly close it. Deposit changed → open the successor first, then
  void the old (release if claimed; drop naming the successor). Path changed → same
  task; the receipt records the detour. Names are create-time intent: stale in the
  how, never in the what.
- **Drop path:** `dropped` is a human-only judgment. Legal from `open` and from `done`;
  never from `claimed` — in-flight claims are protected (only lease expiry ends a
  claim non-consensually). Dropping from `done` is the void: the receipt survives in the
  log; the state marks the task a mistake. `dropped` is the sole terminal state.
- **Judgment-act payload:** judgment events (drop, release, reopen, overturn) wear one
  uniform shape: an optional prose note, nothing else. Classification is reserved for
  verdict kinds, where it is constitutive of the act — a resolution without a kind is
  undefined; a drop without a category is still a drop. Notes are prose at rest
  (strings underneath; enums never enter the query layer).
- **Behavioral acceptance:** completion judgment is revealed, not declared — done +
  silence = accepted; done → dropped = voided (the note carries why). Time-to-verdict
  comes free from the log. An explicit verified event is added only when accepted and
  never-reviewed stop being distinguishable.
- **Hypothesis resolution:** verdict kind (`confirmed`/`refuted`/`split`) is data on the
  resolution event, not states per kind. Resolution guard: a criteria version must exist
  and findings must satisfy it; without ratified criteria, only humans resolve.
  `resolved → verified` is the human step: affirm, or overturn (back to `open`,
  justification logged — the old verdict's full evidence chain survives).
- **Late contradicting evidence** never mutates a verdict: it raises a **contradiction
  flag** (projection). Human rules by appending (reopen or dismiss-with-reason).
  Invalidation events cascade to findings that depended on resolved methodological
  hypotheses (OSF-withdrawal semantics: tombstone + reason + back-pointer, content
  preserved).
- **`inconclusive` findings** park non-answers as recorded knowledge.
- **Hypothesis split** spawns children (child-of edges); parent terminal.
- **Finding graduation:** `superseded-by` pointer (dedup) or `invalidated` tombstone
  (erroneous observation). Content always preserved.
- **Structural invariant:** `task → experiment-slot → hypothesis ← finding`. Findings
  attach to hypotheses (with provenance of the producing run); never to tasks.
- **Derived states are projections, never stored:** `blocked`, `ready`, `stale`,
  `awaiting-verification`, contradiction flags, triage queue contents, digests.

Arbitrary grouping (epics/projects) is a **root convention**, not an object: unclaimed
tasks or hypothesis trees with children; wrap requires the wrap finding. Promotion
trigger: if arc-level queries prove inexpressible as projections, promote arc to an
object (event-sourcing makes that additive).

### Conventional (non-hypothesis) software work

Tasks are first-class free-standing objects; the knowledge graph is opt-in, not a tax.
Goal-tasks ("make datagen 2x faster") close with receipts (before/after numbers) — no
hypothesis required. Hypotheses appear only when the *why* is worth keeping ("generation
is CPU-bound in tokenization, not IO" is real knowledge; "tried multiprocessing, 1.4x,
moving on" is a receipt). The capture question is always: *what does this demand?* —
work, belief change, or nothing-but-done.

Research vs conventional software is a **distribution dial, not a mode**: same machine,
same invariants. The difference is where the deliverable sits — software succeeds when
the world matches the spec; research succeeds when the record matches the world.

## 4. Evidence, verdicts, criteria — three homes

| Concept | What it is | Where it lives |
|---|---|---|
| **Criteria** (the standard) | Versioned data; amendments are justified events | On the hypothesis |
| **Evidence** (the observations) | First-class findings with provenance + relation | Own objects; typed edge to one hypothesis |
| **Verdict** (the judgment) | Resolution event: kind + actor + tier + cited findings | Append-only event on the hypothesis; read via projected state |

Consequences: overturn never destroys history; the non-prescription rule holds by
construction (observation records and judgment records are different rows, written by
different tiers); late evidence can't mutate a locked resolution.

Findings are **non-prescriptive by construction**: an agent literally cannot write its
conclusion into its evidence.

**Pre-registration:** a hypothesis carries versioned confirm/refute criteria from
creation. Agent-created hypotheses carry *unratified* criteria — they gate resolution
only after human ratification (at review). This kills the "agent sets its own bar" trap.

## 5. Provenance and pointers

Two layers:

- **Act provenance** — every event in the log carries `{actor, tier, timestamp,
  justification}`. Uniform, structural, no exceptions.
- **Execution provenance** — pointer sets, only on content that reports the world:
  findings (mandatory, may degrade to `{system: "manual"}`) and receipts.

**Pointer sets are executor-agnostic.** Core shape: `{system, ref, extras, commit,
command, captured_at}`. The tracker semantically interprets only `commit` and
`captured_at`. Everything executor-specific is an opaque string in that system's
namespace. Rationale: *schema stability and generality* — the tracker is an index over
claims; executors are interchangeable from its vantage. (Not circularity-avoidance:
dogfooding Saccade to track Saccade or to track jernerics development needs only tasks
and git-commit provenance.)

**Adapters live outside the core:** per-system config providing shape validation
(syntax — the adapter owns it; the tracker owns storage; semantics stay with the
executor), deep-link templates, and replay-command rendering. A jernerics adapter is one
entry. Unknown `system` values are accepted, rendered verbatim, queryable by ref-equality
only, and **flagged at triage** until normalized.

**Consistency mechanism:** validated pointers extract at write time into
`pointer_index(finding_id, system, key, value)` — parse at write, never at read.
Cross-cutting queries are identity-grade only (`system + ref`, commit, time).
Executor-semantic queries ("which trials had recall < X") belong to the executor's query
surface, reached by link-out.

**The pointer is the receipt:** it makes every claim checkable against the world —
reproduction, auditability, dedup/identity, and completeness (thin pointers = incomplete
stories, surfaced at review). Honest limit: reachability, not immortality — replay is
best-effort over time; `captured_at` and the commit hash are the parts most likely to
survive decay.

**The tracker never executes and never stores executor content** — artifacts live in the
executor's systems; the tracker stores claims, pointers, and outcome summaries. Anti-
graveyard rule: nothing is kept alive solely for bookkeeping, because (git history +
pointer) reconstitutes any result. Cleanup stops being data loss.

## 6. Workflow — triggers, not clocks

Two clocks, each defined by **triggers**, not calendars:

**Operational** (continuous, mostly agents):
- **Capture** → one object, never more. `cap [-t|-h|-f] "text" [--to h-ID] [--criteria]
  [--done-when] [--sys … --ref … --commit …]`. Deterministic — no LLM classification.
  Unanchored findings and criteria-less hypotheses land **flagged**, never blocked.
  Edges to existing objects yes; multi-object creation no (batch API exists for agents).
- **Ready → claim → run → record.** Ready ordering: priority, then age, then ID.
- **Resolve:** only when criteria are satisfied; agent verdicts provisional.
- **Session start (agent):** pointer-only TOC (see §8) + on-demand search/open. No
  content auto-injection.

**Review** (demand-driven, human only):
- **Triage**, triggered by queue contents — *mechanical* questions with right answers:
  contradiction flags, unanchored findings, unratified criteria, unnormalized pointers,
  near-duplicates. Decision cards with inline deciding context, keyboard-first, batch
  ops. "Clean" is a visible state — the anti-graveyard is an empty queue, not a debt
  counter.
- **Arc walk**, triggered by needing direction — *judgment* questions: alive/dead per
  root, incompleteness (thin pointers, thin-evidence verdicts), distillation candidates
  (human-verified, no note pointer), ratify/overturn provisional verdicts. The split:
  **Triage is where things have right answers; Arc is where things have your answers.**
- **Wrap:** close an arc = deposit the wrap finding.
- Learning retrospectives are journal work, not system prompts. The system is
  learning-neutral end to end.

## 7. Agents

- **Human claim parity:** claiming by hand is the same one command / same button as
  agent dispatch. No intent field, no learn semantics anywhere in the schema. Dispatch
  (launching agents) is the human's routing decision, made at launch time.
- **Claims are CAS leases:** `open → claimed` with `(claimed_by, lease_expiry)`
  written atomically; 409 on contention → re-query ready → next item (deterministic
  ordering + jitter). Heartbeat renewal; expiry auto-reverts to `open` (system event).
- **No agent preemption.** Priority reorders `ready`, never interrupts a `claimed`
  task. Human `abort` is the only interruption, and it notifies via the event stream.
- **Concurrent hypothesis resolution:** CAS on the hypothesis version. A second, divergent
  resolution attempt 409s and becomes a **review flag** (disagreement = candidate
  evidence). The losing agent files its findings and moves on; it never renders two
  verdicts on one claim.
- **Tiers:** `Tier {human, agent}` today; `system` arrives with lease expiry. Tier rides
  the record provenance, so history is self-describing and retro queries
  (overturn-rate-by-agent) work.
- **Authority is an event-keyed table.** Each event kind names its requirement —
  `AnyTier` or `Require(tier)` — in an exhaustive match with no wildcard: a new event
  variant is a compile error until its requirement is named (silence is not a policy).
  Event-keyed, not command-keyed: one row per effect, regardless of which command —
  or system sweep — proposes it. Every decide arm routes through the gate (construct →
  enforce → probe → validate → emit), making the table the single point of policy
  mutation and firing authority before any world probing. Judgment acts (drop, verify,
  overturn, ratify, reopen) are human-only; work acts (claim, done) are tier-blind.
- **Agent identity:** per-agent tokens → per-agent actor names. Contention semantics and
  the ratification gate are blind without it.
- **SSE event feed** off the event log for 409s, preemptions, expirations.

## 8. Retrieval — agent-facing: pointers, not prose

**No content auto-injection.** Injection makes coverage free but every effect invisible;
the only honest attribution for injected content is controlled absence. Instead:

- **Session start delivers a pointer-only TOC** (~20 tokens): open hypotheses, valid
  findings, in-flight tasks on the anchor — `open <id>` to pull content. Cures
  unknown-unknowns by pointer, not by prose.
- **Search** (`search <keywords> [--type] [--status] [--valid-only]`): the deliberate
  retrieval primitive — "grep before you build." FTS5-backed.
- Every content access is a recorded event: `toc_served{session, anchor, ids}`,
  `search{query, results}`, `open{object}`. The full retrieval chain is provenance.

**Failure taxonomy (re-derivation detector):** when a new finding near-matches an
existing valid finding:
- TOC listed it, never opened → **pull-discipline miss**
- TOC didn't list it → **coverage/anchor miss**
- Opened and re-derived anyway → **disagreement** — candidate evidence, not a failure;
  should have been filed as a refuting finding or a flag

Primary metric: **re-derivation rate = misses ÷ findings filed**. Digest *usefulness* is
a counterfactual property, measurable only by rare pre-registered ablations (degraded
TOC vs full, comparable sessions) — never by consumption logs. Empty searches are
corpus-gap telemetry.

**Mechanism:** v0 is anchored graph traversal (anchor → hypothesis → verdict lineage +
valid findings + serving tasks + arc context) over the typed edges — deterministic,
explainable, zero LLM — plus FTS5 for the unanchored case. Embeddings are admitted only
via a pre-registered experiment ("misses ≥15%, FTS-irrelevant, over N weeks"). Session
digests are precomputed at write time; session start is one indexed read, zero LLM calls.

## 9. Views — five, trigger-shaped

Derivation rule: *a view exists iff it answers one recurring question, at one trigger,
from one projection.* Tests: the **trigger test** (one trigger per view; two triggers →
split) and the **leave-test** (a view does too little if its question can't be answered
inside it). Presentations are v1 and iterate against usage evidence; the questions are
the contract.

| View | Trigger | Question |
|---|---|---|
| **Digest** | session start | where do I stand? |
| **Triage** | queue contents | what's mechanically broken that I can fix? |
| **Object** | click | everything about this one thing |
| **Arc** | direction needed | where is this going, what needs judgment? |
| **Timeline** | surprise or distrust | what actually happened? |

Wireframes (v1, real items):

```
┌ DIGEST ─────────────────────────── search: [____________] ┐
│ READY · claimable now                                     │
│   t-14 bump postgres, verify restore            [claim]   │
│   t-3  make unicode test fail on demand         [claim]   │
│ OPEN HYPOTHESES                                           │
│   h-2 throughput decay has a named cause        [open]    │
│     ├ f-31 allocator refuted (microbench flat)[refutes]   │
│     └ t-9  instrument allocator at hour 4    [claimed]    │
│   h-10 vector 3x memory expected post-rebuild [agent✓]    │
│ ARCS                                                      │
│   t-20 perf push · alive · f-31 2d ago                    │
│   t-27 flaky nodes · STALE · no findings 6d               │
└───────────────────────────────────────────────────────────┘

┌ TRIAGE · card 2/4 ──────────────── remaining: 3 ──────────┐
│ UNANCHORED  f-41 "loss spike at 41k — epoch boundary?"    │
│ CONTEXT · candidate hypotheses:                           │
│   h-5  loss spikes are epoch-boundary artifacts  [match?] │
│   h-2  throughput decay has a named cause        [match?] │
│ ACTION  [anchor→h-5] [new hypothesis] [discard]           │
│ (1=first candidate, 2=second, n=new, d=discard, s=skip)   │
└───────────────────────────────────────────────────────────┘

┌ OBJECT · h-12: node drops caused by DNS timeouts ─────────┐
│ STATE: resolved(refuted) → verified(human)     [overturn] │
│ CRITERIA v1: "coredns logs show timeouts during drops"    │
│ FINDINGS                                                  │
│   f-33 [refutes]  coredns logs clean in 6 drop events     │
│         src: agent/see#88 · logs query                    │
│   f-37 [supports] LB TLS evictions correlate with drops   │
│         src: session#91 · replay: [copy command]          │
│ LINKS: vault → "flaky-node-shutdowns"                     │
│ HISTORY: created → resolved(refuted, agent) → verified    │
└───────────────────────────────────────────────────────────┘

┌ ARC · t-20 perf push ─────────── alive · started Aug 28 ──┐
│ THE PATH                                                  │
│  Aug 28  arc opened · goal: ingest p99 back to ~40ms      │
│  Sep 01  h-7 opened: regression is tail-scheduling        │
│  Sep 02  t-12 claimed → harness comparison run            │
│  Sep 03  f-22 filed: 200-loop, no unicode repro [inconc.] │
│  Sep 04  f-28 filed: p99 doubles on 1.38→1.47  [supports] │
│  Sep 05  h-13 opened: harness may mis-measure (TAINTS f-28)│
│  today   t-9 running: allocator instrumentation hour-4    │
│ JUDGMENT QUEUE                                            │
│  ◇ f-22 thin provenance · h-10 verified, undistilled      │
│  ◇ direction call: harness question blocks trusting f-28  │
└───────────────────────────────────────────────────────────┘

┌ TIMELINE ─────────────────────────────────────────────────┐
│ 14:02  f-37 filed (agent/see#91) supports h-12            │
│ 13:58  h-12 verified by you                               │
│ 09:12  t-14 claimed by agent/orion (lease 30m)            │
│ 09-05  f-33 filed, refutes h-12                           │
└───────────────────────────────────────────────────────────┘
```

Conceptual split: **Triage = right answers; Arc = your answers.** No dashboards, no
search-as-view (search box lives in the Digest header), no notes management (vault
pointers render via adapter, opaque otherwise), no productivity surfaces.

## 10. IDs and references

- Typed sequential: `t-1`, `h-42`, `f-17`. Self-describing in prose, token-cheap,
  sortable, collision-free on a single authority. Numbers never reused; tombstones burn
  IDs. Events are `(object_id, seq)` internally — not user-addressable. Hierarchy is
  edges, never ID syntax (no `h-42.1`).
- **References are inert.** ID-shaped tokens in any prose at rest — commits, chat,
  findings, event notes — produce backlinks (projection) and nothing else. No
  keyword-driven transitions, ever.
- Beads imports carry `imported_from` aliases (arbitrary id strings — real beads ids
  are `prefix-token`, not `bd-123`) so historical commit backlinks resolve; archive-only
  aliases dangle — the unloved resolving to nothing is the correct answer.
- **Proposals are seq-addressed** — identity is the birth event's log position
  (`accept 614`); no counter, no `p-` syntax. Tasks keep allocated dense ids:
  human-referenced daily (§17).

## 11. Storage and deployment

- **Event-sourced core:** the append-only event log is the only truth; every projection
  (views, ready, triage, digests, pointer_index, retro queries) is a deterministic
  replay — rebuildable, property-testable.
- **Bi-temporal stamps** (event time vs validity time) from day one — stamped
  immediately, consumed by nothing in v0. Event time defaults to entry time when
  unmeasured; backdating is for sourced data only (imports, adapters); human-cited
  "when it really happened" lives in receipt prose. Clock divergence is a provenance
  signal — equal stamps mean typed live. Carried because it cannot be backfilled and
  speculative features (stale-memory forensics, adapter flushes, heard-time retro
  metrics) will want a complete record.
- **SQLite embedded, WAL** as the default runtime everywhere; Postgres only as a later
  adapter if evidence demands. Single-writer + serialized appends make the CAS machinery
  trivial.
- **Single artifact, three configs:** dev/laptop (one command, local data dir, auth off),
  homelab (systemd/NixOS module, token auth, persistent volume), escape hatch (one
  container + one volume; docker-compose; any cheap VPS/PaaS). Not a managed SaaS.
- **Auth:** none in dev; per-actor tokens elsewhere. Agent tokens map to actor names and
  tiers.

## 12. Migration from beads

No built-in migration — and no runbook-as-artifact: three one-shot migrations don't
justify a procedure document. **Migration is an executed mapping over the public API.**
The value is the mapping test — real issues finding honest homes saccade's model would
have chosen — with API acceptance riding along for free. Mapping decisions live here;
`tests/` pins the expression over a curated corpus; each real import executes once, in
a sandbox first, producing a residue report (every gap becomes an amendment or a named
door).

- **Beads fields are detection heuristics, never the target ontology.** The bar for
  log presence is "saccade wants it," never "beads had it." The source's own enums are
  advisory (wild: `enhancement`, `chore`, `in_review`) — undocumented values are normal
  input, not exceptions.
- **The counterfactual-native test decides each item:** would this have been created,
  in this shape, had saccade been the tracker that day? Committed work is checkable
  evidence for yes; failures go archive-only — the mapping succeeding, not a gap.
- **Times and actors:** create at `created_at` under `created_by`; claim + done at
  `closed_at` under a synthetic importer actor (beads records no closer); receipt =
  close_reason verbatim + provenance line. Imports are the sanctioned `--at` path.
- **Judgment-shaped closures — duplicates, garbled creates, superseded-before-delivery,
  human-decision closes — stop at the gate:** imported as open tasks with aliases, with
  a drop-candidate report; humans perform the drops. Classification is agent reading
  work, never keyword or LLM auto-classification (receipts say "deferred" mid-prose);
  uncertain cases escalate.
- **Ids and edges:** aliases are arbitrary strings (real beads ids are prefix + token,
  e.g. `jernerics-cdf`); v0 carries them in the task name, constitutive payload is a
  door. Dotted ids (`jyl.13`) are manual child-numbering — become parent edges.
  `parent-child` deps migrate as `--parent` — ordering-is-grouping: plain children
  hang as stars, ordered ones chain; cross-group ordering and diamond dependencies
  are unexpressible under single-parent trees (door; recoverable from git history). Titles and aliases
  migrate; bodies, comments, labels, priorities don't — git history of the source repo
  is the archive, and nothing references it.
- **Policy is evidence-driven:** the shape distribution argues per-project
  `closed-only | full | archive-only` (jernerics is an archive at 213/214 closed; symlab
  is the live one). Hard cutover: final sync commits, beads goes read-only, then
  `.beads/` leaves the working tree. **Success is removal: receipts + log + git are
  self-sufficient, and anything that would send you back to the corpse is a residue
  item** — non-self-sufficient receipts ("per decision comments 45-48") are flagged and
  folded at import; live-context tasks are re-captured at cutover or wait on the
  task-context door.

## 13. Build order and ownership

Stack: **Rust** (axum, tokio confined to the HTTP edge, rusqlite, server-rendered
templates + htmx). Specific dependencies are chosen as they come up, not batch-decided.
Design was the hard part; implementation is transcription of settled invariants
(~5–7k LOC). Rust earns its keep exactly where the invariants live: state machines as
enums with exhaustive matching, newtype IDs (`TaskId`/`HypothesisId`/`FindingId`),
projections as pure functions over events, property tests on guards. Boring Rust — no
typestate machinery, no DDD ceremony.

Ownership tiers (the project is the first entry in its own triage queue):

- **Tier 1 — you write, agents review** (~1.5–2k LOC): event store, state machines,
  guards, verdict/criteria logic. The executable encoding of the workflow philosophy.
- **Tier 2 — agents write, you review and can modify** (~2k): projections, API, SSE,
  search.
- **Tier 3 — delegated** (~2k): UI templates, packaging, compose files.

Build order (each step independently usable):

1. Event store + guards + JSON API + CLI — the correctness core
2. Projections: ready/blocked, triage queue, digest traversal, FTS5 + search endpoint
3. Tokens + per-agent actors + SSE feed
4. Five UI views over existing projections
5. Instrumentation events + re-derivation detector
6. Packaging: NixOS module + container + compose
7. Migration runbook executed by an agent on a real project — acceptance test

## 14. Instrumentation and the retro loop

- `toc_served{session, anchor, ids}` — the intervention record (not a usefulness claim).
- `search{query, results}`, `open{object}` — deliberate retrieval acts.
- **Drop-reason distribution:** sourced from analysis-time tags on drop records,
  not event payloads — the codebook is ratified at the retro (pre-registered before
  reading the aggregate), and re-tagging appends. Segment by whether a `done` preceded
  the drop: voids and pre-work drops are different populations.
- **Re-derivation detector** (projection): new finding near-matches an existing valid
  finding → classify by served TOC: pull-discipline miss / coverage miss / attention
  miss / disagreement-as-evidence. Primary metric: re-derivation rate.
- **`retro` report** (saved SQL over existing projections, run on demand at review):
  misses by class, search gaps, orphan findings (valid, never opened — insurance vs
  undiscoverable), verdict overturn rate by agent, drop-reason distribution,
  object-type ratio drift, lease health.
- Improvement loop: recurring pattern → `cap -h` → pre-registered experiment → finding →
  adopt/reject. The tracker dogfoods its own epistemology for its own development.
  Limits, honestly: n=1 (patterns over quarters, not significance), counterfactuals
  still require ablations, classes are strong hints not proofs.

## 15. Prior art — what was validated and stolen

Surveyed: beads, Jira, Linear, GitHub, GitLab, Bugzilla, Trac, Fossil, Redmine,
Taskwarrior (dev); eLabFTW, SciNote, OSF Registrations, Registered Reports, MLflow, W&B,
DVC, Sacred, W3C PROV-DM, Wikidata (research); mem0, Letta, Zep/Graphiti, cognee,
LangGraph, Memobase, ADR/MADR, Task Master (memory).

**Steal:** Bugzilla resolve→verify (= verdict tiers); Bugzilla resolution taxonomy;
beads' ready-work primitive + hash-ID lesson (superseded by sequential IDs — no merge);
Fossil artifact-replay (= event sourcing); PROV-DM relation vocabulary; OSF justified
amendments + withdrawal tombstones; Registered Reports confirmatory/exploratory labeling
+ outcome-independent completion; Wikidata rank-based supersession; Graphiti
bi-temporality + supersession-never-delete; Mem0 v3 ADD-only capture; Memobase/Letta
precomputed session-start projection; MADR status-flip events; Jira transition validators
+ Trac permission gating (enforcement machinery, not configurability); MLflow
status-vs-lifecycle dimension split; Sacred QUEUED pre-registration hook + heartbeat;
Redmine per-tracker workflow *precedent* (fixed in Saccade, not configurable); beads
JSON-first CLI + MCP-able ergonomics.

**Fight (encoded as engine rules, not conventions):** single entity type with type tags;
git-JSONL merge as sync; auto-close on artifact merge; free-text close reasons; binary
open/closed; untyped dependency links without cycle guards; acceptance criteria as
editable text; single-user assumptions; per-deployment schema divergence; evidence-free
role gates; LLM-driven mutation of records; content auto-injection; configurable core
semantics.

**The gap (validated):** every individual mechanism exists somewhere; the combination —
typed lifecycles × evidence/verdict separation × versioned criteria × write-time digest
projections × verdict-carrying edges — exists nowhere.

## 16. Non-goals

- Managing or syncing the Zettelkasten (journal stays agent-free; graduation only)
- Storing theory — only its operationalized instances (hypotheses)
- Storing executor content (artifacts, configs, logs, plots) — pointers and summaries
- LLM calls in any read path; LLM classification at capture or import
- Dashboards, productivity/velocity surfaces, notifications, multi-user teams
- Managed SaaS hosting
- Distributed replicas (single authority is the honest homelab contract; IDs break first
  if that changes — accepted, documented)

## 17. Proposals — agent-drafted judgment acts, human-accepted

The gate conflated *authority* with *execution*: judgment verbs (drop, release) are
human-only, so an agent that detects a judgment-shaped fact — a duplicate, a
supersession, a move — had no legal residence for the evidence, and the gate queue
rendered as live work. The fix separates the two: approval is one recorded act,
execution another.

**Object taxonomy.** The log stores events; commands append them; the fold derives the
**world** — a set of **objects**, each an identity plus a state machine, born by one
event and transitioned by others. Species: task (allocated dense ids), proposal
(seq-addressed). Attributes (receipts, notes) ride on events and have no independent
life. Accept is the first two-object transaction; `decide` returning `Vec<Event>` is
what makes transactions representable.

**A proposal is a persisted, non-effective command** — an act awaiting effectiveness.

- **Embed, don't reference:** the payload carries the act self-contained (verb, task
  id, name). Rulings never consult anything outside the log.
- **Verb-set lockstep:** proposal-able verbs ≡ human-gated verbs (`drop`, `release`),
  enforced in the grammar — ungated acts are unrepresentable as proposals.
- **Propose-time validation:** the embedded command is validated like any command
  (target exists, transition legal) minus authority — illegal acts refuse at
  propose; the queue never admits a proposal that is already unacceptable.

**Lifecycle.** `open → accepted | rejected | withdrawn`; all exits terminal; **no
expiry, ever**.

- **Accept** (human-only): executes the embedded act as a compound write
  `[proposal_accepted, task_dropped | task_released]`. The act event keeps its own
  human gate, satisfied by the acceptor — the gate is never bypassed, only discharged.
  Actor = acceptor; the proposer is recovered by log join; receipts name both at read
  time.
- **Reject** (human-only): note required — the ruling is the evidence, and rejection
  stays in the log forever.
- **Withdraw** (agent): tier-gated only, no proposer restriction — validation depends on
  tier, never identity, an invariant kept until token-bound actors land (serve auth is
  the seed). Note required — same criterion as every event: the withdrawal's reason
  dies with the event, and a proposal is a public object, so retracting it edits the
  shared picture of pending judgments. Not a gate — no human input; the note is a
  documentation duty on the actor.
- **Re-propose:** always legal, no anti-repeat rule; the gate is per-act, and rejection
  notes carry conditions.
- **Inertness:** an open proposal has zero effect on its target's legality. No locks,
  no implicit effects, no auto-invalidation — the log changes only via commands.

**Derived staleness.** Effective status is computed, never stored: an open proposal
whose embedded act is currently illegal renders `stale` (with reason) in views and
fails accept loudly (`InvalidStateTransition`) — one check, two consumers. Stale
proposals exit by explicit reject. Door: t-3's system-actor sweep may reconcile them
if the residue offends.

**Guard layering** — same three table-pinned layers as tasks. Authority:
created/withdrawn `AnyTier`; accepted/rejected `Require(Human)`. Transition:
`Proposal::transition`, total table, terminals admit nothing. World: seq resolves,
embedded act re-validated through the task's own code path. Error ordering pinned:
authority → existence → transition → embedded re-check.

**Surfacing.** `sac proposals` with `--json` (bare, like every CLI read — the
version envelope is serve-surface only; CLI and binary share a process, so no
version skew is possible), the state column carrying derived staleness (`stale`
when the embedded act would be refused today); `TaskView` gains `proposal:
Option<{seq, verb}>` (open proposals only — the row is an attention cue and a
pointer; the full story lives at the seq). `sac log` renders proposal events
without special casing. A `--status open|stale` filter is a named door, not built.

**Why no generic envelope.** "Acts awaiting effectiveness" is the concept; the
authority-conditioned proposal is its only live member. Lease expiry (t-3) decomposes
as derivation + sweep, not a pending act. Build the special case; the door is named.

## 18. Glossary

- **Task** — claimable work; closes on deposit (finding or receipt)
- **Hypothesis** — a claim under test with versioned criteria; resolves to
  confirmed/refuted/split; agent verdicts provisional until human-verified
- **Finding** — an immutable observation with provenance and relation
  (support/refute/inconclusive); anchors to exactly one hypothesis
- **Receipt** — done-event payload recording an outcome without claim semantics
- **Arc** — a root task/hypothesis tree spanning weeks; wraps with a wrap finding
- **Pointer set** — executor-agnostic provenance receipt; adapter-validated; indexed at
  write
- **Verdict tiers** — agent-provisional vs human-verified resolution
- **Digest** — session-start projection: TOC for agents, landing view for humans
- **Triage** — the mechanical-integrity queue (right answers)
- **Arc view** — the judgment surface (your answers): path + judgment queue
- **Retro** — the report over the system's own event log consumed at review
- **Deposit** — what every close requires: the record got richer, or it isn't done
- **Proposal** — an agent-drafted judgment act (drop/release) awaiting acceptance;
  inert while open; seq-addressed
- **Accept** — human-only promotion: the proposal's embedded act executes as a
  compound write; the receipt names proposer and acceptor
- **Stale** — derived status of an open proposal whose embedded act is no longer
  legal on its target; exits by explicit reject
