# Saccade — Axioms

Epistemic status: this file names the beliefs the design stands on and the
discipline that admits them. The law layer is the record itself — enforced
behavior in code, ratified decisions in tracker threads, standing
constraints on the not-yet-built in AGENTS.md. This is the axiom layer:
what law and everything built derive from. Files here carry primaries only
and must not drift. Every change here is a regime change and gets its own
task and conversation.

## 1. Granting discipline

Axioms are granted sparingly — closer to never than to rarely.

1. **Independently held.** A belief you would hold even if the feature it
   might justify never existed. No post-hoc axioms.
2. **Multiple bearings.** An axiom serving one consequence is a hypothesis in
   disguise. If only one thing needs it, suspect the thing, not the ontology.
3. **World, not feature.** Axioms describe the world; standing law describes
   the tracker; what is built — the gate, the dock, the serve edge — is
   derived. A proposed axiom whose only customer is one desired build is
   that build pleading.
4. **Special-case axioms do not exist.** A concept that requires its own
   dedicated axiom is likely false; the concept is rejected, not the axiom
   granted.
5. **Every term names its backing** — axiom, standing law, or derivation.
   Unbacked terms are errors, not vocabulary. Silence is not a policy.
6. **Corrections must simplify.** When the understanding stops being simple,
   the carving is wrong. Patches that add special cases are the smell;
   corrections that delete cases are the evidence.

Anti-warp rules — the framing is flexible, so its tests are operational:

- **The destroy-the-state test.** Treat any state outside the record as
  already gone — the session ended, the process died, the cache cleared —
  then check the record's statements. All still true, every attribution
  answerable: it was convenience (a cookie, a memo), no violation. Any
  statement gone false — a claim whose holder no longer exists, an answer
  the record can no longer give: that was dependence, and axiom A is
  violated.
- **Evidence counting.** Only findings already on the books or mechanically
  checkable count as validation. Interpretation-only alignments are anecdote.

## 2. The axioms

### A. The record is the only durable memory

Agent knowledge is a consumable: it degrades as it is used, compacts
lossily when it degrades too far, and ends with its session — even a live
agent is partly forgotten. Mechanical state — browsers, harnesses, cookies —
persists but carries no meaning worth depending on. Nothing outside the
record is both meaning-bearing and durable, so nothing may depend on state
outside the record. Falsified by durable, non-degrading agent memory, if
that ever exists; that discovery narrows the rule rather than widening it.

Derivations:

- Orientation comes from the record. Anything working in the world
  self-locates from the log alone — at birth and mid-life alike; carry-over
  context is already degraded.
- Claims are held by names, not processes. A promise held by a session
  would be state outside the record.
- Attribution is record-side. Even a live agent forgets; the fold may
  summarize, never anonymize (#33 is this rule violated today).
- Receipts are written at done-time, at the peak of the writer's knowledge —
  it is all decay afterward.
- Documentation follows this axiom: a doc is never a second source of truth.
  Docs carry only the underivable — framework, intent, constraints on the
  future — and name their reader (agents: AGENTS.md and this file; the human:
  code, the record, conversation).

### T. Understanding is the limiting resource — telos, not axiom

The telos — the end the system exists to serve — is chosen, not believed of
the world; it is never tested, only served or abandoned.

A knowledgeable human guiding an agent runs at full confidence and full
speed exactly until the work leaves the domain of their knowledge: the bound
is not understanding in general but its edge. The tracker's job is to keep
that edge ahead of the work — the record extends the domain where
full-confidence guidance is possible. Of the system's three memories only
the human's compounds (the record is exact but inert, agent knowledge is
consumable, the human's understanding is the one thing both alive and
durable), so the record is how the durable knower compounds without
full-time contact. That record-substitution can hold the edge is the
hypothesis the episodes test, not a settled derivation: did the record keep
the human effective just past where their knowledge ended?

### S. One human — scope fact, named

The founding line — "a human, their notes, and their agents" — made
load-bearing: the served surface pins the tier because there is one candidate;
the signal, when built (§3), addresses without ambiguity. Multi-human is a
law change (the recurring identity bill), not a surprise.

## 3. Derived model (provisional; episode-tested, not ratified law)

Substrate: the world as a system of **demands**, each with an addressee and a
state (awaiting / satisfied). The founding invariant — nothing is here unless
it demands something — promoted to the unit of account.

Routing — the two ways a demand reaches its addressee:

- **Pull.** The addressee visits (surfaces: canvas, gate, ready list; the
  agent's TOC).
- **Push.** Only an unsatisfied human-addressed demand may arrive unprompted.
  The signal is its future carrier — unbuilt; the roadmap's judgment push.
  Counting rule: signal types are counted in kinds of demands, never kinds
  of events.

Tier-law commentary (not a new axiom): judgment binds the future; binding
requires an actor whose persistence is dependable; by axiom A that is the
record and the human. The human-only judgment acts — the authority table —
read as a consequence.
