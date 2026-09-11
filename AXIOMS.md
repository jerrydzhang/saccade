# Saccade — Axioms

Epistemic status: this file names the beliefs the design stands on and the
discipline that admits them. The law layer is the record itself — enforced
behavior in code, ratified decisions in tracker threads, standing
constraints on the not-yet-built in AGENTS.md. This is the axiom layer:
what law and organs derive from. Files here carry primaries only and must
not drift. Every change here is a regime change and gets its own task and
conversation.

## 1. Granting discipline

Axioms are granted sparingly — closer to never than to rarely.

1. **Independently held.** A belief you would hold even if the feature it
   might justify never existed. No post-hoc axioms.
2. **Multiple bearings.** An axiom serving one consequence is a hypothesis in
   disguise. If only one thing needs it, suspect the thing, not the ontology.
3. **World, not feature.** Axioms describe the world; standing law describes
   the tracker; organs (signals, leases, docks) are derived. A proposed axiom
   whose only customer is an organ is the organ pleading.
4. **Special-case axioms do not exist.** A concept that requires its own
   dedicated axiom is likely false; the concept is rejected, not the axiom
   granted.
5. **Every primitive names its backing** — axiom, standing law, or
   derivation. Unbacked nouns are errors, not vocabulary. Silence is not a
   policy.
6. **Corrections must simplify.** When the understanding stops being simple,
   the carving is wrong. Patches that add special cases are the smell;
   corrections that delete cases are the signal.

Anti-warp rules — the framing is flexible, so its tests are operational:

- **The destroy-the-state test.** To know whether the system depends on some
  state outside the record, destroy it. Nothing true lost: there was no
  dependence. Something lies: there was.
- **Evidence counting.** Only findings already on the books or mechanically
  checkable count as validation. Interpretation-only alignments are anecdote.

## 2. The axioms

### A. Nothing may depend on state outside the record

The record is the only dependable persistence. Executor state — agent
sessions, harnesses, browsers, processes — exists but is architecturally
invisible: nothing may lean on it. The founding three-layer stack (journal /
saccade / executors) restated as the system's one architectural axiom.

Derivations:

- Orientation comes from the record. Anything dropped into the world must
  self-locate from the log alone — session-start orientation is a necessity,
  not an optimization.
- Claims are held by names, not processes. A promise held by a session would
  be state outside the record.
- Attribution is record-side. Who acted lives in the envelope; the fold may
  summarize, never anonymize (#33 is this rule violated today).
- Documentation follows this axiom: a doc is never a second source of truth.
  Docs carry only the underivable — framework, intent, constraints on the
  future — and name their reader (agents: AGENTS.md and this file; the human:
  code, the record, conversation).

### T. Understanding is the limiting resource — telos, not axiom

Implementation speed and decision quality are bounded by the human's live
understanding of the system; the tracker's job is to keep that understanding
alive as contact decreases. Objectives are chosen, not granted. That
record-substitution can replace contact is the hypothesis the episodes test,
not a settled derivation.

### S. One human — scope fact, named

The founding line — "a human, their notes, and their agents" — made
load-bearing: the served surface pins the tier because there is one candidate;
signals address without ambiguity. Multi-human is a law change (the recurring
identity bill), not a surprise.

## 3. Derived model (provisional; episode-tested, not ratified law)

Substrate: the world as a system of **demands**, each with an addressee and a
state (awaiting / satisfied). The founding invariant — nothing is here unless
it demands something — promoted to the unit of account.

Routing — the two ways a demand reaches its addressee:

- **Pull.** The addressee visits (surfaces: canvas, gate, ready list; the
  agent's TOC).
- **Push.** Only an unsatisfied human-addressed demand may arrive unprompted
  (the signal). Counting rule: signal types are counted in kinds of demands,
  never kinds of events.

Tier-law commentary (not a new axiom): judgment binds the future; binding
requires an actor whose persistence is dependable; by axiom A that is the
record and the human. The human-only judgment acts — the authority table —
read as a consequence.

## 4. Joints (owned unknowns)

- **Claim semantics** — t-3 reopened, not assumed. Does the lease survive
  axiom A at all? Propose-release (already law) is the candidate reaper.
  A clock enters only if episodes show judgment-latency hurting. Owner: jerry.
- **Comprehension refresh** — does pull suffice, or does understanding need
  scheduled contact? The telos's open mechanism. Owner: jerry.
- **Dispatch act-kind** — routing decision (judgment-shaped, human-only
  first) or work act? Constrained by the demand model, not settled by it.
  Owner: jerry.
- **Release granularity** — the authority table is event-keyed and cannot
  express self-release (work act) vs other-release (judgment). Found by the
  heat check; rides t-4. Owner: jerry.

## 5. Rejected axioms (do not re-propose without new evidence)

- **Logged-time is the only clock.** Organ pleading: motivated solely by
  lease expiry, and the lease did not survive the framing.
- **Agents are ephemeral.** An empirical claim, false as stated (sessions
  persist, weakly). Replaced by axiom A's architectural form: nothing depends
  on executor state whether or not it persists.
- **Accountability for irreversible acts.** Serves one unification of
  already-ratified law (the authority table); demoted to commentary. The law
  stands on its own ratification.

## 6. Provenance

Ratified in conversation 2026-09-10: roadmap → ontology → axiom discipline →
heat check against the landed architecture. Journey summaries on t-19. Heat-
check evidence: two genuine failures — `Claimed` holds no claimer (#33, on the
books before the framing existed) and the authority table's event-keyed
granularity (mechanically checkable) — plus discounted passes and honest
silences, per the anti-warp rules.
