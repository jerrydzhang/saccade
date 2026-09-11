# AGENTS.md — how agents work in this repo

Saccade is an opinionated, agent-native issue tracker: one Rust binary, an
append-only SQLite event log, a world that is a fold of that log. Read
`AXIOMS.md` before proposing anything architectural — it is the framework
the project exists under, not a suggestion. Ratified decisions live in the
tracker's own threads (`sac log`, `sac show t-N`); standing law lives in
this file. Working context for in-flight tasks lives
as comments on the tasks themselves (`sac comment t-N "…"`). The dogfood
tracker (`sac`, repo's own `saccade.db`) is where
work is claimed and closed; see `.agents/skills/saccade/SKILL.md` for the CLI
discipline.

## How decisions happen

- One decision at a time, in conversation. The human ratifies; then it lands
  in the record — the task's thread, AXIOMS.md, or standing law below, by
  kind. Arguments are welcome — when a counter-argument lands,
  concede cleanly and record why.
- The problem is never silently modified. Deviations get announced before
  enactment. Success claims name their evidence (tests run and counts, files
  touched, commands exercised).
- Tiered ownership: `src/objects/`, `src/events.rs`, `src/decide.rs`, and
  `src/store.rs` core semantics are human-authored. `src/wire.rs`,
  `src/main.rs`, `src/db.rs`, and test scaffolding are agent-drafted and
  reviewed line-by-line. Mechanical moves into human-owned files happen only
  when explicitly delegated.

## Vocabulary discipline

- Doc vocabulary and code vocabulary must consolidate (three incidents:
  RecordId/seq, accept/ratify, an invented "learns"). If a term appears in one
  place, sweep the other in the same change.
- Do not import conversational coinage into code or docs. A term earns its
  place by being ratified, and check for collisions with standing law
  ("no learn semantics anywhere" — don't add one).
- In-memory types take the width of their job: `usize` for indices and counts,
  `u64` for absolute times. Storage/wire convert explicitly at the boundary.

## Standing law

Law that binds the not-yet-built. These survived the DESIGN.md deletion
(2026-09-10) because nothing else carries them: code enforces only what
exists, and `AXIOMS.md` carries world belief, not tracker law. Everything
else in that file was derivable or outdated; git history holds the rest.

- Prescriptions live in code, never configuration; there is no plugin surface.
- Objects earn residency by being addressed after birth; only a rule change
  is an ontology change.
- No learn semantics anywhere: the system never infers from use; standing
  things are admitted by ratification, never by recurrence.
- Pointers, never payloads: artifacts live with executors; the tracker stores
  claims, pointers, and outcome summaries.
- No LLM calls in any read path; no LLM classification at capture or import.
- The journal stays agent-free; the tracker sees only operationalized
  instances, never theory.
- Workflow fires on triggers, not clocks.
- Identity follows reference: dense ids for things cited daily by humans
  (tasks); record-derived identity for things born once (proposals,
  comments).
- Breaking payload changes are allowed while this repo is the sole consumer;
  the promise that old logs always load begins at external adoption or
  painful volume, whichever comes first.
- No counts, no badges — the restraint invariant, restated for every
  rendering.
- Every capability that reduces contact must add record in the same change;
  comprehension rides the record, never the conversation.
- Single authority is the honest homelab contract; distributed replicas are a
  non-goal (IDs break first if that changes — accepted).

## Test philosophy

Placement is a contract:

- `src/decide.rs` tests: pure units over decide's own functions. Fixtures are
  constructed events and `World::new()` — never a `Log`.
- `src/lib.rs` `invariant` tests: drive through `Log::execute` against the
  shared `populate_log` fixture.
- `src/db.rs` tests: the persistence layer. One round-trip test per layer;
  new event families ride the existing round-trip, they don't get their own.
- `tests/mapping_corpus.rs`: worked examples of the executed beads mapping.
  (Known wart: the gate-queue deposit scenario lives there until a second
  non-mapping integration test justifies a file.)

Rules:

- New family, same contract → grow the existing test. New contract → new
  test. Four agent-written tests were deleted in one review round by this
  rule; consolidation is a win, bloat is watched.
- Exhaustive contract tests (authority table, transition tables) are
  inventories: new event kinds register in them, never in sibling tests.
- Test comments: one line, plain, self-contained. Say what the assertion
  pins ("the proposal's id is 1"). No citations of other docs, no
  design-history narration, no vocabulary that isn't in the code.
- Source comments: the repo is self-documenting, enforced by deletion.
  A comment earns its bytes only by saying what the code cannot (a
  non-obvious why, like ProposalCreated's missing id). Rationale lives in
  the record, not `///` blocks; narration of the next ten lines gets removed.
- Fixture data reads as narrative: `implement foo`, `migrate floop`, not
  `test task 2`. The log is a story even in tests.
- Keep the suite green (33 unit + 7 corpus at time of writing) and honest:
  a failing suite from a fixture change means the fixture changed a contract
  — find out which before editing assertions.

## Semantic trip-wires (look wrong, are right)

- `candidate`'s `None` arms emit the unresolvable event anyway — authority
  supersedes existence, and agents must not gain an existence oracle. Do not
  "fix" this to fail early.
- Task ids live in event payloads (dense, world-allocated). Proposal ids are
  log positions of their birth records — `ProposalCreated` carries no id and
  there is no counter. Don't add one "for consistency."
- Open proposals are inert: they never block their target. Rejecting locks
  was deliberate.
- Validation depends on tier, never actor identity. Agents never pass
  `--tier human`; judgment acts are proposed (`sac propose drop t-N --name`),
  not executed.

## Edit mechanics (scar tissue)

- Re-read the exact lines immediately before every edit — files change live
  during review sessions. Read regions fully; a truncated read once anchored
  an edit on a wrong assumption about test structure.
- Edit-tool batches are atomic: one failed hunk rolls back the whole call.
  Unicode quirks (ellipsis, em-dash) in anchor text are a known failure
  family — sed by line number is the fallback.
- `cargo build` is the judge, not grep (brace-list imports defeat grep).
- Verify each change before stacking the next: build, then suite, then a CLI
  smoke on a throwaway db for anything touching verbs or wire.
