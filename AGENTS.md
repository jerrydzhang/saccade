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
- Ownership is conceptual, never file-based. The human authors ontology and
  type distinctions, event payload meaning, authority and transitions,
  cross-object invariants, what effects may claim, view statements, and
  behavioral acceptance assertions. Agents draft codecs and DB plumbing,
  HTTP/CLI adapters, process/Git/Telegram mechanics, formatting and CSS,
  fake-process harnesses, and repetitive registration of already-ratified
  variants. A file may mix both: task briefs name the ratified semantic
  interface and the exact delegated mechanics; ambiguity returns to the
  human, never resolved by path. Mechanical edits in decision-heavy files
  need explicit scope; policy in an effect file stays human-owned.
  Decomposition follows conceptual seams and review responsibility; file
  lists plan collisions and never assign authority.

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
Admission bar: rules name kinds, never task ids — instance facts belong to
the tracker, which ages with them.

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
- Rendering is descriptive, never persuasive: deterministic grouping,
  ordering, staleness, and demand projection present record facts — neutral
  local cardinality included. No unread or aggregate counts of demands,
  tasks, states, incarnations, or records; no detached badges, urgency
  emphasis, inferred priority/progress/success/correctness, or narrative
  annotations. The push signal summons attention; the pull UI never
  competes or coerces.
- Every capability that reduces contact must add record in the same change;
  comprehension rides the record, never the conversation.
- Single authority is the honest homelab contract; distributed replicas are a
  non-goal (IDs break first if that changes — accepted).
- Comments land at the tier they describe: work facts on the work thread,
  episode facts on the episode's container, direction facts on the direction
  home; session-shaped work writes no thread receipts unless it changed law —
  the commit is the record.

## Test philosophy

Placement is a contract:

- Unit tests also encode invariants — the authority and transition tables
  are invariants, pinned where they live. Placement is by entry point, not
  by invariant-hood.

- `src/objects/*.rs` tests: pure exhaustive transition inventory for the
  object the module defines.
- `src/decide.rs` tests: pure command expansion and authority; fixtures are
  constructed events and `World::new()` — never a `Log`.
- `src/store.rs` tests: the fold's failure contracts — every `Reason` pinned,
  plus the command translation table.
- `src/demands.rs` tests: the exhaustive shared Human-demand candidate and
  staleness projection that signal and views both consume.
- `src/lib.rs` `pipeline` tests: what only the composed path can show —
  cross-object contracts driven through `Log::execute` against the shared
  `populate_log` fixture.
- `src/db.rs` tests: the persistence layer — one all-event round-trip; new
  event families ride the existing round-trip, they don't get their own.
- `tests/mapping_corpus.rs`: worked examples of the executed beads mapping,
  mapping only. (Known wart: the gate-queue deposit scenario lives there
  until `tests/executor.rs` takes it.)
- `tests/runner.rs`: the slice-1 runner end to end — real temp Git, real db,
  machinery verbs driven as the server will drive them.
- `tests/executor.rs`: real process/Git/server effect and crash contracts,
  with fake Pi and Telegram.

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

- decide is structurally stateless: it has no world and cannot see existence,
  so authority always precedes it. expand reads the record only after the tier
  gate. Do not give decide a world "for convenience."
- No birth event carries its id; references do. Task ids are fold-derived push
  positions, dense by construction; proposal ids are the record positions of
  their births, with no counter.
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
