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
  HTTP/CLI adapters, process/Git mechanics, formatting and CSS,
  fake-process harnesses, and repetitive registration of already-ratified
  variants. A file may mix both: task briefs name the ratified semantic
  interface and the exact delegated mechanics; ambiguity returns to the
  human, never resolved by path. Mechanical edits in decision-heavy files
  need explicit scope; policy in an effect file stays human-owned.
  Decomposition follows conceptual seams and review responsibility; file
  lists plan collisions and never assign authority.

## Vocabulary discipline

- Doc vocabulary and code vocabulary must consolidate (incidents:
  RecordId/seq, accept/ratify, an invented "learns"). If a term appears in one
  place, sweep the other in the same change.
- Do not import conversational coinage into code or docs. A term earns its
  place by being ratified, and check for collisions with standing law
  ("no learn semantics anywhere" — don't add one).
- In-memory types take the width of their job: `usize` for indices and counts,
  `u64` for absolute times. Storage/wire convert explicitly at the boundary.

## Standing law

Law that binds the not-yet-built: code enforces only what exists, and
`AXIOMS.md` carries world belief, not tracker law; git history holds the
rest. Admission bar: rules name kinds; no task ids, dates, counts, or
names of unborn doors; instance facts belong to the tracker, which ages
with them.

- Prescriptions live in code, never configuration; there is no plugin surface.
- Objects carry their own stages, identity, and born content; indexes over
  records, accumulations, and provenance facts live on the context.
- Transition tables are pure over the event: state variants carry event
  payloads, never envelope facts; who acted is a context's disclosure
  cache; identity and tier guards live in store arms.
- Objects earn residency by being addressed after birth; only a rule change
  is an ontology change.
- No learn semantics anywhere: the system never infers from use; standing
  things are admitted by ratification, never by recurrence.
- Pointers, never payloads: artifact bytes live outside the record; the
  tracker stores claims, pointers, and outcome summaries.
- No LLM calls in any read path; no LLM classification at capture or import.
- The journal stays agent-free; the tracker sees only operationalized
  instances, never theory.
- Workflow fires on triggers, not clocks.
- Identity follows reference: dense ids for things cited daily by humans
  (tasks); record-derived identity for things born once (comments,
  incarnations).
- Breaking payload changes are allowed while this repo is the sole consumer;
  the promise that old logs always load begins at external adoption or
  painful volume, whichever comes first.
- Versioning is 0.x semver: the minor position is breaking. A MINOR bump
  may break log loading, the `/api/v1` wire, or CLI verbs, and demands
  binaries move in step. PATCH covers everything else; UI surfaces and
  internal modules are never breaking alone. 1.0.0 is the external-adoption
  line above.
- Rendering is descriptive, never persuasive: deterministic grouping,
  ordering, staleness, and standing-offer projection present record
  facts — neutral local cardinality included. No unread or aggregate
  counts of tasks, states, incarnations, or records; no detached badges, urgency
  emphasis, inferred priority/progress/success/correctness, or narrative
  annotations. Outreach is an attending actor's judgment, never a
  projection; the pull UI never competes or coerces.
- Every capability that reduces contact must add record in the same change;
  comprehension rides the record, never the conversation.
- Every telemetry emit names its reader and the moment they read it, the
  way comments earn bytes; lines duplicating the record or outliving
  their consumer are cut in the same change that notices them. The
  destination story (and any spans) waits for the consumer — a standing
  service, not before.
- Single authority is the honest homelab contract; distributed replicas are a
  non-goal (IDs break first if that changes — accepted).
- Native pi stays the pinned harness, a flake input — never the
  ambient binary. The tracker supervises nothing: no managed RPC, no
  session ownership, no spawn-time lever composition — start asks the
  host for one terminal and never manages it. An incarnation is a
  record-derived identity (its birth record — the workspace
  establishment, one task, one workspace address) with no lifecycle
  and no session pointer; attachment effects beyond the submitted
  launch remain unbuilt.
- Attention routes through declared dependency and chain judgment,
  never addressing. No comment names a recipient. The human is reached
  by the descriptive pull surface; blocking CLI waits ride record
  cursors (read-side, no residency), and push signals are unbuilt.
- Comments land at the tier they describe: work facts on the work thread,
  episode facts on the episode's container, direction facts on the direction
  home; session-shaped work writes no thread summary unless it changed
  law — the commit is the record.

## Test philosophy

Placement is a contract:

- Unit tests also encode invariants — the authority and transition tables
  are invariants, pinned where they live. Placement is by entry point, not
  by invariant-hood.

- `src/objects/*.rs` tests: pure exhaustive transition inventory for the
  object the module defines.
- `src/decide.rs` tests: pure command translation and authority; the
  authority inventory states its expected cells independently of the
  table it pins; fixtures are constructed events and `World::new()` —
  never a `Log`.
- `src/store.rs` tests: the law's contracts over `store::execute` — cross-object
  invariants, authority ordering, the lifecycle and reliance guards — driven
  through the in-memory `Log` fixture (test-private, no storage layer)
  against the shared `populate_log` story; plus every `Reason` pinned
  and the command translation table.
- `src/views.rs` tests: view derivations over folded worlds, including
  the ownership and standing-offer panels.
- `src/db.rs` tests: the persistence layer — one all-event round-trip; new
  event families ride the existing round-trip, they don't get their own.
- `tests/proptest_harness.rs`: random legal sequences never panic the fold,
  and the world `db::record` returns equals replay and reload.
- `tests/api.rs`: server, wire, and console contracts over HTTP — command
  round-trips, the tier law through the wire, malformed bodies, client send,
  and the CLI face through a real subprocess.

Rules:

- New family, same contract → grow the existing test. New contract → new
  test. Agent-written tests were deleted in one review round by this
  rule; consolidation is a win, bloat is watched.
- Exhaustive contract tests (authority table, transition tables) are
  inventories: new event kinds register in them, never in sibling tests.
- Fixture data reads as narrative: `implement foo`, `migrate floop`, not
  `test task 2`. The log is a story even in tests.
- Keep the suite green and honest:
  a failing suite from a fixture change means the fixture changed a contract
  — find out which before editing assertions.

## Comment law

- Test comments: one line, plain, self-contained. Say what the assertion
  pins ("the claim is the ownership ledger's last entry"). No citations of
  other docs, no design-history narration, no vocabulary that isn't in
  the code.
- Source comments: the repo is self-documenting, enforced by deletion.
  A comment earns its bytes only by saying what the code cannot (a
  non-obvious why, like Accept's missing prose). Rationale lives in
  the record, not `///` blocks; narration of the next ten lines gets removed.

## Semantic trip-wires (look wrong, are right)

- decide is structurally stateless: it has no world and cannot see existence,
  so authority always precedes it, and every command is its events with no
  expansion. Do not give decide a world "for convenience."
- No birth event carries its id; references do. Task ids are fold-derived push
  positions, dense by construction; comment and incarnation ids are the
  record positions of their births, with no counter.
- Validation depends on tier, never actor identity — the exception is the
  self-acceptance door: an agent whose environment-declared task scope
  covers the offered task cannot accept it. Scope is disclosure riding
  the record, never ownership or incarnation identity. The CLI has no
  tier argument: `SACCADE_ACTOR` presence is agent tier, its absence
  human. Judgment acts (cancel, reopen) are human-only and executed
  with an explanation, never proposed.
- Comments carry a payload variant — note, request, offer, accept,
  reject, use — never an address. Offer addresses its whole task;
  accept and reject address the offer they name and refuse a stale
  one; use lives on the consuming task, names another task's standing
  Offered or Concluded account, and never creates the acceptance;
  acceptance never creates the use.

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
