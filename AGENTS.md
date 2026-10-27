# AGENTS.md - how agents work in this repo

Saccade is an opinionated, agent-native issue tracker: one Rust binary, an
append-only SQLite event log, a world that is a fold of that log. Read
`AXIOMS.md` before proposing anything architectural - it is the framework
the project exists under, not a suggestion. Ratified decisions live in the
tracker's own threads (`sac log`, `sac show t-N`); standing law lives in
this file. Working context for in-flight tasks lives
as comments on the tasks themselves (`sac comment t-N "..."`). The dogfood
tracker (`sac`, repo's own `saccade.db`) is where
work is claimed and closed; see `.agents/skills/saccade/SKILL.md` for the CLI
discipline.

## How decisions happen

- One decision at a time, in conversation. The human ratifies; then it lands
  in the record - the task's thread, AXIOMS.md, or standing law below, by
  kind. Arguments are welcome - when a counter-argument lands,
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
  ("no learn semantics anywhere" - don't add one).
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
  tracker stores claims, pointers, and outcome summaries. Publish an artifact
  when the object itself is useful to inspect or reuse, not merely because
  it was produced during execution.
- No LLM calls in any read path; no LLM classification at capture or import.
- The human read path is part of completion: execution capabilities and
  recorded facts alone are insufficient. The human must be able to
  follow work, reasoning, and consequential decisions without depending
  on a workspace host or a separate coordinator briefing.
- The journal stays agent-free; the tracker sees only operationalized
  instances, never theory.
- Workflow fires on triggers, not clocks.
- Identity follows reference: task ids are dense fold-derived positions;
  comments, incarnations, and messages have stable birth-record references.
  Their human-facing numbers are dense within each kind, derived by the fold,
  never competing persisted counters. Record seq remains provenance, not a
  human-facing object number.
- Breaking payload changes are allowed while this repo is the sole consumer;
  the promise that old logs always load begins at external adoption or
  painful volume, whichever comes first.
- Versioning is 0.x semver: the minor position is breaking. A MINOR bump
  may break log loading, the `/api/v1` wire, or CLI verbs, and demands
  binaries move in step. PATCH covers everything else; UI surfaces and
  internal modules are never breaking alone. 1.0.0 is the external-adoption
  line above.
- Required stored-format migrations are bounded version-to-version steps in
  code, run before serving with transactional validation and recovery
  safeguards. Behavioral breaks alone do not justify data rewriting.
- Rendering is descriptive, never persuasive: deterministic grouping,
  ordering, staleness, and standing-offer projection present record
  facts - neutral local cardinality included. No unread or aggregate
  counts of tasks, states, incarnations, or records; no detached badges, urgency
  emphasis, inferred priority/progress/success/correctness, or narrative
  annotations. Outreach is an attending actor's judgment, never a
  projection; the pull UI never competes or coerces.
- Repository-authored text is ASCII: source, identifiers, comments, docs,
  CLI and web output, UI labels, and agent-visible envelopes carry
  ordinary ASCII punctuation, and a repository check holds every tracked
  text file to the rule. The rule reaches authored text only: Unicode
  data - messages, comments, paths, names, artifact bytes, record
  content - stays Unicode-capable, and this rule never transliterates or
  rewrites it. Separately ratified semantic migrations keep their own
  authority over record content. Runtime Unicode needed to recognize or
  test inbound data enters source as escape sequences, and authored
  runtime labels and typography are ASCII bytes themselves, never escaped
  or encoded decoration.
- Every capability that reduces contact must add record in the same change;
  comprehension rides the record, never the conversation.
- Every telemetry emit names its reader and the moment they read it, the
  way comments earn bytes; lines duplicating the record or outliving
  their consumer are cut in the same change that notices them. The
  destination story (and any spans) waits for the consumer - a standing
  service, not before.
- Single authority is the honest homelab contract; distributed replicas are a
  non-goal (IDs break first if that changes - accepted).
- Native Pi stays the pinned harness, never the ambient binary. Its
  Saccade-integrated composition is separately callable, not a replacement
  for ordinary Pi. The host launches and resumes; the integration associates
  native sessions, delivers messages, and yields its own waits to input.
  There is no tracker-owned model loop or managed RPC harness.
- An incarnation identifies a native session, not a task, workspace, or
  running process, and has no lifecycle in the tracker. Turns, compaction,
  resume, and workspace handoff retain identity; fresh sessions and forks
  create another. Historical existence asserts no current execution.
  Native-session associations and saved locations are record facts;
  terminal handles are runtime information.
- Tasks, incarnations, and workspaces have separate identities. Participation
  need not begin with a task, and launching never transfers whole-task
  ownership implicitly. An incarnation can select, create, and switch
  workspaces; several incarnations can use one. New versus existing workspace
  choice is explicit. Established branch/path names survive task and session
  changes and do not derive from task or incarnation numbering.
- Workspace cleanup is reversible eviction, not retirement: preserve its
  identity, branch, saved sessions, and recorded work. Restoration uses the
  retained branch's current local tip, never an implicit reset or pull.
  Available and Cleaned change only through verified completed effects,
  not intermediate progress states. Probes check consistency; finding a
  directory does not silently restore a Cleaned workspace.
- Messages are durable communication, never comment variants or implicit
  thread publications. Agent messages carry machine-supplied sender,
  message, and reply references; native human input remains native. Known
  contacts come from creation, contact, or explicit introduction, not a
  global directory, authority, expertise, or an obligation to work.
- Messages are recorded before an immediate submission attempt. Submission
  means native input queue acceptance, never reading or action. Missing
  confirmation is missing evidence, not a delivery obligation or another
  resident lifecycle. Messages activate already-running recipients at
  supported boundaries, never restart stopped sessions; explicit manual
  resume drains no recorded backlog. Sending and waiting are separate;
  integration-owned waits yield to incoming input, not by aborting arbitrary
  work. Waits are read-side, not resident workflow objects. The human
  participates through native conversation and the descriptive pull surface,
  not unsolicited push signals.
- The requesting incarnation interprets returned contributions against its
  goal and retains continuation unless explicitly handed onward. Delivery
  supplies activation, never judgment or a successor goal. A use records
  reliance, not an execution subscription or automatic continuation agent.
- Comments add information rather than repeat prior accounts. Explicit
  structured references connect contributions; their explanations remain
  prose. Presentation revision never retargets them, substantive corrections
  are new contributions, and historical prose does not acquire inferred
  graph edges.
- Comments land at the tier they describe: work facts on the work thread,
  episode facts on the episode's container, direction facts on the direction
  home; session-shaped work writes no thread summary unless it changed
  law - the commit is the record.

## Test philosophy

Placement is a contract:

- Unit tests also encode invariants - the authority and transition tables
  are invariants, pinned where they live. Placement is by entry point, not
  by invariant-hood.

- `src/objects/*.rs` tests: pure exhaustive transition inventory for the
  object the module defines.
- `src/decide.rs` tests: pure command translation and authority; the
  authority inventory states its expected cells independently of the
  table it pins; fixtures are constructed events and `World::new()` -
  never a `Log`.
- `src/store.rs` tests: the law's contracts over `store::execute` - cross-object
  invariants, authority ordering, the lifecycle and reliance guards - driven
  through the in-memory `Log` fixture (test-private, no storage layer)
  against the shared `populate_log` story; plus every `Reason` pinned
  and the command translation table.
- `src/views.rs` tests: view derivations over folded worlds, including
  the ownership and standing-offer panels.
- `src/db.rs` tests: the persistence layer - one all-event round-trip; new
  event families ride the existing round-trip, they don't get their own.
- `tests/proptest_harness.rs`: random legal sequences never panic the fold,
  and the world `db::record` returns equals replay and reload.
- `tests/api.rs`: server, wire, and console contracts over HTTP - command
  round-trips, the tier law through the wire, malformed bodies, client send,
  and the CLI face through a real subprocess.

Rules:

- New family, same contract -> grow the existing test. New contract -> new
  test. Agent-written tests were deleted in one review round by this
  rule; consolidation is a win, bloat is watched.
- Exhaustive contract tests (authority table, transition tables) are
  inventories: new event kinds register in them, never in sibling tests.
- Fixture data reads as narrative: `implement foo`, `migrate floop`, not
  `test task 2`. The log is a story even in tests.
- Keep the suite green and honest:
  a failing suite from a fixture change means the fixture changed a contract
  - find out which before editing assertions.

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
  positions, dense by construction. Stored comment, incarnation, and message
  references identify birth records; displayed numbers count births of each
  kind, not global record positions.
- Validation depends on tier, never actor identity - the exception is the
  self-acceptance door: an agent whose environment-declared task scope
  covers the offered task cannot accept it. Scope is disclosure riding
  the record, never ownership or incarnation identity. The CLI has no
  tier argument: `SACCADE_ACTOR` presence is agent tier, its absence
  human. Judgment acts (cancel, reopen) are human-only and executed
  with an explanation, never proposed.
- Comments carry note, offer, accept, reject, or use payloads, an explicit
  home task, and structured citation mappings, never a recipient. Notes
  and offers carry their own authored title at birth - required,
  rendered as plain text beside the prose, revisable in presentation
  independently of the body; accept stays untitled and bodyless, and
  no other kind takes a title. Withdrawal hides a note's title and
  body together. Requests are messages. Offer addresses its whole
  task; accept and reject name its current offer. A use declares a
  fixed other source task; explicit binding or rebinding selects that
  task's current Offered or Concluded account with grounds. Earlier
  selections remain recorded, and source replacement or acceptance
  never changes the selection automatically. Changing source task
  creates another use. Use never creates acceptance, nor acceptance
  use.

## Edit mechanics (scar tissue)

- Re-read the exact lines immediately before every edit - files change live
  during review sessions. Read regions fully; a truncated read once anchored
  an edit on a wrong assumption about test structure.
- Edit-tool batches are atomic: one failed hunk rolls back the whole call.
  Unicode quirks (ellipsis, em-dash) in anchor text are a known failure
  family - sed by line number is the fallback.
- `cargo build` is the judge, not grep (brace-list imports defeat grep).
- Verify each change before stacking the next: build, then suite, then a CLI
  smoke on a throwaway db for anything touching verbs or wire.
