# AGENTS.md - how agents work in this repo

Saccade is an opinionated, agent-native issue tracker: one Rust binary, an
append-only SQLite event log, a world that is a fold of that log, and a
tracker-owned Git repository holding each task's board. Read
`AXIOMS.md` before proposing anything architectural - it is the framework
the project exists under, not a suggestion. Ratified decisions live in the
tracker's own threads (`sac log`, `sac show t-N`); standing law lives in
this file. Working context for in-flight tasks lands as board content on
the task's board, revised by commits; logistics between participants is
messages. The dogfood tracker (this repository's own tracker) is where
work is claimed and closed; see `tracker/skill/SKILL.md` for the CLI
discipline. The skill's source lives in `tracker/skill/` and deploys into
projects with `sac skill install` - the installed copy is generated and
git-ignored, never authored here.

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

## Documentation

- Markdown lives in the root law files (`AGENTS.md`, `AXIOMS.md`) and in
  per-package docs - a package README or a package docs directory -
  written for anyone working on the repository, not for agents alone.
  Development know-how that sat in skills becomes package docs. The skill
  is the one exception: it describes `sac`'s commands exactly, shares
  sac's life cycle, and lives in `tracker/skill/`.
- A phase review checks locality and that no stray docs or files remain.
- The `.agents` directory carries no tracked files: deployed skill copies
  are generated, git-ignored, and refreshed by `sac skill install`.

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
- Content lives in the board repository; the tracker stores claims,
  references, and decision prose. A figure's file lives in the board tree,
  and a document too large for a text component signals an overloaded
  account, not a storage need. Publish a board revision when the account
  is ready to be judged, not merely because work happened.
- No LLM calls in any read path; no LLM classification at capture or import.
- The human read path is part of completion: execution capabilities and
  recorded facts alone are insufficient. The human must be able to
  follow work, reasoning, and consequential decisions without depending
  on a workspace host or a separate coordinator briefing.
- The journal stays agent-free; the tracker sees only operationalized
  instances, never theory.
- Workflow fires on triggers, not clocks.
- Identity follows reference: task ids are dense fold-derived positions;
  board revisions are dense per task and derived by the fold (`t-N@rK`);
  incarnations and messages have stable birth-record references, and
  their human-facing numbers are dense within each kind, derived by the
  fold, never competing persisted counters. Record seq remains
  provenance, not a human-facing object number.
- Breaking payload changes are allowed while this repo is the sole consumer;
  the promise that old logs always load begins at external adoption or
  painful volume, whichever comes first.
- Versioning is per package, each 0.x semver over the contract it offers:
  `sac` (the log format, the `/api/v1` wire, the CLI verbs), the board
  grammar crate (the `board.yaml` format), and the viewer. Within a
  package the minor position is breaking: a MINOR bump may break that
  package's contract and demands its binaries move in step; PATCH covers
  everything else; UI surfaces and internal modules are never breaking
  alone. 1.0.0 is the external-adoption line. Tags and flake outputs are
  per package, named `<package>-v<version>`; cross-package changes land
  as one tested commit; components declare the contract ranges they use.
- Shared code is one package per contract: a thing shares only when a
  second consumer needs it and its shape has settled; packages join only
  on an identical life cycle. Internal code used by several packages but
  carrying no public contract may live in an internal workspace crate
  named for what it does, never "common". Code with one user stays inside
  its package. Breaking a shared contract is a major bump and ratified;
  shared code may change internally without a bump as long as its
  behavior holds.
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
  data - messages, board paths, names, figure bytes, record
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
- The host contract is law; each tracker version names one dev host with
  a tested claim for exactly that version, pinned as a repo flake lock
  entry. The pin is a development guarantee, never a runtime gate;
  production joins the machine's own host server. Harnesses are
  interchangeable terminals within the host; the contract's verbs are
  the entire host boundary, and another host is a dialect plus its own
  claim, never an architecture change. The contract is the doors' floor,
  never the deployment ceiling: doors never depend on host capabilities
  beyond the contract, while the deployment may exploit host richness
  through the host's own no-fork surfaces - shipped host artifacts are
  conveniences on the host's faces, and no door routes through them.
  A launch asks the host for one terminal and never supervises it.
  The here door attaches the invoking terminal as a viewer of that
  launched terminal, never adopts the caller's shell address; detaching
  leaves the session running. Message delivery, waits, and attachment
  run through the same contract. There is no tracker-owned model loop
  or managed RPC harness.
- An incarnation identifies a native session, not a task, workspace, or
  running process, and has no lifecycle in the tracker. Turns, compaction,
  resume, and workspace handoff retain identity; fresh sessions and forks
  create another. The harness is part of native-session identity;
  resume uses the recorded harness, never a new default. Historical
  existence asserts no current execution. Session associations - host
  addresses, including historical adopted addresses - and saved locations
  are record facts; transient terminal handles are runtime information.
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
- Messages are durable communication, never board content or implicit
  thread publications. Agent messages carry machine-supplied sender,
  message, and reply references; native human input remains native. Known
  contacts come from creation, contact, or explicit introduction, not a
  global directory, authority, expertise, or an obligation to work.
- Messages are recorded before an immediate submission attempt. Submission
  means host-accepted ordered submission - written and submitted - never
  reading or action. Missing
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
  supplies activation, never judgment or a successor goal. Reliance is
  recorded by the consuming task's board incorporation, never an
  execution subscription or automatic continuation agent.
- Decisions are task events, written once. An offer names the task's
  current revision and carries no prose - the board is the account.
  Rejection requires grounds and reopens the task; grounds are the entry
  point into the why and the home for concerns that attach to no specific
  place. Acceptance certifies a truthful final state, whatever the
  outcome. Corrections are new decisions on later offers; grounds and
  explanations follow the cancellation precedent and are never revisable
  in place.
- The board is the task's account and its content channel: findings, dead
  ends, working state, and review annotations live there, revised by
  commits. A correction removes an annotation in the commit that
  addresses it; history keeps the annotated revision. There is no
  resolved/unresolved lifecycle and no separate withdrawal.

## Test philosophy

Placement is a contract:

- Unit tests also encode invariants - the authority and transition tables
  are invariants, pinned where they live. Placement is by entry point, not
  by invariant-hood.

- `tracker/src/record/objects/*.rs` tests: pure exhaustive transition
  inventory for the object the module defines.
- `tracker/src/record/decide.rs` tests: pure command translation and
  authority; the authority inventory states its expected cells
  independently of the table it pins; fixtures are constructed events and
  `World::new()` - never a `Log`.
- `tracker/src/record/store.rs` tests: the law's contracts over
  `store::execute` - cross-object invariants, authority ordering, the
  lifecycle and reliance guards - driven through the in-memory fixture
  (test-private, no storage layer) against the shared story; plus every
  `Reason` pinned and the command translation table.
- `tracker/src/boards/` tests: the Git effects' contracts - write order,
  crash points, merge behavior - over temporary repositories.
- `tracker/src/record/views.rs` tests: view derivations over folded
  worlds, including the ownership and standing-offer panels.
- `tracker/src/record/db.rs` tests: the persistence layer - one all-event
  round-trip; new event families ride the existing round-trip, they don't
  get their own.
- `tracker/tests/proptest_harness.rs`: random legal sequences never panic
  the fold, and the world `db::record` returns equals replay and reload.
- `tracker/tests/api.rs`: server, wire, and console contracts over HTTP -
  command round-trips, the tier law through the wire, malformed bodies,
  client send, the board routes, and the CLI face through a real
  subprocess, including the full offer/reject/annotate/accept loop.
- `grammar/` tests: fixture-driven parse contracts - every valid fixture
  parses, every invalid fixture refuses with its declared kind.
- `client/` node tests: the presentation decoder's parity with the
  grammar crate over the shared fixtures; the browser, native and remote
  lanes carry the rendered contracts.

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
- Phase reviews diff the test inventory by name against the previous
  phase: additions fine, every removal explained.

## Code comments

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
  positions, dense by construction. Stored incarnation, message, and
  workspace references identify birth records; displayed numbers count
  births of each kind, and board revisions count per task, not global
  record positions.
- Validation depends on tier, never actor identity - the exception is the
  self-acceptance door: an agent whose environment-declared task scope
  covers the offered task cannot accept it. Scope is disclosure riding
  the record, never ownership or incarnation identity. The CLI has no
  tier argument: `SACCADE_ACTOR` presence is agent tier, its absence
  human. Judgment acts (cancel, reopen) are human-only and executed
  with an explanation, never proposed.
- The board publish sequence is Git objects, then the event in one
  SQLite transaction, then the ref; the log decides current, offered
  and accepted, and refs plus the materialized checkout are derived
  caches repaired at startup. A crash leaves at worst orphan objects or
  a lagging ref; a logged commit missing from the repository refuses to
  serve.
- A commit names the commit it builds on and refuses if the current one
  moved; committing refuses while the task is offered, so an offered
  revision cannot move and no revision-moved guard exists.
- Acceptance requires every incorporation to cite exactly its source's
  accepted revision; reconciliation onto an accepted revision is an
  ordinary commit, with no new act.
- Checkpoints are snapshots mid-edit: they validate nothing, carry no
  event, chain onto the actor's own ref, and must be discarded before
  anything accepts the task.

## Native host UI development

- Agents can drive the desktop app on a headless development machine. This
  is a mechanical development loop, not a substitute for testing the human's
  actual client, remote-machine topology, and input behavior.
- Use `just herdr-ui up` (or `nix run .#herdr-ui -- up`) rather than
  reconstructing the runtime environment. The on-demand Linux package carries
  the pinned app/core, Weston/Xwayland, and XTest dependencies. It isolates
  app/core/XDG state, clears inherited terminal and tracker scope, and
  disables sound/telemetry. Do not add production machines to the sandbox.
- `just herdr-ui info` reports the app home and log directory. Read that
  home's installed `AGENTS.md` and generated type declarations before writing
  mods. Edit personal mods/layouts there; live reload does not require a
  rebuild. Read `tools/herdr-ui/herdr-ui.md` for commands and working-copy
  setup.
- After each mod edit, run `just herdr-ui check <mod-directory>`, inspect
  `just herdr-ui status`, then capture and actually inspect `just herdr-ui
  shot`. Use `do` for app commands and `input` for private-display input;
  `input sequence` keeps related actions on one virtual-device connection.
  Verify input effects in saved data as well as screenshots. A frame or
  successful CLI command alone does not establish working interaction.
- Use `--name <label>` before the command for independent sandboxes. `down`
  stops only sandbox-owned processes and retains edits/logs for the next
  `up`. Preserve useful evidence on the task before manually deleting a
  stopped sandbox. Never automate the person's display or stop another
  session's core/compositor. The real-client gate remains separate.
- Native Markdown support does not imply math rendering; exercise actual
  content and record missing capabilities. `just herdr-ui-test` exercises
  isolation and failure recovery without a display; the same contract is
  available as the flake's Linux `herdr-ui` check.

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
