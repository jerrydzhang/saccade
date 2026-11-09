# tracker

`sac`: one Rust binary over an append-only SQLite event log; the world
is a fold of that log; each task's board lives in the tracker's own Git
repository. This README carries the tracker's law - what the code must
honor and where its tests pin it. The board grammar's law is
`grammar/README.md`; world belief lives in the root `AXIOMS.md`.

## Serving as a service

The dogfood server is infrastructure, not a dev artifact: the flake's
devenv process manager runs it from the pinned release, started with
`devenv up` in the project. The `sac` process is declared in
`flake.nix`; this is the line it runs:

```
sac serve --repo /home/jerry/Projects/saccade --bind 0.0.0.0 --port 8811
```

with `restart.on = "on_failure"` (max 10) and a readiness probe on
`GET /` at port 8811. The dogfood server is production: read-only
doors only, unless the task itself is the change.

- `--repo` names the tracker: the shell may sit anywhere, so the db's
  repo is declared, not discovered from the working directory.
- `--bind 0.0.0.0` spans loopback (CLI default) and the tailnet. Any
  device that can reach the port writes at human tier through the
  client - the conceded boundary. The wildcard never lands in the
  address hint: `serve` publishes `http://127.0.0.1:<port>` beside the
  served database after a successful bind, and clients discover the
  server from that file.
- The process manager holds the process output; the warns also land in
  the same `attempts.jsonl` beside the db as every judged write, so
  one surface greps everything.
- The attempts log is the diagnoser's door: refusals carry the full
  request and the storage-seq cursor they died against. Reproduce a
  refusal with `sac clone --at <cursor> --out <path>`, serve the
  clone (`--db` names it), and re-post the logged request.
- Stop is the process manager stopping the process. The server owns no
  child processes. The board operations spawn one-shot git runs under
  the writer lock; nothing long-lived is ever kept.

Consumers pin a package with a flake input naming its tag, e.g.
`saccade.url = "github:jerrydzhang/saccade/sac-v0.20.0"` - the
lockfile is the version identity.

## Model law

Law that binds the not-yet-built: code enforces only what exists.
Admission bar: rules name kinds; no task ids, dates, counts, or names
of unborn doors; instance facts belong to the tracker, which ages with
them.

- Prescriptions live in code, never configuration; there is no plugin
  surface.
- Objects carry their own stages, identity, and born content; indexes
  over records, accumulations, and provenance facts live on the
  context.
- Transition tables are pure over the event: state variants carry
  event payloads, never envelope facts; who acted is a context's
  disclosure cache; identity and tier guards live in store arms.
- Objects earn residency by being addressed after birth; only a rule
  change is an ontology change.
- No learn semantics anywhere: the system never infers from use;
  standing things are admitted by ratification, never by recurrence.
- Content lives in the board repository; the tracker stores claims,
  references, and decision prose. A document too large for a text
  component signals an overloaded account, not a storage need.
- No LLM calls in any read path; no LLM classification at capture or
  import.
- The human read path is part of completion: the human follows work,
  reasoning, and decisions without a workspace host or a separate
  coordinator briefing.
- The journal stays agent-free; the tracker sees only operationalized
  instances, never theory.
- Workflow fires on triggers, not clocks.
- Identity follows reference: task ids are dense fold-derived
  positions; board revisions are dense per task (`t-N@rK`);
  incarnations and messages have stable birth-record references, and
  their human-facing numbers are dense within each kind. Record seq is
  provenance, never an object number.
- Rendering is descriptive, never persuasive: deterministic grouping,
  ordering, staleness, and standing-offer projection present record
  facts. No unread or aggregate counts, no urgency emphasis, no
  inferred priority or progress; outreach is an attending actor's
  judgment.
- Every capability that reduces contact must add record in the same
  change.
- Every telemetry emit names its reader and the moment they read it;
  lines duplicating the record or outliving their consumer are cut in
  the change that notices them.
- Single authority is the honest homelab contract; distributed
  replicas are a non-goal.
- The host contract is law: each tracker version names one dev host
  with a tested claim, pinned as a repo flake lock entry - a
  development guarantee, never a runtime gate. The contract's verbs
  are the entire host boundary; doors never depend on host
  capabilities beyond it. A launch asks the host for one terminal and
  never supervises it; the here door attaches the invoking terminal as
  a viewer; there is no tracker-owned model loop or managed RPC
  harness.
- An incarnation identifies a native session, not a task, workspace,
  or running process, and has no lifecycle in the tracker. The harness
  is part of native-session identity; resume uses the recorded
  harness. Historical existence asserts no current execution. Session
  associations and saved locations are record facts; transient
  terminal handles are runtime information.
- Tasks, incarnations, and workspaces have separate identities.
  Participation need not begin with a task; launching never transfers
  whole-task ownership. New versus existing workspace choice is
  explicit; established branch/path names never derive from task or
  incarnation numbering.
- Workspace cleanup is reversible eviction: preserve identity, branch,
  saved sessions, and recorded work; restoration uses the retained
  branch's current local tip, never an implicit reset or pull.
  Available and Cleaned change only through verified completed
  effects; finding a directory does not silently restore.
- Messages are durable communication, never board content. Agent
  messages carry machine-supplied sender, message, and reply
  references; known contacts come from creation, contact, or explicit
  introduction. Messages are recorded before an immediate submission
  attempt; submission is host-accepted ordered submission, never
  reading or action; missing confirmation is missing evidence, not a
  delivery obligation. Messages activate already-running recipients at
  supported boundaries, never restart stopped sessions. Waits are
  read-side, never resident workflow objects; the human participates
  through native conversation and the descriptive pull surface.
- The requesting incarnation interprets returned contributions against
  its goal and retains continuation unless explicitly handed onward.
  Delivery supplies activation, never judgment or a successor goal;
  reliance is recorded by the consuming task's board incorporation.

## Commands and decide

- decide is structurally stateless: no world, no existence checks;
  authority always precedes it, and every command is its events with
  no expansion.
- The command translation and authority tables are pinned as
  inventories in `decide.rs` and `store.rs`; new commands register in
  them.
- Validation depends on tier, never actor identity - the exception is
  the self-acceptance door: an agent whose environment-declared task
  scope covers the offered task cannot accept it. The CLI has no tier
  argument: `SACCADE_ACTOR` presence is agent tier, its absence human.
  Judgment acts (cancel, reopen) are human-only and executed with an
  explanation, never proposed.
- No birth event carries its id; references do. Stored references
  identify birth records; displayed numbers count births of each kind.

## The board write order

- Publish is Git objects, then the event in one SQLite transaction,
  then the ref. The log decides current, offered, and accepted; refs
  and the materialized checkout are derived caches repaired at
  startup. A crash leaves at worst orphan objects or a lagging ref; a
  logged commit missing from the repository refuses to serve.
- A commit names the commit it builds on and refuses if the current
  one moved; committing refuses while the task is offered, so an
  offered revision cannot move.
- Acceptance requires every incorporation to cite exactly its source's
  accepted revision; reconciliation onto an accepted revision is an
  ordinary commit.
- Checkpoints are snapshots mid-edit: they validate nothing, carry no
  event, chain onto the actor's own ref, and must be discarded before
  anything accepts the task.

## The pages (/view/v1)

The instruments' read surface: server-shaped hypermedia view documents
at `/view/v1/<location>`, settled by t-14 "Pages, the instruments'
read surface" and t-24@r2. The pages are pure record: they read the
same fold and board repository as every other read face, wait on the
record's sequence, and never watch the host - tracker logic does not
depend on Herdr. `/api/v1` stays `sac`'s own API and changes freely;
nothing writes through `/view/v1`.

- **A page belongs to a location, not an endpoint.** The location
  grammar v1 serves: `` (the root, naming the tracker, the view
  grammar majors and board grammar range, and the page families),
  `t-N[@rK]` (a board page), `t-N[@rK]/history` (a timeline of
  commits interleaved with decisions in record order; a pinned
  revision cuts the timeline at the commit that landed it),
  `t-N@rA..rB/diff` (the stored unified diff, one text part),
  `t-N[@rK]/file/<path>` (raw file bytes, typed by extension), and
  `t-N@rK/content/<select path>` (one pinned component). Only the
  file family answers with bytes; every other answer is a view
  document in `application/json`.
- **The view grammar v1 has six kinds**: text, group, fact, time,
  ref, figure. A kind added in a later minor carries a text fallback,
  so older instruments degrade instead of breaking. Two location
  rules hold in every family: a part's own `location` names what it
  is in the tracker's addressing grammar (a wrapped board part is
  `t-N@rK:/content/...`, an agents-page item would be `i-N`); a ref's
  `location` names where to go in the page-location grammar - the
  route tail after `/view/v1/`. A board page wraps the board in these
  kinds (an annotation rides a titled group; an incorporation rides
  its collapsed shape: source revision, receiving text, a ref per
  cited component), and the authored `board.yaml` stays reachable raw
  through the file route, so a viewer may keep decoding the authored
  source. Every history entry carries the full commit message, the
  decision's grounds or explanation, the session behind the record
  (the incarnation its recorded scope names, as a ref), and refs to
  its revision and the diff from the revision before it.
- **Waiting is a cursor on the same route**: every page carries its
  cursor `seq` (the fold's last record the page was computed at);
  `?after=<seq>` answers at once when a record relevant to the
  location has landed past the cursor, else parks on the landing wake
  with a 250 ms re-check floor (a spurious wake costs one read, an
  out-of-process writer costs no more than the floor) until one lands
  or the hold expires. Quiet is a normal answer: the page as it
  stands, with its current cursor. The hold defaults to 25 s and
  `?timeout=` caps at 55 s. Relevant means the records the fold
  attributes to the page's task (the wait filter's activity set); the
  root wakes on any record. A page that refuses today keeps its
  task's watch, so a wait parks through a missing board or revision.
- **Refusals are documents**: kind `refusal`, a fact naming the
  refusal's code, a text naming what would make it answerable, under
  an informative HTTP status (unknown location or bad query 400,
  unknown task, missing board, revision, component or file 404,
  degraded 503, storage 500).
- A later page family extends the one dispatch in
  `src/serve/pages.rs` - one arm in the location match, one line in
  the root page's family list, its own arm of the watch enum - and
  never a second surface.

## Test placement

Placement is by entry point, not by invariant-hood; unit tests also
encode the invariants.

- `src/record/objects/*.rs`: pure exhaustive transition inventory for
  the object the module defines.
- `src/record/decide.rs`: pure command translation and authority; the
  authority inventory states its expected cells independently of the
  table it pins; fixtures are constructed events and `World::new()`.
- `src/record/store.rs`: the law's contracts over `store::execute` -
  cross-object invariants, authority ordering, the lifecycle and
  reliance guards - over the in-memory fixture against the shared
  story; every `Reason` pinned; the command translation table.
- `src/boards/`: the Git effects' contracts - write order, crash
  points, merge behavior - over temporary repositories.
- `src/record/views.rs`: view derivations over folded worlds.
- `src/record/db.rs`: the persistence layer - one all-event
  round-trip; new event families ride it.
- `tests/proptest_harness.rs`: random legal sequences never panic the
  fold; the world `db::record` returns equals replay and reload.
- `tests/api.rs`: server, wire, and console contracts over HTTP,
  including the CLI face through a real subprocess.

Rules: new family, same contract, grows the existing test; new
contract, new test. Exhaustive contract tests are inventories - new
event kinds register in them, never in sibling tests. Fixture data
reads as narrative. A failing suite from a fixture change means the
fixture changed a contract: find out which before editing assertions.

## Vocabulary

- Doc vocabulary and code vocabulary consolidate: a term appears in
  one place or gets swept in the same change.
- No conversational coinage: a term earns its place by ratification.
- In-memory types take the width of their job: `usize` for indices and
  counts, `u64` for absolute times; storage and wire convert
  explicitly at the boundary.

## Code comments

- Test comments: one line, plain, self-contained - what the assertion
  pins, nothing else.
- Source comments: the repo is self-documenting, enforced by deletion.
  A comment earns its bytes only by saying what the code cannot.
  Rationale lives in the record, not `///` blocks.
