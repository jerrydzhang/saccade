# Source/claim map - saccade-and-migration.yaml

Companion to `boards/saccade-and-migration.yaml`. Every row names the
claims the board makes and the exact source inspected. Inspected revision:
main 6821415 (crate 0.18.0, branch migration-explanation-0ccca79b, clean).
Record references are dogfood tracker records read via `sac show`.

Status vocabulary matches the board: current / planned / open.

## 1. The current system

| Claim | Source | Status |
| --- | --- | --- |
| One binary `sac`, crate 0.18.0 | `Cargo.toml` | current |
| Append-only SQLite event log; world is a fold; readable faces are record-derived view statements | `AGENTS.md` (header), `src/store.rs` (`World`, `fold`, `stage`), `src/views.rs` (module doc), `.agents/skills/saccade/SKILL.md` | current |
| Fold-derived claims are view-scoped: artifact bytes, embedded workbench assets, the runtime address hint, the attempts side log, and host probes are not fold statements | `src/paths.rs` (state homes), `src/serve.rs` (asset set, address hint), `src/api.rs` (attempts), `src/host_contract.rs` (probe verbs) | current |
| Command is exactly its events; decide structurally stateless | `src/decide.rs` (`decide`), `AGENTS.md` (semantic trip-wires) | current |
| Actual write order in `store::execute`: decide translates the command first, then the tier gate applies to those events, then command-ingress checks, then fold guards/transition tables stage for append | `src/store.rs` (`execute`, 1043+); authority table `src/decide.rs` (`required_tier`) | current |
| Property test pins write/replay/persistence agreement: world returned == full replay == reload through the stored-format codec (persistence layer, not the HTTP wire); not a necessity proof | `tests/proptest_harness.rs` (`random_sequences_never_panic_and_replay_equals_live`, 596-619); `src/db.rs` (codec) | current |
| No birth event carries its own id | `AGENTS.md` (trip-wires), `src/events.rs` (payloads) | current |
| Task ids are dense fold push positions, not birth seq | `src/objects/task.rs` (`TaskId(pub usize)`), `src/store.rs` (`tasks: Vec<TaskContext>`, `task_born_at` resolves birth -> position), `AGENTS.md` (trip-wire: dense fold-derived push positions) | current |
| Comments/incarnations/messages: identity is the birth record (`RecordId` wrappers); public `c-N`/`i-N`/`m-N` are dense 0-based positions among same-kind births; `seq` is provenance | `src/objects/comment.rs`, `src/objects/incarnation.rs`, `src/objects/message.rs` (id docs), `src/store.rs` (`comment_num`, `incarnation_num`, `message_num`), `src/main.rs` ("c-0 through c-N" refusal prose, 1146) | current |
| Workspaces: identity is the birth-record reference; no public per-kind number derives from these births | `src/objects/workspace.rs` (`WorkspaceId` doc), `src/store.rs` (`World.workspaces` doc) | current |
| Numbering formula | board's own notation for the law in `src/store.rs` (`comment_num`); domain: comments, incarnations, messages | current |
| Tier possessed never passed; `SACCADE_ACTOR` set = agent, unset = human | `.agents/skills/saccade/SKILL.md` (Identity), `src/store.rs` (`Context`, `Tier`) | current |
| Cancel/reopen human-only with explanation; machinery verbs gated to the system tier (the role, not an actor name); cleanup/restoration effect-only | `src/decide.rs` (`required_tier`, `Authority::Require(Tier::System)`), `src/store.rs` (`enforce_tier`) | current |
| Self-acceptance: agent whose declared task scope covers the offered task cannot accept; scope is disclosure | `src/store.rs` (`Scope` doc; accept arm ~493-499; `Reason::SelfAcceptance`), skill (Identity) | current |
| Task states open/offered/concluded/cancelled(+explanation)/reopen; claim separate from state, gates nothing | `src/objects/task.rs` (`TaskState::transition`, `Owner`, `OwnerEntry`) | current |
| Comment kinds: note/offer titled at birth; accept bodyless; reject grounds; use declares fixed source task, bind selects current offered/concluded account | `src/objects/comment.rs` (`CommentKind`, `UseBinding`), skill (The model) | current |
| Concluded never reopens; corrections in new tasks | skill (The model), `src/objects/task.rs` (transition table) | current |
| Workbench single page at `/`, `/t/N`, `/c/N` deep links, `/a/<hash>` artifacts, workbench doors; dashboard/forms retired | `src/serve.rs` (module doc), `docs/service.md` (web UI paragraph) | current |
| Attempts log: one line per judged write beside the db | `src/api.rs` (`Attempts`, append sites), `docs/service.md` | current |
| Host contract verbs are the entire host boundary; tracker supervises no process | `src/host_contract.rs` (`HostContract` trait), `AGENTS.md` (standing law) | current |
| Workspace stages Available/Cleaned; cleanup reversible eviction; restore keeps identity | `src/objects/workspace.rs` (`WorkspaceStage::transition`, module docs) | current |
| Schema stamps: application id "sacd" + user_version; newer refuses by name; older refuses naming migration; foreign stamp refuses | `src/db.rs` (`APPLICATION_ID`, `SCHEMA_VERSION = 5`, `DbError`) | current |
| ASCII over every tracked text file; record content stays Unicode-capable | `prek.toml` (`ascii-authored-text` hook), `AGENTS.md` (standing law) | current |

## 2. What does not move

| Claim | Source | Status |
| --- | --- | --- |
| Axiom A (record is the only durable memory; docs never second truth) | `AXIOMS.md` section 2.A | unchanged foundation |
| Telos: understanding is the limiting resource; episode-tested, not ratified law | `AXIOMS.md` section 2.T | unchanged foundation |
| One human scope fact | `AXIOMS.md` section 2.S | unchanged foundation |
| Standing law carried into the migration: judgment acts human-only, no LLM in read paths, pointers not payloads, descriptive rendering, identity follows reference, contact-reduction adds record | `AGENTS.md` (standing law section) | unchanged foundation |
| Axiom A requires a durable record, not SQLite/append-only forever | Inference from `AXIOMS.md` 2.A (belief-level axiom; storage is derivation/law); the operational-history retention is separately pragmatic per t-56 c-883 | unchanged foundation |

## 3. Task blackboards (planned)

| Claim | Source | Status |
| --- | --- | --- |
| Goal: recover goal/spec/attempts/outcomes/dependencies without re-synthesis; boards versioned, curated | t-56 c-875 | planned |
| Declarative bounded grammar; renderer owns layout; grouping/order/captions/comparisons are content; no widget-operation history | t-56 c-876 | planned |
| Editing vs committing; working state may be invalid; commits are validated self-contained snapshots; structural validity only | t-56 c-876 | planned |
| Checkpoints (cheap, mutable, may be invalid; annotation lands as one) vs commits (coherent, validated, acceptance-eligible) | t-56 c-884 | planned |
| Incorporation = declaration edit with structured source ref, detected at commit; substantive receiving text; retirement grounds; no separate event kind | t-56 c-880 (with c-876 requirements) | planned |
| Provenance tooling must be the ergonomic normal path; agent paraphrase is the primary reliability risk | t-56 c-876 | planned |
| Acceptance names the current published revision and refuses on change; restore-then-choose; freezes the current intended committed state - newer checkpoints must be committed or explicitly set aside/restored, never silently omitted; stacked draft incorporation allowed; consumer acceptance requires accepted sources; older-revision acceptance superseded | t-56 c-883 (supersession), c-884 (committed-state refinement) | planned (resolved semantics) |
| Authority retained: edit/check/commit not owner-gated; self-acceptance guard stays; cancel/reopen human-only; concluded immutable; authored asides are ordinary text, review annotations are the declaration's review comments - no separate widget for authored asides, no resolved/unresolved lifecycle, no review stage | t-56 c-881 | planned |
| Rationale belongs in the account, not only commit messages | t-56 c-884 | planned |
| Explanatory boards are rendered declarations, no task/lifecycle | t-56 c-914 | planned |
| Git-backed boards; Git mechanics probe (six checks); acceptance is a tracker fact naming retained Git content, never merely a ref | t-56 c-877, c-878 | planned |
| Operational event log retained as pragmatic choice; mutable-tables question undecided, failure-trace evaluation | t-56 c-882, c-883 | planned / open |
| Minimal proposed tracker facts (creation, checkpoint advance, acceptance); payloads/guards need approval | t-56 c-883 | planned / open |
| Cutover: bounded agent-curated reset; read-only snapshot preserved; agents seed; no fabricated history; archive recoverable | t-56 c-878 | planned |
| Reset-scope boundary (incarnation/message/workspace/saved-session records, live identities) | t-56 c-879 | open |
| Removing the Offered stage is agreed direction (c-876, c-881); what remains open is the exact replacement and the comment/Use/Bind mapping onto boards | t-56 c-876, c-881 (agreed direction), c-879 (open mapping) | direction agreed / mapping open |

## 4. Client direction (ratified)

| Claim | Source | Status |
| --- | --- | --- |
| Read-only, useful, retained first viewer | t-56 record 2512 (c-1014) | ratified |
| Tauri 2 + Svelte 5 + TypeScript + Vite + shared CSS | t-56 c-997, c-1020; brief r4 section 5 | ratified |
| Independent windows; WM composes; opt-in view coordination; navigation never mutates agent execution/identity/ownership | t-56 c-1002..c-1005 | ratified |
| No app fork, no upstream contribution prerequisite | t-56 c-978 | ratified |
| Math: inherit text color; natural size; display equations scroll locally; wrapped formula one copy/search unit; indivisible inline overflow exception | t-56 c-924; t-64 c-943; t-65 c-963; t-66 c-964/c-973 (artifact 2390); t-56 c-1020 | ratified |
| Mermaid and code as Markdown fences; never execute; source preserved on failure; expanded-only zoom/pan | t-56 c-974, c-1020; t-68 c-981 (artifact 2424; class-diagram byte wobble named) | ratified |
| Literal-TeX reader rejected; math and Mermaid core requirements | t-56 c-980 | ratified |
| Explicit project -> workspace -> board selector, transitional | t-56 c-968, c-1018, c-1020 | ratified |
| Headless agent-friendly development; fast routine checks hard constraint | t-56 c-1007, c-1018 | ratified |
| Inspection fields/transport follow implementation (not a pre-implementation gate) | t-56 record 2519/c-1020, record 2520/c-1021 | deferred to implementation |
| Launch/PATH/distribution: plain Nix tooling; no bespoke launcher; foreground commands | t-56 records 2524-2529 (c-1024..c-1029) | deferred |
| Implementation in progress on t-71, independent worktree, contributor i-74 | t-71 c-1030, c-1033 | in progress |

## 5. Migration mechanics (current)

| Claim | Source | Status |
| --- | --- | --- |
| Bounded version-to-version migrations, run before serving, transactional validation/recovery | `AGENTS.md` (standing law), `src/migrate.rs` (`migrate`) | current |
| Offline chain: v3 or v4 source; step 1 renumbers prose onto public numbering (v3 only); step 2 backfills authored titles; refuses to write if rewritten log cannot fold; source untouched; seq/timestamps preserved; output must not exist | `src/migrate.rs` (header, `migrate`, `renumber_rows`, `backfill_titles`), `src/db.rs` (`MIGRATE_FROM_VERSION = 3`) | current |
| t-9 example: 0.12.0 -> 0.13.0 parent removal; two historical parent links stripped; no renumbering; mutual refusal semantics by name; live-write slip honestly recorded (t-12/t-13) | t-9 c-171 (brief), c-172 (inventory + refusal design), c-176/c-177 (slip + verification) | current (done) |
| Version line: 0.13.0 schema 3; 0.14.0 use/bind + task-free incarnations (schema 4); 0.15.0 authored titles (schema 5, backfill step); 0.16.x messaging + reversible cleanup + reviewed web reader; 0.17.0 host-contract cutover; 0.17.1 pin; main reports 0.18.0 | commits 1ed24fa (0.14.0 prep), 5cb29b9 (0.15.0), 135bb90/8caf94a/6e31124 (0.16.x), 6677ae2 (0.17.0), e02a787 (0.17.1 pin), `Cargo.toml` (0.18.0) | current |
| 0.x semver, minor position breaking; old-logs-always-load begins at external adoption or painful volume, whichever comes first | `AGENTS.md` (standing law), `docs/service.md` (release pinning) | standing law |

## 6. Wording conflicts found (parked, not fixed in this task)

- `src/host_contract.rs` module header says "the six verbs every
  persistent-session host answers", but the trait now defines seven
  methods (launch, place, live, submit, wait, close, attach); `place`
  arrived after the header was written (commits 6f0bdab, 1fde795).
  Wording staleness only; no behavior claim depends on the count.
  Parked on t-72 (seq 2576) for the coordinator.
- `AGENTS.md` (semantic trip-wires) says "authority always precedes it"
  about decide; actual code order in `store::execute` runs decide first
  and applies `enforce_tier` to the translated events afterward. This
  map records the quoted phrase and the call order side by side and
  selects no resolution and no corrective direction; clarification
  belongs to the human.

## Verified record inventory

Threads read for this board: t-72 (c-1031), t-56 (c-875 through c-1029,
artifact records 2069/2071/2077/2078/2080-2083), t-70 (c-1008..c-1029,
brief r1 record 2502), t-71 (c-1030..c-1033), t-9 (c-171..c-177, offer
c-194), t-62 (c-922/c-927, artifact 2201), t-64 (c-935/c-936/c-943/c-951),
t-65 (c-945/c-963), t-66 (c-964/c-973, artifact 2390), t-67 (c-983),
t-68 (c-981/c-982, artifact 2424), t-69 (c-994/c-996), artifact record
2522 (brief r4, sha256 575b2ee2...). Provisional grammar and retained
fixture read from the authoring-exercise and t-62 bundle paths named on
t-72.
