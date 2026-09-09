# AMENDMENTS.md — the manual how-stale ledger

The tracker has no amendments yet (t-11). Until then, this file carries
**how-level working information** that tasks and receipts can't hold:
implementation state, review findings, working decisions, deferred wrangles.

Discipline:

- How-stale content only. What-level changes still spawn tasks (t-10/t-11 pattern).
- This is working memory, not a second DESIGN. Ratified semantics live in
  DESIGN.md; when a decision here gets ratified, it moves there and its entry
  here shrinks to a pointer or dies.
- When t-11 lands: migrate each section into annotations on its task and
  delete this file.

---

## t-9 — proposal + ratification (implementation ledger)

### Done and ratified (see DESIGN §17 for semantics; entries here are code-shape only)

- `ProposalId(pub RecordId)` — identity is the birth record's id, refinement
  typed as nesting. No counter.
- `Event::ProposalCreated` carries **no id** (see comment in events.rs):
  journaled decisions go in payloads, physical positions stay in envelopes.
- `World.proposals: BTreeMap<ProposalId, Proposal>`.
- Authority table: created/withdrawn AnyTier; rejected/accepted Require(Human).
- Notes on rejected/withdrawn are `String` (required); accepted carries none
  (evidence migrates to the task's drop receipt).

### Implementation decisions (not in DESIGN)

- `ProposalAction::target_event(note: &str) -> Event` — ONE function, empty-string
  probe convention. `validate` calls `target_event("")` (transition table reads
  only the state column); candidate's accept arm calls
  `target_event(&p.proposal_name)`. Two-method probe/journal split was rejected:
  discipline-equivalent, fewer parts wins.
- Bare-event-in-None is load-bearing: `candidate` emits `ProposalAccepted { id }`
  for missing proposals so that agents get `HumanOnly` (authority supersedes
  existence — no existence oracle) and humans get `InvalidProposalId` from
  validate. Do not "fix" candidate to error early.
- Act field is `proposal_name` (convention match with `task_name`); the act's
  note semantics live under that name until t-11 maybe relieves it.

### Review findings — status (re-checked after implementation pass)

RESOLVED: all findings F1–F5, plus the follow-on batch (done by agent, review level):
serde derives (ProposalId, RecordId, ProposalAction, TaskId), wire disassemble+assemble
arms for the 4 kinds, `RecordId` field made `pub` (TaskId-consistent), lib root
re-exports, main.rs verbs (`propose <drop|release> t-N --name`, `accept <n>`,
`reject <n> --note`, `withdraw <n> --note`, `proposals`), `parse_proposal_id`
(bare log position), `reject_code` gained `invalid_proposal_id`, DESIGN §10/§17/glossary
swept to accept/accepted (hypothesis-verdict "ratify" in §4/§7 untouched — different
concept).

Vocabulary F6 DECIDED: accept/accepted. DESIGN swept. Hypothesis/verdict ratification
keeps "ratify" (distinct concept, t-1 territory).

### Remaining work

NONE — t-9 closed (dogfood event #24, agent-tier done with evidence receipt).
Landed after the implementation batch: the full test plan (34 unit + 7 corpus
green — authority cells, no-oracle err ordering, propose-time legality trio,
compound order+atomicity, inertness+stale-atomic, re-propose, identity-in-envelope,
db wire round-trip, gate-queue deposit corpus), TaskView `drop#seq` mark + derived
`stale` in the queue (view joins take `&World`; ProposalMark = {seq, verb}, no
actor — the row is a cue and pointer), skill teaches propose-instead-of-execute,
SACCADDE→SACCADE typo fixed in usage message and skill.

Rulings recorded late: CLI `--json` stays bare everywhere (envelope is
serve-surface only — no version skew possible in-process); `--status open|stale`
filter is a named door, not built. Withdraw-on-Accepted pinned by the exhaustive
transition-table test (no separate test needed).

Follow-on tasks: t-10 (notes-required), t-11 (annotation object), t-3/t-4.

### Test plan (in pinning order)

1. ✅ proposal.rs transition table (written).
2. Extend `authority_table_gates_exactly_the_gated_events` with the 4 events.
3. Err-ordering: agent accept/reject on nonexistent id → `HumanOnly`;
   human → `InvalidProposalId`.
4. Propose-time: nonexistent task → `InvalidTaskId`; drop on Claimed →
   `InvalidStateTransition`; release on Open → `InvalidStateTransition`.
5. Compound: human accept of open drop-proposal on Open task → exactly 2
   records in order (accepted, dropped), task Dropped, note = proposal_name.
   Agent accept → `HumanOnly` AND zero records. Stale accept (target claimed
   meanwhile) → `InvalidStateTransition`, zero records.
6. Inertness: open proposal doesn't block claim/complete/direct-drop; accept
   after → stale failure.
7. Re-propose after reject → second proposal Open, first stays Rejected.
8. Withdraw on Accepted → `InvalidStateTransition`.
9. propose's returned record id == new ProposalId; replay round-trip preserves
   proposals.
10. Later (wire/CLI done): gate-queue corpus scenario — propose 15 drops, batch
    accept, one reject (the t-9 deposit as a corpus test).

---

## t-10 — notes-required amendment

EXECUTED (pending review): note `Option<String> → String` on TaskDropped/TaskReleased
(+ commands); backstop `require_reason` in decide::validate — trim-empty refuses with
`Reject::ReasonRequired`, checked AFTER existence/transition (ordering recorded in
DESIGN §17); scope = six fields (drop, release, withdraw, reject, propose-name,
receipt — receipt was previously unenforced, `done --receipt ""` passed); wire
`NotedPayload.note → String`, null payloads hit the existing Corrupt path (verified
live: a null record bricks list and log, naming seq+kind); CLI `--note` mandatory;
fixtures rewritten with narrative notes; new contract test `empty_required_text_refuses`
(six cells, records-unchanged). Dogfood #16 repaired deliberately (trigger dance, backfilled
note). 31 unit + 7 corpus green.

DEFERRED TO t-11: the text taxonomy — whether Receipt/notes/annotation bodies share a
type, whether a note is an inline annotation, whether a finding is an attachment. Ruled
in t-10 only: the RULE is uniform (require_reason), the TYPES stay distinct (receipt ≠
note; merging is syntax-driven typing). No new newtype until t-11's design wants one.

Ratified mid-execution: **null payloads refuse, no shim** (the `Corrupt` path —
zero code). The boundary is recorded in DESIGN §10: format stability is not
promised until external adoption or migration pain, whichever first. The one
affected record (dogfood #16, `task_released` with `note: null`) got a
deliberate owner-side repair: drop trigger, UPDATE, restore trigger.

---

## Open wrangles (parked)

- D4 second half: canvas sections vs pure-tree — leaning sections + collapsed
  DONE via `<details>`; undecided, owner: jerry.
- First-cutover fallback stays: gate-queue report + hand-run drops until t-9
  ships (proposals make the import session the first real dogfood).
