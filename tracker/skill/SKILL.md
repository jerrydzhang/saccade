---
name: saccade
description: Use when running the `sac` CLI - working tasks, authoring task boards, offering and reviewing conclusions, or any question about the tracker's events, task states, or messages.
---

# Saccade (`sac`)

One binary over an append-only event log; the world is a fold of that log.
The record is the reasoning cache: what lands in it survives context loss,
handoff, and reboot. Ratified decisions land by kind: the task's thread
carries the working record; the law files and package docs carry standing
rules. There is no managed execution: the tracker records undertakings and
accounts, it holds no waits and supervises nothing.

Before your first tracker write, run `sac identity`. Tier is **possessed,
never passed**: `SACCADE_ACTOR` set means agent tier, unset means human
tier. Preserve an injected `SACCADE_ACTOR` and task scope; set your own
actor name only when the variable is absent, then check `sac identity`.
A session's environment may declare a scope (`SACCADE_TASK`,
`SACCADE_INCARNATION_SEQ`, `SACCADE_WORKSPACE`); the scope is disclosure
riding every record - never proof, never ownership. It feeds one gate: an
agent whose declared scope covers a task cannot accept that task's offer.

Every command below exists in this build; run `sac <command> --help` for
its flags. Output is human-readable by default and `--json` on every face.

## The model

- Tasks are named undertakings: `open -> offered -> concluded`, with
  `cancelled` as the human's stop and `sac reopen` as the human's resume.
  Ownership is separate from state: `sac claim t-N` records whole-task
  responsibility, transfers freely, and gates nothing.
- **The account is the board.** Each task carries a board document
  (board.yaml plus its figures) whose published revisions live in the
  tracker's own Git repository. A revision is self-contained; the story
  across revisions is told by commit messages.
- **Offer** (`sac offer t-N`) names the task's current revision and moves
  it to offered. No prose: the board is the account.
- **Reject** (`sac reject t-N --grounds "..."`) names the standing offer,
  requires grounds, and reopens the task.
- **Accept** (`sac accept t-N`) concludes the task: it certifies a
  truthful account of the final state, whatever the outcome, and only
  after the reviewer has checked that the board answers the task as
  posed. The common failure is an account of an easier problem: a
  constraint relaxed that the task never said could be relaxed. A board
  that narrows the task must say so up front, and a reviewer rejects one
  that does not. When in doubt, reject with grounds. An agent accepts
  only when whoever asked for the review asked it to decide; otherwise it
  reports and leaves the decision. Cancellation is for set-aside and
  cleanup, not for honest findings.
- **Incorporation and the acceptance rule**: a board may cite another
  task's revision (an `incorporation` component, source named `t-N`, full
  commit hash, positional select paths, plus receiving text). A task can
  be accepted only once every task it incorporates is accepted, at
  exactly the revision it cites; reconcile by editing the citation to the
  accepted revision and committing.
- Cancel and reopen are human-only, each with an explanation. Agents pass
  prose explicitly; with a terminal attached, omitted prose opens $EDITOR.

## Writing a board

A board is one `board.yaml` beside its figures (8 MiB each), naming the
five kinds and only these:

````yaml
grammar: 1
task: t-2
content:
  - text: |
      ## Question

      Why does the tone arm resonate near 7 Hz on the rebuilt plinth?

      ```mermaid
      flowchart LR
        cli["sac CLI"] --> api["/api/v1"]
      ```
  - group:
      title: Recorded setup
      content:
        - text: |
            The plinth rebuild replaced the plate with birch ply.
        - figure:
            input: figures/plinth-section.png
            caption: Section through the rebuilt plinth.
  - incorporation:
      source:
        task: t-3
        revision: 1
        select: [/content/0]
      content:
        - text: |
            The compliance figures set the predicted resonance window.
  - annotation: |
      Reviewer concern: the number assumes a 20 C room; the session log
      says 24 C.
````

`text` is markdown with dollar math, fenced code blocks, and Mermaid
diagrams in a mermaid-fenced block. Mermaid support is uneven across
Markdown renderers; the app renders it (client/port/src/lib/render.ts).
`group` nests content under an optional title; `figure` names a file in
the board tree plus a caption; `incorporation` cites another task's
revision **by number** with select paths and its own receiving text;
`annotation` is a review concern placed near what it addresses. The
file is one YAML document: no anchors, no aliases, no duplicate keys. Images are `figure` components, never Markdown images: `board check` and commit refuse `![alt](src)` in a text.

The steps: `sac create task "..."`, `sac board edit t-N [dir]`, write
board.yaml and its figures with your own tools, `sac board check
[dir]`, `sac board commit t-N [dir] -m "..."`, `sac offer t-N`.

## The authoring loop

Working directories are local to your machine; the server does every Git
effect. One iteration:

1. `sac board edit t-N [dir]` - download the current revision into a
   local directory and print its path. With a terminal and no directory
   it opens board.yaml in $EDITOR and offers commit.
2. Edit board.yaml and its figures with your own tools.
3. `sac board check [dir]` - local validation: the board parses, figures
   exist within the 8 MiB bound.
4. `sac board commit t-N [dir] -m "..."` - validate, write, and record a
   published revision. The parent comes from the directory's recorded
   base; if another author moved the task, `sac board update` merges
   three-way onto the current commit, conflicts marked in place.

## What the current board holds

**The current revision tells the current story; the history tells the
story through time.** Every revision stays readable forever (`sac board
get t-N@rK`, `sac board diff`, `sac board log`), so the current board
never has to carry everything. Write it for a reader who wants the
present state, not the record of how it got there.

- **Rewrite, don't append.** When understanding changes, change the
  board to say what is now true. A correction replaces the claim it
  corrects; it is not a new paragraph beside it. Rewriting, folding and
  deleting are ordinary commits.
- **The commit message carries the move.** Say what changed and where
  the old version lives ("folds the r5 scale derivation into one line").
  A commit is one change in understanding, which may well be a rewrite.
- **Ideas that fell away shrink to a line** naming the revision that
  holds the argument: "Two-sided precision: the likelihood already
  rejects the zero field (r6)." That keeps them from being re-proposed
  without keeping them on the page.
- **The brief is the first revision.** A later board need not restate
  it; keep only what still shapes how the result reads.
- **Evidence supports a sentence.** Show what the claim needs (the
  check to run, the decisive rows); a full output dump belongs in a
  figure, a file in the tree, or an earlier revision.
- **Side findings are not the account.** A problem found along the way
  that the task did not ask about becomes its own task or a message.
- **Name the relaxations before the result.** A constraint relaxed or an
  unstated freedom used changes how the result reads, so it stays on the
  board, ahead of the result. Procedure that does not change the reading
  (defaults kept, files not touched, timings) does not.

Two shapes cover most boards:

- **A change account** (code, a fix, a feature): what changed, in a few
  sentences; how to check it (the behaviour to look at); its size (the
  diff); then any decision a reviewer must know about.
- **A standing account** (research, a design that keeps moving): the
  question as it now stands; the working position and the decisions
  behind it; the evidence it rests on; what is open; proposals not
  adopted; and what was set aside, one line each with its revision.
  Each revision rewrites these sections rather than adding a new one.

**Resuming a task**: read the current board, then `sac board log t-N`;
fetch the revisions its pointers name (`sac board get t-N@rK`) when the
detail matters. `sac board diff t-N@rA..rB` opens with which components
were added, removed, changed or moved, then the line diff.

**Writing a brief**: ask for checks, not content. "Done when the
listing tells the runs apart" leaves the account free to stay short;
"show the listing and state the totals" buys sections that never leave.

Other doors: `sac board get t-N[@rK]:/content/0` prints one component;
`sac board get t-N[@rK] --file <path> [--out <file>]` prints (or writes)
one tree file; `sac board get t-N[@rK] --out <dir>` writes a whole
revision out; `sac board diff t-N@rA..rB` (components first, then
lines); `sac board log t-N` (the task's history: commits with messages
interleaved with the decisions); `sac board path t-N` (the materialized
board's path on the server's machine).

## Reviewing

The reviewed object is the board revision, so review is a real judgment:

1. Read the offered board: `sac board get t-N --out <dir>` or open it in
   the viewer at `tracker/t-N`.
2. `sac reject t-N --grounds "..."` - the grounds are the entry point
   into the why: a short orientation, and the home for concerns that
   attach to no specific place. The task reopens.
3. **Annotate the reopened board**: commit an `annotation` component next
   to the content it addresses; a correction removes the annotation in
   the commit that addresses it. Publishing refuses while the task is
   offered - reject first, then annotate, then offer again.
4. `sac accept t-N` when the board is a truthful account.

Reject for a claim that is wrong or unsupported, or for a narrowing the
board does not state; not for procedure the history already holds.

## Names

An incarnation may carry a name, so the human tells sessions apart by
what they do rather than by i-N. `sac launch --name <name>` names a
fresh launch; `sac name <name>` names your own session (the declared
`SACCADE_INCARNATION_SEQ`), changes it any time, and `sac name --clear`
removes it; `sac identity` shows it. The human names any session with
`--incarnation i-N`; an agent names only its own. Every naming is
recorded, and names are never unique: same-named sessions are told
apart by i-N. A recorded terminal also gets the name as its Herdr
label; a refused label is reported and the name still stands.

## Messages

Coordination between participants is `sac send --to i-N "..."` - durable
addressed communication, never a thread entry. `sac messages` reads your
mailbox; `sac send --reply m-N` answers. Findings and review content
belong on the board; logistics belongs in messages.

## Search

`sac search <terms>` searches the current boards, one line per task.
`sac search --history <terms>` finds the revisions where terms appeared
or disappeared, with commit messages and decision grounds, narrowed by
`--in t-N` or `--in t-N@rA..rB`. A `t-N` or `t-N@rK` term is a reference
search: incorporations citing it, plus messages carrying the task.
Orienting on one task is not search: `sac show t-N` and reading the board.

## Naming

Revisions are dense per task and derived by the fold: `t-N@rK`. A file
inside a revision is `t-N@rK:/path`. The Git hash is provenance and is
what incorporations store; `rK` is what humans and agents address.
