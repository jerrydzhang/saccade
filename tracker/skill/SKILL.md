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

```yaml
grammar: 1
task: t-2
content:
  - text: |
      ## Question

      Why does the tone arm resonate near 7 Hz on the rebuilt plinth?
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
```

`text` is markdown with dollar math; `group` nests content under an
optional title; `figure` names a file in the board tree plus a caption;
`incorporation` cites another task's revision **by number** with select
paths and its own receiving text; `annotation` is a review concern
placed near what it addresses. The file is one YAML document: no
anchors, no aliases, no duplicate keys.

The steps: `sac create task "..."`, `sac board edit t-N [dir]`, write
board.yaml and its figures with your own tools, `sac board check
[dir]`, `sac board commit t-N [dir] -m "..."`, `sac offer t-N`.

## The authoring loop

Working directories are local to your machine; the server does every Git
effect. One iteration:

1. `sac board edit t-N [dir]` - download the current revision (or your
   checkpoint) into a local directory and print its path. With a terminal
   and no directory it opens board.yaml in $EDITOR and offers checkpoint
   or commit.
2. Edit board.yaml and its figures with your own tools.
3. `sac board check [dir]` - local validation: the board parses, figures
   exist within the 8 MiB bound.
4. `sac board commit t-N [dir] -m "..."` - validate, write, and record a
   published revision. The parent comes from the directory's recorded
   base; if another author moved the task, `sac board update` merges
   three-way onto the current commit, conflicts marked in place.
5. `sac board checkpoint t-N [dir]` snapshots mid-edit work without
   validating; `sac board discard t-N` drops your checkpoint chain, which
   must happen before anything accepts the task.

**Commit granularity is the resolution of the story.** Each commit is one
chapter of the account; a task committed once at the end has no story.
Commit when a chapter is complete: an annotation addressed, a finding
landed, a correction made.

**Name the relaxations before the result.** Every constraint relaxed,
assumption changed, or unstated freedom you used is written in the board
before the account presents its result, not after.

Other doors: `sac board get t-N[@rK]:/content/0` prints one component;
`sac board get t-N[@rK] --file <path> [--out <file>]` prints (or writes)
one tree file; `sac board get t-N[@rK] --out <dir>` writes a whole
revision out; `sac board diff t-N@rA..rB`; `sac board log t-N` (the task's
history: commits with messages interleaved with the decisions); `sac board
path t-N` (the materialized board's path on the server's machine).

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

Wake on the offer instead of polling: `sac wait t-N --after <seq>
--until offer` returns once a TaskOffered record lands past the cursor
seq (`sac log` or `sac show t-N` names it).

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
