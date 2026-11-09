// The history mode (t-14 "History"): a mode that engages with the
// board at a revision. The timeline comes from the tracker's history
// page; each revision's authored board is fetched through the file
// family; the diff between neighbouring revisions is computed
// client-side by the pure half (history.ts). This module holds the
// mode's state and produces what the shell paints; it decides nothing
// about the tracker.

import { decodeBoard } from "./board/decode.ts";
import type { Board, Block } from "./board/model.ts";
import {
  alignParts,
  marksToSentinelText,
  structuredLineDiff,
} from "./history.ts";
import type { HistoryEntry, ViewPage } from "./pages.ts";
import type { TrackerSource } from "./tracker.ts";

export type RevisionText = { revision: number; text: string; board: Board };

/** The mode's state: the timeline, the position, the cached revision
 * texts, and the part the reader tracks. */
export class HistoryMode {
  readonly entries: HistoryEntry[] = [];
  /** The revision the board shows. */
  at = 0;
  /** The newest revision in the timeline. */
  newest = 0;
  private texts = new Map<number, Promise<RevisionText>>();
  /** The tracked part's content key; H and L walk its changes. */
  trackedKey: string | null = null;

  constructor(
    private readonly source: TrackerSource,
    private readonly taskPath: string,
  ) {}

  /** Load the timeline and land on `startRevision` (the shown one). */
  async load(startRevision: number | null): Promise<void> {
    const page: ViewPage = await this.source.history(this.taskPath);
    const entries = page.entries ?? [];
    this.entries.splice(0, this.entries.length, ...entries);
    let newest = 0;
    for (const entry of entries) {
      if (entry.event === "task_committed" && entry.revision !== null && entry.revision > newest) {
        newest = entry.revision;
      }
    }
    this.newest = newest;
    this.at = startRevision === null || startRevision < 1 || startRevision > newest ? newest : startRevision;
  }

  /** The commit entry that landed the revision, with its full message. */
  commitOf(revision: number): HistoryEntry | null {
    for (const entry of this.entries) {
      if (entry.event === "task_committed" && entry.revision === revision) return entry;
    }
    return null;
  }

  /** One revision's authored board, cached per revision. */
  revisionBoard(revision: number): Promise<RevisionText> {
    const cached = this.texts.get(revision);
    if (cached) return cached;
    const load = (async () => {
      const text = await this.source.boardTextAt(`t-${this.taskNumber()}@r${revision}`);
      const result = decodeBoard(text);
      const board: Board =
        result.status === "ok"
          ? result.board
          : { grammar: 1, task: `t-${this.taskNumber()}`, content: [], diagnostics: [] };
      return { revision, text, board };
    })();
    this.texts.set(revision, load);
    void load.catch(() => {
      if (this.texts.get(revision) === load) this.texts.delete(revision);
    });
    return load;
  }

  private taskNumber(): number {
    const match = /^t-(\d+)$/.exec(this.taskPath.trim());
    const task = match?.[1];
    return task === undefined ? 0 : Number(task);
  }

  /** The board at the position, with the diff against the revision
   * before it painted into the text parts' markdown. The board's
   * decoded parts carry the sentinel-marked markdown in place of the
   * changed parts' plain markdown; the shell renders it and wraps the
   * sentinels. */
  async boardAtPosition(): Promise<{ board: Board; changedKeys: Set<string>; markedPaths: Set<string> }> {
    const current = await this.revisionBoard(this.at);
    const changedKeys = new Set<string>();
    const markedPaths = new Set<string>();
    const board: Board = { ...current.board, content: current.board.content };
    if (this.at <= 1) {
      return { board, changedKeys, markedPaths };
    }
    const previous = await this.revisionBoard(this.at - 1);
    const alignment = alignParts(previous.board.content, current.board.content);
    const markedContent: Block[] = [...current.board.content];
    for (const step of alignment) {
      if (step.op === "changed" && step.oldIndex !== undefined && step.newIndex !== undefined) {
        const oldPart = previous.board.content[step.oldIndex];
        const newPart = current.board.content[step.newIndex];
        if (
          newPart &&
          oldPart &&
          (newPart.kind === "text" || newPart.kind === "annotation") &&
          (oldPart.kind === "text" || oldPart.kind === "annotation")
        ) {
          const marks = structuredLineDiff(oldPart.markdown, newPart.markdown);
          const marked = marksToSentinelText(marks);
          markedContent[step.newIndex] = { ...newPart, markdown: marked };
          const key = partKeyOf(newPart);
          changedKeys.add(key);
          if (newPart.pos.start.line >= 0) {
            markedPaths.add(`/content/${step.newIndex}`);
          }
        }
      }
      if ((step.op === "added" || step.op === "moved") && step.newIndex !== undefined) {
        const newPart = current.board.content[step.newIndex];
        if (newPart) changedKeys.add(partKeyOf(newPart));
      }
      if ((step.op === "removed" || step.op === "moved") && step.oldIndex !== undefined) {
        const oldPart = previous.board.content[step.oldIndex];
        if (oldPart) changedKeys.add(partKeyOf(oldPart));
      }
    }
    return { board: { ...board, content: markedContent }, changedKeys, markedPaths };
  }

  /** Step h (older) or l (newer); returns the new position. */
  step(direction: "older" | "newer"): number {
    if (direction === "older" && this.at > 1) this.at -= 1;
    if (direction === "newer" && this.at < this.newest) this.at += 1;
    return this.at;
  }

  /** Track the part a content key names: H and L then walk its
   * changes. */
  track(key: string): void {
    this.trackedKey = key;
  }

  /** The nearest revision where the tracked part changed, walking in
   * one direction; null when no change lies that way. */
  async walkTracked(direction: "older" | "newer"): Promise<number | null> {
    if (this.trackedKey === null) return null;
    let revision = this.at;
    while (true) {
      const next = revision + (direction === "older" ? -1 : 1);
      if (next < 1 || next > this.newest) return null;
      const [here, there] = await Promise.all([this.revisionBoard(revision), this.revisionBoard(next)]);
      const keyHere = findKey(here.board.content, this.trackedKey);
      const keyThere = keyHere === null ? null : findKey(there.board.content, this.trackedKey);
      const changed =
        keyHere !== keyThere ||
        (keyThere !== null && keyHere !== null && textAt(there.board.content, keyThere) !== textAt(here.board.content, keyHere));
      revision = next;
      if (changed) return next;
    }
  }

  /** Leave: the revision to read plain. */
  exitPath(): string {
    return `t-${this.taskNumber()}@r${this.at}`;
  }
}

function partKeyOf(block: Block): string {
  if (block.kind === "text" || block.kind === "annotation") return `t:${block.markdown.trim().slice(0, 200)}`;
  if (block.kind === "group") return `g:${block.title ?? ""}`;
  if (block.kind === "incorporation") return `i:${block.source.task}@${block.source.revision}`;
  if (block.kind === "figure") return `f:${block.input}`;
  return `u:${block.name}`;
}

/** The content key of the block at a select path, or null. */
function findKey(blocks: Block[], key: string): string | null {
  for (const block of blocks) {
    if (partKeyOf(block) === key) return key;
  }
  return null;
}

function textAt(blocks: Block[], key: string): string {
  for (const block of blocks) {
    if (partKeyOf(block) === key) {
      if (block.kind === "text" || block.kind === "annotation") return block.markdown;
      return key;
    }
  }
  return "";
}

// -- marginalia (t-14): runs of parts beside the full last-changed message --

export type MarginaliaRun = {
  /** The current revision's part indexes the run covers, in order. */
  indexes: number[];
  /** The revision whose commit last changed every part in the run. */
  lastChanged: number;
};

/** For each top-level part of the newest board: the oldest revision it
 * has been identical through, walking back - that revision's commit is
 * the one that last changed it. Adjacent parts sharing a revision
 * group into runs. */
export async function marginaliaRuns(mode: HistoryMode): Promise<MarginaliaRun[]> {
  const newest = mode.newest;
  if (newest < 1) return [];
  const boards: Board[] = [];
  for (let revision = 1; revision <= newest; revision++) {
    boards.push((await mode.revisionBoard(revision)).board);
  }
  const current = boards[boards.length - 1];
  if (!current) return [];
  const lastChanged: number[] = [];
  current.content.forEach((part, index) => {
    const key = partKeyOf(part);
    let oldest = newest;
    if (key !== undefined) {
      for (let revision = newest; revision >= 2; revision--) {
        const earlier = boards[revision - 2];
        if (!earlier) break;
        const matches = earlier.content.some((candidate) => partKeyOf(candidate) === key);
        if (matches) oldest = revision - 1;
        else break;
      }
    }
    lastChanged.push(oldest);
  });
  const runs: MarginaliaRun[] = [];
  current.content.forEach((_, index) => {
    const at = lastChanged[index];
    const run = runs.at(-1);
    if (run && run.lastChanged === at) run.indexes.push(index);
    else runs.push({ indexes: [index], lastChanged: at ?? newest });
  });
  return runs;
}
