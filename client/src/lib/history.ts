// The history diff (t-14 "History"): a word diff that keeps Markdown's
// structure - the diff marks sit after list, heading and quote markers,
// and a changed list marker is moved in front of them. The parts align
// between revisions first; a changed text part word-diffs line by line
// after its markers. Pure here, pinned by the node suite; the history
// mode paints what this returns.

import type { Block } from "./board/model.ts";

export type Marked = { type: "same" | "add" | "del"; text: string };

// sentinel characters in the private use area: the renderer wraps
// these into real add/del spans after the markdown pass
export const ADD_OPEN = "\uE000";
export const ADD_CLOSE = "\uE001";
export const DEL_OPEN = "\uE002";
export const DEL_CLOSE = "\uE003";

// -- part alignment ----------------------------------------------------

/** The LCS length table for two sequences under a key function: the
 * one table both the part alignment and the line diff walk. */
function lcsLengths<A, B>(a: A[], b: B[], equal: (x: A, y: B) => boolean): number[][] {
  const table: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array<number>(b.length + 1).fill(0),
  );
  for (let r = a.length - 1; r >= 0; r--) {
    const row = table[r];
    const below = table[r + 1];
    if (!row || !below) continue;
    for (let c = b.length - 1; c >= 0; c--) {
      const diagonal = below[c + 1] ?? 0;
      const down = below[c] ?? 0;
      const right = row[c + 1] ?? 0;
      const x = a[r];
      const y = b[c];
      row[c] = x !== undefined && y !== undefined && equal(x, y) ? diagonal + 1 : Math.max(down, right);
    }
  }
  return table;
}

function partKey(block: Block): string {
  if (block.kind === "text" || block.kind === "annotation") return `t:${block.markdown.trim()}`;
  if (block.kind === "group") return `g:${block.title ?? ""}`;
  if (block.kind === "incorporation") return `i:${block.source.task}@${block.source.revision}`;
  if (block.kind === "figure") return `f:${block.input}`;
  return `u:${block.name}`;
}

export type PartAlignment = {
  op: "same" | "changed" | "moved" | "added" | "removed";
  /** The old part's index for same/changed/moved/removed. */
  oldIndex?: number;
  /** The new part's index for same/changed/moved/added. */
  newIndex?: number;
};

/** Align two revisions' top-level parts: an LCS on part identity
 * (kind plus content key), so surviving parts map across shifts and
 * the ops name what the revision did. A remove/add pair whose content
 * is equal is a move, not a change - the part survives. */
export function alignParts(oldBlocks: Block[], newBlocks: Block[]): PartAlignment[] {
  const oldKeys = oldBlocks.map(partKey);
  const newKeys = newBlocks.map(partKey);
  const table = lcsLengths(oldKeys, newKeys, (x, y) => x === y);
  const rows = oldKeys.length;
  const cols = newKeys.length;
  const alignment: PartAlignment[] = [];
  let r = 0;
  let c = 0;
  while (r < rows && c < cols) {
    const below = table[r + 1]?.[c] ?? 0;
    const right = table[r]?.[c + 1] ?? 0;
    if (oldKeys[r] !== undefined && oldKeys[r] === newKeys[c]) {
      alignment.push({ op: "same", oldIndex: r, newIndex: c });
      r++;
      c++;
    } else if (below >= right) {
      alignment.push({ op: "removed", oldIndex: r });
      r++;
    } else {
      alignment.push({ op: "added", newIndex: c });
      c++;
    }
  }
  while (r < rows) {
    alignment.push({ op: "removed", oldIndex: r });
    r++;
  }
  while (c < cols) {
    alignment.push({ op: "added", newIndex: c });
    c++;
  }
  // a removal directly followed by an addition is a change; a removal
  // and an addition of the SAME content anywhere are a move
  const merged: PartAlignment[] = [];
  for (const step of alignment) {
    const last = merged.at(-1);
    if (
      last &&
      last.op === "removed" &&
      step.op === "added" &&
      last.oldIndex !== undefined &&
      step.newIndex !== undefined
    ) {
      const removed = merged.pop();
      const removedIndex = removed?.oldIndex;
      if (
        removedIndex !== undefined &&
        oldKeys[removedIndex] === newKeys[step.newIndex]
      ) {
        merged.push({ op: "moved", oldIndex: removedIndex, newIndex: step.newIndex });
      } else if (removedIndex !== undefined) {
        merged.push({ op: "changed", oldIndex: removedIndex, newIndex: step.newIndex });
      }
      continue;
    }
    merged.push(step);
  }
  // a removal and an addition of equal content, separated in the walk,
  // are still a move: pair them by key
  const openRemovals = new Map<string, number>();
  for (const step of merged) {
    if (step.op === "removed" && step.oldIndex !== undefined) {
      const key = oldKeys[step.oldIndex];
      if (key !== undefined) openRemovals.set(key, step.oldIndex);
    }
  }
  const result: PartAlignment[] = [];
  for (const step of merged) {
    if (step.op === "added" && step.newIndex !== undefined) {
      const key = newKeys[step.newIndex];
      const removedAt = key === undefined ? undefined : openRemovals.get(key);
      if (removedAt !== undefined) {
        // drop the paired removal, keep the move
        const removalIndex = result.findIndex(
          (candidate) => candidate.op === "removed" && candidate.oldIndex === removedAt,
        );
        if (removalIndex >= 0) result.splice(removalIndex, 1);
        if (key !== undefined) openRemovals.delete(key);
        result.push({ op: "moved", oldIndex: removedAt, newIndex: step.newIndex });
        continue;
      }
    }
    result.push(step);
  }
  return result;
}

// -- markers -----------------------------------------------------------

/** A line's leading Markdown structure marker: list bullets and
 * numbers, headings, block quotes. The diff marks sit after these. */
export function markerSplit(line: string): { marker: string; rest: string } {
  const match = /^(\s*(?:>\s+|#{1,6}\s+|[-*+]\s+|\d+[.)]\s+))(.*)$/.exec(line);
  if (!match) return { marker: "", rest: line };
  return { marker: match[1] ?? "", rest: match[2] ?? "" };
}

/** A changed marker pair renders the marker change in front of the
 * content marks; an unchanged marker stays common. */
export function markerChange(oldMarker: string, newMarker: string): Marked[] {
  const marks: Marked[] = [];
  if (oldMarker !== newMarker) {
    if (oldMarker.length > 0) marks.push({ type: "del", text: oldMarker });
    if (newMarker.length > 0) marks.push({ type: "add", text: newMarker });
  } else if (newMarker.length > 0) {
    marks.push({ type: "same", text: newMarker });
  }
  return marks;
}

// -- word diff ---------------------------------------------------------

function words(text: string): string[] {
  return text.split(/(\s+)/).filter((piece) => piece.length > 0);
}

/** A word-level LCS diff of two strings: the marks ride words, the
 * whitespace between them stays common where it can. */
export function wordDiff(oldText: string, newText: string): Marked[] {
  const a = words(oldText);
  const b = words(newText);
  const table = lcsLengths(a, b, (x, y) => x === y);
  const rows = a.length;
  const cols = b.length;
  const marks: Marked[] = [];
  const push = (type: Marked["type"], text: string): void => {
    const last = marks[marks.length - 1];
    if (last && last.type === type) last.text += text;
    else marks.push({ type, text });
  };
  let r = 0;
  let c = 0;
  while (r < rows && c < cols) {
    const below = table[r + 1]?.[c] ?? 0;
    const right = table[r]?.[c + 1] ?? 0;
    if (a[r] !== undefined && a[r] === b[c]) {
      push("same", a[r] ?? "");
      r++;
      c++;
    } else if (below >= right) {
      push("del", a[r] ?? "");
      r++;
    } else {
      push("add", b[c] ?? "");
      c++;
    }
  }
  while (r < rows) {
    push("del", a[r] ?? "");
    r++;
  }
  while (c < cols) {
    push("add", b[c] ?? "");
    c++;
  }
  return marks;
}

/** Pair two multi-line texts line by line (an LCS on full lines), then
 * word-diff each changed line pair after its marker. One output line
 * per content line, newlines between lines, the structure kept. */
export function structuredLineDiff(oldText: string, newText: string): Marked[] {
  const oldLines = oldText.split("\n");
  const newLines = newText.split("\n");
  const table = lcsLengths(oldLines, newLines, (x, y) => x === y);
  const rows = oldLines.length;
  const cols = newLines.length;
  const out: Marked[] = [];
  const pushLine = (marks: Marked[]): void => {
    if (out.length > 0 && marks.length > 0) {
      const first = marks[0];
      if (first) first.text = `\n${first.text}`;
    }
    for (const mark of marks) {
      if (mark.text.length === 0) continue;
      const last = out[out.length - 1];
      if (last && last.type === mark.type) last.text += mark.text;
      else out.push({ type: mark.type, text: mark.text });
    }
  };
  let r = 0;
  let c = 0;
  while (r < rows && c < cols) {
    const oldLine = oldLines[r] ?? "";
    const newLine = newLines[c] ?? "";
    if (oldLine === newLine) {
      pushLine([{ type: "same", text: newLine }]);
      r++;
      c++;
      continue;
    }
    const oldNext = oldLines[r + 1];
    const newNext = newLines[c + 1];
    if (oldNext === newLine) {
      // the old line here is gone
      const parsed = markerSplit(oldLine);
      pushLine([{ type: "del", text: `${parsed.marker}${parsed.rest}` }]);
      r++;
      continue;
    }
    if (newNext === oldLine) {
      // the new line here is new
      const parsed = markerSplit(newLine);
      pushLine([{ type: "add", text: `${parsed.marker}${parsed.rest}` }]);
      c++;
      continue;
    }
    // a changed pair: the marker change moves in front, the words
    // diff after it
    const oldParsed = markerSplit(oldLine);
    const newParsed = markerSplit(newLine);
    pushLine([...markerChange(oldParsed.marker, newParsed.marker), ...wordDiff(oldParsed.rest, newParsed.rest)]);
    r++;
    c++;
  }
  while (r < rows) {
    const parsed = markerSplit(oldLines[r] ?? "");
    pushLine([{ type: "del", text: `${parsed.marker}${parsed.rest}` }]);
    r++;
  }
  while (c < cols) {
    const parsed = markerSplit(newLines[c] ?? "");
    pushLine([{ type: "add", text: `${parsed.marker}${parsed.rest}` }]);
    c++;
  }
  return out;
}

/** The marked lines as the reader's text: add and del segments ride
 * sentinel characters the renderer wraps; the line breaks stay
 * outside the sentinels so a mark never covers a line break. */
export function marksToSentinelText(marks: Marked[]): string {
  let out = "";
  for (const mark of marks) {
    let text = mark.text;
    let lead = "";
    if (text.startsWith("\n")) {
      lead = "\n";
      text = text.slice(1);
    }
    if (mark.type === "same") out += `${lead}${text}`;
    else if (mark.type === "add") out += `${lead}${ADD_OPEN}${text}${ADD_CLOSE}`;
    else out += `${lead}${DEL_OPEN}${text}${DEL_CLOSE}`;
  }
  return out;
}

// -- the marked board (t-34@r6 change 1) ---------------------------------

/** How a part changed between two revisions, carried on the part for
 * the renderer: added whole, modified (its words, or inside a group),
 * removed (kept where it was, struck). */
export type ChangeMark = "add" | "mod" | "del";

/** The change a marked part carries, if any. */
export function changeOf(block: Block): ChangeMark | undefined {
  return (block as Block & { change?: ChangeMark }).change;
}

function withChange(block: Block, change: ChangeMark): Block {
  return Object.assign({}, block, { change }) as Block;
}

function isText(block: Block): block is Extract<Block, { kind: "text" | "annotation" }> {
  return block.kind === "text" || block.kind === "annotation";
}

/** The newer revision's parts with the diff against the older painted
 * in, at every depth: a group that pairs up (same title, or a changed
 * title) is compared part by part inside; a changed text part carries
 * its word diff; an added part is marked whole; a removed part stays
 * where it was, marked. `changes` counts the changed leaf parts. */
export function markBoard(oldBlocks: Block[], newBlocks: Block[]): { content: Block[]; changes: number } {
  const content: Block[] = [];
  let changes = 0;
  for (const step of alignParts(oldBlocks, newBlocks)) {
    const oldPart = step.oldIndex !== undefined ? oldBlocks[step.oldIndex] : undefined;
    const newPart = step.newIndex !== undefined ? newBlocks[step.newIndex] : undefined;
    if (step.op === "added" && newPart) {
      content.push(withChange(newPart, "add"));
      changes += 1;
    } else if (step.op === "removed" && oldPart) {
      content.push(withChange(oldPart, "del"));
      changes += 1;
    } else if (step.op === "moved" && newPart) {
      content.push(withChange(newPart, "mod"));
      changes += 1;
    } else if ((step.op === "same" || step.op === "changed") && oldPart && newPart) {
      if (oldPart.kind === "group" && newPart.kind === "group") {
        const inner = markBoard(oldPart.content, newPart.content);
        const retitled = (oldPart.title ?? "") !== (newPart.title ?? "");
        const group = { ...newPart, content: inner.content };
        content.push(inner.changes > 0 || retitled ? withChange(group, "mod") : group);
        changes += inner.changes + (retitled ? 1 : 0);
      } else if (step.op === "changed" && isText(oldPart) && isText(newPart)) {
        const marked = marksToSentinelText(structuredLineDiff(oldPart.markdown, newPart.markdown));
        content.push(withChange({ ...newPart, markdown: marked }, "mod"));
        changes += 1;
      } else if (step.op === "changed") {
        content.push(withChange(oldPart, "del"), withChange(newPart, "add"));
        changes += 1;
      } else {
        content.push(newPart);
      }
    }
  }
  return { content, changes };
}
