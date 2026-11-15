// History mode's diff (t-49, t-57; t-14 "The viewer", History), its
// pure half: a revision shown against the one before it. The component
// alignment is the tracker's: the diff page serves the changes `sac
// board diff` names (tracker/src/boards/changes.rs), and the shown
// board follows them - the client keeps no alignment of its own. The
// result is ONE board to render: the shown revision's components, each
// added, changed or moved one marked, and every removed one placed back
// where it stood, marked removed, so the reader sees what went without
// leaving the page. Beside it, every shown part's place in both
// revisions: what keeps the part being read in place across steps.
//
// Inside a changed text the diff is by word, and keeps Markdown's
// structure: the marks ride private-use sentinels through the normal
// renderer (paintDiffMarks turns them into ins/del after rendering), a
// line's list, heading or quote marker always stays in front of them
// (a changed marker moves in front of the marks), math and code spans
// change whole, fence lines are never marked, and a Mermaid diagram
// shows its new source unmarked (marks would break the diagram).

import type { Block, Board, GroupBlock, IncorporationBlock } from "./model.ts";
import { selectToPart } from "./targets.ts";

export const INS_OPEN = "\uE000";
export const INS_CLOSE = "\uE001";
export const DEL_OPEN = "\uE002";
export const DEL_CLOSE = "\uE003";

export type ChangeMark = "add" | "mod" | "del" | "move";
type Marked = Block & { change?: ChangeMark };

/** One component change as the diff page serves it: select paths in
 * the revision each side names (`from` the earlier, `to` the later). */
export type ComponentChange = Readonly<{
  change: "added" | "removed" | "changed" | "moved";
  from?: string;
  to?: string;
}>;

/** A shown part and where it stands in each revision: `before` in the
 * earlier, `after` in the shown one; null where it does not exist (an
 * added part has no before, a removed one no after). Paths are
 * canonical part paths (/content/0/content/1). */
export type ShownPart = Readonly<{
  shown: string;
  before: string | null;
  after: string | null;
  change: ChangeMark | null;
}>;

export type BoardDiff = {
  board: Board;
  /** The changed parts in reading order, as rendered part paths: the
   * places n and N jump between. */
  changed: string[];
  /** Every shown component part, in reading order. */
  parts: ShownPart[];
};

/** The shown revision against its predecessor (null: the first
 * revision, everything added), aligned by the tracker's changes. */
export function diffBoards(
  previous: Board | null,
  current: Board,
  changes: readonly ComponentChange[] = [],
): BoardDiff {
  const changed: string[] = [];
  const parts: ShownPart[] = [];
  if (previous === null) {
    const content = current.content.map((block, index) => {
      const path = `/content/${index}`;
      changed.push(path);
      record(parts, block, path, null, path, "add");
      return markAll(block, "add");
    });
    return { board: { ...current, content }, changed, parts };
  }
  const index = indexChanges(changes);
  const content = diffBlocks(previous.content, current.content, "", "", "", index, changed, parts);
  return { board: { ...current, content }, changed, parts };
}

type ChangeIndex = {
  /** Earlier-revision parts that went (removed, or moved away). */
  gone: Set<string>;
  /** Later-revision parts that came (added, or moved here). */
  came: Set<string>;
  /** Moved parts: later path -> earlier path. */
  movedFrom: Map<string, string>;
  /** Earlier paths of the aligned pairs that changed whole. */
  changed: Set<string>;
  /** Every path a change names, either side, to tell a container to
   * walk in: [earlier paths, later paths]. */
  before: string[];
  after: string[];
};

function indexChanges(changes: readonly ComponentChange[]): ChangeIndex {
  const index: ChangeIndex = {
    gone: new Set(),
    came: new Set(),
    movedFrom: new Map(),
    changed: new Set(),
    before: [],
    after: [],
  };
  for (const entry of changes) {
    const from = entry.from === undefined ? null : selectToPart(entry.from);
    const to = entry.to === undefined ? null : selectToPart(entry.to);
    if (from !== null) index.before.push(from);
    if (to !== null) index.after.push(to);
    if (entry.change === "removed" && from !== null) index.gone.add(from);
    if (entry.change === "added" && to !== null) index.came.add(to);
    if (entry.change === "moved" && from !== null && to !== null) {
      index.gone.add(from);
      index.came.add(to);
      index.movedFrom.set(to, from);
    }
    if (entry.change === "changed" && from !== null) index.changed.add(from);
  }
  return index;
}

function childrenOf(block: Block): readonly Block[] {
  return block.kind === "group" || block.kind === "incorporation" ? block.content : [];
}

/** A shown part and its descendants, placed in both revisions by the
 * same child positions (`before` or `after` null: absent there). */
function record(
  parts: ShownPart[],
  block: Block,
  shown: string,
  before: string | null,
  after: string | null,
  change: ChangeMark | null,
): void {
  parts.push({ shown, before, after, change });
  childrenOf(block).forEach((child, index) => {
    const at = (prefix: string | null) => (prefix === null ? null : `${prefix}/content/${index}`);
    record(parts, child, at(shown)!, at(before), at(after), change === "mod" ? null : change);
  });
}

/** One level of the alignment: the earlier and later siblings under a
 * container both revisions hold. What the changes name as gone or come
 * is placed by them; every other sibling pairs in order - exactly the
 * tracker's longest common subsequence, so the pairs need no keys. */
function diffBlocks(
  previous: readonly Block[],
  current: readonly Block[],
  beforePrefix: string,
  afterPrefix: string,
  shownPrefix: string,
  index: ChangeIndex,
  changed: string[],
  parts: ShownPart[],
): Block[] {
  const beforeAt = (i: number) => `${beforePrefix}/content/${i}`;
  const afterAt = (j: number) => `${afterPrefix}/content/${j}`;
  const keptBefore = previous.map((_, i) => i).filter((i) => !index.gone.has(beforeAt(i)));
  const keptAfter = current.map((_, j) => j).filter((j) => !index.came.has(afterAt(j)));
  const pairs: [number, number][] = [];
  for (let k = 0; k < Math.min(keptBefore.length, keptAfter.length); k++) pairs.push([keptBefore[k]!, keptAfter[k]!]);
  const out: Block[] = [];
  const at = () => `${shownPrefix}/content/${out.length}`;
  const removed = (i: number) => {
    // a moved component shows where it went, never where it stood
    if ([...index.movedFrom.values()].includes(beforeAt(i))) return;
    const block = previous[i]!;
    changed.push(at());
    record(parts, block, at(), beforeAt(i), null, "del");
    out.push(markAll(block, "del"));
  };
  const added = (j: number) => {
    const block = current[j]!;
    changed.push(at());
    const from = index.movedFrom.get(afterAt(j));
    if (from !== undefined) {
      record(parts, block, at(), from, afterAt(j), "move");
      out.push({ ...block, change: "move" } as Marked);
      return;
    }
    record(parts, block, at(), null, afterAt(j), "add");
    out.push(markAll(block, "add"));
  };
  const inside = (paths: readonly string[], path: string) => paths.some((other) => other.startsWith(`${path}/`));
  let i = 0;
  let j = 0;
  for (const [pi, pj] of [...pairs, [previous.length, current.length] as [number, number]]) {
    while (i < pi) removed(i++);
    while (j < pj) added(j++);
    if (pi === previous.length && pj === current.length) break;
    const before = previous[pi]!;
    const after = current[pj]!;
    const path = at();
    if (index.changed.has(beforeAt(pi))) {
      changed.push(path);
      record(parts, after, path, beforeAt(pi), afterAt(pj), "mod");
      if (
        (before.kind === "text" && after.kind === "text") ||
        (before.kind === "annotation" && after.kind === "annotation")
      ) {
        const markdown = wordDiffMarkdown(before.markdown, (after as typeof before).markdown);
        out.push({ ...after, markdown, change: "mod" } as Marked);
      } else {
        out.push({ ...after, change: "mod" } as Marked);
      }
    } else if (
      (after.kind === "group" || after.kind === "incorporation") &&
      before.kind === after.kind &&
      (inside(index.before, beforeAt(pi)) || inside(index.after, afterAt(pj)))
    ) {
      // a container whose own fields agree: walk in and mark there
      parts.push({ shown: path, before: beforeAt(pi), after: afterAt(pj), change: null });
      const content = diffBlocks(
        (before as GroupBlock | IncorporationBlock).content,
        (after as GroupBlock | IncorporationBlock).content,
        beforeAt(pi),
        afterAt(pj),
        path,
        index,
        changed,
        parts,
      );
      out.push({ ...after, content } as Block);
    } else {
      // unchanged: the tracker named nothing here
      record(parts, after, path, beforeAt(pi), afterAt(pj), null);
      out.push(after);
    }
    i = pi + 1;
    j = pj + 1;
  }
  return out;
}

/** A whole component added or removed: the mark on it, and its texts'
 * words marked through, so a removed paragraph reads struck. */
function markAll(block: Block, change: "add" | "del"): Block {
  const op = change === "add" ? "ins" : "del";
  const mark = (markdown: string) => emit(tokenize(markdown).map((token) => ({ op, token }) as Op));
  switch (block.kind) {
    case "text":
    case "annotation":
      return { ...block, markdown: mark(block.markdown), change } as Marked;
    case "group":
    case "incorporation":
      return { ...block, content: block.content.map((child) => markAll(child, change)), change } as Marked;
    default:
      return { ...block, change } as Marked;
  }
}

// -- places across revisions -------------------------------------------

/** The block part a rendered part belongs to: a ref, diagram or
 * citation inside a block reads as the block. */
export function blockPartOf(part: string): string {
  return /^(\/content\/\d+)+/.exec(part)?.[0] ?? part;
}

/** The shown part standing at `path` in one revision, or null. A part
 * the diff does not list reads as its nearest listed ancestor (not
 * exact). */
export function shownAt(
  diff: BoardDiff,
  side: "before" | "after",
  path: string,
): { shown: string; exact: boolean } | null {
  for (let at: string | null = blockPartOf(path); at !== null; at = parentOf(at)) {
    const found = diff.parts.find((part) => part[side] === at);
    if (found !== undefined) return { shown: found.shown, exact: at === path };
  }
  return null;
}

/** Where a shown part stands in one revision. When it is not there (an
 * added part before, a removed one after), its nearest surviving
 * neighbour: the closest sibling that is there, the one above first at
 * each distance, else its container's. */
export function placeIn(
  diff: BoardDiff,
  shown: string,
  side: "before" | "after",
): { path: string; exact: boolean } | null {
  const of = (path: string) => diff.parts.find((part) => part.shown === path);
  const own = of(blockPartOf(shown));
  if (own === undefined) return null;
  const direct = own[side];
  if (direct !== null) return { path: direct, exact: true };
  for (let at: string | null = own.shown; at !== null; at = parentOf(at)) {
    const parent = parentOf(at);
    const siblings = diff.parts.filter((part) => parentOf(part.shown) === parent);
    const position = siblings.findIndex((part) => part.shown === at);
    for (let distance = 1; distance < siblings.length; distance++) {
      for (const k of [position - distance, position + distance]) {
        const there = siblings[k]?.[side];
        if (there !== null && there !== undefined) return { path: there, exact: false };
      }
    }
    if (parent !== null) {
      const container = of(parent)?.[side];
      if (container !== null && container !== undefined) return { path: container, exact: false };
    }
  }
  return null;
}

/** Whether the shown revision changed this part: it carries a mark, or
 * something inside it does. */
export function touches(diff: BoardDiff, shown: string): boolean {
  return diff.changed.some((path) => path === shown || path.startsWith(`${shown}/`));
}

function parentOf(path: string): string | null {
  const cut = path.lastIndexOf("/content/");
  return cut <= 0 ? null : path.slice(0, cut);
}

function lcs<T>(a: readonly T[], b: readonly T[]): [number, number][] {
  const n = a.length;
  const m = b.length;
  const table: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i]![j] = a[i] === b[j] ? table[i + 1]![j + 1]! + 1 : Math.max(table[i + 1]![j]!, table[i]![j + 1]!);
    }
  }
  const pairs: [number, number][] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      pairs.push([i, j]);
      i++;
      j++;
    } else if (table[i + 1]![j]! >= table[i]![j + 1]!) i++;
    else j++;
  }
  return pairs;
}

// -- the word diff -----------------------------------------------------

type Token = { text: string; kind: "word" | "space" | "newline" | "atomic" | "fence" };
type Op = { op: "same" | "ins" | "del"; token: Token };

/** Markdown as tokens: structure lines whole (fence lines, and every
 * line of a Mermaid diagram or a display-math block, whose source must
 * parse), math and code spans whole, words, runs of spaces, newlines. */
export function tokenize(markdown: string): Token[] {
  const tokens: Token[] = [];
  const lines = markdown.split("\n");
  let fence: string | null = null;
  let mathBlock = false;
  lines.forEach((line, index) => {
    const fenceLine = /^\s*(```|~~~)\s*(\S*)/.exec(line);
    if (fenceLine !== null) {
      fence = fence === null ? (fenceLine[2] ?? "") : null;
      tokens.push({ text: line, kind: "fence" });
    } else if (fence === "mermaid" || (fence === null && (mathBlock || line.trim() === "$$"))) {
      if (fence === null && line.trim() === "$$") mathBlock = !mathBlock;
      tokens.push({ text: line, kind: "fence" });
    } else {
      const pattern = /\$\$[^$]*\$\$|\$[^$\n]+?\$|`[^`\n]*`|[ \t]+|[^\s]+/g;
      for (const match of line.matchAll(pattern)) {
        const text = match[0];
        const kind = /^[ \t]+$/.test(text) ? "space" : /^(\$|`)/.test(text) && text.length > 1 ? "atomic" : "word";
        tokens.push({ text, kind });
      }
    }
    if (index < lines.length - 1) tokens.push({ text: "\n", kind: "newline" });
  });
  return tokens;
}

/** The size past which a changed middle is shown as whole removed and
 * added text rather than aligned word by word. */
const WORD_DIFF_LIMIT = 4000;

/** The current markdown with the change from `before` marked in it. */
export function wordDiffMarkdown(before: string, after: string): string {
  const a = tokenize(before);
  const b = tokenize(after);
  // the common prefix and suffix need no alignment
  let start = 0;
  while (start < a.length && start < b.length && a[start]!.text === b[start]!.text) start++;
  let endA = a.length;
  let endB = b.length;
  while (endA > start && endB > start && a[endA - 1]!.text === b[endB - 1]!.text) {
    endA--;
    endB--;
  }
  const ops: Op[] = a.slice(0, start).map((token) => ({ op: "same", token }));
  const midA = a.slice(start, endA);
  const midB = b.slice(start, endB);
  if (midA.length > WORD_DIFF_LIMIT || midB.length > WORD_DIFF_LIMIT) {
    ops.push(...midA.map((token) => ({ op: "del" as const, token })));
    ops.push(...midB.map((token) => ({ op: "ins" as const, token })));
  } else {
    // spaces never anchor the alignment (a shared space between two
    // changed words would split one change into interleaved pieces)
    const pairs = lcs(
      midA.map((token, index) => (token.kind === "space" ? `\u0000a${index}` : token.text)),
      midB.map((token, index) => (token.kind === "space" ? `\u0000b${index}` : token.text)),
    );
    let i = 0;
    let j = 0;
    for (const [pi, pj] of [...pairs, [midA.length, midB.length] as [number, number]]) {
      while (i < pi) ops.push({ op: "del", token: midA[i++]! });
      while (j < pj) ops.push({ op: "ins", token: midB[j++]! });
      if (pi < midA.length) ops.push({ op: "same", token: midB[pj]! });
      i = pi + 1;
      j = pj + 1;
    }
  }
  ops.push(...a.slice(endA).map((token) => ({ op: "same" as const, token })));
  return emit(ops);
}

const MARKER = /^(#{1,6}|[-*+]|\d+[.)]|>)$/;

/** The ops as markdown: words wrapped in their mark, whitespace and
 * line markers left bare so the structure stays the current one. */
function emit(ops: readonly Op[]): string {
  let out = "";
  let lineStart = true;
  let previous: Op["op"] | null = null;
  for (const { op, token } of ops) {
    const [open, close] = op === "ins" ? [INS_OPEN, INS_CLOSE] : [DEL_OPEN, DEL_CLOSE];
    if (token.kind === "newline") {
      out += "\n";
      lineStart = true;
      continue;
    }
    if (token.kind === "space") {
      // a removed space is written only inside a removed run; the
      // current text's own spaces are always written
      if (op !== "del" || previous === "del") out += token.text;
      continue;
    }
    previous = op;
    if (token.kind === "fence") {
      // a structure line: a removed one would open a fence that never
      // closes (or break a diagram's or a formula's source), so only
      // the current ones are written, unmarked
      if (op !== "del") out += token.text;
      lineStart = false;
      continue;
    }
    if (op === "same" || (lineStart && MARKER.test(token.text))) {
      out += token.text;
    } else {
      out += open + token.text + close;
    }
    lineStart = false;
  }
  // a run of changed words reads as one mark: the spaces between two
  // marks of the same kind join them (no box around every word)
  return out
    .replace(/\uE001([ \t]+)\uE000/g, "$1")
    .replace(/\uE003([ \t]+)\uE002/g, "$1");
}

/** After rendering: the sentinels become ins and del elements, across
 * text nodes (a highlighted code line splits a word into spans). */
export function paintDiffMarks(root: HTMLElement): void {
  const doc = root.ownerDocument;
  const walker = doc.createTreeWalker(root, 4 /* NodeFilter.SHOW_TEXT */);
  const nodes: Text[] = [];
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) nodes.push(node as Text);
  let state: "none" | "ins" | "del" = "none";
  for (const node of nodes) {
    const text = node.data;
    if (state === "none" && !/[\uE000-\uE003]/.test(text)) continue;
    const fragment = doc.createDocumentFragment();
    let run = "";
    const flush = () => {
      if (run.length === 0) return;
      if (state === "none") fragment.append(doc.createTextNode(run));
      else {
        const mark = doc.createElement(state);
        mark.className = state === "ins" ? "port-diff-ins" : "port-diff-del";
        mark.textContent = run;
        fragment.append(mark);
      }
      run = "";
    };
    for (const ch of text) {
      if (ch === INS_OPEN || ch === DEL_OPEN) {
        flush();
        state = ch === INS_OPEN ? "ins" : "del";
      } else if (ch === INS_CLOSE || ch === DEL_CLOSE) {
        flush();
        state = "none";
      } else run += ch;
    }
    flush();
    node.replaceWith(fragment);
  }
}
