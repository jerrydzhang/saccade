// History mode's diff (t-49; t-14 "The viewer", History), its pure
// half: a revision shown against the one before it. Components are
// aligned as `sac board diff` aligns them (tracker/src/boards/
// changes.rs: by a key - a group's title, a text's first heading, a
// figure's input, an incorporation's source task - over a longest
// common subsequence). The result is ONE board to render: the shown
// revision's components, each added or changed one marked, and every
// removed one placed back where it stood, marked removed, so the
// reader sees what went without leaving the page.
//
// Inside a changed text the diff is by word, and keeps Markdown's
// structure: the marks ride private-use sentinels through the normal
// renderer (paintDiffMarks turns them into ins/del after rendering), a
// line's list, heading or quote marker always stays in front of them
// (a changed marker moves in front of the marks), math and code spans
// change whole, fence lines are never marked, and a Mermaid diagram
// shows its new source unmarked (marks would break the diagram).

import type { Block, Board, GroupBlock, IncorporationBlock } from "./model.ts";

export const INS_OPEN = "\uE000";
export const INS_CLOSE = "\uE001";
export const DEL_OPEN = "\uE002";
export const DEL_CLOSE = "\uE003";

export type ChangeMark = "add" | "mod" | "del";
type Marked = Block & { change?: ChangeMark };

export type BoardDiff = {
  board: Board;
  /** The changed parts in reading order, as rendered part paths: the
   * places n and N jump between. */
  changed: string[];
};

/** The shown revision against its predecessor (null: the first
 * revision, everything added). */
export function diffBoards(previous: Board | null, current: Board): BoardDiff {
  const changed: string[] = [];
  const content = diffBlocks(previous?.content ?? [], current.content, "", changed);
  return { board: { ...current, content }, changed };
}

function blockKey(block: Block): string {
  switch (block.kind) {
    case "group":
      return `group:${block.title ?? ""}`;
    case "text":
      return `text:${headingOf(block.markdown) ?? ""}`;
    case "figure":
      return `figure:${block.input}`;
    case "incorporation":
      return `incorporation:${block.source.task}`;
    case "annotation":
      return "annotation";
    case "unknown":
      return `unknown:${block.name}`;
  }
}

function headingOf(markdown: string): string | null {
  const first = markdown.split("\n").find((line) => line.trim().length > 0)?.trim() ?? "";
  return first.startsWith("#") ? first : null;
}

/** Equality of authored content, ignoring source positions. */
function same(a: Block, b: Block): boolean {
  return JSON.stringify(strip(a)) === JSON.stringify(strip(b));
}

function strip(block: Block): unknown {
  const { pos: _pos, ...rest } = block as Block & { pos?: unknown };
  if (block.kind === "group" || block.kind === "incorporation") {
    return { ...rest, content: block.content.map(strip) };
  }
  return rest;
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

function diffBlocks(previous: readonly Block[], current: readonly Block[], prefix: string, changed: string[]): Block[] {
  const pairs = lcs(previous.map(blockKey), current.map(blockKey));
  const out: Block[] = [];
  const at = () => `${prefix}/content/${out.length}`;
  const removed = (block: Block) => {
    changed.push(at());
    out.push(markAll(block, "del"));
  };
  const added = (block: Block) => {
    changed.push(at());
    out.push(markAll(block, "add"));
  };
  let i = 0;
  let j = 0;
  for (const [pi, pj] of [...pairs, [previous.length, current.length] as [number, number]]) {
    while (i < pi) removed(previous[i++]!);
    while (j < pj) added(current[j++]!);
    if (pi === previous.length && pj === current.length) break;
    const before = previous[pi]!;
    const after = current[pj]!;
    if (same(before, after)) {
      out.push(after);
    } else if (
      (before.kind === "group" && after.kind === "group" && before.title === after.title) ||
      (before.kind === "incorporation" &&
        after.kind === "incorporation" &&
        JSON.stringify(before.source) === JSON.stringify(after.source))
    ) {
      // a container whose own fields agree: walk in and mark there
      const path = at();
      const inner = diffBlocks(
        (before as GroupBlock | IncorporationBlock).content,
        (after as GroupBlock | IncorporationBlock).content,
        path,
        changed,
      );
      out.push({ ...after, content: inner } as Block);
    } else if (
      (before.kind === "text" && after.kind === "text") ||
      (before.kind === "annotation" && after.kind === "annotation")
    ) {
      changed.push(at());
      const markdown = wordDiffMarkdown(before.markdown, (after as typeof before).markdown);
      out.push({ ...after, markdown, change: "mod" } as Marked);
    } else {
      changed.push(at());
      out.push({ ...after, change: "mod" } as Marked);
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
    const pairs = lcs(
      midA.map((token) => token.text),
      midB.map((token) => token.text),
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
  for (const { op, token } of ops) {
    const [open, close] = op === "ins" ? [INS_OPEN, INS_CLOSE] : [DEL_OPEN, DEL_CLOSE];
    if (token.kind === "newline") {
      out += "\n";
      lineStart = true;
      continue;
    }
    if (token.kind === "space") {
      out += token.text;
      continue;
    }
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
  return out;
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
