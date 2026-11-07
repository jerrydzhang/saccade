// Plain-text projection and search over a decoded board. An authored math
// expression is one logical search unit carrying its TeX value; equal
// formulas at distinct source locations stay distinct occurrences.

import type { SourceRange } from "./errors.ts";
import type { Board } from "./model.ts";
import { parseMarkdown, promotedDisplayMath, authoredSlice } from "./markdown.ts";

export type ProjectionUnit = {
  id: string;
  kind: "prose" | "code" | "math" | "mermaid";
  /** Searchable value: rendered text for prose, source bytes for code, TeX for math. */
  text: string;
  /** Authored bytes for math (Source copy mode); equals text elsewhere. */
  source: string;
  pos: SourceRange;
  display?: boolean;
  blockKind: string;
};

function zeroRange(): SourceRange {
  const start = { line: 0, column: 0, offset: 0 };
  return { start, end: { ...start } };
}

export function projectBoard(board: Board): ProjectionUnit[] {
  const units: ProjectionUnit[] = [];
  let counter = 0;
  const nextId = () => `u${counter++}`;

  const projectMarkdown = (markdown: string, blockKind: string, offset: number, line: number): void => {
    const parsed = parseMarkdown(markdown, offset, line);
    void parsed;
    let prose = "";
    let prosePos: SourceRange | null = null;

    const walk = (node: import("mdast").RootContent | void): void => {
      if (!node) return;
      const pos = node.position;
      const range = pos && pos.start.offset != null && pos.end.offset != null
        ? {
            start: { line: line + (pos.start.line - 1), column: pos.start.column, offset: offset + pos.start.offset },
            end: { line: line + (pos.end.line - 1), column: pos.end.column, offset: offset + pos.end.offset },
          }
        : null;
      const onlyChild = node.type === "paragraph" ? node.children[0] : undefined;
      if (node.type === "paragraph" && onlyChild && onlyChild.type === "inlineMath" && promotedDisplayMath(markdown, node, onlyChild)) {
        flushProse();
        units.push({
          id: nextId(),
          kind: "math",
          text: onlyChild.value,
          source: authoredSlice(markdown, onlyChild.position),
          pos: range ?? zeroRange(),
          display: true,
          blockKind,
        });
        return;
      }
      if (node.type === "inlineMath") {
        flushProse();
        units.push({
          id: nextId(),
          kind: "math",
          text: node.value,
          source: authoredSlice(markdown, node.position),
          pos: range ?? zeroRange(),
          display: false,
          blockKind,
        });
        return;
      }
      if (node.type === "math") {
        flushProse();
        units.push({
          id: nextId(),
          kind: "math",
          text: node.value,
          source: authoredSlice(markdown, node.position),
          pos: range ?? zeroRange(),
          display: true,
          blockKind,
        });
        return;
      }
      if (node.type === "inlineCode") {
        flushProse();
        units.push({
          id: nextId(),
          kind: "code",
          text: node.value,
          source: node.value,
          pos: range ?? zeroRange(),
          blockKind,
        });
        return;
      }
      if (node.type === "code") {
        flushProse();
        units.push({
          id: nextId(),
          kind: node.lang === "mermaid" ? "mermaid" : "code",
          text: node.value,
          source: node.value,
          pos: range ?? zeroRange(),
          blockKind,
        });
        return;
      }
      if (node.type === "text") {
        if (prose.length > 0 && prosePos) {
          // prose accumulates contiguous text; keep the widest range
          prose += " ";
        } else {
          prosePos = range;
        }
        prose += node.value;
        return;
      }
      if ("children" in node && Array.isArray(node.children)) {
        for (const child of node.children as import("mdast").RootContent[]) {
          walk(child);
        }
        if (node.type === "paragraph" || node.type === "heading" || node.type === "listItem" || node.type === "blockquote" || node.type === "tableCell") {
          flushProse();
        }
        return;
      }
    };

    const flushProse = (): void => {
      const text = prose.trim();
      if (text.length > 0) {
        units.push({
          id: nextId(),
          kind: "prose",
          text,
          source: text,
          pos: prosePos ?? zeroRange(),
          blockKind,
        });
      }
      prose = "";
      prosePos = null;
    };

    for (const child of parsed.root.children) {
      walk(child);
    }
    flushProse();
  };

  const projectBlocks = (blocks: Board["content"]): void => {
    for (const block of blocks) {
      if (block.kind === "text" || block.kind === "annotation") {
        projectMarkdown(block.markdown, block.kind, block.pos.start.offset, block.pos.start.line);
      } else if (block.kind === "group" || block.kind === "incorporation") {
        if (block.kind === "group" && block.title) {
          units.push({
            id: nextId(),
            kind: "prose",
            text: block.title,
            source: block.title,
            pos: block.pos,
            blockKind: "group-title",
          });
        }
        projectBlocks(block.content);
      } else if (block.kind === "figure") {
        const caption = block.caption;
        units.push({
          id: nextId(),
          kind: "prose",
          text: caption.length > 0 ? caption : block.input,
          source: caption.length > 0 ? caption : block.input,
          pos: block.pos,
          blockKind: "figure",
        });
      } else if (block.kind === "unknown") {
        units.push({
          id: nextId(),
          kind: "code",
          text: block.source,
          source: block.source,
          pos: block.pos,
          blockKind: "unknown",
        });
      }
    }
  };

  projectBlocks(board.content);
  return units;
}

export type SearchOccurrence = {
  unit: ProjectionUnit;
  start: number;
  end: number;
};

export function searchUnits(units: ProjectionUnit[], query: string): SearchOccurrence[] {
  const trimmed = query.trim();
  if (trimmed.length === 0) return [];
  // A query written with math delimiters still finds the bare TeX value.
  const needle = normalizeQuery(trimmed).toLowerCase();
  const out: SearchOccurrence[] = [];
  for (const unit of units) {
    // One authored expression is one logical search unit: matches in the
    // logical text count once; the delimited authored form is only consulted
    // when the plain form did not match at all.
    const textMatches = collectMatches(normalizeQuery(unit.text).toLowerCase(), needle);
    if (textMatches.length > 0) {
      for (const [start, end] of textMatches) out.push({ unit, start, end });
      continue;
    }
    const sourceMatches = collectMatches(normalizeQuery(unit.source).toLowerCase(), needle);
    for (const [start, end] of sourceMatches) out.push({ unit, start, end });
  }
  return out;
}

function collectMatches(haystack: string, needle: string): [number, number][] {
  const out: [number, number][] = [];
  let from = 0;
  while (from <= haystack.length) {
    const idx = haystack.indexOf(needle, from);
    if (idx < 0) break;
    out.push([idx, idx + needle.length]);
    from = idx + Math.max(needle.length, 1);
  }
  return out;
}

/** Strip math delimiters so `$x$` queries match the TeX value `x`. */
export function normalizeQuery(text: string): string {
  return text.replace(/^\$\$?/, "").replace(/\$\$?$/, "");
}
