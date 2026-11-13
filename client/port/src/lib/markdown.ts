// One source-positioned Markdown pipeline. Every block's markdown goes
// through the same parse (remark-parse + gfm + math); mermaid is a fence
// language of that same tree; ordinary fences never execute; raw HTML in
// authored content is surfaced as an explicit diagnostic, never rendered.

import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import { unified } from "unified";
import type { Root, RootContent } from "mdast";
import type { Diagnostic, SourceRange } from "./board/errors.ts";

export type MathNode = {
  type: "inlineMath" | "math";
  value: string;
  pos: SourceRange;
  /** The exact authored bytes of the expression, delimiters included:
   * Source copy and the renderer use this, never a reconstruction. */
  authored: string;
};

/**
 * The exact authored bytes a node occupies in its own block source.
 * Reconstructing delimiters from the value would violate authored
 * spelling, spacing, Unicode and newlines.
 */
export function authoredSlice(
  markdown: string,
  position:
    { start?: { offset?: number | null }; end?: { offset?: number | null } } | null | undefined,
): string {
  const start = position?.start?.offset;
  const end = position?.end?.offset;
  if (start === null || start === undefined || end === null || end === undefined) return "";
  if (end <= start) return "";
  return markdown.slice(start, end);
}

export type ParsedMarkdown = {
  root: Root;
  mathNodes: MathNode[];
  mermaidFences: { code: string; pos: SourceRange }[];
  /** Raw HTML found in authored content: reported, never executed. */
  diagnostics: Diagnostic[];
};

const processor = unified().use(remarkParse).use(remarkGfm).use(remarkMath);

/**
 * A lone single-line $$...$$ in its own paragraph is authored display math:
 * the authored bytes decide, not line position. Remark parses it as math
 * text, so this promotion keeps one AST and one classification rule shared
 * by the parser and the projection.
 */
export function promotedDisplayMath(
  markdown: string,
  paragraph: { children?: unknown[]; position?: unknown },
  only: { type?: string; position?: unknown },
): boolean {
  const children = paragraph.children;
  if (!children || children.length !== 1 || only.type !== "inlineMath") return false;
  const position = paragraph.position as { start?: { offset?: number | null } } | undefined;
  const endOffset = (only.position as { end?: { offset?: number | null } } | undefined)?.end
    ?.offset;
  const start = position?.start?.offset;
  if (start === null || start === undefined || endOffset === null || endOffset === undefined)
    return false;
  return markdown.slice(start, endOffset).startsWith("$$");
}

// One parse per markdown string (m-547 (5)): the renderer, the model
// targets (diagrams, refs) and find (shown text) all read the same
// block, and each parsed it again - on t-36 every find keystroke
// re-parsed all 125 blocks. The tree depends on the text alone
// (positions are relative to it; the block offsets apply on top), and
// no consumer mutates it. Bounded: the oldest entry goes first.
const ROOT_CACHE_LIMIT = 4096;
const rootCache = new Map<string, Root>();
let parses = 0;

function parsedRoot(markdown: string): Root {
  const cached = rootCache.get(markdown);
  if (cached !== undefined) return cached;
  parses += 1;
  const root = processor.parse(markdown) as Root;
  if (rootCache.size >= ROOT_CACHE_LIMIT) {
    const oldest = rootCache.keys().next().value;
    if (oldest !== undefined) rootCache.delete(oldest);
  }
  rootCache.set(markdown, root);
  return root;
}

/** How many real parses ran (the parse cache's mechanism test). */
export function markdownParseCount(): number {
  return parses;
}

export function parseMarkdown(
  markdown: string,
  blockOffset: number,
  blockLine: number,
): ParsedMarkdown {
  const root = parsedRoot(markdown);
  const mathNodes: MathNode[] = [];
  const mermaidFences: { code: string; pos: SourceRange }[] = [];
  const diagnostics: Diagnostic[] = [];

  const visit = (node: RootContent | Root["children"][number] | void): void => {
    if (!node) return;
    const pos = docRange(node.position, blockOffset, blockLine);
    if (node.type === "inlineMath" || node.type === "math") {
      mathNodes.push({
        type: node.type,
        value: node.value,
        pos: pos ?? zeroRange(),
        authored: authoredSlice(markdown, node.position),
      });
      return;
    }
    if (node.type === "paragraph") {
      const onlyChild = node.children[0];
      if (
        onlyChild &&
        onlyChild.type === "inlineMath" &&
        promotedDisplayMath(markdown, node, onlyChild)
      ) {
        const mathPos = docRange(onlyChild.position, blockOffset, blockLine);
        mathNodes.push({
          type: "math",
          value: onlyChild.value,
          pos: mathPos ?? zeroRange(),
          authored: authoredSlice(markdown, onlyChild.position),
        });
        return;
      }
      for (const child of node.children as RootContent[]) {
        visit(child);
      }
      return;
    }
    if (node.type === "code") {
      if (node.lang === "mermaid") {
        mermaidFences.push({ code: node.value, pos: pos ?? zeroRange() });
      }
      // Ordinary fences become inert code, whatever their language.
      return;
    }
    if (node.type === "html") {
      diagnostics.push({
        code: "unknown-key",
        message: "raw HTML in authored content is not rendered in the read-only viewer",
        line: pos ? pos.start.line : undefined,
        column: pos ? pos.start.column : undefined,
      });
      return;
    }
    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children as RootContent[]) {
        visit(child);
      }
    }
  };

  for (const child of root.children) {
    visit(child);
  }

  return { root, mathNodes, mermaidFences, diagnostics };
}

export function docRange(
  position:
    | {
        start?: { line: number; column: number; offset?: number | null };
        end?: { line: number; column: number; offset?: number | null };
      }
    | null
    | undefined,
  blockOffset: number,
  blockLine: number,
): SourceRange | null {
  const start = position?.start;
  const end = position?.end;
  if (
    !start ||
    !end ||
    start.offset === null ||
    start.offset === undefined ||
    end.offset === null ||
    end.offset === undefined
  ) {
    return null;
  }
  const lineDelta = start.line - 1;
  return {
    start: {
      line: blockLine + lineDelta,
      column: start.column,
      offset: blockOffset + start.offset,
    },
    end: {
      line: blockLine + (end.line - 1),
      column: end.column,
      offset: blockOffset + end.offset,
    },
  };
}

function zeroRange(): SourceRange {
  const start = { line: 0, column: 0, offset: 0 };
  return { start, end: { ...start } };
}
