// The refs in a block's prose (m-547 (2), the prototype's linkify,
// inst16.html:197): every t-N, t-N@rK, t-N@rK:/path, i-N and m-N in
// shown text - never inside code, a link, a deletion, math or mermaid -
// is a ref, its own select target <block>/ref/<n> in document order.
// The renderer and the model targets both walk the same parse with the
// same rules (refsIn), so the n-th ref is the same ref on both sides.

import type { Root, RootContent } from "mdast";
import { parseMarkdown } from "../markdown.ts";

/** The prototype's pattern (inst16.html:197), one ref per match. */
export const REF_PATTERN = /\b(t-\d+(?:@r\d+)?(?::\/[\w/]+)?|i-\d+|m-\d+)\b/g;

/** A ref's part path: its block's part plus its ordinal there. */
export function refPart(blockPart: string, ordinal: number): string {
  return `${blockPart}/ref/${ordinal}`;
}

/** Split one text run into plain pieces and refs, in order. */
export function splitRefs(text: string): ({ kind: "text"; text: string } | { kind: "ref"; ref: string })[] {
  const out: ({ kind: "text"; text: string } | { kind: "ref"; ref: string })[] = [];
  let last = 0;
  for (const match of text.matchAll(REF_PATTERN)) {
    const index = match.index ?? 0;
    if (index > last) out.push({ kind: "text", text: text.slice(last, index) });
    out.push({ kind: "ref", ref: match[1]! });
    last = index + match[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

/** The node kinds whose text never carries refs. */
export const REFLESS = new Set(["link", "linkReference", "delete", "inlineCode", "code", "inlineMath", "math", "html"]);

/** Every ref in a block's markdown, in document order. */
export function refsIn(markdown: string): string[] {
  const { root } = parseMarkdown(markdown, 0, 1);
  const out: string[] = [];
  const visit = (node: RootContent | Root): void => {
    if (REFLESS.has(node.type)) return;
    if (node.type === "text") {
      for (const piece of splitRefs(node.value)) if (piece.kind === "ref") out.push(piece.ref);
      return;
    }
    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children as RootContent[]) visit(child);
    }
  };
  visit(root);
  return out;
}

/** A ref's openable address: a task ref's path (t-N or t-N@rK) and the
 * part it names, or null for refs the port cannot open (i-N, m-N). */
export function refAddress(ref: string): { path: string; part: string | null } | null {
  const match = /^(t-\d+(?:@r\d+)?)(?::(\/[\w/]+))?$/.exec(ref);
  if (match === null) return null;
  return { path: match[1]!, part: match[2] ?? null };
}
