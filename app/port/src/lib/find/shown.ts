// The shown prose of a markdown block, derived from the SAME parse the
// renderer walks (lib/markdown.ts): the text a reader sees, without
// markdown syntax (`**alpha**` reads `alpha`), so an occurrence counted
// here is one the reader can see and the mark painter (marks.ts) can
// find again in the rendered text. Math and mermaid are left out on
// both sides: their rendered form is not text a reader searches; the
// session's authored-source fallback still finds a TeX query.

import type { Root, RootContent } from "mdast";
import { parseMarkdown } from "../markdown.ts";

export function shownText(markdown: string): string {
  const { root } = parseMarkdown(markdown, 0, 1);
  let out = "";
  const visit = (node: RootContent | Root): void => {
    switch (node.type) {
      case "text":
      case "inlineCode":
        out += node.value;
        return;
      case "code":
        if (node.lang !== "mermaid") out += node.value;
        return;
      case "inlineMath":
      case "math":
      case "html":
      case "image":
        return;
      default:
        if ("children" in node && Array.isArray(node.children)) {
          for (const child of node.children as RootContent[]) visit(child);
        }
    }
  };
  visit(root);
  return out;
}
