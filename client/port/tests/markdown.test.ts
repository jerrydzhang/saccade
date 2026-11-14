// The parse step's math (t-27): what the board's authored LaTeX becomes
// before MathJax sees it. The stored text is never rewritten; the
// expression's TeX and its authored bytes come out exact.
import assert from "node:assert/strict";
import { test } from "vitest";
import type { Root, RootContent } from "mdast";
import { parseMarkdown } from "../src/lib/markdown.ts";

const math = (markdown: string) =>
  parseMarkdown(markdown, 0, 1).mathNodes.map(({ type, value, authored }) => ({ type, value, authored }));

/** Every text node's value, in document order (math excluded). */
function texts(root: Root): string[] {
  const out: string[] = [];
  const walk = (node: Root | RootContent): void => {
    if (node.type === "text") out.push(node.value);
    if ("children" in node) for (const child of node.children) walk(child);
  };
  walk(root);
  return out;
}

test("inline math keeps underscores and backslashes: blackboard and calligraphic symbols", () => {
  assert.deepEqual(math("Inline $(\\mathbb{Z}_2)^p$ and $\\mathcal{F}_S$."), [
    { type: "inlineMath", value: "(\\mathbb{Z}_2)^p", authored: "$(\\mathbb{Z}_2)^p$" },
    { type: "inlineMath", value: "\\mathcal{F}_S", authored: "$\\mathcal{F}_S$" },
  ]);
});

test("display math with its $$ on their own lines: the aligned body, line breaks and all", () => {
  const body = "\\begin{aligned}\n-X_S^\\top(y-X_S\\beta_S)+\\sigma\\odot w_S &= 0\\\\\n\\beta_S &= G^{-1}\\big(X_S^\\top y-\\sigma\\odot w_S\\big)\n\\end{aligned}";
  const source = `$$\n${body}\n$$`;
  assert.deepEqual(math(source), [{ type: "math", value: body, authored: source }]);
});

test("display math whose $$ share the body's first and last lines is one display expression", () => {
  const body = "\\begin{aligned}\n0 &< \\sigma\\odot\\beta_S\\\\\n0 &< \\sigma\\odot G^{-1}(X_S^\\top y-\\sigma\\odot w_S)\n\\end{aligned}";
  const source = `$$${body}$$`;
  assert.deepEqual(math(source), [{ type: "math", value: body, authored: source }]);
});

test("a same-line display block between prose lines keeps the prose around it as text", () => {
  const body = "\\begin{aligned}\na &= 0\\\\\nb &= 1\n\\end{aligned}";
  const markdown = `On-support:\n$$${body}$$\nSubstitute $x$ after.`;
  const parsed = parseMarkdown(markdown, 0, 1);
  assert.deepEqual(
    parsed.mathNodes.map(({ type, value, authored }) => ({ type, value, authored })),
    [
      { type: "math", value: body, authored: `$$${body}$$` },
      { type: "inlineMath", value: "x", authored: "$x$" },
    ],
  );
  const shown = texts(parsed.root).join("|");
  assert.match(shown, /On-support:/);
  assert.match(shown, /Substitute/);
  assert.match(shown, /after\./);
});

test("a same-line display block does not swallow the paragraphs after it", () => {
  const markdown = "$$\\begin{aligned}\na &= 0\n\\end{aligned}$$\n\nThe next paragraph.\n\n$$\nx\n$$";
  const parsed = parseMarkdown(markdown, 0, 1);
  assert.deepEqual(
    parsed.mathNodes.map(({ type, value }) => ({ type, value })),
    [
      { type: "math", value: "\\begin{aligned}\na &= 0\n\\end{aligned}" },
      { type: "math", value: "x" },
    ],
  );
  assert.deepEqual(texts(parsed.root), ["The next paragraph."]);
});

test("a single-line $$...$$ paragraph stays display math", () => {
  assert.deepEqual(math("$$x_1 + \\mathbb{R}$$"), [
    { type: "math", value: "x_1 + \\mathbb{R}", authored: "$$x_1 + \\mathbb{R}$$" },
  ]);
});
