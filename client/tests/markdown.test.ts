import test from "node:test";
import assert from "node:assert/strict";
import { parseMarkdown } from "../src/lib/markdown.ts";

test("math spans come from the same parse and carry source positions", () => {
  const markdown = "Prose with $a<b$ and display:\n\n$$c = d$$\n";
  const parsed = parseMarkdown(markdown, 100, 10);
  assert.equal(parsed.mathNodes.length, 2);
  const inline = parsed.mathNodes[0];
  const display = parsed.mathNodes[1];
  assert.equal(inline?.type, "inlineMath");
  assert.equal(inline?.value, "a<b");
  assert.equal(display?.type, "math");
  assert.equal(display?.value, "c = d");
  // positions map onto the whole document: block offset 100, block line 10
  assert.ok(inline && inline.pos.start.offset >= 100);
  assert.ok(display && display.pos.start.line >= 10);
});

test("dollar strings inside ordinary code fences never become math", () => {
  const markdown = "```\nnot math $x=y$ here\n```\n\nReal: $x=y$\n";
  const parsed = parseMarkdown(markdown, 0, 1);
  assert.equal(parsed.mathNodes.length, 1);
  assert.equal(parsed.mathNodes[0]?.value, "x=y");
});

test("mermaid fences are found by language and positioned", () => {
  const markdown = "```mermaid\nflowchart LR\n  a --> b\n```\n\n```python\nx = 1\n```\n";
  const parsed = parseMarkdown(markdown, 0, 1);
  assert.equal(parsed.mermaidFences.length, 1);
  assert.match(parsed.mermaidFences[0]?.code ?? "", /flowchart LR/);
  // the python fence is an ordinary inert code fence
  assert.equal(parsed.mathNodes.length, 0);
});

test("raw HTML is reported as a diagnostic, never rendered", () => {
  const markdown = "Before <script>alert(1)</script> after\n";
  const parsed = parseMarkdown(markdown, 0, 1);
  // open and close tags are separate html nodes: two findings, zero execution
  assert.equal(parsed.diagnostics.length, 2);
  assert.match(parsed.diagnostics[0]?.message ?? "", /raw HTML/);
});

test("positions map through block offset and line onto authored bytes", () => {
  const markdown = "first $x$ line\nsecond $y$ line\n";
  const blockOffset = 50;
  const blockLine = 7;
  const parsed = parseMarkdown(markdown, blockOffset, blockLine);
  const second = parsed.mathNodes[1];
  assert.ok(second);
  // the authored document byte at the mapped offset is the dollar of $y$
  const doc = "x".repeat(50) + markdown;
  assert.equal(doc.charAt(second.pos.start.offset), "$");
  assert.equal(second.pos.start.line, blockLine + 1);
});

test("authored slices retain exact spelling, spacing and newlines", () => {
  const markdown = "Mixed: $$ x $$ paragraph\n\n$$\n\\frac{a}{b}\n$$\n";
  const parsed = parseMarkdown(markdown, 0, 1);
  const inline = parsed.mathNodes.find((node) => node.value === "x");
  const flow = parsed.mathNodes.find((node) => node.type === "math" && node.value.includes("frac"));
  assert.equal(inline?.authored, "$$ x $$");
  assert.equal(flow?.authored, "$$\n\\frac{a}{b}\n$$");
});

test("flow math (fenced display) parses with value and authored fence", () => {
  const markdown = "$$\nc = d\n$$\n";
  const parsed = parseMarkdown(markdown, 0, 1);
  const display = parsed.mathNodes[0];
  assert.equal(display?.type, "math");
  assert.equal(display?.value, "c = d");
  assert.equal(display?.authored, "$$\nc = d\n$$");
});
