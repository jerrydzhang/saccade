// History mode's diff (t-49), its pure half: components aligned as
// `sac board diff` aligns them, and words marked inside changed text
// with Markdown's structure kept. Painting the marks needs a DOM: the
// history scenario covers it.

import assert from "node:assert/strict";
import { test } from "vitest";
import { decodeBoard } from "../src/lib/board/decode.ts";
import type { Board } from "../src/lib/board/model.ts";
import {
  DEL_CLOSE,
  DEL_OPEN,
  INS_CLOSE,
  INS_OPEN,
  diffBoards,
  wordDiffMarkdown,
} from "../src/lib/board/history-diff.ts";

function board(yaml: string): Board {
  const result = decodeBoard(yaml);
  if (result.status !== "ok") throw new Error(`fixture: ${JSON.stringify(result.errors)}`);
  return result.board;
}

const ins = (text: string) => `${INS_OPEN}${text}${INS_CLOSE}`;
const del = (text: string) => `${DEL_OPEN}${text}${DEL_CLOSE}`;

const R1 = `grammar: 1
task: t-1
content:
  - text: |
      # Title

      The loop holds.
  - group:
      title: Old idea (r5)
      content:
        - text: |
            long argument
  - group:
      title: Evidence
      content:
        - text: |
            one two three
`;

test("a removed group stays where it stood, marked; a changed nested text is marked by word", () => {
  const r2 = R1.replace(/  - group:\n      title: Old idea \(r5\)\n      content:\n        - text: \|\n            long argument\n/, "").replace(
    "one two three",
    "one four three",
  );
  const diff = diffBoards(board(R1), board(r2));
  const content = diff.board.content as (Board["content"][number] & { change?: string })[];
  assert.equal(content.length, 3, "the removed group is placed back");
  assert.equal(content[1]?.change, "del");
  assert.equal(content[0]?.change, undefined, "an unchanged text carries no mark");
  const evidence = content[2];
  if (evidence?.kind !== "group") throw new Error("shape");
  const text = evidence.content[0] as { markdown: string; change?: string };
  assert.equal(text.change, "mod");
  assert.equal(text.markdown, `one ${del("two")}${ins("four")} three\n`);
  assert.deepEqual(diff.changed, ["/content/1", "/content/2/content/0"]);
});

test("the first revision is all added", () => {
  const diff = diffBoards(null, board(R1));
  assert.deepEqual(diff.changed, ["/content/0", "/content/1", "/content/2"]);
  const first = diff.board.content[0] as { markdown: string; change?: string };
  assert.equal(first.change, "add");
  assert.ok(first.markdown.startsWith(`# ${ins("Title")}`), "the heading marker stays in front");
});

test("list, heading and quote markers stay in front of the marks", () => {
  assert.equal(wordDiffMarkdown("- a\n", "- a\n- b\n"), `- a\n- ${ins("b")}\n`);
  assert.equal(wordDiffMarkdown("## Old\n", "## New\n"), `## ${del("Old")}${ins("New")}\n`);
  assert.equal(wordDiffMarkdown("> x\n", "> y\n"), `> ${del("x")}${ins("y")}\n`);
});

test("math and code spans change whole; display math and diagrams show their new source unmarked", () => {
  assert.equal(wordDiffMarkdown("as $a + b$ said", "as $a - b$ said"), `as ${del("$a + b$")}${ins("$a - b$")} said`);
  assert.equal(wordDiffMarkdown("run `x y`", "run `x z`"), `run ${del("`x y`")}${ins("`x z`")}`);
  const display = wordDiffMarkdown("$$\na + b\n$$\n", "$$\na - b\n$$\n");
  assert.ok(!/[\uE000-\uE003]/.test(display), display);
  assert.ok(display.includes("a - b") && !display.includes("a + b"), display);
  const mermaid = wordDiffMarkdown("```mermaid\nflowchart LR\n  a --> b\n```\n", "```mermaid\nflowchart LR\n  a --> c\n```\n");
  assert.ok(!/[\uE000-\uE003]/.test(mermaid), mermaid);
  assert.ok(mermaid.includes("a --> c") && !mermaid.includes("a --> b"), mermaid);
});

test("code inside an ordinary fence is marked by word, and its fence lines are never marked", () => {
  const marked = wordDiffMarkdown("```python\nx = 1\n```\n", "```python\nx = 2\n```\n");
  assert.equal(marked, `\`\`\`python\nx = ${del("1")}${ins("2")}\n\`\`\`\n`);
});
