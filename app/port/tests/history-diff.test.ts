// History mode's diff (t-49, t-57), its pure half: components aligned
// by the changes the tracker serves (the shapes `sac board diff` names;
// saccade/tracker/tests/api.rs pins the page carrying them), words marked
// inside changed text with Markdown's structure kept, and every shown
// part placed in both revisions. Painting the marks needs a DOM: the
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
  placeIn,
  shownAt,
  touches,
  wordDiffMarkdown,
  type ComponentChange,
} from "../src/lib/board/history-diff.ts";
import { decodePage } from "../src/lib/pages.ts";

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
  // what changes.rs names for this pair (its own test fixture's shape)
  const changes: ComponentChange[] = [
    { change: "removed", from: "/content/1" },
    { change: "changed", from: "/content/2/group/content/0", to: "/content/1/group/content/0" },
  ];
  const diff = diffBoards(board(R1), board(r2), changes);
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
  // a run of changed words is one mark, its spaces inside
  assert.equal(wordDiffMarkdown("a b c d\n", "a x y d\n"), `a ${del("b c")}${ins("x y")} d\n`);
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

test("a moved component shows where it went, marked moved, and is not placed back", () => {
  // the Title text moves to the board's end, unchanged
  const title = "  - text: |\n      # Title\n\n      The loop holds.\n";
  const r2 = R1.replace(title, "") + title;
  const diff = diffBoards(board(R1), board(r2), [{ change: "moved", from: "/content/0", to: "/content/2" }]);
  const content = diff.board.content as (Board["content"][number] & { change?: string })[];
  assert.equal(content.length, 3);
  assert.equal(content[2]?.change, "move");
  assert.equal((content[2] as { markdown: string }).markdown, "# Title\n\nThe loop holds.\n", "a move marks no words");
  assert.deepEqual(diff.changed, ["/content/2"]);
  assert.deepEqual(placeIn(diff, "/content/2", "before"), { path: "/content/0", exact: true });
  assert.deepEqual(placeIn(diff, "/content/0", "before"), { path: "/content/1", exact: true });
});

test("every shown part stands in both revisions; a gone one reads as its nearest surviving neighbour", () => {
  const r2 = R1.replace(/  - group:\n      title: Old idea \(r5\)\n      content:\n        - text: \|\n            long argument\n/, "").replace(
    "one two three",
    "one four three",
  );
  const diff = diffBoards(board(R1), board(r2), [
    { change: "removed", from: "/content/1" },
    { change: "changed", from: "/content/2/group/content/0", to: "/content/1/group/content/0" },
  ]);
  // the shown board: Title, the struck Old idea, Evidence
  assert.deepEqual(placeIn(diff, "/content/2/content/0", "before"), { path: "/content/2/content/0", exact: true });
  assert.deepEqual(placeIn(diff, "/content/2/content/0", "after"), { path: "/content/1/content/0", exact: true });
  // the removed group is gone after: the part above it stands in
  assert.deepEqual(placeIn(diff, "/content/1", "after"), { path: "/content/0", exact: false });
  assert.deepEqual(placeIn(diff, "/content/1/content/0", "after"), { path: "/content/0", exact: false });
  // the later revision's paths find their shown parts
  assert.deepEqual(shownAt(diff, "after", "/content/1/content/0"), { shown: "/content/2/content/0", exact: true });
  assert.deepEqual(shownAt(diff, "before", "/content/1"), { shown: "/content/1", exact: true });
  // a ref inside a part reads as the part
  assert.deepEqual(shownAt(diff, "after", "/content/0/ref/0"), { shown: "/content/0", exact: false });
  // a group with a changed text inside is touched; the unchanged title is not
  assert.ok(touches(diff, "/content/2"));
  assert.ok(touches(diff, "/content/1"));
  assert.ok(!touches(diff, "/content/0"));
});

test("an added part has no place before: its neighbour stands in, the one above first", () => {
  const r2 = R1.replace("  - group:\n      title: Evidence", "  - text: |\n      inserted\n  - group:\n      title: Evidence");
  const diff = diffBoards(board(R1), board(r2), [{ change: "added", to: "/content/2" }]);
  assert.deepEqual(diff.changed, ["/content/2"]);
  assert.deepEqual(placeIn(diff, "/content/2", "before"), { path: "/content/1", exact: false });
  assert.deepEqual(placeIn(diff, "/content/3", "before"), { path: "/content/2", exact: true });
});

test("the diff page's changes decode; an unknown kind rides past", () => {
  const page = decodePage({
    view: { major: 1 },
    kind: "document",
    location: "t-1@r1..r2/diff",
    seq: 3,
    content: [{ kind: "text", text: "--- a\n+++ b\n" }],
    changes: [
      { change: "added", label: "annotation", to: "/content/2" },
      { change: "moved", label: "figure f.png", from: "/content/0", to: "/content/1" },
      { change: "renamed", label: "a later kind" },
    ],
  });
  assert.deepEqual(page.changes, [
    { change: "added", label: "annotation", to: "/content/2" },
    { change: "moved", label: "figure f.png", from: "/content/0", to: "/content/1" },
  ]);
  assert.equal(decodePage({ view: { major: 1 }, kind: "document", location: "", seq: 1 }).changes, undefined);
});
