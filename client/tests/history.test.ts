import test from "node:test";
import assert from "node:assert/strict";
import {
  alignParts,
  markerChange,
  markerSplit,
  marksToSentinelText,
  structuredLineDiff,
  wordDiff,
} from "../src/lib/history.ts";
import type { Block } from "../src/lib/board/model.ts";

function text(markdown: string): Block {
  return { kind: "text", markdown, pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } };
}

function figure(input: string): Block {
  return { kind: "figure", input, caption: "", pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } };
}

test("parts align across shifts: same, changed, added, removed", () => {
  const oldBlocks = [text("one"), text("two"), figure("a.png")];
  const newBlocks = [text("one"), text("two revised"), figure("a.png"), text("three")];
  const alignment = alignParts(oldBlocks, newBlocks);
  assert.deepEqual(
    alignment.map((step) => step.op),
    ["same", "changed", "same", "added"],
  );
  assert.equal(alignment[0]?.oldIndex, 0);
  assert.equal(alignment[1]?.oldIndex, 1);
  assert.equal(alignment[1]?.newIndex, 1);
});

test("a moved part is a move, not a remove-plus-add", () => {
  const oldBlocks = [text("one"), text("two")];
  const newBlocks = [text("two"), text("one")];
  const alignment = alignParts(oldBlocks, newBlocks);
  const moved = alignment.filter((step) => step.op === "moved");
  const same = alignment.filter((step) => step.op === "same");
  assert.equal(moved.length + same.length, 2, "both parts survive the swap");
  assert.equal(moved.length, 1, "the displaced part is a move");
});

test("markers split off bullets, numbers, headings and quotes", () => {
  assert.deepEqual(markerSplit("- a point"), { marker: "- ", rest: "a point" });
  assert.deepEqual(markerSplit("  3. third"), { marker: "  3. ", rest: "third" });
  assert.deepEqual(markerSplit("## Heading"), { marker: "## ", rest: "Heading" });
  assert.deepEqual(markerSplit("> quoted"), { marker: "> ", rest: "quoted" });
  assert.deepEqual(markerSplit("plain prose"), { marker: "", rest: "plain prose" });
});

test("a changed list marker moves in front of the content marks", () => {
  const marks = markerChange("- ", "* ");
  assert.deepEqual(marks, [
    { type: "del", text: "- " },
    { type: "add", text: "* " },
  ]);
  const same = markerChange("- ", "- ");
  assert.deepEqual(same, [{ type: "same", text: "- " }]);
});

test("the word diff marks what moved between two texts", () => {
  const marks = wordDiff("the tone arm resonates", "the tone arm hums");
  assert.deepEqual(marks, [
    { type: "same", text: "the tone arm " },
    { type: "del", text: "resonates" },
    { type: "add", text: "hums" },
  ]);
});

test("the structured line diff keeps the marks after the markers", () => {
  const old = "- the tone arm resonates near 7 Hz\n- the plinth is birch ply";
  const newer = "- the tone arm hums near 7 Hz\n- the plinth is birch ply";
  const marks = structuredLineDiff(old, newer);
  const text = marksToSentinelText(marks);
  // the bullet stays common, the change rides the words after it
  assert.match(text, /^- the tone arm /);
  assert.match(text, /\uE002resonates\uE003/);
  assert.match(text, /\uE000hums\uE001/);
  assert.match(text, /- the plinth is birch ply/);
});

test("a dropped line renders as a del mark, a new line as an add mark", () => {
  const old = "first\nsecond\nthird";
  const newer = "first\nthird\nfourth";
  const text = marksToSentinelText(structuredLineDiff(old, newer));
  assert.match(text, /\uE002second\uE003/);
  assert.match(text, /\uE000fourth\uE001/);
  assert.match(text, /first/);
  assert.match(text, /third/);
});

test("a heading's marker change rides in front of the content", () => {
  const text = marksToSentinelText(structuredLineDiff("## The Question", "### The Question"));
  assert.match(text, /\uE002## \uE003\uE000### \uE001/);
  assert.match(text, /The Question/);
});
