import test from "node:test";
import assert from "node:assert/strict";
import {
  annotatedBoard,
  locateBlockSpan,
  type Draft,
} from "../src/lib/annotations.ts";

const BOARD = `grammar: 1
task: t-7
content:
  - text: the first part
  - text: the second part
  - group:
      title: the group
      content:
        - text: the nested part
  - text: the last part
`;

test("a top-level draft inserts its annotation right after the part", () => {
  const drafts: Draft[] = [{ path: "/content/1", text: "the reviewer's concern" }];
  const result = annotatedBoard(BOARD, drafts);
  assert.equal(result.missing.length, 0);
  assert.ok(result.text !== null);
  const lines = (result.text ?? "").split("\n");
  const secondPart = lines.findIndex((line) => line === "  - text: the second part");
  assert.ok(secondPart >= 0);
  assert.equal(lines[secondPart + 1], "  - annotation: |");
  assert.equal(lines[secondPart + 2], "      the reviewer's concern");
  assert.equal(lines[secondPart + 3], "  - group:");
});

test("a nested draft lands inside its group, after the nested part", () => {
  const drafts: Draft[] = [{ path: "/content/2/group/content/0", text: "on the nested part" }];
  const result = annotatedBoard(BOARD, drafts);
  assert.equal(result.missing.length, 0);
  const lines = (result.text ?? "").split("\n");
  const nested = lines.findIndex((line) => line === "        - text: the nested part");
  assert.ok(nested >= 0);
  assert.equal(lines[nested + 1], "        - annotation: |");
  assert.equal(lines[nested + 2], "            on the nested part");
});

test("multiple drafts insert at each of their parts, order independent", () => {
  const drafts: Draft[] = [
    { path: "/content/0", text: "first concern" },
    { path: "/content/3", text: "last concern" },
  ];
  const result = annotatedBoard(BOARD, drafts);
  assert.equal(result.missing.length, 0);
  const text = result.text ?? "";
  assert.match(text, /first concern/);
  assert.match(text, /last concern/);
  // the untouched middle stays byte-identical
  assert.match(text, /  - text: the second part\n  - group:/);
});

test("a multi-line annotation keeps its lines indented", () => {
  const drafts: Draft[] = [{ path: "/content/0", text: "line one\n\nline three" }];
  const result = annotatedBoard(BOARD, drafts);
  const text = result.text ?? "";
  assert.match(text, /      line one\n\n      line three/);
});

test("a path the text cannot locate refuses and names it", () => {
  const drafts: Draft[] = [{ path: "/content/9", text: "nowhere" }];
  const result = annotatedBoard(BOARD, drafts);
  assert.equal(result.text, null);
  assert.deepEqual(result.missing, ["/content/9"]);
});

test("the result parses as the same board plus annotations only", () => {
  const drafts: Draft[] = [{ path: "/content/1", text: "concern" }];
  const result = annotatedBoard(BOARD, drafts);
  const before = BOARD.split("\n").filter((line) => line.length > 0);
  const after = (result.text ?? "").split("\n").filter((line) => line.length > 0 && !line.includes("annotation") && line.trim() !== "concern");
  // every original line survives, in order, unchanged
  let cursor = 0;
  for (const line of before) {
    const found = after.indexOf(line, cursor);
    assert.ok(found >= cursor, `line missing or reordered: ${line}`);
    cursor = found;
  }
});

test("locateBlockSpan walks the grammar's select paths", () => {
  const lines = BOARD.split("\n");
  const top = locateBlockSpan(lines, "/content/1");
  assert.ok(top !== null);
  assert.equal(lines[top.start], "  - text: the second part");
  const nested = locateBlockSpan(lines, "/content/2/group/content/0");
  assert.ok(nested !== null);
  assert.equal(lines[nested.start], "        - text: the nested part");
});
