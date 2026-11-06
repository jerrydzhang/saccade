import test from "node:test";
import assert from "node:assert/strict";
import { decodeBoard } from "../src/lib/decode.ts";

test("a valid narrative board decodes with authored task metadata", () => {
  const text = [
    "task: investigate-step-size",
    "content:",
    "  - text: |",
    "      ## Goal",
    "      Explain the rates.",
    "  - group:",
    "      title: Two step sizes",
    "      relation: comparison",
    "      content:",
    "        - text: |",
    "            At eta=0.01 the factor vanishes.",
  ].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.task, "investigate-step-size");
  assert.equal(result.board.content.length, 2);
  const group = result.board.content[1];
  if (!group || group.kind !== "group") throw new Error("second block is a group");
  assert.equal(group.title, "Two step sizes");
  assert.equal(group.relation, "comparison");
  assert.equal(group.content.length, 1);
});

test("a declaration without a task field still decodes; task is optional metadata", () => {
  const text = ["content:", "  - text: |", "      No task line at all."].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.task, null);
});

test("malformed YAML is a structured error, never a board", () => {
  const text = ["task: broken", "content:", "  - text: |", "      line", "  - group:", "      title: [unclosed"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.length >= 1);
  assert.equal(result.errors[0]?.code, "yaml-syntax");
  assert.ok(typeof result.errors[0]?.line === "number");
});

test("duplicate top-level keys are refused, not silently last-wins", () => {
  const text = ["task: one", "task: two", "content:", "  - text: body"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((error) => error.code === "duplicate-key"));
});

test("unknown top-level keys become diagnostics and decoding continues", () => {
  const text = ["task: demo", "retirement: retired", "content:", "  - text: body"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.diagnostics.length, 1);
  assert.equal(result.board.diagnostics[0]?.code, "unknown-key");
  assert.match(result.board.diagnostics[0]?.message ?? "", /retirement/);
});

test("unknown keys inside a group become diagnostics, not failures", () => {
  const text = [
    "content:",
    "  - group:",
    "      title: T",
    "      layout: spatial",
    "      content:",
    "        - text: body",
  ].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.diagnostics.length, 1);
  assert.match(result.board.diagnostics[0]?.message ?? "", /layout/);
});

test("an unknown block kind is a structural error carrying its path", () => {
  const text = ["content:", "  - widget: {}", "  - text: body"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  const error = result.errors.find((issue) => issue.code === "unknown-block");
  assert.ok(error, "unknown-block error present");
  assert.deepEqual(error?.path, ["content", 0]);
});

test("an incorporation keeps its claimed task, revision and select path", () => {
  const text = [
    "content:",
    "  - incorporation:",
    "      id: model",
    "      source:",
    "        task: source-model",
    "        revision: 647de6e6d2b90c507bf2d05dacc32740558e0bfc",
    "        select: [/content/1/group/content/0]",
    "      content:",
    "        - text: |",
    "            Claimed content.",
  ].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  const block = result.board.content[0];
  if (!block || block.kind !== "incorporation") throw new Error("first block is an incorporation");
  assert.equal(block.id, "model");
  assert.equal(block.source.task, "source-model");
  assert.equal(block.source.revision, "647de6e6d2b90c507bf2d05dacc32740558e0bfc");
  assert.deepEqual(block.source.select, ["/content/1/group/content/0"]);
  assert.equal(block.content.length, 1);
});

test("an incorporation missing source or content is a structured error", () => {
  const text = ["content:", "  - incorporation:", "      id: incomplete"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  const codes = result.errors.map((issue) => issue.code);
  assert.ok(codes.includes("missing-key"));
});

test("a figure requires a path", () => {
  const text = ["content:", "  - figure:", "      caption: no path"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  const codes = result.errors.map((issue) => issue.code);
  assert.ok(codes.includes("missing-key"));
});

test("an annotation decodes as its own block kind", () => {
  const text = ["content:", "  - text: body", "  - annotation: |", "      A review note."].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  const block = result.board.content[1];
  if (!block || block.kind !== "annotation") throw new Error("second block is an annotation");
  assert.equal(block.markdown, "A review note.\n");
});

test("an empty content list is a structured error", () => {
  const text = ["task: empty", "content: []"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((issue) => issue.code === "empty-board"));
});

test("block positions point at authored bytes", () => {
  const text = ["content:", "  - text: |", "      first", "  - text: |", "      second"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  const first = result.board.content[0];
  const second = result.board.content[1];
  if (!first || !second) throw new Error("two blocks decoded");
  // each block's range lands inside its own item, strictly ordered
  assert.ok(first.pos.start.line < second.pos.start.line);
  assert.ok(first.pos.start.offset < second.pos.start.offset);
  assert.ok(first.pos.end.offset <= text.length);
  assert.ok(text.slice(first.pos.start.offset, first.pos.end.offset).includes("first"));
  assert.ok(text.slice(second.pos.start.offset, second.pos.end.offset).includes("second"));
});
