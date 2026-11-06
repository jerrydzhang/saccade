import test from "node:test";
import assert from "node:assert/strict";
import { decodeBoard } from "../src/lib/decode.ts";
import { projectBoard, searchUnits, normalizeQuery } from "../src/lib/projection.ts";

function boardOf(text: string) {
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") throw new Error("fixture must decode");
  return result.board;
}

test("math units carry the TeX value as text and authored bytes as source", () => {
  const board = boardOf(["content:", "  - text: |", "      Value: $\\eta < 2/b$ inline.", "  - text: |", "      $$x = 1$$", "", "      display."].join("\n"));
  const units = projectBoard(board);
  const math = units.filter((unit) => unit.kind === "math");
  assert.equal(math.length, 2);
  const inline = math.find((unit) => unit.display === false);
  const display = math.find((unit) => unit.display === true);
  assert.equal(inline?.text, "\\eta < 2/b");
  assert.equal(inline?.source, "$\\eta < 2/b$");
  assert.equal(display?.source, "$$x = 1$$");
});

test("equal formulas at distinct source locations are distinct occurrences", () => {
  const board = boardOf([
    "content:",
    "  - text: |",
    "      first $0 < \\eta < 2/b$ here",
    "  - text: |",
    "      second $0 < \\eta < 2/b$ there",
  ].join("\n"));
  const units = projectBoard(board);
  const math = units.filter((unit) => unit.kind === "math");
  assert.equal(math.length, 2);
  assert.notEqual(math[0]?.pos.start.offset, math[1]?.pos.start.offset);
  assert.notEqual(math[0]?.pos.start.line, math[1]?.pos.start.line);
  const matches = searchUnits(units, "\\eta");
  assert.equal(matches.length, 2);
});

test("search finds prose and math as logical units", () => {
  const board = boardOf([
    "content:",
    "  - text: |",
    "      Decay without oscillation is possible.",
    "  - text: |",
    "      The factor is $0.99$.",
  ].join("\n"));
  const units = projectBoard(board);
  const prose = searchUnits(units, "oscillation");
  assert.equal(prose.length, 1);
  const tex = searchUnits(units, "0.99");
  assert.equal(tex.length, 1);
  // a delimited query still finds the bare TeX value, once
  const delimited = searchUnits(units, "$0.99$");
  assert.equal(delimited.length, 1);
});

test("a query matching the delimited form alone still finds the unit once", () => {
  const board = boardOf(["content:", "  - text: |", "      Mention $q$ only.", "  - text: |", "      Also $z$."].join("\n"));
  const units = projectBoard(board);
  const matches = searchUnits(units, "$q$");
  assert.equal(matches.length, 1);
  assert.equal(matches[0]?.unit.text, "q");
});

test("code fences and mermaid fences project as code and mermaid units", () => {
  const board = boardOf([
    "content:",
    "  - text: |",
    "      ```mermaid",
    "      flowchart LR",
    "        a --> b",
    "      ```",
    "  - text: |",
    "      ```python",
    "      x = 1",
    "      ```",
  ].join("\n"));
  const units = projectBoard(board);
  assert.ok(units.some((unit) => unit.kind === "mermaid" && unit.text.includes("flowchart")));
  assert.ok(units.some((unit) => unit.kind === "code" && unit.text === "x = 1"));
});

test("group titles project as searchable prose", () => {
  const board = boardOf([
    "content:",
    "  - group:",
    "      title: Ratified content behavior",
    "      content:",
    "        - text: body",
  ].join("\n"));
  const units = projectBoard(board);
  assert.ok(searchUnits(units, "Ratified content").length === 1);
});

test("normalizeQuery strips math delimiters from both ends", () => {
  assert.equal(normalizeQuery("$x$"), "x");
  assert.equal(normalizeQuery("$$x$$"), "x");
  assert.equal(normalizeQuery("plain"), "plain");
  assert.equal(normalizeQuery("$a + b$"), "a + b");
});

test("projection sources are exact authored bytes, not reconstructions", () => {
  const board = boardOf([
    "content:",
    "  - text: |",
    "      Mixed: $$ x $$ paragraph",
    "  - text: |",
    "      $$",
    "      \\frac{a}{b}",
    "      $$",
  ].join("\n"));
  const units = projectBoard(board);
  const mixed = units.find((unit) => unit.kind === "math" && unit.text === "x");
  const flow = units.find((unit) => unit.kind === "math" && unit.text.includes("frac"));
  assert.equal(mixed?.source, "$$ x $$");
  assert.ok(flow?.source.includes("\n"), "flow math source keeps the authored newlines");
  assert.equal(flow?.source, "$$\n\\frac{a}{b}\n$$");
});
