import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { decodeBoard } from "../src/lib/board/decode.ts";
import { projectBoard, searchUnits } from "../src/lib/board/projection.ts";
import type { Block } from "../src/lib/board/model.ts";

const fixturesDir = fileURLToPath(new URL("../fixtures/", import.meta.url));

function load(name: string): string {
  return readFileSync(`${fixturesDir}${name}`, "utf8");
}

// the viewer's grammar corpus: every supported-version board, the soft
// unknown-kind board, and the refused boundary boards, with the verdicts
// pinned here and pinned again on the Rust side over the same files
const VALID_FIXTURES = [
  "real-board.yaml",
  "unknown-kind.yaml",
  "missing-asset.yaml",
  "wrapped-math.yaml",
  "display-scroll.yaml",
  "repeated-math.yaml",
  "long-code.yaml",
  "large-diagram.yaml",
  "saccade-and-migration.yaml",
];

const REFUSED_FIXTURES: Array<[string, string]> = [
  ["grammar-2.yaml", "grammar-version"],
  ["missing-grammar.yaml", "missing-key"],
  ["too-deep.yaml", "too-deep"],
  ["invalid.yaml", "yaml-syntax"],
];

test("every committed fixture is ASCII bytes", () => {
  const names = [...VALID_FIXTURES, ...REFUSED_FIXTURES.map(([name]) => name), "saccade-and-migration-sources.md"];
  for (const name of names) {
    const bytes = readFileSync(`${fixturesDir}${name}`);
    for (const byte of bytes) {
      assert.ok(byte === 10 || byte === 9 || (byte >= 32 && byte <= 126), `${name} carries non-ASCII byte ${byte}`);
    }
  }
});

test("every supported-version fixture decodes under grammar 1 with no diagnostics", () => {
  for (const name of VALID_FIXTURES) {
    if (name === "unknown-kind.yaml") continue; // the soft case, pinned below
    const result = decodeBoard(load(name));
    assert.equal(result.status, "ok", `${name} must decode`);
    if (result.status !== "ok") return;
    assert.equal(result.board.grammar, 1, `${name} declares grammar 1`);
    assert.equal(result.board.diagnostics.length, 0, `${name} carries no stray keys`);
  }
});

test("the refused fixtures refuse under their pinned family", () => {
  for (const [name, code] of REFUSED_FIXTURES) {
    const result = decodeBoard(load(name));
    assert.equal(result.status, "invalid", `${name} must refuse`);
    if (result.status !== "invalid") return;
    assert.ok(
      result.errors.some((issue) => issue.code === code),
      `${name} refuses as ${code}`,
    );
  }
});

test("the real board carries every ratified kind", () => {
  const result = decodeBoard(load("real-board.yaml"));
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  const kinds = new Set(result.board.content.map((block) => block.kind));
  const expected: Block["kind"][] = ["text", "group", "annotation", "incorporation", "figure"];
  for (const kind of expected) {
    assert.ok(kinds.has(kind), `${kind} present`);
  }
  const units = projectBoard(result.board);
  assert.ok(units.some((unit) => unit.kind === "math" && unit.display));
  assert.ok(units.some((unit) => unit.kind === "math" && !unit.display));
  assert.ok(units.some((unit) => unit.kind === "mermaid"));
  assert.ok(units.some((unit) => unit.kind === "code"));
});

test("the figure fixture's asset exists beside its declaration", () => {
  const bytes = readFileSync(`${fixturesDir}assets/loop-figure.svg`, "utf8");
  assert.match(bytes, /<svg/);
});

test("the unknown-kind fixture renders its unknown component as raw source", () => {
  const result = decodeBoard(load("unknown-kind.yaml"));
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.diagnostics.length, 1);
  const diagnostic = result.board.diagnostics[0];
  assert.equal(diagnostic?.code, "unknown-kind");
  assert.match(diagnostic?.message ?? "", /changelog/);
  const unknown = result.board.content.find((block) => block.kind === "unknown");
  if (!unknown || unknown.kind !== "unknown") throw new Error("the unknown block decoded");
  assert.equal(unknown.name, "changelog");
  assert.ok(unknown.source.includes("added"), "the raw source is the authored bytes");
  const units = projectBoard(result.board);
  assert.ok(units.some((unit) => unit.blockKind === "unknown" && unit.kind === "code"), "the raw source stays searchable");
});

test("the missing-asset fixture references a file that truly is absent", () => {
  assert.throws(() => readFileSync(`${fixturesDir}assets/absent-figure.svg`), /ENOENT/);
  const result = decodeBoard(load("missing-asset.yaml"));
  assert.equal(result.status, "ok");
});

test("the wrapped and display math fixtures carry their overflow cases", () => {
  const wrapped = projectBoard(decodeOk(load("wrapped-math.yaml")));
  const longInline = wrapped.filter((unit) => unit.kind === "math" && !unit.display && unit.text.length > 40);
  assert.ok(longInline.length >= 1, "a genuinely long inline expression exists");

  const display = projectBoard(decodeOk(load("display-scroll.yaml")));
  const wideDisplay = display.filter((unit) => unit.kind === "math" && unit.display && unit.text.length > 100);
  assert.ok(wideDisplay.length >= 1, "a genuinely wide display expression exists");
});

test("the repeated-math fixture holds the same formula at distinct locations", () => {
  const units = projectBoard(decodeOk(load("repeated-math.yaml")));
  const matches = searchUnits(units, "0 < \\eta < 2/b");
  assert.equal(matches.length, 2);
  assert.notEqual(matches[0]?.unit.pos.start.offset, matches[1]?.unit.pos.start.offset);
});

test("the large-diagram fixture carries one big mermaid fence", () => {
  const units = projectBoard(decodeOk(load("large-diagram.yaml")));
  const diagrams = units.filter((unit) => unit.kind === "mermaid");
  assert.equal(diagrams.length, 1);
  assert.ok(diagrams[0] && diagrams[0].text.split("\n").length > 15, "the diagram is genuinely large");
});

const MIGRATION_BOARD_SHA256 = "9c0109e0beba5c62ce255f9f74b42b86494146fe6c5a2994ff3cf972852eb38f";

function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

test("the migration board's grammar 1 conversion is pinned byte for byte", () => {
  // the published pre-grammar bytes were converted to grammar 1 in the
  // viewer's grammar change; the converted fixture is the pinned copy
  // from here on, and the conversion is visible in the pinned hash
  const bytes = readFileSync(`${fixturesDir}saccade-and-migration.yaml`);
  assert.equal(sha256Hex(bytes), MIGRATION_BOARD_SHA256);
});

test("the migration board decodes under grammar 1", () => {
  const result = decodeBoard(readFileSync(`${fixturesDir}saccade-and-migration.yaml`, "utf8"));
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.grammar, 1);
  assert.equal(result.board.task, "t-18");
  assert.equal(result.board.diagnostics.length, 0, "no stray keys in the converted board");
  const kinds = new Set(result.board.content.map((block) => block.kind));
  assert.ok(kinds.has("text"));
  assert.ok(kinds.has("annotation"));
  assert.ok(kinds.has("group"));
  const units = projectBoard(result.board);
  assert.ok(units.some((unit) => unit.kind === "mermaid"), "the board carries diagrams");
  assert.ok(units.some((unit) => unit.kind === "code"), "the board carries code");
  assert.ok(units.length > 40, "the board is substantial");
});

test("the source-map companion decodes nothing and stays plain prose", () => {
  const sources = readFileSync(`${fixturesDir}saccade-and-migration-sources.md`, "utf8");
  assert.match(sources, /sources/);
  // it is a companion document, not a board: decoding it as YAML must not
  // be attempted by the fixture adapter (no .yaml extension)
  assert.ok(!sources.trimStart().startsWith("grammar:"));
});

function decodeOk(text: string) {
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") throw new Error("fixture must decode");
  return result.board;
}
