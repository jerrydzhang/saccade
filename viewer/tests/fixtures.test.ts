import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { decodeBoard } from "../src/lib/decode.ts";
import { projectBoard, searchUnits } from "../src/lib/projection.ts";
import type { Block } from "../src/lib/model.ts";

const fixturesDir = fileURLToPath(new URL("../fixtures/", import.meta.url));

function load(name: string): string {
  return readFileSync(`${fixturesDir}${name}`, "utf8");
}

test("every committed fixture is ASCII bytes", () => {
  const names = [
    "real-board.yaml",
    "invalid.yaml",
    "unknown-keys.yaml",
    "missing-asset.yaml",
    "wrapped-math.yaml",
    "display-scroll.yaml",
    "repeated-math.yaml",
    "long-code.yaml",
    "large-diagram.yaml",
    "saccade-and-migration.yaml",
    "saccade-and-migration-sources.md",
  ];
  for (const name of names) {
    const bytes = readFileSync(`${fixturesDir}${name}`);
    for (const byte of bytes) {
      assert.ok(byte === 10 || byte === 9 || (byte >= 32 && byte <= 126), `${name} carries non-ASCII byte ${byte}`);
    }
  }
});

test("the real board decodes with every ratified feature present", () => {
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

test("the invalid fixture produces structured errors with a position", () => {
  const result = decodeBoard(load("invalid.yaml"));
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.length > 0);
  assert.ok(typeof result.errors[0]?.line === "number");
});

test("the unknown-keys fixture decodes with diagnostics for every stray key", () => {
  const result = decodeBoard(load("unknown-keys.yaml"));
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  const messages = result.board.diagnostics.map((diagnostic) => diagnostic.message);
  assert.ok(messages.some((message) => message.includes("retirement")));
  assert.ok(messages.some((message) => message.includes("layout")));
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

test("the real board's figure asset exists beside its declaration", () => {
  const bytes = readFileSync(`${fixturesDir}assets/loop-figure.svg`, "utf8");
  assert.match(bytes, /<svg/);
});

const REAL_BOARD_SHA256 = "ef65496fe48510d44367dac0ae187cf270d2fa2b184b35e312ce9ade78ce3104";

function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

test("the copied migration board keeps the published bytes bit for bit", () => {
  const bytes = readFileSync(`${fixturesDir}saccade-and-migration.yaml`);
  assert.equal(sha256Hex(bytes), REAL_BOARD_SHA256);
});

test("the real migration board decodes under the provisional grammar", () => {
  const result = decodeBoard(readFileSync(`${fixturesDir}saccade-and-migration.yaml`, "utf8"));
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.task, "saccade-current-system-and-migration");
  assert.ok(result.board.diagnostics.length === 0, "no stray keys in the published board");
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
  assert.ok(!sources.trimStart().startsWith("task:"));
});

function decodeOk(text: string) {
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") throw new Error("fixture must decode");
  return result.board;
}
