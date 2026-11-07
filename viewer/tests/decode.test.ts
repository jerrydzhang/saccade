import test from "node:test";
import assert from "node:assert/strict";
import { decodeBoard, MAX_NESTING_DEPTH, SUPPORTED_GRAMMAR_VERSIONS } from "../src/lib/decode.ts";

const VALID_REVISION = "647de6e6d2b90c507bf2d05dacc32740558e0bfc";

function validBoard(...content: string[]): string {
  return ["grammar: 1", "task: investigate-step-size", "content:", ...content].join("\n");
}

test("the supported grammar range is exactly version 1", () => {
  assert.deepEqual(SUPPORTED_GRAMMAR_VERSIONS, [1]);
  assert.equal(MAX_NESTING_DEPTH, 100);
});

test("a grammar 1 board decodes with authored task metadata", () => {
  const text = validBoard(
    "  - text: |",
    "      ## Goal",
    "      Explain the rates.",
    "  - group:",
    "      title: Two step sizes",
    "      content:",
    "        - text: |",
    "            At eta=0.01 the factor vanishes.",
  );
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.grammar, 1);
  assert.equal(result.board.task, "investigate-step-size");
  assert.equal(result.board.content.length, 2);
  const group = result.board.content[1];
  if (!group || group.kind !== "group") throw new Error("second block is a group");
  assert.equal(group.title, "Two step sizes");
  assert.equal(group.content.length, 1);
});

test("a board without a grammar field is refused, not treated as legacy", () => {
  const text = ["task: pre-grammar", "content:", "  - text: body"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  const error = result.errors[0];
  assert.equal(error?.code, "missing-key");
  assert.match(error?.message ?? "", /grammar: 1/);
});

test("an unknown major version refuses with a message naming both versions", () => {
  for (const version of ["2", "0", "17"]) {
    const text = validBoard("  - text: body").replace("grammar: 1", `grammar: ${version}`);
    const result = decodeBoard(text);
    assert.equal(result.status, "invalid", `grammar ${version} must refuse`);
    if (result.status !== "invalid") return;
    const error = result.errors[0];
    assert.equal(error?.code, "grammar-version");
    assert.match(error?.message ?? "", new RegExp(`grammar ${version}`));
    assert.match(error?.message ?? "", /grammar 1/);
  }
});

test("a grammar value that is not the plain integer 1 refuses", () => {
  for (const written of ['"1"', "1.0", "yes", "one"]) {
    const text = validBoard("  - text: body").replace("grammar: 1", `grammar: ${written}`);
    const result = decodeBoard(text);
    assert.equal(result.status, "invalid", `grammar: ${written} must refuse`);
    if (result.status !== "invalid") return;
    assert.equal(result.errors[0]?.code, "grammar-version");
  }
});

test("malformed YAML is a structured error, never a board", () => {
  const text = validBoard("  - text: |", "      line", "  - group:", "      title: [unclosed");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.length >= 1);
  assert.equal(result.errors[0]?.code, "yaml-syntax");
  assert.ok(typeof result.errors[0]?.line === "number");
});

test("duplicate keys are refused, not silently last-wins", () => {
  const text = ["grammar: 1", "task: one", "task: two", "content:", "  - text: body"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((error) => error.code === "duplicate-key"));
});

test("several YAML documents in one file are refused", () => {
  const text = validBoard("  - text: body") + "\n---\n" + validBoard("  - text: other");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((error) => error.code === "multiple-documents"));
});

test("YAML anchors and aliases are refused", () => {
  const text = validBoard("  - text: &body", "      shared prose", "  - text: *body");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((error) => error.code === "anchor-alias"));
});

test("an unknown field at the board top is a refusal, matching the crate", () => {
  const text = ["grammar: 1", "task: demo", "retirement: retired", "content:", "  - text: body"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  const error = result.errors.find((issue) => issue.code === "unknown-key");
  assert.ok(error, "unknown-key refusal present");
  assert.match(error?.message ?? "", /retirement/);
});

test("a group relation hint is a refusal: grammar 1 has no layout hints", () => {
  const text = validBoard(
    "  - group:",
    "      title: T",
    "      relation: comparison",
    "      content:",
    "        - text: body",
  );
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  const error = result.errors.find((issue) => issue.code === "unknown-key");
  assert.ok(error, "unknown-key refusal present");
  assert.match(error?.message ?? "", /relation/);
});

test("an unknown component kind softens into a raw-source block with a diagnostic", () => {
  const text = validBoard("  - text: body", "  - changelog:", "      added:", "        - something");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.diagnostics.length, 1);
  const diagnostic = result.board.diagnostics[0];
  assert.equal(diagnostic?.code, "unknown-kind");
  assert.match(diagnostic?.message ?? "", /changelog/);
  assert.match(diagnostic?.message ?? "", /\/content\/1/);
  const block = result.board.content[1];
  if (!block || block.kind !== "unknown") throw new Error("second block is the unknown kind");
  assert.equal(block.name, "changelog");
  assert.ok(block.source.includes("added"), "the raw source is the authored bytes");
  assert.ok(block.pos.start.line > 0, "the raw block points at its authored position");
});

test("an incorporation keeps its claimed task, revision and select paths", () => {
  const text = validBoard(
    "  - incorporation:",
    "      source:",
    "        task: source-model",
    `        revision: ${VALID_REVISION}`,
    "        select: [/content/1/group/content/0]",
    "      content:",
    "        - text: |",
    "            Claimed content.",
  );
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  const block = result.board.content[0];
  if (!block || block.kind !== "incorporation") throw new Error("first block is an incorporation");
  assert.equal(block.source.task, "source-model");
  assert.equal(block.source.revision, VALID_REVISION);
  assert.deepEqual(block.source.select, ["/content/1/group/content/0"]);
  assert.equal(block.content.length, 1);
});

test("an incorporation missing source or content is a structured error", () => {
  const text = validBoard("  - incorporation:", "      source:");
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  const codes = result.errors.map((issue) => issue.code);
  assert.ok(codes.includes("missing-key"));
});

test("a source revision must be the full lowercase hex commit", () => {
  for (const revision of ["647de6e", VALID_REVISION.toUpperCase(), "z" + VALID_REVISION.slice(1)]) {
    const text = validBoard(
      "  - incorporation:",
      "      source:",
      "        task: source-model",
      `        revision: ${revision}`,
      "        select: [/content/0]",
      "      content:",
      "        - text: receiving",
    );
    const result = decodeBoard(text);
    assert.equal(result.status, "invalid", `revision ${revision} must refuse`);
    if (result.status !== "invalid") return;
    assert.ok(result.errors.some((issue) => issue.code === "bad-revision"));
  }
});

test("a select path must be positional and name whole components", () => {
  for (const path of ["content/0", "/content", "/content/0/text", "/content/x", "/content/0/text/content/0", "/content/0/widget/content/0", "//content/0"]) {
    const text = validBoard(
      "  - incorporation:",
      "      source:",
      "        task: source-model",
      `        revision: ${VALID_REVISION}`,
      `        select: [${path}]`,
      "      content:",
      "        - text: receiving",
    );
    const result = decodeBoard(text);
    assert.equal(result.status, "invalid", `select path ${path} must refuse`);
    if (result.status !== "invalid") return;
    assert.ok(result.errors.some((issue) => issue.code === "bad-select-path"), `${path}: bad-select-path expected`);
  }
});

test("an incorporation selects at least one path", () => {
  const text = validBoard(
    "  - incorporation:",
    "      source:",
    "        task: source-model",
    `        revision: ${VALID_REVISION}`,
    "        select: []",
    "      content:",
    "        - text: receiving",
  );
  const result = decodeBoard(text);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((issue) => issue.code === "empty-select"));
});

test("an incorporation carries receiving text of its own", () => {
  const withoutText = validBoard(
    "  - incorporation:",
    "      source:",
    "        task: source-model",
    `        revision: ${VALID_REVISION}`,
    "        select: [/content/0]",
    "      content:",
    "        - annotation: only a note",
  );
  const result = decodeBoard(withoutText);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((issue) => issue.code === "no-receiving-text"));

  // text through a group counts; text inside a nested incorporation belongs
  // to that incorporation's own source and does not
  const throughGroup = validBoard(
    "  - incorporation:",
    "      source:",
    "        task: source-model",
    `        revision: ${VALID_REVISION}`,
    "        select: [/content/0]",
    "      content:",
    "        - group:",
    "            content:",
    "              - text: receiving through the group",
  );
  assert.equal(decodeBoard(throughGroup).status, "ok");

  const onlyNested = validBoard(
    "  - incorporation:",
    "      source:",
    "        task: source-model",
    `        revision: ${VALID_REVISION}`,
    "        select: [/content/0]",
    "      content:",
    "        - incorporation:",
    "            source:",
    "              task: inner-source",
    "              revision: " + VALID_REVISION,
    "              select: [/content/0]",
    "            content:",
    "              - text: belongs to the inner incorporation",
  );
  const nested = decodeBoard(onlyNested);
  assert.equal(nested.status, "invalid");
  if (nested.status !== "invalid") return;
  assert.ok(nested.errors.some((issue) => issue.code === "no-receiving-text"));
});

test("a figure names its input file and a caption, both required", () => {
  const withoutCaption = validBoard("  - figure:", "      input: assets/loop-figure.svg");
  const result = decodeBoard(withoutCaption);
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((issue) => issue.code === "missing-key"));
});

test("a figure input stays inside the board tree", () => {
  for (const [input, code] of [
    ["/etc/loop.svg", "bad-figure-path"],
    ["assets/../loop.svg", "bad-figure-path"],
    ["", "bad-figure-path"],
  ] as const) {
    const text = validBoard("  - figure:", `      input: ${input === "" ? '""' : input}`, "      caption: a figure");
    const result = decodeBoard(text);
    assert.equal(result.status, "invalid", `figure input ${input} must refuse`);
    if (result.status !== "invalid") return;
    assert.ok(result.errors.some((issue) => issue.code === code), `${input}: ${code} expected`);
  }
});

test("an empty content list decodes; grammar 1 allows an empty board", () => {
  const text = ["grammar: 1", "task: empty-demo", "content: []"].join("\n");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  assert.equal(result.board.content.length, 0);
});

test("an annotation decodes as its own block kind", () => {
  const text = validBoard("  - text: body", "  - annotation: |", "      A review note.");
  const result = decodeBoard(text);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") return;
  const block = result.board.content[1];
  if (!block || block.kind !== "annotation") throw new Error("second block is an annotation");
  assert.equal(block.markdown, "A review note.\n");
});

test("nesting past the depth cap refuses, matching the crate", () => {
  const depth = MAX_NESTING_DEPTH + 1;
  const lines = ["grammar: 1", "task: too-deep", "content:"];
  for (let level = 0; level < depth; level++) {
    lines.push("  ".repeat(1 + 2 * level) + "- group:");
    lines.push("  ".repeat(3 + 2 * level) + "content:");
  }
  lines.push("  ".repeat(1 + 2 * depth) + "- text: past the cap");
  const result = decodeBoard(lines.join("\n"));
  assert.equal(result.status, "invalid");
  if (result.status !== "invalid") return;
  assert.ok(result.errors.some((issue) => issue.code === "too-deep"));
});

test("block positions point at authored bytes", () => {
  const text = validBoard("  - text: |", "      first", "  - text: |", "      second");
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
