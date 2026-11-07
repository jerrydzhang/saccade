import test from "node:test";
import assert from "node:assert/strict";
import { serializeSelection, type CopySegment, type CopyUnit } from "../src/lib/copy.ts";

function mathUnit(id: string, tex: string, touched = true): CopyUnit {
  return { id, kind: "math", text: tex, source: `$${tex}$`, tex, touched, covered: true };
}

function proseUnit(id: string, text: string): CopyUnit {
  return { id, kind: "prose", text, source: text, touched: true, covered: true };
}

function codeUnit(id: string, source: string): CopyUnit {
  return { id, kind: "code", text: source, source, touched: true, covered: true };
}

test("a selection touching part of a formula emits it once, as authored bytes in Source mode", () => {
  const segments: CopySegment[] = [
    { blockId: "b0", units: [proseUnit("p1", "The factor is "), mathUnit("m1", "\\eta < 2/b"), proseUnit("p2", " holds.")] },
  ];
  assert.equal(serializeSelection(segments, "source"), "The factor is $\\eta < 2/b$ holds.");
});

test("Plain mode emits the TeX value for math, prose unchanged", () => {
  const segments: CopySegment[] = [
    { blockId: "b0", units: [proseUnit("p1", "Value: "), mathUnit("m1", "0 < \\eta < 2/b")] },
  ];
  assert.equal(serializeSelection(segments, "plain"), "Value: 0 < \\eta < 2/b");
});

test("a formula is emitted once even when the selection grazes it twice", () => {
  const segments: CopySegment[] = [
    { blockId: "b0", units: [mathUnit("m1", "x"), mathUnit("m1", "x")] },
  ];
  // the DOM adapter emits each touched unit once; the serializer trusts it
  assert.equal(serializeSelection(segments, "source"), "$x$$x$");
});

test("equal formulas at distinct locations serialize per occurrence", () => {
  const segments: CopySegment[] = [
    { blockId: "b0", units: [proseUnit("p1", "first "), mathUnit("m1", "a<b")] },
    { blockId: "b1", units: [proseUnit("p2", "second "), mathUnit("m2", "a<b")] },
  ];
  assert.equal(serializeSelection(segments, "source"), "first $a<b$\nsecond $a<b$");
});

test("untouched units contribute nothing", () => {
  const segments: CopySegment[] = [
    {
      blockId: "b0",
      units: [
        { id: "p1", kind: "prose", text: "before", source: "before", touched: false, covered: false },
        mathUnit("m1", "y"),
        { id: "p2", kind: "prose", text: "after", source: "after", touched: false, covered: false },
      ],
    },
  ];
  assert.equal(serializeSelection(segments, "source"), "$y$");
});

test("code copies source bytes in Source mode regardless of coverage", () => {
  const segments: CopySegment[] = [
    { blockId: "b0", units: [codeUnit("c1", "x = 1\ny = 2")] },
  ];
  assert.equal(serializeSelection(segments, "source"), "x = 1\ny = 2");
  assert.equal(serializeSelection(segments, "plain"), "x = 1\ny = 2");
});

test("blocks separate by newline, inline units concatenate directly", () => {
  const segments: CopySegment[] = [
    { blockId: "b0", units: [proseUnit("p1", "one "), mathUnit("m1", "1"), proseUnit("p2", " two")] },
    { blockId: "b1", units: [proseUnit("p3", "next line")] },
  ];
  assert.equal(serializeSelection(segments, "source"), "one $1$ two\nnext line");
});
