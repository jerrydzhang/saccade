// The shared label overlay's placement model (order of work step
// 3). Pure math: host-relative placement, no chip for unmeasurable
// parts, the visibility contract F3 violated.

import assert from "node:assert/strict";
import { test } from "vitest";

import { CHIP_GAP, CHIP_MARGIN_ROOM, chipPlacements, estimatedChipSize, isWithinHost } from "../src/lib/select/overlay.ts";

test("a block's chip sits in the margin to its left, level with its top", () => {
  const placements = chipPlacements(
    [
      { part: "/content/0", label: "A" },
      { part: "/content/3", label: "S" },
    ],
    (part) =>
      part === "/content/0"
        ? { left: 120, top: 340, width: 500, height: 60 }
        : { left: 48, top: 11279, width: 500, height: 30 },
  );
  assert.deepEqual(placements, [
    { part: "/content/0", label: "A", left: 120 - CHIP_GAP, top: 340, side: "margin" },
    { part: "/content/3", label: "S", left: 48 - CHIP_GAP, top: 11279, side: "margin" },
  ]);
});

test("an inline target's chip stands just before it, centred on its line - its text stays clear", () => {
  const [placement] = chipPlacements([{ part: "/content/2/ref/0", label: "D" }], () => ({
    left: 300,
    top: 400,
    width: 40,
    height: 20,
    inline: true,
  }));
  assert.equal(placement!.side, "before");
  assert.ok(placement!.left < 300, "the chip's right edge is left of the link's first character");
  assert.equal(placement!.top, 410, "centred on the link's line");
});

test("a block with no margin to stand in takes the chip inside its top-left corner", () => {
  const [placement] = chipPlacements([{ part: "/row", label: "A" }], () => ({ left: CHIP_MARGIN_ROOM - 1, top: 10 }));
  assert.deepEqual(placement, { part: "/row", label: "A", left: CHIP_MARGIN_ROOM - 1, top: 10, side: "inside" });
});

test("an unmeasurable part gets no chip - a label over nothing is a lie", () => {
  const placements = chipPlacements(
    [
      { part: "/gone", label: "A" },
      { part: "/here", label: "S" },
    ],
    (part) => (part === "/here" ? { left: 60, top: 10 } : null),
  );
  assert.equal(placements.length, 1);
  assert.equal(placements[0]!.part, "/here");
});

test("the visibility contract: inside the host's content box", () => {
  const host = { width: 800, height: 600 };
  assert.equal(isWithinHost({ left: 0, top: 0 }, host), true, "the origin is inside");
  assert.equal(isWithinHost({ left: 799, top: 599 }, host), true, "the far corner is inside");
  assert.equal(isWithinHost({ left: 800, top: 0 }, host), false, "past the right edge");
  assert.equal(isWithinHost({ left: 0, top: 600 }, host), false, "past the bottom - F3's shape");
  assert.equal(isWithinHost({ left: -1, top: 0 }, host), false, "left of the host");
});

test("F3's arithmetic caught as a visibility failure: a chip placed a content-height low", () => {
  // the host is 6000px of content; the chip's part sits at top 100
  // but the broken layer added the full height beneath the host
  const host = { width: 800, height: 6000 };
  const partBox = { left: 40, top: 100 };
  const brokenLayerOffset = 6000;
  assert.equal(isWithinHost(partBox, host), true, "the true placement is visible");
  assert.equal(
    isWithinHost({ left: partBox.left, top: partBox.top + brokenLayerOffset }, host),
    false,
    "the F3 placement (part top + the host's own height) is out of the host",
  );
});

test("a link that opens its block keeps the nearest chip; the block's steps further into the margin", () => {
  const boxes: Record<string, { left: number; top: number; width: number; height: number; inline?: boolean }> = {
    "/content/2": { left: 120, top: 200, width: 500, height: 52 },
    "/content/2/ref/0": { left: 120, top: 203, width: 24, height: 20, inline: true },
  };
  const placements = chipPlacements(
    [
      { part: "/content/2", label: "J" },
      { part: "/content/2/ref/0", label: "K" },
    ],
    (part) => boxes[part] ?? null,
  );
  assert.deepEqual(placements.map((entry) => entry.part), ["/content/2", "/content/2/ref/0"], "the input order holds");
  const [block, link] = placements;
  assert.equal(link!.left, 120 - CHIP_GAP / 2, "the link's chip stands just before it");
  const width = estimatedChipSize("K").width;
  assert.ok(block!.left <= link!.left - width, "the block's chip clears the link's");
});
