// The shared label overlay's placement model (order of work step
// 3). Pure math: host-relative placement, no chip for unmeasurable
// parts, the visibility contract F3 violated.

import assert from "node:assert/strict";
import { test } from "vitest";

import { CHIP_GAP, CHIP_MARGIN_ROOM, chipPlacements, estimatedChipSize, isWithinHost } from "../src/lib/select/overlay.ts";

test("a block's chip sits in the margin to its left, centred on its first line", () => {
  const placements = chipPlacements(
    [
      { part: "/content/0", label: "A" },
      { part: "/content/3", label: "S" },
    ],
    (part) =>
      part === "/content/0"
        ? { left: 0, top: 340, width: 500, height: 30, room: 400 }
        : { left: 48, top: 11279, width: 500, height: 30 },
  );
  assert.deepEqual(placements, [
    { part: "/content/0", label: "A", left: -CHIP_GAP, top: 355, side: "margin" },
    { part: "/content/3", label: "S", left: 48 - CHIP_GAP, top: 11294, side: "margin" },
  ]);
});

test("the margin is the room on screen, not inside the host: a column at the host's edge still has one", () => {
  const [placement] = chipPlacements([{ part: "/content/0", label: "A" }], () => ({ left: 0, top: 10, room: 380 }));
  assert.equal(placement!.side, "margin", "the sheet's margin left of the column is room");
});

test("an inline target's chip stands in the room its line reserves before it, centred on its line", () => {
  const [placement] = chipPlacements([{ part: "/content/2/ref/0", label: "D" }], () => ({
    left: 300,
    top: 400,
    width: 40,
    height: 20,
    inline: true,
  }));
  assert.deepEqual(placement, { part: "/content/2/ref/0", label: "D", left: 300, top: 410, side: "inline" });
});

test("a block with no margin to stand in holds the chip inside, in the indent select mode gives it", () => {
  const [placement] = chipPlacements([{ part: "/row", label: "A" }], () => ({
    left: 0,
    top: 10,
    height: 20,
    room: CHIP_MARGIN_ROOM - 1,
  }));
  assert.deepEqual(placement, { part: "/row", label: "A", left: 0, top: 20, side: "inside" });
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

test("two margin chips on one line stand side by side, the later further out", () => {
  const placements = chipPlacements(
    [
      { part: "/content/2", label: "J" },
      { part: "/content/2/content/0", label: "K" },
    ],
    () => ({ left: 120, top: 200, width: 500, height: 30 }),
  );
  const [group, block] = placements;
  assert.equal(group!.left, 120 - CHIP_GAP);
  assert.ok(block!.left <= group!.left - estimatedChipSize("J").width, "the second clears the first");
});
