// The shared label overlay's placement model (order of work step
// 3). Pure math: host-relative placement, no chip for unmeasurable
// parts, the visibility contract F3 violated.

import assert from "node:assert/strict";
import test from "node:test";

import { chipPlacements, isWithinHost } from "../src/lib/select/overlay.ts";

test("chips place at their part's host-relative box", () => {
  const placements = chipPlacements(
    [
      { part: "/content/0", label: "A" },
      { part: "/content/3", label: "S" },
    ],
    (part) =>
      part === "/content/0" ? { left: 12, top: 340 } : { left: 48, top: 11279 },
  );
  assert.deepEqual(placements, [
    { part: "/content/0", label: "A", left: 12, top: 340 },
    { part: "/content/3", label: "S", left: 48, top: 11279 },
  ]);
});

test("an unmeasurable part gets no chip - a label over nothing is a lie", () => {
  const placements = chipPlacements(
    [
      { part: "/gone", label: "A" },
      { part: "/here", label: "S" },
    ],
    (part) => (part === "/here" ? { left: 0, top: 10 } : null),
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
