// The act menu's placement rules (M3): pointer-anchored,
// beside-the-target with the flip, clamped. Pure math tests.

import assert from "node:assert/strict";
import { test } from "vitest";

import { menuPlacement } from "../src/lib/menu/geometry.ts";

const viewport = { width: 800, height: 600 };
const menu = { width: 200, height: 120 };

test("a right-click point anchors the menu there", () => {
  const at = menuPlacement({ at: { x: 300, y: 220 }, target: null, menu, viewport });
  assert.deepEqual(at, { x: 300, y: 220 });
});

test("without a point, the menu sits beside its target, below", () => {
  const at = menuPlacement({
    at: null,
    target: { left: 160, top: 400, bottom: 440 },
    menu,
    viewport,
  });
  assert.deepEqual(at, { x: 170, y: 444 }, "left+10, bottom+4");
});

test("no room below flips the menu above the target", () => {
  const at = menuPlacement({
    at: null,
    target: { left: 160, top: 520, bottom: 560 },
    menu,
    viewport,
  });
  // below would be 564 + 120 + 4 > 600: flip to top - 4 - height
  assert.deepEqual(at, { x: 170, y: 396 });
});

test("the clamp keeps the menu inside the viewport", () => {
  const at = menuPlacement({ at: { x: 790, y: 590 }, target: null, menu, viewport });
  assert.deepEqual(at, { x: 596, y: 476 }, "4px margins on the far edges");
  const negative = menuPlacement({ at: { x: -50, y: -50 }, target: null, menu, viewport });
  assert.deepEqual(negative, { x: 4, y: 4 }, "never left of or above the viewport");
});

test("no anchor at all lands at the clamped corner", () => {
  const at = menuPlacement({ at: null, target: null, menu, viewport });
  assert.deepEqual(at, { x: 4, y: 4 });
});
