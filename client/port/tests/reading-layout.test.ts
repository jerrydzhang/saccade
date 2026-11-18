// The board's reading layout (lib/board/reading-layout.ts): the steps
// stay on whole hundredths and inside their bounds, the resets touch
// only their own scale, and the holder applies each step at once -
// per window, for the run, written nowhere (t-71@r3).


import assert from "node:assert/strict";
import { test } from "vitest";
import {
  changeReadingLayout,
  createReadingLayoutHolder,
  DEFAULT_READING_LAYOUT,
  readingLayoutVars,
  WIDTH_MAX,
  ZOOM_MIN,
  type ReadingLayoutChange,
} from "../src/lib/board/reading-layout.ts";

function run(...changes: ReadingLayoutChange[]) {
  return changes.reduce(changeReadingLayout, DEFAULT_READING_LAYOUT);
}

test("steps land on whole tenths and each scale moves alone", () => {
  assert.deepEqual(run("zoom-in", "zoom-in", "zoom-in"), { zoom: 1.3, width: 1 });
  assert.deepEqual(run("widen", "widen", "narrow"), { zoom: 1, width: 1.1 });
  assert.deepEqual(run("zoom-in", "widen", "zoom-reset"), { zoom: 1, width: 1.1 });
  assert.deepEqual(run("zoom-in", "widen", "width-reset"), { zoom: 1.1, width: 1 });
});

test("the scales stop at their bounds", () => {
  assert.equal(run(...Array<ReadingLayoutChange>(20).fill("zoom-out")).zoom, ZOOM_MIN);
  assert.equal(run(...Array<ReadingLayoutChange>(40).fill("widen")).width, WIDTH_MAX);
});

test("the CSS custom properties the stylesheet reads", () => {
  assert.deepEqual(readingLayoutVars({ zoom: 1.2, width: 0.8 }), {
    "--port-reading-zoom": "1.2",
    "--port-reading-width": "0.8",
  });
});

function holder() {
  const applied: Record<string, string>[] = [];
  const reading = createReadingLayoutHolder({
    apply: (vars) => {
      applied.push(vars);
    },
  });
  return { reading, applied };
}

test("the holder applies each step at once, from the defaults", () => {
  const h = holder();
  assert.deepEqual(h.reading.current(), DEFAULT_READING_LAYOUT, "a fresh window starts at the defaults");
  h.reading.change("zoom-in");
  h.reading.change("widen");
  assert.deepEqual(h.applied.at(-1), { "--port-reading-zoom": "1.1", "--port-reading-width": "1.1" });
  assert.deepEqual(h.reading.current(), { zoom: 1.1, width: 1.1 });
});
