// The board's reading layout (lib/board/reading-layout.ts): the steps
// stay on whole hundredths and inside their bounds, the resets touch
// only their own scale, a stored layout never refuses, and the
// window's holder writes once the steps settle.

import assert from "node:assert/strict";
import { test } from "vitest";
import {
  changeReadingLayout,
  createReadingLayoutHolder,
  DEFAULT_READING_LAYOUT,
  readingLayoutOf,
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

test("a stored layout falls back per field, never refuses", () => {
  assert.deepEqual(readingLayoutOf({}), DEFAULT_READING_LAYOUT);
  assert.deepEqual(readingLayoutOf({ zoom: 1.4, width: null }), { zoom: 1.4, width: 1 });
  assert.deepEqual(readingLayoutOf({ zoom: 99, width: Number.NaN }), { zoom: 2, width: 1 });
});

test("the CSS custom properties the stylesheet reads", () => {
  assert.deepEqual(readingLayoutVars({ zoom: 1.2, width: 0.8 }), {
    "--port-reading-zoom": "1.2",
    "--port-reading-width": "0.8",
  });
});

function holder() {
  const applied: Record<string, string>[] = [];
  const written: { zoom: number; width: number }[] = [];
  let job: (() => void) | null = null;
  const reading = createReadingLayoutHolder({
    apply: (vars) => {
      applied.push(vars);
    },
    write: (layout) => {
      written.push(layout);
    },
    schedule: (_ms, run) => {
      job = run;
      return 1;
    },
    cancel: () => {
      job = null;
    },
  });
  return {
    reading,
    applied,
    written,
    settle: () => {
      const run = job;
      job = null;
      run?.();
    },
  };
}

test("the holder applies each step at once and writes once the steps settle", () => {
  const h = holder();
  h.reading.change("zoom-in");
  h.reading.change("widen");
  assert.deepEqual(h.applied.at(-1), { "--port-reading-zoom": "1.1", "--port-reading-width": "1.1" });
  assert.deepEqual(h.written, [], "no write per key press");
  h.settle();
  assert.deepEqual(h.written, [{ zoom: 1.1, width: 1.1 }]);
});

test("a bound state's stored layout applies, unless this window changed its own", () => {
  const restored = holder();
  restored.reading.bound({ zoom: 1.3, width: null });
  assert.deepEqual(restored.reading.current(), { zoom: 1.3, width: 1 });
  restored.settle();
  assert.deepEqual(restored.written, [], "restoring writes nothing back");

  const local = holder();
  local.reading.change("zoom-in");
  local.reading.bound({ zoom: 1.6, width: 2 });
  assert.deepEqual(local.reading.current(), { zoom: 1.1, width: 1 }, "the window's own change wins");
  local.settle();
  assert.deepEqual(local.written, [{ zoom: 1.1, width: 1 }]);
});
