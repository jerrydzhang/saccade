// S05 and S07, migrated from the t36 Python port lane
// (t36_scenario_bodies.py check_s05_temporary_buffers,
// check_s07_return_place). S05: the picker is a temporary surface - an
// opening through it adds exactly one trail entry, the picker itself is
// never a destination, and Back lands on the first board at the place
// it was left. S07: Back restores the passage (the addressed place AND
// the reading position) by moving the cursor, never the trail; Forward
// retraces. Back/Forward are the pinned Ctrl-o / Ctrl-i.
import { expect, test } from "#scenario";
import { longBoard, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

// a trail return shows the pinned revision it recorded (t-N@rK)
async function backOn(app: AppDriver, path: string): Promise<void> {
  await expect
    .poll(async () => {
      const board = (await app.readout()).board;
      return board.status === "loaded" && (board.location ?? "").startsWith(`${path}@r`);
    })
    .toBe(true);
}

async function readTallBoardAtItsEnd(app: AppDriver, tracker: Tracker): Promise<{ path: string; left: number; depth: number }> {
  const task = await longBoard(tracker);
  const shown = await openLocation(app, spec(tracker, task));
  await app.press("Escape");
  await app.press("G");
  await expect.poll(async () => (await app.readout()).viewport.top).toBeGreaterThan(100);
  // settled: two reads agree
  let left = (await app.readout()).viewport.top;
  await expect.poll(async () => {
    const now = (await app.readout()).viewport.top;
    const same = Math.abs(now - left) <= 1;
    left = now;
    return same;
  }).toBe(true);
  return { path: `${tracker.repo}/${task}`, left, depth: shown.trail.depth };
}

test("S05 the picker is never a trail destination; Back returns to the place left", { tags: ["native"] }, async ({ app, tracker }) => {
  const first = await readTallBoardAtItsEnd(app, tracker);
  await openLocation(app, spec(tracker, tracker.tasks.cited));
  const trail = (await app.readout()).trail;
  expect(trail.depth, "the picker journey adds exactly one entry").toBe(first.depth + 1);
  expect(trail.entries.map((entry) => entry.location).filter((location) => /picker/i.test(location))).toEqual([]);
  await app.press("Ctrl-o");
  await backOn(app, first.path);
  await expect.poll(async () => Math.abs((await app.readout()).viewport.top - first.left)).toBeLessThanOrEqual(3);
});

test("S07 Back restores the passage by moving the cursor, not the trail; Forward retraces", { tags: ["native"] }, async ({ app, tracker }) => {
  const first = await readTallBoardAtItsEnd(app, tracker);
  const landing = `${tracker.repo}/${tracker.tasks.landing}`;
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  const depth = (await app.readout()).trail.depth;
  expect(depth).toBeGreaterThanOrEqual(2);
  await app.press("Ctrl-o");
  await backOn(app, first.path);
  await expect.poll(async () => Math.abs((await app.readout()).viewport.top - first.left)).toBeLessThanOrEqual(3);
  expect((await app.readout()).trail).toMatchObject({ depth, cursor: depth - 2 });
  await app.press("Ctrl-i");
  await backOn(app, landing);
  expect((await app.readout()).trail).toMatchObject({ depth, cursor: depth - 1 });
});
