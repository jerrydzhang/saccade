// D5, the paging steps, verified against the prototype (inst16.html:495
// Ctrl f/b 0.9 and d/u 0.5 of the body's clientHeight, :504 j/k 60px,
// :505 d/u 0.5), the same bindings on boards and lists, ignored inside
// inputs (inst16.html:489). Written red before the code: the port paged
// at 1.0 and stepped 40px, and every Ctrl chord but o/i passed native
// (src/App.ts; now contexts.ts PINNED_CHORDS). Read through the
// readout's viewport.
import { expect, test } from "#scenario";
import { find, longBoard, openLocation, spec, submitLocation } from "./kit/helpers.ts";
import type { AppDriver } from "./kit/driver.ts";

const STEPS: [string, (height: number) => number][] = [
  ["j", () => 60],
  ["k", () => -60],
  ["d", (h) => 0.5 * h],
  ["u", (h) => -0.5 * h],
  ["Ctrl-f", (h) => 0.9 * h],
  ["Ctrl-b", (h) => -0.9 * h],
  ["Ctrl-d", (h) => 0.5 * h],
  ["Ctrl-u", (h) => -0.5 * h],
];

async function top(app: AppDriver): Promise<number> {
  return (await app.readout()).viewport.top;
}

// from a mid-page start, so every step has room in both directions
async function stepsMatch(app: AppDriver): Promise<void> {
  const height = (await app.readout()).viewport.height;
  expect(height).toBeGreaterThan(0);
  await app.press("Ctrl-f", "Ctrl-f");
  await app.press("j", "j", "j");
  for (const [key, step] of STEPS) {
    const before = await top(app);
    await app.press(key);
    // within 1px: scrollTop lands on whole (device) pixels while a
    // fractional step (0.5 or 0.9 of an odd height) does not, so the
    // exact target is unreachable by a pixel's fraction; the app
    // never rounds
    await expect
      .poll(async () => Math.abs((await top(app)) - (before + step(height))), { message: `${key} moves ${step(height)}px` })
      .toBeLessThanOrEqual(1);
  }
}

test("boards page by the prototype's steps", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await app.press("Escape");
  await stepsMatch(app);
});

test("lists page by the same steps", async ({ app, tracker }) => {
  // 150 rows: the steps start mid-page (two Ctrl-f and three j, about
  // 2h) and still need a Ctrl-f of room; 60 rows scrolled only ~1.4h
  // on a 775px viewport, so the setup ended at the bottom
  for (let index = 0; index < 150; index++) await tracker.createTask(`paging row ${index + 1}`);
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  await app.press("Escape");
  await stepsMatch(app);
});

test("paging keys inside an input are text, not paging (a guard: it held before the paging landed)", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await find(app, "par");
  const before = await top(app);
  await app.type("dju");
  await app.press("Ctrl-d");
  await expect.poll(async () => (await app.readout()).find.query).toBe("pardju");
  expect(await top(app)).toBe(before);
});
