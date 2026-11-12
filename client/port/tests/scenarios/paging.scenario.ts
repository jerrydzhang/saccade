// PENDING (batch 3, red before the code): the paging steps, verified
// against the prototype (inst16.html:495 Ctrl f/b 0.9 and d/u 0.5 of the
// body's clientHeight, :504 j/k 60px, :505 d/u 0.5), the same bindings
// on boards and lists, ignored inside inputs (inst16.html:489). The port
// today pages "full" at 1.0 of the height (src/components/Buffer.ts
// scrollPage) and steps lines at 40px; Ctrl chords other than o/i pass
// native (src/App.ts). Read through the readout's viewport.
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
    await expect.poll(() => top(app), { message: `${key} moves ${step(height)}px` }).toBeCloseTo(before + step(height), 0);
  }
}

test("boards page by the prototype's steps", { tags: ["pending"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await app.press("Escape");
  await stepsMatch(app);
});

test("lists page by the same steps", { tags: ["pending"] }, async ({ app, tracker }) => {
  for (let index = 0; index < 60; index++) await tracker.createTask(`paging row ${index + 1}`);
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  await app.press("Escape");
  await stepsMatch(app);
});

test("paging keys inside an input are text, not paging (holds today: a guard, not pending)", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await find(app, "par");
  const before = await top(app);
  await app.type("dju");
  await app.press("Ctrl-d");
  await expect.poll(async () => (await app.readout()).find.query).toBe("pardju");
  expect(await top(app)).toBe(before);
});
