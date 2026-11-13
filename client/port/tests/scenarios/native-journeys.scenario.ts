// D1 and S20, migrated from the t36 Python port lane
// (t36_scenario_bodies.py check_d1_unfocused_scrolling,
// check_s20_relaunch). Real windows: native only.
// D1: j/k were dead in a NEW window while gg/G worked. Open a tall
// board (its own task: the lane committed to the first row's task, a
// seed, and broke S13's revision) through the task row's Enter (one window, no link: a new
// window), switch the DRIVER to it (no native focus), press j: the new
// window's scroller moves by the line step; G still reaches the end.
// S20: the launch restore - after a relaunch into the same data
// directory, the windows that were open come back.
import { expect, test } from "#scenario";
import { citationAct, openLocation, spec, submitLocation, whenBound } from "./kit/helpers.ts";

test("D1 j scrolls a window born by the task row's Enter, with the driver in it", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  // its own task (the seeds stay as seeded for every other scenario),
  // committed before the list loads so its row is there
  const task = await tracker.createTask("D1: a board taller than a window");
  const blocks = Array.from({ length: 150 }, (_, index) => `  - text: D1 overflow block ${index + 1}\n`).join("");
  await tracker.commit(task, `grammar: 1\ntask: ${task}\ncontent:\n${blocks}`, "D1: a board taller than a window");
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null, { timeout: 20_000 }).toBe("loaded");
  const first = (await app.readout()).tasks!.rows.find((row) => row.task === task)!;
  expect(first, "the D1 task's row is listed").toBeDefined();
  await whenBound(app);
  await app.press("Escape");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  const chip = (await app.labels()).find((label) => label.part === first.part)!;
  await app.type(chip.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("Enter");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const born = (await app.windows()).find((label) => label !== "main")!;
  await app.focusWindow(born);
  await expect.poll(async () => (await app.readout(born)).board.status, { timeout: 20_000 }).toBe("loaded");
  const before = (await app.readout(born)).viewport.top;
  await app.press("j");
  await expect.poll(async () => (await app.readout(born)).viewport.top).toBe(before + 60);
  await app.press("G");
  await expect.poll(async () => (await app.readout(born)).viewport.top).toBeGreaterThan(before + 1000);
});

test("S20 a relaunch on the same data directory restores the open windows", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  await app.relaunch();
  await expect.poll(async () => (await app.windows()).length, { timeout: 40_000 }).toBe(2);
});
