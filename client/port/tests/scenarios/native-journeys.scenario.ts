// D1 and S20, migrated from the t36 Python port lane
// (t36_scenario_bodies.py check_d1_unfocused_scrolling,
// check_s20_relaunch). Real windows: native only.
// D1: j/k were dead in a NEW window while gg/G worked. Open a tall
// board (its own task: the lane committed to the first row's task, a
// seed, and broke S13's revision) through the task row's Enter (one window, no link: a new
// window), switch the DRIVER to it (no native focus), press j: the new
// window's scroller moves by the line step; G still reaches the end.
// S20: the launch (t-71@r3) - a relaunch into the same data directory
// opens one fresh window; no window, place, zoom, width or link comes
// back, and nothing is read from any file.
import { expect, test } from "#scenario";
import { citationAct, openLocation, spec, whenBound, openPlane } from "./kit/helpers.ts";

test("D1 j scrolls a window born by the task row's Enter, with the driver in it", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  // its own task (the seeds stay as seeded for every other scenario),
  // committed before the list loads so its row is there
  const task = await tracker.createTask("D1: a board taller than a window");
  const blocks = Array.from({ length: 150 }, (_, index) => `  - text: D1 overflow block ${index + 1}\n`).join("");
  await tracker.commit(task, `grammar: 1\ntask: ${task}\ncontent:\n${blocks}`, "D1: a board taller than a window");
  await openPlane(app, tracker);
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null, { timeout: 20_000 }).toBe("loaded");
  const first = (await app.readout()).tasks!.rows.find((row) => row.task === task)!;
  expect(first, "the D1 task's row is listed").toBeDefined();
  await whenBound(app);
  await app.press("Escape");
  // labels name what is on screen: page the list (d, half a view) until
  // the row is in view - earlier scenarios' tasks can push it past the fold
  const inView = async (): Promise<boolean> => {
    const box = await app.partBox(first.part);
    return box !== null && box.top >= 0 && box.bottom <= (await app.readout()).viewport.height;
  };
  for (let page = 0; page < 20 && !(await inView()); page++) await app.press("d");
  expect(await inView(), "the D1 row is on screen").toBe(true);
  await app.press("s");
  // select mode measures twice (it reserves inline room, then places):
  // wait for the row's own chip, not merely for some chip
  await expect.poll(async () => (await app.labels()).some((label) => label.part === first.part)).toBe(true);
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

test("S20 a relaunch on the same data directory opens one fresh window", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  // the run's arrangement: a second window born by o, its link set
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  await app.relaunch();
  // the ratified truth: exactly the launch window comes back, fresh -
  // the elsewhere surface ready, no link restored, nothing else open
  await expect.poll(async () => (await app.windows()).length, { timeout: 40_000 }).toBe(1);
  await expect.poll(async () => (await app.readout()).window.bound, { timeout: 20_000 }).toBe(true);
  const obs = await app.readout();
  expect(obs.window).toMatchObject({ label: "main", link: null });
  expect((await app.strip()).link).toBe("unlinked");
  // fresh means the control plane: the launch named no place and no
  // test-open rides a relaunch; the plane loads its list from the
  // tracker the run already reached (the data directory rides the
  // relaunch), and the board surface stays empty
  await expect.poll(async () => (await app.readout()).surface).toBe("tasks");
  await expect
    .poll(async () => {
      const obs = await app.readout();
      return { status: obs.tasks?.status ?? null, asked: obs.board.asked };
    })
    .toEqual({ status: "loaded", asked: null });
});
