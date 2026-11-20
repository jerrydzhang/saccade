// The rail is the frame's own pane, so its entries are selectable with
// no buffer shown: in the empty window, Space r shows the rail, s
// labels its entries, typing a label opens the entry's act menu, and
// the menu's keys run its acts - the same select session and act menu
// a shown buffer carries, by keys alone.
import { expect, test } from "./kit/browser.ts";
import type { AppDriver } from "./kit/driver.ts";
import { submitLocation } from "./kit/helpers.ts";

async function railLabel(app: AppDriver, part: string): Promise<string> {
  await app.press("s");
  await expect.poll(async () => (await app.labels()).map((chip) => chip.part)).toContain(part);
  return (await app.labels()).find((chip) => chip.part === part)!.label;
}

test("the empty window: s labels the rail's project entry; its label then Enter opens the task list", async ({ app, tracker }) => {
  // the window began empty: no location, no buffer shown
  expect(await app.shows("empty-surface")).toBe(true);
  await app.press(" ", "r");
  await expect.poll(() => app.shows("rail")).toBe(true);
  const part = `rail/project/${tracker.repo}`;
  await expect.poll(() => app.shows(`project-${tracker.repo}`), { timeout: 20_000 }).toBe(true);

  const label = await railLabel(app, part);
  await app.type(label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  // Enter runs the entry's first act: the project's claimed view
  await app.press("Enter");
  await expect
    .poll(async () => {
      const tasks = (await app.readout()).tasks;
      return { project: tasks?.project ?? null, status: tasks?.status ?? null };
    }, { timeout: 20_000 })
    .toEqual({ project: tracker.repo, status: "loaded" });
});

test("the empty window: a rail entry's act menu is reached by keys and runs its acts", async ({ app, tracker }) => {
  expect(await app.shows("empty-surface")).toBe(true);
  await app.press(" ", "r");
  await expect.poll(() => app.shows("rail")).toBe(true);
  const part = `rail/project/${tracker.repo}`;
  await expect.poll(() => app.shows(`project-${tracker.repo}`), { timeout: 20_000 }).toBe(true);

  const label = await railLabel(app, part);
  await app.type(label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect(await app.menuActs()).toEqual(["f", "a", "d"]);
  // the register act opens the rail's add field; the window stays empty
  await app.press("a");
  await expect.poll(() => app.shows("project-input")).toBe(true);
  expect(await app.menuOpen()).toBe(false);
  expect(await app.shows("empty-surface")).toBe(true);
  await app.press("Escape");
  await expect.poll(() => app.shows("project-input")).toBe(false);

  // Escape on an open menu closes it, and s labels again afterwards
  const again = await railLabel(app, part);
  await app.type(again.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("Escape");
  await expect.poll(() => app.menuOpen()).toBe(false);
  // the primary act key runs from the menu: the claimed view opens
  const third = await railLabel(app, part);
  await app.type(third.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("f");
  await expect
    .poll(async () => (await app.readout()).tasks?.project ?? null, { timeout: 20_000 })
    .toBe(tracker.repo);
});

test("the empty window: select left open closes when a buffer shows; the next bufferless window starts without labels", async ({ app, tracker }) => {
  expect(await app.shows("empty-surface")).toBe(true);
  await app.press(" ", "r");
  await expect.poll(() => app.shows("rail")).toBe(true);
  const part = `rail/project/${tracker.repo}`;
  await expect.poll(() => app.shows(`project-${tracker.repo}`), { timeout: 20_000 }).toBe(true);
  await railLabel(app, part);

  // with the rail's labels still up (no Escape: it would leave select
  // mode), the picker opens the task list
  await app.press(" ", " ");
  await expect.poll(() => app.shows("picker-input")).toBe(true);
  expect((await app.labels()).map((chip) => chip.part)).toContain(part);
  await app.type(`${tracker.repo}/tasks`);
  await app.press("Enter");
  await expect
    .poll(async () => (await app.readout()).tasks?.status ?? null, { timeout: 20_000 })
    .toBe("loaded");
  expect((await app.labels()).filter((chip) => chip.part.startsWith("rail/"))).toEqual([]);

  // a board that cannot load draws no buffer: the rail rides the
  // frame's own session again, and no labels linger from before
  await submitLocation(app, `${tracker.repo}/t-9999`);
  await expect.poll(async () => (await app.readout()).board.status, { timeout: 20_000 }).toBe("unavailable");
  await expect.poll(() => app.shows(`project-${tracker.repo}`)).toBe(true);
  expect(await app.labels()).toEqual([]);
  expect(await app.menuOpen()).toBe(false);
  const label = await railLabel(app, part);
  await app.type(label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect(await app.menuActs()).toEqual(["f", "a", "d"]);
});
