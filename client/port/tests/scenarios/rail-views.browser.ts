// The rail keeps views: Space k keeps the location the window shows
// beneath its repository, as an ordinary entry that opens it again by
// one click or one select label - in the window that kept it, in a
// relaunched window that began empty. The repository's own file may
// offer views too (kit/tracker.ts seeds two: one readable, one not);
// those show on a device that kept nothing, have no forget act, and
// an unreadable one is named without losing the rest. The shown mark
// sits on a view exactly when the window shows that location.
import { afterEach } from "vitest";

import { expect, test } from "./kit/browser.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";
import { openPlane, submitLocation } from "./kit/helpers.ts";

/** The app iframe's realm: the device store lives there. */
function frameRealm(): Window {
  const frame = document.querySelector("iframe[data-port-suite=app]") as HTMLIFrameElement | null;
  if (frame === null) throw new Error("the app iframe is not mounted");
  return frame.contentWindow as unknown as Window;
}

const viewsStore = (): Record<string, string[]> =>
  JSON.parse(frameRealm().localStorage.getItem("saccade.views") ?? "{}") as Record<string, string[]>;

/** The device's kept views start empty and leave none behind: the
 * iframe's storage outlives one app, and a view kept here would show
 * on every later scenario's rail. */
function startWithNoViews(): void {
  frameRealm().localStorage.removeItem("saccade.views");
}
afterEach(() => frameRealm().localStorage.removeItem("saccade.views"));

const control = (tracker: Tracker, path: string): string => `rail-view-${tracker.repo}/${path}`;
const part = (tracker: Tracker, path: string): string => `rail/view/${tracker.repo}/${path}`;

async function tasksAsked(app: AppDriver): Promise<string | null> {
  const tasks = (await app.readout()).tasks;
  return tasks !== null && tasks !== undefined && tasks.status === "loaded" ? tasks.asked : null;
}

async function showRail(app: AppDriver): Promise<void> {
  await app.press(" ", "r");
  await expect.poll(() => app.shows("rail")).toBe(true);
}

/** Select mode, the entry's label typed: its act menu opens. */
async function menuOf(app: AppDriver, partId: string): Promise<void> {
  await app.press("s");
  await expect.poll(async () => (await app.labels()).map((chip) => chip.part)).toContain(partId);
  const chip = (await app.labels()).find((entry) => entry.part === partId)!;
  await app.type(chip.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
}

/** The window as a relaunch leaves it: the same device storage, a
 * fresh app that begins empty. */
async function relaunch(app: AppDriver): Promise<void> {
  frameRealm().location.reload();
  await expect
    .poll(async () => {
      try {
        return await app.shows("empty-surface");
      } catch {
        return false;
      }
    }, { timeout: 20_000 })
    .toBe(true);
  await app.click("content");
}

test("Space k keeps the shown narrowed list beneath its repository; one click opens it again; the mark follows the window", async ({ app, tracker }) => {
  startWithNoViews();
  await tracker.createTag("vwapp", "the views scenario's tag");
  const tagged = await tracker.createTask("views: tagged for the kept list");
  await tracker.tagTask(tagged, "vwapp");
  const narrowed = `${tracker.repo}/tasks?tag=vwapp`;
  await submitLocation(app, `${tracker.repo}/tasks?tag=VWAPP`);
  await expect.poll(() => tasksAsked(app), { timeout: 20_000 }).toBe(narrowed);
  await showRail(app);
  const entry = control(tracker, "tasks?tag=vwapp");
  expect(await app.shows(entry)).toBe(false);

  await app.press(" ", "k");
  await expect.poll(() => app.shows(entry)).toBe(true);
  expect((await app.strip()).note).toBe(`kept ${narrowed} beneath ${tracker.repo}`);
  expect(await app.controlText(entry)).toBe("vwapp");
  expect(await app.shownMark(entry)).toBe(true);
  expect(viewsStore()).toEqual({ [await app.seededPath()]: ["tasks?tag=vwapp"] });

  // the whole list is another location: the mark leaves the view
  await openPlane(app, tracker);
  await expect.poll(() => app.shownMark(entry)).toBe(false);
  // one click opens the kept list, and the mark returns
  await app.click(entry);
  await expect.poll(() => tasksAsked(app), { timeout: 20_000 }).toBe(narrowed);
  await expect.poll(() => app.shownMark(entry)).toBe(true);
});

test("keeping a kept location again says so and adds nothing", async ({ app, tracker }) => {
  startWithNoViews();
  await tracker.createTag("vwtwice", "the keep-twice scenario's tag");
  await submitLocation(app, `${tracker.repo}/tasks?tag=vwtwice`);
  await expect.poll(() => tasksAsked(app), { timeout: 20_000 }).toBe(`${tracker.repo}/tasks?tag=vwtwice`);
  await showRail(app);
  await app.press(" ", "k");
  await expect.poll(() => app.shows(control(tracker, "tasks?tag=vwtwice"))).toBe(true);
  await app.press(" ", "k");
  await expect.poll(async () => (await app.strip()).note).toBe(`already kept: ${tracker.repo}/tasks?tag=vwtwice`);
  expect(Object.values(viewsStore())).toEqual([["tasks?tag=vwtwice"]]);
  await app.press("s");
  const parts = (await app.labels()).map((chip) => chip.part).filter((id) => id === part(tracker, "tasks?tag=vwtwice"));
  expect(parts).toHaveLength(1);
  await app.press("Escape");
});

test("a window that has nothing to keep says why", async ({ app }) => {
  startWithNoViews();
  expect(await app.shows("empty-surface")).toBe(true);
  await app.press(" ", "k");
  await expect.poll(async () => (await app.strip()).note).toMatch(/shows no location/);
  expect(await app.shows("rail-view-problem")).toBe(false);
});

test("a kept view survives a relaunch and is reached in the empty window by one select label, then forgotten by keys alone", async ({ app, tracker }) => {
  startWithNoViews();
  await tracker.createTag("vwkeep", "the relaunch scenario's tag");
  const tagged = await tracker.createTask("views: tagged for the relaunch");
  await tracker.tagTask(tagged, "vwkeep");
  const narrowed = `${tracker.repo}/tasks?tag=vwkeep`;
  await submitLocation(app, narrowed);
  await expect.poll(() => tasksAsked(app), { timeout: 20_000 }).toBe(narrowed);
  await app.press(" ", "k");
  await expect.poll(async () => (await app.strip()).note).toBe(`kept ${narrowed} beneath ${tracker.repo}`);

  await relaunch(app);
  expect(Object.values(viewsStore())).toEqual([["tasks?tag=vwkeep"]]);
  await showRail(app);
  const entry = control(tracker, "tasks?tag=vwkeep");
  await expect.poll(() => app.shows(entry), { timeout: 20_000 }).toBe(true);
  // the empty window's own rail session: its label opens the act menu,
  // and f shows the list here
  await menuOf(app, part(tracker, "tasks?tag=vwkeep"));
  expect(await app.menuActs()).toEqual(["f", "o", "i", "w", "d"]);
  await app.press("f");
  await expect.poll(() => tasksAsked(app), { timeout: 20_000 }).toBe(narrowed);
  await expect.poll(() => app.shownMark(entry)).toBe(true);

  // forget it by keys alone: s, its label, d
  await menuOf(app, part(tracker, "tasks?tag=vwkeep"));
  await app.press("d");
  await expect.poll(() => app.shows(entry)).toBe(false);
  expect(viewsStore()).toEqual({});
});

test("the file's views show on a device that kept nothing; a file view has no forget act; an unreadable one is named and the rest stand", async ({ app, tracker }) => {
  startWithNoViews();
  await tracker.createTag("vwfile", "the file's view");
  expect(viewsStore()).toEqual({});
  await showRail(app);
  const entry = control(tracker, "tasks?tag=vwfile");
  await expect.poll(() => app.shows(entry), { timeout: 20_000 }).toBe(true);
  expect(await app.controlText(entry)).toBe("vwfile");
  await expect.poll(() => app.controlText("rail-view-problem")).toMatch(
    /^workbench\.toml view 'tasks\?tag=' cannot be read: tag= names no tag/,
  );

  await menuOf(app, part(tracker, "tasks?tag=vwfile"));
  expect(await app.menuActs()).toEqual(["f", "o", "i", "w"]);
  expect(await app.menuLabel()).toContain("offered by workbench.toml");
  await app.press("Escape");
  // choosing it opens it; the mark sits on it while it shows
  await app.click(entry);
  await expect.poll(() => tasksAsked(app), { timeout: 20_000 }).toBe(`${tracker.repo}/tasks?tag=vwfile`);
  await expect.poll(() => app.shownMark(entry)).toBe(true);
});

test("unregistering the repository forgets the views the device kept beneath it", async ({ app, tracker }) => {
  startWithNoViews();
  await tracker.createTag("vwgone", "the unregister scenario's tag");
  await submitLocation(app, `${tracker.repo}/tasks?tag=vwgone`);
  await expect.poll(() => tasksAsked(app), { timeout: 20_000 }).toBe(`${tracker.repo}/tasks?tag=vwgone`);
  await showRail(app);
  await app.press(" ", "k");
  await expect.poll(() => app.shows(control(tracker, "tasks?tag=vwgone"))).toBe(true);
  expect(Object.values(viewsStore())).toEqual([["tasks?tag=vwgone"]]);
  await menuOf(app, `rail/project/${tracker.repo}`);
  await app.press("d");
  await expect.poll(() => app.shows(`project-${tracker.repo}`)).toBe(false);
  expect(viewsStore()).toEqual({});
});
