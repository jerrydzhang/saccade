// The window's place across a relaunch (t-38,
// lib/navigation/window-place.ts): the launch restore brought back the
// SET of windows, each one empty - a window's state got its location
// only at birth and never the part it was reading. Now each window's
// state receives the location it shows and the part at the reading
// height once they settle, and a relaunch reopens every window there:
// a bare location bare and still following, a pinned revision pinned.
import { expect, test } from "#scenario";
import { citationAct, openPicker, spec, waitShown, whenBound } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

/** A board long enough to scroll, committed twice: answers the task
 * and the first revision (pinning it shows a revision that is not
 * current). */
async function twiceCommitted(tracker: Tracker, name: string): Promise<{ task: string; first: number }> {
  const task = await tracker.createTask(name);
  const board = (label: string) =>
    `grammar: 1\ntask: ${task}\ncontent:\n` +
    Array.from({ length: 150 }, (_, index) => `  - text: ${label} paragraph ${index + 1}\n`).join("");
  const { revision } = await tracker.commit(task, board("first"), "the first revision");
  await tracker.commit(task, board("second"), "the second revision");
  return { task, first: revision };
}

/** Open a location in the driver's window by keys alone (the picker's
 * chord, typing, Enter): it holds in a window that lost focus. */
async function openByKeys(app: AppDriver, location: string, window: string): Promise<void> {
  await openPicker(app);
  await app.type(location);
  await app.press("Enter");
  await waitShown(app, location.slice(location.indexOf("/") + 1), window);
}

/** Page down by keys until the reading height sits past the board's
 * top: answers the window's place. */
async function readFurther(app: AppDriver, window: string) {
  await app.press("Ctrl-d", "Ctrl-d");
  await expect
    .poll(async () => (await app.readout(window)).window.place?.part ?? null)
    .toMatch(/\/content\/([1-9]\d*)$/);
  return (await app.readout(window)).window.place!;
}

test("a relaunch reopens each window at its location and its reading part", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  const following = await twiceCommitted(tracker, "window place: the following board");
  const pinned = await twiceCommitted(tracker, "window place: the pinned board");

  // a second window, born by o from a citation
  await openByKeys(app, spec(tracker, tracker.tasks.citingChild), "main");
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const born = (await app.windows()).find((label) => label !== "main")!;

  // the born window (it has the focus): a pinned revision, read further
  await app.focusWindow(born);
  await whenBound(app);
  await openByKeys(app, spec(tracker, `${pinned.task}@r${pinned.first}`), born);
  const bornPlace = await readFurther(app, born);
  expect(bornPlace.location).toBe(`${tracker.repo}/${pinned.task}@r${pinned.first}`);

  // main, unfocused now: by keys, a bare location, read further
  await app.focusWindow("main");
  await openByKeys(app, spec(tracker, following.task), "main");
  const mainPlace = await readFurther(app, "main");
  expect(mainPlace.location).toBe(`${tracker.repo}/${following.task}`);

  // the places settle into the window states: a hold, then the relaunch
  await new Promise((resolve) => setTimeout(resolve, 1_500));
  await app.relaunch();
  await expect.poll(async () => (await app.windows()).length, { timeout: 40_000 }).toBe(2);

  // which relaunched window claims which state is the shell's memory's
  // order: the places are compared as a set
  const places = async () => {
    const shown = [];
    for (const label of await app.windows()) shown.push((await app.readout(label)).window.place);
    return shown.map((place) => `${place?.location ?? "(empty)"} ${place?.part ?? "(no part)"}`).sort();
  };
  await expect
    .poll(places, { timeout: 30_000 })
    .toEqual([mainPlace, bornPlace].map((place) => `${place.location} ${place.part}`).sort());

  // the bare location still follows; the pinned revision does not
  for (const label of await app.windows()) {
    const bare = (await app.readout(label)).window.place?.location === mainPlace.location;
    await expect
      .poll(async () => (await app.readout(label)).follow?.armed ?? false, { message: `${label} follows exactly when bare` })
      .toBe(bare);
  }
});
