// The frame's registration and the rail: Space a shows the rail (if
// hidden) and opens the add field in it; Enter registers the
// repository whose machine:/path the human typed - its declaration
// file names the project - and the project joins the rail; nothing
// opens. d on a rail entry unregisters: the path leaves the device's
// list, nothing on the machine is touched. Space r shows and hides
// the rail, per window; no window starts with it. A window with no
// location is the empty window; the plane is no location.
import { expect, test } from "#scenario";
import { openPlane, submitLocation } from "./kit/helpers.ts";

/** The app iframe's realm: the device store lives there, and WebKit
 * partitions the test page's storage away from it. */
function frameRealm(): Window {
  const frame = document.querySelector("iframe[data-port-suite=app]") as HTMLIFrameElement | null;
  if (frame === null) throw new Error("the app iframe is not mounted");
  return frame.contentWindow as unknown as Window;
}

/** The device store is the app realm's storage (the lane seeds it per
 * launch); the registration journey here clears the seed first - the
 * empty-registry case. */
function forgetEveryProject(): void {
  frameRealm().localStorage.removeItem("saccade.projects");
}

test("Space a shows the rail, registers machine:/path, and opens nothing", async ({ app, tracker }) => {
  // the boot opens nothing (a window with no location is the empty
  // window); the seeded registration clears - the empty-registry case
  forgetEveryProject();
  // the plane is no location: the open refuses, the window stays
  await submitLocation(app, "plane");
  await app.press("Escape");
  expect(await app.shows("rail")).toBe(false);

  // Space a shows the rail with the add field; typing registers
  // through the declaration file
  await app.press(" ", "a");
  await expect.poll(() => app.shows("project-input"), { timeout: 10_000 }).toBe(true);
  await app.type(`machine:${tracker.repoPath}`);
  await app.press("Enter");
  // the project joined the rail: its entry is on screen, nothing
  // opened (the window stays the empty window)
  const after = await app.readout();
  expect(after.tasks?.project ?? null).toBe(null);
  expect(await app.windows()).toEqual(["main"]);
  // the field closes once the registration's declaration read answers
  await expect.poll(async () => (await app.readout()).railAdd?.open ?? null).toBe(false);
  await expect.poll(() => app.shows(`project-${tracker.repo}`), { timeout: 20_000 }).toBe(true);
  expect(await app.shows("empty-surface")).toBe(true);

  // Escape left no field behind: Space a again opens it, Escape leaves
  await app.press(" ", "a");
  await expect.poll(() => app.shows("project-input")).toBe(true);
  await app.press("Escape");
  await expect.poll(() => app.shows("project-input")).toBe(false);
});

test("Space r shows and hides the rail; a rail entry's d unregisters", async ({ app, tracker }) => {
  await openPlane(app, tracker);
  // the rail starts hidden: no window begins with it
  expect(await app.shows("rail")).toBe(false);
  await app.press(" ", "r");
  await expect.poll(() => app.shows("rail")).toBe(true);
  const project = `project-${tracker.repo}`;
  await expect.poll(() => app.shows(project)).toBe(true);
  // reading keys scroll the content only; the rail stays
  await app.press("j");
  await expect.poll(() => app.shows("rail")).toBe(true);

  // d on the rail entry unregisters: the path leaves the device's
  // list; the rail entry leaves; the shown list stays
  await app.press("s");
  const labels = await app.labels();
  const chip = labels.find((entry) => entry.part === `rail/project/${tracker.repo}`);
  expect(chip, JSON.stringify(labels)).toBeDefined();
  await app.type(chip!.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect(await app.menuActs()).toEqual(["f", "a", "d"]);
  await app.press("d");
  await expect.poll(() => app.shows(project), { timeout: 20_000 }).toBe(false);

  // Space r hides the rail again
  await app.press(" ", "r");
  await expect.poll(() => app.shows("rail")).toBe(false);
});
