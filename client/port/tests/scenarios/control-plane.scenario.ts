// The control plane: its own address, "plane", which
// names no tracker - a window with nothing opened shows it, and which
// project it shows lives in the session, never in the window's place. The
// task list carries a rail of the projects this device has registered,
// the shown one marked; choosing a project switches this window in
// place - the list it shows, never a new window. Registering a project
// is its own act: a opens the add field, Enter registers the repository
// at machine:/path (its declaration file names the project) and opens
// nothing. Rail entries are the list's own targets: s labels
// them with the rows, and a label, a click or Enter runs the same act.
// Agents join from Herdr core's words, which the lane's sandbox may not
// run; their grouping is pinned by the unit tests
// (tests/tasks-rail.test.ts).
import { expect, test } from "#scenario";
import { openPlane, submitLocation } from "./kit/helpers.ts";

/** The device store is this origin's storage (the lane seeds it per
 * launch); this scenario drives the real add field, so it clears the
 * seed first - the empty-registry case. */
function forgetEveryProject(): void {
  localStorage.removeItem("saccade.projects");
}

test("a window with nothing opened shows the plane; with no project it says how to add one, and a adds one opening nothing", async ({ app, tracker }) => {
  // the boot opens the plane: its location names no tracker
  await expect.poll(async () => (await app.readout()).tasks?.asked ?? null, { timeout: 20_000 }).toBe("plane");
  expect((await app.readout()).surface).toBe("tasks");

  forgetEveryProject();
  await submitLocation(app, "plane");
  await expect
    .poll(async () => {
      const tasks = (await app.readout()).tasks;
      return { project: tasks === null ? "(no plane)" : tasks.project, status: tasks?.status ?? null };
    })
    .toEqual({ project: null, status: "loaded" });
  // the rail's own snapshot refreshes on its poll, after the frame
  await expect.poll(() => app.shows("plane-empty"), { timeout: 20_000 }).toBe(true);
  expect(await app.controlText("plane-empty")).toContain("press a");
  const depth = (await app.readout()).trail.depth;

  // a refused address keeps the field and says why
  await app.press("a");
  await expect.poll(() => app.shows("project-input")).toBe(true);
  await app.type("nothing");
  await app.press("Enter");
  await expect.poll(() => app.shows("project-input-problem")).toBe(true);
  await app.press("Escape");
  await expect.poll(() => app.shows("project-input")).toBe(false);

  // the real registration: the repository's machine and path; the
  // declaration file there names the project
  await app.press("a");
  await expect.poll(() => app.shows("project-input")).toBe(true);
  await app.type(`machine:${tracker.repoPath}`);
  expect((await app.readout()).tasks?.add.value).toBe(`machine:${tracker.repoPath}`);
  await app.press("Enter");
  await expect.poll(() => app.shows("project-input")).toBe(false);
  const project = `project-${tracker.repo}`;
  await expect.poll(() => app.shows(project), { timeout: 20_000 }).toBe(true);
  // registering opened nothing: no project shows yet, no trail entry,
  // no window
  const after = await app.readout();
  expect(after.tasks?.project).toBe(null);
  expect(after.trail.depth).toBe(depth);
  expect(await app.windows()).toEqual(["main"]);
});

test("the rail lists the reached project and switches this window to its tasks", async ({ app, tracker }) => {
  await openPlane(app, tracker);
  const project = `project-${tracker.repo}`;
  await expect.poll(() => app.shows(project)).toBe(true);
  expect(await app.controlText(project)).toBe(tracker.repo);

  await app.click(project);
  await expect.poll(async () => (await app.readout()).tasks?.project ?? null).toBe(tracker.repo);
  const shown = await app.readout();
  expect(shown.tasks?.location).toBe("plane");
  // the plane remembers no project: its place is the bare plane
  await expect.poll(async () => (await app.readout()).window.place).toEqual({ location: "plane", part: null });
  expect(await app.windows()).toEqual(["main"]);

  // the rail is the list's own selection: s labels the project with
  // the rows, its label opens the act menu, and Enter switches here
  await app.press("s");
  const labels = await app.labels();
  const chip = labels.find((entry) => entry.part === `rail/project/${tracker.repo}`);
  expect(chip, JSON.stringify(labels)).toBeDefined();
  expect(labels.some((entry) => entry.part.startsWith("/"))).toBe(true);
  await app.type(chip!.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect(await app.menuActs()).toEqual(["f", "a", "d"]);
  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).tasks?.project ?? null).toBe(tracker.repo);
  expect(await app.windows()).toEqual(["main"]);
});
