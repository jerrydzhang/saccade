// The control plane (t-48): the task list carries a rail of the
// projects this device has reached, the shown one marked; choosing a
// project switches this window in place - the list it shows, never a
// new window. Rail entries are the list's own targets: s labels them
// with the rows, and a label, a click or Enter runs the same act. Agents join from Herdr core's words, which the lane's
// sandbox may not run; their grouping is pinned by the unit tests
// (tests/tasks-rail.test.ts).
import { expect, test } from "#scenario";
import { spec, submitLocation } from "./kit/helpers.ts";

test("the rail lists the reached project and switches this window to its tasks", async ({ app, tracker }) => {
  await submitLocation(app, spec(tracker, tracker.tasks.cited));
  await expect.poll(async () => (await app.readout()).board.status, { timeout: 20_000 }).toBe("loaded");
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  const project = `project-${tracker.repo}`;
  await expect.poll(() => app.shows(project)).toBe(true);
  expect(await app.controlText(project)).toBe(tracker.repo);

  await app.click(project);
  await expect.poll(async () => (await app.readout()).tasks?.location ?? null).toBe(`${tracker.repo}/tasks`);
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
  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).tasks?.location ?? null).toBe(`${tracker.repo}/tasks`);
  expect(await app.windows()).toEqual(["main"]);
});
