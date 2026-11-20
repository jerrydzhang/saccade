// S08 and S13, migrated from the t36 Python port lane
// (t36_differential.py _s08_port, t36_scenario_bodies.py _s13_port).
// The seeds: the cited board's group (/content/0) holds two children;
// the second citing board's incorporation names the FIRST child
// (/content/0/group/content/0).
// S08: navigation reveals its actual anchor through folds - fold the
// group on the cited board, then follow the citation from the citing
// board: the cited child renders. S13: a citation's copy names the
// shown revision and the child's full canonical path, never the
// parent; the group's own copy names its own full path.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";

const GROUP = "/content/0";
const FIRST_CHILD = "/content/0/content/0";

test("S08 a citation's navigation reveals its anchor through a fold", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.cited));
  await app.press("Escape");
  await app.clickPart(GROUP, { button: "right", at: "header" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("z");
  await expect.poll(() => app.partBox(FIRST_CHILD), { message: "the fold premise: the child stops showing" }).toBeNull();
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await app.click("citation");
  await expect
    .poll(async () => (await app.readout()).board.location)
    .toBe(`${tracker.repo}/${tracker.tasks.cited}@r1`);
  await expect.poll(() => app.partBox(FIRST_CHILD), { message: "the anchor shows: the fold opened for it" }).not.toBeNull();
});

test("S13 a nested citation copies the child's full path; a group copies its own", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await app.press("Escape");
  await app.click("citation", { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("y");
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toBe(`${tracker.tasks.cited}@r1:${FIRST_CHILD}`);
  await openLocation(app, spec(tracker, tracker.tasks.cited));
  await app.press("Escape");
  await app.clickPart(GROUP, { button: "right", at: "header" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("y");
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toBe(`${tracker.tasks.cited}@r1:${GROUP}`);
});
