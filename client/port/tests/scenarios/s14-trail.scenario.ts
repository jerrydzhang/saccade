// S14: the trail records explicit openings only
// (t36_differential._s14_port). Two picker openings land two trail
// entries with the cursor on the newest; a fold through the act menu
// leaves the trail untouched.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";

test("S14 the trail records explicit openings only", async ({ app, tracker }) => {
  const { citing, landing } = tracker.tasks;
  await openLocation(app, spec(tracker, citing));
  const obs = await openLocation(app, spec(tracker, landing));

  const trail = obs.trail;
  expect(trail.depth).toBeGreaterThanOrEqual(2);
  expect(trail.cursor).toBe(trail.depth - 1);
  const locations = trail.entries.map((entry) => entry.location).join(" ");
  expect(locations).toContain(`/${citing}`);
  expect(locations).toContain(`/${landing}`);
  expect(trail).toMatchObject({ canBack: true, canForward: false });

  // the fold act must not record an opening
  const group = obs.content?.parts.find((part) => part.kind === "group");
  expect(group, "the landing board carries a group to fold").toBeDefined();
  await app.clickPart(group!.part, { button: "right", at: "header" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("z");
  await expect.poll(() => app.menuOpen()).toBe(false);
  expect((await app.readout()).trail).toEqual(trail);
});
