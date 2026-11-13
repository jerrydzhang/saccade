// S16: find is a facility of the shown revision
// (t36_differential.check_s16_find_entry, the port half; the
// prototype differential half stays in the Python harness). Bare /
// opens find, typing matches, Enter commits (the field closes, the
// matches stay), Escape clears.
import { expect, test } from "#scenario";
import { find, openLocation, spec } from "./kit/helpers.ts";

test("S16 find opens, matches, commits and clears on the shown board", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  const typed = await find(app, "landing");
  expect(typed.count).toBeGreaterThanOrEqual(1);

  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).find).toMatchObject({ open: false, query: "landing", count: typed.count });

  await app.press("Escape");
  await expect.poll(async () => (await app.readout()).find).toMatchObject({ open: false, count: 0 });
  // leaving find moved nothing: the shown revision is the same
  expect((await app.readout()).board.asked).toBe(`${tracker.repo}/${tracker.tasks.landing}`);
});
