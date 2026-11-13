// The human's traces (m-536): after s, typed letters that named no
// label left the select session alive with nothing to select - every
// later key consumed, j/k dead, no exit but Escape the person did not
// know to press. The prototype's hintKey (inst16.html:106), ruled
// exactly (m-541): letters match labels only, a letter that leaves
// ZERO matching labels clears them and exits; a non-letter is
// ignored; Escape exits.
import { expect, test } from "#scenario";
import { longBoard, openLocation, spec } from "./kit/helpers.ts";

test("a letter that matches nothing ends the selection; j scrolls again", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await app.press("Escape");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  // the long board's labels are letters a..; none starts with z
  expect((await app.labels()).some((label) => label.label.toLowerCase().startsWith("z"))).toBe(false);
  await app.type("z");
  await expect.poll(async () => (await app.labels()).length).toBe(0);
  const before = (await app.readout()).viewport.top;
  await app.press("j");
  await expect.poll(async () => (await app.readout()).viewport.top).toBe(before + 60);
});

test("a non-letter is ignored: the labels stay, Escape ends them", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  await app.press("Escape");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  const labels = await app.labels();
  await app.type("1");
  expect(await app.labels()).toEqual(labels);
  await app.press("Escape");
  await expect.poll(async () => (await app.labels()).length).toBe(0);
});
