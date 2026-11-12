// S18: the OPEN acts deliver under real conditions
// (t36_checks._s18_port). One window shows the citing board; the
// citation's o opens a receiver AT the cited board and the source's
// link remembers it; i asks, the receiver's chip answers, the
// delivery lands on the receiver and the link is untouched; w opens a
// third viewer at the citation. Native windows only.
import { expect, test } from "#scenario";
import { citationAct, openLocation, spec } from "./kit/helpers.ts";

test("S18 linked delivery: o, i and w deliver the citation", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  const { citingChild: citing, cited } = tracker.tasks;
  const citedShown = `${tracker.repo}/${cited}@r1`;
  expect(await app.windows()).toEqual(["main"]);
  await openLocation(app, spec(tracker, citing));

  // o: the receiver opens AT the cited board; the link remembers it
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const receiver = (await app.windows()).find((label) => label !== "main")!;
  await expect.poll(async () => (await app.readout(receiver)).board.location, { timeout: 20_000 }).toBe(citedShown);
  const source = await app.readout("main");
  expect(source.window.link).toBe(receiver);
  expect(source.board.asked).toBe(`${tracker.repo}/${citing}`);

  // i: the ask answers with the receiver's chip; the link stays
  const before = (await app.readout(receiver)).doors.delivered.length;
  await citationAct(app, "i");
  await expect.poll(() => app.shows("ask-input")).toBe(true);
  const chip = (await app.readout(receiver)).doors.jumpChip;
  expect(chip, "the receiver broadcasts a jump chip").toBeTruthy();
  await app.type(chip!);
  await app.press("Enter");
  await expect.poll(async () => (await app.readout(receiver)).doors.delivered.length, { timeout: 20_000 }).toBeGreaterThan(before);
  expect((await app.readout("main")).window.link).toBe(receiver);

  // w: a third viewer opens at the citation
  await citationAct(app, "w");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(3);
  const third = (await app.windows()).find((label) => label !== "main" && label !== receiver)!;
  await expect.poll(async () => (await app.readout(third)).board.location, { timeout: 20_000 }).toBe(citedShown);
});
