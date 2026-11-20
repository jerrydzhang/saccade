// S09 and S10, migrated from the t36 Python port lane
// (t36_scenario_bodies.py _s09_port, _s10_port).
// S09: one action, every invocation agrees - the citation's first act
// (f, follow here) runs the same from a plain click, right-click +
// Enter, right-click + its key, right-click + a click on its item, and
// (since links got labels, m-547) s + its label + Enter: each lands on
// the cited pinned revision with the menu closed.
// S10: sequences go through one matcher - G down, gg back to the top;
// an unmatched key changes nothing (no surface opens, nothing scrolls).
import { expect, test } from "#scenario";
import { longBoard, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

async function followed(app: AppDriver, tracker: Tracker): Promise<void> {
  await expect
    .poll(async () => (await app.readout()).board.location)
    .toBe(`${tracker.repo}/${tracker.tasks.cited}@r1`);
  expect(await app.menuOpen(), "the menu closed before the act ran").toBe(false);
}

async function onCiting(app: AppDriver, tracker: Tracker): Promise<void> {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await app.press("Escape");
}

async function rightClickCitation(app: AppDriver): Promise<void> {
  await app.click("citation", { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
}

const INVOCATIONS: [string, (app: AppDriver) => Promise<void>][] = [
  ["a plain click", (app) => app.click("citation")],
  ["right-click + Enter", async (app) => { await rightClickCitation(app); await app.press("Enter"); }],
  ["right-click + its key f", async (app) => { await rightClickCitation(app); await app.press("f"); }],
  ["right-click + a click on its item", async (app) => { await rightClickCitation(app); await app.clickMenuAct("f"); }],
];

for (const [name, invoke] of INVOCATIONS) {
  test(`S09 ${name} follows the citation`, { tags: ["native"] }, async ({ app, tracker }) => {
    await onCiting(app, tracker);
    await invoke(app);
    await followed(app, tracker);
  });
}

test("S09 s + the citation's label + Enter follows it", { tags: ["native"] }, async ({ app, tracker }) => {
  await onCiting(app, tracker);
  const citation = ((await app.readout()).content?.parts ?? []).find((part) => part.kind === "incorporation")!.part + "/citation";
  await app.press("s");
  await expect.poll(async () => (await app.labels()).map((label) => label.part)).toContain(citation);
  await app.type((await app.labels()).find((label) => label.part === citation)!.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("Enter");
  await followed(app, tracker);
});

test("S10 G and gg run through the matcher; an unmatched key changes nothing", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await app.press("Escape");
  await app.press("G");
  await expect.poll(async () => (await app.readout()).viewport.top).toBeGreaterThan(100);
  await app.press("g", "g");
  await expect.poll(async () => (await app.readout()).viewport.top).toBeLessThanOrEqual(2);
  const before = { top: (await app.readout()).viewport.top, menu: await app.menuOpen(), labels: (await app.labels()).length, picker: await app.shows("picker-input") };
  await app.press("q");
  // a hold: an unmatched key must leave everything as it was
  await new Promise((resolve) => setTimeout(resolve, 500));
  expect({ top: (await app.readout()).viewport.top, menu: await app.menuOpen(), labels: (await app.labels()).length, picker: await app.shows("picker-input") }).toEqual(before);
});
