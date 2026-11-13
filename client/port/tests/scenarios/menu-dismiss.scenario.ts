// The act menu dismisses the way the prototype's does (inst16.html:517:
// a pointer-down outside the menu closes it without acting). The human
// hit it on the native app: after a right-click on the board the menu
// never went away - a left click elsewhere did nothing, another
// right-click only moved it, and scrolling carried it off screen. The
// session had the outside-pointer close; nothing on the page called it.
import { expect, test } from "#scenario";
import { longBoard, openLocation, spec, submitLocation } from "./kit/helpers.ts";
import type { AppDriver } from "./kit/driver.ts";

async function twoVisibleParts(app: AppDriver): Promise<[string, string]> {
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(4);
  const labels = await app.labels();
  await app.press("Escape");
  await expect.poll(async () => (await app.labels()).length).toBe(0);
  return [labels[1]!.part, labels[3]!.part];
}

test("a left click outside the open menu closes it without acting", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker, 40)));
  const [first, second] = await twoVisibleParts(app);
  await app.clickPart(first, { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.clickPart(second);
  await expect.poll(() => app.menuOpen()).toBe(false);
  // dismissing ran no act: nothing was copied
  expect((await app.readout()).copy).toBeNull();
});

test("a right click elsewhere opens the menu for the new target", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker, 40)));
  const [first, second] = await twoVisibleParts(app);
  await app.clickPart(first, { button: "right" });
  await expect.poll(async () => (await app.menuGeometry())?.part ?? null).toBe(first);
  await app.clickPart(second, { button: "right" });
  await expect.poll(async () => (await app.menuGeometry())?.part ?? null).toBe(second);
});

test("scrolling the board with the wheel closes the open menu", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker, 40)));
  const [first] = await twoVisibleParts(app);
  await app.clickPart(first, { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.wheel("content", 240);
  await expect.poll(() => app.menuOpen()).toBe(false);
});

test("a left click outside a task row's menu closes it", { tags: ["native"] }, async ({ app, tracker }) => {
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.rows.length ?? 0).toBeGreaterThan(1);
  await app.press("Escape");
  const { rows, groups } = (await app.readout()).tasks!;
  await app.clickPart(rows[0]!.part, { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  // a group's heading carries no act: a plain click there is outside
  await app.clickPart(groups[0]!.part, { at: "header" });
  await expect.poll(() => app.menuOpen()).toBe(false);
  expect((await app.readout()).surface).toBe("tasks");
});
