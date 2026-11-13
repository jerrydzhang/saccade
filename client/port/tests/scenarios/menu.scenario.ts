// The act menu after a selection (the human's m-545 item 2): it showed
// in odd places and Enter did not run the first act. The prototype's
// openAt (inst16.html:61-64): at the pointer for a right-click, else
// beside the target - its left + 10, 4px below, flipped above when it
// would not fit - clamped to the VIEWPORT; and Enter runs act 0 (the
// menu branch, :496). On a scrolled board the port placed the menu in
// content coordinates clamped by the window height, and offset a
// right-click by the stage twice. Read through menuGeometry (viewport
// coordinates) and the readout.
import { expect, test } from "#scenario";
import { longBoard, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

async function scrolledLongBoard(app: AppDriver, tracker: Tracker): Promise<void> {
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await app.press("Escape");
  await app.press("Ctrl-f", "Ctrl-f", "Ctrl-f");
  await expect.poll(async () => (await app.readout()).viewport.top).toBeGreaterThan(1500);
}

async function selectVisibleParagraph(app: AppDriver): Promise<string> {
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(4);
  // a paragraph in the middle of the screen: room below for the menu
  const labels = await app.labels();
  const chip = labels[Math.floor(labels.length / 2)]!;
  await app.type(chip.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  return chip.part;
}

test("after s and a label, the menu sits beside its target, on screen", { tags: ["native"] }, async ({ app, tracker }) => {
  await scrolledLongBoard(app, tracker);
  const part = await selectVisibleParagraph(app);
  const geometry = (await app.menuGeometry())!;
  expect(geometry.part).toBe(part);
  const { menu, target, viewport } = geometry;
  expect(target).not.toBeNull();
  // on screen
  expect(menu.top).toBeGreaterThanOrEqual(0);
  expect(menu.bottom).toBeLessThanOrEqual(viewport.height);
  // beside the target: its left + 10, 4px below it (or flipped above)
  expect(menu.left).toBeCloseTo(target!.left + 10, 0);
  const below = Math.abs(menu.top - (target!.bottom + 4)) <= 1;
  const above = Math.abs(menu.bottom - (target!.top - 4)) <= 1;
  expect(below || above, `menu ${JSON.stringify(menu)} target ${JSON.stringify(target)}`).toBe(true);
});

test("after s and a label, Enter runs the first act", { tags: ["native"] }, async ({ app, tracker }) => {
  await scrolledLongBoard(app, tracker);
  const part = await selectVisibleParagraph(app);
  expect((await app.readout()).copy).toBeNull();
  await app.press("Enter");
  // a paragraph's first act copies its citation (src/lib/menu/acts.ts)
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toContain(part);
  await expect.poll(() => app.menuOpen()).toBe(false);
});

test("a right-click on a scrolled board opens the menu at the pointer", { tags: ["native"] }, async ({ app, tracker }) => {
  await scrolledLongBoard(app, tracker);
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(4);
  const labels = await app.labels();
  const part = labels[Math.floor(labels.length / 2)]!.part;
  await app.press("Escape");
  await app.clickPart(part, { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  const { menu, target } = (await app.menuGeometry())!;
  // the pointer is the part's center (clickPart's default)
  const x = (target!.left + target!.right) / 2;
  const y = (target!.top + target!.bottom) / 2;
  expect(menu.left).toBeCloseTo(x, -1);
  expect(menu.top).toBeCloseTo(y, -1);
});
