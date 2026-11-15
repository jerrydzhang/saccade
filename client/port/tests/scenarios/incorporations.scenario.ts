// t-58: an incorporation shows collapsed - its source line and its
// receiving text - until its act expands it; expanded, it shows only
// the components its select paths cite; the same act collapses it.
// The seeds: the second citing board's incorporation names the FIRST
// child of the cited board's group (/content/0/group/content/0); the
// group's title and its second child are never cited.
import { expect, test } from "#scenario";
import type { AppDriver, Tracker } from "./kit/driver.ts";
import { find, openLocation, spec } from "./kit/helpers.ts";

const SELECT = "/content/0/group/content/0";

async function onCiting(app: AppDriver, tracker: Tracker): Promise<string> {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await app.press("Escape");
  return ((await app.readout()).content?.parts ?? []).find((part) => part.kind === "incorporation")!.part;
}

async function expanded(app: AppDriver): Promise<unknown> {
  return (await app.readout()).content?.expanded ?? null;
}

test("an incorporation is collapsed; z expands it to the cited component alone; z collapses it", async ({ app, tracker }) => {
  const part = await onCiting(app, tracker);
  // collapsed: the source line and the receiving text, nothing cited
  expect(await expanded(app)).toEqual([]);
  expect(await app.shows("cited")).toBe(false);
  const collapsed = (await app.partText(part)) ?? "";
  expect(collapsed).toContain(`incorporates ${tracker.tasks.cited}`);
  expect(collapsed).toContain("the receiving text for the first group child");

  await app.clickPart(part, { button: "right", at: "header" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect(await app.menuActs()).toContain("z");
  await app.press("z");
  await expect.poll(() => expanded(app)).toEqual([
    { part, status: "loaded", cited: [{ select: SELECT, part: `${part}/cited/0` }] },
  ]);
  await expect.poll(() => app.partText(`${part}/cited/0`)).toContain("the cited account's first child, inside a group");
  const cited = (await app.controlText("cited")) ?? "";
  expect(cited, "only the cited component: not its group's title").not.toContain("Recorded setup");
  expect(cited, "only the cited component: not its sibling").not.toContain("a second child");
  // the receiving text stays
  expect(await app.partText(part)).toContain("the receiving text for the first group child");

  await app.clickPart(part, { button: "right", at: "header" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("z");
  await expect.poll(() => expanded(app)).toEqual([]);
  await expect.poll(() => app.shows("cited")).toBe(false);
});

test("a plain click on the source line expands and collapses; the citation link still follows", async ({ app, tracker }) => {
  const part = await onCiting(app, tracker);
  await app.clickPart(part, { at: "header" });
  await expect.poll(async () => ((await expanded(app)) as { status: string }[]).map((entry) => entry.status)).toEqual(["loaded"]);
  expect(await app.menuOpen(), "the act ran directly, no menu left open").toBe(false);
  await app.clickPart(part, { at: "header" });
  await expect.poll(() => expanded(app)).toEqual([]);
  await app.click("citation");
  await expect.poll(async () => (await app.readout()).board.location).toBe(`${tracker.repo}/${tracker.tasks.cited}@r1`);
});

test("s and the incorporation's label open its acts; Enter expands", async ({ app, tracker }) => {
  const part = await onCiting(app, tracker);
  await app.press("s");
  await expect.poll(async () => (await app.labels()).map((label) => label.part)).toContain(part);
  await app.type((await app.labels()).find((label) => label.part === part)!.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("Enter");
  await expect.poll(async () => ((await expanded(app)) as { part: string }[]).map((entry) => entry.part)).toEqual([part]);
});

test("find searches the board's own text: an expanded incorporation's cited components are skipped", async ({ app, tracker }) => {
  const part = await onCiting(app, tracker);
  await app.clickPart(part, { at: "header" });
  await expect.poll(() => app.partText(`${part}/cited/0`)).toContain("inside a group");
  expect((await find(app, "inside a group")).count).toBe(0);
  // commit, then clear, before the next query
  await app.press("Enter");
  await app.press("Escape");
  await expect.poll(async () => (await app.readout()).find).toMatchObject({ open: false, count: 0 });
  expect((await find(app, "receiving text")).count).toBe(1);
});
