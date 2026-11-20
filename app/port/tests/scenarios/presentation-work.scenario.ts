// The mechanism under "folding a section has visible lag" (m-547 (5)):
// a fold rebuilt the whole board and hydrated every figure, math node
// and diagram again (t-36: ~500ms a fold). A fold, a find and a
// selection are presentation changes over the same content: the board
// renders once per load and hydrates each unit once. Read through the
// readout's presentation counts - a mechanism, never a time threshold.
import { expect, test } from "#scenario";
import { find, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

const PNG = Uint8Array.from(
  atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="),
  (ch) => ch.charCodeAt(0),
);

async function workBoard(app: AppDriver, tracker: Tracker): Promise<void> {
  const task = await tracker.createTask("presentation work: a group, a figure, math");
  await tracker.commit(
    task,
    [
      "grammar: 1",
      `task: ${task}`,
      "content:",
      "  - group:",
      "      title: A section to fold",
      "      content:",
      "        - text: Inside the section, with $x^2$ math.",
      "        - figure:",
      "            input: figures/plot.png",
      "            caption: the plot",
      "  - text: After the section, the word alpha.",
      "",
    ].join("\n"),
    "presentation work",
    [{ path: "figures/plot.png", content: PNG }],
  );
  await openLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).render.length, { timeout: 15_000 }).toBe(2);
  await app.press("Escape");
}

const work = async (app: AppDriver) => (await app.readout()).presentation;

test("folding and unfolding a section neither re-renders nor re-hydrates", { tags: ["native"] }, async ({ app, tracker }) => {
  await workBoard(app, tracker);
  const before = await work(app);
  expect(before).toMatchObject({ renders: 1, hydrated: 2 });
  for (let fold = 0; fold < 2; fold++) {
    await app.clickPart("/content/0", { button: "right", at: "header" });
    await expect.poll(() => app.menuOpen()).toBe(true);
    await app.press("z");
    await expect.poll(async () => ((await app.partText("/content/0")) ?? "").includes("the plot")).toBe(fold === 1);
  }
  expect(await work(app)).toEqual(before);
});

test("find and a selection neither re-render nor re-hydrate", { tags: ["native"] }, async ({ app, tracker }) => {
  await workBoard(app, tracker);
  const before = await work(app);
  await find(app, "alpha");
  await app.press("Enter");
  await app.press("n");
  await app.press("Escape");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  await app.press("Escape");
  expect(await work(app)).toEqual(before);
});
