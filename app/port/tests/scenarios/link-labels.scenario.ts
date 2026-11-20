// Links get labels (human ruling, m-547 (2)): a ref in prose (t-N,
// t-N@rK, t-N@rK:/path, i-N, m-N - the prototype's linkify,
// inst16.html:197) and an incorporation's citation are select targets
// of their own. s labels them, the label's menu offers their acts with
// act 0 = follow (the prototype's ref kind, :43), and Enter runs it.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

async function refBoard(app: AppDriver, tracker: Tracker): Promise<string> {
  const task = await tracker.createTask("link labels: refs in prose");
  const cited = tracker.tasks.cited;
  await tracker.commit(
    task,
    [
      "grammar: 1",
      `task: ${task}`,
      "content:",
      `  - text: See ${cited}@r1 for the cited board, and i-3 for who asked.`,
      "",
    ].join("\n"),
    "link labels: a task ref and an incarnation ref",
  );
  await openLocation(app, spec(tracker, task));
  await app.press("Escape");
  return task;
}

async function labelOf(app: AppDriver, part: string): Promise<string> {
  await app.press("s");
  await expect.poll(async () => (await app.labels()).map((label) => label.part)).toContain(part);
  return (await app.labels()).find((label) => label.part === part)!.label;
}

test("s labels a task ref in prose; Enter follows it", { tags: ["native"] }, async ({ app, tracker }) => {
  await refBoard(app, tracker);
  const label = await labelOf(app, "/content/0/ref/0");
  await app.type(label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect((await app.menuGeometry())?.part).toBe("/content/0/ref/0");
  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).board.location).toBe(`${tracker.repo}/${tracker.tasks.cited}@r1`);
});

test("an incarnation ref is labeled; its only act copies it", { tags: ["native"] }, async ({ app, tracker }) => {
  await refBoard(app, tracker);
  const label = await labelOf(app, "/content/0/ref/1");
  await app.type(label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toBe("i-3");
});

test("an incorporation's citation is labeled; Enter follows it", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await app.press("Escape");
  const parts = ((await app.readout()).content?.parts ?? []).filter((part) => part.kind === "incorporation").map((part) => part.part);
  expect(parts.length).toBeGreaterThan(0);
  const label = await labelOf(app, `${parts[0]}/citation`);
  await app.type(label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).board.location).toBe(`${tracker.repo}/${tracker.tasks.cited}@r1`);
});

test("a plain click on a task ref follows it", { tags: ["native"] }, async ({ app, tracker }) => {
  await refBoard(app, tracker);
  await app.clickPart("/content/0/ref/0");
  await expect.poll(async () => (await app.readout()).board.location).toBe(`${tracker.repo}/${tracker.tasks.cited}@r1`);
});
