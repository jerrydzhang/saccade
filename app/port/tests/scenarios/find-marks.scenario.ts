// D2 (the human's first-use defect): find marks the matched TEXT, not
// the whole part. Every occurrence carries a mark whose text is the
// match itself, exactly one is active, and n/N step occurrence by
// occurrence in document order - including several in one paragraph,
// where a part-level mark cannot move at all (src/lib/find/units.ts
// units are whole parts; BoardView painted the part). Read through the
// painted marks (kit/driver.ts findMarks) and the readout's find.
import { expect, test } from "#scenario";
import { find, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, FindMark, Tracker } from "./kit/driver.ts";

const active = (marks: FindMark[]): number => marks.findIndex((mark) => mark.active);

async function threeInTwoParagraphs(app: AppDriver, tracker: Tracker): Promise<void> {
  const task = await tracker.createTask("find marks: occurrences, two in one paragraph");
  await tracker.commit(
    task,
    `grammar: 1\ntask: ${task}\ncontent:\n  - text: |\n      The Alpha sweep and the second alpha sweep share a paragraph.\n  - text: |\n      A **alpha** in bold sits alone here.\n`,
    "find marks: three occurrences",
  );
  await openLocation(app, spec(tracker, task));
  expect((await find(app, "alpha")).count).toBe(3);
}

test("each occurrence is marked on its own text, one active", { tags: ["native"] }, async ({ app, tracker }) => {
  await threeInTwoParagraphs(app, tracker);
  await expect.poll(async () => (await app.findMarks()).map((mark) => mark.text.toLowerCase())).toEqual(["alpha", "alpha", "alpha"]);
  const marks = await app.findMarks();
  expect(marks.map((mark) => mark.part)).toEqual(["/content/0", "/content/0", "/content/1"]);
  expect(marks.filter((mark) => mark.active)).toHaveLength(1);
});

test("n and N step occurrence by occurrence, within a paragraph too", { tags: ["native"] }, async ({ app, tracker }) => {
  await threeInTwoParagraphs(app, tracker);
  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).find.open).toBe(false);
  const start = (await app.readout()).find.active;
  await expect.poll(async () => active(await app.findMarks())).toBe(start);
  // n, twice: through the second mark of the SAME paragraph, then the
  // next paragraph; then wrap
  for (const step of [1, 2, 3]) {
    await app.press("n");
    await expect.poll(async () => active(await app.findMarks())).toBe((start + step) % 3);
    expect((await app.readout()).find.active).toBe((start + step) % 3);
  }
  await app.press("N");
  await expect.poll(async () => active(await app.findMarks())).toBe((start + 2) % 3);
  // Escape clears the search and every mark
  await app.press("Escape");
  await expect.poll(() => app.findMarks()).toEqual([]);
});
