// The task list narrowed by tag: a narrowed list is an ordinary
// location. It stays narrowed through the follow's poll, a re-open and
// Back, a row whose tag was removed leaves at the next poll, an unknown
// tag shows the tracker's refusal on the problem line, and the picker
// offers the repository's tags. Each scenario makes its own tags and
// tasks on the shared throwaway tracker.
import { expect, test } from "#scenario";
import { openPicker, openPlane, submitLocation } from "./kit/helpers.ts";
import type { AppDriver } from "./kit/driver.ts";

const rowsOf = async (app: AppDriver): Promise<string[]> =>
  ((await app.readout()).tasks?.rows ?? []).map((row) => row.task).sort();

test("a narrowed list stays narrowed through a poll, a re-open and Back, and drops a row whose tag was removed", async ({ app, tracker }) => {
  await tracker.createTag("ntscope", "the narrowing scenario's own tag");
  const kept = await tracker.createTask("narrowed: tagged and kept");
  const dropped = await tracker.createTask("narrowed: tagged then untagged");
  const other = await tracker.createTask("narrowed: never tagged");
  await tracker.tagTask(kept, "ntscope");
  await tracker.tagTask(dropped, "ntscope");

  const narrowed = `${tracker.repo}/tasks?tag=ntscope`;
  // two spellings of one location: the shown one is the canonical
  await submitLocation(app, `${tracker.repo}/tasks?tag=NTSCOPE,ntscope`);
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null, { timeout: 20_000 }).toBe("loaded");
  expect((await app.readout()).tasks?.asked).toBe(narrowed);
  expect(await rowsOf(app)).toEqual([kept, dropped].sort());
  expect(await rowsOf(app)).not.toContain(other);

  // a tag added while the list is shown: its row arrives by the poll,
  // and the untagged one stays out
  const late = await tracker.createTask("narrowed: tagged later");
  await tracker.tagTask(late, "ntscope");
  await expect.poll(() => rowsOf(app), { timeout: 20_000 }).toEqual([kept, dropped, late].sort());

  // a tag removed while the list is shown: its row leaves at the next poll
  await tracker.untagTask(dropped, "ntscope");
  await expect.poll(() => rowsOf(app), { timeout: 20_000 }).toEqual([kept, late].sort());

  // Back returns to the narrowed list; the whole list shows between
  await openPlane(app, tracker);
  await expect.poll(async () => (await rowsOf(app)).includes(other)).toBe(true);
  await app.press("Ctrl-o");
  await expect.poll(async () => (await app.readout()).tasks?.asked ?? null).toBe(narrowed);
  await expect.poll(() => rowsOf(app), { timeout: 20_000 }).toEqual([kept, late].sort());

  // opening it again reads it afresh, still narrowed
  await submitLocation(app, narrowed);
  await expect.poll(() => rowsOf(app), { timeout: 20_000 }).toEqual([kept, late].sort());
  expect((await app.readout()).trail.entries.map((entry) => entry.location)).toContain(narrowed);
});

test("an unknown tag in a location shows the tracker's refusal with the tags that exist", async ({ app, tracker }) => {
  await tracker.createTag("ntknown", "a tag that does exist");
  await submitLocation(app, `${tracker.repo}/tasks?tag=ntmissing`);
  await expect
    .poll(async () => (await app.readout()).tasks?.problem ?? null, { timeout: 20_000 })
    .toMatch(/no tag 'ntmissing'/);
  const tasks = (await app.readout()).tasks!;
  expect(tasks.status).toBe("invalid");
  expect(tasks.problem).toContain("ntknown - a tag that does exist");
  expect(tasks.rows).toEqual([]);
});

test("the picker offers the repository's tags, each opening its narrowed list", async ({ app, tracker }) => {
  await tracker.createTag("ntpick", "a tag the picker offers");
  const tagged = await tracker.createTask("narrowed: picked by its tag");
  await tracker.tagTask(tagged, "ntpick");
  await openPlane(app, tracker);
  await openPicker(app);
  await app.type("ntpick");
  await app.press("Enter");
  await expect
    .poll(async () => (await app.readout()).tasks?.asked ?? null, { timeout: 20_000 })
    .toBe(`${tracker.repo}/tasks?tag=ntpick`);
  await expect.poll(() => rowsOf(app), { timeout: 20_000 }).toEqual([tagged]);
});
