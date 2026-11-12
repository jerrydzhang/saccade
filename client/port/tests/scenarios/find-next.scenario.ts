// The n/N ruling (t-36 r110) on a board whose three matches are all on
// screen at once. The readout's find.active steps on n today; what
// the person sees does not (the third test): the port renders no active-match marker
// (src/lib/find/interaction.ts reveal only scrolls, a no-op when every
// match is visible), so n looks dead. The third test is that KNOWN
// DEFECT, expected to fail (test.fails) until the fix lands; it reads
// the proposed observable, a data-port-find-active attribute on the
// active match's part (kit/driver.ts activeMatch). When the fix lands
// with that attribute this test starts "failing": drop `fails` then.
import { expect, test } from "#scenario";
import { find, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

async function committedSearch(app: AppDriver, tracker: Tracker): Promise<number> {
  const task = await tracker.createTask("find-next: three matches, all on screen");
  await tracker.commit(
    task,
    `grammar: 1\ntask: ${task}\ncontent:\n  - text: alpha one\n  - text: alpha two\n  - text: alpha three\n`,
    "find-next: three short matches",
  );
  await openLocation(app, spec(tracker, task));
  expect((await find(app, "alpha")).count).toBe(3);
  await app.press("Enter");
  await expect.poll(async () => (await app.readout()).find.open).toBe(false);
  return (await app.readout()).find.active;
}

test("n steps the committed search's active match (readout)", async ({ app, tracker }) => {
  const before = await committedSearch(app, tracker);
  await app.press("n");
  await expect.poll(async () => (await app.readout()).find.active).toBe((before + 1) % 3);
});

// BASE DEFECT, expected to fail: src/lib/input/contexts.ts binds
// find-prev as shift("n"), but a real Shift+n arrives as key "N" and
// the matcher is exact (src/lib/input/input-context.ts strokeMatches),
// so N never fires on real input (G is bound as shift("G") and works).
test("N steps the committed search back", async ({ app, tracker }) => {
  const before = await committedSearch(app, tracker);
  await app.press("n");
  await expect.poll(async () => (await app.readout()).find.active).toBe((before + 1) % 3);
  await app.press("N");
  await expect.poll(async () => (await app.readout()).find.active, { timeout: 3_000 }).toBe(before);
});

test("n visibly moves the active match with every match on screen", async ({ app, tracker }) => {
  await committedSearch(app, tracker);
  const shown = await app.activeMatch();
  await app.press("n");
  await expect.poll(() => app.activeMatch(), { timeout: 3_000 }).not.toBe(shown);
  expect(await app.activeMatch()).not.toBeNull();
});
