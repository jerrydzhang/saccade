// D6 (written red before the code, batch 3): '?' is which-key with an empty
// prefix over the active context's whole command table. RECORDED
// DIFFERENCE: the prototype's showHelp (inst16.html:90-96) is a
// hand-curated key list; the port's help is table-driven (the
// design-agnostic first pass). So this asserts the information - the
// bound commands are listed with their keys - not any curation. The
// exhaustive table equality is the unit test's (tests/which-key.test.ts,
// help); here the active board's layered contexts are spot-checked by
// key, across the window and reading tables.
import { expect, test } from "#scenario";
import { openLocation, readInput, spec } from "./kit/helpers.ts";

test("? lists the active context's whole command table", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  await app.press("Escape");
  await app.press("?");
  await expect.poll(async () => (await readInput(app)).whichKey?.prefix ?? null).toBe("");
  expect(await app.shows("which-key")).toBe(true);
  const entries = (await readInput(app)).whichKey!.entries;
  const keys = entries.map((entry) => entry.keys);
  // the window table (src/lib/input/contexts.ts windowContext) and the
  // board reading table (boardReadingContext), by their bound keys
  for (const bound of ["spc spc", "Ctrl-o", "Ctrl-i", "j", "k", "g g", "Shift-G", "/", "n", "Shift-N", "s"]) {
    expect(keys).toContain(bound);
  }
  for (const entry of entries) {
    expect(entry.label.length).toBeGreaterThan(0);
    expect(entry.actionId.length).toBeGreaterThan(0);
  }
  // Escape closes the help
  await app.press("Escape");
  await expect.poll(async () => (await readInput(app)).whichKey).toBeNull();
});
