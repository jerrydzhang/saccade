// D6, which-key, a matcher property
// (human ruling r110). Any stroke that leaves a sequence pending shows
// that prefix's continuations, read from the ACTIVE context's tables:
// space and g alike. The popup delay and the timeout-suspend are named
// tunables, read from the readout, never pinned here. Observable: the
// readout's `input` field (kit/driver.ts InputView) and the popup's
// control id `which-key`. Written red before the code (batch 3).
import { expect, test } from "#scenario";
import { openLocation, readInput, spec } from "./kit/helpers.ts";

test("space shows its continuations after the named delay", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  const { tunables } = await readInput(app);
  await app.press("Escape");
  await app.press(" ");
  const pressed = Date.now();
  expect((await readInput(app)).pending).toEqual(["spc"]);
  await expect.poll(async () => (await readInput(app)).whichKey?.prefix ?? null, { timeout: tunables.whichKeyDelayMs + 2_000 }).toBe("spc");
  // not before the delay: the popup is a pause's answer, not a flash
  expect(Date.now() - pressed).toBeGreaterThanOrEqual(tunables.whichKeyDelayMs - 50);
  expect(await app.shows("which-key")).toBe(true);
  const keys = (await readInput(app)).whichKey!.entries.map((entry) => entry.keys);
  // the ruled space continuations: picker, find, link, unlink
  for (const continuation of ["spc spc", "spc /", "spc l", "spc u"]) expect(keys).toContain(continuation);
  for (const entry of (await readInput(app)).whichKey!.entries) {
    expect(entry.keys.startsWith("spc ")).toBe(true);
    expect(entry.label.length).toBeGreaterThan(0);
  }
});

test("g shows its continuations too: a matcher property, not a space special case", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  const { tunables } = await readInput(app);
  await app.press("Escape");
  await app.press("g");
  await expect.poll(async () => (await readInput(app)).whichKey?.prefix ?? null, { timeout: tunables.whichKeyDelayMs + 2_000 }).toBe("g");
  expect((await readInput(app)).whichKey!.entries.map((entry) => entry.keys)).toContain("g g");
});

test("an open popup suspends the sequence timeout; the continuation completes", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  const { tunables } = await readInput(app);
  expect(tunables.whichKeySuspendsTimeout, "the default suspends the timeout").toBe(true);
  await app.press("Escape");
  await app.press(" ");
  await expect.poll(async () => (await readInput(app)).whichKey?.prefix ?? null, { timeout: tunables.whichKeyDelayMs + 2_000 }).toBe("spc");
  // a hold, asserted over a window: past the sequence timeout the
  // sequence is still pending and the popup still open
  await new Promise((resolve) => setTimeout(resolve, tunables.sequenceTimeoutMs + 500));
  expect(await readInput(app)).toMatchObject({ pending: ["spc"], whichKey: { prefix: "spc" } });
  await app.press(" ");
  await expect.poll(() => app.shows("picker-input")).toBe(true);
  expect(await readInput(app)).toMatchObject({ pending: [], whichKey: null });
});
