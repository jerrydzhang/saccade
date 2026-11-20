// A far machine answers late: every request to another machine is a
// program run, and a slow one must hold up neither drawing nor keys.
// One request through the shell's local request command waits on a
// loopback server that answers a second late, keys are pressed
// meanwhile and a plain command is asked of the interface's own
// thread (the one that draws the window and takes its keys): the
// window takes every key and the thread answers before the late
// answer lands. The real shell only: native.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";

const LATE_MS = 1000;

test("a late answer holds up neither drawing nor keys", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  const seen = await app.lateAnswer(LATE_MS, ["j", "k", "j"]);
  // the numbers a stall is measured by, kept in the lane's log
  console.log(`[late-answer] ${JSON.stringify({ lateMs: LATE_MS, ...seen })}`);
  expect(seen.answeredMs).toBeGreaterThanOrEqual(LATE_MS - 50);
  expect(seen.keysBeforeAnswer, "every key reached the page while the request waited").toBe(true);
  expect(seen.firstKeyMs, "the first key did not wait on the request").toBeLessThan(LATE_MS / 2);
  expect(Math.max(...seen.keyDelaysMs), "no key waited on the request").toBeLessThan(LATE_MS / 2);
  expect(seen.plainBeforeAnswer, "the interface's thread answered while the request waited").toBe(true);
  expect(seen.plainAskMs, "the interface's thread was free").toBeLessThan(LATE_MS / 2);
});
