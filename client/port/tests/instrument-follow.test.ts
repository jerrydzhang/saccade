// The instruments' shared follow loop (lib/instruments/follow.ts), as
// the task list uses it: a bounded poll that refreshes when the page's
// content differs from the shown one, quiet on equal content and on
// failure, ended by stop. The board's adapter has its own tests
// (follow.test.ts). Pure tests with a manual timer.

import assert from "node:assert/strict";
import { test } from "vitest";
import { createFollow } from "../src/lib/instruments/follow.ts";

function harness() {
  const log: string[] = [];
  let job: (() => void) | null = null;
  let current: string | null | Error = "rows-a";
  let shown: string | null = "rows-a";
  const follow = createFollow<string>({
    intervalMs: 3000,
    same: (a, b) => a === b,
    current: async () => {
      if (current instanceof Error) throw current;
      return current;
    },
    shown: () => shown,
    refresh: () => {
      log.push("refresh");
    },
    schedule: (_ms, run) => {
      job = run;
      return 1;
    },
    cancel: () => {
      job = null;
    },
  });
  return {
    follow,
    log,
    pending: () => job !== null,
    tick: async () => {
      const run = job;
      job = null;
      run?.();
      await new Promise((resolve) => setImmediate(resolve));
    },
    set current(value: string | null | Error) {
      current = value;
    },
    set shown(value: string | null) {
      shown = value;
    },
  };
}

test("changed rows refresh; equal rows do not", async () => {
  const h = harness();
  h.follow.start("tasks");
  await h.tick();
  assert.deepEqual(h.log, []);
  h.current = "rows-b";
  await h.tick();
  assert.deepEqual(h.log, ["refresh"]);
  h.shown = "rows-b";
  await h.tick();
  assert.deepEqual(h.log, ["refresh"], "in step again: quiet");
});

test("a failing read is silent and the poll keeps going; stop ends it", async () => {
  const h = harness();
  h.follow.start("tasks");
  h.current = new Error("the tracker is down");
  await h.tick();
  h.current = null;
  await h.tick();
  assert.deepEqual(h.log, []);
  assert.equal(h.pending(), true, "still polling after failures");
  h.follow.stop();
  assert.equal(h.pending(), false);
  assert.deepEqual(h.follow.running(), { armed: false, key: null });
});
