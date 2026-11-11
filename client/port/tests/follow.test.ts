// The follow intent (chunk G, scenario 15): a normally opened board
// follows current - a bounded poll, a refresh on advance, silence on
// failure, no follow for pinned opens. Pure tests with a manual timer.

import assert from "node:assert/strict";
import test from "node:test";
import { createFollow, pinnedEqual } from "../src/lib/board/follow.ts";

function harness() {
  const log: string[] = [];
  let job: (() => void) | null = null;
  let current: string | null = "t-1@r1";
  let shown: string | null = "seedtracker/t-1@r1";
  const follow = createFollow({
    currentRevision: async () => current,
    shownRevision: () => shown,
    refresh: () => log.push("refresh"),
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
    tick: async () => {
      job?.();
      await new Promise((resolve) => setImmediate(resolve));
    },
    set current(value: string | null) {
      current = value;
    },
    set shown(value: string | null) {
      shown = value;
    },
  };
}

test("an advancing tracker revision refreshes; an equal one does not", async () => {
  const h = harness();
  h.follow.start(1);
  await h.tick();
  assert.deepEqual(h.log, [], "same revision: no refresh");
  h.current = "t-1@r2";
  await h.tick();
  assert.deepEqual(h.log, ["refresh"], "advanced revision: one refresh");
  h.shown = "seedtracker/t-1@r2";
  await h.tick();
  assert.deepEqual(h.log, ["refresh"], "back in step: quiet again");
});

test("a failing poll is silent and retries; stop cancels", async () => {
  const h = harness();
  h.follow.start(1);
  h.current = null;
  await h.tick();
  assert.deepEqual(h.log, []);
  h.current = "t-1@r9";
  await h.tick();
  assert.deepEqual(h.log, ["refresh"]);
  h.follow.stop();
  h.current = "t-1@r10";
  await h.tick();
  assert.deepEqual(h.log, ["refresh"], "stopped: no further polls run");
});

test("pinned comparison tolerates the machine/path prefix each surface carries", () => {
  assert.equal(pinnedEqual("seedtracker/t-1@r3", "t-1@r3"), true);
  assert.equal(pinnedEqual("seedtracker/t-1@r3", "t-1@r4"), false);
  assert.equal(pinnedEqual("other/t-2@r1", "t-2@r1"), true);
});
