import assert from "node:assert/strict";
import { test, onTestFinished, vi } from "vitest";
import { SliceBufferHost } from "../src/components/Buffer.ts";
import type { InstrumentAttachment } from "../src/components/Buffer.ts";
import { connectBoard } from "../src/instruments/boardController.ts";
import type { BoardData, BoardUI } from "../src/instruments/boardController.ts";
import type { BoardAnchor } from "../src/lib/contracts/places.ts";
import { sourceFor } from "../src/lib/sources.ts";
import { TrackerSource } from "../src/lib/tracker.ts";
import { FOLLOW_POLL_MS } from "../src/lib/board/follow.ts";

// Controller integration: a real pinned resolution must NOT erase the
// bare follow intent. Consume timers like the browser, and change the
// source's current revision twice to catch one-shot implementations.
test("bare board follows multiple pinned resolutions, stops on pinned open, and releases", async (t) => {
  const machine = "tracker:follow-test";
  const source = sourceFor(machine) as TrackerSource;
  let revision = 1;
  let failRead = false;
  const reads: string[] = [];
  const polls: number[] = [];
  vi.spyOn(source, "loadText").mockImplementation(async (location: { path: string }) => {
    reads.push(location.path);
    if (failRead) throw new Error("temporary source failure");
    return {
      text: `grammar: 1\ntask: t-1\ncontent:\n  - text: revision ${revision}\n`,
      location: { machine, path: `t-1@r${revision}` },
    };
  });
  vi.spyOn(source, "currentBoardPage").mockImplementation(async (task: number) => {
    polls.push(task);
    return { location: `t-1@r${revision}` } as never;
  });
  let job: (() => void) | null = null;
  const realTimeout = globalThis.setTimeout;
  const realClear = globalThis.clearTimeout;
  const timerId = 99999;
  vi.spyOn(globalThis, "setTimeout").mockImplementation((run: () => void, ms?: number, ...args: unknown[]) => {
    if (ms === FOLLOW_POLL_MS) { job = run; return timerId as never; }
    return realTimeout(run, ms, ...args) as never;
  });
  vi.spyOn(globalThis, "clearTimeout").mockImplementation((id) => {
    if (id === timerId) job = null;
    else realClear(id);
  });
  let attachment!: InstrumentAttachment<BoardUI, BoardAnchor>;
  const controller = new SliceBufferHost<BoardData, BoardUI, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" },
    viewport: {
      metrics: () => ({ position: { left: 0, top: 0 }, width: 100, height: 100 }),
      scrollLines() {}, scrollPage() {}, start() {}, end() {}, reveal() {}, moveTo() {},
 scrollFraction: () => {},
    },
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) { attachment = connectBoard(buffer); return attachment; },
  });
  const built = controller.wire();
  onTestFinished(() => built.host.destroy());
  const tick = async () => {
    const run = job;
    job = null;
    run?.();
    await new Promise((resolve) => setImmediate(resolve));
  };
  await attachment.places.request({ machine, path: "t-1" });
  assert.notEqual(job, null);
  built.host.activate();
  const press = (key: string) => built.host.handleKey({ key, ctrl: false, shift: false, alt: false, meta: false });
  press("/");
  attachment.presentation.typeFind("revision");
  assert.equal(attachment.presentation.findView().open, true);
  assert.equal(attachment.presentation.findView().count, 1);
  assert.equal(press("s"), "allow-default", "ordinary typing in captured text entry never starts selection");
  assert.equal(attachment.presentation.getSelection(), null);
  await tick();
  assert.deepEqual(reads, ["t-1"]);
  revision = 2;
  await tick();
  revision = 3;
  await tick();
  assert.deepEqual(reads, ["t-1", "t-1", "t-1"]);
  assert.deepEqual(polls, [1, 1, 1]);
  assert.equal(attachment.presentation.findView().open, true, "background follow does not end Find");
  const beforeFailure = built.host.snapshot();
  revision = 4;
  failRead = true;
  await tick();
  const afterFailure = built.host.snapshot();
  assert.equal(beforeFailure.kind, "alive");
  assert.equal(afterFailure.kind, "alive");
  if (beforeFailure.kind === "alive" && afterFailure.kind === "alive") {
    assert.equal(beforeFailure.content.kind, "available");
    assert.equal(afterFailure.content.kind, "available");
    if (beforeFailure.content.kind === "available" && afterFailure.content.kind === "available") {
      assert.equal(afterFailure.content.value, beforeFailure.content.value, "failed refresh retains shown frame");
      assert.notEqual(afterFailure.content.lastFailure, null, "failure is reported separately");
    }
  }
  failRead = false;
  await tick();
  assert.equal(reads.length, 5, "same bare intent retries after failure");
  assert.notDeepEqual(built.host.snapshot(), afterFailure, "recovery publishes the newer revision");
  assert.equal(attachment.presentation.findView().open, true, "failure/recovery does not restore old interaction state");
  await attachment.places.request({ machine, path: "t-1@r4" });
  assert.equal(attachment.presentation.findView().open, false, "explicit open ends Find");
  assert.equal(job, null, "explicit pinned open stops follow");
  await tick();
  assert.equal(reads.length, 6);
  press("/");
  assert.equal(attachment.presentation.findView().query, "revision");
  press("Escape");
  assert.equal(attachment.presentation.findView().open, false, "declared Escape releases the captured context");
  press("/");
  const restore = built.host.places.restore({
    bufferId: "b", address: { machine, path: "t-1@r4" }, position: { kind: "default" },
  });
  assert.equal(attachment.presentation.findView().open, false, "trail restore shares the explicit-open funnel");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(restore.snapshot(), "restored");
  await attachment.places.request({ machine, path: "t-1" });
  assert.notEqual(job, null);
  press("/");
  built.host.destroy();
  assert.equal(attachment.presentation.findView().open, false);
  assert.equal(job, null, "release cancels follow");
});
