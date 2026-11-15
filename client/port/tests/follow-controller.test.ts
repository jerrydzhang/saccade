import assert from "node:assert/strict";
import { test, onTestFinished, vi } from "vitest";
import { createViewport, SliceBufferHost } from "../src/components/Buffer.ts";
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

// t-43: a capture names the location as it was ASKED. A bare board
// captures bare (never the revision it resolved to), so a trail return
// requests it bare and the follow re-arms; a pinned board captures
// pinned and stays unfollowed.
test("a bare board captures bare and its restore follows again; a pinned one stays pinned", async () => {
  const machine = "tracker:capture-asked-test";
  const source = sourceFor(machine) as TrackerSource;
  const reads: string[] = [];
  vi.spyOn(source, "loadText").mockImplementation(async (location: { path: string }) => {
    reads.push(location.path);
    const revision = /@r(\d+)$/.exec(location.path)?.[1] ?? "3";
    return {
      text: `grammar: 1\ntask: t-5\ncontent:\n  - text: revision ${revision}\n`,
      location: { machine, path: `t-5@r${revision}` },
    };
  });
  vi.spyOn(source, "currentBoardPage").mockImplementation(async () => ({ location: "t-5@r3" }) as never);
  let attachment!: InstrumentAttachment<BoardUI, BoardAnchor>;
  const controller = new SliceBufferHost<BoardData, BoardUI, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" },
    viewport: createViewport(),
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) { attachment = connectBoard(buffer); return attachment; },
  });
  const built = controller.wire();
  onTestFinished(() => built.host.destroy());
  const followState = () => attachment.presentation.followState();

  await attachment.places.request({ machine, path: "t-5" });
  const bare = built.host.places.capture();
  assert.deepEqual(bare.address, { machine, path: "t-5" }, "the capture is the asked bare location, not t-5@r3");

  await attachment.places.request({ machine, path: "t-5@r1" });
  const pinned = built.host.places.capture();
  assert.deepEqual(pinned.address, { machine, path: "t-5@r1" }, "a pinned open captures pinned");
  assert.equal(followState().armed, false);

  const back = built.host.places.restore(bare);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(back.snapshot(), "restored");
  assert.equal(reads.at(-1), "t-5", "the restore requests the bare location");
  assert.equal(followState().armed, true, "the restored bare board follows again");
  assert.equal(followState().task, 5);

  const forward = built.host.places.restore(pinned);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(forward.snapshot(), "restored");
  assert.equal(followState().armed, false, "the restored pinned board does not follow");
});

// An older tracker (before sac 0.20.7) serves no component changes:
// history refuses in words where its card would stand, and never
// opens to fall straight back (the human saw only a flicker).
test("history on a tracker without component changes refuses visibly and stays out of the mode", async () => {
  const machine = "tracker:history-old-test";
  const source = sourceFor(machine) as TrackerSource;
  vi.spyOn(source, "loadText").mockImplementation(async () => ({
    text: "grammar: 1\ntask: t-5\ncontent:\n  - text: revision 3\n",
    location: { machine, path: "t-5@r3" },
  }));
  vi.spyOn(source, "currentBoardPage").mockImplementation(async () => ({ location: "t-5@r3" }) as never);
  vi.spyOn(source, "history").mockImplementation(async () => ({ entries: [] }) as never);
  vi.spyOn(source, "revisionChanges").mockImplementation(async () => {
    throw new Error("the tracker's diff page for t-5@r2..r3 carries no component changes; move the binaries in step");
  });
  let attachment!: InstrumentAttachment<BoardUI, BoardAnchor>;
  const controller = new SliceBufferHost<BoardData, BoardUI, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" },
    viewport: createViewport(),
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) { attachment = connectBoard(buffer); return attachment; },
  });
  const built = controller.wire();
  onTestFinished(() => built.host.destroy());
  await attachment.places.request({ machine, path: "t-5" });
  let refusal: string | null = null;
  const off = attachment.presentation.subscribeHistoryRefusal((next) => {
    refusal = next;
  });
  onTestFinished(off);
  attachment.presentation.toggleHistory();
  for (let tries = 0; tries < 50 && refusal === null; tries++) await new Promise((resolve) => setImmediate(resolve));
  assert.match(refusal ?? "", /sac 0\.20\.7/);
  const state = built.host.snapshot();
  assert.ok(state.kind === "alive" && state.content.kind === "available");
  assert.equal((state.content.value.value as BoardData).history ?? null, null, "the mode never opened");
  // the next open clears the note
  await attachment.places.request({ machine, path: "t-5" });
  assert.equal(refusal, null);
});
