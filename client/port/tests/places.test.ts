// The anchor placement (chunk D1): reveal with fold-unfolding,
// pending-waits (pending is not absence), confirmed-missing default
// position AT the same place, scroll cancellation, and the part
// ancestor walk. Pure tests with fakes - no DOM.

import assert from "node:assert/strict";
import test from "node:test";
import { partAncestors, selectToPart } from "../src/lib/board/targets.ts";
import { SliceBufferHost as BufferController } from "../src/components/Buffer.ts";
import type { BufferBuild, BuiltBuffer } from "../src/components/Buffer.ts";
import type { BoardAnchor } from "../src/lib/contracts/places.ts";
import type { AnchorData } from "../src/lib/contracts/places.ts";
import type { InstrumentAttachment } from "../src/components/Buffer.ts";
import type { InputContext } from "../src/lib/contracts/input.ts";
import { commandContext } from "../src/lib/contracts/input.ts";
import type { Bounds, Point, Viewport } from "../src/lib/contracts/buffer.ts";

test("grammar select paths map to canonical part paths", () => {
  assert.equal(selectToPart("/content/0/group/content/1"), "/content/0/content/1");
  assert.equal(selectToPart("/content/3"), "/content/3");
  assert.equal(selectToPart("group/content/2"), "/content/2");
  assert.deepEqual(partAncestors(selectToPart("/content/0/group/content/1")), ["/content/0"]);
});

test("part ancestors walk outermost-first", () => {
  assert.deepEqual(partAncestors("/content/0"), []);
  assert.deepEqual(partAncestors("/content/0/content/1"), ["/content/0"]);
  assert.deepEqual(partAncestors("/content/0/content/1/content/2"), [
    "/content/0",
    "/content/0/content/1",
  ]);
});

function buildFake(behaviour: {
  reveal: (call: number) => { kind: "found"; bounds: Bounds } | { kind: "pending" } | { kind: "missing" };
  unfoldCalls: string[];
}): {
  built: BuiltBuffer<{ block: string }, Record<string, never>>;
  revealed: Bounds[];
  moved: Point[];
} {
  const revealed: Bounds[] = [];
  const moved: Point[] = [];
  let position = { left: 0, top: 0 };
  const viewport: Viewport = {
    metrics: () => ({ position, width: 100, height: 100 }),
    scrollLines: () => {},
    scrollPage: () => {},
    start: () => {},
    end: () => {},
    reveal: (bounds) => {
      revealed.push(bounds);
      if (bounds.top < position.top) position = { ...position, top: bounds.top };
      else if (bounds.bottom > position.top + 100) position = { ...position, top: bounds.bottom - 100 };
    },
    moveTo: (next) => { moved.push(next); position = { ...next }; },
  };
  let revealCalls = 0;
  const attachment: InstrumentAttachment<Record<string, never>, BoardAnchor> = {
    presentation: {},
    input: commandContext(() => []),
    places: {
      captureDisplayed: () => ({ address: { machine: "m", path: "/p" }, position: { kind: "default" } }),
      request: async () => "applied",
      acceptsAnchor: (data: AnchorData): data is BoardAnchor =>
        (data as { kind?: string }).kind === "board-part",
      revealAnchor: (anchor: BoardAnchor) => {
        behaviour.unfoldCalls.push(anchor.path);
        return behaviour.reveal(revealCalls++);
      },
      defaultPosition: () => ({ left: 0, top: 0 }),
      subscribeChanges: () => () => {},
    },
    release: () => {},
  };
  const build: BufferBuild<{ block: string }, Record<string, never>, BoardAnchor> = {
    context: { windowId: "w", bufferId: "b" },
    viewport,
    diagnostics: { report: () => "", unexpected: () => "" },
    connect: () => attachment,
  };
  const controller = new BufferController(build);
  const built = controller.wire();
  return { built, revealed, moved };
}

test("a found anchor places at its bounds with the saved offset", async () => {
  const unfoldCalls: string[] = [];
  const { built } = buildFake({
    reveal: () => ({ kind: "found", bounds: { left: 0, top: 500, right: 100, bottom: 540 } }),
    unfoldCalls,
  });
  const job = built.host.places.restore({
    bufferId: "b",
    address: { machine: "m", path: "/p" },
    position: {
      kind: "anchor",
      anchor: { kind: "board-part", path: "/content/1/content/0" },
      viewportOffset: { left: 0, top: -10 },
    },
  });
  await settled(job.snapshot.bind(job));
  assert.equal(job.snapshot(), "restored");
  assert.deepEqual(built.host.viewport.metrics().position, { left: 0, top: 510 }, "anchor offset = content bounds minus scroll position");
});

test("pending waits for usable geometry (pending is not absence)", async () => {
  const unfoldCalls: string[] = [];
  const { built } = buildFake({
    reveal: (call) => (call < 2 ? { kind: "pending" } : { kind: "found", bounds: { left: 0, top: 50, right: 10, bottom: 60 } }),
    unfoldCalls,
  });
  const job = built.host.places.restore({
    bufferId: "b",
    address: { machine: "m", path: "/p" },
    position: { kind: "anchor", anchor: { kind: "board-part", path: "/content/0" }, viewportOffset: { left: 0, top: 0 } },
  });
  await settled(job.snapshot.bind(job));
  assert.equal(job.snapshot(), "restored");
  assert.equal(unfoldCalls.length, 3);
  assert.deepEqual(built.host.viewport.metrics().position, { left: 0, top: 50 });
});

test("restoring an already visible anchor still places its exact saved offset", async () => {
  const { built } = buildFake({
    reveal: () => ({ kind: "found", bounds: { left: 80, top: 50, right: 90, bottom: 60 } }),
    unfoldCalls: [],
  });
  const job = built.host.places.restore({
    bufferId: "b", address: { machine: "m", path: "/p" },
    position: {
      kind: "anchor", anchor: { kind: "board-part", path: "/content/0" },
      viewportOffset: { left: 20, top: 15 },
    },
  });
  await settled(job.snapshot.bind(job));
  assert.equal(job.snapshot(), "restored");
  assert.deepEqual(built.host.viewport.metrics().position, { left: 60, top: 35 });
});

test("a confirmed-missing anchor defaults AT the same place", async () => {
  const { built, revealed } = buildFake({
    reveal: () => ({ kind: "missing" }),
    unfoldCalls: [],
  });
  const job = built.host.places.restore({
    bufferId: "b",
    address: { machine: "m", path: "/p" },
    position: { kind: "anchor", anchor: { kind: "board-part", path: "/content/9" }, viewportOffset: { left: 0, top: 0 } },
  });
  await settled(job.snapshot.bind(job));
  assert.equal(job.snapshot(), "defaulted");
  assert.deepEqual(revealed, []);
});

async function settled(snapshot: () => string): Promise<void> {
  const deadline = Date.now() + 5000;
  while (snapshot() === "pending-content" || snapshot() === "pending-layout") {
    assert.ok(Date.now() < deadline, "the restore job never settled");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
