// The anchor placement (chunk D1): reveal with fold-unfolding,
// pending-waits (pending is not absence), confirmed-missing default
// position AT the same place, scroll cancellation, and the part
// ancestor walk. Pure tests with fakes - no DOM.

import assert from "node:assert/strict";
import { test } from "vitest";
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
  vetAfterRequest?: boolean;
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
  let appliedRequests = 0;
  const attachment: InstrumentAttachment<Record<string, never>, BoardAnchor> = {
    presentation: {},
    input: commandContext(() => []),
    places: {
      captureDisplayed: () => ({ address: { machine: "m", path: "/p" }, position: { kind: "default" } }),
      request: async () => {
        appliedRequests += 1;
        return "applied";
      },
      // when the vet must be destination-aware, the anchor is only
      // acceptable AFTER a request applied: a pre-request vet fails
      // this check (the S08 regression shape)
      acceptsAnchor: behaviour.vetAfterRequest === true
        ? (data: AnchorData): data is BoardAnchor =>
            appliedRequests > 0 && (data as { kind?: string }).kind === "board-part"
        : (data: AnchorData): data is BoardAnchor =>
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

test("a cross-board anchored open vets the anchor against the destination, not the display", async () => {
  // the S08 live finding: a citation names a part of the board it
  // opens. The anchor exists in the DESTINATION model and not in the
  // currently displayed one - the pre-request vet rejected every
  // such open, so the citation landed with its fold never unfolded.
  // The fake only accepts an anchor AFTER a request applied: a
  // pre-request vet fails the job, the destination-aware vet restores
  // it with the reveal.
  const unfoldCalls: string[] = [];
  const { built } = buildFake({
    reveal: () => ({ kind: "found", bounds: { left: 0, top: 20, right: 100, bottom: 40 } }),
    unfoldCalls,
    vetAfterRequest: true,
  });
  const job = built.host.places.restore({
    bufferId: "b",
    address: { machine: "m", path: "/destination" },
    position: { kind: "anchor", anchor: { kind: "board-part", path: "/content/0/content/0" }, viewportOffset: { left: 0, top: 0 } },
  });
  await settled(job.snapshot.bind(job));
  assert.equal(job.snapshot(), "restored", "the anchor of the opened board reveals instead of being rejected pre-open");
  assert.deepEqual(unfoldCalls, ["/content/0/content/0"], "the reveal ran after the open applied");
});

test("an anchor missing at the destination defaults AT the same requested place", async () => {
  // the vet runs AFTER the open applies: a destination-missing anchor
  // is confirmed missing - default position at the same place with
  // the rejection reported, never a silent nothing (scenario 7)
  const unfoldCalls: string[] = [];
  const { built } = buildFake({
    reveal: () => ({ kind: "found", bounds: { left: 0, top: 0, right: 10, bottom: 10 } }),
    unfoldCalls,
    vetAfterRequest: true,
  });
  const job = built.host.places.restore({
    bufferId: "b",
    address: { machine: "m", path: "/p" },
    position: { kind: "anchor", anchor: { kind: "not-a-board-part" } as unknown as BoardAnchor, viewportOffset: { left: 0, top: 0 } },
  });
  await settled(job.snapshot.bind(job));
  assert.equal(job.snapshot(), "defaulted");
  assert.deepEqual(unfoldCalls, [], "no reveal for an anchor the destination does not carry");
});

async function settled(snapshot: () => string): Promise<void> {
  const deadline = Date.now() + 5000;
  while (snapshot() === "pending-content" || snapshot() === "pending-layout") {
    assert.ok(Date.now() < deadline, "the restore job never settled");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
