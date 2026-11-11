import assert from "node:assert/strict";
import test from "node:test";
import { createViewport, SliceBufferHost } from "../src/components/Buffer.ts";
import type { BufferAccess, ContentFrame } from "../src/components/Buffer.ts";
import { commandContext } from "../src/lib/contracts/input.ts";
import { knownIssue } from "../src/lib/contracts/buffer.ts";
import type { AnchorData, BoardAnchor } from "../src/lib/contracts/places.ts";

type Data = { location: string; folded: boolean };
const frame = (location: string, folded = false): ContentFrame<Data> => ({ value: { location, folded }, targets: [] });
function deferred() {
  let finish!: () => void;
  const promise = new Promise<void>((resolve) => { finish = resolve; });
  return { promise, finish };
}
function harness() {
  let access!: BufferAccess<Data>;
  const controller = new SliceBufferHost<Data, Record<string, never>, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" }, viewport: createViewport(),
    diagnostics: { report: () => "diagnostic", unexpected: () => "diagnostic" },
    connect(buffer) {
      access = buffer;
      return {
        presentation: {}, input: commandContext(() => []), release() {},
        places: {
          captureDisplayed: () => ({ address: { machine: "fixture", path: "/board.yaml" }, position: { kind: "default" } }),
          request: async () => "applied",
          acceptsAnchor: (_data: AnchorData): _data is BoardAnchor => false,
          revealAnchor: () => ({ kind: "missing" }),
          defaultPosition: () => ({ left: 0, top: 0 }), subscribeChanges: () => () => {},
        },
      };
    },
  });
  const { host } = controller.wire();
  const shown = () => {
    const state = host.snapshot();
    if (state.kind !== "alive") return null;
    return state.content.kind === "available" ? state.content.value
      : state.content.kind === "pending" ? state.content.previous : null;
  };
  return {
    access, host, shown,
    fold() {
      // The same scenario now drives the displayed-state mechanism,
      // rather than using a queued content replacement for a redraw.
      access.updates.updateDisplayed((current) => ({
        ...current, value: { ...current.value, folded: true },
      }));
    },
  };
}

test("a displayed fold during a delayed open does not cancel that open", async (t) => {
  const h = harness(); t.after(() => h.host.destroy());
  await h.access.updates.refresh(() => ({ kind: "ready", value: frame("A") }));
  const gate = deferred();
  const opening = h.access.updates.refresh(async () => { await gate.promise; return { kind: "ready", value: frame("B") }; });
  h.fold();
  assert.equal(h.shown()?.value.folded, true);
  gate.finish();
  assert.equal(await opening, "applied");
  assert.equal(h.shown()?.value.location, "B");
});

test("displayed patches preserve pending state and only a newer load supersedes a load", async (t) => {
  const h = harness(); t.after(() => h.host.destroy());
  await h.access.updates.refresh(() => ({ kind: "ready", value: frame("A") }));
  const b = deferred(), c = deferred();
  const openingB = h.access.updates.refresh(async () => { await b.promise; return { kind: "ready", value: frame("B") }; });
  h.fold();
  const state = h.host.snapshot();
  assert.equal(state.kind === "alive" ? state.content.kind : state.kind, "pending");
  const openingC = h.access.updates.refresh(async () => { await c.promise; return { kind: "ready", value: frame("C") }; });
  h.fold();
  b.finish();
  assert.equal(await openingB, "superseded");
  c.finish();
  assert.equal(await openingC, "applied");
  assert.equal(h.shown()?.value.location, "C");
});

test("a failed delayed load retains the patched displayed frame and reports failure", async (t) => {
  const h = harness(); t.after(() => h.host.destroy());
  await h.access.updates.refresh(() => ({ kind: "ready", value: frame("A") }));
  const gate = deferred();
  const opening = h.access.updates.refresh(async () => {
    await gate.promise;
    return { kind: "failed", issue: knownIssue({ code: "asset-missing", message: "B missing" }) };
  });
  h.fold(); gate.finish();
  assert.equal(await opening, "failed");
  assert.deepEqual(h.shown()?.value, { location: "A", folded: true });
  const state = h.host.snapshot();
  assert.equal(state.kind === "alive" && state.content.kind === "available" ? state.content.lastFailure : null, "diagnostic");
  h.fold();
  const patched = h.host.snapshot();
  assert.equal(patched.kind === "alive" && patched.content.kind === "available" ? patched.content.lastFailure : null, "diagnostic",
    "a display update does not clear a source failure");
});

test("display updates cannot create content or revive a destroyed buffer", async () => {
  const h = harness();
  let calls = 0;
  const update = (current: Readonly<ContentFrame<Data>>) => { calls += 1; return current; };
  h.access.updates.updateDisplayed(update);
  assert.equal(calls, 0);
  await h.access.updates.refresh(() => ({ kind: "ready", value: frame("A") }));
  h.access.updates.updateDisplayed(update);
  assert.equal(calls, 1);
  h.host.destroy();
  h.access.updates.updateDisplayed(update);
  assert.equal(calls, 1);
  assert.equal(h.host.snapshot().kind, "destroyed");
});
