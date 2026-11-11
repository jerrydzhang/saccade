import assert from "node:assert/strict";
import test from "node:test";
import { createViewport, SliceBufferHost } from "../src/components/Buffer.ts";
import type { BufferAccess } from "../src/components/Buffer.ts";
import { commandContext, key, sequence } from "../src/lib/contracts/input.ts";
import type { AnchorData, BoardAnchor } from "../src/lib/contracts/places.ts";

// Scenario 11: these are the actual base input leases, not re-stubbed
// release callbacks. A return targets the CURRENT valid base; stale
// release/restore callers must not displace a newer capture.
function harness() {
  const log: string[] = [];
  const context = (name: string) => commandContext(() => [{
    action: { id: name, label: name, run: () => { log.push(name); } },
    keys: [sequence(key("x"))],
  }]);
  let access!: BufferAccess<{ text: string }>;
  const base = context("base");
  const controller = new SliceBufferHost<{ text: string }, Record<string, never>, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" }, viewport: createViewport(),
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) {
      access = buffer;
      return {
        presentation: {}, input: base, release() {},
        places: {
          captureDisplayed: () => ({ address: { machine: "fixture", path: "/board.yaml" }, position: { kind: "default" } }),
          request: async () => "applied",
          acceptsAnchor: (_data: AnchorData): _data is BoardAnchor => false,
          revealAnchor: () => ({ kind: "missing" }),
          defaultPosition: () => ({ left: 0, top: 0 }),
          subscribeChanges: () => () => {},
        },
      };
    },
  });
  const { host } = controller.wire();
  host.activate();
  return { host, access, context, base, log, stroke: () => host.handleKey({ key: "x", ctrl: false, shift: false, alt: false, meta: false }) };
}

test("stale lease release and caller restore cannot displace a newer capture", () => {
  const h = harness();
  try {
    const a = h.access.interactions.capture(h.context("A"));
    const b = h.access.interactions.capture(h.context("B"));
    a.release();
    h.access.setReadingContext(h.base); // existing session exit path
    h.stroke();
    assert.deepEqual(h.log, ["B"]);
    b.release();
    h.stroke();
    assert.deepEqual(h.log, ["B", "base"]);
  } finally { h.host.destroy(); }
});

test("release returns to the current base, not the context captured at entry", () => {
  const h = harness();
  try {
    const lease = h.access.interactions.capture(h.context("interaction"));
    h.access.setReadingContext(h.context("new base"));
    h.stroke();
    lease.release();
    h.stroke();
    assert.deepEqual(h.log, ["interaction", "new base"]);
  } finally { h.host.destroy(); }
});

test("hidden and destroyed buffers cannot route input or be revived by a late release", () => {
  const h = harness();
  const lease = h.access.interactions.capture(h.context("interaction"));
  h.host.deactivate();
  assert.equal(h.stroke(), "allow-default");
  lease.release();
  h.access.setReadingContext(h.context("new base"));
  assert.equal(h.stroke(), "allow-default");
  assert.deepEqual(h.log, []);
  h.host.activate();
  h.stroke();
  assert.deepEqual(h.log, ["new base"]);
  const late = h.access.interactions.capture(h.context("late"));
  h.host.destroy();
  late.release();
  h.access.setReadingContext(h.base);
  assert.equal(h.stroke(), "allow-default");
  assert.equal(h.host.snapshot().kind, "destroyed");
  assert.deepEqual(h.log, ["new base"]);
});
