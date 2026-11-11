import assert from "node:assert/strict";
import test from "node:test";
import { createViewport, SliceBufferHost } from "../src/components/Buffer.ts";
import type { BufferAccess } from "../src/components/Buffer.ts";
import { commandContext } from "../src/lib/contracts/input.ts";
import type { AnchorData, BoardAnchor } from "../src/lib/contracts/places.ts";

// The geometry consumer compares bounds with scroll offsets and sends
// them to Viewport.reveal, so DOM measurements must share that content
// coordinate frame. Include a border and nonzero horizontal scroll.
test("part bounds and viewport reveal share content coordinates after scrolling", (t) => {
  class FakeElement {
    scrollTop = 500;
    scrollLeft = 40;
    clientTop = 2;
    clientLeft = 3;
    clientWidth = 300;
    clientHeight = 200;
    getBoundingClientRect() { return { left: 50, top: 100 }; }
    querySelector() { return part; }
  }
  const scroller = new FakeElement();
  const part = new FakeElement();
  part.getBoundingClientRect = () => ({
    left: 50 + scroller.clientLeft + 80 - scroller.scrollLeft,
    top: 100 + scroller.clientTop + 550 - scroller.scrollTop,
    right: 50 + scroller.clientLeft + 180 - scroller.scrollLeft,
    bottom: 100 + scroller.clientTop + 590 - scroller.scrollTop,
  });
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "HTMLElement");
  Object.defineProperty(globalThis, "HTMLElement", { value: FakeElement, configurable: true });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, "HTMLElement", descriptor);
    else Reflect.deleteProperty(globalThis, "HTMLElement");
  });
  const viewport = createViewport();
  viewport.attachElement(scroller as unknown as HTMLElement);
  let access!: BufferAccess<{ text: string }>;
  const controller = new SliceBufferHost<{ text: string }, Record<string, never>, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" }, viewport,
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) {
      access = buffer;
      return {
        presentation: {}, input: commandContext(() => []), release() {},
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
  const built = controller.wire();
  t.after(() => built.host.destroy());
  const expected = { left: 80, top: 550, right: 180, bottom: 590 };
  assert.deepEqual(access.geometry.bounds("/content/0"), expected);
  viewport.reveal(expected);
  assert.equal(scroller.scrollTop, 500, "already visible part must not jump");
  scroller.scrollTop = 0;
  assert.deepEqual(access.geometry.bounds("/content/0"), expected, "scrolling does not change content bounds");
  viewport.reveal(expected);
  assert.equal(scroller.scrollTop, 390, "reveal scrolls to the measured content bounds");
});
