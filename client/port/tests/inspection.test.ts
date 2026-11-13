// The figure inspection surface (S04): the zoom ladder, the modal
// context, and the readout's inspection field.

import assert from "node:assert/strict";
import { test } from "vitest";

import { ZOOM_MAX, ZOOM_MIN, zoomIn, zoomOut, zoomReset, zoomWheel } from "../src/lib/figures/inspection.ts";
import { createHostMatcher } from "../src/lib/input/host.ts";
import type { KeyInput } from "../src/lib/contracts/input.ts";
import { inspectionContext } from "../src/lib/input/contexts.ts";
import { buildReadout } from "../src/lib/observation.ts";

const stroke = (key: string): KeyInput => ({ key, ctrl: false, shift: false, alt: false, meta: false });

test("the zoom is the prototype's: start 1.2, x1.2 per + and -, x1.12 per wheel notch, 0.3 to 8", () => {
  // inst16.html:396-400: z = 1.2; + - multiply by 1.2; the wheel by
  // 1.12 per notch; = resets to 1.2 and the centre; clamped to 0.3-8.
  // (The quarter-step "half to five" ladder this replaces was never
  // the prototype's.)
  assert.equal(zoomReset(), 1.2);
  assert.ok(Math.abs(zoomIn(1.2) - 1.44) < 1e-9);
  assert.ok(Math.abs(zoomOut(1.44) - 1.2) < 1e-9);
  assert.ok(Math.abs(zoomWheel(1.2, -100) - 1.2 * 1.12) < 1e-9);
  assert.ok(Math.abs(zoomWheel(1.2, 100) - 1.2 / 1.12) < 1e-9);
  assert.equal(ZOOM_MIN, 0.3);
  assert.equal(ZOOM_MAX, 8);
  assert.equal(zoomIn(7.9), 8);
  assert.equal(zoomOut(0.31), 0.3);
});

test("the inspection context: its own keys act, everything else is consumed inert", () => {
  const ran: string[] = [];
  const host = createHostMatcher(
    inspectionContext({
      zoomIn: () => { ran.push("in"); },
      zoomOut: () => { ran.push("out"); },
      zoomReset: () => { ran.push("reset"); },
      close: () => { ran.push("close"); },
    }),
  );
  const apply = (key: string): string => {
    const resolution = host.stroke(stroke(key));
    return host.apply(resolution);
  };
  assert.equal(apply("+"), "consume");
  assert.equal(apply("-"), "consume");
  assert.equal(apply("="), "consume");
  assert.equal(apply("Escape"), "consume");
  // a board scroll key is consumed WITHOUT acting: no fall-through
  // to the board beneath, no silent scroll (scenario 4)
  assert.equal(apply("j"), "consume");
  assert.equal(apply("g"), "consume");
  assert.deepEqual(ran, ["in", "out", "reset", "close"]);
});

test("the readout carries the inspection while open, null when closed", () => {
  const base = {
    windowLabel: "main",
    link: null,
    bound: true,
    ask: null,
    surface: "board" as const,
    state: { kind: "alive" as const, active: true, content: { kind: "empty" as const } } as Parameters<typeof buildReadout>[0]["state"],
    diagnostics: [],
    doors: { jumpChip: null, delivered: [] },
    trail: { depth: 0, cursor: 0, entries: [], canBack: false, canForward: false },
    viewport: { left: 0, top: 0, width: 0, height: 0 },
    copy: null,
    follow: null,
    input: { pending: [], whichKey: null, tunables: { whichKeyDelayMs: 400, whichKeySuspendsTimeout: true, sequenceTimeoutMs: 1000 } },
  };
  assert.equal(buildReadout({ ...base, inspection: null }).inspection, null);
  const open = buildReadout({ ...base, inspection: { part: "/content/0", zoom: 1.5, x: 10, y: -4 } });
  assert.deepEqual(open.inspection, { part: "/content/0", zoom: 1.5, x: 10, y: -4 });
});

test("a figure shows by its file: images by extension, text as text, other bytes as binary", async () => {
  const { figureShape } = await import("../src/lib/figures/shape.ts");
  const encode = (text: string) => new TextEncoder().encode(text);
  assert.deepEqual(figureShape("figures/plot.png", new Uint8Array([137, 80])), { kind: "image", mime: "image/png" });
  assert.deepEqual(figureShape("a/b.JPG", new Uint8Array([1])), { kind: "image", mime: "image/jpeg" });
  assert.deepEqual(figureShape("x.svg", encode("<svg/>")), { kind: "image", mime: "image/svg+xml" });
  assert.deepEqual(figureShape("sweep.py", encode("def f():\n  pass\n")), { kind: "text", text: "def f():\n  pass\n" });
  assert.deepEqual(figureShape("change.diff", encode("+a\n-b\n")), { kind: "text", text: "+a\n-b\n" });
  assert.deepEqual(figureShape("blob.gz", new Uint8Array([0x1f, 0x8b, 0x08, 0x00, 0xff])), { kind: "binary" });
});
