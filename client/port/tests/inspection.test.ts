// The figure inspection surface (S04): the zoom ladder, the modal
// context, and the readout's inspection field.

import assert from "node:assert/strict";
import { test } from "vitest";

import { ZOOM_MAX, ZOOM_MIN, zoomIn, zoomOut, zoomReset } from "../src/lib/figures/inspection.ts";
import { createHostMatcher } from "../src/lib/input/host.ts";
import type { KeyInput } from "../src/lib/contracts/input.ts";
import { inspectionContext } from "../src/lib/input/contexts.ts";
import { buildReadout } from "../src/lib/observation.ts";

const stroke = (key: string): KeyInput => ({ key, ctrl: false, shift: false, alt: false, meta: false });

test("the zoom ladder steps by quarters between half and five", () => {
  assert.equal(zoomReset(), 1);
  assert.equal(zoomIn(1), 1.25);
  assert.equal(zoomOut(1.25), 1);
  // the clamps hold: a step past the range keeps the value
  assert.equal(zoomIn(ZOOM_MAX), ZOOM_MAX);
  assert.equal(zoomOut(ZOOM_MIN), ZOOM_MIN);
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
  const open = buildReadout({ ...base, inspection: { part: "/content/0", zoom: 1.5 } });
  assert.deepEqual(open.inspection, { part: "/content/0", zoom: 1.5 });
});
