// The port's input foundation (chunk A of the Board instrument
// slice): collision-at-load, the host matcher's chain and expiry, the
// window and reading contexts, and the picker controller. Pure tests
// - no DOM, no Svelte; fake schedulers and recorders only.

import assert from "node:assert/strict";
import test from "node:test";
import { commandContext, key, sequence } from "../src/lib/input/input-context.ts";
import { createMatcher, SEQUENCE_TIMEOUT_MS, validateNoCollisions } from "../src/lib/input/key-matcher.ts";
import { createHostMatcher, dispatchStroke, type HostScheduler } from "../src/lib/input/host.ts";
import { boardReadingContext, windowContext } from "../src/lib/input/contexts.ts";
import { createPickerController } from "../src/lib/input/picker.ts";
import type { KeyDisposition, KeyInput } from "../src/lib/input/input-context.ts";

const stroke = (k: string, over: Partial<KeyInput> = {}): KeyInput => ({
  key: k,
  ctrl: false,
  shift: false,
  alt: false,
  meta: false,
  ...over,
});

// a scheduler that records and runs nothing without being told to
function manualScheduler(): HostScheduler & { fire(): void } {
  let job: (() => void) | null = null;
  return {
    schedule(_ms, run) {
      job = run;
      return 1;
    },
    cancel() {
      job = null;
    },
    fire() {
      const held = job;
      job = null;
      if (held !== null) held();
    },
  };
}

test("collisions between different actions are errors at load", () => {
  const clash = commandContext(() => [
    {
      action: { id: "a.one", label: "one", run: () => {} },
      keys: [sequence(key("x"))],
    },
    {
      action: { id: "a.two", label: "two", run: () => {} },
      keys: [sequence(key("x"))],
    },
  ]);
  assert.throws(() => validateNoCollisions(clash), /input collision at load/);
  // alternatives for the SAME action stay legal
  const alternatives = commandContext(() => [
    {
      action: { id: "a.one", label: "one", run: () => {} },
      keys: [sequence(key("x")), sequence(key("y"), key("y"))],
    },
  ]);
  validateNoCollisions(alternatives);
});

test("the real port contexts validate at construction", () => {
  windowContext({ openPicker: () => {}, closeOverlays: () => {} });
  boardReadingContext({ scrollToTop: () => {}, scrollToBottom: () => {} });
});

test("gg runs the scroll-top action through the host matcher; a lone g expires consumed", () => {
  const ran: string[] = [];
  const scheduler = manualScheduler();
  const host = createHostMatcher(
    boardReadingContext({
      scrollToTop: () => ran.push("top"),
      scrollToBottom: () => ran.push("bottom"),
    }),
    { scheduler },
  );
  assert.equal(host.apply(host.stroke(stroke("g"))), "consume");
  assert.equal(host.apply(host.stroke(stroke("g"))), "consume");
  assert.deepEqual(ran, ["top"]);
  // the timeout drain: an abandoned prefix consumes without firing
  assert.equal(host.apply(host.stroke(stroke("g"))), "consume");
  scheduler.fire();
  assert.deepEqual(ran, ["top"]);
});

test("shift-G scrolls to bottom; unmatched keys are consumed, not native", () => {
  const ran: string[] = [];
  const host = createHostMatcher(
    boardReadingContext({ scrollToTop: () => ran.push("top"), scrollToBottom: () => ran.push("bottom") }),
    { scheduler: manualScheduler() },
  );
  assert.equal(host.apply(host.stroke(stroke("G", { shift: true }))), "consume");
  assert.deepEqual(ran, ["bottom"]);
  assert.equal(host.apply(host.stroke(stroke("q"))), "consume");
});

test("the matcher's expiry fires the completed spelling when it is its own binding", () => {
  let clock = 0;
  const ran: string[] = [];
  const matcher = createMatcher(
    boardReadingContext({ scrollToTop: () => ran.push("top"), scrollToBottom: () => ran.push("bottom") }),
    { now: () => clock },
  );
  // a binding of 'g' alone would make g-g ambiguous; here only gg is
  // declared, so expiry of a lone g consumes (scenario 10: else are
  // consumed)
  assert.equal(matcher.resolve(stroke("g")).kind, "prefix");
  clock += SEQUENCE_TIMEOUT_MS + 1;
  const drained = matcher.expire();
  assert.equal(drained.length, 1);
  assert.equal(drained[0]?.kind, "consume");
  assert.deepEqual(ran, []);
});

test("the window chain: space-space opens the picker at the window level; other keys reach the buffer", () => {
  const windowEvents: string[] = [];
  const bufferEvents: KeyInput[] = [];
  const scheduler = manualScheduler();
  const windowHost = createHostMatcher(
    windowContext({
      openPicker: () => windowEvents.push("picker"),
      closeOverlays: () => windowEvents.push("close"),
    }),
    { scheduler },
  );
  const buffer = {
    handleKey(input: KeyInput): KeyDisposition {
      bufferEvents.push(input);
      return "consume";
    },
  };
  // space space: prefix then invoke - all at the window level
  assert.equal(dispatchStroke(windowHost, buffer, stroke(" ")), "consume");
  assert.equal(dispatchStroke(windowHost, buffer, stroke(" ")), "consume");
  assert.deepEqual(windowEvents, ["picker"]);
  assert.equal(bufferEvents.length, 0);
  // an unmatched key flows to the buffer, whose answer is final
  assert.equal(dispatchStroke(windowHost, buffer, stroke("j")), "consume");
  assert.equal(bufferEvents.length, 1);
  assert.equal(bufferEvents[0]?.key, "j");
  // escape closes overlays at the window level
  assert.equal(dispatchStroke(windowHost, buffer, stroke("Escape")), "consume");
  assert.deepEqual(windowEvents, ["picker", "close"]);
});

test("the picker controller: submit goes through the one open funnel, then closes", () => {
  const opened: string[] = [];
  const picker = createPickerController((spec) => opened.push(spec));
  const seen: boolean[] = [];
  picker.subscribe((state) => seen.push(state.open));
  picker.open();
  picker.submit("  host:8811/repo/t-1  ");
  picker.open();
  picker.close();
  assert.deepEqual(opened, ["host:8811/repo/t-1"]);
  assert.deepEqual(seen, [false, true, false, true, false]);
  // an empty submit never reaches the open funnel
  picker.open();
  picker.submit("   ");
  assert.deepEqual(opened, ["host:8811/repo/t-1"]);
});
