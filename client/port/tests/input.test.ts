// The port's input foundation (chunk A of the Board instrument
// slice): collision-at-load, the host matcher's chain and expiry, the
// window and reading contexts, and the picker controller. Pure tests
// - no DOM, no Svelte; fake schedulers and recorders only.

import assert from "node:assert/strict";
import { test } from "vitest";
import { commandContext, key, sequence } from "../src/lib/input/input-context.ts";
import { createMatcher, SEQUENCE_TIMEOUT_MS, validateNoCollisions } from "../src/lib/input/key-matcher.ts";
import { createHostMatcher, dispatchStroke, type HostScheduler } from "../src/lib/input/host.ts";
import { boardReadingContext, isPinnedChord, windowContext } from "../src/lib/input/contexts.ts";
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

test("the pinned window chords run back/forward through the host matcher", () => {
  const ran: string[] = [];
  const scheduler = manualScheduler();
  const host = createHostMatcher(
    windowContext({
      openPicker: () => {}, closeOverlays: () => {}, pickerIsOpen: () => false,
      trailBack: (): boolean => { ran.push("back"); return true; },
      trailForward: (): boolean => { ran.push("forward"); return true; },
      linkByLabels: () => {},
    enterFind: () => {}, unlink: () => {}, showHelp: () => {},
    }),
    { scheduler },
  );
  const chord = (k: string): KeyInput => ({ key: k, ctrl: true, shift: false, alt: false, meta: false });
  const back = host.stroke(chord("o"));
  const forward = host.stroke(chord("i"));
  assert.strictEqual(back.kind, "invoke");
  assert.strictEqual(back.action.id, "window.trail-back");
  assert.strictEqual(forward.kind, "invoke");
  assert.strictEqual(forward.action.id, "window.trail-forward");
  host.apply(back);
  host.apply(forward);
  assert.deepStrictEqual(ran, ["back", "forward"]);
});

test("the seam carve-out names exactly the prototype's six Ctrl chords", () => {
  // only bare ctrl+o/i/f/b/d/u pass the refusal (inst16.html:494-495);
  // shift-combos, alt/meta variants and other keys never do
  const pinned = (k: string, ctrl: boolean, shift: boolean, alt: boolean, meta: boolean) =>
    isPinnedChord({ key: k, ctrl, shift, alt, meta });
  assert.ok(pinned("o", true, false, false, false));
  assert.ok(pinned("i", true, false, false, false));
  for (const k of ["f", "b", "d", "u"]) assert.ok(pinned(k, true, false, false, false), k);
  assert.ok(!pinned("p", true, false, false, false));
  assert.ok(!pinned("c", true, false, false, false));
  assert.ok(!pinned("f", true, true, false, false));
  assert.ok(!pinned("o", true, true, false, false));
  assert.ok(!pinned("o", true, false, true, false));
  assert.ok(!pinned("o", true, false, false, true));
  assert.ok(!pinned("o", false, false, false, false));
});

test("the real port contexts validate at construction", () => {
  windowContext({ openPicker: () => {}, closeOverlays: () => {}, pickerIsOpen: () => false, trailBack: (): boolean => false, trailForward: (): boolean => false, linkByLabels: () => {},
    enterFind: () => {}, unlink: () => {}, showHelp: () => {} });
  boardReadingContext({ scrollToTop: () => {}, scrollToBottom: () => {}, scrollDown: () => {}, scrollUp: () => {}, scrollFraction: () => {}, enterSelect: () => {}, enterFind: () => {}, nextMatch: () => {}, prevMatch: () => {}, clearSearch: () => {} });
});

test("gg runs the scroll-top action through the host matcher; a lone g expires consumed", () => {
  const ran: string[] = [];
  const scheduler = manualScheduler();
  const host = createHostMatcher(
    boardReadingContext({
      scrollToTop: () => ran.push("top"),
      scrollToBottom: () => ran.push("bottom"),
      scrollDown: () => ran.push("down"),
      scrollUp: () => ran.push("up"),
      scrollFraction: () => {},
      enterSelect: () => {},
      enterFind: () => {},
      nextMatch: () => ran.push("next"),
      prevMatch: () => ran.push("prev"),
      clearSearch: () => ran.push("clear"),
    }),
    { scheduler },
  );
  assert.equal(host.apply(host.stroke(stroke("j"))), "consume");
  assert.deepEqual(ran, ["down"], "j scrolls a line down");
  assert.equal(host.apply(host.stroke(stroke("k"))), "consume");
  assert.deepEqual(ran, ["down", "up"], "k scrolls a line up");
  assert.equal(host.apply(host.stroke(stroke("g"))), "consume");
  assert.equal(host.apply(host.stroke(stroke("g"))), "consume");
  assert.deepEqual(ran, ["down", "up", "top"]);
  // the timeout drain: an abandoned prefix consumes without firing
  assert.equal(host.apply(host.stroke(stroke("g"))), "consume");
  scheduler.fire();
  assert.deepEqual(ran, ["down", "up", "top"]);
});

test("shift-G scrolls to bottom; unmatched keys are consumed, not native", () => {
  const ran: string[] = [];
  const host = createHostMatcher(
    boardReadingContext({ scrollToTop: () => ran.push("top"), scrollToBottom: () => ran.push("bottom"), scrollDown: () => ran.push("down"), scrollUp: () => ran.push("up"), scrollFraction: () => {}, enterSelect: () => {}, enterFind: () => {} , nextMatch: () => ran.push("next"), prevMatch: () => ran.push("prev"), clearSearch: () => ran.push("clear") }),
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
    boardReadingContext({ scrollToTop: () => ran.push("top"), scrollToBottom: () => ran.push("bottom"), scrollDown: () => ran.push("down"), scrollUp: () => ran.push("up"), scrollFraction: () => {}, enterSelect: () => {}, enterFind: () => {} , nextMatch: () => ran.push("next"), prevMatch: () => ran.push("prev"), clearSearch: () => ran.push("clear") }),
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
  let pickerOpen = false;
  const scheduler = manualScheduler();
  const windowHost = createHostMatcher(
    windowContext({
      openPicker: () => { pickerOpen = true; windowEvents.push("picker"); },
      closeOverlays: () => { pickerOpen = false; windowEvents.push("close"); },
      pickerIsOpen: () => pickerOpen,
      trailBack: (): boolean => { windowEvents.push("back"); return true; },
      trailForward: (): boolean => { windowEvents.push("forward"); return true; },
      linkByLabels: () => {},
    enterFind: () => {}, unlink: () => {}, showHelp: () => {},
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
  // Closed window surfaces cannot steal Escape from a buffer capture.
  dispatchStroke(windowHost, buffer, stroke("Escape"));
  assert.equal(bufferEvents.at(-1)?.key, "Escape");
  dispatchStroke(windowHost, buffer, stroke("/"));
  assert.equal(bufferEvents.at(-1)?.key, "/", "Find entry is content-owned");
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

test("m-552: a printable character carries its own Shift; letters keep their case; ctrl/alt/meta stay exact", async () => {
  const { strokeMatches, key: plain } = await import("../src/lib/input/input-context.ts");
  const at = (k: string, over: Partial<KeyInput> = {}): KeyInput => ({ key: k, ctrl: false, shift: false, alt: false, meta: false, ...over });
  // US Shift+= is "+" with Shift held; the keypad's + is "+" without it
  assert.ok(strokeMatches(plain("+"), at("+", { shift: true })), "Shift+= is +");
  assert.ok(strokeMatches(plain("+"), at("+")), "keypad + is +");
  assert.ok(strokeMatches(plain("?"), at("?", { shift: true })), "Shift+/ is ?");
  // letters: the character's case carries Shift
  assert.ok(strokeMatches(plain("N"), at("N", { shift: true })));
  assert.ok(!strokeMatches(plain("N"), at("n")), "N is not n");
  assert.ok(!strokeMatches(plain("n"), at("N", { shift: true })), "n is not N");
  assert.ok(!strokeMatches(plain("g"), at("G", { shift: true })), "g is not G");
  // modifiers other than Shift stay exact
  assert.ok(!strokeMatches(plain("+"), at("+", { ctrl: true })));
  assert.ok(!strokeMatches(plain("o"), at("o", { alt: true })));
  // a named key keeps Shift exact (Shift+Enter is not Enter)
  assert.ok(!strokeMatches(plain("Enter"), at("Enter", { shift: true })));
});
