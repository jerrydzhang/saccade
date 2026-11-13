// Tests for the key matcher (key-matcher.ts): scenario 10 clauses -
// unmatched-key consumption, per-context declarations, sequences
// through one matcher, timeout semantics (SEQUENCE_TIMEOUT_MS=1000,
// abandon pending at expiry, received keys then match their own
// bindings or are consumed), handoff reset, help reading the same
// declarations.
import test from "node:test";
import assert from "node:assert/strict";

import {
  commandContext,
  ctrl,
  describeCommands,
  key,
  sequence,
  shift,
  textEntryContext,
  type Action,
  type CommandBinding,
  type InputContext,
  type KeyInput,
  type KeySequence,
} from "./input-context.ts";
import { createMatcher, SEQUENCE_TIMEOUT_MS } from "./key-matcher.ts";

function makeAction(id: string): Action {
  return { id, label: id, run: () => {} };
}

function binding(action: Action, keys: readonly KeySequence[]): CommandBinding {
  return { action, keys };
}

function stroke(k: string, mods: Partial<Exclude<KeyInput, "key">> = {}): KeyInput {
  return { key: k, ctrl: false, shift: false, alt: false, meta: false, ...mods };
}

// The contract's readingExample (t-36@r46): j scroll, d/Ctrl-d half
// page, g g top.
function readingContext(): InputContext {
  const scrollDown = makeAction("scroll-down");
  const halfPageDown = makeAction("half-page-down");
  const goToStart = makeAction("go-to-start");
  return commandContext(() => [
    binding(scrollDown, [sequence(key("j"))]),
    binding(halfPageDown, [sequence(key("d")), sequence(ctrl("d"))]),
    binding(goToStart, [sequence(key("g"), key("g"))]),
  ]);
}

// The contract's findExample: Escape leaveFind, Enter nextMatch,
// Shift-Enter previousMatch.
function findContext(): InputContext {
  const leaveFind = makeAction("leave-find");
  const nextMatch = makeAction("next-match");
  const previousMatch = makeAction("previous-match");
  return textEntryContext(() => [
    binding(leaveFind, [sequence(key("Escape"))]),
    binding(nextMatch, [sequence(key("Enter"))]),
    binding(previousMatch, [sequence(shift("Enter"))]),
  ]);
}

test("the ratified timeout value is 1000 ms", () => {
  assert.equal(SEQUENCE_TIMEOUT_MS, 1000);
});

test("a single-stroke binding invokes its own action object", () => {
  const matcher = createMatcher(readingContext());
  const resolution = matcher.resolve(stroke("j"));
  assert.equal(resolution.kind, "invoke");
  assert.equal(resolution.kind === "invoke" && resolution.action.id, "scroll-down");
});

test("unmatched application keys are consumed in the command context", () => {
  // Scenario 10: never dispatched to inactive buffers underneath. The
  // prototype's keydown chain preventDefaults every branch
  // (inst16.html:458-516).
  const matcher = createMatcher(readingContext());
  assert.deepEqual(matcher.resolve(stroke("x")), { kind: "consume" });
  assert.deepEqual(matcher.resolve(stroke("1")), { kind: "consume" });
});

test("text entry matches declared commands first, other typing edits natively", () => {
  // Scenario 10: other typing never reaches the board keymap. The
  // contract invariant: text-entry contexts never try the underlying
  // reading context.
  const matcher = createMatcher(findContext());
  const enter = matcher.resolve(stroke("Enter"));
  assert.equal(enter.kind === "invoke" && enter.action.id, "next-match");
  const shifted = matcher.resolve(stroke("Enter", { shift: true }));
  assert.equal(shifted.kind === "invoke" && shifted.action.id, "previous-match");
  const escape = matcher.resolve(stroke("Escape"));
  assert.equal(escape.kind === "invoke" && escape.action.id, "leave-find");
  assert.deepEqual(matcher.resolve(stroke("a")), { kind: "native-editing" });
  assert.deepEqual(matcher.resolve(stroke(" ")), { kind: "native-editing" });
});

test("alternatives of one action both invoke it (d and Ctrl-d)", () => {
  const matcher = createMatcher(readingContext());
  const plain = matcher.resolve(stroke("d"));
  const chord = matcher.resolve(stroke("d", { ctrl: true }));
  assert.equal(plain.kind === "invoke" && plain.action.id, "half-page-down");
  assert.equal(chord.kind === "invoke" && chord.action.id, "half-page-down");
});

test("a two-key sequence never fires early and completes exactly", () => {
  // Scenario 10: sequences run through the shared matcher; the
  // prototype's gg waits for its second g (inst16.html:499).
  const matcher = createMatcher(readingContext());
  assert.deepEqual(matcher.resolve(stroke("g")), { kind: "prefix" });
  const done = matcher.resolve(stroke("g"));
  assert.equal(done.kind === "invoke" && done.action.id, "go-to-start");
  // The buffer is spent: the next g starts a fresh pending.
  assert.deepEqual(matcher.resolve(stroke("g")), { kind: "prefix" });
});

test("an unrelated key during a pending sequence resolves on its own; pending survives", () => {
  // Executed prototype behavior (inst16.html:499): non-g keys fall
  // through their own branches while gPending stays.
  const matcher = createMatcher(readingContext());
  assert.deepEqual(matcher.resolve(stroke("g")), { kind: "prefix" });
  const j = matcher.resolve(stroke("j"));
  assert.equal(j.kind === "invoke" && j.action.id, "scroll-down");
  // The pending gg was not disturbed: the next g completes it.
  const done = matcher.resolve(stroke("g"));
  assert.equal(done.kind === "invoke" && done.action.id, "go-to-start");
});

test("an unmatched key during a pending sequence is consumed; pending survives", () => {
  const matcher = createMatcher(readingContext());
  assert.deepEqual(matcher.resolve(stroke("g")), { kind: "prefix" });
  assert.deepEqual(matcher.resolve(stroke("x")), { kind: "consume" });
  const done = matcher.resolve(stroke("g"));
  assert.equal(done.kind === "invoke" && done.action.id, "go-to-start");
});

test("a stroke that is both a completion and a prefix waits, and expiry fires its own binding", () => {
  // d alone and d d declared: the first d waits (could extend). The
  // ratified expiry: received keys then match their own bindings if
  // declared (scenario 10).
  let clock = 0;
  const matcher = createMatcher(readingContext(), { now: () => clock });
  const both = commandContext(() => [
    binding(makeAction("d-own"), [sequence(key("d"))]),
    binding(makeAction("d-d"), [sequence(key("d"), key("d"))]),
  ]);
  const bothMatcher = createMatcher(both, { now: () => clock });
  assert.deepEqual(bothMatcher.resolve(stroke("d")), { kind: "prefix" });
  clock += SEQUENCE_TIMEOUT_MS;
  const drained = bothMatcher.expire();
  assert.equal(drained.length, 1);
  assert.equal(drained[0]?.kind === "invoke" && drained[0].action.id, "d-own");
  // The plain matcher (d only declared alone) never waits.
  assert.equal(matcher.resolve(stroke("d")).kind, "invoke");
});

test("expire before the timeout does nothing; at it, the pending is abandoned and consumed", () => {
  // Adapter A1: the app abandons a pending sequence at
  // SEQUENCE_TIMEOUT_MS=1000; the received keys then match their own
  // bindings or are consumed. gg's g alone is undeclared -> consumed.
  let clock = 0;
  const matcher = createMatcher(readingContext(), { now: () => clock });
  assert.deepEqual(matcher.resolve(stroke("g")), { kind: "prefix" });
  clock += SEQUENCE_TIMEOUT_MS - 1;
  assert.deepEqual(matcher.expire(), []);
  // Still pending just before expiry: the next g would complete.
  clock += 1;
  const drained = matcher.expire();
  assert.deepEqual(drained, [{ kind: "consume" }]);
  // The buffer is spent: a later g starts a fresh pending, it does
  // not complete the abandoned one.
  clock += 2000;
  assert.deepEqual(matcher.resolve(stroke("g")), { kind: "prefix" });
});

test("expiry of a fully typed but extendable sequence fires the completed spelling", () => {
  let clock = 0;
  const context = commandContext(() => [
    binding(makeAction("d-d"), [sequence(key("d"), key("d"))]),
    binding(makeAction("d-d-d"), [sequence(key("d"), key("d"), key("d"))]),
  ]);
  const matcher = createMatcher(context, { now: () => clock });
  matcher.resolve(stroke("d"));
  assert.deepEqual(matcher.resolve(stroke("d")), { kind: "prefix" });
  clock += SEQUENCE_TIMEOUT_MS;
  const drained = matcher.expire();
  assert.equal(drained.length, 1);
  assert.equal(drained[0]?.kind === "invoke" && drained[0].action.id, "d-d");
});

test("reset drops a pending sequence without firing (never survives a handoff)", () => {
  // Scenario 10: a partial sequence never survives a handoff; the
  // draft: replacing/releasing a context must not leak its pending
  // prefix into another.
  const matcher = createMatcher(readingContext());
  assert.deepEqual(matcher.resolve(stroke("g")), { kind: "prefix" });
  matcher.reset();
  const after = matcher.resolve(stroke("g"));
  assert.deepEqual(after, { kind: "prefix" });
});

test("matchers hold per-context state, not global state", () => {
  const reading = createMatcher(readingContext());
  const find = createMatcher(findContext());
  assert.deepEqual(reading.resolve(stroke("g")), { kind: "prefix" });
  // The find matcher is untouched by the reading matcher's pending:
  // Enter invokes next-match immediately.
  const enter = find.resolve(stroke("Enter"));
  assert.equal(enter.kind === "invoke" && enter.action.id, "next-match");
  // And the reading matcher's own sequence still completes.
  const done = reading.resolve(stroke("g"));
  assert.equal(done.kind === "invoke" && done.action.id, "go-to-start");
});

test("help and execution read the same current declarations", () => {
  // inst16.html:90-97: showHelp and runCmd both call commands().
  const scrollDown = makeAction("scroll-down");
  const jumpChange = makeAction("next-change");
  let bindings: readonly CommandBinding[] = [binding(scrollDown, [sequence(key("j"))])];
  const context = commandContext(() => bindings);
  const matcher = createMatcher(context);
  assert.deepEqual(matcher.resolve(stroke("n")), { kind: "consume" });
  assert.deepEqual(
    describeCommands(context),
    [{ actionId: "scroll-down", label: "scroll-down", keyLabels: ["j"] }],
  );
  // History opens and adds bindings: help lists them, the matcher
  // invokes them, with no new context or matcher.
  bindings = [binding(scrollDown, [sequence(key("j"))]), binding(jumpChange, [sequence(key("n"))])];
  assert.deepEqual(describeCommands(context).map((entry) => entry.actionId), ["scroll-down", "next-change"]);
  const next = matcher.resolve(stroke("n"));
  assert.equal(next.kind === "invoke" && next.action.id, "next-change");
});

test("a duplicate sequence resolves to the first declared binding (staged fallback)", () => {
  // Collision-as-validation-error is a draft invariant, not ratified;
  // the staged fallback is declaration order, recorded in README.md.
  const first = makeAction("first");
  const second = makeAction("second");
  const context = commandContext(() => [
    binding(first, [sequence(key("z"))]),
    binding(second, [sequence(key("z"))]),
  ]);
  const matcher = createMatcher(context);
  const resolution = matcher.resolve(stroke("z"));
  assert.equal(resolution.kind === "invoke" && resolution.action.id, "first");
});

test("exact modifiers: a shifted stroke does not match an unshifted pattern", () => {
  // Modifier normalization is draft-open; the matcher invents none and
  // matches fields exactly (strokeMatches, input-context.ts).
  const matcher = createMatcher(readingContext());
  assert.deepEqual(matcher.resolve(stroke("j", { shift: true })), { kind: "consume" });
});
