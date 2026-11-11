// Tests for the implemented input-context contract surface
// (input-context.ts, from t-36@r46 reference/input-context-contract.ts).
import test from "node:test";
import assert from "node:assert/strict";

import {
  commandContext,
  ctrl,
  describeCommands,
  formatSequence,
  formatStroke,
  key,
  sequence,
  shift,
  textEntryContext,
  type Action,
  type CommandBinding,
  type KeySequence,
} from "./input-context.ts";

function makeAction(id: string, label: string, run: () => void = () => {}): Action {
  return { id, label, run };
}

function binding(action: Action, keys: readonly KeySequence[]): CommandBinding {
  return { action, keys };
}

test("stroke constructors produce canonical exact patterns", () => {
  // Field-wise: the patterns also carry the constructor brand symbol,
  // so deep-equality is checked per field.
  const plain = key("g");
  assert.equal(plain.key, "g");
  assert.deepEqual(
    [plain.ctrl, plain.shift, plain.alt, plain.meta],
    [false, false, false, false],
  );
  const chord = ctrl("d");
  assert.equal(chord.key, "d");
  assert.deepEqual([chord.ctrl, chord.shift, chord.alt, chord.meta], [true, false, false, false]);
  const shifted = shift("Enter");
  assert.equal(shifted.key, "Enter");
  assert.deepEqual([shifted.ctrl, shifted.shift, shifted.alt, shifted.meta], [false, true, false, false]);
  const pair = sequence(key("g"), key("g"));
  assert.equal(pair.length, 2);
  assert.equal(pair[0]?.key, "g");
  assert.equal(pair[1]?.key, "g");
});

test("formatting: modifiers in canonical order, space reads spc, sequences join", () => {
  assert.equal(formatStroke(key(" ")), "spc");
  assert.equal(formatStroke(key("g")), "g");
  assert.equal(formatStroke(ctrl("d")), "Ctrl-d");
  assert.equal(formatStroke(shift("Enter")), "Shift-Enter");
  assert.equal(formatSequence(sequence(key("g"), key("g"))), "g g");
  assert.equal(formatSequence(sequence(shift("Enter"))), "Shift-Enter");
});

test("describeCommands projects the current declarations, never a frozen copy", () => {
  // The prototype reads the live commands() table in both the runner
  // and the help sheet (inst16.html:71-97); help must follow a table
  // change (history opening adds bindings) without a new context.
  const find = makeAction("find", "Find text");
  const follow = makeAction("follow", "Follow here");
  let bindings: readonly CommandBinding[] = [binding(find, [sequence(key("/"))])];
  const context = commandContext(() => bindings);
  assert.deepEqual(describeCommands(context), [
    { actionId: "find", label: "Find text", keyLabels: ["/"] },
  ]);
  bindings = [binding(follow, [sequence(key("f"))]), binding(find, [sequence(key("/"))])];
  assert.deepEqual(describeCommands(context), [
    { actionId: "follow", label: "Follow here", keyLabels: ["f"] },
    { actionId: "find", label: "Find text", keyLabels: ["/"] },
  ]);
});

test("alternatives of one binding list as separate key labels", () => {
  // The contract's readingExample: halfPageDown carries d and Ctrl-d
  // as alternatives for the SAME action.
  const halfPageDown = makeAction("half-page-down", "half page down");
  const context = commandContext(() => [
    binding(halfPageDown, [sequence(key("d")), sequence(ctrl("d"))]),
  ]);
  assert.deepEqual(describeCommands(context), [
    { actionId: "half-page-down", label: "half page down", keyLabels: ["d", "Ctrl-d"] },
  ]);
});

test("context kinds are commands and text-entry", () => {
  assert.equal(commandContext(() => []).kind, "commands");
  assert.equal(textEntryContext(() => []).kind, "text-entry");
});

test("action is one object with id, label and run", () => {
  let ran = 0;
  const follow = makeAction("follow", "Follow here", () => {
    ran += 1;
  });
  assert.equal(follow.id, "follow");
  assert.equal(follow.label, "Follow here");
  follow.run();
  assert.equal(ran, 1);
});
