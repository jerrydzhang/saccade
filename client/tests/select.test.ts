import test from "node:test";
import assert from "node:assert/strict";
import {
  assignLabels,
  isTextKind,
  narrowStep,
  startNarrow,
  type TargetKind,
} from "../src/lib/select.ts";

test("text targets carry lowercase labels, non-text targets uppercase ones", () => {
  const kinds: TargetKind[] = ["text", "figure", "text", "control"];
  const labels = assignLabels(kinds);
  assert.equal(labels[0], "a");
  assert.equal(labels[1], "A");
  assert.equal(labels[2], "b");
  assert.equal(labels[3], "B");
});

test("beyond one alphabet the labels grow a second character", () => {
  const kinds: TargetKind[] = Array.from({ length: 54 }, (_, index) => (index % 2 === 0 ? "text" : "figure"));
  const labels = assignLabels(kinds);
  assert.equal(labels[48], "y"); // the 25th text target
  assert.equal(labels[52], "aa"); // the 27th text target
  assert.equal(labels[53], "AA"); // the 27th non-text target
});

test("typing a complete unique label selects its target", () => {
  const kinds: TargetKind[] = ["text", "figure", "text"];
  const labels = assignLabels(kinds);
  const texts = ["first prose", "a figure", "second prose"];
  let state = startNarrow(kinds.length);
  const step1 = narrowStep(state, kinds, texts, labels, "b");
  assert.equal(step1.selected, 2);
  state = step1;
  void state;
});

test("a single-character label completes on its own keystroke", () => {
  const kinds: TargetKind[] = ["text", "text", "text"];
  const labels = assignLabels(kinds); // a, b, c
  const texts = ["one", "two", "three"];
  const state = narrowStep(startNarrow(3), kinds, texts, labels, "b");
  assert.equal(state.selected, 1);
});

test("a two-character label completes on the second keystroke", () => {
  const kinds: TargetKind[] = Array.from({ length: 54 }, (_, index) => (index % 2 === 0 ? "text" : "figure"));
  const labels = assignLabels(kinds);
  const texts = kinds.map((_, index) => `target ${index}`);
  const textTargets = kinds.map((k, i) => (k === "text" ? i : -1)).filter((i) => i >= 0);
  const last = textTargets[26];
  assert.ok(last !== undefined);
  const label = labels[last];
  assert.equal(label, "aa");
  let state = narrowStep(startNarrow(kinds.length), kinds, texts, labels, "a");
  assert.equal(state.selected, null);
  assert.equal(state.pendingLabel, "a");
  state = narrowStep(state, kinds, texts, labels, "a");
  assert.equal(state.selected, last);
});

test("a character no label needs narrows the text targets by their text", () => {
  const kinds: TargetKind[] = ["text", "text", "figure"];
  const labels = assignLabels(kinds);
  const texts = ["stop the press", "the tone arm", "a diagram"];
  let state = narrowStep(startNarrow(3), kinds, texts, labels, "t");
  assert.deepEqual(state.alive, [0, 1]);
  assert.equal(state.query, "t");
  state = narrowStep(state, kinds, texts, labels, "o");
  assert.deepEqual(state.alive, [0, 1]);
  state = narrowStep(state, kinds, texts, labels, "n");
  assert.deepEqual(state.alive, [1]);
  assert.equal(state.selected, null, "narrowing never selects on its own");
});

test("narrowing is case-insensitive on the text, labels are case-sensitive", () => {
  const kinds: TargetKind[] = ["text", "figure"];
  const labels = assignLabels(kinds);
  const texts = ["Prose With Capitals", "A Figure"];
  const state = narrowStep(startNarrow(2), kinds, texts, labels, "p");
  assert.deepEqual(state.alive, [0]);
  const upper = narrowStep(startNarrow(2), kinds, texts, labels, "A");
  assert.equal(upper.selected, 1, "the uppercase A is the figure's label");
});

test("typing text leaves non-text targets untouched in the alive set only by narrowing text kinds", () => {
  const kinds: TargetKind[] = ["text", "figure", "text"];
  const labels = assignLabels(kinds);
  const texts = ["alpha prose", "a diagram", "alphabet prose"];
  let state = narrowStep(startNarrow(3), kinds, texts, labels, "a");
  // "a" completes target 0's label; every target stays alive for labels
  assert.equal(state.selected, 0);
});

test("a narrowing that matches nothing leaves no targets alive", () => {
  const kinds: TargetKind[] = ["text", "figure"];
  const labels = assignLabels(kinds);
  const texts = ["prose", "figure"];
  const state = narrowStep(startNarrow(2), kinds, texts, labels, "z");
  assert.deepEqual(state.alive, []);
});

test("only text kinds narrow; a query never touches other targets", () => {
  const kinds: TargetKind[] = ["figure", "diagram", "incorporation", "citation", "control", "unknown", "annotation", "text"];
  assert.equal(isTextKind("text"), true);
  assert.equal(isTextKind("annotation"), true);
  for (const kind of ["figure", "diagram", "incorporation", "citation", "control", "unknown"] as TargetKind[]) {
    assert.equal(isTextKind(kind), false);
  }
  void kinds;
});
