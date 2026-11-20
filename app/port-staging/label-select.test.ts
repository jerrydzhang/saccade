// Tests for label-select narrowing (label-select.ts): pending-label
// matching, the case-fold rule, text-target narrowing, exact-single
// selection (scenario 12; the case fold is scenario 19's held break,
// green as S12 via commit b8240b5).
import test from "node:test";
import assert from "node:assert/strict";

import {
  assignLabels,
  isTextKind,
  narrowStep,
  startNarrow,
  type TargetKind,
} from "./label-select.ts";

test("text targets are the prose kinds", () => {
  assert.equal(isTextKind("text"), true);
  assert.equal(isTextKind("annotation"), true);
  assert.equal(isTextKind("figure"), false);
  assert.equal(isTextKind("group"), false);
});

test("labels: lowercase on text targets, uppercase on the rest, singles then combos", () => {
  const kinds: TargetKind[] = ["text", "group", "text", "figure"];
  const labels = assignLabels(kinds);
  assert.deepEqual(labels, ["a", "A", "b", "B"]);
  // Beyond the single letters, two-letter combinations in stable order.
  const many: TargetKind[] = new Array<TargetKind>(28).fill("group");
  const combo = assignLabels(many);
  assert.deepEqual(combo.slice(0, 3), ["A", "B", "C"]);
  assert.deepEqual(combo.slice(26, 28), ["AA", "AB"]);
});

test("a completed label selects its target (exact-single)", () => {
  // hintKey's last branch (inst16.html:106): one label left and equal
  // to the typed prefix.
  const kinds: TargetKind[] = ["group", "text"];
  const labels = assignLabels(kinds);
  let state = startNarrow(2);
  const step = narrowStep(state, kinds, ["first prose", "second prose"], labels, "A");
  assert.equal(step.selected, 0);
  assert.equal(step.pendingLabel, "");
  state = step;
  assert.equal(narrowStep(state, kinds, ["first prose", "second prose"], labels, "a").selected, 1);
});

test("a two-letter label pends on the first letter and selects on the second", () => {
  // A partial label never selects early (the prototype's prefix
  // narrowing, inst16.html:106); a full label selects (exact-single).
  const kinds: TargetKind[] = ["group", "group"];
  const labels = ["AA", "AB"];
  const state = startNarrow(2);
  const first = narrowStep(state, kinds, ["", ""], labels, "A");
  assert.deepEqual(first, { alive: [0, 1], pendingLabel: "A", query: "", selected: null });
  const second = narrowStep(first, kinds, ["", ""], labels, "A");
  assert.equal(second.selected, 0);
  const other = narrowStep(first, kinds, ["", ""], labels, "B");
  assert.equal(other.selected, 1);
});

test("the case-fold rule: a letter naming no label in its own case tries the folded case", () => {
  // b8240b5; no label ever needs Shift. Uppercase labels exist, the
  // typed lowercase letter names none in its own case, the folded
  // case matches.
  const kinds: TargetKind[] = ["group"];
  const labels = assignLabels(kinds); // ["A"]
  const state = startNarrow(1);
  const step = narrowStep(state, kinds, [""], labels, "a");
  assert.equal(step.selected, 0);
  assert.equal(step.pendingLabel, "");
  // And the mirror: a typed uppercase letter reaches lowercase labels.
  const textKinds: TargetKind[] = ["text"];
  const textLabels = assignLabels(textKinds); // ["a"]
  const textState = startNarrow(1);
  const textStep = narrowStep(textState, textKinds, ["prose"], textLabels, "A");
  assert.equal(textStep.selected, 0);
  assert.equal(textStep.pendingLabel, "");
});

test("the fold applies mid-label: a mixed-case typing completes a label", () => {
  // The pending prefix is carried into the folded retry
  // (state.pendingLabel + flipped, b8240b5).
  const kinds: TargetKind[] = ["group", "group"];
  const labels = ["AB", "AC"];
  let state = startNarrow(2);
  // "a" names no label in its own case ("AB" does not start with a
  // lowercase a): the fold lifts it to "A" and both labels pend.
  state = narrowStep(state, kinds, ["", ""], labels, "a");
  assert.deepEqual(state, { alive: [0, 1], pendingLabel: "A", query: "", selected: null });
  // "b" completes "Ab" only through the fold to "AB".
  const done = narrowStep(state, kinds, ["", ""], labels, "b");
  assert.equal(done.selected, 0);
  const other = narrowStep(state, kinds, ["", ""], labels, "c");
  assert.equal(other.selected, 1);
});

test("only a letter that names no label at all narrows the text targets", () => {
  // The narrowing falls through to the text search only after both
  // cases fail (the mandate's "only then narrows text targets").
  const kinds: TargetKind[] = ["group", "text", "text"];
  const labels = assignLabels(kinds); // A, a, b
  let state = startNarrow(3);
  // "c" names no label in either case: it narrows text by "c"; the
  // group target is not a text target and falls out.
  state = narrowStep(state, kinds, ["", "the cited account", "a citation"], labels, "c");
  assert.equal(state.pendingLabel, "");
  assert.equal(state.query, "c");
  assert.deepEqual(state.alive, [1, 2]);
  // "a" matches the surviving target's own label instead: labels win
  // over text narrowing.
  const picked = narrowStep(state, kinds, ["", "the cited account", "a citation"], labels, "a");
  assert.equal(picked.selected, 1);
  assert.equal(picked.query, "c");
});

test("text narrowing accumulates case-insensitively over the alive targets", () => {
  const kinds: TargetKind[] = ["text", "text", "text"];
  // Labels that no query letter folds onto, so the steps below are
  // pure text narrowing.
  const labels = ["A", "B", "D"];
  let state = startNarrow(3);
  state = narrowStep(state, kinds, ["The Citing Account", "a citation", "another account"], labels, "c");
  // "c" names no label in either case (the fold reaches "C", no such
  // label): all three prose texts contain it.
  assert.deepEqual(state.alive, [0, 1, 2]);
  assert.equal(state.query, "c");
  state = narrowStep(state, kinds, ["The Citing Account", "a citation", "another account"], labels, "i");
  // "ci" survives in the first two (case-insensitively: the first
  // target's prose carries capitals); the third never has "ci".
  assert.deepEqual(state.alive, [0, 1]);
  assert.equal(state.query, "ci");
});

test("narrowing never resurrects a target that already fell out", () => {
  const kinds: TargetKind[] = ["text", "text"];
  const labels = ["C", "D"];
  let state = startNarrow(2);
  state = narrowStep(state, kinds, ["alpha", "cod"], labels, "a");
  assert.deepEqual(state.alive, [0]);
  // "l" keeps only targets holding "al": target 1 is already gone.
  state = narrowStep(state, kinds, ["alpha", "cod"], labels, "l");
  assert.deepEqual(state.alive, [0]);
  state = narrowStep(state, kinds, ["alpha", "cod"], labels, "p");
  assert.deepEqual(state.alive, [0]);
  // A letter of the dropped target's text cannot bring it back.
  state = narrowStep(state, kinds, ["alpha", "cod"], labels, "t");
  assert.deepEqual(state.alive, []);
  state = narrowStep(state, kinds, ["alpha", "cod"], labels, "o");
  assert.deepEqual(state.alive, []);
});

test("a label match runs over the narrowed target set only", () => {
  const kinds: TargetKind[] = ["group", "text"];
  const labels = assignLabels(kinds); // A, a
  let state = startNarrow(2);
  // "z" names no label in either case and no text holds it: the group
  // target (never a text target) falls out of alive.
  state = narrowStep(state, kinds, ["", "zen prose"], labels, "z");
  assert.deepEqual(state.alive, [1]);
  assert.equal(state.query, "z");
  // The surviving target's own label still selects it.
  const step = narrowStep(state, kinds, ["", "zen prose"], labels, "a");
  assert.equal(step.selected, 1);
  assert.equal(step.pendingLabel, "");
});

test("multi-character input is ignored (one character per step)", () => {
  const kinds: TargetKind[] = ["group"];
  const labels = assignLabels(kinds);
  const state = startNarrow(1);
  const step = narrowStep(state, kinds, [""], labels, "AB");
  assert.deepEqual(step, { alive: [0], pendingLabel: "", query: "", selected: null });
});

test("a digit names no label and narrows text (staged app behavior)", () => {
  // The prototype's hintKey ignores non-letters (inst16.html:106);
  // the app's accepted narrowing lets any character search the text.
  // Recorded as a staged deviation in README.md.
  const kinds: TargetKind[] = ["text", "text"];
  const labels = assignLabels(kinds);
  const state = startNarrow(2);
  const step = narrowStep(state, kinds, ["revision 7 text", "plain prose"], labels, "7");
  assert.deepEqual(step.alive, [0]);
  assert.equal(step.query, "7");
});
