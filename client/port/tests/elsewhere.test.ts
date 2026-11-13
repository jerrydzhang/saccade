// The elsewhere decision core (S18): which move each OPEN act makes,
// the ask's chips, and their honest edges. Pure: no store, no shell.

import assert from "node:assert/strict";
import { test } from "vitest";

import { assignChips, elsewhereDecision, type SuiteWindow } from "../src/lib/elsewhere.ts";

const two: SuiteWindow[] = [
  { label: "main", state_id: "s-1" },
  { label: "w-2", state_id: "s-2" },
];

test("o delivers at a remembered link without asking", () => {
  assert.deepEqual(elsewhereDecision("link", { linkLabel: "w-2", windows: two }), {
    kind: "deliver",
    label: "w-2",
  });
});

test("o with no link asks when other windows exist; with one window it opens new AND becomes the link", () => {
  assert.deepEqual(elsewhereDecision("link", { linkLabel: null, windows: two }), { kind: "ask" });
  assert.deepEqual(elsewhereDecision("link", { linkLabel: null, windows: two.slice(0, 1) }), {
    kind: "new-window",
    becomeLink: true,
  });
});

test("i never uses the link: it asks when others exist, opens new (no link) with one window", () => {
  assert.deepEqual(elsewhereDecision("once", { linkLabel: "w-2", windows: two }), { kind: "ask" });
  assert.deepEqual(elsewhereDecision("once", { linkLabel: "w-2", windows: two.slice(0, 1) }), {
    kind: "new-window",
    becomeLink: false,
  });
});

test("w always opens a new viewer, link or not", () => {
  assert.deepEqual(elsewhereDecision("new", { linkLabel: "w-2", windows: two }), {
    kind: "new-window",
    becomeLink: false,
  });
  assert.deepEqual(elsewhereDecision("new", { linkLabel: null, windows: two.slice(0, 1) }), {
    kind: "new-window",
    becomeLink: false,
  });
});

test("a dead link (no live window carries it) does not deliver: o asks instead", () => {
  // the link names a state no window shows - refreshLink resolves
  // null, and the decision must never deliver into a dead label
  assert.deepEqual(elsewhereDecision("link", { linkLabel: null, windows: two }), { kind: "ask" });
});

test("chips assign a, b, c in stable order and grow past one alphabet", () => {
  assert.deepEqual(
    assignChips(["main", "w-2"]).map((entry) => [entry.window, entry.chip]),
    [
      ["main", "a"],
      ["w-2", "b"],
    ],
  );
  const many = assignChips(Array.from({ length: 28 }, (_, n) => `w-${n}`));
  assert.equal(many[25]!.chip, "z");
  assert.equal(many[26]!.chip, "aa");
  assert.equal(many[27]!.chip, "ab");
});
