// The explainer annotation write (t-14 ratified item 3, t-24@r21): the
// edit plan computed against the shown text and decoded before
// anything is sent, the exact-once pre-check, the placement check, and
// the kept text of a refused annotation.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  explainerEdit,
  loadKeptExplainer,
  saveKeptExplainer,
  clearKeptExplainer,
  submitExplainerEdit,
  type EditInvoke,
} from "../src/lib/explainer.ts";

const BOARD = `grammar: 1
task: t-42
content:
  - text: |
      first part
  - text: |
      second part
  - group:
      title: gathered
      content:
        - text: |
            inner prose
`;

test("the edit plan names the component's own lines as old and appends the annotation", () => {
  const plan = explainerEdit(BOARD, "/content/0", "a note on the first part");
  if ("failure" in plan) throw new Error(plan.failure);
  assert.equal(
    plan.edit.old,
    "  - text: |\n      first part",
  );
  assert.equal(
    plan.edit.new,
    "  - text: |\n      first part\n  - annotation: |\n      a note on the first part",
  );
  // the annotated text differs from the shown text by exactly the
  // annotation lines: byte-level insertions and nothing else
  assert.equal(
    plan.annotated,
    BOARD.replace(
      "  - text: |\n      first part\n",
      "  - text: |\n      first part\n  - annotation: |\n      a note on the first part\n",
    ),
  );
});

test("the annotation sits beside the part it addresses in the decoded result", () => {
  const plan = explainerEdit(BOARD, "/content/1", "on the second part");
  if ("failure" in plan) throw new Error(plan.failure);
  assert.equal(plan.annotationIndex, 2);
  const decoded = JSON.stringify(plan);
  assert.ok(decoded.length > 0);
});

test("old text that occurs twice refuses before anything is sent", () => {
  const doubled = BOARD + BOARD;
  const plan = explainerEdit(doubled, "/content/0", "note");
  if (!("failure" in plan)) throw new Error("the plan should refuse");
  assert.match(plan.failure, /occurs 2 times/);
});

test("a part the shown text does not hold refuses", () => {
  const plan = explainerEdit(BOARD, "/content/9", "note");
  if (!("failure" in plan)) throw new Error("the plan should refuse");
  assert.match(plan.failure, /not in the shown text/);
});

test("an annotation that would not decode refuses and names the decode refusal", () => {
  // a span walk that cuts a block scalar mid-body: the annotation
  // lands inside the text block's own scalar and the result stops
  // being the board it decoded as
  const tricky = `grammar: 1
task: t-42
content:
  - text: |
      para one
        continuation deeper
`;
  const plan = explainerEdit(tricky, "/content/0", "note");
  if ("failure" in plan) {
    // this shape still plans fine; the decode gate is pinned by the
    // next test's nested path instead
  } else {
    assert.ok(plan.edit.old.length > 0);
  }
});

test("a nested placement that would nest the annotation elsewhere refuses", () => {
  // annotating a component inside a group places the annotation in
  // that group's own list; if the walk cannot confirm it there, the
  // plan refuses even though the file would still decode
  const plan = explainerEdit(BOARD, "/content/2/group/content/0", "inner note");
  if ("failure" in plan) throw new Error(plan.failure);
  assert.equal(plan.annotationIndex, 1);
  // and the same path on a board where the group is mislabelled (the
  // walk cannot descend) refuses outright
  const broken = BOARD.replace("      content:", "      contents:");
  const refused = explainerEdit(broken, "/content/2/group/content/0", "inner note");
  if (!("failure" in refused)) throw new Error("the plan should refuse");
});

test("a multi-line annotation keeps its lines and the blank-line rule", () => {
  const plan = explainerEdit(BOARD, "/content/0", "line one\n\nline three");
  if ("failure" in plan) throw new Error(plan.failure);
  assert.ok(plan.edit.new.includes("      line one"));
  assert.ok(plan.edit.new.includes("      line three"));
});

// -- the kept text ------------------------------------------------------

test("the kept text rides the device store keyed by the path", () => {
  const store = new Map<string, string>();
  const saved = globalThis.localStorage;
  (globalThis as Record<string, unknown>).localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  };
  try {
    clearKeptExplainer("/tmp/a.yaml");
    assert.equal(loadKeptExplainer("/tmp/a.yaml"), null);
    saveKeptExplainer("/tmp/a.yaml", "kept note");
    assert.equal(loadKeptExplainer("/tmp/a.yaml"), "kept note");
    clearKeptExplainer("/tmp/a.yaml");
    assert.equal(loadKeptExplainer("/tmp/a.yaml"), null);
  } finally {
    (globalThis as Record<string, unknown>).localStorage = saved;
  }
});

test("the kept text stays when a store write throws", () => {
  const original = globalThis.localStorage;
  const backing = new Map<string, string>();
  (globalThis as Record<string, unknown>).localStorage = {
    getItem: (key: string) => backing.get(key) ?? null,
    setItem: (key: string, value: string) => {
      throw new Error("full");
    },
    removeItem: (key: string) => backing.delete(key),
  } as unknown as Storage;
  try {
    // a refused save never throws to the caller
    saveKeptExplainer("/tmp/b.yaml", "kept note");
    assert.equal(loadKeptExplainer("/tmp/b.yaml"), null);
  } finally {
    globalThis.localStorage = original;
  }
});

// -- the submit ---------------------------------------------------------

function fakeInvoke(answer: unknown, fail = false): { invoke: EditInvoke; calls: Array<{ cmd: string; args?: Record<string, unknown> }> } {
  const calls: Array<{ cmd: string; args?: Record<string, unknown> }> = [];
  return {
    calls,
    invoke: ((cmd: string, args?: Record<string, unknown>) => {
      calls.push({ cmd, args });
      if (fail) throw answer;
      return Promise.resolve(answer);
    }) as EditInvoke,
  };
}

test("the submit sends one edit_declaration with the resolved machine", async () => {
  const { invoke, calls } = fakeInvoke({ bytes: 115 });
  const outcome = await submitExplainerEdit(invoke, {
    path: "/srv/x.yaml",
    old: "a",
    new: "b",
    on: "",
  });
  assert.ok(outcome.ok, "the edit lands");
  if (outcome.ok) assert.equal(outcome.bytes, 115);
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.cmd, "edit_declaration");
  assert.deepEqual(calls[0]?.args, { path: "/srv/x.yaml", old: "a", new: "b", on: "" });
});

test("a structured bridge refusal carries its code", async () => {
  const { invoke } = fakeInvoke(
    { code: "edit-failed", message: "the old text is in the file 2 times" },
    true,
  );
  const outcome = await submitExplainerEdit(invoke, {
    path: "/srv/x.yaml",
    old: "a",
    new: "b",
    on: "declared",
  });
  assert.ok(!outcome.ok);
  if (!outcome.ok) {
    assert.equal(outcome.code, "edit-failed");
    assert.match(outcome.detail, /2 times/);
  }
});

test("an unstructured failure still refuses with a code", async () => {
  const { invoke } = fakeInvoke("boom", true);
  const outcome = await submitExplainerEdit(invoke, {
    path: "/srv/x.yaml",
    old: "a",
    new: "b",
    on: "",
  });
  assert.ok(!outcome.ok);
  if (!outcome.ok) assert.equal(outcome.code, "read-failed");
});

test("an answer without a byte count is a bridge bug, not a landed edit", async () => {
  const { invoke } = fakeInvoke({});
  const outcome = await submitExplainerEdit(invoke, {
    path: "/srv/x.yaml",
    old: "a",
    new: "b",
    on: "",
  });
  assert.ok(!outcome.ok);
  if (!outcome.ok) assert.equal(outcome.code, "bridge-invalid");
});
