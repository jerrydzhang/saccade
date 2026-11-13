// The select session (chunk B of the Board instrument), tested
// directly: labels for visible targets only, narrowing through the
// shared matcher's onUnmatched hook, exact-single selection,
// identity-carrying targets, lease capture/restore lifecycle, and the
// model-derived target walk. Pure - no DOM, no sources.

import assert from "node:assert/strict";
import { test } from "vitest";
import { createSelectSession } from "../src/lib/select/session.ts";
import type { SelectableTarget } from "../src/lib/select/session.ts";
import { assignLabels } from "../src/lib/select/label-select.ts";
import { boardTargetsOf } from "../src/lib/board/targets.ts";
import type { Board } from "../src/lib/board/model.ts";
import type { InputContext, KeyInput } from "../src/lib/contracts/input.ts";
import { commandContext } from "../src/lib/contracts/input.ts";

const key = (k: string): KeyInput => ({ key: k, ctrl: false, shift: false, alt: false, meta: false });

function harness(targets: readonly SelectableTarget[], visible: (part: string) => boolean) {
  const events: string[] = [];
  const base = commandContext(() => []);
  let readingNow: InputContext | null = base;
  const deps = {
    targets: () => targets,
    visible: (target: SelectableTarget) => visible(target.part),
    capture: (context: InputContext) => {
      events.push("capture");
      readingNow = context;
      return { release: () => {
        events.push("release", "restore");
        readingNow = base; // the host lease owns return, not the session
      } };
    },
    onSelected: (target: SelectableTarget) => events.push(`selected:${target.part}`),
  };
  const session = createSelectSession(deps);
  return { session, events, context: () => readingNow };
}

test("targets come from the model, nested included, identity is the part path", () => {
  const board = {
    content: [
      { kind: "text", markdown: "alpha prose", pos: { start: { line: 1 }, end: { line: 2 } } },
      {
        kind: "group",
        title: "the group",
        content: [
          { kind: "text", markdown: "nested prose", pos: { start: { line: 3 }, end: { line: 3 } } },
          { kind: "figure", input: "figures/a.png", caption: "a figure", pos: { start: { line: 4 }, end: { line: 4 } } },
        ],
        pos: { start: { line: 2 }, end: { line: 4 } },
      },
    ],
  } as unknown as Board;
  assert.deepEqual(
    boardTargetsOf(board).map((t) => [t.part, t.kind]),
    [
      ["/content/0", "text"],
      ["/content/1", "group"],
      ["/content/1/content/0", "text"],
      ["/content/1/content/1", "figure"],
    ],
  );
});

test("enter captures and installs the session; exact-single selects (case-folded) and restores", () => {
  // two targets: labels 'A' then 'S' (one namespace, the prototype's
  // home-row order, inst16.html:100)
  const targets: SelectableTarget[] = [
    { part: "/content/0", kind: "group", text: "the group" },
    { part: "/content/1", kind: "figure", text: "a figure" },
  ];
  const { session, events, context } = harness(targets, () => true);
  assert.equal(session.view(), null);
  session.enter();
  assert.notEqual(session.view(), null, "labels painted on enter");
  const view = session.view();
  assert.deepEqual(
    view?.labels.map((l) => [l.part, l.label]),
    [
      ["/content/0", "A"],
      ["/content/1", "S"],
    ],
  );
  const active = context();
  assert.ok(active !== null && active.onUnmatched !== undefined, "the session context installed");
  // typing is case-insensitive: 's' selects the label painted 'S'
  // (no label ever needs Shift)
  const selection = active.onUnmatched?.(key("s"));
  assert.equal(selection?.kind, "consume");
  assert.deepEqual(events, ["capture", "release", "restore", "selected:/content/1"]);
  assert.equal(session.view(), null, "selection exits the session");
});

test("invisible targets get no labels: existence is never visibility", () => {
  const targets: SelectableTarget[] = [
    { part: "/content/0", kind: "text", text: "alpha prose" },
    { part: "/content/1", kind: "group", text: "the group" },
  ];
  const visible = new Set(["/content/0"]);
  const { session, context, events } = harness(targets, (part) => visible.has(part));
  session.enter();
  const view = session.view();
  assert.deepEqual(view?.labels.map((l) => l.label), ["A"], "only the visible target is labeled");
  // the invisible group is never labeled; 'z' names no label and
  // narrows no text: ZERO matches left, so the stroke is consumed and
  // the session EXITS (the prototype's hintKey, inst16.html:106; the
  // human's trap, m-536) - never staying alive with nothing to select,
  // and never selecting the invisible group
  const active = context();
  const step = active?.onUnmatched?.(key("z"));
  assert.equal(step?.kind, "consume");
  assert.equal(session.view(), null);
  assert.ok(!events.some((event) => event.startsWith("selected:")), "nothing selected");
});

test("enter is a no-op with nothing visible; escape exits without selecting", () => {
  const targets: SelectableTarget[] = [{ part: "/content/0", kind: "text", text: "alpha" }];
  const { session, events } = harness(targets, () => false);
  session.enter();
  assert.equal(session.view(), null);
  assert.deepEqual(events, [], "nothing captured");
  // with visibility, escape through the declared binding exits clean
  const open = harness(targets, () => true);
  open.session.enter();
  const exitBinding = open.context()?.commands().find((b) => b.action.id === "select.exit");
  assert.ok(exitBinding !== undefined);
  void exitBinding.action.run();
  assert.equal(open.session.view(), null);
  assert.deepEqual(open.events, ["capture", "release", "restore"]);
});

test("labels: ONE namespace in the prototype's home-row order, painted uppercase (human ruling m-545)", () => {
  // inst16.html:99-100: LETTERS = ASDFGHJKLQWERTYUIOPZXCVBNM, single
  // letters up to 26 targets whatever their kind, two-letter pairs past
  assert.deepEqual(assignLabels(["text", "group", "text", "figure"]), ["A", "S", "D", "F"]);
  const many = assignLabels(Array.from({ length: 27 }, () => "text" as const));
  assert.deepEqual(many.slice(0, 4), ["AA", "AS", "AD", "AF"]);
  assert.ok(many.every((label) => label === label.toUpperCase()));
});

test("labels are prefix-free past 26 targets: every label is reachable by typing it", async () => {
  // the prototype's labelsFor (inst16.html:100): up to 26 targets take
  // single letters, more take two letters ALL of them - a single
  // letter that prefixed a two-letter label could never complete
  const { narrowStep, startNarrow } = await import("../src/lib/select/label-select.ts");
  for (const count of [26, 27, 30]) {
    const kinds = Array.from({ length: count }, () => "task" as const);
    const labels = assignLabels(kinds);
    for (const label of labels) {
      assert.ok(!labels.some((other) => other !== label && other.startsWith(label)), `${label} prefixes another label (${count} targets)`);
    }
    labels.forEach((label, index) => {
      let state = startNarrow(count);
      let selected: number | null = null;
      for (const ch of label) {
        const step = narrowStep(state, labels, ch);
        state = step;
        selected = step.selected;
      }
      assert.equal(selected, index, `${label} selects its own target (${count} targets)`);
    });
  }
});

test("m-541: letters match labels only - a letter only text would match ends the session", () => {
  // the prototype's hintKey (inst16.html:106): no text narrowing; the
  // staging's narrowing turned trace 1's 'co' into an invisible text
  // search that held every key
  const targets: SelectableTarget[] = [
    { part: "/content/0", kind: "text", text: "alpha prose" },
    { part: "/content/1", kind: "text", text: "beta prose" },
  ];
  const { session, context, events } = harness(targets, () => true);
  session.enter();
  assert.deepEqual(session.view()?.labels.map((l) => l.label), ["A", "S"]);
  // 'p' names no label (it is in both texts): zero matches, exit
  const step = context()?.onUnmatched?.(key("p"));
  assert.equal(step?.kind, "consume");
  assert.equal(session.view(), null);
  assert.ok(!events.some((event) => event.startsWith("selected:")));
});

test("m-541: the typed prefix filters the painted labels; the next letter selects", () => {
  const targets: SelectableTarget[] = Array.from({ length: 30 }, (_, index) => ({
    part: `/content/${index}`,
    kind: "text" as const,
    text: `paragraph ${index}`,
  }));
  const { session, context, events } = harness(targets, () => true);
  session.enter();
  const all = session.view()!.labels;
  assert.equal(all.length, 30);
  // case-insensitive: a lowercase 's' extends the prefix 'S'
  context()?.onUnmatched?.(key("s"));
  const view = session.view()!;
  assert.equal(view.pendingLabel, "S");
  assert.ok(view.labels.length > 0 && view.labels.every((l) => l.label.startsWith("S")), "only S- labels stay painted");
  assert.ok(view.labels.length < all.length);
  const target = view.labels[0]!;
  context()?.onUnmatched?.(key(target.label[1]!.toLowerCase()));
  assert.ok(events.includes(`selected:${target.part}`));
  assert.equal(session.view(), null);
});

test("m-545: every mermaid fence in a block is a diagram target of its own, in document order", () => {
  const board = {
    content: [
      { kind: "text", markdown: "before\n\n```mermaid\nflowchart LR\n  a --> b\n```\n\nbetween\n\n```mermaid\nflowchart LR\n  c --> d\n```\n", pos: { start: { line: 1 }, end: { line: 9 } } },
      { kind: "text", markdown: "```js\nnot a diagram\n```\n", pos: { start: { line: 10 }, end: { line: 12 } } },
    ],
  } as unknown as Board;
  assert.deepEqual(
    boardTargetsOf(board).map((t) => [t.part, t.kind]),
    [
      ["/content/0", "text"],
      ["/content/0/diagram/0", "diagram"],
      ["/content/0/diagram/1", "diagram"],
      ["/content/1", "text"],
    ],
  );
});
