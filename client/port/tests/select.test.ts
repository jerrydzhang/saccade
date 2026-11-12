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
  // two non-text targets: labels 'A' then 'B' (the other namespace)
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
      ["/content/1", "B"],
    ],
  );
  const active = context();
  assert.ok(active !== null && active.onUnmatched !== undefined, "the session context installed");
  // typing 'b' names no label in its own case and folds to 'B' - the
  // case-fold rule selects the second target (no label ever needs Shift)
  const selection = active.onUnmatched?.(key("b"));
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
  const { session, context } = harness(targets, (part) => visible.has(part));
  session.enter();
  const view = session.view();
  assert.deepEqual(view?.labels.map((l) => l.label), ["a"], "only the visible target is labeled");
  // 'z' names no label and narrows no text: consumed, session stays,
  // the invisible group is never labeled and never selects
  const active = context();
  const step = active?.onUnmatched?.(key("z"));
  assert.equal(step?.kind, "consume");
  assert.equal(session.view()?.labels.length, 1);
  assert.equal(session.view()?.labels[0]?.part, "/content/0");
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

test("labels: two namespaces, stable order, single letters first", () => {
  const labels = assignLabels(["text", "group", "text", "figure"]);
  assert.deepEqual(labels, ["a", "A", "b", "B"]);
});
