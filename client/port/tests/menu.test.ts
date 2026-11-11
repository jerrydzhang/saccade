// The board menu wiring (chunk C): one action object behind every
// invocation path, close-before-execute, fold as the first real act,
// and the canonical part-path rendering the geometry depends on.
// Pure tests - no DOM, no sources.

import assert from "node:assert/strict";
import test from "node:test";
import { createBoardMenu, actsFor } from "../src/lib/menu/acts.ts";
import { createMenuSession } from "../src/lib/menu/session.ts";
import type { Action } from "../src/lib/contracts/input.ts";
import type { InputContext } from "../src/lib/contracts/input.ts";

test("groups carry fold/unfold; other kinds carry nothing yet", () => {
  const folded = new Set<string>(["/content/1"]);
  const acts = actsFor({ kind: "group", label: "the group" }, "/content/1", {
    isFolded: (part) => folded.has(part),
    toggleFold: (part) => { folded.delete(part); },
  });
  assert.equal(acts.length, 1);
  assert.equal(acts[0]?.key, "z");
  assert.equal(acts[0]?.action.label, "unfold");
  assert.deepEqual(actsFor({ kind: "text", label: "prose" }, "/content/0", {
    isFolded: () => false,
    toggleFold: () => {},
  }), []);
  assert.deepEqual(actsFor({ kind: "group", label: "g" }, null, {
    isFolded: () => false,
    toggleFold: () => {},
  }), []);
});

test("the menu session: four invocation paths run the SAME action object; close before execute", () => {
  const journal: string[] = [];
  const action: Action = { id: "board.fold", label: "fold", run: () => { journal.push("run"); } };
  const session = createMenuSession(() => [{ key: "z", action }]);
  const target = { kind: "group", label: "g" };
  // path 1: selection opens; Enter fires the first act
  session.openFromSelection(target);
  assert.deepEqual(session.key("Enter"), { kind: "act", index: 0, action });
  // path 2: right-click opens; Enter fires
  session.openAtPointer(target, { x: 1, y: 1 });
  assert.deepEqual(session.key("Enter"), { kind: "act", index: 0, action });
  // path 3: right-click opens; the act item is clicked (choose)
  session.openAtPointer(target, { x: 1, y: 1 });
  assert.deepEqual(session.choose(0), { kind: "act", index: 0, action });
  // path 4: the declared act key fires while open
  session.openFromSelection(target);
  assert.deepEqual(session.key("z"), { kind: "act", index: 0, action });
  assert.equal(journal.length, 4);
  // close-before-execute ordering: the close event precedes the act event
  const closeIndex = session.journal().findIndex((e) => e.kind === "close" && e.reason === "act");
  const actIndex = session.journal().findIndex((e) => e.kind === "act");
  assert.ok(closeIndex >= 0 && actIndex >= 0 && closeIndex < actIndex);
  // escape dismisses without acting; an unrelated key is consumed and
  // leaves the menu open (no universal close rule)
  session.openFromSelection(target);
  assert.deepEqual(session.key("Escape"), { kind: "dismissed" });
  session.openFromSelection(target);
  assert.deepEqual(session.key("q"), { kind: "ignored" });
  assert.equal(session.isOpen(), true);
  session.outsidePointerDown();
  assert.equal(session.isOpen(), false);
  assert.equal(journal.length, 4, "dismissals never act");
});

function menuHarness(folded: Set<string>) {
  const events: string[] = [];
  let readingNow: InputContext | null = null;
  const menu = createBoardMenu({
    isFolded: (part) => folded.has(part),
    toggleFold: (part) => {
      if (folded.has(part)) folded.delete(part);
      else folded.add(part);
      events.push(`fold:${part}`);
    },
    capture: (context) => {
      events.push("capture");
      readingNow = context;
      return { release: () => {
        events.push("release", "restore");
        readingNow = null; // the host lease owns the return
      } };
    },
  });
  return { menu, events, context: () => readingNow };
}

test("the board menu host: selection opens for groups, capture/restore ride the lease, z folds", () => {
  const folded = new Set<string>();
  const { menu, events, context } = menuHarness(folded);
  // an act-less kind opens nothing
  menu.openFor({ part: "/content/0", kind: "text", label: "prose" });
  assert.deepEqual(events, []);
  assert.equal(context(), null);
  // a group opens, captures, and the declared act key folds it
  menu.openFor({ part: "/content/1", kind: "group", label: "the group" });
  assert.ok(menu.session.isOpen());
  assert.notEqual(context(), null, "the menu context installed");
  const outcome = menu.session.key("z");
  assert.equal(outcome.kind, "act");
  assert.deepEqual(events, ["capture", "release", "restore", "fold:/content/1"]);
  assert.deepEqual(folded, new Set(["/content/1"]));
  assert.equal(menu.session.isOpen(), false, "acting closes the menu");
});
