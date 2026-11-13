// Tests for menu invocation (menu-invocation.ts): scenario 9 clauses -
// one action object behind every invocation path, close-before-
// execute, Escape/outside-click dismissal without acting, unrelated
// keys left alone (no universal close rule).
import test from "node:test";
import assert from "node:assert/strict";

import { createMenuSession, type MenuAct, type MenuSession, type MenuTarget } from "./menu-invocation.ts";
import type { Action } from "./input-context.ts";

// The prototype's ref kind (inst16.html:43): Follow here first, then
// the open-elsewhere acts, then copy. ONE Action object per operation,
// shared by every opening (the contract's Action identity).
const follow: Action = { id: "follow-here", label: "Follow here", run: () => {} };
const openLink: Action = { id: "open-link", label: "Open board (through the link)", run: () => {} };
const copyRef: Action = { id: "copy-reference", label: "Copy reference", run: () => {} };
const refActs: readonly MenuAct[] = [
  { key: "f", action: follow },
  { key: "o", action: openLink },
  { key: "y", action: copyRef },
];
const refTarget: MenuTarget = { kind: "ref", label: "t-3@r14" };

function refSession(): MenuSession {
  return createMenuSession((target) => (target.kind === "ref" ? refActs : []));
}

// Acts whose runs record themselves, for ordering and act-id checks.
function actsWith(ran: string[]): readonly MenuAct[] {
  const mk = (id: string): Action => ({
    id,
    label: id,
    run: () => {
      ran.push(id);
    },
  });
  return [
    { key: "f", action: mk("follow-here") },
    { key: "o", action: mk("open-link") },
    { key: "y", action: mk("copy-reference") },
  ];
}

function sessionWith(ran: string[]): MenuSession {
  const acts = actsWith(ran);
  return createMenuSession((target) => (target.kind === "ref" ? acts : []));
}

test("menu key after selection runs the same operation as plain click", () => {
  // Scenario 9, paths 1 and 4: select the reference (label-select
  // opens its menu, select :55-59), then the act's key fires (keydown
  // :504); a plain click runs the FIRST act with no menu
  // (:69 primary) - plain click means follow here (:43).
  const ran: string[] = [];
  const session = sessionWith(ran);
  session.openFromSelection(refTarget);
  const outcome = session.key("f");
  assert.equal(outcome.kind, "act");
  assert.equal(outcome.kind === "act" && outcome.action.id, "follow-here");
  assert.deepEqual(ran, ["follow-here"]);
  const plain = session.primaryClick(refTarget);
  assert.equal(plain.kind, "act");
  assert.equal(plain.kind === "act" && plain.action.id, "follow-here");
  assert.equal(session.isOpen(), false);
  assert.deepEqual(ran, ["follow-here", "follow-here"]);
});

test("right-click then Enter and right-click then click run the first act", () => {
  // Paths 2 and 3: contextmenu opens at the pointer (:531); Enter
  // fires the first act (:504); clicking a rendered item fires
  // act(i) (:59).
  const ran: string[] = [];
  const session = sessionWith(ran);
  session.openAtPointer(refTarget, { x: 40, y: 60 });
  const enter = session.key("Enter");
  assert.equal(enter.kind === "act" && enter.action.id, "follow-here");
  assert.deepEqual(ran, ["follow-here"]);
  session.openAtPointer(refTarget, { x: 40, y: 60 });
  const item = session.choose(0);
  assert.equal(item.kind === "act" && item.action.id, "follow-here");
  assert.deepEqual(ran, ["follow-here", "follow-here"]);
});

test("identity: every invocation path resolves to the same Action object", () => {
  // Scenario 9: all four invocations run the same action object.
  const session = refSession();
  session.openFromSelection(refTarget);
  const menuKey = session.key("f");
  session.openAtPointer(refTarget, { x: 1, y: 2 });
  const enter = session.key("Enter");
  session.openAtPointer(refTarget, { x: 1, y: 2 });
  const item = session.choose(0);
  const plain = session.primaryClick(refTarget);
  const actions = [menuKey, enter, item, plain].map((outcome) =>
    outcome.kind === "act" ? outcome.action : null,
  );
  assert.ok(actions.every((a) => a !== null && a.id === "follow-here"));
  // Reference identity with the one declared Action.
  assert.equal(actions[0], follow);
  assert.equal(actions[1], follow);
  assert.equal(actions[2], follow);
  assert.equal(actions[3], follow);
});

test("the menu closes before executing (journal order: close, then act)", () => {
  // act(i), inst16.html:66: closeMenu(); if (el && a) a[2](el).
  const ran: string[] = [];
  const session = sessionWith(ran);
  session.openFromSelection(refTarget);
  session.key("f");
  const journal = session.journal();
  const closeAt = journal.findIndex((event) => event.kind === "close" && event.reason === "act");
  const actAt = journal.findIndex((event) => event.kind === "act");
  assert.ok(closeAt >= 0);
  assert.ok(actAt > closeAt);
  assert.deepEqual(ran, ["follow-here"]);
});

test("the close callback runs before the action's run", () => {
  // closeMenu (:68) clears selection state first; the app's onClose
  // contract runs on every close (menu.ts@b8240b5).
  const order: string[] = [];
  const tracked: Action = {
    id: "follow-here",
    label: "Follow here",
    run: () => {
      order.push("run");
    },
  };
  const session = createMenuSession((target) => (target.kind === "ref" ? [{ key: "f", action: tracked }] : []));
  session.onClose(() => order.push("close"));
  session.openFromSelection(refTarget);
  session.key("f");
  assert.deepEqual(order, ["close", "run"]);
});

test("Escape dismisses without acting", () => {
  const ran: string[] = [];
  const session = sessionWith(ran);
  session.openFromSelection(refTarget);
  const outcome = session.key("Escape");
  assert.equal(outcome.kind, "dismissed");
  assert.equal(session.isOpen(), false);
  assert.deepEqual(ran, []);
});

test("an outside pointer-down dismisses without acting", () => {
  const ran: string[] = [];
  const session = sessionWith(ran);
  session.openAtPointer(refTarget, { x: 5, y: 5 });
  session.outsidePointerDown();
  assert.equal(session.isOpen(), false);
  assert.deepEqual(ran, []);
  // Dismissal is a close, not an act.
  const last = session.journal().at(-1);
  assert.ok(last !== undefined && last.kind === "close" && last.reason === "outside-click");
});

test("an unrelated key is consumed, acts nothing, and the menu stays open", () => {
  // inst16.html:504: preventDefault only - no universal close rule.
  const ran: string[] = [];
  const session = sessionWith(ran);
  session.openFromSelection(refTarget);
  const outcome = session.key("q");
  assert.deepEqual(outcome, { kind: "ignored" });
  assert.equal(session.isOpen(), true);
  assert.deepEqual(ran, []);
  // A declared act key still fires afterwards.
  const later = session.key("y");
  assert.equal(later.kind === "act" && later.action.id, "copy-reference");
  assert.deepEqual(ran, ["copy-reference"]);
});

test("choosing an out-of-range index closes without acting", () => {
  // act()'s guard (:66): closeMenu(); if (el && a) a[2](el).
  const ran: string[] = [];
  const session = sessionWith(ran);
  session.openFromSelection(refTarget);
  const outcome = session.choose(9);
  assert.equal(outcome.kind, "dismissed");
  assert.equal(session.isOpen(), false);
  assert.deepEqual(ran, []);
});

test("a plain click on a kind with no acts does nothing", () => {
  // The prototype's click handler returns early for block targets
  // (:526-527); a kind with no declared acts has no primary act.
  const session = createMenuSession(() => []);
  const outcome = session.primaryClick({ kind: "block", label: "part /content/0" });
  assert.deepEqual(outcome, { kind: "ignored" });
  assert.equal(session.isOpen(), false);
});

test("opening while a menu is open closes the first (select, :55)", () => {
  const closes: string[] = [];
  const session = refSession();
  session.onClose(() => closes.push("closed"));
  session.openFromSelection(refTarget);
  session.openFromSelection({ kind: "ref", label: "t-5@r4" });
  assert.equal(session.isOpen(), true);
  assert.equal(session.target()?.label, "t-5@r4");
  assert.deepEqual(closes, ["closed"]);
  const journal = session.journal();
  const reopen = journal.findIndex((event) => event.kind === "close" && event.reason === "reopen");
  const secondOpen = journal.findIndex((event) => event.kind === "open" && event.target.label === "t-5@r4");
  assert.ok(reopen >= 0 && secondOpen > reopen);
});

test("keys with no menu open are ignored", () => {
  const session = refSession();
  assert.deepEqual(session.key("f"), { kind: "ignored" });
  assert.deepEqual(session.key("Escape"), { kind: "ignored" });
  assert.deepEqual(session.choose(0), { kind: "ignored" });
});
