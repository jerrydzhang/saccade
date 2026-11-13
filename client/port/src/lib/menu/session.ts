// Menu invocation, lifted from client/port-staging/menu-invocation.ts
// (t-36@r65) into the port with only the import path adjusted.
//
// One action object behind every invocation path; the menu closes
// before executing; Escape and an outside click dismiss without
// acting. Scenario clauses (t-36@r50 reference/acceptance-scenarios.txt,
// scenario 9 lines 203-219): from the same starting board, menu key
// after selection, right-click then Enter, right-click then click, and
// plain click all run the same action object; plain click means follow
// here; the menu closes before executing; Escape and outside click
// dismiss without acting; unrelated keys behave as the prototype does -
// no universal close rule is invented.
//
// Executed prototype source (inst16.html):
//   - :37-43 KINDS: each target kind declares its acts; ref's first act
//     is "Follow here" (:43) - plain click follows here.
//   - :55-59 select(): closes any open menu, marks the selection, and
//     builds the same act list for every opening path; each rendered
//     item's click runs act(i).
//   - :66 act(i): closeMenu() runs BEFORE the act - close-before-
//     execute; an invalid index closes without acting.
//   - :69 primary(el): a plain click runs the FIRST act directly,
//     without opening a menu.
//   - :504 (the keydown chain's menu branch): Escape closes without
//     acting; Enter fires act(0); a declared act key fires act(i); an
//     unrelated key is consumed and leaves the menu open - no action,
//     no close (no universal close rule).
//   - :517: a pointer-down outside the menu closes it without acting.
//   - :531: right-click (contextmenu) opens the same act list at the
//     pointer.
// The leader which-key menu (space then a key: inst16.html:84-97) is a
// distinct surface with different close behavior and is not part of
// this module; the ratified scenario 9 surface is the ActionMenu.
//
// The Action type is the ratified contract's (input-context.ts, from
// t-36@r46): one action object per operation, so all four invocation
// paths run the same object - the tests assert identity.

import type { Action } from "../input/input-context.ts";

/** The live menu view a component paints: the heading line and the
 * acts with their keys. The point a pointer-open carried rides for
 * positioning hosts; logic never reads it. */
export type MenuView = Readonly<{
  open: true;
  label: string;
  acts: readonly { key: string; label: string }[];
  at: Readonly<{ x: number; y: number }> | null;
  /** The target's part path, for beside-the-target placement when
   * no pointer point rides the open. */
  part: string | null;
}>;

export type MenuTarget = Readonly<{
  kind: string;
  /** The heading line above the acts (the prototype's menu head,
   * select() :58 `el.dataset.label || el.dataset.loc || el.dataset.kind`). */
  label: string;
  /** The target's canonical part path - identity rides the VALUE
   * OBJECT, never a side map keyed by kind:label (two targets of one
   * kind with equal first-line labels collide there; t-36@r90's
   * part-carriage constraint). The reference target of a citation
   * carries its incorporation part's path. */
  part: string | null;
}>;

export type MenuAct = Readonly<{
  /** The key that fires the act while the menu is open (:504's
   * items.findIndex by act key). */
  key: string;
  action: Action;
}>;

// The host's per-kind act declarations - the staged KINDS (:37-43).
// Returning [] models a kind with no acts (the prototype's click
// handler returns early for block targets, :526-527).
export type MenuActs = (target: MenuTarget) => readonly MenuAct[];

export type MenuEvent =
  | Readonly<{ kind: "open"; target: MenuTarget; at: Readonly<{ x: number; y: number }> | null }>
  | Readonly<{ kind: "close"; reason: "act" | "escape" | "outside-click" | "reopen" }>
  | Readonly<{ kind: "act"; index: number; actionId: string }>;

export type MenuKeyOutcome =
  | Readonly<{ kind: "act"; index: number; action: Action }>
  | Readonly<{ kind: "dismissed" }>
  | Readonly<{ kind: "ignored" }>;

export type MenuSession = Readonly<{
  isOpen(): boolean;
  target(): MenuTarget | null;
  acts(): readonly MenuAct[];
  journal(): readonly MenuEvent[];
  /** Opening from a label selection (select(), :55-59: any open menu
   * closes first, then the acts are built for the target). */
  openFromSelection(target: MenuTarget): void;
  /** Right-click opens the same act list at the pointer (:531); the
   * point is carried for the positioning host, never used for logic. */
  openAtPointer(target: MenuTarget, at: { x: number; y: number }): void;
  /** The point the open menu was anchored at (a right-click's
   * coordinates), or null when it opened from a selection. */
  at(): { x: number; y: number } | null;
  /** A key with the menu open (:504): Escape dismisses without
   * acting; Enter fires the first act; a declared act key fires it;
   * an unrelated key is consumed, acts nothing, and leaves the menu
   * open. With no menu open every key is "ignored". */
  key(k: string): MenuKeyOutcome;
  /** Clicking a rendered act item runs act(i) (:59). An out-of-range
   * index closes without acting (:66's guard). */
  choose(index: number): MenuKeyOutcome;
  /** A plain click runs the FIRST act directly, without opening a
   * menu (:69 primary) - plain click means follow here (:43). With no
   * menu-independent acts declared for the kind, nothing happens. */
  primaryClick(target: MenuTarget): MenuKeyOutcome;
  /** A pointer-down outside the menu dismisses without acting (:517). */
  outsidePointerDown(): void;
  /** The host's callback: runs on every close so selection state
   * follows (closeMenu :68; the app's onClose contract,
   * menu.ts@b8240b5 "the caller's onClose runs on every close"). */
  onClose(callback: () => void): void;
}>;

export function createMenuSession(actsFor: MenuActs): MenuSession {
  let open = false;
  let current: MenuTarget | null = null;
  let acts: readonly MenuAct[] = [];
  const events: MenuEvent[] = [];
  let closeCallback: (() => void) | null = null;

  function close(reason: "act" | "escape" | "outside-click" | "reopen"): void {
    if (!open) return;
    open = false;
    current = null;
    acts = [];
    events.push({ kind: "close", reason });
    if (closeCallback !== null) closeCallback();
  }

  let point: { x: number; y: number } | null = null;

  function openWith(target: MenuTarget, at: { x: number; y: number } | null): void {
    // select() closes any open menu before opening (:55).
    if (open) close("reopen");
    current = target;
    acts = actsFor(target);
    point = at;
    open = true;
    events.push({ kind: "open", target, at });
  }

  // act(i): close BEFORE execute (:66: closeMenu(); if (el && a) a[2](el)).
  // The invalid-index case still closes (:66's guard); nothing runs.
  function act(index: number): MenuKeyOutcome {
    const entry = acts[index];
    close("act");
    if (entry === undefined) return { kind: "dismissed" };
    events.push({ kind: "act", index, actionId: entry.action.id });
    entry.action.run();
    return { kind: "act", index, action: entry.action };
  }

  function keyOnOpen(k: string): MenuKeyOutcome {
    // The menu branch of the keydown chain, :504.
    if (k === "Escape") {
      close("escape");
      return { kind: "dismissed" };
    }
    if (k === "Enter") return act(0);
    const index = acts.findIndex((entry) => entry.key === k);
    if (index >= 0) return act(index);
    // Unrelated key: consumed, no action, the menu stays open - the
    // prototype invents no universal close rule and neither does this
    // module (:504: preventDefault only).
    return { kind: "ignored" };
  }

  return {
    isOpen: () => open,
    target: () => current,
    acts: () => acts,
    journal: () => events,
    openFromSelection: (target: MenuTarget) => openWith(target, null),
    openAtPointer: (target: MenuTarget, at: { x: number; y: number }) => openWith(target, at),
    at: () => (open ? point : null),
    key: (k: string) => (open ? keyOnOpen(k) : { kind: "ignored" }),
    choose: (index: number) => (open ? act(index) : { kind: "ignored" }),
    primaryClick: (target: MenuTarget) => {
      // The prototype's plain click is two events: the pointer-down
      // outside any open menu dismisses it (:517), then the click runs
      // primary(el) (:69) - the first act of the kind, no menu, plain
      // click means follow here (:43). The close-before-execute order
      // is therefore observable on this path too.
      if (open) close("outside-click");
      const entries = actsFor(target);
      const entry = entries[0];
      if (entry === undefined) return { kind: "ignored" };
      events.push({ kind: "act", index: 0, actionId: entry.action.id });
      entry.action.run();
      return { kind: "act", index: 0, action: entry.action };
    },
    outsidePointerDown: () => close("outside-click"),
    onClose: (callback: () => void) => {
      closeCallback = callback;
    },
  };
}
