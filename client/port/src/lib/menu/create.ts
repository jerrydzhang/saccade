// The shared act-menu factory (order of work step 2, audit defect
// c): the menu's input wiring - lease, context, Enter/Escape, the
// per-act keys, the single-character dispatch, open-at-pointer vs
// open-from-selection - written once. An instrument supplies only
// its acts function; the menu opens (and captures input) when the
// target has acts, and is a no-op for act-less kinds.

import type { Action, InputContext, KeyInput, KeyResolution } from "../contracts/input.ts";
import { commandContext, key, sequence } from "../input/input-context.ts";
import { createMenuSession, type MenuSession, type MenuTarget } from "./session.ts";
import type { MenuAct } from "./session.ts";

export interface ActMenuCapture {
  /** Lease the input context for the menu's lifetime (the window's
   * capture funnel; released on close). */
  capture(context: InputContext): { release(): void };
}

export interface ActMenu {
  readonly session: MenuSession;
  /** The active menu input context, null while closed. */
  context(): InputContext | null;
  /** Selection (or a right-click) opens the menu when the target
   * has acts; a no-op for act-less kinds. */
  openFor(
    target: Readonly<{ part: string | null; kind: string; label: string }>,
    at?: { x: number; y: number },
  ): void;
  /** The view-change notification (open, act, close). */
  setOnNotify(callback: () => void): void;
}

export function createActMenu(
  actsOf: (target: MenuTarget) => readonly MenuAct[],
  deps: ActMenuCapture,
): ActMenu {
  const session = createMenuSession(actsOf);

  let lease: { release(): void } | null = null;
  let active: InputContext | null = null;
  let notify: (() => void) | null = null;

  function close(): void {
    if (lease !== null) {
      lease.release();
      lease = null;
    }
    active = null;
  }

  session.onClose(() => {
    close();
    notify?.();
  });

  function menuKey(k: string): KeyResolution {
    void session.key(k);
    notify?.();
    return { kind: "consume" } as const;
  }

  function currentActs(): readonly MenuAct[] {
    const target = session.target();
    if (target === null) return [];
    return actsOf(target);
  }

  function buildContext(): InputContext {
    return {
      ...commandContext(() => [
        {
          action: {
            id: "menu.primary",
            label: "run the first act",
            run: () => void menuKey("Enter"),
          },
          keys: [sequence(key("Enter"))],
        },
        {
          action: {
            id: "menu.dismiss",
            label: "dismiss",
            run: () => void menuKey("Escape"),
          },
          keys: [sequence(key("Escape"))],
        },
        ...currentActs().map(
          (act, index): { action: Action; keys: readonly ReturnType<typeof sequence>[] } => ({
            action: {
              id: `menu.act.${index}`,
              label: act.action.label,
              run: () => void menuKey(act.key),
            },
            keys: [sequence(key(act.key))],
          }),
        ),
      ]),
      onUnmatched: (stroke: KeyInput): KeyResolution | undefined => {
        if (active === null) return undefined;
        if (stroke.key.length !== 1) return { kind: "consume" };
        return menuKey(stroke.key);
      },
    };
  }

  return {
    session,
    context: () => active,
    setOnNotify(callback) {
      notify = callback;
    },
    openFor(target, at) {
      const menuTarget: MenuTarget = { kind: target.kind, label: target.label, part: target.part };
      if (actsOf(menuTarget).length === 0) return;
      if (active === null) {
        active = buildContext();
        lease = deps.capture(active);
      }
      if (at !== undefined) session.openAtPointer(menuTarget, at);
      else session.openFromSelection(menuTarget);
      notify?.();
    },
  };
}
