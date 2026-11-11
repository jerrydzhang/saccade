// The board's menu wiring: the per-kind act declarations (the staged
// KINDS shape) and the board's menu host - selection opens, keys
// route through the shared matcher, close-before-execute comes from
// the session module. Pure: the controller supplies fold state.

import { commandContext, key, sequence } from "../input/input-context.ts";
import type { Action, InputContext, KeyInput, KeyResolution } from "../input/input-context.ts";
import { createMenuSession, type MenuAct, type MenuSession, type MenuTarget } from "./session.ts";

export interface BoardMenuDeps {
  /** Whether a group part path is folded now. */
  isFolded(part: string): boolean;
  /** Fold or unfold a group part path. */
  toggleFold(part: string): void;
  /** Capture input while open. Lease release returns to the CURRENT
   * base; a menu never reinstalls its own saved reading context. */
  capture(context: InputContext): { release(): void };
}

export interface BoardMenu {
  session: MenuSession;
  /** The context installed while the menu is open (null closed). */
  context(): InputContext | null;
  /** Selection (or a right-click) opens the menu when the target has
   * acts; a no-op for act-less kinds (plain blocks). */
  openFor(target: Readonly<{ part: string; kind: string; label: string }>, at?: { x: number; y: number }): void;
  /** The view-change notification (open, act, close). */
  setOnNotify(callback: () => void): void;
}

export function createBoardMenu(deps: BoardMenuDeps): BoardMenu {
  const partOf = new Map<string, string>();
  const menuTargetOf = (part: string, kind: string, label: string): MenuTarget => {
    const target: MenuTarget = { kind, label };
    // the menu module's targets are value objects; the part rides
    // alongside through this side map keyed by open order
    partOf.set(`${kind}:${label}`, part);
    return target;
  };
  const partOfTarget = (target: MenuTarget): string | null => partOf.get(`${target.kind}:${target.label}`) ?? null;

  const session = createMenuSession((target) => actsFor(target, partOfTarget(target), deps));

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

  const context = (): InputContext | null => active;

  function buildContext(): InputContext {
    const menuKey = (k: string): KeyResolution => {
      void session.key(k);
      notify?.();
      return { kind: "consume" } as const;
    };
    const ctx: InputContext = {
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
    return ctx;
  }

  function currentActs(): readonly MenuAct[] {
    const target = session.target();
    if (target === null) return [];
    const part = partOfTarget(target);
    if (part === null) return [];
    return actsFor(target, part, deps);
  }

  return {
    session,
    context,
    setOnNotify(callback) {
      notify = callback;
    },
    openFor(target, at) {
      const acts = actsFor(
        { kind: target.kind, label: target.label },
        target.part,
        deps,
      );
      if (acts.length === 0) return;
      if (active === null) {
        active = buildContext();
        lease = deps.capture(active);
      }
      if (at !== undefined) session.openAtPointer(menuTargetOf(target.part, target.kind, target.label), at);
      else session.openFromSelection(menuTargetOf(target.part, target.kind, target.label));
      notify?.();
    },
  };
}

/** The per-kind act declarations. Chunk C: groups fold/unfold; every
 * other kind carries no acts yet (their acts land with the citations
 * and figure chunks). */
export function actsFor(
  target: MenuTarget,
  part: string | null,
  deps: Pick<BoardMenuDeps, "isFolded" | "toggleFold">,
): readonly MenuAct[] {
  if (target.kind !== "group" || part === null) return [];
  const folded = deps.isFolded(part);
  return [
    {
      key: "z",
      action: {
        id: "board.fold",
        label: folded ? "unfold" : "fold",
        run: () => deps.toggleFold(part),
      },
    },
  ];
}
