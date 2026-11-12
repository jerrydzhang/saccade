// The Tasks instrument's menu wiring: the task-row act declarations
// and the menu host. The shape mirrors the board's menu host
// (lib/menu/acts.ts) - selection or a right-click opens, the captured
// context routes keys, close-before-execute is the session module's
// own rule - but the acts are the task row's (scenario 17 with the
// prototype's KINDS as the contract, inst16.html :40):
//
//   f  open the task's activity here   (the review-resolved primary:
//      the activity is a LOCATION that replaces the window's content;
//      plain click and Enter run this first act)
//   o  open board (through the link, or ask once)
//   i  open in... (ask; keep the link)
//   w  open in a new window
//
// The o/i/w acts are the window's elsewhere surface (S18's OPEN acts,
// window-wide since ea51669): they appear only when the window has
// injected the ports and bound its window-store state. The primary
// needs only the open funnel. No other act is offered - not the
// prototype's 'l' (launch) or 'e' (inline activity), which the
// agreement superseded.

import { commandContext, key, sequence } from "../input/input-context.ts";
import type { Action, InputContext, KeyInput, KeyResolution } from "../input/input-context.ts";
import { formatLocation } from "../location.ts";
import { createMenuSession, type MenuAct, type MenuSession, type MenuTarget } from "../menu/session.ts";
import type { ElsewhereAct } from "../elsewhere.ts";

/** The window services the task-row acts ride: opening goes through
 * the window's ONE open funnel (trail + restore), the elsewhere acts
 * through the window's OPEN surface. The window injects them; until
 * then no acts are offered. */
export type TaskPorts = Readonly<{
  open(spec: string): void;
  elsewhere?(spec: string, act: ElsewhereAct): void;
}>;

export interface TasksMenuDeps {
  /** The tracker machine the shown list resolved at: the acts compose
   * their specs against it (formatLocation, never a colon join). */
  machine(): string | null;
  /** The row's link - the location its acts open (the task's
   * activity) - for a row part path, or null when the part names no
   * row of the shown page. */
  rowLink(part: string): string | null;
  /** The window's services, or null before injection. */
  ports(): TaskPorts | null;
  /** Capture input while open. Lease release returns to the CURRENT
   * base; a menu never reinstalls its own saved reading context. */
  capture(context: InputContext): { release(): void };
}

export interface TasksMenu {
  session: MenuSession;
  /** The context installed while the menu is open (null closed). */
  context(): InputContext | null;
  /** Selection (or a right-click) opens the menu when the target has
   * acts; a no-op when nothing is offered. */
  openFor(target: Readonly<{ part: string | null; kind: string; label: string }>, at?: { x: number; y: number }): void;
  /** The view-change notification (open, act, close). */
  setOnNotify(callback: () => void): void;
}

export function taskRowActs(
  target: MenuTarget,
  deps: Pick<TasksMenuDeps, "machine" | "rowLink" | "ports">,
): readonly MenuAct[] {
  const ports = deps.ports();
  if (ports === null) return [];
  const machine = deps.machine();
  if (machine === null) return [];
  // the ROW'S LINK composed against the list's own tracker - never
  // the part path (that is identity, not a location)
  const link = target.part === null ? null : deps.rowLink(target.part);
  if (link === null || link.length === 0) return [];
  const spec = formatLocation({ machine, path: link });
  const acts: MenuAct[] = [
    {
      key: "f",
      action: {
        id: "tasks.open-activity",
        label: "open the task's activity here",
        run: () => ports.open(spec),
      },
    },
  ];
  if (ports.elsewhere !== undefined) {
    const elsewhere = ports.elsewhere;
    const open = (act: ElsewhereAct): Action => ({
      id: `tasks.elsewhere.${act}`,
      label:
        act === "link"
          ? "open board (through the link, or ask once)"
          : act === "once"
            ? "open in... (ask; keep the link)"
            : "open in a new window",
      run: () => elsewhere(spec, act),
    });
    acts.push(
      { key: "o", action: open("link") },
      { key: "i", action: open("once") },
      { key: "w", action: open("new") },
    );
  }
  return acts;
}

export function createTasksMenu(deps: TasksMenuDeps): TasksMenu {
  const session = createMenuSession((target) => taskRowActs(target, deps));

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

  function currentActs(): readonly MenuAct[] {
    const target = session.target();
    if (target === null) return [];
    return taskRowActs(target, deps);
  }

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
        ...currentActs().map((act) => ({
          action: {
            id: `menu.act.${act.key}`,
            label: act.action.label,
            run: () => void menuKey(act.key),
          },
          keys: [sequence(key(act.key))],
        })),
      ]),
      onUnmatched: (stroke: KeyInput): KeyResolution | undefined => {
        if (active === null) return undefined;
        if (stroke.key.length !== 1) return { kind: "consume" };
        return menuKey(stroke.key);
      },
    };
    return ctx;
  }

  return {
    session,
    context: () => active,
    setOnNotify(callback) {
      notify = callback;
    },
    openFor(target, at) {
      const menuTarget: MenuTarget = { kind: target.kind, label: target.label, part: target.part };
      const acts = taskRowActs(menuTarget, deps);
      if (acts.length === 0) return;
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
