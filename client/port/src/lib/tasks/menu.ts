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
  /** Copy text to the clipboard; injected so tests observe it and
   * the menu never touches the DOM's clipboard itself. */
  copy?(text: string): void;
}>;

import { createActMenu } from "../menu/create.ts";

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
  // F6 (the ratified primary, t-36 r109): the row's FIRST act is
  // o 'open board through the link' - Enter and a plain click run
  // it. With one window and no link, the elsewhere surface's
  // one-window rule opens a new linked window without asking; the
  // next Enter reuses the link. The prototype's f/in-place open
  // never existed - the port's earlier first act reversed this.
  // e 'show activity' is RECORDED ABSENT: no activity location is
  // built yet (the inventory's B3); it returns with the surface.
  const acts: MenuAct[] = [];
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
  const copy = ports.copy;
  if (copy !== undefined) {
    acts.push({
      key: "y",
      action: {
        id: "tasks.copy-reference",
        label: "copy reference",
        run: () => copy(spec),
      },
    });
  }
  return acts;
}

export function createTasksMenu(deps: TasksMenuDeps): TasksMenu {
  return createActMenu(
    (target) => taskRowActs(target, deps),
    { capture: (context) => deps.capture(context) },
  );
}
