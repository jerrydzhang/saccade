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
// The control plane's rail entries (t-48) are targets of this same
// menu: a project's f shows its tasks here (its first act - Enter and a
// plain click switch) and d forgets it, any project; an agent's acts are its first
// task's, a task row's.
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
import type { RailTarget } from "./rail.ts";

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
  /** The rail entry a rail part names (t-48), or undefined. */
  railTarget(part: string): RailTarget | undefined;
  /** Forget a reached project (lib/tracker.ts forgetTracker). */
  forget(project: string): void;
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

/** The acts on a task's board, composed spec in hand: a task row's, and
 * an agent's for its first task (t-14: tasks and agents have no here). */
function boardActs(spec: string, ports: TaskPorts): MenuAct[] {
  // F6 (the ratified primary, t-36 r109): the FIRST act is o 'open
  // board through the link' - Enter and a plain click run it. With one
  // window and no link, the elsewhere surface's one-window rule opens a
  // new linked window without asking; the next Enter reuses the link.
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

/** A project's acts (t-48): f shows its tasks in this window - the
 * first act, so Enter and a plain click switch, as the picker's Enter
 * does - and d forgets it. Any project may be forgotten, the home and
 * the shown one included (the human, 2026-11-14: with one project,
 * forgetting it must be possible). */
function projectActs(target: RailTarget, deps: Pick<TasksMenuDeps, "ports" | "forget">, ports: TaskPorts): MenuAct[] {
  const acts: MenuAct[] = [
    {
      key: "f",
      action: {
        id: "tasks.project.show",
        label: "show its tasks here",
        run: () => ports.open(formatLocation({ machine: `tracker:${target.project}`, path: "tasks" })),
      },
    },
  ];
  acts.push({
    key: "d",
    action: {
      id: "tasks.project.forget",
      label: "forget this project (type its host to reach it again)",
      run: () => deps.forget(target.project),
    },
  });
  return acts;
}

/** The menu heading for a target: a project names the host it was
 * reached at, so a stray entry shows what it is. */
export function tasksMenuLabel(target: Readonly<{ kind: string; label: string }>, rail: RailTarget | undefined): string {
  if (target.kind !== "project" || rail === undefined || rail.host === null) return target.label;
  return `${target.label} (${rail.host})`;
}

export function taskRowActs(
  target: MenuTarget,
  deps: Pick<TasksMenuDeps, "machine" | "rowLink" | "ports" | "railTarget" | "forget">,
): readonly MenuAct[] {
  const ports = deps.ports();
  if (ports === null || target.part === null) return [];
  if (target.kind === "project" || target.kind === "agent") {
    const rail = deps.railTarget(target.part);
    if (rail === undefined) return [];
    if (rail.kind === "project") return projectActs(rail, deps, ports);
    if (rail.task === null) return [];
    return boardActs(formatLocation({ machine: `tracker:${rail.project}`, path: rail.task }), ports);
  }
  const machine = deps.machine();
  if (machine === null) return [];
  // the ROW'S LINK composed against the list's own tracker - never
  // the part path (that is identity, not a location)
  const link = deps.rowLink(target.part);
  if (link === null || link.length === 0) return [];
  return boardActs(formatLocation({ machine, path: link }), ports);
}

export function createTasksMenu(deps: TasksMenuDeps): TasksMenu {
  return createActMenu(
    (target) => taskRowActs(target, deps),
    { capture: (context) => deps.capture(context) },
  );
}
