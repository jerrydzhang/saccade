// The one interface both scenario drivers implement (browser.ts:
// WebKit through vitest browser mode; native.ts: the built app
// through tauri-driver and WebdriverIO). Scenarios and helpers.ts
// speak only this: state is read through the port's observation
// readout, input is real keys and clicks, and controls are named by
// their data-port-control id - never by a selector in a scenario.
import type { PortObservation } from "../../../src/lib/observation.ts";

export type { PortObservation };

/** The throwaway seeded tracker every scenario run starts from. */
export type SeededTracker = {
  /** http://127.0.0.1:<port> */
  base: string;
  /** 127.0.0.1:<port>, the form a location spec carries */
  host: string;
  /** the served repo's name, the location's first path segment */
  repo: string;
  /** the t36 seed tasks (client/harness/t36_seeds.py), as t-N ids */
  tasks: { cited: string; citing: string; citingChild: string; landing: string };
};

/** One stored window state as the tracker's window store answers it
 * (client/port/src/lib/windows.ts WindowState, the fields read here). */
export type WindowStateView = { id: string; location: string | null; attachedHere: boolean; link: string | null };

/** The throwaway tracker's door, mid-scenario (kit/tracker-wire.ts). */
export interface TrackerDoor {
  createTask(name: string): Promise<string>;
  /** Commit a board, with any figure files beside board.yaml. */
  commit(
    task: string,
    board: string,
    message: string,
    files?: readonly { path: string; content: string | Uint8Array }[],
  ): Promise<{ revision: number }>;
  /** A window state, seen from `device`. */
  windowState(id: string, device: string): Promise<WindowStateView>;
  /** Fixture: a window state born and attached on `device`. */
  attachedWindow(id: string, device: string): Promise<void>;
}

export type Tracker = SeededTracker & TrackerDoor;

declare module "vitest" {
  interface ProvidedContext {
    tracker: SeededTracker;
  }
}

export type Box = { left: number; top: number; right: number; bottom: number };
/** AppDriver.menuGeometry's answer. */
export type MenuGeometry = { menu: Box; target: Box | null; part: string; viewport: { width: number; height: number } };

/** One painted find mark (AppDriver.findMarks). */
export type FindMark = { part: string | null; text: string; active: boolean };

/** A fresh app: one window, no launch restore, its own data. */
export interface AppDriver {
  /** The readout of a window: the focused one by default. */
  readout(window?: string): Promise<PortObservation>;
  /** Real key presses in order. Names: one character ("n", "/", " "),
   * a named key ("Escape", "Enter", "Backspace", "ArrowDown"), or a
   * Ctrl chord ("Ctrl-f"). */
  press(...keys: string[]): Promise<void>;
  /** Real typing into whatever holds focus, character by character. */
  type(text: string): Promise<void>;
  /** A real click on a port control (its data-port-control id). */
  click(control: string, options?: { button?: "left" | "right" }): Promise<void>;
  /** A real click on a rendered content part (its canonical content
   * path, as the readout's content.parts names it). "header" clicks
   * the part's top strip - a group's own hit area, where its center
   * would land on a child. */
  clickPart(part: string, options?: { button?: "left" | "right"; at?: "center" | "header" }): Promise<void>;
  /** A real pointer move onto a content part's center (no click). */
  pointAt(part: string): Promise<void>;
  /** How a rendered part looks: its computed background and text
   * colour, or null when not rendered. */
  partLook(part: string): Promise<{ background: string; color: string } | null>;
  /** A real mouse drag from a control's center by (dx, dy). */
  drag(control: string, dx: number, dy: number): Promise<void>;
  /** A real wheel notch over a control (deltaY < 0 is up). The native
   * stack delivers no wheel events (WebKitWebDriver), so the native
   * driver refuses it: wheel scenarios are browser-only. */
  wheel(control: string, deltaY: number): Promise<void>;
  /** A rendered control's box in viewport coordinates, or null. */
  controlBox(control: string): Promise<Box | null>;
  /** A rendered control's visible text, or null. */
  controlText(control: string): Promise<string | null>;
  /** Whether a port control is rendered now. */
  shows(control: string): Promise<boolean>;
  /** The window strip's fields as rendered (location, status, link). */
  strip(): Promise<{ location: string; status: string; link: string }>;
  /** The select labels painted now (after s): each chip's text and
   * the part it marks. */
  labels(): Promise<{ label: string; part: string }[]>;
  /** Whether an act menu is open now. */
  menuOpen(): Promise<boolean>;
  /** The open act menu's box and its target part's box, in viewport
   * (client) coordinates, with the viewport's size; null when no menu
   * is open. The menu names its target by data-port-menu-part. */
  menuGeometry(): Promise<MenuGeometry | null>;
  /** The content part visibly marked as find's active match, or null.
   * Reads the data-port-find-active mark (D2, lib/find/marks.ts). */
  activeMatch(): Promise<string | null>;
  /** Find's painted marks in document order (D2): each mark's text as
   * shown, the content part it sits in, and whether it is the active
   * one. A mark is a [data-port-find-match] element; a part-level
   * fallback mark reports the whole part's text. */
  findMarks(): Promise<FindMark[]>;
  /** A rendered content part's visible text (innerText), or null when
   * the part is not rendered: what a reader sees there. */
  partText(part: string): Promise<string | null>;
  /** The labels of the live windows ("main" first). The browser has
   * exactly one. */
  windows(): Promise<string[]>;
  /** Route input and the default readout to a window. */
  focusWindow(label: string): Promise<void>;
}

/** The readout's input field (batch 3, served by D6): the input
 * matcher's live state, the observable the which-key, help and timeout
 * scenarios read. Strokes and keys use the port's own help labels
 * (src/lib/input/input-context.ts formatStroke/formatSequence: "spc",
 * "g g", "Ctrl-f", "Shift-G"). */
export type InputView = {
  /** The pending strokes of an unfinished sequence, or [] */
  pending: string[];
  /** The which-key popup's content while it is shown, else null: the
   * pending prefix ("" for help) and every continuation in the active
   * context's tables, with its full key sequence and label. */
  whichKey: { prefix: string; entries: { keys: string; label: string; actionId: string }[] } | null;
  /** The named tunables, read by the scenarios, never pinned there. */
  tunables: { whichKeyDelayMs: number; whichKeySuspendsTimeout: boolean; sequenceTimeoutMs: number };
};

/** The input field of a readout (undefined on a build without it). */
export function inputOf(obs: PortObservation): InputView | undefined {
  return (obs as PortObservation & { input?: InputView }).input;
}
