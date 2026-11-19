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
  /** the served repo's name, the declaration's project name */
  repo: string;
  /** the served repo's absolute path: what the app registers */
  repoPath: string;
  /** the t36 seed tasks (client/harness/t36_seeds.py), as t-N ids */
  tasks: { cited: string; citing: string; citingChild: string; landing: string };
};

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
  /** Real key presses in order. Names: one character ("n", "/", " ";
   * a US-layout shifted symbol such as "+" or "?" arrives with Shift
   * held, as a person types it), a named key ("Escape", "Enter",
   * "Backspace", "ArrowDown", "NumpadAdd"), a Ctrl chord ("Ctrl-f"), or
   * a Shift chord on a named key ("Shift-Enter"). */
  press(...keys: string[]): Promise<void>;
  /** Real typing into whatever holds focus, character by character. */
  type(text: string): Promise<void>;
  /** A real click on a port control (its data-port-control id). */
  /** A real click on a port control (its data-port-control id);
   * `shift` holds Shift through the click, "middle" is the wheel
   * button (native only: the open-elsewhere gestures, t-47). */
  click(control: string, options?: { button?: "left" | "right" | "middle"; shift?: boolean }): Promise<void>;
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
  /** A rendered part's box in viewport coordinates, or null when it is
   * not rendered or has no size (hidden inside a folded group). */
  partBox(part: string): Promise<Box | null>;
  /** The open act menu's act keys, in its order; [] when none is open. */
  menuActs(): Promise<string[]>;
  /** A real click on the open act menu's item for an act key. */
  clickMenuAct(key: string): Promise<void>;
  /** Quit the app and launch it again on the SAME data directory (the
   * launch restore's journey, S20). Native only: the browser driver
   * has no app process to relaunch and refuses. */
  relaunch(): Promise<void>;
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
  /** The window strip's fields as rendered (location, status, link);
   * status is "" while the view is simply loaded (the strip shows it
   * only when loading or failed). */
  strip(): Promise<{ location: string; status: string; link: string }>;
  /** The select labels painted now (after s): each chip's text, the
   * part it marks, and whether it is drawn nested (t-63: its target
   * sits inside another labelled one). */
  labels(): Promise<{ label: string; part: string; nested: boolean }[]>;
  /** Whether an act menu is open now. */
  menuOpen(): Promise<boolean>;
  /** The open act menu's label as rendered (its target named in full),
   * or null when no menu is open. */
  menuLabel(): Promise<string | null>;
  /** The window's viewport resized to (width, height) CSS pixels, as a
   * person narrows a window; resolves once the page sees the size. The
   * native lane's kiosk-shell holds every window at the output's size,
   * so the native driver refuses it: resize scenarios are browser-only. */
  resize(width: number, height: number): Promise<void>;
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
 * "g g", "Ctrl-f", "G"). */
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
