// The port's real contexts: the window's own surface commands and the
// Board instrument's reading commands. Every context validates at
// construction - collisions are errors at load (t-36@r49 standing
// policy, wired by r65), never silent shadowing.

import {
  commandContext,
  ctrl,
  key,
  sequence,
  shift,
  type InputContext,
} from "./input-context.ts";
import { validateNoCollisions } from "./key-matcher.ts";

// The window's own surfaces. The picker is the open-location input (a
// window surface, not a board-content action): space-space opens it,
// Escape closes the picker only while it is open. Find belongs to
// the reading instrument, not this window command table.
export interface WindowActions {
  openPicker(): void;
  closeOverlays(): void;
  pickerIsOpen(): boolean;
  trailBack(): boolean;
  trailForward(): boolean;
  /** space-/: find in the active board (the leader's find command;
   * the bare-/ reading binding stays). */
  enterFind(): void;
  /** space-l: the link act - labels over windows, a key sets the
   * link (the prototype's D1 modal shape). */
  linkByLabels(): void;
  /** space-u: clear this window's link. */
  unlink(): void;
  /** ?: the help surface - the which-key popup over the ACTIVE
   * context's whole command table (a recorded deliberate
   * difference from the prototype's hand-curated list). */
  showHelp(): void;
}

// The pinned chords: exactly the Ctrl chords the prototype claims
// (inst16.html:494-495) - Ctrl-o / Ctrl-i (the trail, the app's own
// back/forward since the old app) and Ctrl-f / b / d / u (paging, D5).
// The seam consults these before its modifier-chord refusal; every
// other ctrl/alt/meta chord still passes native (the IME/native-
// editing boundary set stays draft-open, grouped decision).
export const PINNED_CHORDS: readonly { key: string; ctrl: boolean }[] = [
  { key: "o", ctrl: true },
  { key: "i", ctrl: true },
  { key: "f", ctrl: true },
  { key: "b", ctrl: true },
  { key: "d", ctrl: true },
  { key: "u", ctrl: true },
];

// The modifier keys themselves: a bare press is never a command and
// never consumed (m-536: the trace showed key=Super consumed). The
// seam passes them through before any routing.
export const MODIFIER_KEYS: ReadonlySet<string> = new Set([
  "Shift", "Control", "Alt", "AltGraph", "Meta", "Super", "Hyper", "OS", "CapsLock", "NumLock", "Fn", "FnLock",
]);

export function isPinnedChord(stroke: {
  key: string;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
}): boolean {
  return PINNED_CHORDS.some(
    (chord) =>
      stroke.ctrl === chord.ctrl &&
      !stroke.shift &&
      !stroke.alt &&
      !stroke.meta &&
      stroke.key.toLowerCase() === chord.key,
  );
}

export function windowContext(actions: WindowActions): InputContext {
  const context = commandContext(() => [
    {
      action: {
        id: "window.open-picker",
        label: "open a location",
        run: () => actions.openPicker(),
      },
      keys: [sequence(key(" "), key(" "))],
    },
    {
      action: {
        id: "window.find",
        label: "find in this board",
        run: () => actions.enterFind(),
      },
      keys: [sequence(key(" "), key("/"))],
    },
    {
      action: {
        id: "window.link",
        label: "link to a window (labels)",
        run: () => actions.linkByLabels(),
      },
      keys: [sequence(key(" "), key("l"))],
    },
    {
      action: {
        id: "window.unlink",
        label: "unlink this window",
        run: () => actions.unlink(),
      },
      keys: [sequence(key(" "), key("u"))],
    },
    {
      action: {
        id: "window.help",
        label: "show the keys",
        run: () => actions.showHelp(),
      },
      // a real keyboard's ? is Shift+slash (key "?", shift set); a
      // synthesized one may carry either shape - both spellings fire
      // the same action (alternatives for one action)
      keys: [sequence(key("?")), sequence(shift("?"))],
    },
    {
      action: {
        id: "window.trail-back",
        label: "go back through the trail",
        run: () => {
          actions.trailBack();
        },
      },
      keys: [sequence(ctrl("o"))],
    },
    {
      action: {
        id: "window.trail-forward",
        label: "go forward through the trail",
        run: () => {
          actions.trailForward();
        },
      },
      keys: [sequence(ctrl("i"))],
    },
    ...(actions.pickerIsOpen() ? [{
      action: {
        id: "window.close-overlays",
        label: "close overlays",
        run: () => actions.closeOverlays(),
      },
      keys: [sequence(key("Escape"))],
    }] : []),
  ]);
  validateNoCollisions(context);
  return context;
}

// The figure inspection context (S04): the temporary surface's own
// keys - the prototype's `+ - =` (zoom in, zoom out, reset) and Escape
// (close, handing input back to the board's reading context). Every
// stroke is answered: unmatched keys are CONSUMED inert while the
// inspector is open (no silent fall-through to the board beneath).
export interface InspectionActions {
  zoomIn(): void;
  zoomOut(): void;
  zoomReset(): void;
  close(): void;
}

export function inspectionContext(actions: InspectionActions): InputContext {
  const ctx: InputContext = {
    ...commandContext(() => [
      {
        action: { id: "inspection.zoom-in", label: "zoom in", run: () => actions.zoomIn() },
        keys: [sequence(key("+"))],
      },
      {
        action: { id: "inspection.zoom-out", label: "zoom out", run: () => actions.zoomOut() },
        keys: [sequence(key("-"))],
      },
      {
        action: { id: "inspection.zoom-reset", label: "reset the zoom", run: () => actions.zoomReset() },
        keys: [sequence(key("="))],
      },
      {
        action: { id: "inspection.close", label: "close the inspection", run: () => actions.close() },
        keys: [sequence(key("Escape"))],
      },
    ]),
    // the inspector is modal: anything it does not define is consumed
    // WITHOUT action - the board beneath never sees it (no silent
    // fall-through, scenario 4)
    onUnmatched: () => ({ kind: "consume" }),
  };
  validateNoCollisions(ctx);
  return ctx;
}

// The Board instrument's reading context: viewport reading commands
// and the select-mode entry, declared by the content owner. The
// viewport operations belong to the buffer's own viewport port
// (start = top, end = bottom); the instrument wires them at connect
// time.
export interface ReadingActions {
  scrollToTop(): void;
  scrollToBottom(): void;
  scrollDown(): void;
  scrollUp(): void;
  /** The prototype's paging steps (inst16 :495/:505): d/u half the
   * client height, Ctrl-f/b 0.9, Ctrl-d/u 0.5. */
  scrollFraction(direction: -1 | 1, fraction: number): void;
  enterSelect(): void;
  enterFind(): void;
  /** n/N: step the committed search's matches (the r110 ruling);
   * a no-op when none exist. */
  nextMatch(): void;
  prevMatch(): void;
  /** Escape: clear the committed search when one is live; an inert
   * consume otherwise (the reading context claims the stroke). */
  clearSearch(): void;
}

// The Tasks instrument's reading context: the same viewport reading
// commands as the board's, with the pipeline's own reading shape - j/k
// is pure scrolling by task row (the scenario 17 resolution: NO row
// cursor), and find is a facility of a shown revision, so / is not
// offered on a task list.
export interface TasksReadingActions {
  scrollDown(): void;
  scrollUp(): void;
  scrollToTop(): void;
  scrollToBottom(): void;
  scrollFraction(direction: -1 | 1, fraction: number): void;
  enterSelect(): void;
}

export function tasksReadingContext(actions: TasksReadingActions): InputContext {
  const context = commandContext(() => [
    {
      action: {
        id: "tasks.page-half-down",
        label: "page down half the view",
        run: () => actions.scrollFraction(1, 0.5),
      },
      keys: [sequence(key("d"))],
    },
    {
      action: {
        id: "tasks.page-half-up",
        label: "page up half the view",
        run: () => actions.scrollFraction(-1, 0.5),
      },
      keys: [sequence(key("u"))],
    },
    {
      action: {
        id: "tasks.page-mostly-down",
        label: "page down most of the view",
        run: () => actions.scrollFraction(1, 0.9),
      },
      keys: [sequence(ctrl("f"))],
    },
    {
      action: {
        id: "tasks.page-mostly-up",
        label: "page up most of the view",
        run: () => actions.scrollFraction(-1, 0.9),
      },
      keys: [sequence(ctrl("b"))],
    },
    {
      action: {
        id: "tasks.page-ctrl-half-down",
        label: "page down half the view",
        run: () => actions.scrollFraction(1, 0.5),
      },
      keys: [sequence(ctrl("d"))],
    },
    {
      action: {
        id: "tasks.page-ctrl-half-up",
        label: "page up half the view",
        run: () => actions.scrollFraction(-1, 0.5),
      },
      keys: [sequence(ctrl("u"))],
    },
    {
      action: {
        id: "tasks.scroll-down",
        label: "scroll down",
        run: () => actions.scrollDown(),
      },
      keys: [sequence(key("j"))],
    },
    {
      action: {
        id: "tasks.scroll-up",
        label: "scroll up",
        run: () => actions.scrollUp(),
      },
      keys: [sequence(key("k"))],
    },
    {
      action: {
        id: "tasks.scroll-top",
        label: "scroll to the top",
        run: () => actions.scrollToTop(),
      },
      keys: [sequence(key("g"), key("g"))],
    },
    {
      action: {
        id: "tasks.scroll-bottom",
        label: "scroll to the bottom",
        run: () => actions.scrollToBottom(),
      },
      keys: [sequence(shift("G"))],
    },
    {
      action: {
        id: "tasks.select",
        label: "select a task by label",
        run: () => actions.enterSelect(),
      },
      keys: [sequence(key("s"))],
    },
  ]);
  validateNoCollisions(context);
  return context;
}

export function boardReadingContext(actions: ReadingActions): InputContext {
  const context = commandContext(() => [
    {
      action: {
        id: "board.scroll-down",
        label: "scroll down a line",
        run: () => actions.scrollDown(),
      },
      keys: [sequence(key("j"))],
    },
    {
      action: {
        id: "board.scroll-up",
        label: "scroll up a line",
        run: () => actions.scrollUp(),
      },
      keys: [sequence(key("k"))],
    },
    {
      action: {
        id: "board.scroll-top",
        label: "scroll to the top",
        run: () => actions.scrollToTop(),
      },
      keys: [sequence(key("g"), key("g"))],
    },
    {
      action: {
        id: "board.scroll-bottom",
        label: "scroll to the bottom",
        run: () => actions.scrollToBottom(),
      },
      keys: [sequence(shift("G"))],
    },
    {
      action: {
        id: "board.page-half-down",
        label: "page down half the view",
        run: () => actions.scrollFraction(1, 0.5),
      },
      keys: [sequence(key("d"))],
    },
    {
      action: {
        id: "board.page-half-up",
        label: "page up half the view",
        run: () => actions.scrollFraction(-1, 0.5),
      },
      keys: [sequence(key("u"))],
    },
    {
      action: {
        id: "board.page-mostly-down",
        label: "page down most of the view",
        run: () => actions.scrollFraction(1, 0.9),
      },
      keys: [sequence(ctrl("f"))],
    },
    {
      action: {
        id: "board.page-mostly-up",
        label: "page up most of the view",
        run: () => actions.scrollFraction(-1, 0.9),
      },
      keys: [sequence(ctrl("b"))],
    },
    {
      action: {
        id: "board.page-ctrl-half-down",
        label: "page down half the view",
        run: () => actions.scrollFraction(1, 0.5),
      },
      keys: [sequence(ctrl("d"))],
    },
    {
      action: {
        id: "board.page-ctrl-half-up",
        label: "page up half the view",
        run: () => actions.scrollFraction(-1, 0.5),
      },
      keys: [sequence(ctrl("u"))],
    },
    {
      action: {
        id: "board.find-next",
        label: "next match of the committed search",
        run: () => actions.nextMatch(),
      },
      keys: [sequence(key("n"))],
    },
    {
      action: {
        id: "board.find-prev",
        label: "previous match of the committed search",
        run: () => actions.prevMatch(),
      },
      // a real Shift+n arrives as key "N" with shift set (G's own
      // lesson, recorded): the binding names the arriving key
      keys: [sequence(shift("N"))],
    },
    {
      action: {
        id: "board.find-clear",
        label: "clear the committed search",
        run: () => actions.clearSearch(),
      },
      keys: [sequence(key("Escape"))],
    },
    {
      action: {
        id: "board.select",
        label: "select a target by label",
        run: () => actions.enterSelect(),
      },
      keys: [sequence(key("s"))],
    },
    {
      action: { id: "board.find", label: "find in this revision", run: () => actions.enterFind() },
      keys: [sequence(key("/"))],
    },
  ]);
  validateNoCollisions(context);
  return context;
}
