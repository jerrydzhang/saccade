// The port's real contexts: the window's own surface commands and the
// Board instrument's reading commands. Every context validates at
// construction - collisions are errors at load (the standing
// policy), never silent shadowing.

import {
  commandContext,
  ctrl,
  key,
  sequence,
  type CommandBinding,
  type InputContext,
} from "./input-context.ts";
import { validateNoCollisions } from "./key-matcher.ts";
import type { ReadingLayoutChange } from "../board/reading-layout.ts";
import { INSPECTION_PAN_PX } from "../figures/inspection.ts";

// The window's own surfaces. The picker is the open-location input (a
// window surface, not a board-content action): space-space opens it,
// Escape closes the picker only while it is open. Find belongs to
// the reading instrument, not this window command table: it is `/`
// alone, bound in the board's reading context (no space /).
/** The window chords, declared once. */
export const WINDOW_CHORDS = {
  picker: [key(" "), key(" ")],
  rail: [key(" "), key("r")],
  register: [key(" "), key("a")],
  keep: [key(" "), key("k")],
} as const;

export interface WindowActions {
  openPicker(): void;
  closeOverlays(): void;
  pickerIsOpen(): boolean;
  trailBack(): boolean;
  trailForward(): boolean;
  /** space-h: history mode on the active board: enter or leave. */
  toggleHistory?(): void;
  /** space-l: the link act - labels over windows, a key sets the
   * link (the prototype's D1 modal shape). */
  linkByLabels(): void;
  /** space-u: clear this window's link. */
  unlink(): void;
  /** space-r: show or hide the rail, per window, for the run. The
   * two rail keys are the human's first choice - they bind here, so
   * one line changes them. */
  toggleRail(): void;
  /** space-a: show the rail if it is hidden and open the add field
   * in it; Enter registers, nothing opens. */
  registerProject(): void;
  /** space-k: keep the location this window shows as a view on the
   * rail, beneath its repository. */
  keepView(): void;
  /** ?: the help surface - the which-key popup over the ACTIVE
   * context's whole command table (a recorded deliberate
   * difference from the prototype's hand-curated list). */
  showHelp(): void;
}

// The pinned chords: the Ctrl chords the prototype claims
// (inst16.html:494-495) - Ctrl-o / Ctrl-i (the trail, the app's own
// back/forward since the old app) and Ctrl-f / b / d / u (paging, D5) -
// and the human's column-width chords Ctrl-+ / Ctrl-- / Ctrl-= (the
// board's reading layout). The seam consults these before its
// modifier-chord refusal; every other ctrl/alt/meta chord still passes
// native (the IME/native-editing boundary set stays draft-open,
// grouped decision).
export const PINNED_CHORDS: readonly { key: string; ctrl: boolean }[] = [
  { key: "o", ctrl: true },
  { key: "i", ctrl: true },
  { key: "f", ctrl: true },
  { key: "b", ctrl: true },
  { key: "d", ctrl: true },
  { key: "u", ctrl: true },
  { key: "+", ctrl: true },
  { key: "-", ctrl: true },
  { key: "=", ctrl: true },
];

// The modifier keys themselves: a bare press is never a command and
// never consumed (the trace showed key=Super consumed). The
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
      // a symbol's character already carries Shift (the shift rule): Ctrl-+ is
      // Ctrl+Shift+= on a US layout; a letter chord never takes Shift
      (!stroke.shift || !/^[a-z]$/.test(chord.key)) &&
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
      keys: [sequence(...WINDOW_CHORDS.picker)],
    },
    ...(actions.toggleHistory === undefined
      ? []
      : [
          {
            action: {
              id: "window.history",
              label: "enter or leave history",
              run: () => actions.toggleHistory?.(),
            },
            keys: [sequence(key(" "), key("h"))],
          },
        ]),
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
        id: "window.toggle-rail",
        label: "show or hide the rail",
        run: () => actions.toggleRail(),
      },
      keys: [sequence(...WINDOW_CHORDS.rail)],
    },
    {
      action: {
        id: "window.register-project",
        label: "register a project (machine:/path/to/repository)",
        run: () => actions.registerProject(),
      },
      keys: [sequence(...WINDOW_CHORDS.register)],
    },
    {
      action: {
        id: "window.keep-view",
        label: "keep this location on the rail (beneath its repository)",
        run: () => actions.keepView(),
      },
      keys: [sequence(...WINDOW_CHORDS.keep)],
    },
    {
      action: {
        id: "window.help",
        label: "show the keys",
        run: () => actions.showHelp(),
      },
      // the character carries its Shift (input-context.ts
      // strokeMatches' shift rule): one spelling for every layout
      keys: [sequence(key("?"))],
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
  /** Move the stage by (dx, dy) CSS pixels. */
  pan(dx: number, dy: number): void;
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
      // h/j/k/l move the VIEW as j/k move it over a board (the
      // figure's own mapping): j
      // goes down the figure, so the stage moves up by the step
      {
        action: { id: "inspection.pan-left", label: "pan left", run: () => actions.pan(INSPECTION_PAN_PX, 0) },
        keys: [sequence(key("h"))],
      },
      {
        action: { id: "inspection.pan-down", label: "pan down", run: () => actions.pan(0, -INSPECTION_PAN_PX) },
        keys: [sequence(key("j"))],
      },
      {
        action: { id: "inspection.pan-up", label: "pan up", run: () => actions.pan(0, INSPECTION_PAN_PX) },
        keys: [sequence(key("k"))],
      },
      {
        action: { id: "inspection.pan-right", label: "pan right", run: () => actions.pan(-INSPECTION_PAN_PX, 0) },
        keys: [sequence(key("l"))],
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
  /** n/N: step the committed search's matches;
   * a no-op when none exist. */
  nextMatch(): void;
  prevMatch(): void;
  /** Escape: clear the committed search when one is live; an inert
   * consume otherwise (the reading context claims the stroke). */
  clearSearch(): void;
  /** + - = zoom the board's text; Ctrl-+ Ctrl-- Ctrl-= its column's
   * width (lib/board/reading-layout.ts). */
  readingLayout(change: ReadingLayoutChange): void;
  /** h l: step time in history mode; nothing outside it. */
  historyStep?(direction: -1 | 1): void;
  /** t: track the part at the reading height, or stop. */
  historyTrack?(): void;
  /** H L: the revision before or after that changed the tracked part. */
  historyWalk?(direction: -1 | 1): void;
}

// The frame's reading context: the shared reading keys,
// defined ONCE - j k d u f b g G over the viewport and s for select -
// owned by the frame, never by an instrument. An instrument extends it
// with bindings of its own (the task list adds `a`); a binding that
// names a frame key is a collision and fails at construction (the
// standing policy), never a silent rebinding. The task list's context
// IS this one, via the lease's readingContext(extra); the board's
// composes it with its own prefix until its own task.
export interface FrameReadingActions {
  scrollDown(): void;
  scrollUp(): void;
  scrollToTop(): void;
  scrollToBottom(): void;
  scrollFraction(direction: -1 | 1, fraction: number): void;
  enterSelect(): void;
}

export function frameReadingContext(
  actions: FrameReadingActions,
  extra: readonly CommandBinding[] = [],
  // The action ids' owner: the frame's own instruments read the
  // frame.* ids; an instrument still carrying its own observable ids
  // passes its own prefix - the key trace records invoke:<id> per
  // stroke, and the board's scroll ids are pinned observable facts
  // (tests/scenarios/key-trace.browser.ts pins invoke:board.scroll-down),
  // so the board's composition passes "board" until its own task
  // rules otherwise.
  owner = "frame",
): InputContext {
  const context = commandContext(() => [
    {
      action: {
        id: `${owner}.scroll-down`,
        label: "scroll down a line",
        run: () => actions.scrollDown(),
      },
      keys: [sequence(key("j"))],
    },
    {
      action: {
        id: `${owner}.scroll-up`,
        label: "scroll up a line",
        run: () => actions.scrollUp(),
      },
      keys: [sequence(key("k"))],
    },
    {
      action: {
        id: `${owner}.scroll-top`,
        label: "scroll to the top",
        run: () => actions.scrollToTop(),
      },
      keys: [sequence(key("g"), key("g"))],
    },
    {
      action: {
        id: `${owner}.scroll-bottom`,
        label: "scroll to the bottom",
        run: () => actions.scrollToBottom(),
      },
      keys: [sequence(key("G"))],
    },
    {
      action: {
        id: `${owner}.page-half-down`,
        label: "page down half the view",
        run: () => actions.scrollFraction(1, 0.5),
      },
      keys: [sequence(key("d"))],
    },
    {
      action: {
        id: `${owner}.page-half-up`,
        label: "page up half the view",
        run: () => actions.scrollFraction(-1, 0.5),
      },
      keys: [sequence(key("u"))],
    },
    {
      action: {
        id: `${owner}.page-mostly-down`,
        label: "page down most of the view",
        run: () => actions.scrollFraction(1, 0.9),
      },
      keys: [sequence(ctrl("f"))],
    },
    {
      action: {
        id: `${owner}.page-mostly-up`,
        label: "page up most of the view",
        run: () => actions.scrollFraction(-1, 0.9),
      },
      keys: [sequence(ctrl("b"))],
    },
    {
      action: {
        id: `${owner}.page-ctrl-half-down`,
        label: "page down half the view",
        run: () => actions.scrollFraction(1, 0.5),
      },
      keys: [sequence(ctrl("d"))],
    },
    {
      action: {
        id: `${owner}.page-ctrl-half-up`,
        label: "page up half the view",
        run: () => actions.scrollFraction(-1, 0.5),
      },
      keys: [sequence(ctrl("u"))],
    },
    {
      action: {
        id: `${owner}.select`,
        label: "select a target by label",
        run: () => actions.enterSelect(),
      },
      keys: [sequence(key("s"))],
    },
    ...extra,
  ]);
  validateNoCollisions(context);
  return context;
}

export function boardReadingContext(actions: ReadingActions): InputContext {
  // The shared keys are the frame's table, composed with the board's
  // own prefix (the key trace pins invoke:board.scroll-down); what is
  // below is the board's own: find and its stepping, Escape's
  // search-clear, the reading layout's zoom and column chords, and the
  // history keys when history mode exists. The board's select, menu
  // and targets stay hand-wired until its own task.
  return frameReadingContext(
    {
      scrollDown: () => actions.scrollDown(),
      scrollUp: () => actions.scrollUp(),
      scrollToTop: () => actions.scrollToTop(),
      scrollToBottom: () => actions.scrollToBottom(),
      scrollFraction: (direction, fraction) => actions.scrollFraction(direction, fraction),
      enterSelect: () => actions.enterSelect(),
    },
    [
      {
        action: { id: "board.find", label: "find in this revision", run: () => actions.enterFind() },
        keys: [sequence(key("/"))],
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
        // N is the character (its case carries Shift)
        keys: [sequence(key("N"))],
      },
      {
        action: {
          id: "board.find-clear",
          label: "clear the committed search (in history: stop tracking)",
          run: () => actions.clearSearch(),
        },
        keys: [sequence(key("Escape"))],
      },
      ...(
        [
          ["zoom-in", "larger text", key("+")],
          ["zoom-out", "smaller text", key("-")],
          ["zoom-reset", "text at its normal size", key("=")],
          ["widen", "a wider column", ctrl("+")],
          ["narrow", "a narrower column", ctrl("-")],
          ["width-reset", "the column at its normal width", ctrl("=")],
        ] as const
      ).map(([change, label, stroke]) => ({
        action: { id: `board.${change}`, label, run: () => actions.readingLayout(change) },
        keys: [sequence(stroke)],
      })),
      ...(actions.historyStep === undefined
        ? []
        : [
            {
              action: {
                id: "board.history-older",
                label: "history: the revision before",
                run: () => actions.historyStep?.(-1),
              },
              keys: [sequence(key("h"))],
            },
            {
              action: {
                id: "board.history-newer",
                label: "history: the revision after",
                run: () => actions.historyStep?.(1),
              },
              keys: [sequence(key("l"))],
            },
            {
              action: {
                id: "board.history-track",
                label: "history: track the part being read, or stop",
                run: () => actions.historyTrack?.(),
              },
              keys: [sequence(key("t"))],
            },
            {
              action: {
                id: "board.history-track-older",
                label: "history: the change to the tracked part before",
                run: () => actions.historyWalk?.(-1),
              },
              // H and L are the characters (their case carries Shift)
              keys: [sequence(key("H"))],
            },
            {
              action: {
                id: "board.history-track-newer",
                label: "history: the change to the tracked part after",
                run: () => actions.historyWalk?.(1),
              },
              keys: [sequence(key("L"))],
            },
          ]),
    ],
    "board",
  );
}
