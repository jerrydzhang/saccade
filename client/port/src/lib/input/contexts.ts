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
}

// The two oracle-pinned window chords (the app's own back/forward
// bindings since the old app: Ctrl-o, Ctrl-i). The seam consults
// these before its modifier-chord refusal; every other ctrl/alt/meta
// chord still passes native (the IME/native-editing boundary set
// stays draft-open, grouped decision).
export const PINNED_WINDOW_CHORDS: readonly { key: string; ctrl: boolean }[] = [
  { key: "o", ctrl: true },
  { key: "i", ctrl: true },
];

export function isPinnedWindowChord(stroke: {
  key: string;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
}): boolean {
  return PINNED_WINDOW_CHORDS.some(
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

// The Board instrument's reading context: viewport reading commands
// and the select-mode entry, declared by the content owner. The
// viewport operations belong to the buffer's own viewport port
// (start = top, end = bottom); the instrument wires them at connect
// time.
export interface ReadingActions {
  scrollToTop(): void;
  scrollToBottom(): void;
  enterSelect(): void;
  enterFind(): void;
}

export function boardReadingContext(actions: ReadingActions): InputContext {
  const context = commandContext(() => [
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
