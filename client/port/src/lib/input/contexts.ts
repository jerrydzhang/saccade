// The port's real contexts: the window's own surface commands and the
// Board instrument's reading commands. Every context validates at
// construction - collisions are errors at load (t-36@r49 standing
// policy, wired by r65), never silent shadowing.

import {
  commandContext,
  key,
  sequence,
  shift,
  type InputContext,
} from "./input-context.ts";
import { validateNoCollisions } from "./key-matcher.ts";

// The window's own surfaces. The picker is the open-location input (a
// window surface, not a board-content action): space-space opens it,
// / enters Find, Escape closes overlays. Bindings beyond these ride
// the slices that build their surfaces (jump).
export interface WindowActions {
  openPicker(): void;
  closeOverlays(): void;
  openFind(): void;
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
        id: "window.enter-find",
        label: "find in this revision",
        run: () => actions.openFind(),
      },
      keys: [sequence(key("/"))],
    },
    {
      action: {
        id: "window.close-overlays",
        label: "close overlays",
        run: () => actions.closeOverlays(),
      },
      keys: [sequence(key("Escape"))],
    },
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
  ]);
  validateNoCollisions(context);
  return context;
}
