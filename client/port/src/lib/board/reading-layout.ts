// The board's reading layout: two independent scales the human asked
// for, held per window and carried by its window state.
//
// - zoom (+ - =) scales the board's text. The column grows with it, so
//   a line keeps its length in characters.
// - width (Ctrl-+ Ctrl-- Ctrl-=) scales the column alone: more or
//   fewer characters per line, so the margins shrink or grow. A wide
//   monitor need not stretch the text across the screen, and a narrow
//   one can drop its margins.
//
// The column never exceeds the window: the stage's max-width is the
// measure times both scales, and the window caps it.

export type ReadingLayout = Readonly<{ zoom: number; width: number }>;

export type ReadingLayoutChange =
  | "zoom-in"
  | "zoom-out"
  | "zoom-reset"
  | "widen"
  | "narrow"
  | "width-reset";

// The named tunables (value changes are mechanical, t-36@r64).
export const ZOOM_STEP = 0.1;
export const ZOOM_MIN = 0.7;
export const ZOOM_MAX = 2;
export const WIDTH_STEP = 0.1;
export const WIDTH_MIN = 0.6;
export const WIDTH_MAX = 3;

export const DEFAULT_READING_LAYOUT: ReadingLayout = { zoom: 1, width: 1 };

function clamp(value: number, min: number, max: number): number {
  // whole tenths: repeated float steps never drift (1.2000000000000002)
  return Math.min(max, Math.max(min, Math.round(value * 100) / 100));
}

export function changeReadingLayout(layout: ReadingLayout, change: ReadingLayoutChange): ReadingLayout {
  switch (change) {
    case "zoom-in":
      return { ...layout, zoom: clamp(layout.zoom + ZOOM_STEP, ZOOM_MIN, ZOOM_MAX) };
    case "zoom-out":
      return { ...layout, zoom: clamp(layout.zoom - ZOOM_STEP, ZOOM_MIN, ZOOM_MAX) };
    case "zoom-reset":
      return { ...layout, zoom: DEFAULT_READING_LAYOUT.zoom };
    case "widen":
      return { ...layout, width: clamp(layout.width + WIDTH_STEP, WIDTH_MIN, WIDTH_MAX) };
    case "narrow":
      return { ...layout, width: clamp(layout.width - WIDTH_STEP, WIDTH_MIN, WIDTH_MAX) };
    case "width-reset":
      return { ...layout, width: DEFAULT_READING_LAYOUT.width };
  }
}

/** A stored layout as the window state carries it: absent or
 * out-of-range values fall back to the default, never refuse. */
export function readingLayoutOf(stored: { zoom?: number | null; width?: number | null }): ReadingLayout {
  const valid = (value: number | null | undefined, min: number, max: number, fallback: number): number =>
    typeof value === "number" && Number.isFinite(value) ? clamp(value, min, max) : fallback;
  return {
    zoom: valid(stored.zoom, ZOOM_MIN, ZOOM_MAX, DEFAULT_READING_LAYOUT.zoom),
    width: valid(stored.width, WIDTH_MIN, WIDTH_MAX, DEFAULT_READING_LAYOUT.width),
  };
}

/** The CSS custom properties the board's stylesheet reads
 * (styles/app.css: .port-board-stage, .port-board-view). */
export function readingLayoutVars(layout: ReadingLayout): Record<string, string> {
  return {
    "--port-reading-zoom": String(layout.zoom),
    "--port-reading-width": String(layout.width),
  };
}

/** The window's holder of its reading layout: the current scales, a
 * key's change applied at once and written once the steps settle
 * (never per key press), and a bound state's stored layout restored
 * unless this window already changed its own - then this window's
 * layout is written to the state instead. The window supplies where
 * the CSS goes and how a write reaches its state. */
export interface ReadingLayoutHolder {
  current(): ReadingLayout;
  change(change: ReadingLayoutChange): void;
  /** The window's state bound (null when the bind read nothing). */
  bound(stored: { zoom?: number | null; width?: number | null } | null): void;
  /** Re-apply the current layout (a newly attached scroller). */
  apply(): void;
  destroy(): void;
}

export const READING_LAYOUT_SETTLE_MS = 400;

export function createReadingLayoutHolder(deps: {
  apply(vars: Record<string, string>): void;
  /** Write the layout to the window's state; a refusal is the
   * caller's to swallow (the next change writes again). */
  write(layout: ReadingLayout): void;
  schedule(ms: number, run: () => void): number;
  cancel(id: number): void;
}): ReadingLayoutHolder {
  let layout: ReadingLayout = DEFAULT_READING_LAYOUT;
  let changedHere = false;
  let timer: number | null = null;

  function show(next: ReadingLayout): void {
    layout = next;
    deps.apply(readingLayoutVars(layout));
  }

  function scheduleWrite(): void {
    if (timer !== null) deps.cancel(timer);
    timer = deps.schedule(READING_LAYOUT_SETTLE_MS, () => {
      timer = null;
      deps.write(layout);
    });
  }

  return {
    current: () => layout,
    change(change) {
      changedHere = true;
      show(changeReadingLayout(layout, change));
      scheduleWrite();
    },
    bound(stored) {
      if (changedHere) scheduleWrite();
      else if (stored !== null) show(readingLayoutOf(stored));
    },
    apply: () => show(layout),
    destroy() {
      if (timer !== null) deps.cancel(timer);
      timer = null;
    },
  };
}
