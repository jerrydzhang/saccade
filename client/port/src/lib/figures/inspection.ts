// The figure inspection surface's pure half (S04): the zoom ladder
// and the state shape. The inspector is a TEMPORARY window surface -
// it dies on close, never enters the trail, and while it is open the
// board beneath keeps its updates but receives no input (deactivation
// decouples input from content, the r46-55 contracts).

export type InspectionState = Readonly<{
  /** The inspected figure's canonical part path. */
  part: string;
  /** The displayed asset's src at open time (the honest "enlarge
   * what is shown" - a data: URL the hydration produced). */
  src: string;
  /** A text figure's text (src is then empty): enlarged as text. */
  text?: string;
  /** A rendered diagram's SVG markup (src is then empty): the shown
   * diagram itself, cloned as the prototype clones it. */
  svg?: string;
  /** The figure's caption, as the board rendered it. */
  caption: string;
  zoom: number;
  /** The pan offset in CSS pixels from the centred position. */
  x: number;
  y: number;
}>;

// The prototype's enlarge (inst16.html:396-400): z starts at 1.2, the
// keys + and - multiply by 1.2, a wheel notch by 1.12, = resets to 1.2
// and the centre, always clamped to 0.3-8. Multiplicative - never a
// fixed-step ladder.
export const ZOOM_MIN = 0.3;
export const ZOOM_MAX = 8;
const KEY_FACTOR = 1.2;
const WHEEL_FACTOR = 1.12;

export function zoomIn(zoom: number): number {
  return clamp(zoom * KEY_FACTOR);
}

export function zoomOut(zoom: number): number {
  return clamp(zoom / KEY_FACTOR);
}

/** One wheel notch: up (deltaY < 0) zooms in, down zooms out. */
export function zoomWheel(zoom: number, deltaY: number): number {
  return clamp(deltaY < 0 ? zoom * WHEEL_FACTOR : zoom / WHEEL_FACTOR);
}

export function zoomReset(): number {
  return 1.2;
}

function clamp(value: number): number {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, value));
}
