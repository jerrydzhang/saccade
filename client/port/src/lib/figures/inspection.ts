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
  /** The figure's caption, as the board rendered it. */
  caption: string;
  zoom: number;
}>;

export const ZOOM_MIN = 0.5;
export const ZOOM_MAX = 5;
const ZOOM_STEP = 0.25;

/** The prototype's `+ - =` ladder: quarter steps between half and
 * five, clamped - a step that would leave the range keeps the value
 * (the + at max does nothing, exactly as the felt behaviour). */
export function zoomIn(zoom: number): number {
  return clamp(round(zoom + ZOOM_STEP));
}

export function zoomOut(zoom: number): number {
  return clamp(round(zoom - ZOOM_STEP));
}

export function zoomReset(): number {
  return 1;
}

function clamp(value: number): number {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, value));
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
