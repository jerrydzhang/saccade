// The frame's pointer guards and the middle button's lifecycle,
// extracted from the board's element-by-element wiring (lib/board/
// hydrate.ts) so the frame's surface runs them for every instrument.
// Pure over an injected clock: the surface supplies performance.now
// and the DOM events, the tests supply their own.

// A plain click within this window of a right gesture is the
// WebDriver's synthesized artifact, never a user click: the stack
// synthesizes one click right after the right-button release (the
// recorded live probe: contextmenu, then click:button=0 ~120ms
// later). The window is generous against the synthesis order and
// tiny against any human press.
export const RIGHT_GESTURE_WINDOW_MS = 250;

// The synthesized click a middle release produces is dropped within
// the same window: the release already ran the modified click, and
// the stray click would run the plain one after it.
export const STRAY_CLICK_WINDOW_MS = 250;

export interface PointerGuards {
  /** A right gesture landed on the surface: the artifact window
   * opens. */
  contextmenu(): void;
  /** Whether a plain click now is the synthesized post-contextmenu
   * artifact, to drop. */
  rightGestureArtifact(): boolean;
  /** A middle button press (the caller preventDefaults): armed, not
   * acted - a middle press does not arrive as a click. */
  middlePress(): void;
  /** The release over the element, when armed: true, and the caller
   * runs the modified click. The release's own button is not trusted
   * (this stack reports a middle release as button 0) - the press
   * decides, the mouseup over the armed element acts. */
  middleRelease(): boolean;
  /** The pointer left the surface: the press is cancelled. */
  middleLeave(): void;
  /** Whether this auxclick is the middle button's, to prevent (the
   * platform's middle-click behaviour stays out). */
  middleAuxClick(button: number): boolean;
  /** Whether a plain click now is the stray click a middle release
   * synthesizes after acting, to drop. */
  middleStrayClick(): boolean;
}

export function createPointerGuards(now: () => number): PointerGuards {
  let contextmenuAt: number | null = null;
  let armed = false;
  let actedAt: number | null = null;

  return {
    contextmenu(): void {
      contextmenuAt = now();
    },

    rightGestureArtifact(): boolean {
      return contextmenuAt !== null && now() - contextmenuAt < RIGHT_GESTURE_WINDOW_MS;
    },

    middlePress(): void {
      armed = true;
    },

    middleRelease(): boolean {
      if (!armed) return false;
      armed = false;
      actedAt = now();
      return true;
    },

    middleLeave(): void {
      armed = false;
    },

    middleAuxClick(button: number): boolean {
      return button === 1;
    },

    middleStrayClick(): boolean {
      return actedAt !== null && now() - actedAt < STRAY_CLICK_WINDOW_MS;
    },
  };
}
