// The window's place as its window state carries it: the location the
// window shows (as it was asked - a bare location stays bare and keeps
// following, a pinned revision stays pinned) and the part at the
// reading height. A relaunch, or another device opening the same
// state, reopens there (t-38).
//
// Writes settle, never per scroll frame or per follow refresh: any
// change of what the window shows or where it reads schedules one
// check after the settle delay, and the check writes only a place that
// differs from the last one the state holds. A placement in flight (an
// open's anchor still being revealed, a trail restore) holds the check
// until it ends, so a half-placed window never overwrites the place it
// is restoring.

import type { RestoreStatus, RestoringPlace, SavedPlace } from "../contracts/places.ts";
import { isPlane } from "../location.ts";

export type WindowPlace = Readonly<{ location: string; part: string | null }>;

/** The place a shown frame stands for: its ASKED location (never the
 * revision a bare location resolved to, or the relaunch would pin
 * what was following) and the board part the reading height sits in
 * (the trail's own capture; a task row or the default position is no
 * part). The control plane remembers no project (the human,
 * 2026-11-14): its place is the bare plane, which shows the home
 * project on a relaunch. Null while the frame is loading or carries no
 * location. */
export function shownPlaceOf(
  content: { asked?: unknown; status?: unknown } | undefined,
  capture: () => SavedPlace,
): WindowPlace | null {
  if (content === undefined || typeof content.asked !== "string" || content.asked.length === 0) return null;
  if (content.status === "loading") return null;
  let part: string | null = null;
  try {
    const captured = capture();
    const position = captured.position;
    if (isPlane(captured.address)) {
      part = null;
    } else if (position.kind === "anchor") {
      const anchor = position.anchor as { kind?: unknown; path?: unknown } | null;
      if (anchor !== null && anchor.kind === "board-part" && typeof anchor.path === "string") part = anchor.path;
    }
  } catch {
    // a buffer between lives captures nothing: no part
  }
  return { location: content.asked, part };
}

/** A placement job's end, whatever it ended as. */
export function placementDone(job: RestoringPlace): Promise<RestoreStatus> {
  return new Promise((resolve) => {
    const pending = (status: RestoreStatus) => status === "pending-content" || status === "pending-layout";
    if (!pending(job.snapshot())) {
      resolve(job.snapshot());
      return;
    }
    const stop = job.subscribe((status) => {
      if (pending(status)) return;
      stop();
      resolve(status);
    });
  });
}

export interface WindowPlaceHolder {
  /** What the window shows or where it reads may have moved. */
  changed(): void;
  /** A placement started; the check waits for `done`. */
  placing(done: Promise<unknown>): void;
  /** The window's state bound: the place it already holds (null
   * fields when it holds none). */
  bound(stored: { location: string | null; part: string | null }): void;
  destroy(): void;
}

export const WINDOW_PLACE_SETTLE_MS = 400;

export function samePlace(a: WindowPlace | null, b: WindowPlace | null): boolean {
  if (a === null || b === null) return a === b;
  return a.location === b.location && a.part === b.part;
}

export function createWindowPlaceHolder(deps: {
  /** The place the window shows now; null while it shows nothing
   * settled (empty, loading). */
  current(): WindowPlace | null;
  /** Write the place to the window's state; a refusal is the caller's
   * to swallow (the next change writes again). */
  write(place: WindowPlace): void;
  schedule(ms: number, run: () => void): number;
  cancel(id: number): void;
}): WindowPlaceHolder {
  let isBound = false;
  let written: WindowPlace | null = null;
  let placements = 0;
  let timer: number | null = null;
  let destroyed = false;

  function check(): void {
    if (destroyed || !isBound || placements > 0) return;
    const place = deps.current();
    if (place === null || samePlace(place, written)) return;
    written = place;
    deps.write(place);
  }

  function scheduleCheck(): void {
    if (destroyed) return;
    if (timer !== null) deps.cancel(timer);
    timer = deps.schedule(WINDOW_PLACE_SETTLE_MS, () => {
      timer = null;
      check();
    });
  }

  return {
    changed: scheduleCheck,
    placing(done) {
      placements += 1;
      const end = () => {
        placements -= 1;
        scheduleCheck();
      };
      done.then(end, end);
    },
    bound(stored) {
      isBound = true;
      written =
        stored.location === null ? null : { location: stored.location, part: stored.part };
      // a change made before the state bound (the first open is what
      // binds it) writes now
      scheduleCheck();
    },
    destroy() {
      destroyed = true;
      if (timer !== null) deps.cancel(timer);
      timer = null;
    },
  };
}
