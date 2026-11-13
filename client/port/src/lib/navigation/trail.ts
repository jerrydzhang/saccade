// The window's trail: a map of explicit openings, never a time
// machine. Only explicit openings add entries (same-board references
// included; scrolling, scrubbing and Shift-G add nothing); each entry
// carries the DEPARTURE place captured as the window left it; Back
// restores the departure place of the previous entry, Forward
// retraces; neither ever touches folds, zoom or drafts - the saved
// place is address + reading anchor + viewport offset only (t-36@r50
// scenario 14). Pure: the host supplies capture and restore.

import type { SavedPlace } from "../contracts/places.ts";

export type TrailEntry = Readonly<{
  /** The location this entry was opened AS (the asked spec). */
  location: string;
  /** The place the window left when it moved on (captured at
   * departure); Back restores exactly this. */
  departure: SavedPlace | null;
}>;

export interface TrailHost {
  /** Capture the currently displayed place (the departure capture). */
  capture(): SavedPlace;
  /** Restore a place through the buffer's normal placement. */
  restore(place: SavedPlace): void;
}

export interface TrailView {
  readonly depth: number;
  readonly cursor: number;
  readonly entries: readonly { location: string; hasDeparture: boolean }[];
  readonly canBack: boolean;
  readonly canForward: boolean;
}

export interface Trail {
  /** An explicit opening: the departure place lands on the CURRENT
   * entry, the new location pushes; forward history truncates. */
  opened(location: string): void;
  back(): boolean;
  forward(): boolean;
  view(): TrailView;
}

export function createTrail(host: TrailHost): Trail {
  const entries: { location: string; departure: SavedPlace | null }[] = [];
  let cursor = -1;

  function captureDeparture(): void {
    const current = cursor >= 0 ? entries[cursor] : undefined;
    if (current !== undefined) current.departure = host.capture();
  }

  return {
    opened(location) {
      captureDeparture();
      entries.splice(cursor + 1);
      entries.push({ location, departure: null });
      cursor = entries.length - 1;
    },
    back() {
      if (cursor <= 0) return false;
      // The place to return to is the one recorded when the window
      // LEFT the previous entry (captured at the opening that moved
      // on) - Back never invents a position.
      const previous = entries[cursor - 1];
      if (previous === undefined || previous.departure === null) return false;
      captureDeparture();
      host.restore(previous.departure);
      cursor -= 1;
      return true;
    },
    forward() {
      if (cursor >= entries.length - 1) return false;
      const next = entries[cursor + 1];
      if (next === undefined) return false;
      // Forward retraces: restore the departure the window recorded
      // when it left that entry (its own departure), or nothing when
      // it never moved on.
      if (next.departure === null) return false;
      captureDeparture();
      host.restore(next.departure);
      cursor += 1;
      return true;
    },
    view() {
      return {
        depth: entries.length,
        cursor,
        entries: entries.map((entry) => ({
          location: entry.location,
          hasDeparture: entry.departure !== null,
        })),
        canBack: cursor > 0 && entries[cursor - 1]?.departure != null,
        canForward: cursor < entries.length - 1 && entries[cursor + 1]?.departure != null,
      };
    },
  };
}
