// The follow intent (scenario 15): a normally opened board (a bare
// tracker location, no pinned revision) follows current. A bounded
// poll re-reads the tracker's current revision; when it advances, the
// same address re-requests through the normal refresh pipeline -
// latest intent wins, failures retain content, and the poll itself
// never notifies. A pinned open (t-N@rK) never follows. The interval
// is a named tunable (t-36@r64 policy: value changes are mechanical).
//
// The loop itself is the instruments' shared one
// (lib/instruments/follow.ts); this module names the board's key (the
// task), its comparison (the pinned revision) and its interval.

import { createFollow as createSharedFollow } from "../instruments/follow.ts";

export const FOLLOW_POLL_MS = 3000;

export interface FollowDeps {
  /** The tracker's current pinned spec for the followed task
   * (t-N@rK), or null when the poll could not answer. */
  currentRevision(task: number): Promise<string | null>;
  /** The pinned spec the shown frame was resolved at (t-N@rK), or
   * null when nothing resolved. */
  shownRevision(): string | null;
  /** Re-request the bare address through the normal pipeline. */
  refresh(): void | Promise<unknown>;
  schedule(ms: number, run: () => void): number;
  cancel(id: number): void;
}

export interface Follow {
  /** Begin following a bare address; replaces any earlier follow. */
  start(task: number): void;
  stop(): void;
  /** Whether a follow poll is armed, and for which task (the S15
   * diagnosis's readout surface). */
  running(): { armed: boolean; task: number | null };
}

export function createFollow(deps: FollowDeps): Follow {
  const follow = createSharedFollow<number>({
    intervalMs: FOLLOW_POLL_MS,
    current: (task) => deps.currentRevision(task),
    shown: () => deps.shownRevision(),
    same: pinnedEqual,
    refresh: () => deps.refresh(),
    schedule: (ms, run) => deps.schedule(ms, run),
    cancel: (id) => deps.cancel(id),
  });
  return {
    start: (task) => follow.start(task),
    stop: () => follow.stop(),
    running() {
      const { armed, key } = follow.running();
      return { armed, task: key };
    },
  };
}

/** Both sides name a pinned spec; bare suffix comparison tolerates the
 * machine/path prefix each surface carries. */
export function pinnedEqual(shown: string, current: string): boolean {
  const tail = (spec: string): string => {
    const index = spec.indexOf("t-");
    return index < 0 ? spec : spec.slice(index);
  };
  return tail(shown) === tail(current);
}
