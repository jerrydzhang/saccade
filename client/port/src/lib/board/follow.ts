// The follow intent (scenario 15): a normally opened board (a bare
// tracker location, no pinned revision) follows current. A bounded
// poll re-reads the tracker's current revision; when it advances, the
// same address re-requests through the normal refresh pipeline -
// latest intent wins, failures retain content, and the poll itself
// never notifies. A pinned open (t-N@rK) never follows. The interval
// is a named tunable (t-36@r64 policy: value changes are mechanical).

export const FOLLOW_POLL_MS = 3000;

export interface FollowDeps {
  /** The tracker's current pinned spec for the followed task
   * (t-N@rK), or null when the poll could not answer. */
  currentRevision(): Promise<string | null>;
  /** The pinned spec the shown frame was resolved at (t-N@rK), or
   * null when nothing resolved. */
  shownRevision(): string | null;
  /** Re-request the bare address through the normal pipeline. */
  refresh(): void;
  schedule(ms: number, run: () => void): number;
  cancel(id: number): void;
}

export interface Follow {
  /** Begin following a bare address; replaces any earlier follow. */
  start(task: number): void;
  stop(): void;
}

export function createFollow(deps: FollowDeps): Follow {
  let timer: number | null = null;

  function stop(): void {
    if (timer !== null) {
      deps.cancel(timer);
      timer = null;
    }
  }

  async function poll(): Promise<void> {
    const current = await deps.currentRevision();
    if (current === null) return;
    const shown = deps.shownRevision();
    if (shown === null) return;
    if (pinnedEqual(shown, current)) return;
    deps.refresh();
  }

  return {
    start(task) {
      stop();
      void task; // the task rides in the deps' closures (the source
      // reads it); the follow itself only polls
      timer = deps.schedule(FOLLOW_POLL_MS, () => {
        void poll();
      });
    },
    stop,
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
