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
  let timer: number | null = null;
  let generation = 0;

  function stop(): void {
    generation += 1;
    armedTask = null;
    if (timer !== null) {
      deps.cancel(timer);
      timer = null;
    }
  }

  function arm(task: number, token: number): void {
    if (token !== generation) return;
    timer = deps.schedule(FOLLOW_POLL_MS, () => {
      timer = null;
      void poll(task, token);
    });
  }

  async function poll(task: number, token: number): Promise<void> {
    try {
      const current = await deps.currentRevision(task);
      if (token !== generation || current === null) return;
      const shown = deps.shownRevision();
      if (shown === null || !pinnedEqual(shown, current)) await deps.refresh();
    } catch {
      // A failed read retries. Refresh failures report through the
      // normal pipeline; neither kind ends the follow intent.
    } finally {
      // Schedule after completion: slow reads/refreshes never overlap.
      arm(task, token);
    }
  }

  let armedTask: number | null = null;
  return {
    start(task) {
      stop();
      armedTask = task;
      arm(task, generation);
    },
    stop,
    /** Whether a follow poll is armed, and for which task (the S15
     * diagnosis's readout surface). */
    running(): { armed: boolean; task: number | null } {
      return { armed: armedTask !== null, task: armedTask };
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
