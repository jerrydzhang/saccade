// The follow loop every instrument that shows the record's now shares
// (the board's S15 follow, the task list's): a bounded poll reads what
// the source answers now; when it differs from what is shown, the
// instrument refreshes through its normal pipeline - failures retain
// content, and the poll itself never notifies. Each read is scheduled
// after the previous one completes, so slow reads never overlap; stop
// invalidates any read in flight.

export interface FollowDeps<K> {
  /** The interval between reads (a named tunable per instrument). */
  intervalMs: number;
  /** What the source answers now for the followed key, or null when
   * the read failed (it retries on the next tick). */
  current(key: K): Promise<string | null>;
  /** What the shown frame was read at, or null when nothing is shown. */
  shown(): string | null;
  /** Whether the shown and current answers name the same content. */
  same(shown: string, current: string): boolean;
  /** Re-request through the instrument's normal pipeline. */
  refresh(): void | Promise<unknown>;
  schedule(ms: number, run: () => void): number;
  cancel(id: number): void;
}

export interface Follow<K> {
  /** Begin following; replaces any earlier follow. */
  start(key: K): void;
  stop(): void;
  /** Whether a poll is armed, and for which key. */
  running(): { armed: boolean; key: K | null };
}

export function createFollow<K>(deps: FollowDeps<K>): Follow<K> {
  let timer: number | null = null;
  let generation = 0;
  let armedKey: K | null = null;

  function stop(): void {
    generation += 1;
    armedKey = null;
    if (timer !== null) {
      deps.cancel(timer);
      timer = null;
    }
  }

  function arm(key: K, token: number): void {
    if (token !== generation) return;
    timer = deps.schedule(deps.intervalMs, () => {
      timer = null;
      void poll(key, token);
    });
  }

  async function poll(key: K, token: number): Promise<void> {
    try {
      const current = await deps.current(key);
      if (token !== generation || current === null) return;
      const shown = deps.shown();
      if (shown === null || !deps.same(shown, current)) await deps.refresh();
    } catch {
      // A failed read retries. Refresh failures report through the
      // normal pipeline; neither kind ends the follow.
    } finally {
      arm(key, token);
    }
  }

  return {
    start(key) {
      stop();
      armedKey = key;
      arm(key, generation);
    },
    stop,
    running: () => ({ armed: armedKey !== null, key: armedKey }),
  };
}
