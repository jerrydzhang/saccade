// The task list follows the record (the board's S15 follow, for the
// pipeline page): while the list is shown, a bounded poll re-reads the
// page; when its rows differ from the shown ones, the list re-requests
// through the normal refresh pipeline - failures retain content, and
// the poll itself never notifies. The page has no revision to compare,
// so the comparison is the page model's own content (the caller's
// signature). The interval is the board follow's (FOLLOW_POLL_MS's
// value), a named tunable here.

export const TASKS_FOLLOW_POLL_MS = 3000;

export interface TasksFollowDeps {
  /** The page as it stands now, as a signature; null when the read
   * failed (a failed read retries on the next tick). */
  current(): Promise<string | null>;
  /** The shown page's signature, or null when nothing is shown. */
  shown(): string | null;
  /** Re-request the list through the normal pipeline. */
  refresh(): void | Promise<unknown>;
  schedule(ms: number, run: () => void): number;
  cancel(id: number): void;
}

export interface TasksFollow {
  start(): void;
  stop(): void;
  armed(): boolean;
}

export function createTasksFollow(deps: TasksFollowDeps): TasksFollow {
  let timer: number | null = null;
  let generation = 0;
  let armed = false;

  function stop(): void {
    generation += 1;
    armed = false;
    if (timer !== null) {
      deps.cancel(timer);
      timer = null;
    }
  }

  function arm(token: number): void {
    if (token !== generation) return;
    timer = deps.schedule(TASKS_FOLLOW_POLL_MS, () => {
      timer = null;
      void poll(token);
    });
  }

  async function poll(token: number): Promise<void> {
    try {
      const current = await deps.current();
      if (token !== generation || current === null) return;
      if (current !== deps.shown()) await deps.refresh();
    } catch {
      // a failed read or refresh never ends the follow
    } finally {
      // scheduled after completion: slow reads never overlap
      arm(token);
    }
  }

  return {
    start() {
      stop();
      armed = true;
      arm(generation);
    },
    stop,
    armed: () => armed,
  };
}
