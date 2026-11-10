// The live words store: the frontend half of the app's ONE Herdr
// event feed. The native shell holds the subscription - one per
// process by this client's own design (t-24@r23: the core allows
// more; sharing one where simpler is what the client holds) - and
// keeps the table; windows read it
// through the bounded hold (`words_wait`, the wait_look_config
// pattern: the loop parks, wakes on a landing, re-enters - never a
// timer). A window with no host (the browser lane) has no feed; the
// inspection setter injects words for the lanes, test-only machinery
// named as such, never human functionality.

export type WordRow = {
  state?: string;
  app?: string;
  msg?: string;
  progress?: string;
  title?: string;
  finished?: string;
  at: number;
};

export type WordsTable = Record<string, WordRow>;

let table: WordsTable = {};
// whether the shell's feed holds a live Herdr connection; null where
// there is no shell (the browser) - nothing to report there
let connected: boolean | null = null;
let problem: string | null = null;
export type WordsConnection = { connected: boolean | null; problem: string | null };
const connectionListeners = new Set<(state: WordsConnection) => void>();

/** Follow the feed's connection: the strip says "Herdr unreachable"
 * when it is down (t-34@r6 change 6). */
export function onWordsConnection(listener: (state: WordsConnection) => void): () => void {
  connectionListeners.add(listener);
  listener({ connected, problem });
  return () => connectionListeners.delete(listener);
}
let generation = 0;
let following = false;
let started = false;

const listeners = new Set<(words: WordsTable) => void>();

function publish(next: WordsTable): void {
  table = next;
  for (const listener of listeners) listener(table);
}

/** The live words, terminal address -> row. The instruments join the
 * record's sessions to these by the session's terminal address. */
export function words(): WordsTable {
  return table;
}

export function wordsGeneration(): number {
  return generation;
}

export function onWords(listener: (words: WordsTable) => void): () => void {
  listeners.add(listener);
  listener(table);
  return () => listeners.delete(listener);
}

async function followLoop(): Promise<void> {
  const { tauriInvoke } = await import("./sources.ts");
  for (;;) {
    let invoke;
    try {
      invoke = tauriInvoke();
    } catch {
      // no host: the browser lane; the store stays whatever the
      // inspection setter injected, and the loop ends
      return;
    }
    try {
      const snapshot = (await invoke("words_wait", { since: generation })) as {
        generation: number;
        words: WordsTable;
        connected?: boolean;
        problem?: string;
      };
      const now = snapshot.connected ?? null;
      const why = snapshot.problem ?? null;
      if (now !== connected || why !== problem) {
        connected = now;
        problem = why;
        for (const listener of connectionListeners) listener({ connected, problem });
      }
      if (snapshot.generation > generation) {
        generation = snapshot.generation;
        publish(snapshot.words);
      } else {
        // quiet: the hold answered with the table as it stands; the
        // loop re-enters the hold without touching the store
        generation = snapshot.generation;
      }
    } catch {
      // the bridge failed: back off once, then re-enter; the feed
      // reconnects on its own side
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
}

/** Start following the feed where a host exists; resolves false (and
 * leaves the store to the inspection setter) in a window with no
 * host. Runs once; the loop lives for the window's life. */
export async function startWords(): Promise<boolean> {
  if (started) return following;
  started = true;
  try {
    const { tauriInvoke } = await import("./sources.ts");
    tauriInvoke(); // refuses outside the native shell
    following = true;
  } catch {
    following = false;
    return false;
  }
  void followLoop();
  return true;
}

/** The lanes' test-only setter: words injected as if the feed had
 * landed them. Named as inspection machinery, never human
 * functionality. */
export function setWords(next: WordsTable, atGeneration?: number): void {
  generation = atGeneration ?? generation + 1;
  publish(next);
}
