// The picker's model (F1, order of work step 3): what the picker
// lists and in what order, transcribed from the prototype's own
// rules (inst16.html:410-466, the m-490 spec) as pure functions.
// The component composes over this; the tests pin every tier.
//
// The scope rule (t-34@r6 change 5): the entries come from the
// window's ONE tracker - never every tracker merged.

/** One pickable entry: a task board, or a file. */
export interface PickEntry {
  readonly kind: "board" | "file";
  /** The task's id (t-N) or the file's path. */
  readonly id: string;
  /** The task's title; files carry theirs or an empty string. */
  readonly name: string;
  /** The full location spec the entry opens. */
  readonly spec: string;
  /** The entry's recency: its index in the recent list (0 = newest),
   * or null when it was never opened. */
  readonly recentIndex: number | null;
}

export interface PickContext {
  /** The window's recent opens, newest first (specs). */
  readonly recent: readonly string[];
  /** The tracker's tasks (id + title), from the one tracker. */
  readonly tasks: readonly { id: string; title: string }[];
  /** The spec composing an entry's open: tracker base + id. */
  readonly specOf: (id: string) => string;
  /** The location open in THIS window, if any (sorts last on ties). */
  readonly openHere: string | null;
}

/** The scored list for a query: every task and file, ranked by the
 * prototype's tiers; an empty query lists EVERYTHING (the list is
 * the discovery). */
export function pickList(query: string, context: PickContext): PickEntry[] {
  const entries = candidateEntries(context);
  const q = query.trim().toLowerCase();
  if (q.length === 0) return order(entries, context, (entry) => 1000);
  const scored: { entry: PickEntry; score: number }[] = [];
  for (const entry of entries) {
    const score = scoreOf(q, entry);
    if (score <= 0) continue;
    scored.push({ entry, score });
  }
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return tieBreak(a.entry, b.entry, context);
  });
  return scored.map((pair) => pair.entry);
}

function candidateEntries(context: PickContext): PickEntry[] {
  // dedup by TASK identity: a recent open pinned at a revision and
  // the task's plain entry are ONE entry (the recent form wins -
  // its spec reopens where the human last was)
  const seen = new Set<string>();
  const entries: PickEntry[] = [];
  for (const spec of context.recent) {
    const id = idOfSpec(spec);
    if (seen.has(id)) continue;
    seen.add(id);
    const task = context.tasks.find((entry) => entry.id === id);
    entries.push({
      kind: "board",
      id: id,
      name: task?.title ?? "",
      spec,
      recentIndex: entries.length,
    });
  }
  // then every task not already listed (kind board, id t-N, the
  // title as its name)
  for (const task of context.tasks) {
    if (seen.has(task.id)) continue;
    seen.add(task.id);
    entries.push({
      kind: "board",
      id: task.id,
      name: task.title,
      spec: context.specOf(task.id),
      recentIndex: null,
    });
  }
  return entries;
}

/** The prototype's tiers (inst16.html:441-447, m-490 verbatim):
 * an all-digits query equal to the task number 1000; exact id 950;
 * id starts with the query 800; word start in id+name 600;
 * substring of id+name 450; in the path 300; scattered letters (a
 * subsequence) 100; no match drops the entry. */
export function scoreOf(query: string, entry: PickEntry): number {
  const id = entry.id.toLowerCase();
  const name = entry.name.toLowerCase();
  const digits = /^\d+$/.test(query);
  const taskNumber = /^t-(\d+)$/.exec(id)?.[1] ?? null;
  if (digits && taskNumber === query) return 1000;
  if (query === id) return 950;
  if (id.startsWith(query)) return 800;
  if (wordStarts(query, `${id} ${name}`)) return 600;
  if (`${id} ${name}`.includes(query)) return 450;
  if (entry.spec.toLowerCase().includes(query)) return 300;
  if (isSubsequence(query, `${id} ${name}`)) return 100;
  return 0;
}

function wordStarts(query: string, text: string): boolean {
  return text.split(/[\s/_-]+/).some((word) => word.startsWith(query));
}

function isSubsequence(query: string, text: string): boolean {
  let at = 0;
  for (const character of query) {
    at = text.indexOf(character, at);
    if (at < 0) return false;
    at += 1;
  }
  return true;
}

function order(
  entries: PickEntry[],
  context: PickContext,
  score: (entry: PickEntry) => number,
): PickEntry[] {
  return [...entries].sort((a, b) => {
    const difference = score(b) - score(a);
    if (difference !== 0) return difference;
    return tieBreak(a, b, context);
  });
}

/** Ties (m-490): the location already open here sorts last, then by
 * recency index (null - never opened - after every opened one). */
function tieBreak(a: PickEntry, b: PickEntry, context: PickContext): number {
  const aHere = context.openHere !== null && a.spec === context.openHere ? 1 : 0;
  const bHere = context.openHere !== null && b.spec === context.openHere ? 1 : 0;
  if (aHere !== bHere) return aHere - bHere;
  const aRecent = a.recentIndex ?? Number.MAX_SAFE_INTEGER;
  const bRecent = b.recentIndex ?? Number.MAX_SAFE_INTEGER;
  return aRecent - bRecent;
}

function idOfSpec(spec: string): string {
  // the trailing t-N of a tracker spec (the entry's id for boards)
  const match = /(^|\/)(t-\d+)(@r\d+)?(:[^/]*)?$/.exec(spec);
  return match?.[2] ?? spec;
}

/** Path mode (m-490): a query starting with '/', '~' or a
 * 'machine:'/'machine:/' prefix lists that machine's files by
 * prefix instead of scoring tasks. */
export function isPathMode(query: string): boolean {
  return /^\//.test(query) || /^~/.test(query) || /^[^/\s]+:\//.test(query);
}
