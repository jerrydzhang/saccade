// The picker (t-14 "The viewer"): `space space` opens it; match quality
// before recency - an id match first, then a name, a word start, a
// substring, scattered letters - recency breaking ties, the open item
// last. A result shows only its kind and name. Typed locations always
// resolve even when nothing enumerated them (t-24@r11); raw paths
// resolve on the tracker's machine or a named machine, never locally.

/** What a result row shows: only its kind and its name. */
export function pickerLabelKind(item: PickerItem): { kind: string; label: string } {
  return { kind: item.kind, label: item.name.length > 0 ? item.name : item.location };
}

export type PickerItem = {
  /** What opens: a location string in the header field's grammar. */
  location: string;
  kind: "task" | "board" | "file";
  name: string;
};

export type Rank =
  | "id"
  | "name"
  | "word-start"
  | "substring"
  | "scattered"
  | "none";

/** How a query matches a candidate's id and name, best first: an id
 * match (the id contains it - ids are short), a name that starts with
 * it, a word start, a mid-word substring, scattered letters. */
export function rankOf(query: string, item: PickerItem): Rank {
  const q = query.trim().toLowerCase();
  if (q.length === 0) return "none";
  const id = item.location.toLowerCase();
  const name = item.name.toLowerCase();
  if (id.includes(q)) return "id";
  if (name.startsWith(q)) return "name";
  // a word start: any word of the id or name begins with the query
  const words = `${id} ${name}`.split(/[^a-z0-9]+/).filter((word) => word.length > 0);
  if (words.some((word) => word.startsWith(q))) return "word-start";
  if (name.includes(q)) return "substring";
  // scattered: every query character appears, in order
  let at = 0;
  for (const character of q) {
    const found = `${id} ${name}`.indexOf(character, at);
    if (found < 0) return "none";
    at = found + 1;
  }
  return "scattered";
}

const RANK_ORDER: Record<Rank, number> = {
  id: 0,
  name: 1,
  "word-start": 2,
  substring: 3,
  scattered: 4,
  none: 5,
};

/** The ranked candidates: match quality first, recency (the item's
 * position in the window's own recent list, best first) breaking ties,
 * and the item currently open dead last. Unmatched candidates answer
 * only when the query is empty. */
export function rankedPicker(
  query: string,
  items: PickerItem[],
  recent: string[],
  openLocation: string | null,
): PickerItem[] {
  const scored = items
    .map((item) => ({ item, rank: rankOf(query, item) }))
    .filter((entry) => entry.rank !== "none" || query.trim().length === 0);
  const recencyOf = (item: PickerItem): number => {
    const at = recent.indexOf(item.location);
    return at < 0 ? Number.MAX_SAFE_INTEGER : at;
  };
  scored.sort((a, b) => {
    if (RANK_ORDER[a.rank] !== RANK_ORDER[b.rank]) return RANK_ORDER[a.rank] - RANK_ORDER[b.rank];
    const recency = recencyOf(a.item) - recencyOf(b.item);
    if (recency !== 0) return recency;
    return a.item.location.localeCompare(b.item.location);
  });
  if (openLocation !== null) {
    const openAt = scored.findIndex((entry) => entry.item.location === openLocation);
    if (openAt >= 0) {
      const [open] = scored.splice(openAt, 1);
      if (open) scored.push(open);
    }
  }
  return scored.map((entry) => entry.item);
}

// -- what the picker lists (t-34@r6 change 5) ---------------------------

/** The project a picker query is about: a query that starts with a
 * known project's name and a slash names that project; anything else is
 * about the window's own project. One project at a time - the picker
 * never lists every project's boards together. */
export function pickerProject(query: string, current: string | null, known: string[]): string | null {
  const match = /^([A-Za-z0-9][A-Za-z0-9._-]*)\//.exec(query.trim());
  if (match && match[1] !== undefined && known.includes(match[1])) return match[1];
  return current;
}

/** One project's candidates: its tasks window, its agents window and
 * every task's board, named by title. */
export function projectCandidates(project: string, tasks: Array<{ task: string; title: string }>): PickerItem[] {
  return [
    { location: `${project}/tasks`, kind: "board", name: `${project} tasks` },
    { location: `${project}/agents`, kind: "board", name: `${project} agents` },
    ...tasks.map((row) => ({ location: `${project}/${row.task}`, kind: "task" as const, name: `${row.task} ${row.title}` })),
  ];
}

/** Candidates once each, the first naming kept (the project's own list
 * comes first, so a task keeps its title). */
export function uniqueCandidates(items: PickerItem[]): PickerItem[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.location)) return false;
    seen.add(item.location);
    return true;
  });
}
