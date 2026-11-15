// What the floating key surfaces show (t-59): the which-key popup
// (a pending prefix's continuations) and help (`?`, every key, set in
// columns by the context the keys come from). Pure projections over
// the matcher's own views (key-matcher.ts WhichKeyView), so the
// surfaces read the same declarations the matcher runs and cannot
// drift from them. Display only: "spc" reads "space".

import type { KeyInput } from "./input-context.ts";
import type { WhichKeyView } from "./key-matcher.ts";

export type KeyRow = Readonly<{ key: string; label: string }>;

/** One column of help: the context's name and its keys. */
export type HelpGroup = Readonly<{ title: string; rows: readonly KeyRow[] }>;

/** The help surface: one group per context, the active instrument's
 * first, the window's last. */
export type HelpView = Readonly<{ groups: readonly HelpGroup[] }>;

/** The which-key popup: the pending prefix (its head) and the
 * strokes that continue it. */
export type PendingKeysView = Readonly<{ prefix: string; rows: readonly KeyRow[] }>;

function displayToken(token: string): string {
  return token === "spc" ? "space" : token;
}

/** A sequence's help label for display: "spc h" reads "space h". */
export function displayKeys(keys: string): string {
  return keys.split(" ").map(displayToken).join(" ");
}

/** A typed stroke for display, the way formatStroke spells a
 * declared one. */
export function displayStroke(stroke: KeyInput): string {
  let out = "";
  if (stroke.ctrl) out += "Ctrl-";
  if (stroke.alt) out += "Alt-";
  if (stroke.meta) out += "Meta-";
  return out + (stroke.key === " " ? "space" : stroke.key);
}

/** The pending prefix's popup: each row is the NEXT stroke (in the
 * declared spelling, case kept) and the action it leads to; strokes
 * shared by several sequences show once, first declared first. */
export function pendingKeys(view: WhichKeyView): PendingKeysView {
  const depth = view.prefix.length;
  const rows = new Map<string, KeyRow>();
  for (const entry of view.entries) {
    const next = entry.keys.split(" ")[depth];
    if (next === undefined) continue;
    const key = displayToken(next);
    if (!rows.has(key)) rows.set(key, { key, label: entry.label });
  }
  return { prefix: view.prefix.map(displayStroke).join(" "), rows: [...rows.values()] };
}

/** Help's columns: each context's whole table, an action's
 * alternative keys on one row ("d, Ctrl-d" for one label); a context
 * with no keys shows no column. */
export function helpGroups(
  contexts: readonly Readonly<{ title: string; view: WhichKeyView }>[],
): HelpView {
  const groups: HelpGroup[] = [];
  for (const { title, view } of contexts) {
    const rows = new Map<string, string[]>();
    for (const entry of view.entries) {
      const keys = rows.get(entry.label) ?? [];
      const shown = displayKeys(entry.keys);
      if (!keys.includes(shown)) keys.push(shown);
      rows.set(entry.label, keys);
    }
    if (rows.size === 0) continue;
    groups.push({
      title,
      rows: [...rows].map(([label, keys]) => ({ key: keys.join(", "), label })),
    });
  }
  return { groups };
}
