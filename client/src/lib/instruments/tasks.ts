// The tasks instrument's pure half (t-14 "Tasks and agents"): the
// pipeline page decoded into sections and rows, and the age the row
// renders. The page is the record's shape - sections are titled
// groups each holding one table - so the model only restates it and
// computes the age against a now the caller hands in. The agents'
// state shapes are the client's live join, never page data: the
// instrument joins the row's agent ids against the agents page and
// the words store at render time.

import type { GroupPart, TablePart, ViewPage, ViewPart } from "../pages.ts";

export type TasksRow = {
  task: string;
  title: string;
  agents: string;
  latestCommit: string;
  /** The task's birth moment, epoch seconds; null when the row
   * carries none. */
  at: number | null;
  /** The page location the row's primary act follows - the board,
   * through the link (t-14: tasks and agents have no here). */
  location: string | null;
};

export type TasksSection = {
  title: string;
  rows: TasksRow[];
};

/** The pipeline sections in the page's order: one titled group per
 * recorded state, each holding one table. */
export function tasksSections(page: ViewPage): TasksSection[] {
  const sections: TasksSection[] = [];
  for (const part of page.content ?? []) {
    if (part.kind !== "group") continue;
    const group = part as GroupPart;
    const table = group.content.find((child): child is TablePart => child.kind === "table");
    if (!table) continue;
    sections.push({ title: group.title ?? "", rows: tasksRows(table) });
  }
  return sections;
}

function tasksRows(table: TablePart): TasksRow[] {
  return table.rows.map((row) => ({
    task: row.cells[0] ?? "",
    title: row.cells[1] ?? "",
    agents: row.cells[2] ?? "",
    latestCommit: row.cells[3] ?? "",
    at: row.at ?? null,
    location: row.location ?? null,
  }));
}

/** The age a row renders, against now: the ledger's terse form. Null
 * renders as the empty cell - the page sent no moment. */
export function taskAge(at: number | null, now: number): string {
  if (at === null || now < at) return "";
  const seconds = Math.max(0, now - at);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  if (seconds < 60) return "now";
  if (minutes < 60) return `${minutes}m`;
  if (days > 0) return `${days}d ${hours % 24}h`;
  return `${hours}h ${minutes % 60}m`;
}

/** The agent ids a row names, in order: the page sends them
 * space-joined. */
export function rowAgentIds(agents: string): string[] {
  return agents.split(/\s+/).filter((id) => id.length > 0);
}

/** The table part of one section, for tests and the type-checked
 * walk. */
export function sectionTables(parts: ViewPart[]): TablePart[] {
  return parts.filter((part): part is TablePart => part.kind === "table");
}
