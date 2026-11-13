// The Tasks instrument's pure model: the tracker's pipeline page
// (the /view/v1 answer at the "tasks" location) restated as the rows
// the surface serves. Svelte-free and sources-free - unit-testable
// without any transport, exactly like lib/board's model layer.
//
// The wire shape is the page grammar's own (lib/pages.ts): groups
// carrying tables, rows carrying cells plus the row's location (its
// link - "t-N") and its latest-activity moment. Nothing here switches
// on the page kind; a page that carries no tables simply has no rows.
//
// Identity (scenario 12): a row's identity is its canonical part path
// (/content/{group}/content/{table}/row/{index}) - stable in the
// model, independent of geometry; a row that leaves the page leaves
// the dataset and its target ceases to exist.

import type { GroupPart, TablePart, TableRowPart, ViewPage } from "../pages.ts";

export type TaskRow = Readonly<{
  /** Canonical identity path within the tasks page. */
  part: string;
  /** The task the row names ("t-N"): the identity the return anchor
   * carries and the observation surface exposes. */
  task: string;
  /** The row's link - the location its primary act opens (the task's
   * activity), relative to the tracker that serves the list. */
  location: string;
  title: string;
  agents: string;
  commit: string;
  revision: string;
  at: number | null;
}>;

export type TaskGroup = Readonly<{
  /** Canonical identity path of the group part. */
  part: string;
  title: string;
  rows: readonly TaskRow[];
}>;

export type TasksPageModel = Readonly<{
  groups: readonly TaskGroup[];
}>;

function cell(row: TableRowPart, index: number): string {
  return row.cells[index] ?? "";
}


function taskOf(row: TableRowPart, index: number): string {
  // the wire's canonical link is the row's task ("t-N"); the first
  // cell repeats it - the link wins when they disagree
  const fromLocation = row.location ?? "";
  if (fromLocation.startsWith("t-")) return fromLocation;
  return cell(row, 0);
}

function rowsOf(table: TablePart, tablePath: string): TaskRow[] {
  const out: TaskRow[] = [];
  table.rows.forEach((row, index) => {
    const part = `${tablePath}/row/${index}`;
    const task = taskOf(row, index);
    if (task.length === 0) return;
    out.push({
      part,
      task,
      location: row.location ?? task,
      title: cell(row, 1),
      agents: cell(row, 2),
      commit: cell(row, 3),
      revision: cell(row, 4),
      at: typeof row.at === "number" ? row.at : null,
    });
  });
  return out;
}

/** The row's age (the prototype's ago, inst16 :116): minutes under
 * an hour, hours under two days, else days - from the row's latest
 * activity moment; null when the row carries none. */
export function ageOf(at: number | null, now: number): string {
  if (at === null) return "";
  const minutes = Math.max(0, Math.round((now - at) / 60_000));
  if (minutes < 60) return `${minutes}m`;
  if (minutes < 48 * 60) return `${Math.round(minutes / 60)}h`;
  return `${Math.round(minutes / (24 * 60))}d`;
}

/** The tasks page's model: every group's every table's rows, with
 * their canonical part paths. A page with no content answers an empty
 * model - never an error. */
export function tasksPageOf(page: ViewPage): TasksPageModel {
  const groups: TaskGroup[] = [];
  (page.content ?? []).forEach((part, groupIndex) => {
    if (part.kind !== "group") return;
    const group = part as GroupPart;
    const groupPath = `/content/${groupIndex}`;
    const rows: TaskRow[] = [];
    group.content.forEach((child, tableIndex) => {
      if (child.kind !== "table") return;
      const tablePath = `${groupPath}/content/${tableIndex}`;
      rows.push(...rowsOf(child as TablePart, tablePath));
    });
    groups.push({ part: groupPath, title: group.title ?? "", rows });
  });
  return { groups };
}

/** The row's narrowing/label text: the task id and the title - the
 * prototype's row label shape (data-label="t-N  title"). */
export function rowText(row: TaskRow): string {
  return `${row.task} ${row.title}`.trim();
}

// The select-session target shape (part/kind/text) plus the facts the
// acts need: the task id and the row's link.
export type TaskTarget = Readonly<{
  part: string;
  kind: "task";
  text: string;
  task: string;
  location: string;
}>;

/** The page's selectable targets: one per row, in document order. */
export function tasksTargetsOf(model: TasksPageModel): readonly TaskTarget[] {
  const out: TaskTarget[] = [];
  for (const group of model.groups) {
    for (const row of group.rows) {
      out.push({ part: row.part, kind: "task", text: rowText(row), task: row.task, location: row.location });
    }
  }
  return out;
}

/** The row a part path names, or undefined when the part names no row
 * of the shown page (scenario 12: the identity left the dataset). */
export function taskRowOf(model: TasksPageModel, part: string): TaskRow | undefined {
  for (const group of model.groups) {
    const row = group.rows.find((candidate) => candidate.part === part);
    if (row !== undefined) return row;
  }
  return undefined;
}
