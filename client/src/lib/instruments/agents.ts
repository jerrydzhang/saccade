// The agents instrument's pure half (t-14 "Tasks and agents"): the
// agents page's sessions joined with the live words by the session's
// terminal address, sectioned by Herdr's word. One view, sectioned by
// the word (error, blocked, working, idle, done); within a section
// rows order by task, then workspace. A row is the state shape, the
// incarnation, Herdr's msg and progress exactly as the program sent
// them (nothing when nothing was sent), the task, and the time. The
// proposed item is built as proposed: available workspaces with no
// agent hold no row - the view is the sessions, nothing else keeps it
// open. A word Herdr does not report is shown as absent (t-24@r19):
// those sessions ride the last section, with no shape.

import type { GroupPart, ViewPage } from "../pages.ts";
import type { WordsTable } from "../words.ts";

/** Herdr's word vocabulary the spec names, in section order. */
export const WORD_ORDER = ["error", "blocked", "working", "idle", "done"] as const;

export type AgentRow = {
  incarnation: string;
  /** The terminal address the live join rides; absent when the
   * record names none. */
  address: string | null;
  harness: string | null;
  /** The tasks this incarnation currently owns, latest claim first
   * (the page's order). The row's task is the first of them - the
   * same convention the companion's window title holds (t-24@r6). */
  tasks: string[];
  workspace: string | null;
  /** The session's last board commit, from the page. */
  lastCommit: string | null;
  /** The live word, joined by the terminal address: null when the
   * session's terminal reports nothing. */
  word: {
    state: string | null;
    app: string | null;
    msg: string | null;
    progress: string | null;
    title: string | null;
    finished: string | null;
    at: number;
  } | null;
};

export type AgentsSection = {
  /** The section's word; null is the absent-word section, last. */
  word: string | null;
  rows: AgentRow[];
};

type SessionFacts = {
  incarnation: string;
  address: string | null;
  harness: string | null;
  tasks: string[];
  workspace: string | null;
  lastCommit: string | null;
};

function sessionFacts(group: GroupPart): SessionFacts | null {
  let incarnation: string | null = null;
  const facts: SessionFacts = {
    incarnation: "",
    address: null,
    harness: null,
    tasks: [],
    workspace: null,
    lastCommit: null,
  };
  for (const part of group.content) {
    if (part.kind !== "fact") continue;
    switch (part.label) {
      case "incarnation":
        incarnation = part.value;
        facts.incarnation = part.value;
        break;
      case "terminal address":
        facts.address = part.value;
        break;
      case "harness":
        facts.harness = part.value;
        break;
      case "tasks":
        facts.tasks = part.value.split(/\s+/).filter((task) => task.length > 0);
        break;
      case "workspace":
        facts.workspace = part.value;
        break;
      case "last commit":
        facts.lastCommit = part.value;
        break;
    }
  }
  if (incarnation === null) return null;
  return facts;
}

/** The agents view: the page's session groups, joined with the words,
 * sectioned by the word. Sections appear in the spec's word order;
 * the absent-word section rides last when any session has no word.
 * Within a section rows order by task, then workspace, then
 * incarnation (a session owning no task sits by its workspace). */
export function agentsSections(page: ViewPage, words: WordsTable): AgentsSection[] {
  const rows: AgentRow[] = [];
  for (const part of page.content ?? []) {
    if (part.kind !== "group") continue;
    const facts = sessionFacts(part as GroupPart);
    if (facts === null) continue;
    const live = facts.address !== null ? (words[facts.address] ?? null) : null;
    rows.push({
      incarnation: facts.incarnation,
      address: facts.address,
      harness: facts.harness,
      tasks: facts.tasks,
      workspace: facts.workspace,
      lastCommit: facts.lastCommit,
      word:
        live === null
          ? null
          : {
              state: live.state ?? null,
              app: live.app ?? null,
              msg: live.msg ?? null,
              progress: live.progress ?? null,
              title: live.title ?? null,
              finished: live.finished ?? null,
              at: live.at,
            },
    });
  }
  const sections: AgentsSection[] = [];
  for (const word of WORD_ORDER) {
    const inSection = rows.filter((row) => row.word?.state === word);
    sections.push({ word, rows: orderRows(inSection) });
  }
  const absent = rows.filter((row) => {
    const state = row.word?.state;
    return state === null || state === undefined || !WORD_ORDER.includes(state as (typeof WORD_ORDER)[number]);
  });
  if (absent.length > 0) sections.push({ word: null, rows: orderRows(absent) });
  return sections;
}

function orderRows(rows: AgentRow[]): AgentRow[] {
  const taskKey = (row: AgentRow): string => row.tasks[0] ?? "";
  return [...rows].sort((a, b) => {
    const byTask = taskKey(a).localeCompare(taskKey(b), undefined, { numeric: true });
    if (byTask !== 0) return byTask;
    const byWorkspace = (a.workspace ?? "").localeCompare(b.workspace ?? "");
    if (byWorkspace !== 0) return byWorkspace;
    return a.incarnation.localeCompare(b.incarnation, undefined, { numeric: true });
  });
}

/** The row's task: the most recently claimed of the tasks it owns -
 * the board Enter opens through the link. */
export function rowTask(row: AgentRow): string | null {
  return row.tasks[0] ?? null;
}

/** The attach command the row copies: the human's own viewer on the
 * agent's terminal, the companion's shape (t-20). */
export function attachCommand(row: AgentRow): string | null {
  if (row.address === null) return null;
  return `hc attach ${row.address}`;
}

/** The row's Herdr detail, exactly as sent: msg and progress, joined
 * the way the glance reads them. Nothing renders when nothing was
 * sent (t-24@r19). */
export function rowDetail(row: AgentRow): string {
  const parts: string[] = [];
  if (row.word?.msg) parts.push(row.word.msg);
  if (row.word?.progress) parts.push(row.word.progress);
  return parts.join(" \u00b7 ");
}

/** The row's time: the word's last change, epoch seconds. Nothing
 * when the terminal reports nothing. */
export function rowTime(row: AgentRow): number | null {
  if (row.word === null) return null;
  return row.word.at;
}
