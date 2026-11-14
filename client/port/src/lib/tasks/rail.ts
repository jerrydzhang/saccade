// The control plane's rail (t-48), its pure half: the projects this
// device has reached and, under each, the agents that are running now,
// grouped by Herdr's word. "Running" is Herdr core's say alone: an
// incarnation from a project's agents page shows only while the local
// core's word table carries a live row for its terminal address, and
// it shows exactly the words the program sent (t-14 "Agents: the
// glance"; the human, 2026-11-14: whatever Herdr core reports, nothing
// else). Terminals that are no project's agent never appear. Nothing
// here counts or advertises pending work.

import type { ViewPage } from "../pages.ts";

/** One live terminal's words, as the shell's feed serializes them
 * (client/src-tauri/src/words.rs WordRow): a field the program did not
 * send is absent. */
export type HerdrWord = {
  state?: string;
  msg?: string;
  progress?: string;
  title?: string;
  finished?: string;
  at: number;
};

export type WordsSnapshot = {
  generation: number;
  words: Record<string, HerdrWord>;
  connected: boolean;
  problem?: string;
};

export const NO_WORDS: WordsSnapshot = { generation: 0, words: {}, connected: false };

/** One incarnation as a project's agents page records it. */
export type AgentFacts = { incarnation: string; address: string | null; tasks: string[] };

export type RailAgent = {
  incarnation: string;
  state: string;
  msg: string | null;
  progress: string | null;
  /** The tasks it owns, in the record's order: Enter opens the first. */
  tasks: string[];
  at: number;
};

export type RailProject = {
  name: string;
  current: boolean;
  /** Herdr's words in t-14's section order, empty sections left out. */
  groups: { state: string; agents: RailAgent[] }[];
  /** Why this project's agents could not be read, when they could not. */
  problem: string | null;
};

export type Rail = {
  projects: RailProject[];
  /** Herdr core unreachable: why, so the human can fix it. */
  herdr: string | null;
};

/** t-14's sections: error, blocked, working, idle, done; any other
 * word the core sends follows them, alphabetically. */
export const WORD_ORDER = ["error", "blocked", "working", "idle", "done"] as const;

function wordRank(state: string): number {
  const index = (WORD_ORDER as readonly string[]).indexOf(state);
  return index === -1 ? WORD_ORDER.length : index;
}

/** The incarnations an agents page lists: one group per incarnation,
 * its facts by label (tracker/src/serve/pages.rs agents_page). */
export function agentFactsOf(page: ViewPage): AgentFacts[] {
  const out: AgentFacts[] = [];
  for (const part of page.content ?? []) {
    if (part.kind !== "group") continue;
    const fact = (label: string): string | null => {
      for (const inner of part.content) {
        if (inner.kind === "fact" && inner.label === label) return inner.value;
      }
      return null;
    };
    const incarnation = fact("incarnation") ?? part.location ?? null;
    if (incarnation === null) continue;
    const tasks = fact("tasks");
    out.push({
      incarnation,
      address: fact("terminal address"),
      tasks: tasks === null ? [] : tasks.split(/\s+/).filter((task) => task.length > 0),
    });
  }
  return out;
}

/** The rail: every project in name order, the shown one marked, each
 * with its running agents grouped by word. */
export function railOf(
  projects: readonly { name: string; agents: readonly AgentFacts[] | null; problem: string | null }[],
  words: WordsSnapshot,
  current: string | null,
): Rail {
  const ordered = [...projects].sort((a, b) => a.name.localeCompare(b.name));
  return {
    herdr: words.connected ? null : (words.problem ?? "Herdr core unreachable"),
    projects: ordered.map((project) => {
      const sections = new Map<string, RailAgent[]>();
      for (const agent of project.agents ?? []) {
        if (agent.address === null) continue;
        const word = words.words[agent.address];
        // a terminal that ended carries its reason and no word: it is
        // not running
        if (word === undefined || word.finished !== undefined) continue;
        const state = word.state ?? "";
        const list = sections.get(state) ?? [];
        list.push({
          incarnation: agent.incarnation,
          state,
          msg: word.msg ?? null,
          progress: word.progress ?? null,
          tasks: agent.tasks,
          at: word.at,
        });
        sections.set(state, list);
      }
      const groups = [...sections.entries()]
        .sort(([a], [b]) => wordRank(a) - wordRank(b) || a.localeCompare(b))
        .map(([state, agents]) => ({
          state,
          // within a section, by task, then incarnation (t-14)
          agents: agents.sort(
            (a, b) =>
              (a.tasks[0] ?? "~").localeCompare(b.tasks[0] ?? "~", undefined, { numeric: true }) ||
              a.incarnation.localeCompare(b.incarnation, undefined, { numeric: true }),
          ),
        }));
      return { name: project.name, current: project.name === current, groups, problem: project.problem };
    }),
  };
}

/** Decode the shell's words snapshot; anything malformed reads as no
 * connection rather than a guess. */
export function wordsOf(raw: unknown): WordsSnapshot {
  if (typeof raw !== "object" || raw === null) return NO_WORDS;
  const value = raw as Record<string, unknown>;
  const words: Record<string, HerdrWord> = {};
  if (typeof value.words === "object" && value.words !== null) {
    for (const [address, row] of Object.entries(value.words as Record<string, unknown>)) {
      if (typeof row !== "object" || row === null) continue;
      const fields = row as Record<string, unknown>;
      const text = (key: string) => (typeof fields[key] === "string" ? (fields[key] as string) : undefined);
      words[address] = {
        state: text("state"),
        msg: text("msg"),
        progress: text("progress"),
        title: text("title"),
        finished: text("finished"),
        at: typeof fields.at === "number" ? fields.at : 0,
      };
    }
  }
  return {
    generation: typeof value.generation === "number" ? value.generation : 0,
    words,
    connected: value.connected === true,
    problem: typeof value.problem === "string" ? value.problem : undefined,
  };
}
