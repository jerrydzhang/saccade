// The control plane's rail (t-48), its pure half: the projects this
// device has reached and, under each, the agents that are running now,
// grouped by Herdr's word. "Running" is Herdr core's say alone: an
// incarnation from a project's agents page shows only while the local
// core's word table carries a live row for its terminal address, and
// it shows exactly the words the program sent (t-14 "Agents: the
// glance"; the human, 2026-11-14: whatever Herdr core reports, nothing
// else). Nothing here counts or advertises pending work.
//
// Agents no tracker registered (t-55, the human, 2026-11-14: stray
// agents are not stranded). The rule: a live terminal whose Herdr
// status names an app (status.app present, as the core reports it -
// a plain shell sends no status and never shows) and whose address no
// incarnation on any agents page this device read names. One whose
// cwd is inside a project shows under it, marked unregistered: the
// cwd is the project's repo path (its root page's repo_path) or under
// it, or under the project's worktree home - the tracker's state-dir
// slug (<repo basename>-<fnv1a of the repo path>,
// tracker/src/host/paths.rs) followed by /worktrees/. The deepest repo
// path wins. Any other shows in the unregistered group at the rail's
// foot, named by Herdr's `where` label (else the terminal's title)
// and its address.

import type { ViewPage } from "../pages.ts";

/** One live terminal's words, as the shell's feed serializes them
 * (client/src-tauri/src/words.rs WordRow): a field the program did not
 * send is absent. */
export type HerdrWord = {
  state?: string;
  /** The program that reported the word (status.app). */
  app?: string;
  msg?: string;
  progress?: string;
  title?: string;
  finished?: string;
  /** The terminal's working directory, as the core holds it. */
  cwd?: string;
  /** Herdr's `where` label ("saccade / general"). */
  where?: string;
  /** The terminal's own title (not the program's status title). */
  terminalTitle?: string;
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
  /** The tracker's incarnation; null for an agent no tracker
   * registered (t-55). */
  incarnation: string | null;
  /** The terminal's address as the local core names it (other/t63 on
   * a core federating that machine): the attach act's argument. */
  address: string;
  /** What names an unregistered agent: Herdr's where label, else the
   * terminal's title; null when the core sent neither (and on a
   * registered agent, which its incarnation names). */
  label: string | null;
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
  /** The home tracker: it holds this device's window store, so it is
   * never forgotten. */
  home: boolean;
  /** The host this device reached it at, when registered. */
  host: string | null;
  /** Herdr's words in t-14's section order, empty sections left out. */
  groups: { state: string; agents: RailAgent[] }[];
  /** Why this project's agents could not be read, when they could not. */
  problem: string | null;
};

export type Rail = {
  projects: RailProject[];
  /** Agents no tracker registered and no project's repo holds, at the
   * rail's foot: by word in t-14's order, then label, then address. */
  unregistered: RailAgent[];
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

/** The word for an incarnation's terminal address. The tracker records
 * the address on the agent's own machine (t81); a core that federates
 * other machines names their terminals machine-first (other/t81), so a
 * laptop's core reaching the agents' machine carries the prefix. An
 * exact key wins; else the one key that is some machine's terminal of
 * that address. Two machines holding the same address is ambiguous:
 * no word is guessed. */
export function wordFor(words: Readonly<Record<string, HerdrWord>>, address: string): HerdrWord | undefined {
  const key = wordKeyFor(words, address);
  return key === undefined ? undefined : words[key];
}

/** The word table's keys a recorded address could name: the exact key
 * alone when present, else every machine's terminal of that address. */
function candidateKeys(words: Readonly<Record<string, HerdrWord>>, address: string): string[] {
  if (words[address] !== undefined) return [address];
  const suffix = `/${address}`;
  return Object.keys(words).filter((key) => key.endsWith(suffix));
}

/** The local core's name for a recorded address (wordFor's join), or
 * undefined when none or more than one terminal could be it. */
export function wordKeyFor(words: Readonly<Record<string, HerdrWord>>, address: string): string | undefined {
  const keys = candidateKeys(words, address);
  return keys.length === 1 ? keys[0] : undefined;
}

/** FNV-1a over the path's UTF-8 bytes: the tracker's state-dir slug
 * hash (tracker/src/host/paths.rs fnv1a). */
function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** The tracker's state-dir slug for a repo path: the directory its
 * default db and its worktree home live in (state_dir_in). */
export function stateSlugOf(repoPath: string): string {
  const trimmed = repoPath.replace(/\/+$/, "");
  const basename = trimmed.slice(trimmed.lastIndexOf("/") + 1) || "repo";
  return `${basename}-${fnv1a(trimmed).toString(16).padStart(6, "0")}`;
}

/** How deep a project's repo path is when a terminal's cwd is inside
 * the project - the repo or under it, or under its worktree home -
 * so the deepest project wins; -1 when it is not. */
function placedDepth(cwd: string, repoPath: string): number {
  const repo = repoPath.replace(/\/+$/, "");
  if (repo.length === 0) return -1;
  if (cwd === repo || cwd.startsWith(`${repo}/`)) return repo.length;
  if (`${cwd}/`.includes(`/saccade/${stateSlugOf(repo)}/worktrees/`)) return repo.length;
  return -1;
}

const bySection = (a: RailAgent, b: RailAgent): number =>
  wordRank(a.state) - wordRank(b.state) || a.state.localeCompare(b.state);

const byName = (a: RailAgent, b: RailAgent): number =>
  (a.label ?? "~").localeCompare(b.label ?? "~", undefined, { numeric: true }) ||
  a.address.localeCompare(b.address, undefined, { numeric: true });

/** The rail: every project in name order, the shown one marked, each
 * with its running agents grouped by word. */
export function railOf(
  projects: readonly {
    name: string;
    agents: readonly AgentFacts[] | null;
    problem: string | null;
    /** The tracker's repository path (its root page), when read. */
    repoPath?: string | null;
  }[],
  words: WordsSnapshot,
  current: string | null,
  home: string | null = null,
  hosts: Readonly<Record<string, string>> = {},
): Rail {
  const ordered = [...projects].sort((a, b) => a.name.localeCompare(b.name));
  // the keys some registration names: its joined terminal, or - when
  // two machines hold its address - every candidate, so a registered
  // agent never doubles as a stray one
  const registered = new Set<string>();
  for (const project of projects) {
    for (const agent of project.agents ?? []) {
      if (agent.address === null) continue;
      for (const key of candidateKeys(words.words, agent.address)) registered.add(key);
    }
  }
  // the strays, each placed in the deepest project holding its cwd
  const strays = new Map<string, RailAgent[]>();
  const unplaced: RailAgent[] = [];
  for (const [address, word] of Object.entries(words.words)) {
    if (registered.has(address) || word.finished !== undefined) continue;
    if (word.app === undefined || word.app.length === 0) continue;
    const agent: RailAgent = {
      incarnation: null,
      address,
      // an empty label or title (a fresh terminal's) names nothing
      label: [word.where, word.terminalTitle].find((name) => name !== undefined && name.length > 0) ?? null,
      state: word.state ?? "",
      msg: word.msg ?? null,
      progress: word.progress ?? null,
      tasks: [],
      at: word.at,
    };
    let holder: string | null = null;
    let depth = -1;
    if (word.cwd !== undefined) {
      for (const project of projects) {
        if (project.repoPath === undefined || project.repoPath === null) continue;
        const placed = placedDepth(word.cwd, project.repoPath);
        if (placed > depth) {
          depth = placed;
          holder = project.name;
        }
      }
    }
    if (holder === null) unplaced.push(agent);
    else strays.set(holder, [...(strays.get(holder) ?? []), agent]);
  }
  return {
    herdr: words.connected ? null : (words.problem ?? "Herdr core unreachable"),
    unregistered: unplaced.sort((a, b) => bySection(a, b) || byName(a, b)),
    projects: ordered.map((project) => {
      const sections = new Map<string, RailAgent[]>();
      for (const agent of project.agents ?? []) {
        if (agent.address === null) continue;
        const key = wordKeyFor(words.words, agent.address);
        const word = key === undefined ? undefined : words.words[key];
        // a terminal that ended carries its reason and no word: it is
        // not running
        if (key === undefined || word === undefined || word.finished !== undefined) continue;
        const state = word.state ?? "";
        const list = sections.get(state) ?? [];
        list.push({
          incarnation: agent.incarnation,
          address: key,
          label: null,
          state,
          msg: word.msg ?? null,
          progress: word.progress ?? null,
          tasks: agent.tasks,
          at: word.at,
        });
        sections.set(state, list);
      }
      for (const agent of strays.get(project.name) ?? []) {
        sections.set(agent.state, [...(sections.get(agent.state) ?? []), agent]);
      }
      const groups = [...sections.entries()]
        .sort(([a], [b]) => wordRank(a) - wordRank(b) || a.localeCompare(b))
        .map(([state, agents]) => ({
          state,
          // within a section, by task, then incarnation (t-14); the
          // unregistered after the registered, by name
          agents: agents.sort(
            (a, b) =>
              Number(a.incarnation === null) - Number(b.incarnation === null) ||
              (a.tasks[0] ?? "~").localeCompare(b.tasks[0] ?? "~", undefined, { numeric: true }) ||
              (a.incarnation ?? "").localeCompare(b.incarnation ?? "", undefined, { numeric: true }) ||
              byName(a, b),
          ),
        }));
      return {
        name: project.name,
        current: project.name === current,
        home: project.name === home,
        host: hosts[project.name] ?? null,
        groups,
        problem: project.problem,
      };
    }),
  };
}

/** A rail entry as a selectable target: the same label selection, act
 * menu and plain click a task row has. A project's acts switch this
 * window to its tasks or forget it; an agent's are its first task's,
 * a task row's acts, and y copies its attach command (t-55); an
 * unregistered agent's only act is y. The part path is the rail's own
 * identity scheme (never a page path): the project's name, the
 * agent's incarnation, an unregistered agent's terminal address. */
export type RailTarget = Readonly<{
  part: string;
  kind: "project" | "agent";
  text: string;
  /** The project the entry sits under; null in the foot group. */
  project: string | null;
  /** The agent's first task; null on a project, or an agent with none. */
  task: string | null;
  /** The agent's terminal as the local core names it; null on a project. */
  address: string | null;
  home: boolean;
  current: boolean;
  host: string | null;
}>;

export const projectPart = (name: string): string => `rail/project/${name}`;
export const agentPart = (project: string, incarnation: string): string => `rail/agent/${project}/${incarnation}`;
/** An unregistered agent's part: its terminal, the one identity it has. */
export const terminalPart = (address: string): string => `rail/terminal/${address}`;

/** The part an agent entry rides. */
export const railAgentPart = (project: string | null, agent: RailAgent): string =>
  agent.incarnation === null || project === null ? terminalPart(agent.address) : agentPart(project, agent.incarnation);

/** The words an agent entry is named by. */
export function railAgentText(agent: RailAgent): string {
  if (agent.incarnation === null) return [agent.label, agent.address].filter((part) => part !== null).join(" ");
  return [agent.incarnation, agent.tasks[0]].filter((part) => part !== undefined).join(" ");
}

/** The rail's targets in its own order: each project, then its
 * agents; the unregistered group's agents last. */
export function railTargetsOf(rail: Rail): RailTarget[] {
  const out: RailTarget[] = [];
  for (const project of rail.projects) {
    out.push({
      part: projectPart(project.name),
      kind: "project",
      text: project.name,
      project: project.name,
      task: null,
      address: null,
      home: project.home,
      current: project.current,
      host: project.host,
    });
    for (const group of project.groups) {
      for (const agent of group.agents) {
        out.push({
          part: railAgentPart(project.name, agent),
          kind: "agent",
          text: railAgentText(agent),
          project: project.name,
          task: agent.tasks[0] ?? null,
          address: agent.address,
          home: project.home,
          current: project.current,
          host: project.host,
        });
      }
    }
  }
  for (const agent of rail.unregistered) {
    out.push({
      part: terminalPart(agent.address),
      kind: "agent",
      text: railAgentText(agent),
      project: null,
      task: null,
      address: agent.address,
      home: false,
      current: false,
      host: null,
    });
  }
  return out;
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
      const word: HerdrWord = { at: typeof fields.at === "number" ? fields.at : 0 };
      // only what the shell sent: an absent field stays absent
      const keep = (key: Exclude<keyof HerdrWord, "at">, wire: string): void => {
        const value = text(wire);
        if (value !== undefined) word[key] = value;
      };
      keep("state", "state");
      keep("app", "app");
      keep("msg", "msg");
      keep("progress", "progress");
      keep("title", "title");
      keep("finished", "finished");
      keep("cwd", "cwd");
      keep("where", "where");
      keep("terminalTitle", "terminal_title");
      words[address] = word;
    }
  }
  return {
    generation: typeof value.generation === "number" ? value.generation : 0,
    words,
    connected: value.connected === true,
    problem: typeof value.problem === "string" ? value.problem : undefined,
  };
}
