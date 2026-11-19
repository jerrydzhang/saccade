// The frame's rail: the projects this device has registered and, under
// each, its running agents. The rail is the frame's own surface - any
// window can show it, none starts with it - so its data lives with the
// frame: the registered projects come from the device's registry, the
// agents from the projects' agents pages joined with the shell's word
// feed, and the add field rides here. The rail knows nothing of tasks:
// a chosen project opens its task list through the window's open
// funnel, and what an instrument claims for a project is wave 2's
// seam.

import { createAddProject, type AddProject, type AddView } from "../tasks/add.ts";
import {
  agentFactsOf,
  railOf,
  railTargetsOf,
  wordsOf,
  NO_WORDS,
  type AgentFacts,
  type Rail,
  type RailTarget,
  type WordsSnapshot,
} from "../tasks/rail.ts";
import { theProjects } from "../asks.ts";
import { sourceFor, tauriInvoke } from "../sources.ts";
import { formatLocation } from "../location.ts";
import type { InputContext } from "../contracts/input.ts";
import type { TargetDefinition } from "../contracts/buffer.ts";
import type { ElsewhereAct } from "../elsewhere.ts";
import type { MenuAct } from "../menu/session.ts";
import { TrackerSource } from "../tracker.ts";
import type { TaskPorts } from "../tasks/menu.ts";

/** How often the rail re-reads its agents and its words (a named
 * tunable). */
export const RAIL_POLL_MS = 3000;

export interface FrameRailDeps {
  /** The frame's input capture: the add field's text entry and the
   * act menus ride it while they hold the keys. */
  capture(context: InputContext): { release(): void };
  /** Open a location in this window: a chosen project's task list. */
  open(spec: string): void;
  /** The elsewhere surface: an agent's o/i/w acts. */
  elsewhere(spec: string, act: ElsewhereAct): void;
  /** The clipboard, observed so the lane can see the act. */
  copy(text: string): void;
  /** The project this window shows, when it shows one: the rail's
   * mark. */
  shownProject(): string | null;
}

/** The acts on a task's board, composed spec in hand: an agent's for
 * its first task. */
function boardActs(spec: string, ports: TaskPorts, reference = true): MenuAct[] {
  const acts: MenuAct[] = [];
  if (ports.elsewhere !== undefined) {
    const elsewhere = ports.elsewhere;
    const open = (act: ElsewhereAct): { id: string; label: string; run: () => void } => ({
      id: `tasks.elsewhere.${act}`,
      label:
        act === "link"
          ? "open board (through the link, or ask once)"
          : act === "once"
            ? "open in... (ask; keep the link)"
            : "open in a new window",
      run: () => elsewhere(spec, act),
    });
    acts.push(
      { key: "o", action: open("link") },
      { key: "i", action: open("once") },
      { key: "w", action: open("new") },
    );
  }
  const copy = ports.copy;
  if (copy !== undefined && reference) {
    acts.push({ key: "y", action: { id: "tasks.copy-reference", label: "copy reference", run: () => copy(spec) } });
  }
  return acts;
}

/** An agent's acts: y first - the attach command for its terminal,
 * the address exactly as the local core names it - so Enter and a
 * plain click copy it; then its first task's board acts, when it owns
 * one. */
function agentActs(target: RailTarget, ports: TaskPorts): MenuAct[] {
  const acts: MenuAct[] = [];
  const copy = ports.copy;
  const address = target.address;
  if (copy !== undefined && address !== null) {
    acts.push({
      key: "y",
      action: {
        id: "tasks.agent.attach",
        label: `copy hc attach ${address}`,
        run: () => copy(`hc attach ${address}`),
      },
    });
  }
  if (target.project !== null && target.task !== null) {
    acts.push(...boardActs(formatLocation({ machine: `tracker:${target.project}`, path: target.task }), ports, false));
  }
  return acts;
}

export function createFrameRail(deps: FrameRailDeps) {
  const railObservers = new Set<(rail: Rail) => void>();
  let words: WordsSnapshot = NO_WORDS;
  let agentPages: {
    name: string;
    agents: AgentFacts[] | null;
    problem: string | null;
    repoPath: string | null;
  }[] = [];
  let railRunning = false;
  let railTimer: ReturnType<typeof setTimeout> | null = null;

  // projects unregistered this session: the shown one stays off the
  // rail though the window still shows it, until it is registered
  // again
  const forgotten = new Set<string>();
  let registeredNames: string[] = [];
  let pathsByName = new Map<string, { machine: string; path: string }>();

  function railCurrent(): string | null {
    const current = deps.shownProject();
    if (current === null) return null;
    if (registeredNames.includes(current)) forgotten.delete(current);
    return forgotten.has(current) ? null : current;
  }

  function railNow(): Rail {
    // every registered project is listed, read or not yet, the shown
    // one among them
    const names = new Set([...registeredNames, ...agentPages.map((entry) => entry.name)]);
    const current = railCurrent();
    if (current !== null) names.add(current);
    const projects = [...names].map(
      (name) =>
        agentPages.find((entry) => entry.name === name) ?? { name, agents: null, problem: null, repoPath: null },
    );
    return railOf(projects, words, current);
  }

  function notifyRail(): void {
    const rail = railNow();
    for (const observer of [...railObservers]) observer(rail);
  }

  async function readAgentPages(): Promise<void> {
    // the registry reads every registered declaration through the
    // module; a device with no asks has no registered projects at all
    let projects: { machine: string; path: string; declaration: { name: string } }[] = [];
    try {
      projects = await theProjects().projects();
    } catch {
      projects = [];
    }
    registeredNames = projects.map((project) => project.declaration.name);
    pathsByName = new Map(
      projects.map((project) => [project.declaration.name, { machine: project.machine, path: project.path }]),
    );
    const current = railCurrent();
    const names = [...registeredNames];
    if (current !== null && !names.includes(current)) names.push(current);
    agentPages = await Promise.all(
      names.map(async (name) => {
        try {
          const source = sourceFor(`tracker:${name}`);
          if (!(source instanceof TrackerSource)) {
            return { name, agents: null, problem: "not a tracker", repoPath: null };
          }
          const agents = agentFactsOf(await source.agentsPage());
          // the agents page read through the contract, so the root's
          // identity is in hand; a tracker that cannot say its repo
          // path still lists its agents, and places no stray
          const repoPath = await source.identity().then(
            (identity) => identity.repoPath,
            () => null,
          );
          return { name, agents, problem: null, repoPath };
        } catch (error) {
          return {
            name,
            agents: null,
            problem: error instanceof Error ? error.message : String(error),
            repoPath: null,
          };
        }
      }),
    );
  }

  /** Unregister a project: the path leaves the device's list, nothing
   * on the machine is touched. */
  function forget(project: string): void {
    const registered = pathsByName.get(project);
    if (registered !== undefined) theProjects().forget(registered);
    forgotten.add(project);
    agentPages = agentPages.filter((entry) => entry.name !== project);
    registeredNames = registeredNames.filter((name) => name !== project);
    notifyRail();
  }

  // -- the add field: registering opens nothing -----------------------
  const add: AddProject = createAddProject({
    capture: deps.capture,
    async register(registered) {
      const declaration = await theProjects().register(registered);
      forgotten.delete(declaration.name);
      if (!registeredNames.includes(declaration.name)) registeredNames = [...registeredNames, declaration.name];
      pathsByName.set(declaration.name, registered);
      notifyRail();
      // its agents read now rather than at the next poll
      if (railRunning) void readAgentPages().then(() => railRunning && notifyRail());
    },
  });

  /** The acts on one rail entry, resolved when asked: a project's
   * f/a/d, an agent's y and its first task's board acts. */
  function railEntryActs(entry: { part: string; kind: string }): MenuAct[] {
    const ports: TaskPorts = {
      open: (spec) => deps.open(spec),
      elsewhere: (spec, act) => deps.elsewhere(spec, act),
      copy: deps.copy,
    };
    if (entry.kind === "project" || entry.kind === "agent") {
      const target = railTargetsOf(railNow()).find((candidate) => candidate.part === entry.part);
      if (target === undefined) return [];
      if (target.kind === "project") {
        const project = target.project;
        if (project === null) return [];
        const acts: MenuAct[] = [
          {
            key: "f",
            action: {
              id: "rail.project.show",
              label: "show its tasks here",
              run: () => deps.open(`${project}/tasks`),
            },
          },
          {
            key: "a",
            action: {
              id: "rail.project.add",
              label: "register a project (machine:/path/to/repository)",
              run: () => add.enter(),
            },
          },
          {
            key: "d",
            action: {
              id: "rail.project.unregister",
              label: "unregister this project (register its path again to reach it)",
              run: () => forget(project),
            },
          },
        ];
        return acts;
      }
      return agentActs(target, ports);
    }
    return [];
  }

  async function agentsLoop(): Promise<void> {
    if (!railRunning) return;
    await readAgentPages();
    if (!railRunning) return;
    notifyRail();
    railTimer = setTimeout(() => void agentsLoop(), RAIL_POLL_MS);
  }

  // The shell's word feed: the shell's bounded hold wakes on a change
  // or answers quiet; outside the native shell there is no feed
  async function wordsLoop(): Promise<void> {
    let invoke;
    try {
      invoke = tauriInvoke();
    } catch {
      words = { ...NO_WORDS, problem: "Herdr words reach only the native app" };
      notifyRail();
      return;
    }
    while (railRunning) {
      try {
        const next = wordsOf(await invoke("words_wait", { since: words.generation, timeout: 25 }));
        if (!railRunning) return;
        if (next.generation !== words.generation || next.connected !== words.connected) {
          words = next;
          notifyRail();
        }
      } catch {
        await new Promise((resolve) => setTimeout(resolve, RAIL_POLL_MS));
      }
    }
  }

  return {
    /** The rail's live view, notified on every change. */
    subscribeRail(observer: (rail: Rail) => void): () => void {
      railObservers.add(observer);
      observer(railNow());
      return () => {
        railObservers.delete(observer);
      };
    },
    railNow,
    /** Re-notify: the shown project's mark moves with the window's
     * open, not with the rail's own polls. */
    refresh(): void {
      notifyRail();
    },
    /** The rail's entries as the frame's targets: the acts resolve
     * when asked, over the rail's live view. */
    targets(): TargetDefinition[] {
      return railTargetsOf(railNow()).map((entry) => ({
        id: entry.part,
        kind: entry.kind,
        label: entry.text,
        inside: entry.within ?? null,
        clickable: true,
        acts: () => railEntryActs(entry),
        anchor: null,
      }));
    },
    forget,
    /** The add field: the frame's registration act. */
    add,
    start(): void {
      if (railRunning) return;
      railRunning = true;
      void agentsLoop();
      void wordsLoop();
    },
    stop(): void {
      railRunning = false;
      if (railTimer !== null) clearTimeout(railTimer);
      railTimer = null;
      add.close();
    },
  };
}

export type FrameRail = ReturnType<typeof createFrameRail>;
export type { AddView };
