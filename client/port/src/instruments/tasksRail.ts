// The task list instrument's rail publication: what sits beneath a
// registered project's rail entry - the agents its tracker serves,
// joined with the shell's word feed, their acts (the attach copy, the
// task's board opens) - and the terminals no registered project holds,
// at the rail's foot. The instrument owns this data and polls it; the
// frame draws the published targets without knowing their kind.
//
// A project whose asks failed is re-asked with exponential backoff
// (one poll interval doubled per consecutive failure, capped at a
// minute, reset on success): each ask is a program run on that
// machine, and the cap keeps a returning machine's wait bounded.

import type { TargetDefinition } from "../lib/contracts/buffer.ts";
import { theProjects } from "../lib/asks.ts";
import { sourceFor, tauriInvoke } from "../lib/sources.ts";
import { TrackerSource } from "../lib/tracker.ts";
import { formatLocation } from "../lib/location.ts";
import { agentFactsOf, railOf, railAgentPart, railAgentText, terminalPart, wordsOf, type RailAgent, type WordsSnapshot, NO_WORDS } from "../lib/tasks/rail.ts";
import type { TaskPorts } from "../lib/tasks/menu.ts";
import type { Component } from "svelte";

/** The rail poll's live cadence (a named tunable). */
export const RAIL_POLL_MS = 3000;

/** The re-ask cadence for a project whose last ask failed: one poll
 * interval doubled per consecutive failure, capped - each ask is a
 * program run on the refusing machine, and the cap keeps a returning
 * machine's wait bounded. */
const RAIL_RETRY_MS = 6000;
const RAIL_RETRY_CAP_MS = 60_000;

/** The acts on a task's board, composed spec in hand: an agent's for
 * its first task. */
type EntryActs = import("../lib/contracts/buffer.ts").TargetAct[];

function boardActs(spec: string, ports: Pick<TaskPorts, "elsewhere" | "copy">, reference = true): EntryActs {
  const acts: EntryActs = [];
  if (ports.elsewhere !== undefined) {
    const elsewhere = ports.elsewhere;
    const open = (act: "link" | "once" | "new"): { id: string; key: string; label: string; run(): void } => ({
      key: act,
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
    acts.push({
      key: "y",
      action: { id: "tasks.copy-reference", label: "copy reference", run: () => copy(spec) },
    });
  }
  return acts;
}

/** An agent's acts: y first - the attach command for its terminal,
 * the address exactly as the local core names it - so Enter and a
 * plain click copy it; then its first task's board acts, when it owns
 * one. */
function agentActs(
  target: { address: string | null; project: string | null; task: string | null },
  ports: Pick<TaskPorts, "elsewhere" | "copy">,
): EntryActs {
  const acts: EntryActs = [];
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
    acts.push(
      ...boardActs(formatLocation({ machine: `tracker:${target.project}`, path: target.task }), ports, false),
    );
  }
  return acts;
}

export interface TasksRailDeps {
  /** The window services slice the beneath-acts ride. */
  ports(): Pick<TaskPorts, "open" | "elsewhere" | "copy">;
  /** The clock: the backoff's schedule reads it. Defaults to the
   * real clock; tests inject a stepped one. */
  now?(): number;
  /** One poll, driven by the caller (tests drive the backoff by
   * hand; the loop drives it live). */
  tick?(): Promise<void>;
}

/** The tasks instrument's rail publication: per registered project,
 * the agents beneath its rail entry; the strays at the foot. The poll
 * lives here, with the backoff; the frame reads the published targets
 * through the claims seam and draws them generically. */
/** The live agent lookup behind the tasks instrument's row component:
 * the instrument points it at its own rail view (the specimen, at its
 * fixture). */
let railAgentLookup: (id: string) => RailAgent | null = () => null;
export function serveRailAgents(lookup: (id: string) => RailAgent | null): void {
  railAgentLookup = lookup;
}
/** The agent behind one published rail target's id, or null. */
export function railAgentOf(id: string): RailAgent | null {
  return railAgentLookup(id);
}

export function createTasksRail(deps: TasksRailDeps) {
  const now = (): number => (deps.now ? deps.now() : Date.now());
  const observers = new Set<() => void>();
  let words: WordsSnapshot = NO_WORDS;
  let rail: ReturnType<typeof railOf> | null = null;
  let running = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  // per project: the consecutive ask failures (the backoff's exponent)
  const failures = new Map<string, number>();
  // per project: when the next ask is due (ms epoch)
  const nextDue = new Map<string, number>();
  // per project: the last ask's failure (the unreachable note's text),
  // cleared on success
  const notes = new Map<string, string>();

  async function poll(): Promise<void> {
    let projects: { machine: string; path: string; declaration: { name: string; port: number | null } }[] = [];
    try {
      projects = await theProjects().projects();
    } catch {
      projects = [];
    }
    // one ask per project: due only (a failed project's backoff holds
    // it back; a success resets it). A project whose declaration has
    // no tracker port is never asked: nothing of its tracker is known
    // (its plain rail entry stands, no note)
    const moment = now();
    const due = projects.filter(
      (project) => project.declaration.port !== null && (nextDue.get(project.declaration.name) ?? 0) <= moment,
    );
    const pages = await Promise.all(
      due.map(async (project) => {
        try {
          const source = sourceFor(`tracker:${project.declaration.name}`);
          if (!(source instanceof TrackerSource)) {
            return { name: project.declaration.name, agents: null, repoPath: null };
          }
          const agents = agentFactsOf(await source.agentsPage());
          const repoPath = await source.identity().then(
            (identity) => identity.repoPath,
            () => null,
          );
          failures.delete(project.declaration.name);
          nextDue.delete(project.declaration.name);
          notes.delete(project.declaration.name);
          return { name: project.declaration.name, agents, repoPath };
        } catch (error) {
          const failed = (failures.get(project.declaration.name) ?? 0) + 1;
          failures.set(project.declaration.name, failed);
          nextDue.set(
            project.declaration.name,
            Math.min(moment + RAIL_RETRY_MS * 2 ** (failed - 1), moment + RAIL_RETRY_CAP_MS),
          );
          notes.set(
            project.declaration.name,
            error instanceof Error ? error.message : String(error),
          );
          return { name: project.declaration.name, agents: null, repoPath: null };
        }
      }),
    );
    rail = railOf(
      pages.map((page) => ({
        name: page.name,
        agents: page.agents,
        problem: null,
        repoPath: page.repoPath,
      })),
      words,
      null,
    );
    notify();
  }

  async function wordsLoop(): Promise<void> {
    let invoke;
    try {
      invoke = tauriInvoke();
    } catch {
      words = { ...NO_WORDS, problem: "Herdr words reach only the native app" };
      notify();
      return;
    }
    while (running) {
      try {
        const next = wordsOf(await invoke("words_wait", { since: words.generation, timeout: 25 }));
        if (!running) return;
        if (next.generation !== words.generation || next.connected !== words.connected) {
          words = next;
          await poll();
          notify();
        }
      } catch {
        await new Promise((resolve) => setTimeout(resolve, RAIL_POLL_MS));
      }
    }
  }

  async function loop(): Promise<void> {
    if (!running) return;
    await poll();
    if (!running) return;
    notify();
    timer = setTimeout(() => void loop(), RAIL_POLL_MS);
  }

  function notify(): void {
    for (const observer of [...observers]) observer();
  }

  /** The beneath-targets for one project, from the current rail: the
   * agents in importance order, their label carrying the word (the
   * frame draws no state glyphs), the attach first then the task's
   * board acts. */
  function targetsFor(project: string, ports: { elsewhere(spec: string, act: "link" | "once" | "new"): void; copy(text: string): void }): TargetDefinition[] {
    const view = rail;
    if (view === null) return [];
    const entry = view.projects.find((candidate) => candidate.name === project);
    if (entry === undefined) return [];
    const out: TargetDefinition[] = [];
    for (const group of entry.groups) {
      for (const agent of group.agents) {
        const state = group.state === "" ? null : group.state;
        const text = [state, railAgentText(agent)].filter((piece) => piece !== null).join(" - ");
        out.push({
          id: railAgentPart(project, agent),
          kind: "agent",
          label: text,
          inside: `rail/project/${project}`,
          clickable: true,
          acts: () =>
            agentActs({ address: agent.address, project, task: agent.tasks[0] ?? null }, ports),
          anchor: null,
        });
      }
    }
    return out;
  }

  /** The foot targets: the terminals no registered project holds, in
   * importance order, their label carrying the word and the
   * terminal's own title. */
  function footTargets(ports: { elsewhere(spec: string, act: "link" | "once" | "new"): void; copy(text: string): void }): TargetDefinition[] {
    const view = rail;
    if (view === null) return [];
    return view.unregistered.map((agent) => {
      const state = agent.state === "" ? null : agent.state;
      const text = [state, railAgentText(agent)].filter((piece) => piece !== null).join(" - ");
      return {
        id: terminalPart(agent.address),
        kind: "agent",
        label: text,
        inside: null,
        clickable: true,
        acts: () => agentActs({ address: agent.address, project: null, task: null }, ports),
        anchor: null,
      };
    });
  }

  return {
      /** The beneath-targets the claims seam asks for: the agents
     * beneath a project's rail entry, or the foot's strays. */
    railTargets(project: string | null): TargetDefinition[] {
      const ports = {
        open: (spec: string) => deps.ports().open(spec),
        elsewhere: (spec: string, act: "link" | "once" | "new") => deps.ports().elsewhere?.(spec, act),
        copy: (text: string) => deps.ports().copy?.(text),
      };
      return project === null ? footTargets(ports) : targetsFor(project, ports);
    },
    /** The unreachable note for a project, or null (a plain entry
     * stands). */
    railNote(project: string): string | null {
      return notes.get(project) ?? null;
    },
    /** The agent behind one published rail target's id, or null (the
     * row component draws by it). */
    railAgentFor(id: string): RailAgent | null {
      const view = rail;
      if (view === null) return null;
      for (const entry of view.projects) {
        for (const group of entry.groups) {
          for (const agent of group.agents) {
            if (railAgentPart(entry.name, agent) === id) return agent;
          }
        }
      }
      return view.unregistered.find((agent) => terminalPart(agent.address) === id) ?? null;
    },
    /** The words feed's note, or null. */
    railFootnote(): string | null {
      return words.problem ?? null;
    },
    /** The current rail view (the rows' agent marks read its words). */
    words(): WordsSnapshot {
      return words;
    },
    subscribe(observer: () => void): () => void {
      observers.add(observer);
      observer();
      return () => {
        observers.delete(observer);
      };
    },
    start(): void {
      if (running) return;
      running = true;
      // a test that injected the clock drives the poll by hand (tick);
      // the live loop and word feed spawn only under the real clock
      if (deps.now === undefined) {
        void loop();
        void wordsLoop();
      }
    },
    /** One poll, for the caller that drives the clock. */
    tick(): Promise<void> {
      return loop();
    },
    stop(): void {
      running = false;
      if (timer !== null) clearTimeout(timer);
      timer = null;
    },
  };
}
