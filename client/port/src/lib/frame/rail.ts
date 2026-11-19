// The frame's rail, the frame's own: the projects this device has
// registered - names and the shown mark - the add field, register and
// unregister, show and hide, the scroll region and the cramped rule.
// What sits beneath a project is not the frame's: the rail asks the
// claims seam (the registered instruments) for the targets published
// beneath each project and draws them without knowing their kind; a
// project whose instruments publish nothing beneath it is a plain
// entry.

import { createAddProject, type AddProject, type AddView } from "./add.ts";
import { theProjects } from "../asks.ts";
import { projectViewFor } from "../instruments/claims.ts";
import { registeredInstruments } from "../instruments/registry.ts";
import type { InputContext } from "../contracts/input.ts";
import type { TargetDefinition } from "../contracts/buffer.ts";

/** The frame rail's project entry: the registry's name and the shown
 * mark. What sits beneath it comes from the claims seam, not from
 * here. */
export type FrameRailProject = {
  name: string;
  current: boolean;
  /** The project's rail id: the header the beneath-targets nest
   * inside. */
  part: string;
};

export const projectPart = (name: string): string => `rail/project/${name}`;

export interface FrameRailDeps {
  /** The frame's input capture: the add field's text entry rides it
   * while it holds the keys. */
  capture(context: InputContext): { release(): void };
  /** Open a location in this window: a project's claimed view. */
  open(spec: string): void;
  /** The project this window shows, when it shows one: the rail's
   * mark. */
  shownProject(): string | null;
}

/** The seam's publications: beneath each project's header, then the
 * foot group's projectless targets. */
function targetsBeneathAndFoot(projects: FrameRailProject[]): TargetDefinition[] {
  const out: TargetDefinition[] = [];
  for (const registration of registeredInstruments()) {
    for (const project of projects) {
      for (const target of registration.railTargets(project.name)) {
        out.push(target);
      }
    }
    for (const target of registration.railTargets(null)) {
      out.push(target);
    }
  }
  return out;
}

export function createFrameRail(deps: FrameRailDeps) {
  const railObservers = new Set<(projects: FrameRailProject[]) => void>();
  let railTimer: ReturnType<typeof setInterval> | null = null;

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

  function projectsNow(): FrameRailProject[] {
    // the registry's names, in name order; the shown one marked
    const names = [...registeredNames].sort((a, b) => a.localeCompare(b));
    const current = railCurrent();
    if (current !== null && !names.includes(current)) names.push(current);
    return names.map((name) => ({ name, current: name === current, part: projectPart(name) }));
  }

  function notifyRail(): void {
    const projects = projectsNow();
    for (const observer of [...railObservers]) observer(projects);
  }

  /** The registry's names and addresses, read through the module: a
   * device with no asks has no registered projects at all. */
  async function readRegisteredNames(): Promise<void> {
    try {
      const projects = await theProjects().projects();
      registeredNames = projects.map((project) => project.declaration.name);
      pathsByName = new Map(
        projects.map((project) => [project.declaration.name, { machine: project.machine, path: project.path }]),
      );
    } catch {
      registeredNames = [];
      pathsByName = new Map();
    }
    notifyRail();
  }

  /** Unregister a project: the path leaves the device's list, nothing
   * on the machine is touched. */
  function forget(project: string): void {
    const registered = pathsByName.get(project);
    if (registered !== undefined) theProjects().forget(registered);
    forgotten.add(project);
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
    },
  });

  /** The rail's targets: the project headers the frame owns, then
   * whatever the claims seam publishes beneath each, then the foot
   * group (the seam's projectless targets). The frame draws them
   * without knowing their kind. */
  function targets(): TargetDefinition[] {
    const projects = projectsNow();
    const out: TargetDefinition[] = projects.map((project) => ({
      id: project.part,
      kind: "project",
      label: project.name,
      inside: null,
      clickable: true,
      acts: () => projectActs(project),
      anchor: null,
    }));
    out.push(...targetsBeneathAndFoot(projects));
    return out;
  }

  /** A project's own header acts: its claimed view first (the
   * instrument's claim names the location), then register and
   * unregister. */
  function projectActs(project: FrameRailProject): import("../contracts/buffer.ts").TargetAct[] {
    const acts: import("../contracts/buffer.ts").TargetAct[] = [];
    const view = projectViewFor(project.name);
    if (view !== null) {
      acts.push({
        key: "f",
        action: { id: "rail.project.show", label: `show ${view} here`, run: () => deps.open(view) },
      });
    }
    acts.push({
      key: "a",
      action: {
        id: "rail.project.add",
        label: "register a project (machine:/path/to/repository)",
        run: () => add.enter(),
      },
    });
    acts.push({
      key: "d",
      action: {
        id: "rail.project.unregister",
        label: "unregister this project (register its path again to reach it)",
        run: () => forget(project.name),
      },
    });
    return acts;
  }

  return {
    /** The rail's live view, notified on every change. */
    subscribeRail(observer: (projects: FrameRailProject[]) => void): () => void {
      railObservers.add(observer);
      observer(projectsNow());
      return () => {
        railObservers.delete(observer);
      };
    },
    projectsNow,
    /** Re-notify: the shown project's mark moves with the window's
     * open, not with the rail's own polls. */
    refresh(): void {
      notifyRail();
    },
    forget,
    /** The add field: the frame's registration act. */
    add,
    /** Whether any project is registered (the empty window's third
     * line). */
    registered(): boolean {
      return registeredNames.length > 0;
    },
    /** The rail's targets: the project headers the frame owns, then
     * whatever the claims seam publishes beneath each, then the foot
     * group. */
    targets(): TargetDefinition[] {
      return targets();
    },
    /** The targets beneath one project's rail entry (its header and
     * the seam's publications), or the foot group's when the project
     * is null. */
    railTargets(project: string | null): TargetDefinition[] {
      if (project === null) {
        const foot: TargetDefinition[] = [];
        for (const registration of registeredInstruments()) {
          for (const target of registration.railTargets(null)) {
            foot.push(target);
          }
        }
        return foot;
      }
      const entry = projectsNow().find((candidate) => candidate.name === project);
      if (entry === undefined) return [];
      return [
        {
          id: entry.part,
          kind: "project",
          label: entry.name,
          inside: null,
          clickable: true,
          acts: () => projectActs(entry),
          anchor: null,
        },
        ...registeredInstruments().flatMap((registration) =>
          registration.railTargets(project),
        ),
      ];
    },
    start(): void {
      if (railTimer !== null) return;
      void readRegisteredNames();
      // the registry's names re-read at a calm cadence: a store
      // changed outside this window (another window's registration)
      // still lands here; the reads ride the registry's session
      // cache, no wire
      railTimer = setInterval(() => void readRegisteredNames(), 3000);
    },
    stop(): void {
      if (railTimer !== null) clearInterval(railTimer);
      railTimer = null;
      add.close();
    },
  };
}

export type FrameRail = ReturnType<typeof createFrameRail>;
export type { AddView };
