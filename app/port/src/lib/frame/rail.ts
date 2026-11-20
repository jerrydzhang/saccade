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
import { canonicalAddress } from "../instruments/registry.ts";
import { parseLocation } from "../location.ts";
import { formatRegisteredPath, type Declaration } from "../projects.ts";
import { deviceViews, forgetDeviceView, keepDeviceView } from "../views.ts";
import { readView, resolveViews, type RailView } from "./views.ts";
import { registeredInstruments } from "../instruments/registry.ts";
import type { InputContext } from "../contracts/input.ts";
import type { RailPublication, TargetDefinition } from "../contracts/buffer.ts";

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

/** A kept view as the pane draws it: the entry's rail id, the words,
 * and whether this window shows exactly that location. */
export type RailViewEntry = RailView & { shown: boolean };

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
  /** The canonical location this window shows, when it shows one. */
  shownLocation(): string | null;
  /** Open a location through the elsewhere acts (the window's own
   * link, ask or new window). */
  elsewhere(spec: string, act: "link" | "once" | "new"): void;
  /** Say something on the window's strip note. */
  say(text: string): void;
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
  let declarationsByName = new Map<string, Declaration>();
  // what the pane was last told is shown: the mark moves when the
  // window's location does, not only on the poll
  let notifiedShown: string | null = null;

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
    notifiedShown = deps.shownLocation();
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
      declarationsByName = new Map(projects.map((project) => [project.declaration.name, project.declaration]));
    } catch {
      registeredNames = [];
      pathsByName = new Map();
      declarationsByName = new Map();
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

  // -- the kept views: the file's and this device's ------------------
  function viewsOf(name: string): { views: RailView[]; problems: string[] } {
    const declaration = declarationsByName.get(name);
    const registered = pathsByName.get(name);
    if (declaration === undefined || registered === undefined) return { views: [], problems: [] };
    const resolved = resolveViews(
      name,
      declaration.views,
      deviceViews(formatRegisteredPath(registered)),
    );
    if (declaration.viewsProblem !== null) {
      resolved.problems.unshift(`workbench.toml views cannot be read: ${declaration.viewsProblem}`);
    }
    return resolved;
  }

  /** A project's views as the pane draws them. */
  function railViews(name: string): { entries: RailViewEntry[]; notes: string[] } {
    const shown = deps.shownLocation();
    const { views, problems } = viewsOf(name);
    return {
      entries: views.map((view) => ({ ...view, shown: shown === view.location })),
      notes: problems,
    };
  }

  function viewActs(name: string, view: RailView): import("../contracts/buffer.ts").TargetAct[] {
    const elsewhere = (act: "link" | "once" | "new", label: string): import("../contracts/buffer.ts").TargetAct => ({
      key: act === "link" ? "o" : act === "once" ? "i" : "w",
      action: { id: `rail.view.elsewhere.${act}`, label, run: () => deps.elsewhere(view.location, act) },
    });
    const acts: import("../contracts/buffer.ts").TargetAct[] = [
      {
        key: "f",
        action: { id: "rail.view.show", label: `show ${view.location} here`, run: () => deps.open(view.location) },
      },
      elsewhere("link", "open (through the link, or ask once)"),
      elsewhere("once", "open in... (ask; keep the link)"),
      elsewhere("new", "open in a new window"),
    ];
    // a view the file offers is the file's to drop; only a view this
    // device kept can be forgotten here
    if (view.source === "device") {
      acts.push({
        key: "d",
        action: {
          id: "rail.view.forget",
          label: "forget this view (the location itself is untouched)",
          run: () => forgetView(name, view),
        },
      });
    }
    return acts;
  }

  function forgetView(name: string, view: RailView): void {
    const registered = pathsByName.get(name);
    if (registered === undefined) return;
    forgetDeviceView(formatRegisteredPath(registered), view.path);
    notifyRail();
  }

  /** Space k: keep the location this window shows beneath its
   * repository. The strip note says why when nothing is kept. */
  function keepShown(): void {
    const asked = deps.shownLocation();
    if (asked === null) {
      deps.say("no view to keep: this window shows no location");
      return;
    }
    const address = parseLocation(asked);
    const name = address !== null && address.machine.startsWith("tracker:") ? address.machine.slice("tracker:".length) : null;
    const registered = name === null ? undefined : pathsByName.get(name);
    if (address === null || name === null || registered === undefined) {
      deps.say(`cannot keep ${asked}: it belongs to no registered repository`);
      return;
    }
    const canonical = canonicalAddress(address);
    if (!canonical.ok) {
      deps.say(`cannot keep ${asked}: ${canonical.refusal}`);
      return;
    }
    const verdict = readView(name!, canonical.address.path);
    if (!verdict.ok) {
      deps.say(`cannot keep ${asked}: ${verdict.refusal}`);
      return;
    }
    if (viewsOf(name!).views.some((view) => view.location === verdict.location)) {
      deps.say(`already kept: ${verdict.location}`);
      return;
    }
    keepDeviceView(formatRegisteredPath(registered), verdict.path);
    notifyRail();
    deps.say(`kept ${verdict.location} beneath ${name}`);
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
    for (const project of projects) {
      for (const view of viewsOf(project.name).views) {
        out.push({
          id: view.part,
          kind: "view",
          label: view.source === "file" ? `${view.location} (offered by workbench.toml)` : view.location,
          inside: project.part,
          clickable: true,
          acts: () => viewActs(project.name, view),
          anchor: null,
        });
      }
    }
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
    /** The project's kept views, as the pane draws them. */
    railViews,
    /** Space k. */
    keepShown,
    /** The pane's mark follows the window: re-notify when the shown
     * location moved since the pane was last told. */
    refreshIfShownMoved(): void {
      if (deps.shownLocation() !== notifiedShown) notifyRail();
    },
    /** The add field: the frame's registration act. */
    add,

    /** The rail's targets: the project headers the frame owns, then
     * whatever the claims seam publishes beneath each, then the foot
     * group. */
    targets(): TargetDefinition[] {
      return targets();
    },
    /** The targets beneath one project's rail entry: the claims
     * seam's publications for it (the header is the frame's own), or
     * the foot group's when the project is null. Each publication
     * pairs the target with its instrument's row component, if it
     * handed one over the registration; the frame hosts it. */
    railTargets(project: string | null): RailPublication[] {
      if (project === null) {
        const foot: RailPublication[] = [];
        for (const registration of registeredInstruments()) {
          for (const target of registration.railTargets(null)) {
            foot.push({ target, row: registration.railRow ?? null });
          }
        }
        return foot;
      }
      const out: RailPublication[] = [];
      for (const registration of registeredInstruments()) {
        for (const target of registration.railTargets(project)) {
          out.push({ target, row: registration.railRow ?? null });
        }
      }
      return out;
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
