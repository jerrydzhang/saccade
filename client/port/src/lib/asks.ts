// The app's one asks source: everything that reaches a machine goes
// through the one core module, and this is where its implementation
// is chosen. A lane may install a direct implementation before boot
// (window.__saccadeAsks - the scenarios' pages, like the shell
// mock's install); otherwise the asks ride the native shell's bridge,
// the local core. The project registry rides the same asks: the
// device keeps the registered paths and nothing else about a project,
// so every fact a rail or an open needs is read from the declaration
// file through the module.

import { coreAsks, directAsks, type MachineAsks } from "./herdr-core.ts";
import { ProjectRegistry } from "./projects.ts";

export type { CommandOutcome, LocalAddress, Machine, MachineAsks } from "./herdr-core.ts";
export type { Declaration, Project, RegisteredPath } from "./projects.ts";

type AsksWindow = { __saccadeAsks?: MachineAsks };

function asksWindow(): AsksWindow | null {
  return typeof window === "undefined" ? null : (window as unknown as AsksWindow);
}

/** The asks this window uses: the lane's install when one is present,
 * the bridge otherwise. */
export function theAsks(): MachineAsks {
  const lane = asksWindow()?.__saccadeAsks;
  if (lane) return lane;
  return coreAsks();
}

let projects: ProjectRegistry | null = null;
let projectsAsks: MachineAsks | null = null;

/** The device's project registry over this window's asks: one
 * registry per window, its facts read through the module. A lane that
 * replaces its install gets a fresh registry over the new asks. */
export function theProjects(): ProjectRegistry {
  const asks = theAsks();
  if (projects === null || projectsAsks !== asks) {
    projects = new ProjectRegistry(asks);
    projectsAsks = asks;
  }
  return projects;
}

export { directAsks };
