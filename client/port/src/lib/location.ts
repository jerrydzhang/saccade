// The explicit location grammar, one parser for every input surface
// (open control, picker, inspection open): "machine:/absolute/path"
// names a machine from the local core's list - a repository's
// declaration lives at such a path - "name/t-N[@rK]" names a task
// board served by the registered project of that name, "plane" is the
// control plane (it names no tracker), and nothing else is a location.
// A bare relative path is refused, loudly, never guessed; a bare
// "/absolute/path" is no location either - the machine is always
// named.
//
// Machine names may carry colons themselves, so the separator is the
// first colon whose remainder starts with "/".

export type ParsedLocation = { machine: string; path: string } | null;

// The tracker-named form: a registered project's name, then a task
// spec or the glance's entry page (agents). The name carries no colon
// or slash, so it never collides with the machine:/path form. A
// tracker's pipeline is no location: the control plane shows it.
const TRACKER_LOCATION = /^([A-Za-z0-9][A-Za-z0-9._-]*)\/(t-\d+(?:@r\d+)?|agents)$/;

/** The control plane's address: it names no tracker. Which
 * project the plane shows is the window's own state, carried in the
 * address's path ("" - the window's default) and persisted as the
 * window place's part; the location string is always "plane". */
export const PLANE_LOCATION = "plane";
export const PLANE_MACHINE = "plane";

/** Whether an address is the control plane: a core machine that
 * happens to be named "plane" is still reached as plane:/path. */
export function isPlane(address: { machine: string; path: string }): boolean {
  return address.machine === PLANE_MACHINE && !address.path.startsWith("/");
}

/** The entry pages the instruments read from a tracker: the pipeline
 * (tasks) and the glance (agents), per t-14 "Tasks and agents". */
export type InstrumentPath = "tasks" | "agents";

export function instrumentPath(path: string): InstrumentPath | null {
  const trimmed = path.trim();
  return trimmed === "tasks" || trimmed === "agents" ? trimmed : null;
}

export function trackerLocation(input: string): ParsedLocation {
  const match = TRACKER_LOCATION.exec(input.trim());
  if (!match || match[1] === undefined || match[2] === undefined) return null;
  return { machine: `tracker:${match[1]}`, path: match[2] };
}

/** "t-N" or "t-N@rK": the task spec a tracker location's path carries. */
export type TaskSpec = { task: number; revision: number | null };

export function parseTaskSpec(path: string): TaskSpec | null {
  const match = /^(t-\d+)(?:@r(\d+))?$/.exec(path.trim());
  if (!match || match[1] === undefined) return null;
  const task = Number(match[1].slice(2));
  if (!Number.isFinite(task)) return null;
  return { task, revision: match[2] === undefined ? null : Number(match[2]) };
}

export function parseLocation(input: string): ParsedLocation {
  const raw = input.trim();
  if (raw.length === 0) return null;
  if (raw === PLANE_LOCATION) return { machine: PLANE_MACHINE, path: "" };
  const tracker = trackerLocation(raw);
  if (tracker) return tracker;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === ":" && raw[i + 1] === "/") {
      const machine = raw.slice(0, i);
      const path = raw.slice(i + 1);
      if (machine.length > 0 && !machine.includes("\0")) {
        return { machine, path };
      }
      return null;
    }
  }
  return null;
}

/** The read surface's route for a page or file location: the tracker
 * requests ride the one module's loopback ask, so a route is the
 * address's path alone, never an absolute URL. */
export function viewRoute(route: string): string {
  return `/view/v1/${route.replace(/^\//, "")}`;
}

/** The parseable string form of a location: the form the header field,
 * the picker and the inspection open all take back. */
export function formatLocation(location: { machine: string; path: string }): string {
  if (isPlane(location)) return PLANE_LOCATION;
  if (location.machine.startsWith("tracker:")) {
    return `${location.machine.slice("tracker:".length)}/${location.path}`;
  }
  return `${location.machine}:${location.path}`;
}
