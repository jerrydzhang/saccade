// The explicit location grammar, one parser for every input surface
// (open control, picker, inspection open): "machine:/absolute/path"
// names a machine from the local core's list, a bare "/absolute/path"
// is a file on the TRACKER's machine (t-24@r12: never the device the
// human sits at - the host reader is test-only machinery),
// "tracker/t-N[@rK]" names a task board served by that tracker's own
// read surface, and nothing else is a location. A bare relative path
// is refused, loudly, never guessed.
//
// Machine names may carry colons themselves (user@host:port), so the
// separator is the first colon whose remainder starts with "/".

export type ParsedLocation = { machine: string; path: string } | null;

// The tracker-named form: a repository directory name, then a task
// spec or an entry page (tasks, agents - the pipeline and the glance,
// t-14 step 3). The name carries no colon or slash, so it never
// collides with the machine:/path form.
const TRACKER_LOCATION = /^([A-Za-z0-9][A-Za-z0-9._-]*)\/(t-\d+(?:@r\d+)?|tasks|agents)$/;

/** The entry pages the instruments read: the pipeline (tasks) and the
 * glance (agents), per t-14 "Tasks and agents". */
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

// The reaching form (t-34@r6 change 5): the tracker location with the
// host it answers at in front, "host[:port]/tracker/spec". Typing it
// once reaches the tracker and remembers its host; afterwards the bare
// "tracker/spec" form finds it. A missing port is sac serve's default.
const HOSTED_LOCATION =
  /^((?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9][A-Za-z0-9.-]*)(?::(\d{1,5}))?)\/([A-Za-z0-9][A-Za-z0-9._-]*)\/(t-\d+(?:@r\d+)?|tasks|agents)$/;
export const DEFAULT_TRACKER_PORT = 8811;

export type HostedLocation = { host: string; tracker: string; path: string };

export function hostedTrackerLocation(input: string): HostedLocation | null {
  const match = HOSTED_LOCATION.exec(input.trim());
  if (!match || match[1] === undefined || match[3] === undefined || match[4] === undefined)
    return null;
  const host = match[2] === undefined ? `${match[1]}:${DEFAULT_TRACKER_PORT}` : match[1];
  return { host, tracker: match[3], path: match[4] };
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
  if (raw.startsWith("/")) return { machine: "tracker-machine", path: raw };
  const tracker = trackerLocation(raw);
  if (tracker) return tracker;
  const hosted = hostedTrackerLocation(raw);
  if (hosted) return { machine: `tracker:${hosted.tracker}`, path: hosted.path };
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

/** The server base for a host setting: a bare host:port gets http. */
export function baseFromHost(host: string): URL | null {
  const raw = host.includes("://") ? host : `http://${host}`;
  try {
    return new URL(raw);
  } catch {
    return null;
  }
}

/** A route on the server, resolved against the base: a URL's string
 * form ends in a slash, so string concatenation would double it. */
export function apiUrl(base: URL, route: string): string {
  return new URL(route, base).toString();
}

/** The route tail after /view/v1/ for a page or file location. */
export function viewUrl(base: URL, route: string): string {
  return new URL(`/view/v1/${route.replace(/^\//, "")}`, base).toString();
}

/** The parseable string form of a location: the form the header field,
 * the picker and the inspection open all take back. */
export function formatLocation(location: { machine: string; path: string }): string {
  if (location.machine.startsWith("tracker:")) {
    return `${location.machine.slice("tracker:".length)}/${location.path}`;
  }
  if (location.machine === "tracker-machine") {
    return location.path;
  }
  return `${location.machine}:${location.path}`;
}
