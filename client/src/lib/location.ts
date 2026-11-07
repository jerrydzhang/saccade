// The explicit location grammar, one parser for every input surface
// (launch argument, open control, inspection open): "machine:/absolute/
// path" names a machine from the local core's list, "/absolute/path" is
// the machine this viewer runs on, "tracker/t-N[@rK]" names a task
// board served by that tracker's own server, and nothing else is a
// location. A bare relative path is refused, loudly, never guessed.
//
// Machine names may carry colons themselves (user@host:port), so the
// separator is the first colon whose remainder starts with "/".

export type ParsedLocation = { machine: string; path: string } | null;

// The tracker-named form: a repository directory name, then a task
// spec. The name carries no colon or slash, so it never collides with
// the machine:/path form.
const TRACKER_LOCATION = /^([A-Za-z0-9][A-Za-z0-9._-]*)\/(t-\d+(?:@r\d+)?)$/;

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
  if (raw.startsWith("/")) return { machine: "host", path: raw };
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
