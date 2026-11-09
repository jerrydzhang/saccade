// The window-state pool's client (t-14 "The suite", t-24@r20): every
// suite window shows one pool state - what it shows, the part at the
// reading height, its zoom, its annotation drafts - and the state
// lives on the tracker's server, outside the record. Two devices
// showing the same state stay in sync: writes settle before sending,
// every write names its device, and a device recognises its own echo
// by the sequence its writes returned and never re-applies it. The
// device is an opaque client-generated id persisted on the device.

import { serverBase } from "./tracker.ts";
import { SourceUnavailableError } from "./board/errors.ts";

export type PoolDraft = { path: string; text: string };

/** One pooled window state, as the wire answers it with this device's
 * view beside it. Cleared fields arrive absent. */
export type WindowState = {
  id: string;
  location: string | null;
  part: string | null;
  zoom: number | null;
  drafts: PoolDraft[] | null;
  lastDevice: string | null;
  changed: number;
  attachedHere: boolean;
  link: string | null;
};

export type PoolFields = {
  location?: string | null;
  part?: string | null;
  zoom?: number | null;
  drafts?: PoolDraft[] | null;
};

type Fetcher = (url: string, init?: RequestInit) => Promise<Response>;

/** The route on the pool surface, resolved against the base: a URL's
 * string form ends in a slash, so string concatenation would double
 * it (the t-2 defect, still the rule). */
export function poolUrl(base: URL, route: string): string {
  return new URL(`pool/v1/${route.replace(/^\//, "")}`, base).toString();
}

/** The device id: opaque, generated once, persisted on the device
 * (t-24@r20: not the machine id - a browser lane has none, and two
 * app instances on one machine are two devices). */
const DEVICE_KEY = "saccade.device.id";

export function deviceId(): string {
  try {
    const existing = localStorage.getItem(DEVICE_KEY);
    if (existing && existing.trim().length > 0) return existing;
    const fresh =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `dev-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(DEVICE_KEY, fresh);
    return fresh;
  } catch {
    // no store (a hostile context): a per-run id still names writes
    return "dev-unstored";
  }
}

/** The wire omits a cleared field instead of answering null (t-24@r20's
 * doc shape), so every answer normalizes absent to null here: the
 * client's own rule - `null !== undefined` - reads one way. */
export function normalizeState(state: WindowState): WindowState {
  return {
    id: state.id,
    location: state.location ?? null,
    part: state.part ?? null,
    zoom: state.zoom ?? null,
    drafts: state.drafts ?? null,
    lastDevice: state.lastDevice ?? null,
    changed: state.changed,
    attachedHere: state.attachedHere,
    link: state.link ?? null,
  };
}

/** The echo rule: a pool write returned the sequence it landed at; a
 * watch answer at or below the newest sequence this client produced is
 * its own echo or history it already applied - never re-applied, so a
 * device does not jump its own view. */
export function isOwnEcho(changed: number, lastWrittenSeq: number): boolean {
  return changed <= lastWrittenSeq;
}

/** The no-op rule: a settled write that changes nothing the wire
 * carries is not sent - it would bump the sequence and make every
 * other device re-apply identical values. */
export function fieldsDiffer(
  fields: PoolFields,
  current: {
    location: string | null;
    part: string | null;
    zoom: number | null;
    drafts: PoolDraft[] | null;
  },
): boolean {
  if (fields.location !== undefined && fields.location !== current.location) return true;
  if (fields.part !== undefined && fields.part !== current.part) return true;
  if (fields.zoom !== undefined && fields.zoom !== current.zoom) return true;
  if (fields.drafts !== undefined) {
    const existing = JSON.stringify(current.drafts ?? null);
    const next = JSON.stringify(fields.drafts ?? null);
    if (existing !== next) return true;
  }
  return false;
}

/** The one-time interim-draft move (t-24@r20): each device moves its
 * interim drafts into the pool once, per task - into this device's
 * window state showing that task if there is one, otherwise into a
 * new detached state showing the task, which the window picker then
 * lists. The interim store is marked moved only after the pool
 * confirms the write; the rows stay where they are and nothing reads
 * them again. Zoom owes nothing; it was never kept. */
const MOVED_KEY = "saccade.annotation.moved";
const INTERIM_PREFIX = "saccade.annotation.drafts.";

export function interimDraftKeys(): string[] {
  const keys: string[] = [];
  try {
    for (let index = 0; index < localStorage.length; index++) {
      const key = localStorage.key(index);
      if (key !== null && key.startsWith(INTERIM_PREFIX)) keys.push(key);
    }
  } catch {
    return [];
  }
  return keys;
}

export function readInterimDrafts(key: string): PoolDraft[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (entry): entry is PoolDraft =>
        typeof entry === "object" &&
        entry !== null &&
        typeof (entry as PoolDraft).path === "string" &&
        typeof (entry as PoolDraft).text === "string",
    );
  } catch {
    return [];
  }
}

export function markMoved(keys: string[]): void {
  try {
    const moved = new Set(JSON.parse(localStorage.getItem(MOVED_KEY) ?? "[]") as string[]);
    for (const key of keys) moved.add(key);
    localStorage.setItem(MOVED_KEY, JSON.stringify([...moved]));
  } catch {
    // a hostile store: the move retries on the next start, harmless
  }
}

function movedKeys(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(MOVED_KEY) ?? "[]") as string[]);
  } catch {
    return new Set();
  }
}

/** The draft lists merge by path: the moved drafts take their part's
 * slot, the state's own drafts keep every path the move does not
 * carry (t-24@r20: no draft is ever lost, none moved twice). */
export function mergeDrafts(existing: PoolDraft[] | null, moved: PoolDraft[]): PoolDraft[] {
  const byPath = new Map<string, PoolDraft>();
  for (const draft of existing ?? []) byPath.set(draft.path, draft);
  for (const draft of moved) byPath.set(draft.path, draft);
  // the same order everywhere: the merged list reads by part position,
  // so two devices sharing the state compute one list
  return [...byPath.values()].sort((a, b) =>
    a.path.localeCompare(b.path, undefined, { numeric: true }),
  );
}

/** The window state on this device showing the task, if there is one:
 * the move's target (t-24@r20). */
export function stateShowingTask(states: WindowState[], location: string): WindowState | null {
  return (
    states.find(
      (state) => state.location === location && state.attachedHere && state.id !== "",
    ) ?? null
  );
}

export class PoolClient {
  readonly device: string;
  readonly base: URL;
  private readonly fetcher: Fetcher;
  // the newest sequence one of this client's own writes landed at; the
  // echo filter compares watch answers against it
  private lastWrittenSeq = 0;

  private constructor(base: URL, device: string, fetcher: Fetcher) {
    this.base = base;
    this.device = device;
    this.fetcher = fetcher;
  }

  /** The client for the tracker's pool, or null when no tracker
   * server is set (the window runs unbacked, like every read face
   * without a server). */
  static open(fetcher: Fetcher = (url, init) => fetch(url, init)): PoolClient | null {
    let base: URL;
    try {
      base = serverBase();
    } catch (error) {
      if (error instanceof SourceUnavailableError) return null;
      return null;
    }
    return new PoolClient(base, deviceId(), fetcher);
  }

  private async call(
    route: string,
    init: RequestInit & { body?: string },
  ): Promise<{ status: number; doc: unknown }> {
    const response = await this.fetcher(poolUrl(this.base, route), {
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
    const text = await response.text();
    let doc: unknown = null;
    try {
      doc = text.length > 0 ? JSON.parse(text) : null;
    } catch {
      doc = null;
    }
    return { status: response.status, doc };
  }

  private fail(route: string, status: number, doc: unknown): never {
    const detail =
      (doc as { error?: { detail?: string } } | null)?.error?.detail ?? "no detail";
    throw new Error(`pool ${route} refused (${status}): ${detail}`);
  }

  private stateOf(doc: unknown): WindowState {
    const state = (doc as { state?: WindowState } | null)?.state;
    if (!state) throw new Error("the pool answered no state");
    return normalizeState(state);
  }

  /** The pool's own facts and standing sequence. */
  async seq(): Promise<number> {
    const { status, doc } = await this.call("", { method: "GET" });
    if (status !== 200) this.fail("root", status, doc);
    return (doc as { seq: number }).seq;
  }

  /** Every state, with this device's view beside them. */
  async list(): Promise<WindowState[]> {
    const { status, doc } = await this.call(`states?device=${encodeURIComponent(this.device)}`, {
      method: "GET",
    });
    if (status !== 200) this.fail("states", status, doc);
    const states = (doc as { states?: WindowState[] } | null)?.states;
    return (states ?? []).map(normalizeState);
  }

  /** Birth of the window's state: the id is client-generated and
   * opaque; an id that already exists is returned as it stands (the
   * caller re-reads instead of clobbering). */
  async create(id: string, fields: PoolFields = {}): Promise<WindowState> {
    const { status, doc } = await this.call("states", {
      method: "POST",
      body: JSON.stringify({ id, device: this.device, ...fields }),
    });
    if (status === 409) return await this.getState(id);
    if (status !== 200) this.fail("states", status, doc);
    this.noteWritten(doc);
    return this.stateOf(doc);
  }

  async getState(id: string): Promise<WindowState> {
    const { status, doc } = await this.call(
      `states/${encodeURIComponent(id)}?device=${encodeURIComponent(this.device)}`,
      { method: "GET" },
    );
    if (status !== 200) this.fail(`states/${id}`, status, doc);
    return this.stateOf(doc);
  }

  /** This device starts showing the state. */
  async attach(id: string): Promise<WindowState> {
    const { status, doc } = await this.call(
      `devices/${encodeURIComponent(this.device)}/attach`,
      { method: "POST", body: JSON.stringify({ state: id }) },
    );
    if (status !== 200) this.fail(`attach ${id}`, status, doc);
    return this.stateOf(doc);
  }

  /** This device stops showing the state; the state stays. A link on
   * this device pointing at it lapses server-side (t-14). */
  async detach(id: string): Promise<WindowState> {
    const { status, doc } = await this.call(
      `devices/${encodeURIComponent(this.device)}/detach`,
      { method: "POST", body: JSON.stringify({ state: id }) },
    );
    if (status !== 200) this.fail(`detach ${id}`, status, doc);
    return this.stateOf(doc);
  }

  /** A settled change to the window's own fields. Absent fields stay;
   * null clears. The landed sequence is remembered for the echo
   * filter; a write that changes nothing is not sent. */
  async patch(id: string, fields: PoolFields, current?: {
    location: string | null;
    part: string | null;
    zoom: number | null;
    drafts: PoolDraft[] | null;
  }): Promise<number | null> {
    if (current && !fieldsDiffer(fields, current)) return null;
    const { status, doc } = await this.call(`states/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ device: this.device, ...fields }),
    });
    if (status !== 200) this.fail(`patch ${id}`, status, doc);
    this.noteWritten(doc);
    return (doc as { seq: number }).seq;
  }

  /** The rare act: the state leaves the pool on every device. */
  async discard(id: string): Promise<void> {
    const { status, doc } = await this.call(`states/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (status !== 200 && status !== 404) this.fail(`discard ${id}`, status, doc);
  }

  /** Set or replace this device's one outgoing link (t-14: space l). */
  async setLink(from: string, to: string): Promise<void> {
    const { status, doc } = await this.call(
      `devices/${encodeURIComponent(this.device)}/links/${encodeURIComponent(from)}`,
      { method: "PUT", body: JSON.stringify({ to }) },
    );
    if (status !== 200) this.fail(`link ${from}`, status, doc);
  }

  /** Unlink (t-14: space u). Unlinking a window that carries no link
   * changes nothing. */
  async clearLink(from: string): Promise<void> {
    const { status, doc } = await this.call(
      `devices/${encodeURIComponent(this.device)}/links/${encodeURIComponent(from)}`,
      { method: "DELETE" },
    );
    if (status !== 200) this.fail(`unlink ${from}`, status, doc);
  }

  private noteWritten(doc: unknown): void {
    const seq = (doc as { seq?: number } | null)?.seq;
    if (typeof seq === "number" && seq > this.lastWrittenSeq) this.lastWrittenSeq = seq;
  }

  /** The sync loop (t-24@r20): long-poll the state with `?after=`,
   * apply every answer that is not this client's own echo. A dropped
   * connection retries with backoff and then re-reads the state whole,
   * like the companion reads its page. Answers `stop` when the caller
   * stops it. */
  watch(id: string, onState: (state: WindowState) => void): { stop: () => void } {
    let stopped = false;
    let after = this.lastWrittenSeq;
    let backoff = 500;
    const run = async (): Promise<void> => {
      while (!stopped) {
        try {
          const { status, doc } = await this.call(
            `states/${encodeURIComponent(id)}?device=${encodeURIComponent(this.device)}&after=${after}&timeout=25`,
            { method: "GET" },
          );
          if (stopped) return;
          if (status !== 200) {
            if (status === 404) {
              // the state is gone (discarded elsewhere): answer and stop
              onState({
                id,
                location: null,
                part: null,
                zoom: null,
                drafts: null,
                lastDevice: null,
                changed: 0,
                attachedHere: false,
                link: null,
              });
              return;
            }
            throw new Error(`pool watch refused (${status})`);
          }
          backoff = 500;
          const seq = (doc as { seq: number }).seq;
          if (seq > after) {
            after = seq;
            const state = this.stateOf(doc);
            if (!isOwnEcho(state.changed, this.lastWrittenSeq)) {
              if (seq > this.lastWrittenSeq) this.lastWrittenSeq = seq;
              onState(state);
            }
          }
        } catch {
          if (stopped) return;
          // the connection dropped: retry with backoff, then re-read
          // whole - `after` resets so the next answer is the state as
          // it stands
          await new Promise((resolve) => setTimeout(resolve, backoff));
          backoff = Math.min(backoff * 2, 10000);
          after = 0;
        }
      }
    };
    void run();
    return {
      stop: () => {
        stopped = true;
      },
    };
  }

  /** The one-time interim-draft move (t-24@r20), per pooled start. */
  async moveInterimDrafts(): Promise<void> {
    const keys = interimDraftKeys().filter((key) => !movedKeys().has(key));
    if (keys.length === 0) return;
    const states = await this.list();
    const confirmed: string[] = [];
    for (const key of keys) {
      const tail = key.slice(INTERIM_PREFIX.length);
      const slash = tail.lastIndexOf(".");
      if (slash <= 0) continue;
      const tracker = tail.slice(0, slash);
      const task = tail.slice(slash + 1);
      const drafts = readInterimDrafts(key);
      if (drafts.length === 0) {
        confirmed.push(key);
        continue;
      }
      const location = `${tracker}/${task}`;
      try {
        const target = stateShowingTask(states, location);
        if (target) {
          const merged = mergeDrafts(target.drafts, drafts);
          await this.patch(target.id, { drafts: merged });
        } else {
          await this.create(`moved-${tracker}-${task}-${crypto.randomUUID?.() ?? Date.now()}`, {
            location,
            drafts,
          });
        }
        confirmed.push(key);
      } catch {
        // this task moves on the next pooled start; nothing is lost
      }
    }
    // the interim store is marked moved only after the pool confirmed
    markMoved(confirmed);
  }
}
