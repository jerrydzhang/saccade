// The window store: per-device persisted window states on the home
// tracker (t-14 "space l" lineage; the S18 elsewhere surface). The
// store is the TRACKER's HTTP surface at api/v1/windows/ - a wire
// contract, mirrored here in port idioms (the injected fetcher, the
// honest error). One state per window: the shown location, the link
// this device's window points at (the o act's receiver), the attach
// fact. The device id is opaque, generated once, persisted on the
// device - two app instances on one machine are two devices.

export type WindowDraft = { path: string; text: string };

/** One stored window state, this device's view beside it - as the
 * client holds it: every field present (null when the wire left it
 * out), the client's names (stateOf normalizes the wire). */
export type WindowState = {
  id: string;
  location: string | null;
  part: string | null;
  zoom: number | null;
  /** The board column's width scale (lib/board/reading-layout.ts);
   * null from a tracker that predates it. */
  width: number | null;
  drafts: WindowDraft[] | null;
  lastDevice: string | null;
  changed: number;
  attachedHere: boolean;
  link: string | null;
};

export type WindowFields = {
  location?: string | null;
  part?: string | null;
  zoom?: number | null;
  width?: number | null;
  drafts?: WindowDraft[] | null;
};

const DEVICE_KEY = "saccade.device.id";

export function deviceId(): string {
  try {
    const existing = localStorage.getItem(DEVICE_KEY);
    if (existing !== null && existing.length > 0) return existing;
  } catch {
    // a hostile store: a session-scoped id still serves this window
  }
  const fresh =
    typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `dev-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  try {
    localStorage.setItem(DEVICE_KEY, fresh);
  } catch {
    // unreadable store: the id lives for this session only
  }
  return fresh;
}

export function windowsUrl(base: URL | string, route: string): string {
  return new URL(`api/v1/windows/${route.replace(/^\//, "")}`, base).toString();
}

/** The state document as the tracker serializes it (its StateDoc):
 * snake_case names, and a field with no value left out. */
type WireState = {
  id: string;
  location?: string | null;
  part?: string | null;
  zoom?: number | null;
  width?: number | null;
  drafts?: WindowDraft[] | null;
  last_device?: string | null;
  changed: number;
  attached_here: boolean;
  link?: string | null;
};

/** The one boundary between the wire and the client: every read
 * passes here, so nothing past it meets an absent field or a wire
 * name (the old app's rule, 5a4b96d: an absent location reached as
 * undefined once, and the restore died on it). */
function normalizeState(wire: WireState): WindowState {
  return {
    id: wire.id,
    location: wire.location ?? null,
    part: wire.part ?? null,
    zoom: wire.zoom ?? null,
    width: wire.width ?? null,
    drafts: wire.drafts ?? null,
    lastDevice: wire.last_device ?? null,
    changed: wire.changed,
    attachedHere: wire.attached_here === true,
    link: wire.link ?? null,
  };
}

function stateOf(doc: unknown): WindowState {
  const state = (doc as { state?: WireState } | null)?.state;
  if (state === null || state === undefined || typeof state !== "object") {
    throw new Error("the window store answered no state");
  }
  return normalizeState(state);
}

export class WindowStore {
  readonly device: string;
  readonly #base: URL | string | (() => URL | string);
  readonly #fetcher: (url: string, init: RequestInit) => Promise<Response>;

  constructor(options: {
    /** A fixed base, or a resolver answered per call - a re-registered
     * tracker host serves the next store call without a rebuild. */
    base: URL | string | (() => URL | string);
    fetcher?: (url: string, init: RequestInit) => Promise<Response>;
    device?: string;
  }) {
    this.device = options.device ?? deviceId();
    this.#base = options.base;
    this.#fetcher =
      options.fetcher ??
      ((url, init) => fetch(url, { ...init, headers: { "Content-Type": "application/json" } }));
  }

  async #call(
    route: string,
    init: RequestInit & { body?: string },
  ): Promise<{ status: number; doc: unknown }> {
    const base = typeof this.#base === "function" ? this.#base() : this.#base;
    const response = await this.#fetcher(windowsUrl(base, route), {
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

  #fail(route: string, status: number, doc: unknown): never {
    throw new Error(`the window store refused ${route} (${status}): ${JSON.stringify(doc)}`);
  }

  /** Birth of a window's state: the id is client-generated and
   * opaque; an id that already exists returns as it stands. */
  async create(id: string, fields: WindowFields = {}): Promise<WindowState> {
    const { status, doc } = await this.#call("states", {
      method: "POST",
      body: JSON.stringify({ id, device: this.device, ...fields }),
    });
    if (status === 409) return await this.getState(id);
    if (status !== 200) this.#fail("states", status, doc);
    return stateOf(doc);
  }

  async getState(id: string): Promise<WindowState> {
    const { status, doc } = await this.#call(
      `states/${encodeURIComponent(id)}?device=${encodeURIComponent(this.device)}`,
      { method: "GET" },
    );
    if (status !== 200) this.#fail(`states/${id}`, status, doc);
    return stateOf(doc);
  }

  /** This device starts showing the state (the boot attach). */
  async attach(id: string): Promise<WindowState> {
    const { status, doc } = await this.#call(`devices/${encodeURIComponent(this.device)}/attach`, {
      method: "POST",
      body: JSON.stringify({ state: id }),
    });
    if (status !== 200) this.#fail("attach", status, doc);
    return stateOf(doc);
  }

  /** This device stops showing the state; the state stays. A link on
   * this device pointing at it lapses server-side. */
  async detach(id: string): Promise<WindowState> {
    const { status, doc } = await this.#call(`devices/${encodeURIComponent(this.device)}/detach`, {
      method: "POST",
      body: JSON.stringify({ state: id }),
    });
    if (status !== 200) this.#fail("detach", status, doc);
    return stateOf(doc);
  }

  /** A settled change to the window's own fields; absent fields
   * stay, null clears. */
  async patch(id: string, fields: WindowFields): Promise<WindowState> {
    const { status, doc } = await this.#call(`states/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ device: this.device, ...fields }),
    });
    if (status !== 200) this.#fail(`patch ${id}`, status, doc);
    return stateOf(doc);
  }

  /** Set or replace this device's outgoing link: the o act's
   * remembered receiver (a window STATE id). */
  async setLink(from: string, to: string): Promise<void> {
    const { status, doc } = await this.#call(
      `devices/${encodeURIComponent(this.device)}/links/${encodeURIComponent(from)}`,
      { method: "PUT", body: JSON.stringify({ to }) },
    );
    if (status !== 200) this.#fail(`link ${from}`, status, doc);
  }

  /** Unlink; unlinking a window that carries no link changes
   * nothing. */
  async clearLink(from: string): Promise<void> {
    const { status, doc } = await this.#call(
      `devices/${encodeURIComponent(this.device)}/links/${encodeURIComponent(from)}`,
      { method: "DELETE" },
    );
    if (status !== 200) this.#fail(`unlink ${from}`, status, doc);
  }

  /** Every state, this device's view beside them (the ask's detached
   * candidates). */
  async list(): Promise<WindowState[]> {
    const { status, doc } = await this.#call(
      `states?device=${encodeURIComponent(this.device)}`,
      { method: "GET" },
    );
    if (status !== 200) this.#fail("states", status, doc);
    const states = (doc as { states?: WireState[] } | null)?.states;
    if (!Array.isArray(states)) throw new Error("the window store answered no states");
    return states.map(normalizeState);
  }
}
