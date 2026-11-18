// The app's device-local window store (t-71): window state belongs to
// the app and lives on the device it runs on, in one document the
// backend holds. This revises the tracker store's shape (the
// tracker's api/v1/windows; the client of it, the old windows.ts,
// retired when the app switched): the store is single-device and
// local, so there is no
// device id, no lastDevice, no changed counter and no stored attach
// set - attachedHere is derived, answered from the states bound to
// live windows of this app instance - and a window's link lives on
// its state, the from-state's outgoing link target. The backend is
// dumb text persistence; the serialization, the schema version and
// every refusal live here. This same code runs in the app and in the
// browser tests: there is no HTTP anywhere in it.

export type WindowDraft = { path: string; text: string };

/** One stored window state, as a reader answers it: every field
 * present (null when the document leaves it out). */
export type WindowState = {
  id: string;
  location: string | null;
  part: string | null;
  zoom: number | null;
  /** The board column's width scale (lib/board/reading-layout.ts). */
  width: number | null;
  drafts: WindowDraft[] | null;
  /** Derived, never stored: whether a live window of this app
   * instance is bound to the state right now. */
  attachedHere: boolean;
  /** The from-state's outgoing link target (the o act's remembered
   * receiver); null when the window carries no link. */
  link: string | null;
};

/** The fields a create or patch carries - the old WindowFields' three
 * states: an absent field stays as it stands, null clears. The link
 * is not among them: it moves by setLink and clearLink. */
export type WindowFields = {
  location?: string | null;
  part?: string | null;
  zoom?: number | null;
  width?: number | null;
  drafts?: WindowDraft[] | null;
};

/** One state's stored fields: a field with no value may be absent or
 * null; both read as null. */
type StoredEntry = {
  location?: string | null;
  part?: string | null;
  zoom?: number | null;
  width?: number | null;
  drafts?: WindowDraft[] | null;
  link?: string | null;
};

type StoredDocument = { schema: number; states: Record<string, StoredEntry> };

const SCHEMA = 1;

/** The store document's key, wherever this origin keeps text: the
 * bridge's file is <app-data-dir>/window-store.json (window_store.rs);
 * a realm with no bridge and the mocked shell keep it in localStorage
 * under this key. Exported for them - the kit may import app code,
 * never the reverse. */
export const WINDOW_STORE_KEY = "saccade.window-store";

/** The one boundary between the stored text and the store: every read
 * passes here, so a document that parses but lies about its shape
 * refuses with its reason instead of reaching a reader. */
function parseDocument(text: string): StoredDocument {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`the window store document is malformed: it does not parse (${String(error)})`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("the window store document is malformed: it is not a document object");
  }
  const document = parsed as { schema?: unknown; states?: unknown };
  if (!("schema" in parsed)) {
    throw new Error("the window store document is malformed: it names no schema");
  }
  if (document.schema !== SCHEMA) {
    throw new Error(
      `the window store document's schema is ${JSON.stringify(document.schema)}, not ${SCHEMA}`,
    );
  }
  if (typeof document.states !== "object" || document.states === null || Array.isArray(document.states)) {
    throw new Error("the window store document is malformed: it carries no states map");
  }
  for (const [id, entry] of Object.entries(document.states)) {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      throw new Error(`the window store document is malformed: the state ${id} is not an object`);
    }
  }
  return document as StoredDocument;
}

function stateOf(id: string, entry: StoredEntry, live: ReadonlySet<string>): WindowState {
  return {
    id,
    location: entry.location ?? null,
    part: entry.part ?? null,
    zoom: entry.zoom ?? null,
    width: entry.width ?? null,
    drafts: entry.drafts ?? null,
    attachedHere: live.has(id),
    link: entry.link ?? null,
  };
}

function applyFields(entry: StoredEntry, fields: WindowFields): void {
  if ("location" in fields) entry.location = fields.location;
  if ("part" in fields) entry.part = fields.part;
  if ("zoom" in fields) entry.zoom = fields.zoom;
  if ("width" in fields) entry.width = fields.width;
  if ("drafts" in fields) entry.drafts = fields.drafts;
}

export class WindowStore {
  readonly #backend: { load(): Promise<string | null>; save(text: string): Promise<void> };
  readonly #liveStates: () => Promise<ReadonlySet<string>>;

  constructor(options: {
    backend: { load(): Promise<string | null>; save(text: string): Promise<void> };
    /** The states bound to live windows of this app instance, asked
     * fresh at every read that answers attachedHere and at every
     * link: the attach fact is never stored. */
    liveStates: () => Promise<ReadonlySet<string>>;
  }) {
    this.#backend = options.backend;
    this.#liveStates = options.liveStates;
  }

  /** The stored document as it stands: a store with no document yet
   * answers an empty one, and its first save starts schema 1. */
  async #document(): Promise<StoredDocument> {
    const text = await this.#backend.load();
    if (text === null) return { schema: SCHEMA, states: {} };
    return parseDocument(text);
  }

  // Every mutation reads the document as it stands, touches only the
  // one state's own entry and writes the document back: the document
  // is shared by this device's windows, so a write built on anything
  // but the document as it stands would drop a sibling window's
  // state. A window only ever writes its own entry; it never
  // rewrites another window's fields.

  /** Birth of a window's state: the id is client-generated and
   * opaque; an id that already exists returns as it stands. */
  async create(id: string, fields: WindowFields = {}): Promise<WindowState> {
    const document = await this.#document();
    const existing = document.states[id];
    if (existing !== undefined) return stateOf(id, existing, await this.#liveStates());
    const entry: StoredEntry = {};
    applyFields(entry, fields);
    document.states[id] = entry;
    await this.#write(document);
    return stateOf(id, entry, await this.#liveStates());
  }

  async getState(id: string): Promise<WindowState> {
    const document = await this.#document();
    const entry = document.states[id];
    if (entry === undefined) throw new Error(`the window store holds no state ${id}`);
    return stateOf(id, entry, await this.#liveStates());
  }

  /** A settled change to the window's own fields; absent fields
   * stay, null clears. */
  async patch(id: string, fields: WindowFields): Promise<WindowState> {
    const document = await this.#document();
    const entry = document.states[id];
    if (entry === undefined) throw new Error(`the window store holds no state ${id}`);
    applyFields(entry, fields);
    await this.#write(document);
    return stateOf(id, entry, await this.#liveStates());
  }

  /** Set or replace the from-state's outgoing link: the o act's
   * remembered receiver. The tracker's not_attached invariant, now
   * checked against live windows: both sides must carry a live
   * window, and a window does not link to itself. */
  async setLink(from: string, to: string): Promise<void> {
    if (from === to) {
      throw new Error(`setLink refused: a window does not link to itself (${from})`);
    }
    const live = await this.#liveStates();
    if (!live.has(from)) throw new Error(`setLink refused: no live window carries ${from}`);
    if (!live.has(to)) throw new Error(`setLink refused: no live window carries ${to}`);
    const document = await this.#document();
    const entry = document.states[from];
    if (entry === undefined) throw new Error(`the window store holds no state ${from}`);
    entry.link = to;
    await this.#write(document);
  }

  /** Unlink; unlinking a window that carries no link changes
   * nothing. */
  async clearLink(from: string): Promise<void> {
    const document = await this.#document();
    const entry = document.states[from];
    if (entry === undefined) return;
    if (entry.link === undefined || entry.link === null) return;
    delete entry.link;
    await this.#write(document);
  }

  async #write(document: StoredDocument): Promise<void> {
    await this.#backend.save(JSON.stringify(document));
  }
}
