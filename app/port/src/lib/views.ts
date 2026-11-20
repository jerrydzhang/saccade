// The device's kept views: a view is a kept location, shown on the
// rail beneath the repository it belongs to. The store is a map from
// each registered path ("machine:/path", as the project store spells
// it) to the locations kept there, each written as the canonical path
// inside the repository WITHOUT the repository's name ("tasks?tag=app"):
// a repository renamed in its declaration keeps its views. Nothing
// here reads a location; the claiming instrument canonicalizes before
// anything is kept. Reads and writes never throw: a refused store
// leaves the views of this session only.

const VIEWS_KEY = "saccade.views";

type ViewStore = Record<string, string[]>;

function readStore(): ViewStore {
  try {
    const raw = localStorage.getItem(VIEWS_KEY);
    const parsed: unknown = raw === null ? {} : JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
    const out: ViewStore = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (Array.isArray(value)) out[key] = value.filter((entry): entry is string => typeof entry === "string");
    }
    return out;
  } catch {
    return {};
  }
}

function writeStore(store: ViewStore): void {
  try {
    localStorage.setItem(VIEWS_KEY, JSON.stringify(store));
  } catch {
    // storage refused: the view stays kept for this session only
  }
}

/** The locations kept beneath a registered path, in the order kept. */
export function deviceViews(registered: string): string[] {
  return readStore()[registered] ?? [];
}

/** Keep a canonical path beneath a registered path; keeping one
 * already kept adds nothing. */
export function keepDeviceView(registered: string, path: string): void {
  const store = readStore();
  const kept = store[registered] ?? [];
  if (kept.includes(path)) return;
  store[registered] = [...kept, path];
  writeStore(store);
}

/** Forget one kept path. */
export function forgetDeviceView(registered: string, path: string): void {
  const store = readStore();
  const kept = store[registered];
  if (kept === undefined) return;
  const rest = kept.filter((entry) => entry !== path);
  if (rest.length > 0) store[registered] = rest;
  else delete store[registered];
  writeStore(store);
}

/** Forget every view kept beneath a registered path: unregistering a
 * repository takes its views with it. */
export function forgetDeviceViews(registered: string): void {
  const store = readStore();
  if (!(registered in store)) return;
  delete store[registered];
  writeStore(store);
}
