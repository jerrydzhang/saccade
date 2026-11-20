// Board sources: one interface, explicit machine + path. The fixture
// adapter serves the deterministic development fixtures; the host adapter
// reads through the narrow Tauri file commands in the native shell (the
// lanes' test-only reader); the remote adapter reads a machine-named
// location through the shell bridge's read-only face; anywhere else
// refuses loudly. No silent adapter swapping.
//
// Port carry (t-36 slice 1, the source/transport layer allowlist): the
// module is the old tree's sources.ts with one port-specific adaptation -
// the fixture corpus stays at the repository's app/fixtures while this
// tree's vite root is app/port, so the bundled globs are relative to
// this module and rekeyed to the same /fixtures namespace the adapter's
// key logic speaks.

import { resolveRelative } from "./resolve.ts";
import { TrackerSource } from "./tracker.ts";
import { LANE_CODES, SourceUnavailableError, type LaneCode } from "./board/errors.ts";
import { theAsks } from "./asks.ts";

// "fixture" and "host" are this viewer's own machines; "host" is the
// test-only local-file reader (the lanes' fixture reader, never
// presented as human functionality); any other non-empty name is a
// machine from the local core's list.
export type Machine = string;

export type Location = {
  machine: Machine;
  /** Fixture name, or an explicit declaration file path on the host. */
  path: string;
};

export type LoadedText = {
  text: string;
  location: Location;
};

export type LoadedAsset = {
  bytes: Uint8Array;
  path: string;
};

export interface BoardSource {
  readonly machine: Machine;
  loadText(location: Location): Promise<LoadedText>;
  loadAsset(location: Location, relative: string): Promise<LoadedAsset>;
}

export { SourceUnavailableError };

// Fixture boards and companions are bundled as raw text; binary assets
// are bundled as URLs and fetched, so bytes reach the read model exactly
// as committed - PNG/JPEG never pass through a UTF-8 decode.
// The globs reach the shared fixture corpus three directories up from
// this module (app/port/src/lib -> app/fixtures); vite keys
// relative globs relative to the importing module, so both maps rekey to
// the adapter's own /fixtures namespace below. Glob patterns must be
// string literals, so the prefix repeats in each call.
const fixtureGlobBase = "../../../fixtures";

function rekeyToFixtureNamespace(globbed: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(globbed)) {
    const cut = key.lastIndexOf(`${fixtureGlobBase}/`);
    if (cut >= 0) out[`/${key.slice(cut + fixtureGlobBase.length + 1)}`] = value;
  }
  return out;
}

// Plain Node can exercise the tracker/controller contracts without
// Vite's bundled fixtures. Vite supplies import.meta.env and still
// transforms the literal globs (the macro itself is not a runtime API).
const fixtureRaw = rekeyToFixtureNamespace(
  import.meta.env !== undefined
    ? import.meta.glob<string>("../../../fixtures/*.{yaml,md}", {
        query: "?raw",
        import: "default",
        eager: true,
      }) as Record<string, string>
    : {},
);

const fixtureAssetUrls = rekeyToFixtureNamespace(
  import.meta.env !== undefined
    ? import.meta.glob<string>("../../../fixtures/assets/*", {
        query: "?url",
        import: "default",
        eager: true,
      }) as Record<string, string>
    : {},
);

const fixtureBase = "/fixtures";

function fixtureKey(name: string): string | null {
  const candidate = name.startsWith("/") ? name : `${fixtureBase}/${name}`;
  if (candidate in fixtureRaw) return candidate;
  const withExt = `${candidate}.yaml`;
  if (withExt in fixtureRaw) return withExt;
  return null;
}

function fixtureText(key: string): string {
  const text = fixtureRaw[key];
  if (text === undefined)
    throw new SourceUnavailableError("asset-missing", `fixture content vanished for ${key}`);
  return text;
}

export class FixtureSource implements BoardSource {
  readonly machine = "fixture" as const;

  async loadText(location: Location): Promise<LoadedText> {
    const key = fixtureKey(location.path);
    if (key === null) {
      throw new SourceUnavailableError(
        "asset-missing",
        `no fixture named ${JSON.stringify(location.path)}`,
      );
    }
    return { text: fixtureText(key), location };
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    // Relative assets resolve against the declaration's directory, exactly
    // as the host lane does on the selected machine.
    const dirIndex = location.path.lastIndexOf("/");
    const baseDir =
      dirIndex < 0 ? fixtureBase : `${fixtureBase}/${location.path.slice(0, dirIndex)}`;
    const joined = resolveRelative(baseDir, relative);
    if (!joined.ok) {
      throw new SourceUnavailableError(joined.failure.code, joined.failure.reason);
    }
    const key = joined.joined;
    const url = fixtureAssetUrls[key];
    if (url === undefined) {
      // A text asset referenced as a figure still comes through its
      // bundled URL so bytes stay byte-exact.
      if (!(key in fixtureRaw)) {
        throw new SourceUnavailableError("asset-missing", `fixture asset not found: ${relative}`);
      }
    }
    const response = await fetch(url ?? key);
    if (!response.ok) {
      throw new SourceUnavailableError("asset-missing", `fixture asset not found: ${relative}`);
    }
    return {
      bytes: new Uint8Array(await response.arrayBuffer()),
      path: relative,
    };
  }
}

export interface TauriInvoke {
  (cmd: string, args?: Record<string, unknown>): Promise<unknown>;
}

export function tauriInvoke(): TauriInvoke {
  const w = (typeof window === "undefined" ? {} : window) as unknown as {
    __TAURI_INTERNALS__?: { invoke: TauriInvoke };
  };
  const invoke = w.__TAURI_INTERNALS__?.invoke;
  if (!invoke) {
    throw new SourceUnavailableError(
      "lane-unavailable",
      "the host lane exists only inside the native shell; use an explicit fixture in the browser lane",
    );
  }
  return invoke;
}

export class HostFileSource implements BoardSource {
  readonly machine = "host" as const;

  async loadText(location: Location): Promise<LoadedText> {
    return { text: await theAsks().readFile(this.machine, location.path), location };
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const baseDir = location.path.replace(/\/[^/]*$/, "");
    const resolved = resolveRelative(baseDir, relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(resolved.failure.code, resolved.failure.reason);
    }
    return {
      bytes: await theAsks().readAsset(this.machine, baseDir, relative),
      path: relative,
    };
  }
}

// A machine-named board: every read rides the one module's asks on
// that machine (the text read, and the asset read beside it), and
// containment is proven on the remote side. Adding machines, logging in
// or updating cores is the human's own host setup; every machine state
// arrives here as a structured error, never a retry.
export class RemoteSource implements BoardSource {
  readonly machine: string;

  constructor(machine: string) {
    this.machine = machine;
  }

  async loadText(location: Location): Promise<LoadedText> {
    return { text: await theAsks().readFile(this.machine, location.path), location };
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const baseDir = location.path.replace(/\/[^/]*$/, "");
    const resolved = resolveRelative(baseDir, relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(resolved.failure.code, resolved.failure.reason);
    }
    return {
      bytes: await theAsks().readAsset(this.machine, baseDir, relative),
      path: relative,
    };
  }
}

// One tracker source per machine: its page and contract caches must
// live across the text and figure loads of one open, so one open
// resolves its revision once and text and figures read one immutable
// tree.
const trackerSources = new Map<Machine, TrackerSource>();

/** Every tracker the window has touched, for the picker's candidates. */
export function trackerSourcesForPicker(): TrackerSource[] {
  return [...trackerSources.values()];
}

export function sourceFor(machine: Machine): BoardSource {
  if (machine === "fixture") return new FixtureSource();
  if (machine === "host") return new HostFileSource();
  // A tracker-named board rides that tracker's own read surface.
  if (machine.startsWith("tracker:")) {
    let source = trackerSources.get(machine);
    if (source === undefined) {
      source = new TrackerSource(machine.slice("tracker:".length));
      trackerSources.set(machine, source);
    }
    return source;
  }
  // Runtime callers can pass unknown machines despite the static type.
  // An empty name is a refusal; anything else is a core machine name.
  if (machine.length === 0) {
    throw new SourceUnavailableError("lane-unavailable", "empty machine name");
  }
  return new RemoteSource(machine);
}

export function fixtureNames(): string[] {
  return Object.keys(fixtureRaw)
    .filter((key) => key.endsWith(".yaml"))
    .map((key) => key.slice(fixtureBase.length + 1))
    .sort();
}
