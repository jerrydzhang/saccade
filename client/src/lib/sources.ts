// Board sources: one interface, explicit machine + path. The fixture
// adapter serves the deterministic development fixtures; the host adapter
// reads through the narrow Tauri file commands in the native shell; the
// remote adapter reads a machine-named location through the local
// herdr-core's read-only face; anywhere else refuses loudly. No silent
// adapter swapping.

import { resolveRelative } from "./resolve.ts";
import {
  TrackerSource,
  serverHost,
  trackerIsThisMachine,
} from "./tracker.ts";
import {
  LANE_CODES,
  SourceUnavailableError,
  type LaneCode,
} from "./board/errors.ts";

// "fixture" and "host" are this viewer's own machines; "host" is the
// test-only local-file reader (t-24@r11: the lanes' fixture reader,
// never presented as human functionality); any other non-empty name is
// a machine from the local core's list.
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
const fixtureRaw = import.meta.glob<string>("/fixtures/*.{yaml,md}", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const fixtureAssetUrls = import.meta.glob<string>("/fixtures/assets/*", {
  query: "?url",
  import: "default",
  eager: true,
}) as Record<string, string>;

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
    throw new SourceUnavailableError(
      "asset-missing",
      `fixture content vanished for ${key}`,
    );
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
      dirIndex < 0
        ? fixtureBase
        : `${fixtureBase}/${location.path.slice(0, dirIndex)}`;
    const joined = resolveRelative(baseDir, relative);
    if (!joined.ok) {
      throw new SourceUnavailableError(
        joined.failure.code,
        joined.failure.reason,
      );
    }
    const key = joined.joined;
    const url = fixtureAssetUrls[key];
    if (url === undefined) {
      // A text asset referenced as a figure still comes through its
      // bundled URL so bytes stay byte-exact.
      if (!(key in fixtureRaw)) {
        throw new SourceUnavailableError(
          "asset-missing",
          `fixture asset not found: ${relative}`,
        );
      }
    }
    const response = await fetch(url ?? key);
    if (!response.ok) {
      throw new SourceUnavailableError(
        "asset-missing",
        `fixture asset not found: ${relative}`,
      );
    }
    return {
      bytes: new Uint8Array(await response.arrayBuffer()),
      path: relative,
    };
  }
}

interface TauriInvoke {
  (cmd: string, args?: Record<string, unknown>): Promise<unknown>;
}

export function tauriInvoke(): TauriInvoke {
  const w = window as unknown as {
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

// The bridge rejects with its BridgeError {code, message}; keep the
// structured code and human message instead of flattening them into a
// string. An unknown code is a bridge bug, surfaced as one.
function bridgeFailure(error: unknown): SourceUnavailableError {
  if (typeof error === "object" && error !== null) {
    const candidate = error as { code?: unknown; message?: unknown };
    if (
      typeof candidate.code === "string" &&
      typeof candidate.message === "string"
    ) {
      const known = (LANE_CODES as readonly string[]).includes(candidate.code);
      const code = known ? (candidate.code as LaneCode) : "bridge-invalid";
      return new SourceUnavailableError(code, candidate.message);
    }
  }
  return new SourceUnavailableError("read-failed", String(error));
}

export class HostFileSource implements BoardSource {
  readonly machine = "host" as const;

  async loadText(location: Location): Promise<LoadedText> {
    const invoke = tauriInvoke();
    try {
      const result = (await invoke("read_declaration", {
        path: location.path,
      })) as { text: string };
      return { text: result.text, location };
    } catch (error) {
      throw bridgeFailure(error);
    }
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const invoke = tauriInvoke();
    const baseDir = location.path.replace(/\/[^/]*$/, "");
    const resolved = resolveRelative(baseDir, relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(
        resolved.failure.code,
        resolved.failure.reason,
      );
    }
    try {
      const bytes = (await invoke("read_asset", {
        base: baseDir,
        relative,
      })) as number[];
      return { bytes: new Uint8Array(bytes), path: relative };
    } catch (error) {
      throw bridgeFailure(error);
    }
  }
}

// A machine-named board: every read rides the LOCAL core's read-only
// face (file.read for text, fixed-argv exec for bytes and remote path
// resolution), and containment is proven on the remote side. Adding
// machines, logging in or updating cores is the human's own host setup;
// every machine state arrives here as a structured error, never a retry.
export class RemoteSource implements BoardSource {
  readonly machine: string;

  constructor(machine: string) {
    this.machine = machine;
  }

  async loadText(location: Location): Promise<LoadedText> {
    const invoke = tauriInvoke();
    try {
      const result = (await invoke("read_declaration", {
        path: location.path,
        on: this.machine,
      })) as { text: string };
      return { text: result.text, location };
    } catch (error) {
      throw bridgeFailure(error);
    }
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const invoke = tauriInvoke();
    const baseDir = location.path.replace(/\/[^/]*$/, "");
    const resolved = resolveRelative(baseDir, relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(
        resolved.failure.code,
        resolved.failure.reason,
      );
    }
    try {
      const bytes = (await invoke("read_asset", {
        base: baseDir,
        relative,
        on: this.machine,
      })) as number[];
      return { bytes: new Uint8Array(bytes), path: relative };
    } catch (error) {
      throw bridgeFailure(error);
    }
  }
}

// A bare path is a file on the TRACKER's machine (t-24@r12), read
// through Herdr core like any named machine - never the device the
// human sits at. The hub proof is machine identity (t-24@r14): the
// tracker root page's machine id equals this device's own
// /etc/machine-id, and then the local core's own machine is read with
// no declaration. Otherwise - a different machine, or an unavailable
// id on either side - the client configuration's `tracker-machine`
// declaration applies, and without it the open refuses saying what to
// set. No bridge (no local core), no tracker-machine read at all.
export class TrackerMachineSource implements BoardSource {
  readonly machine = "tracker-machine" as const;

  /** The Herdr machine name the read rides: the declared tracker
   * machine, or "" for the local core's own machine on the hub. */
  private async trackerCoreMachine(): Promise<string> {
    let invoke: TauriInvoke;
    try {
      invoke = tauriInvoke();
    } catch (error) {
      if (error instanceof SourceUnavailableError) {
        throw new SourceUnavailableError(
          "lane-unavailable",
          "a bare path reads on the tracker's machine through Herdr core; no Herdr connection exists here - name a machine explicitly (machine:/path)",
        );
      }
      throw error;
    }
    let ownId: string | null = null;
    try {
      ownId = (await invoke("machine_id")) as string | null;
    } catch (error) {
      throw bridgeFailure(error);
    }
    let declared: string | null = null;
    try {
      declared = (await invoke("tracker_machine")) as string | null;
    } catch (error) {
      throw bridgeFailure(error);
    }
    if (declared !== null && declared.trim().length > 0) return declared.trim();
    if (serverHost() === null) {
      throw new SourceUnavailableError(
        "lane-unavailable",
        "a bare path reads on the tracker's machine; no tracker server is set - set the tracker server host, or tracker-machine in the client configuration",
      );
    }
    const hub = await trackerIsThisMachine(async () => ownId);
    if (hub) return ""; // the local core's own machine: the hub itself
    throw new SourceUnavailableError(
      "lane-unavailable",
      "a bare path reads on the tracker's machine, which this device is not; set tracker-machine in the client configuration to the Herdr machine that hosts the tracker",
    );
  }

  private async resolve(): Promise<string> {
    return this.trackerCoreMachine();
  }

  async loadText(location: Location): Promise<LoadedText> {
    const coreMachine = await this.resolve();
    const invoke = tauriInvoke();
    try {
      const result = (await invoke("read_declaration", {
        path: location.path,
        on: coreMachine,
      })) as { text: string };
      return { text: result.text, location };
    } catch (error) {
      throw bridgeFailure(error);
    }
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const coreMachine = await this.resolve();
    const invoke = tauriInvoke();
    const baseDir = location.path.replace(/\/[^/]*$/, "");
    const resolved = resolveRelative(baseDir, relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(
        resolved.failure.code,
        resolved.failure.reason,
      );
    }
    try {
      const bytes = (await invoke("read_asset", {
        base: baseDir,
        relative,
        on: coreMachine,
      })) as number[];
      return { bytes: new Uint8Array(bytes), path: relative };
    } catch (error) {
      throw bridgeFailure(error);
    }
  }
}

// One tracker source per machine: its page and contract caches must
// live across the text and figure loads of one open, so one open
// resolves its revision once and text and figures read one immutable
// tree.
const trackerSources = new Map<Machine, TrackerSource>();

export function sourceFor(machine: Machine): BoardSource {
  if (machine === "fixture") return new FixtureSource();
  if (machine === "host") return new HostFileSource();
  if (machine === "tracker-machine") return new TrackerMachineSource();
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
