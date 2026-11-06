// Board sources: one interface, explicit machine + path. The fixture
// adapter serves the deterministic development fixtures; the host adapter
// reads through the narrow Tauri file commands in the native shell and
// refuses (loudly) anywhere else. No silent adapter swapping.

import { resolveRelative } from "./resolve.ts";

export type Machine = "fixture" | "host";

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

export class SourceUnavailableError extends Error {
  readonly code: "asset-missing" | "asset-denied" | "lane-unavailable";
  constructor(code: "asset-missing" | "asset-denied" | "lane-unavailable", message: string) {
    super(message);
    this.code = code;
  }
}

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
  if (text === undefined) throw new SourceUnavailableError("asset-missing", `fixture content vanished for ${key}`);
  return text;
}

export class FixtureSource implements BoardSource {
  readonly machine = "fixture" as const;

  async loadText(location: Location): Promise<LoadedText> {
    const key = fixtureKey(location.path);
    if (key === null) {
      throw new SourceUnavailableError("asset-missing", `no fixture named ${JSON.stringify(location.path)}`);
    }
    return { text: fixtureText(key), location };
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    // Relative assets resolve against the declaration's directory, exactly
    // as the host lane does on the selected machine.
    const dirIndex = location.path.lastIndexOf("/");
    const baseDir = dirIndex < 0 ? fixtureBase : `${fixtureBase}/${location.path.slice(0, dirIndex)}`;
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
    return { bytes: new Uint8Array(await response.arrayBuffer()), path: relative };
  }
}

interface TauriInvoke {
  (cmd: string, args?: Record<string, unknown>): Promise<unknown>;
}

export function tauriInvoke(): TauriInvoke {
  const w = window as unknown as { __TAURI_INTERNALS__?: { invoke: TauriInvoke } };
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
    const invoke = tauriInvoke();
    try {
      const result = (await invoke("read_declaration", { path: location.path })) as { text: string };
      return { text: result.text, location };
    } catch (error) {
      throw new SourceUnavailableError("asset-missing", String(error));
    }
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const invoke = tauriInvoke();
    const baseDir = location.path.replace(/\/[^/]*$/, "");
    const resolved = resolveRelative(baseDir, relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(resolved.failure.code, resolved.failure.reason);
    }
    try {
      const bytes = (await invoke("read_asset", { base: baseDir, relative })) as number[];
      return { bytes: new Uint8Array(bytes), path: relative };
    } catch (error) {
      throw new SourceUnavailableError("asset-missing", String(error));
    }
  }
}

export function sourceFor(machine: Machine): BoardSource {
  if (machine === "fixture") return new FixtureSource();
  if (machine === "host") return new HostFileSource();
  // Runtime callers can pass unknown machines despite the static type.
  // Refuse loudly: reading the desktop for a remote target is exactly the
  // silence this adapter refuses.
  throw new SourceUnavailableError(
    "lane-unavailable",
    `unsupported machine: ${JSON.stringify(String(machine))}; expected "fixture" or "host"`,
  );
}

export function fixtureNames(): string[] {
  return Object.keys(fixtureRaw)
    .filter((key) => key.endsWith(".yaml"))
    .map((key) => key.slice(fixtureBase.length + 1))
    .sort();
}
