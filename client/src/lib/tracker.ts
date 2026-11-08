// Tracker-named boards: "<tracker>/t-N[@rK]" served by that tracker's
// own HTTP API. The tracker is named by its repository directory; the
// server is found through one setting, the server host, whose value is
// the address the tracker publishes beside its database. Boards ride
// the /api/v1/boards routes - never a file-system read of any state
// directory. The version contract is checked on every open: the server
// must serve the named tracker, and its grammar range must overlap the
// versions this viewer decodes.

import { SUPPORTED_GRAMMAR_VERSIONS } from "./board/decode.ts";
import type { BoardSource, LoadedAsset, LoadedText, Location } from "./sources.ts";
import { SourceUnavailableError } from "./sources.ts";
import { apiUrl, baseFromHost, parseTaskSpec } from "./location.ts";
import { resolveRelative } from "./resolve.ts";

// The one viewer setting: the tracker server's host. The value is the
// address the tracker publishes beside its database (the runtime
// address hint), pasted once.
const SERVER_KEY = "saccade.server.host";

export function serverHost(): string | null {
  try {
    return localStorage.getItem(SERVER_KEY);
  } catch {
    return null;
  }
}

export function setServerHost(host: string): void {
  localStorage.setItem(SERVER_KEY, host.trim());
}

function serverBase(): URL {
  const host = serverHost();
  if (!host || host.trim().length === 0) {
    throw new SourceUnavailableError(
      "lane-unavailable",
      "no tracker server set; the tracker publishes its address beside its database - set it in the header once",
    );
  }
  const base = baseFromHost(host);
  if (base === null) {
    throw new SourceUnavailableError("lane-unavailable", `the tracker server '${host}' is not a host or URL`);
  }
  return base;
}

type VersionReply = {
  tracker: string | null;
  board_grammar: { min: number; max: number } | null;
};

async function fetchJson(url: string): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new SourceUnavailableError("core-unavailable", `the tracker server did not answer at ${url}: ${String(error)}`);
  }
  if (!response.ok) {
    let detail = `status ${response.status}`;
    try {
      const body = (await response.json()) as { error?: { code?: string; detail?: string } };
      if (body.error?.code) detail = `${body.error.code}: ${body.error.detail ?? ""}`;
    } catch {
      // a non-JSON body still refuses with the status
    }
    throw new SourceUnavailableError("core-protocol", `the tracker server refused ${url}: ${detail}`);
  }
  try {
    return await response.json();
  } catch (error) {
    throw new SourceUnavailableError("core-protocol", `the tracker server answered ${url} with non-JSON: ${String(error)}`);
  }
}

type TreeFile = { path: string; content: string };
// the server nests the list: {"files": {"files": [...]}, commit, revision}
type TreeReply = { files: { files: TreeFile[] }; commit: string; revision: number };

function decodeBase64(content: string): Uint8Array {
  const binary = atob(content);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** One tracker's boards over the wire: the tree arrives once per
 * revision and both the board text and its figures read from it. */
export class TrackerSource implements BoardSource {
  private trees = new Map<string, Promise<TreeReply>>();

  readonly tracker: string;

  constructor(tracker: string) {
    this.tracker = tracker;
  }

  get machine(): string {
    return `tracker:${this.tracker}`;
  }

  /** The version contract, checked before any board read: the server
   * must serve this tracker, and its grammar range must overlap the
   * versions this viewer decodes. */
  private async checkContract(): Promise<void> {
    const base = serverBase();
    const version = (await fetchJson(apiUrl(base, "/api/v1/version"))) as Partial<VersionReply>;
    if (version.tracker !== null && version.tracker !== undefined && version.tracker !== this.tracker) {
      throw new SourceUnavailableError(
        "machine-unknown",
        `the server serves tracker '${version.tracker}', not '${this.tracker}'; name the tracker by its repository directory`,
      );
    }
    const range = version.board_grammar;
    if (!range) {
      throw new SourceUnavailableError(
        "core-protocol",
        "the server's version answer carries no grammar range; move the binaries in step",
      );
    }
    const viewer = SUPPORTED_GRAMMAR_VERSIONS;
    const lowest = viewer[0];
    const highest = viewer[viewer.length - 1];
    const overlaps = lowest !== undefined && highest !== undefined && range.min <= highest && range.max >= lowest;
    if (!overlaps) {
      throw new SourceUnavailableError(
        "core-protocol",
        `the tracker's grammar range is [${range.min},${range.max}], this viewer supports [${viewer[0]},${viewer[viewer.length - 1]}]; no version works on both sides - move the binaries in step`,
      );
    }
  }

  private tree(location: Location): Promise<TreeReply> {
    const cached = this.trees.get(location.path);
    if (cached) return cached;
    const spec = parseTaskSpec(location.path);
    if (!spec) {
      return Promise.reject(
        new SourceUnavailableError("asset-missing", `'${location.path}' is not a task spec (t-N or t-N@rK)`),
      );
    }
    const load = (async () => {
      await this.checkContract();
      const base = serverBase();
      const suffix = spec.revision === null ? "" : `&revision=${spec.revision}`;
      const reply = (await fetchJson(apiUrl(base, `/api/v1/boards/get?task=${spec.task}${suffix}`))) as TreeReply;
      if (!Array.isArray(reply.files?.files)) {
        throw new SourceUnavailableError("core-protocol", "the boards/get answer carries no file list");
      }
      return reply;
    })();
    this.trees.set(location.path, load);
    return load;
  }

  async loadText(location: Location): Promise<LoadedText> {
    const tree = await this.tree(location);
    const board = tree.files.files.find((file) => file.path === "board.yaml");
    if (!board) {
      throw new SourceUnavailableError("asset-missing", `t-${location.path} holds no board.yaml`);
    }
    const bytes = decodeBase64(board.content);
    return { text: new TextDecoder().decode(bytes), location };
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const tree = await this.tree(location);
    const resolved = resolveRelative("", relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(resolved.failure.code, resolved.failure.reason);
    }
    const file = tree.files.files.find((entry) => entry.path === resolved.joined);
    if (!file) {
      throw new SourceUnavailableError("asset-missing", `revision t-${location.path} holds no file '${relative}'`);
    }
    return { bytes: decodeBase64(file.content), path: relative };
  }

}
