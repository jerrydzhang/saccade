// Tracker-named boards: "<tracker>/t-N[@rK]" served by that tracker's
// own read surface, /view/v1 (t-14 "Pages", t-24@r2). The tracker is
// named by its repository directory; the server is found through one
// setting, the server host, whose value is the address the tracker
// publishes beside its database. The viewer reads /view/v1 only - the
// root page is the version contract, the board page resolves a bare
// location to its pinned revision and names the task, and the authored
// board.yaml and its figures ride the file family at that pinned
// revision, so one open's text and figures are one immutable tree.
// /api/v1 stays sac's own API; nothing here touches it.

import { SUPPORTED_GRAMMAR_VERSIONS } from "./board/decode.ts";
import type {
  BoardSource,
  LoadedAsset,
  LoadedText,
  Location,
} from "./sources.ts";
import { SourceUnavailableError } from "./board/errors.ts";
import { baseFromHost, parseTaskSpec } from "./location.ts";
import { viewUrl } from "./location.ts";
import { resolveRelative } from "./resolve.ts";
import {
  fetchFileBytes,
  fetchPage,
  pageFact,
  type Fetcher,
  type ViewPage,
} from "./pages.ts";

// The tracker's identity facts, from the root page (t-24@r14, r16).
export type TrackerIdentity = {
  machineId: string | null;
  repoPath: string | null;
};

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

export function serverBase(): URL {
  const host = serverHost();
  if (!host || host.trim().length === 0) {
    throw new SourceUnavailableError(
      "lane-unavailable",
      "no tracker server set; the tracker publishes its address beside its database - set it in the header once",
    );
  }
  const base = baseFromHost(host);
  if (base === null) {
    throw new SourceUnavailableError(
      "lane-unavailable",
      `the tracker server '${host}' is not a host or URL`,
    );
  }
  return base;
}

/** True when the tracker host is this machine's loopback: the hub, where
 * the local core reads the tracker's own files (t-24@r12). */
/** The hub proof, per t-24@r14: the root page's machine id equals this
 * device's own machine id. Never address shape; null never equals
 * null; an unavailable id on either side means not the hub. */
export async function trackerIsThisMachine(
  ownMachineId: () => Promise<string | null>,
  fetcher: Fetcher = (url) => fetch(url),
): Promise<boolean> {
  const own = (await ownMachineId()) ?? "";
  if (own.length === 0) return false;
  const root = await fetchPage(fetcher, viewUrl(serverBase(), ""));
  const trackerId = root.machineId ?? "";
  if (trackerId.length === 0) return false;
  return trackerId === own;
}

type Contract = {
  tracker: string | null;
  identity: TrackerIdentity;
  boardGrammar: { min: number; max: number };
  viewMajors: number[];
};

/** The task facts a window has seen, for the picker (t-24@r11: the
 * picker enumerates what the window can know). */
export type TaskFact = { spec: string; name: string | null };

export class TrackerSource implements BoardSource {
  private contract: Promise<Contract> | null = null;
  // pinned spec -> the board page (an immutable revision: the page and
  // its facts cache; the file routes ride the same revision)
  private pages = new Map<string, Promise<ViewPage>>();
  // every task the window has seen, newest resolution first
  readonly seenTasks: TaskFact[] = [];

  readonly tracker: string;
  private readonly fetcher: Fetcher;

  constructor(tracker: string, fetcher: Fetcher = (url) => fetch(url)) {
    this.tracker = tracker;
    this.fetcher = fetcher;
  }

  get machine(): string {
    return `tracker:${this.tracker}`;
  }

  /** The version contract, checked before any board read: the server
   * must serve this tracker, its grammar range must overlap the
   * versions this viewer decodes, and it must serve a view major this
   * viewer reads. */
  private async checkContract(): Promise<Contract> {
    if (this.contract !== null) return this.contract;
    const load = (async () => {
      const base = serverBase();
      const root = await fetchPage(this.fetcher, viewUrl(base, ""));
      if (
        root.tracker !== null &&
        root.tracker !== undefined &&
        root.tracker !== this.tracker
      ) {
        throw new SourceUnavailableError(
          "machine-unknown",
          `the server serves tracker '${root.tracker}', not '${this.tracker}'; name the tracker by its repository directory`,
        );
      }
      const range = root.boardGrammar;
      if (!range) {
        throw new SourceUnavailableError(
          "core-protocol",
          "the tracker's root page carries no grammar range; move the binaries in step",
        );
      }
      const viewer = SUPPORTED_GRAMMAR_VERSIONS;
      const lowest = viewer[0];
      const highest = viewer[viewer.length - 1];
      const overlaps =
        lowest !== undefined &&
        highest !== undefined &&
        range.min <= highest &&
        range.max >= lowest;
      if (!overlaps) {
        throw new SourceUnavailableError(
          "core-protocol",
          `the tracker's grammar range is [${range.min},${range.max}], this viewer supports [${viewer[0]},${viewer[viewer.length - 1]}]; no version works on both sides - move the binaries in step`,
        );
      }
      if (
        root.viewMajors !== undefined &&
        !root.viewMajors.includes(root.view.major)
      ) {
        throw new SourceUnavailableError(
          "core-protocol",
          `the tracker serves view majors [${root.viewMajors.join(",")}], not the major its own pages carry - move the binaries in step`,
        );
      }
      return {
        tracker: root.tracker ?? null,
        identity: {
          machineId: root.machineId ?? null,
          repoPath: root.repoPath ?? null,
        },
        boardGrammar: range,
        viewMajors: root.viewMajors ?? [root.view.major],
      };
    })();
    this.contract = load;
    void load.catch(() => {
      // a refused contract is not a fact about the tracker: drop it,
      // so the next open tries the wire again
      if (this.contract === load) this.contract = null;
    });
    return load;
  }

  /** The board page at a pinned revision: the open's resolver and the
   * task-facts source. Cached per revision; a failure clears. */
  /** The board page at a pinned revision: the open's resolver and the
   * task-facts source. Cached per PINNED revision only - an immutable
   * revision's page never moves; a bare location never touches the
   * cache, so a later open re-resolves instead of replaying an old
   * current. A failure clears its entry. */
  private boardPage(pinned: string): Promise<ViewPage> {
    const spec = parseTaskSpec(pinned);
    const cacheable = spec !== null && spec.revision !== null;
    if (cacheable) {
      const cached = this.pages.get(pinned);
      if (cached) return cached;
    }
    const load = (async () => {
      await this.checkContract();
      const base = serverBase();
      const page = await fetchPage(this.fetcher, viewUrl(base, pinned));
      if (page.kind !== "document") {
        throw new SourceUnavailableError(
          "core-protocol",
          `'${pinned}' answered a ${page.kind} page, not a board`,
        );
      }
      this.rememberTask(pinned, page);
      return page;
    })();
    if (cacheable) {
      this.pages.set(pinned, load);
      void load.catch(() => {
        if (this.pages.get(pinned) === load) this.pages.delete(pinned);
      });
    }
    return load;
  }

  /** The pinned spec a location reads at: a pinned location is itself;
   * a bare location resolves through the board page. The page rides
   * along only for a bare resolution; a pinned read never blocks on
   * one - its facts are learned in the background. */
  private async resolvePinned(
    path: string,
  ): Promise<{ pinned: string; page: ViewPage | null }> {
    const spec = parseTaskSpec(path);
    if (!spec) {
      throw new SourceUnavailableError(
        "asset-missing",
        `'${path}' is not a task spec (t-N or t-N@rK)`,
      );
    }
    if (spec.revision !== null) {
      const pinned = path.trim();
      // the facts a pinned open contributes to the picker arrive
      // out of band; the read itself rides the pinned file routes
      void this.boardPage(pinned).catch(() => {});
      return { pinned, page: null };
    }
    // a bare location re-resolves through the board page on EVERY
    // open: a later open never replays the revision an earlier one
    // answered
    const page = await this.boardPage(`t-${spec.task}`);
    return { pinned: page.location, page };
  }

  private rememberTask(pinned: string, page: ViewPage): void {
    const spec = page.location || pinned;
    const task = page.task ?? pageFact(page, "task");
    if (!task) return;
    const name = pageFact(page, "name");
    this.seenTasks.splice(
      0,
      this.seenTasks.length,
      ...this.seenTasks.filter((fact) => fact.spec !== spec),
    );
    this.seenTasks.unshift({ spec, name });
  }

  async loadText(location: Location): Promise<LoadedText> {
    const { pinned, page } = await this.resolvePinned(location.path);
    const base = serverBase();
    const bytes = await fetchFileBytes(
      this.fetcher,
      viewUrl(base, `${pinned}/file/board.yaml`),
    );
    const text = new TextDecoder().decode(bytes);
    // a bare location resolved: the caller renders the revision that
    // answered, and its figures read the same immutable revision
    const resolved =
      location.path.trim() === pinned
        ? location
        : { ...location, path: pinned };
    void page;
    return { text, location: resolved };
  }

  /** Re-read the current revision's board page right now: the publish
   * guard compares against it (a stale window never overwrites a
   * revision that landed after its load). */
  async currentBoardPage(task: number): Promise<ViewPage> {
    return this.boardPage(`t-${task}`);
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const { pinned } = await this.resolvePinned(location.path);
    const resolved = resolveRelative("", relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(
        resolved.failure.code,
        resolved.failure.reason,
      );
    }
    const base = serverBase();
    const bytes = await fetchFileBytes(
      this.fetcher,
      viewUrl(base, `${pinned}/file/${resolved.joined}`),
    );
    return { bytes, path: relative };
  }

  /** The history page of a task, current or pinned: the timeline the
   * history mode walks. */
  async history(path: string): Promise<ViewPage> {
    const spec = parseTaskSpec(path);
    if (!spec) {
      throw new SourceUnavailableError(
        "asset-missing",
        `'${path}' is not a task spec (t-N or t-N@rK)`,
      );
    }
    await this.checkContract();
    const base = serverBase();
    return fetchPage(this.fetcher, viewUrl(base, `${path.trim()}/history`));
  }

  /** The authored board.yaml at a pinned revision, as text: the source
   * the history mode diffs and the annotations edit. */
  async boardTextAt(pinned: string): Promise<string> {
    const spec = parseTaskSpec(pinned);
    if (!spec || spec.revision === null) {
      throw new SourceUnavailableError(
        "asset-missing",
        `'${pinned}' is not a pinned task spec (t-N@rK)`,
      );
    }
    const base = serverBase();
    const bytes = await fetchFileBytes(
      this.fetcher,
      viewUrl(base, `${pinned}/file/board.yaml`),
    );
    return new TextDecoder().decode(bytes);
  }

  /** Re-read the current revision's board text right now: the
   * publish guard compares against it (t-17: a stale window never
   * overwrites a revision that landed after its load). */
  async currentBoardText(
    task: number,
  ): Promise<{ pinned: string; text: string }> {
    const page = await this.boardPage(`t-${task}`);
    const pinned = page.location;
    const text = await this.boardTextAt(pinned);
    return { pinned, text };
  }
}
