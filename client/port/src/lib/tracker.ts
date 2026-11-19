// Tracker-named boards: "<project>/t-N[@rK]" served by that project's
// tracker, reached through the one core module: the project's
// declaration declares its tracker's loopback port, and every page and
// file route here rides the module's request on the project's machine.
// The project is the one the device registered by its repository path;
// the declaration at that path names it. The viewer reads /view/v1
// only - the root page is the version contract, the board page
// resolves a bare location to its pinned revision and names the task,
// and the authored board.yaml and its figures ride the file family at
// that pinned revision, so one open's text and figures are one
// immutable tree.

import { SUPPORTED_GRAMMAR_VERSIONS } from "./board/decode.ts";
import type { BoardSource, LoadedAsset, LoadedText, Location } from "./sources.ts";
import { SourceUnavailableError } from "./board/errors.ts";
import { instrumentPath, parseTaskSpec, viewRoute } from "./location.ts";
import { resolveRelative } from "./resolve.ts";
import { fetchFileBytes, fetchPage, pageFact, type Fetcher, type PageChange, type ViewPage } from "./pages.ts";
import { theAsks, theProjects } from "./asks.ts";
import type { Project } from "./projects.ts";

// The tracker's identity facts, from the root page.
export type TrackerIdentity = {
  machineId: string | null;
  repoPath: string | null;
};

type Contract = {
  tracker: string | null;
  identity: TrackerIdentity;
  boardGrammar: { min: number; max: number };
  viewMajors: number[];
};

/** The task facts a window has seen, for the picker (the picker
 * enumerates what the window can know). */
export type TaskFact = { spec: string; name: string | null };

export class TrackerSource implements BoardSource {
  private contract: Promise<Contract> | null = null;
  // the registered path the cached contract was checked against: a
  // re-registration at another path re-checks, never replays
  private contractFor: string | null = null;
  // pinned spec -> the board page (an immutable revision: the page and
  // its facts cache; the file routes ride the same revision)
  private pages = new Map<string, Promise<ViewPage>>();
  // every task the window has seen, newest resolution first
  readonly seenTasks: TaskFact[] = [];

  readonly tracker: string;

  constructor(tracker: string) {
    this.tracker = tracker;
  }

  get machine(): string {
    return `tracker:${this.tracker}`;
  }

  /** The registered project naming itself as this source's tracker,
   * read fresh from the registry (its declaration cache makes that
   * cheap): the refusal here is the add field's to explain, not the
   * rail's. */
  private async registeredProject(): Promise<Project> {
    const project = await theProjects().projectByName(this.tracker);
    if (project === null) {
      throw new SourceUnavailableError(
        "lane-unavailable",
        `no registered project names itself '${this.tracker}'; add it on the plane (a, then machine:/path/to/repository)`,
      );
    }
    return project;
  }

  /** The route fetcher: the declared loopback address of the
   * registered project naming itself as this source's tracker, asked
   * through the one module on the project's machine. Resolved per
   * call - the registry's answer is read fresh, so a register or a
   * forget takes hold without a stale wire. */
  private async wire(): Promise<Fetcher> {
    const project = await this.registeredProject();
    const port = project.declaration.port;
    if (port === null) {
      throw new SourceUnavailableError(
        "lane-unavailable",
        `the project '${this.tracker}' declares no [tracker] port; its declaration names the project alone`,
      );
    }
    const machine = project.machine;
    const address = { host: "127.0.0.1", port };
    return async (route) => new Response(await theAsks().requestLocal(machine, address, route));
  }

  /** The version contract, checked before any board read: the server
   * must serve this tracker, its grammar range must overlap the
   * versions this viewer decodes, and it must serve a view major this
   * viewer reads. */
  private async checkContract(): Promise<Contract> {
    const project = await this.registeredProject();
    if (this.contract !== null && this.contractFor === project.path) return this.contract;
    const load = (async () => {
      const root = await fetchPage(await this.wire(), viewRoute(""));
      if (root.tracker !== null && root.tracker !== undefined && root.tracker !== this.tracker) {
        throw new SourceUnavailableError(
          "machine-unknown",
          `the server serves tracker '${root.tracker}', not '${this.tracker}'; name the tracker by its repository directory`,
        );
      }
      // the port lives in two places: the declaration the app read and
      // the address the server was started at. The server says which
      // repository it serves; when it is not the registered one, the
      // app says so and names both
      if (root.repoPath !== undefined && root.repoPath !== null && root.repoPath !== project.path) {
        throw new SourceUnavailableError(
          "machine-unknown",
          `'${this.tracker}' is registered at ${project.path}, but the tracker on the port its declaration names serves ${root.repoPath}; the two places name different repositories`,
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
      if (root.viewMajors !== undefined && !root.viewMajors.includes(root.view.major)) {
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
    this.contractFor = project.path;
    void load.catch(() => {
      // a refused contract is not a fact about the tracker: drop it,
      // so the next open tries the wire again
      if (this.contract === load) {
        this.contract = null;
        this.contractFor = null;
      }
    });
    return load;
  }

  /** The board page at a pinned revision: the open's resolver and the
   * task-facts source. Cached per PINNED revision only - an immutable
   * revision's page never moves; a bare location never touches the
   * cache, so a later open re-resolves instead of replaying an old
   * current. A failure clears its entry. */
  private boardPage(fetcher: Fetcher, pinned: string): Promise<ViewPage> {
    const spec = parseTaskSpec(pinned);
    const cacheable = spec !== null && spec.revision !== null;
    if (cacheable) {
      const cached = this.pages.get(pinned);
      if (cached) return cached;
    }
    const load = (async () => {
      await this.checkContract();
      const page = await fetchPage(fetcher, viewRoute(pinned));
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
    fetcher: Fetcher,
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
      void this.boardPage(fetcher, pinned).catch(() => {});
      return { pinned, page: null };
    }
    // a bare location re-resolves through the board page on EVERY
    // open: a later open never replays the revision an earlier one
    // answered
    const page = await this.boardPage(fetcher, `t-${spec.task}`);
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

  /** The entry pages the instruments read: the pipeline (tasks) and
   * the glance (agents), fetched fresh every time - they are views of
   * the record's now, never cached (a bare location re-resolves on
   * every open, and these are the barest of them all). */
  async instrumentPage(path: "tasks" | "agents"): Promise<ViewPage> {
    await this.checkContract();
    const page = await fetchPage(await this.wire(), viewRoute(path));
    if (page.kind !== "document" || page.location !== path) {
      throw new SourceUnavailableError(
        "core-protocol",
        `'${path}' answered a ${page.kind} page at '${page.location}'`,
      );
    }
    return page;
  }

  /** The agents page, for the tasks instrument's live join: the
   * rows' agent ids reach the words only through the sessions'
   * terminal addresses, which this page carries. */
  async agentsPage(): Promise<ViewPage> {
    return this.instrumentPage("agents");
  }

  /** The tracker's identity facts (its root page, read once per
   * contract): the rail places an agent no tracker registered by the
   * repo path. */
  async identity(): Promise<TrackerIdentity> {
    return (await this.checkContract()).identity;
  }

  async loadText(location: Location): Promise<LoadedText> {
    if (instrumentPath(location.path)) {
      throw new SourceUnavailableError(
        "asset-denied",
        `'${location.path}' is an instrument page, not board text; the instruments read it as a view document`,
      );
    }
    const fetcher = await this.wire();
    const { pinned, page } = await this.resolvePinned(fetcher, location.path);
    const bytes = await fetchFileBytes(fetcher, viewRoute(`${pinned}/file/board.yaml`));
    const text = new TextDecoder().decode(bytes);
    // a bare location resolved: the caller renders the revision that
    // answered, and its figures read the same immutable revision
    const resolved = location.path.trim() === pinned ? location : { ...location, path: pinned };
    void page;
    return { text, location: resolved };
  }

  /** Re-read the current revision's board page right now: the publish
   * guard compares against it (a stale window never overwrites a
   * revision that landed after its load). */
  async currentBoardPage(task: number): Promise<ViewPage> {
    return this.boardPage(await this.wire(), `t-${task}`);
  }

  /** Raw bytes of a board-tree file at a pinned revision: the
   * publish act re-posts the figures unchanged. */
  async loadAssetBytes(pinned: string, path: string): Promise<Uint8Array> {
    await this.checkContract();
    return fetchFileBytes(await this.wire(), viewRoute(`${pinned}/file/${path}`));
  }

  /** The cited components at a select path: the incorporation expand's
   * only source - expanding shows just the cited parts. */
  async citedComponents(pinned: string, selectPath: string): Promise<ViewPage> {
    await this.checkContract();
    return fetchPage(await this.wire(), viewRoute(`${pinned}/content${selectPath}`));
  }

  async loadAsset(location: Location, relative: string): Promise<LoadedAsset> {
    const fetcher = await this.wire();
    const { pinned } = await this.resolvePinned(fetcher, location.path);
    const resolved = resolveRelative("", relative);
    if (!resolved.ok) {
      throw new SourceUnavailableError(resolved.failure.code, resolved.failure.reason);
    }
    const bytes = await fetchFileBytes(fetcher, viewRoute(`${pinned}/file/${resolved.joined}`));
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
    return fetchPage(await this.wire(), viewRoute(`${path.trim()}/history`));
  }

  /** The component changes between two revisions of a task: the diff
   * page's `changes`, the alignment history mode shows. A tracker that
   * serves none predates them: refused, never guessed. */
  async revisionChanges(task: number, from: number, to: number): Promise<PageChange[]> {
    await this.checkContract();
    const page = await fetchPage(await this.wire(), viewRoute(`t-${task}@r${from}..r${to}/diff`));
    if (page.changes === undefined) {
      throw new SourceUnavailableError(
        "core-protocol",
        `the tracker's diff page for t-${task}@r${from}..r${to} carries no component changes; move the binaries in step`,
      );
    }
    return page.changes;
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
    const bytes = await fetchFileBytes(await this.wire(), viewRoute(`${pinned}/file/board.yaml`));
    return new TextDecoder().decode(bytes);
  }

  /** Re-read the current revision's board text right now: the
   * publish guard compares against it (a stale window never
   * overwrites a revision that landed after its load). */
  async currentBoardText(task: number): Promise<{ pinned: string; text: string }> {
    const page = await this.boardPage(await this.wire(), `t-${task}`);
    const pinned = page.location;
    const text = await this.boardTextAt(pinned);
    return { pinned, text };
  }
}
