// The Tasks instrument's controller (svelte-free): the control
// plane (the "plane" location, which names no tracker) showing
// one project's pipeline page - the project is the window's own state,
// carried in the plane address's path - served as rows with
// identities, and the return places scenario 17 needs - the departed
// row's identity plus viewport offset, restored by Back.
//
// The pattern is the Board instrument's, end to end: the controller
// loads through the source/transport layer (TrackerSource's
// instrumentPage - fetched fresh every time, the pipeline is a view
// of the record's now), publishes the domain frame through the
// buffer's refresh pipeline, and never reads the DOM for identity
// (targets come from the page model; geometry only decides
// visibility and placement).
//
// The shown list follows the record (lib/instruments/follow.ts): a bounded
// poll re-reads the page and refreshes through the pipeline when its
// rows changed.
//
// The frame presents what this instrument publishes: select
// mode over the targets in view, the act menu, the click paths and
// the shared reading keys are the frame's (lib/frame/presentations.ts,
// through the buffer access's presentations face). This controller
// only PUBLISHES the targets - every row and every rail entry, the
// rail staying with the task list per the brief - with acts that
// resolve when asked (taskRowActs over the live page, rail and window
// services), and keeps what is about its own data: the rail's feed,
// the add field, the return places.
//
// The control plane: beside the list, a rail of the projects
// this device has reached (switching shows a project's tasks here) and,
// under each, its running agents grouped by Herdr's word (lib/tasks/rail.ts).
// The rail's entries are published as targets like the rows: s labels
// them with the rows, and a click, a right-click or a label runs the
// same acts. Rows label first, so a row's label is the one it had
// before the rail.
//
// Registering a project is its own act (lib/tasks/add.ts): `a`
// opens the plane's add field, Enter registers host:port/name and the
// rail lists it; nothing opens. With no project known, the plane says
// so and names `a` - an empty plane, never an error. The `a` key is
// the task list's one reading-key extra; everything else it reads is
// the frame's table.
//
// Deliberately not here (recorded): find (a facility of a shown
// revision - a task list is not one) and any row cursor (j/k is pure
// scrolling, the scenario 17 resolution).

import type { BufferAccess, InstrumentAttachment } from "../components/Buffer.ts";
import { createLoadScope, failureIssue } from "../lib/instruments/scaffold.ts";
import type { ContentFrame, TargetDefinition } from "../lib/contracts/buffer.ts";
import type {
  Point,
  PreparedContent,
  RefreshOutcome,
} from "../lib/contracts/buffer.ts";
import { key, sequence } from "../lib/contracts/input.ts";
import { createFollow } from "../lib/instruments/follow.ts";
import type {
  Address,
  ContentCapture,
  ContentPlaces,
  TaskAnchor,
} from "../lib/contracts/places.ts";
import type { AnchorData, AnchorLocation, ViewportMetrics } from "../lib/contracts/places.ts";
export type { TaskAnchor };
import type { StructuredIssue } from "../lib/board/errors.ts";
import { PLANE_MACHINE, formatLocation, isPlane } from "../lib/location.ts";
import { createAddProject, type AddView } from "../lib/tasks/add.ts";
import { sourceFor, type Location } from "../lib/sources.ts";
import { TrackerSource, forgetTracker, homeTracker, knownTrackers, registerTracker } from "../lib/tracker.ts";
import { tauriInvoke } from "../lib/sources.ts";
import {
  NO_WORDS,
  agentFactsOf,
  railOf,
  railTargetsOf,
  wordsOf,
  type AgentFacts,
  type Rail,
  type RailTarget,
  type WordsSnapshot,
} from "../lib/tasks/rail.ts";
import { taskRowActs, tasksMenuLabel, type TaskPorts } from "../lib/tasks/menu.ts";
import {
  taskRowOf,
  tasksPageOf,
  tasksTargetsOf,
  type TasksPageModel,
} from "../lib/tasks/page.ts";

/** The tasks frame's published domain data. Carries no `surface`
 * field on purpose: the observation layer discriminates frames
 * structurally (a board frame names a board; this one names rows),
 * so the Board instrument's module stays untouched by this slice. The
 * targets are not here either: they ride the publication's own slot
 * (ContentFrame.targets), and this controller refreshes them there
 * when the rail changes. */
export type TasksData = Readonly<{
  /** The load that produced THIS frame; retained frames keep their
   * own token while a newer request is pending. */
  generation: number;
  /** The location the open ASKED for, formatted. */
  asked: string;
  /** The location that answered: the plane. */
  location: string;
  /** The project the plane shows (its tracker's name), the window's
   * own state; null when no project is known yet. */
  project: string | null;
  /** The raw location the shown list was read at (the project's
   * tracker): the row acts compose their specs against this machine.
   * Null when nothing resolved. */
  resolved: Location | null;
  /** invalid: the shown project's list could not be read (problem
   * says why); the rail stays, so another project is one act away. */
  status: "loading" | "loaded" | "invalid";
  /** Why the shown project's list could not be read; null otherwise. */
  problem: string | null;
  page: TasksPageModel | null;
  errors: readonly StructuredIssue[];
}>;

/** The tasks controller's face to its own component: the rail's and
 * the add field's live views and the add field's typing path. Select,
 * menu, clicks and the reading keys are the frame's (the buffer's
 * presentations face); the readout's selection fact forwards the
 * frame's own. Lifecycle lives on the buffer, not here. */
export interface TasksUI {
  /** The last label-selected row's task id, if any (identity rule:
   * the frame clears it when no published target carries the anchor
   * any more). */
  getSelection(): string | null;
  /** The control plane's rail: projects and their running agents. */
  subscribeRail(observer: (rail: Rail) => void): () => void;
  /** The add field: its live view, and the typing path. */
  subscribeAdd(observer: (view: AddView) => void): () => void;
  addView(): AddView;
  typeAdd(value: string): void;
}

/** How often a shown list re-reads its page (a named tunable,
 * a named tunable; the board follows at the same pace). */
export const TASKS_FOLLOW_POLL_MS = 3000;

const EMPTY_FRAME: TasksData = {
  generation: 0,
  asked: "",
  location: "",
  project: null,
  resolved: null,
  status: "loading",
  problem: null,
  page: null,
  errors: [],
};

export function connectTasks(buffer: BufferAccess<TasksData>, ports: TaskPorts): InstrumentAttachment<TasksUI, TaskAnchor> {
  const scope = createLoadScope<TasksData>(EMPTY_FRAME);

  // The follow (lib/instruments/follow.ts): the asked address while a
  // list from a tracker is shown, and the shown page's content
  // signature - the page has no revision, so its rows are compared.
  let followAddress: Address | null = null;
  let shownSignature: string | null = null;
  // the latest explicit open: an older one answering late never
  // starts a follow
  let openIntent = 0;
  const follow = createFollow<Address>({
    intervalMs: TASKS_FOLLOW_POLL_MS,
    async current(address) {
      try {
        const source = sourceFor(`tracker:${address.path}`);
        if (!(source instanceof TrackerSource)) return null;
        return JSON.stringify(tasksPageOf(await source.instrumentPage("tasks")));
      } catch {
        return null;
      }
    },
    shown: () => shownSignature,
    same: (shown, current) => shown === current,
    async refresh() {
      const address = followAddress;
      if (address === null) return;
      return buffer.updates.refresh(async () => load(address));
    },
    // the DOM timer id and Node's Timeout meet across lib mixes; the
    // id is opaque to the follow module
    schedule: (ms, run) => setTimeout(run, ms) as unknown as number,
    cancel: (id) => clearTimeout(id as unknown as ReturnType<typeof setTimeout>),
  });

  // -- the control plane's rail ----------------------------------------
  const railObservers = new Set<(rail: Rail) => void>();
  let words: WordsSnapshot = NO_WORDS;
  let agentPages: {
    name: string;
    agents: AgentFacts[] | null;
    problem: string | null;
    repoPath: string | null;
  }[] = [];
  let railRunning = false;
  let railTimer: ReturnType<typeof setTimeout> | null = null;

  function currentProject(): string | null {
    return scope.frame().project;
  }

  // projects forgotten this session: the shown one stays off the rail
  // though the window still shows it, until it is reached again
  const forgotten = new Set<string>();

  /** The shown project when it belongs on the rail. */
  function railCurrent(): string | null {
    const current = currentProject();
    if (current === null) return null;
    if (knownTrackers()[current] !== undefined) forgotten.delete(current);
    return forgotten.has(current) ? null : current;
  }

  function railNow(): Rail {
    // every reached project is listed, read or not yet
    const known = knownTrackers();
    const names = new Set([...Object.keys(known), ...agentPages.map((entry) => entry.name)]);
    const current = railCurrent();
    if (current !== null) names.add(current);
    const projects = [...names].map(
      (name) =>
        agentPages.find((entry) => entry.name === name) ?? { name, agents: null, problem: null, repoPath: null },
    );
    return railOf(projects, words, currentProject(), homeTracker(), known);
  }

  // -- the published targets -------------------------------------------
  // The frame presents what is published here: every row, then every
  // rail entry (the brief's rule: the rail stays with the task list,
  // its entries published as targets like the rows). The acts resolve
  // when asked - taskRowActs over the live page, rail and services -
  // never frozen at publication. The rail's changes republish the
  // targets alone (the page object is shared), gated on a signature,
  // so a words tick costs nothing.
  const actDeps = (): Parameters<typeof taskRowActs>[1] => ({
    machine: () => scope.frame().resolved?.machine ?? null,
    rowLink: (part) => taskRowOf(scope.frame().page ?? { groups: [] }, part)?.location ?? null,
    ports: () => ports,
    railTarget: (part) => railTargetsOf(railNow()).find((target) => target.part === part),
    forget,
    addProject: () => add.enter(),
  });

  const rowTargets = (page: TasksPageModel | null): TargetDefinition[] =>
    page === null
      ? []
      : tasksTargetsOf(page).map((row) => ({
          id: row.part,
          kind: row.kind,
          label: row.text,
          inside: null,
          acts: () => taskRowActs({ part: row.part, kind: row.kind, label: row.text }, actDeps()),
          anchor: row.task,
        }));

  const railTargetDefs = (): TargetDefinition[] =>
    railTargetsOf(railNow()).map((entry) => ({
      id: entry.part,
      kind: entry.kind,
      label: tasksMenuLabel({ kind: entry.kind, label: entry.text }, entry),
      inside: entry.within ?? null,
      acts: () => taskRowActs({ part: entry.part, kind: entry.kind, label: entry.text }, actDeps()),
      anchor: null,
    }));

  const publishedTargets = (): TargetDefinition[] => [
    ...rowTargets(scope.frame().page),
    ...railTargetDefs(),
  ];

  const targetsSignature = (targets: readonly TargetDefinition[]): string =>
    JSON.stringify(targets.map((target) => [target.id, target.label, target.inside]));

  let publishedSignature: string | null = null;

  function refreshPublishedTargets(): void {
    if (scope.isDestroyed()) return;
    const next = publishedTargets();
    const signature = targetsSignature(next);
    if (signature === publishedSignature) return;
    publishedSignature = signature;
    buffer.updates.updateDisplayed((current) => ({ ...current, targets: next }));
  }

  function forget(project: string): void {
    forgetTracker(project);
    forgotten.add(project);
    agentPages = agentPages.filter((entry) => entry.name !== project);
    notifyRail();
  }

  function notifyRail(): void {
    const rail = railNow();
    for (const observer of [...railObservers]) observer(rail);
    refreshPublishedTargets();
  }

  async function readAgentPages(): Promise<void> {
    const names = Object.keys(knownTrackers());
    const current = railCurrent();
    if (current !== null && !names.includes(current)) names.push(current);
    agentPages = await Promise.all(
      names.map(async (name) => {
        try {
          const source = sourceFor(`tracker:${name}`);
          if (!(source instanceof TrackerSource)) {
            return { name, agents: null, problem: "not a tracker", repoPath: null };
          }
          const agents = agentFactsOf(await source.agentsPage());
          // the agents page read through the contract, so the root's
          // identity is in hand; a tracker that cannot say its repo
          // path still lists its agents, and places no stray
          const repoPath = await source.identity().then(
            (identity) => identity.repoPath,
            () => null,
          );
          return { name, agents, problem: null, repoPath };
        } catch (error) {
          return {
            name,
            agents: null,
            problem: error instanceof Error ? error.message : String(error),
            repoPath: null,
          };
        }
      }),
    );
  }

  async function agentsLoop(): Promise<void> {
    if (!railRunning) return;
    await readAgentPages();
    if (!railRunning) return;
    notifyRail();
    railTimer = setTimeout(() => void agentsLoop(), TASKS_FOLLOW_POLL_MS);
  }

  // Herdr's word feed: the shell's bounded hold wakes on a change or
  // answers quiet; outside the native shell there is no feed
  async function wordsLoop(): Promise<void> {
    let invoke;
    try {
      invoke = tauriInvoke();
    } catch {
      words = { ...NO_WORDS, problem: "Herdr words reach only the native app" };
      notifyRail();
      return;
    }
    while (railRunning) {
      try {
        const next = wordsOf(await invoke("words_wait", { since: words.generation, timeout: 25 }));
        if (!railRunning) return;
        if (next.generation !== words.generation || next.connected !== words.connected) {
          words = next;
          notifyRail();
        }
      } catch {
        await new Promise((resolve) => setTimeout(resolve, TASKS_FOLLOW_POLL_MS));
      }
    }
  }

  function startRail(): void {
    if (railRunning || scope.isDestroyed()) return;
    railRunning = true;
    void agentsLoop();
    void wordsLoop();
  }

  function stopRail(): void {
    railRunning = false;
    if (railTimer !== null) clearTimeout(railTimer);
    railTimer = null;
  }

  // -- the add field: registering opens nothing -----------------------
  const addObservers = new Set<(view: AddView) => void>();
  const add = createAddProject({
    capture: (context) => buffer.interactions.capture(context),
    register(tracker, host) {
      registerTracker(tracker, host);
      forgotten.delete(tracker);
      notifyRail();
      // its agents read now rather than at the next poll
      if (railRunning) void readAgentPages().then(() => railRunning && notifyRail());
    },
  });
  add.subscribe((view) => {
    for (const observer of [...addObservers]) observer(view);
  });

  const host: TasksUI = {
    getSelection() {
      // the readout's selection fact is the frame's own: the anchor of
      // the label-selected target, cleared when no published target
      // carries it any more (scenario 12's identity rule)
      return buffer.presentations().selectedAnchor();
    },
    subscribeRail(observer) {
      railObservers.add(observer);
      observer(railNow());
      return () => {
        railObservers.delete(observer);
      };
    },
    subscribeAdd(observer) {
      addObservers.add(observer);
      observer(add.view());
      return () => {
        addObservers.delete(observer);
      };
    },
    addView: () => add.view(),
    typeAdd: (value) => add.type(value),
  };

  /** The project a plane address shows: the one it names, else the
   * one this plane already shows, else the home, else any reached -
   * only a project this device still knows, for a bare "plane". */
  function projectFor(address: Address): string | null {
    const named = address.path.trim();
    if (named.length > 0) return named;
    const known = knownTrackers();
    const current = currentProject();
    if (current !== null && known[current] !== undefined) return current;
    return homeTracker() ?? Object.keys(known)[0] ?? null;
  }

  /** Publish a loaded frame with its targets: the domain data rides
   * the publication's value, the targets its own slot - built AFTER
   * the frame is adopted, since the rail's entries depend on the
   * frame's own facts (the project it shows marks the rail). */
  function publish(frame: TasksData): PreparedContent<ContentFrame<TasksData>> {
    scope.adopt(frame);
    const targets = publishedTargets();
    publishedSignature = targetsSignature(targets);
    return { kind: "ready", value: { value: frame, targets } };
  }

  async function load(address: Address): Promise<PreparedContent<ContentFrame<TasksData>>> {
    if (scope.isDestroyed())
      return {
        kind: "failed",
        issue: failureIssue("lane-unavailable", "the buffer was destroyed"),
      };
    const token = scope.begin();
    const asked = formatLocation({ machine: address.machine, path: address.path });
    scope.setAsked(asked);
    if (!isPlane(address)) {
      // this buffer serves the control plane; a tracker's pages are
      // no location of their own - the gap is named, never
      // guessed
      return {
        kind: "failed",
        issue: failureIssue("asset-denied", `${asked} is not the plane; this surface serves the control plane`),
      };
    }
    const project = projectFor(address);
    const base = (shown: string | null, problem: string | null): TasksData => ({
      generation: token,
      asked: scope.asked(),
      location: formatLocation({ machine: PLANE_MACHINE, path: "" }),
      project: shown,
      resolved: shown === null ? null : { machine: `tracker:${shown}`, path: "tasks" },
      status: problem === null ? "loaded" : "invalid",
      problem,
      page: null,
      errors: [],
    });
    // no project known yet: the plane says how to add one (a), never
    // an error; the rail still publishes, so another project stays
    // one act away
    if (project === null) {
      shownSignature = null;
      startRail();
      return publish(base(null, null));
    }
    const location: Location = { machine: `tracker:${project}`, path: "tasks" };
    const problemOf = (error: unknown): string => (error instanceof Error ? error.message : String(error));
    let source: unknown;
    try {
      source = sourceFor(location.machine);
    } catch (error) {
      if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
      shownSignature = null;
      startRail();
      return publish(base(project, problemOf(error)));
    }
    if (!(source instanceof TrackerSource)) {
      shownSignature = null;
      startRail();
      return publish(base(project, `'${project}' is not a tracker`));
    }
    let viewPage;
    try {
      viewPage = await source.instrumentPage("tasks");
    } catch (error) {
      if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
      // the shown project is unreachable: the plane keeps its rail
      // and says why, so another project stays one act away
      shownSignature = null;
      startRail();
      return publish(base(project, problemOf(error)));
    }
    if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
    const model = tasksPageOf(viewPage);
    shownSignature = JSON.stringify(model);
    startRail();
    // the shown project moves the rail's mark; the rail's entries
    // ride the same publication
    return publish({
      generation: token,
        asked: scope.asked(),
        location: formatLocation({ machine: PLANE_MACHINE, path: "" }),
        project,
        resolved: location,
        status: "loaded",
        problem: null,
        page: model,
        errors: [],
      });
  }

  // The frame's presentation: the places below read its anchors (the
  // reading position's, and the restore's) instead of walking the
  // targets themselves - the same walk in any instrument is the
  // frame's, not this controller's. The reading keys ride the same
  // face.
  const presentations = buffer.presentations();

  const places: ContentPlaces<TaskAnchor> = {
    captureDisplayed(_viewport: ViewportMetrics): ContentCapture<TaskAnchor> {
      // the plane's address carries the project it shows: Back
      // returns to that project, a relaunch reopens it. The anchor is
      // the frame's: the published target the reading position sits
      // inside, named by its durable anchor; a target without an
      // anchor (a rail entry) cannot carry a return place and is
      // skipped, never guessed.
      const address: Address = { machine: PLANE_MACHINE, path: scope.frame().project ?? "" };
      const reading = presentations.readingAnchor();
      if (reading.kind === "default") return { address, position: { kind: "default" } };
      return {
        address,
        position: { kind: "anchor", anchor: { kind: "task", task: reading.anchor }, bounds: reading.bounds },
      };
    },
    async request(address: Address): Promise<RefreshOutcome> {
      // an explicit open (or a trail restore) ends any running follow
      // and starts one for the asked address once it has answered
      const intent = ++openIntent;
      follow.stop();
      followAddress = null;
      const outcome = await buffer.updates.refresh(async () => load(address));
      const shown = scope.frame().project;
      // a shown project follows even while unreachable: its list
      // returns when its tracker answers again
      if (!scope.isDestroyed() && intent === openIntent && shown !== null) {
        followAddress = { machine: PLANE_MACHINE, path: shown };
        follow.start(followAddress);
      }
      return outcome;
    },
    acceptsAnchor(data: AnchorData): data is TaskAnchor {
      return (
        typeof data === "object" &&
        data !== null &&
        !Array.isArray(data) &&
        (data as { kind?: unknown }).kind === "task" &&
        typeof (data as { task?: unknown }).task === "string" &&
        // the anchor must name a target of THIS frame's publication
        presentations.locateAnchor((data as { task: string }).task).kind !== "missing"
      );
    },
    revealAnchor(anchor: TaskAnchor): AnchorLocation {
      return presentations.locateAnchor(anchor.task);
    },
    defaultPosition(_viewport: ViewportMetrics): Point {
      return { left: 0, top: 0 };
    },
    subscribeChanges(_observer: () => void) {
      return () => {};
    },
  };

  const reading = presentations.readingContext([
    {
      action: {
        id: "tasks.add-project",
        label: "add a project (host:port/name)",
        run: () => add.enter(),
      },
      keys: [sequence(key("a"))],
    },
  ]);

  return {
    presentation: host,
    input: reading,
    places,
    release() {
      scope.destroy();
      follow.stop();
      stopRail();
      add.close();
    },
  };
}
