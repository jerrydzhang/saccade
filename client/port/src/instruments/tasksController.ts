// The Tasks instrument's controller (svelte-free): the tracker's
// pipeline page (the "tasks" location of a tracker location) served
// as rows with identities, label selection over visible rows, the
// task-row act menu, and the return places scenario 17 needs - the
// departed row's identity plus viewport offset, restored by Back.
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
// The control plane (t-48): beside the list, a rail of the projects
// this device has reached (switching opens a project's tasks here) and,
// under each, its running agents grouped by Herdr's word - the agents
// pages joined with the shell's Herdr word feed (lib/tasks/rail.ts).
// The rail is the instrument's chrome, published to its component
// beside the menu, never through the buffer's frames.
//
// The rail's entries are targets of the list's own selection and act
// menu (lib/tasks/rail.ts railTargetsOf, lib/tasks/menu.ts): s labels
// rows and rail entries together, and a click, a right-click or a
// label runs the same acts. Rows label first, so a row's label is the
// one it had before the rail.
//
// Deliberately not here (recorded): find (a facility of a shown
// revision - a task list is not one) and any row cursor (j/k is pure
// scrolling, the scenario 17 resolution).

import type { BufferAccess, InstrumentAttachment } from "../components/Buffer.ts";
import { createLoadScope, failureIssue, issueOf, readyFrame } from "../lib/instruments/scaffold.ts";
import { knownIssue } from "../lib/contracts/buffer.ts";
import type {
  KnownIssue,
  Point,
  PreparedContent,
  RefreshOutcome,
  Viewport,
} from "../lib/contracts/buffer.ts";
import type { InputContext } from "../lib/contracts/input.ts";
import { tasksReadingContext } from "../lib/input/contexts.ts";
import { createSelectSession, type SelectSessionView } from "../lib/select/session.ts";
import { createTasksMenu, tasksMenuLabel, type TaskPorts, type TasksMenu } from "../lib/tasks/menu.ts";
import { createFollow } from "../lib/instruments/follow.ts";
import type { MenuView } from "../lib/menu/session.ts";
import type {
  Address,
  ContentCapture,
  ContentPlaces,
  TaskAnchor,
} from "../lib/contracts/places.ts";
import type { ViewportMetrics } from "../lib/contracts/places.ts";
export type { TaskAnchor };
import type { AnchorData, AnchorLocation } from "../lib/contracts/places.ts";
import type { InstrumentProps } from "../lib/instruments/contract.ts";
import type { StructuredIssue } from "../lib/board/errors.ts";
import { SourceUnavailableError } from "../lib/board/errors.ts";
import { instrumentPath } from "../lib/location.ts";
import { formatLocation } from "../lib/location.ts";
import { sourceFor, type Location } from "../lib/sources.ts";
import { TrackerSource, forgetTracker, homeTracker, knownTrackers } from "../lib/tracker.ts";
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
import {
  taskRowOf,
  tasksPageOf,
  tasksTargetsOf,
  type TaskRow,
  type TaskTarget,
  type TasksPageModel,
} from "../lib/tasks/page.ts";

/** The tasks frame's published domain data. Carries no `surface`
 * field on purpose: the observation layer discriminates frames
 * structurally (a board frame names a board; this one names rows),
 * so the Board instrument's module stays untouched by this slice. */
export type TasksData = Readonly<{
  /** The load that produced THIS frame; retained frames keep their
   * own token while a newer request is pending. */
  generation: number;
  /** The location the open ASKED for, formatted. */
  asked: string;
  /** The location that answered (the tasks location of the tracker). */
  location: string;
  /** The raw location the shown frame was read at: the row acts
   * compose their specs against this machine. Null when nothing
   * resolved. */
  resolved: Location | null;
  status: "loading" | "loaded" | "invalid";
  page: TasksPageModel | null;
  /** The page's selectable targets: the rows, with their task ids and
   * links (TaskTarget is a SelectableTarget plus the act facts). */
  targets: readonly TaskTarget[];
  errors: readonly StructuredIssue[];
}>;

/** The tasks controller's face to its own component: the select and
 * menu sessions' live views, and the row act invocation paths.
 * Lifecycle lives on the buffer, not here. */
export interface TasksUI {
  subscribeSelect(observer: (view: SelectSessionView | null) => void): () => void;
  subscribeMenu(observer: (view: MenuView | null) => void): () => void;
  menuSession(): import("../lib/menu/session.ts").MenuSession;
  /** The last label-selected row's task id, if any (identity rule:
   * cleared when the row leaves the page). */
  getSelection(): string | null;
  /** A plain click on a row or a rail entry runs its FIRST act
   * directly - the same action object every invocation path runs
   * (scenario 9). */
  rowPrimaryClick(part: string): void;
  /** Open the act menu for a row or a rail entry (right-click). */
  openMenuFor(target: Readonly<{ part: string; kind: string; label: string }>, at?: { x: number; y: number }): void;
  /** The window services the row acts ride (the one open funnel and
   * the elsewhere surface); injected by the window, and no act is
   * offered before then. */
  setTaskPorts(ports: TaskPorts): void;
  /** The control plane's rail: projects and their running agents. */
  subscribeRail(observer: (rail: Rail) => void): () => void;
}

/** How often a shown list re-reads its page (a named tunable,
 * t-36@r64; the board follows at the same pace). */
export const TASKS_FOLLOW_POLL_MS = 3000;

const EMPTY_FRAME: TasksData = {
  generation: 0,
  asked: "",
  location: "",
  resolved: null,
  status: "loading",
  page: null,
  targets: [],
  errors: [],
};

/** Whether the page still carries a row naming this task. */
function lastFrameStillCarries(model: TasksPageModel, task: string): boolean {
  return model.groups.some((group) => group.rows.some((row) => row.task === task));
}

export function connectTasks(buffer: BufferAccess<TasksData>): InstrumentAttachment<TasksUI, TaskAnchor> {
  const scope = createLoadScope<TasksData>(EMPTY_FRAME);
  let resolved: Location | null = null;

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
        const source = sourceFor(address.machine);
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
      return buffer.updates.refresh(async () => {
        const prepared = await load(address);
        return prepared.kind === "ready" ? readyFrame(prepared.value) : prepared;
      });
    },
    // the DOM timer id and Node's Timeout meet across lib mixes; the
    // id is opaque to the follow module
    schedule: (ms, run) => setTimeout(run, ms) as unknown as number,
    cancel: (id) => clearTimeout(id as unknown as ReturnType<typeof setTimeout>),
  });

  const menuObservers = new Set<(view: MenuView | null) => void>();

  // -- the control plane's rail (t-48) ---------------------------------
  const railObservers = new Set<(rail: Rail) => void>();
  let words: WordsSnapshot = NO_WORDS;
  let agentPages: { name: string; agents: AgentFacts[] | null; problem: string | null }[] = [];
  let railRunning = false;
  let railTimer: ReturnType<typeof setTimeout> | null = null;

  function currentProject(): string | null {
    const machine = scope.frame().resolved?.machine ?? null;
    return machine !== null && machine.startsWith("tracker:") ? machine.slice("tracker:".length) : null;
  }

  function railNow(): Rail {
    // every reached project is listed, read or not yet
    const names = new Set([...Object.keys(knownTrackers()), ...agentPages.map((entry) => entry.name)]);
    const current = currentProject();
    if (current !== null) names.add(current);
    const projects = [...names].map(
      (name) => agentPages.find((entry) => entry.name === name) ?? { name, agents: null, problem: null },
    );
    return railOf(projects, words, current, homeTracker());
  }

  function notifyRail(): void {
    const rail = railNow();
    for (const observer of [...railObservers]) observer(rail);
  }

  async function readAgentPages(): Promise<void> {
    const names = Object.keys(knownTrackers());
    const current = currentProject();
    if (current !== null && !names.includes(current)) names.push(current);
    agentPages = await Promise.all(
      names.map(async (name) => {
        try {
          const source = sourceFor(`tracker:${name}`);
          if (!(source instanceof TrackerSource)) return { name, agents: null, problem: "not a tracker" };
          return { name, agents: agentFactsOf(await source.agentsPage()), problem: null };
        } catch (error) {
          return { name, agents: null, problem: error instanceof Error ? error.message : String(error) };
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

  // -- the row acts' window services -----------------------------------
  let ports: TaskPorts | null = null;

  function machine(): string | null {
    return scope.frame().resolved?.machine ?? null;
  }

  function menuView(): MenuView | null {
    if (!tasksMenu.session.isOpen()) return null;
    const target = tasksMenu.session.target();
    if (target === null) return null;
    const acts = tasksMenu.session.acts();
    return {
      open: true,
      label: target.label,
      acts: acts.map((act) => ({ key: act.key, label: act.action.label })),
      at: tasksMenu.session.at(),
      part: target.part,
    };
  }

  function notifyMenu(): void {
    const view = menuView();
    for (const observer of [...menuObservers]) observer(view);
  }

  function railTarget(part: string): RailTarget | undefined {
    return railTargetsOf(railNow()).find((target) => target.part === part);
  }

  /** Every selectable target: the rows, then the rail's entries. */
  function allTargets(): readonly (TaskTarget | RailTarget)[] {
    return [...scope.frame().targets, ...railTargetsOf(railNow())];
  }

  function forget(project: string): void {
    if (forgetTracker(project) !== "forgotten") return;
    agentPages = agentPages.filter((entry) => entry.name !== project);
    notifyRail();
  }

  const tasksMenu: TasksMenu = createTasksMenu({
    machine,
    rowLink: (part) => taskRowOf(scope.frame().page ?? { groups: [] }, part)?.location ?? null,
    railTarget,
    forget,
    ports: () => ports,
    capture: (context) => buffer.interactions.capture(context),
  });
  tasksMenu.setOnNotify(() => notifyMenu());

  // -- the select session (label selection over visible rows) ----------
  // The selection's identity is the TASK (scenario 12's identity
  // rules on a task list): a positional part path changes meaning
  // across a reorder, the task does not. A selection whose task
  // leaves the page clears - never a silent neighbour.
  let selectedTask: string | null = null;
  const selectSession = createSelectSession({
    targets: allTargets,
    visible(target) {
      const metrics = buffer.viewport.metrics();
      const viewTop = metrics.position.top;
      const viewBottom = metrics.position.top + metrics.height;
      const bounds = buffer.geometry.bounds(target.part);
      if (bounds === null) return false;
      return bounds.top < viewBottom && bounds.bottom > viewTop;
    },
    capture: (context) => buffer.interactions.capture(context),
    onSelected(target) {
      if (target.kind === "task") selectedTask = (target as TaskTarget).task;
      // scenario 9: selection opens the act menu
      const label = tasksMenuLabel({ kind: target.kind, label: target.text }, railTarget(target.part));
      tasksMenu.openFor({ part: target.part, kind: target.kind, label });
    },
  });

  const host: TasksUI = {
    subscribeSelect(observer) {
      return selectSession.subscribe(observer);
    },
    subscribeMenu(observer) {
      menuObservers.add(observer);
      observer(menuView());
      return () => {
        menuObservers.delete(observer);
      };
    },
    menuSession() {
      return tasksMenu.session;
    },
    getSelection() {
      return selectedTask;
    },
    rowPrimaryClick(part) {
      const target = allTargets().find((entry) => entry.part === part);
      if (target === undefined) return;
      tasksMenu.session.primaryClick({ kind: target.kind, label: target.text, part });
      notifyMenu();
    },
    openMenuFor(target, at) {
      const rail = target.kind === "task" ? undefined : railTarget(target.part);
      tasksMenu.openFor({ ...target, label: tasksMenuLabel(target, rail) }, at);
    },
    setTaskPorts(injected) {
      ports = injected;
    },
    subscribeRail(observer) {
      railObservers.add(observer);
      observer(railNow());
      return () => {
        railObservers.delete(observer);
      };
    },
  };

  async function load(address: Address): Promise<PreparedContent<TasksData>> {
    if (scope.isDestroyed())
      return {
        kind: "failed",
        issue: failureIssue("lane-unavailable", "the buffer was destroyed"),
      };
    const token = scope.begin();
    const asked = formatLocation({ machine: address.machine, path: address.path });
    scope.setAsked(asked);
    const location: Location = { machine: address.machine, path: address.path };
    const page = instrumentPath(location.path);
    if (page !== "tasks") {
      // this buffer serves the tasks page; other entry pages are other
      // instruments (the agents page is a later slice) - the gap is
      // named, never guessed
      const message =
        page === null
          ? `${location.path} is not an instrument page; this surface serves a tracker's tasks page`
          : `the ${page} page has no instrument in this build yet`;
      return { kind: "failed", issue: failureIssue("asset-denied", message) };
    }
    let source: unknown;
    try {
      source = sourceFor(location.machine);
    } catch (error) {
      if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
      resolved = null;
      return { kind: "failed", issue: issueOf(error) };
    }
    if (!(source instanceof TrackerSource)) {
      return {
        kind: "failed",
        issue: failureIssue("asset-denied", `'${location.machine}' is not a tracker`),
      };
    }
    let viewPage;
    try {
      viewPage = await source.instrumentPage("tasks");
    } catch (error) {
      if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
      resolved = null;
      return { kind: "failed", issue: issueOf(error) };
    }
    if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
    const model = tasksPageOf(viewPage);
    resolved = location;
    shownSignature = JSON.stringify(model);
    // scenario 12's identity rule: a selection whose task left the
    // page clears - never a silent neighbour
    if (selectedTask !== null && !lastFrameStillCarries(model, selectedTask)) {
      selectedTask = null;
    }
    startRail();
    scope.adopt({
      generation: token,
      asked: scope.asked(),
      location: formatLocation(location),
      resolved: location,
      status: "loaded",
      page: model,
      targets: tasksTargetsOf(model),
      errors: [],
    });
    // the shown project moves the rail's mark
    notifyRail();
    return { kind: "ready", value: scope.frame() };
  }

  const places: ContentPlaces<TaskAnchor> = {
    captureDisplayed(viewport: ViewportMetrics): ContentCapture<TaskAnchor> {
      const address: Address = resolved
        ? { machine: resolved.machine, path: resolved.path }
        : { machine: "", path: "" };
      // The reading anchor (the hybrid default, r49): the row the
      // reading position sits inside, else the default position.
      // Geometry is live measurement; a row with no measurable
      // presentation is skipped, never guessed.
      const viewTop = viewport.position.top;
      for (const target of scope.frame().targets) {
        const bounds = buffer.geometry.bounds(target.part);
        if (bounds === null) continue;
        if (viewTop >= bounds.top && viewTop < bounds.bottom) {
          const row = taskRowOf(scope.frame().page ?? { groups: [] }, target.part);
          if (row === undefined) continue;
          return {
            address,
            position: { kind: "anchor", anchor: { kind: "task", task: row.task }, bounds },
          };
        }
      }
      return { address, position: { kind: "default" } };
    },
    async request(address: Address): Promise<RefreshOutcome> {
      // an explicit open (or a trail restore) ends any running follow
      // and starts one for the asked address once it has answered
      const intent = ++openIntent;
      follow.stop();
      followAddress = null;
      const outcome = await buffer.updates.refresh(async () => {
        const prepared = await load(address);
        return prepared.kind === "ready" ? readyFrame(prepared.value) : prepared;
      });
      if (!scope.isDestroyed() && intent === openIntent && address.machine.startsWith("tracker:")) {
        followAddress = { ...address };
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
        // the anchor must name a row of THIS frame's page
        scope.frame().targets.some((target) => target.task === (data as { task: string }).task)
      );
    },
    revealAnchor(anchor: TaskAnchor): AnchorLocation {
      const target = scope.frame().targets.find(
        (entry) => entry.task === anchor.task,
      );
      if (target === undefined) return { kind: "missing" };
      const bounds = buffer.geometry.bounds(target.part);
      if (bounds === null) return { kind: "pending" };
      return { kind: "found", bounds };
    },
    defaultPosition(_viewport: ReturnType<Viewport["metrics"]>): Point {
      return { left: 0, top: 0 };
    },
    subscribeChanges(_observer: () => void) {
      return () => {};
    },
  };

  const reading: InputContext = tasksReadingContext({
    scrollDown: () => buffer.viewport.scrollLines(1),
    scrollUp: () => buffer.viewport.scrollLines(-1),
    scrollFraction: (direction, fraction) => buffer.viewport.scrollFraction(direction, fraction),
    scrollToTop: () => buffer.viewport.start(),
    scrollToBottom: () => buffer.viewport.end(),
    enterSelect: () => selectSession.enter(),
  });

  return {
    presentation: host,
    input: reading,
    places,
    release() {
      scope.destroy();
      follow.stop();
      stopRail();
      selectSession.exit();
      tasksMenu.session.outsidePointerDown();
    },
  };
}
