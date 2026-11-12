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
// Deliberately not here (recorded): live reordering mid-display (the
// page re-resolves on every open; a standing poll is the S12 live
// clause, a differential-adapter matter), the agents page (a later
// instrument), find (a facility of a shown revision - a task list is
// not one), and any row cursor (j/k is pure scrolling, the scenario
// 17 resolution).

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
import { createTasksMenu, type TaskPorts, type TasksMenu } from "../lib/tasks/menu.ts";
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
import { TrackerSource } from "../lib/tracker.ts";
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
  /** A plain click on a row runs its FIRST act directly - the same
   * action object every invocation path runs (scenario 9). */
  rowPrimaryClick(part: string): void;
  /** Open the act menu for a row (right-click). */
  openMenuFor(target: Readonly<{ part: string; kind: string; label: string }>, at?: { x: number; y: number }): void;
  /** The window services the row acts ride (the one open funnel and
   * the elsewhere surface); injected by the window, and no act is
   * offered before then. */
  setTaskPorts(ports: TaskPorts): void;
}

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

  const menuObservers = new Set<(view: MenuView | null) => void>();

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
      at: null,
    };
  }

  function notifyMenu(): void {
    const view = menuView();
    for (const observer of [...menuObservers]) observer(view);
  }

  const tasksMenu: TasksMenu = createTasksMenu({
    machine,
    rowLink: (part) => taskRowOf(scope.frame().page ?? { groups: [] }, part)?.location ?? null,
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
    targets: () => scope.frame().targets,
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
      selectedTask = (target as TaskTarget).task;
      // scenario 9: selection opens the act menu
      tasksMenu.openFor({ part: target.part, kind: target.kind, label: target.text });
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
      const target = scope.frame().targets.find((entry) => entry.part === part);
      if (target === undefined) return;
      tasksMenu.session.primaryClick({ kind: target.kind, label: target.text, part });
      notifyMenu();
    },
    openMenuFor(target, at) {
      tasksMenu.openFor(target, at);
    },
    setTaskPorts(injected) {
      ports = injected;
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
    // scenario 12's identity rule: a selection whose task left the
    // page clears - never a silent neighbour
    if (selectedTask !== null && !lastFrameStillCarries(model, selectedTask)) {
      selectedTask = null;
    }
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
      return buffer.updates.refresh(async () => {
        const prepared = await load(address);
        return prepared.kind === "ready" ? readyFrame(prepared.value) : prepared;
      });
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
      selectSession.exit();
      tasksMenu.session.outsidePointerDown();
    },
  };
}
