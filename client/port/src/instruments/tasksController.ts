// The Tasks instrument's controller (svelte-free): a project's task
// list - the "name/tasks" location names the project's tracker
// machine and the list - served as rows with identities, and the
// return places scenario 17 needs - the departed row's identity plus
// viewport offset, restored by Back. The rail beside it is the
// frame's (lib/frame/rail.ts), not this instrument's: the controller
// publishes its rows and nothing else.
//
// The pattern is the Board instrument's, end to end: the controller
// loads through the source/transport layer (TrackerSource's
// instrumentPage - fetched fresh every time, the pipeline is a view
// of the record's now), publishes the domain frame through the
// buffer's refresh pipeline, and never reads the DOM for identity
// (targets come from the page model; geometry only decides
// visibility and placement).
//
// The shown list follows the record (lib/instruments/follow.ts): a
// bounded poll re-reads the page and refreshes through the pipeline
// when its rows changed.
//
// The frame presents what this instrument publishes: select mode
// over the targets in view, the act menu, the click paths and the
// shared reading keys are the frame's (lib/frame/presentations.ts,
// through the buffer access's presentations face). This controller
// only PUBLISHES the targets - every row, with acts that resolve
// when asked (taskRowActs over the live page and the window
// services).
//
// Deliberately not here (recorded): find (a facility of a shown
// revision - a task list is not one), any row cursor (j/k is pure
// scrolling, the scenario 17 resolution), the rail and registration
// (the frame's).

import type { BufferAccess, InstrumentAttachment } from "../components/Buffer.ts";
import { createLoadScope, failureIssue } from "../lib/instruments/scaffold.ts";
import type { ContentFrame, TargetDefinition } from "../lib/contracts/buffer.ts";
import type {
  Point,
  PreparedContent,
  RefreshOutcome,
} from "../lib/contracts/buffer.ts";
import { createFollow } from "../lib/instruments/follow.ts";
import type {
  Address,
  ContentCapture,
  ContentPlaces,
  TaskAnchor,
} from "../lib/contracts/places.ts";
import type { AnchorData, AnchorLocation, ViewportMetrics } from "../lib/contracts/places.ts";
export type { TaskAnchor };
import type { WordsSnapshot } from "../lib/tasks/rail.ts";
import type { StructuredIssue } from "../lib/board/errors.ts";
import { TASKS_PATH, formatLocation } from "../lib/location.ts";
import { sourceFor, type Location } from "../lib/sources.ts";
import { TrackerSource } from "../lib/tracker.ts";
import { taskRowActs, type TaskPorts } from "../lib/tasks/menu.ts";
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
 * when the rows change. */
export type TasksData = Readonly<{
  /** The load that produced THIS frame; retained frames keep their
   * own token while a newer request is pending. */
  generation: number;
  /** The location the open ASKED for, formatted. */
  asked: string;
  /** The location that answered: the same list. */
  location: string;
  /** The project the list belongs to (the location's name). */
  project: string;
  /** The raw location the shown list was read at. */
  resolved: Location;
  /** invalid: the list could not be read (problem says why). */
  status: "loading" | "loaded" | "invalid";
  /** Why the list could not be read; null otherwise. */
  problem: string | null;
  page: TasksPageModel | null;
  errors: readonly StructuredIssue[];
}>;

/** The tasks controller's face to its own component. Select, menu,
 * clicks and the reading keys are the frame's (the buffer's
 * presentations face); the readout's selection fact forwards the
 * frame's own. Lifecycle lives on the buffer, not here. */
export interface TasksUI {
  /** The last label-selected row's task id, if any (identity rule:
   * the frame clears it when no published target carries the anchor
   * any more). */
  getSelection(): string | null;
  /** The frame rail's words feed (the rows' agent marks read the
   * words). The rail publication is the instrument's own. */
  subscribeWords(observer: (words: WordsSnapshot) => void): () => void;
}

/** How often a shown list re-reads its page (a named tunable; the
 * board follows at the same pace). */
export const TASKS_FOLLOW_POLL_MS = 3000;

const EMPTY_FRAME: TasksData = {
  generation: 0,
  asked: "",
  location: "",
  project: "",
  resolved: { machine: "tracker:", path: TASKS_PATH },
  status: "loading",
  problem: null,
  page: null,
  errors: [],
};

export function connectTasks(buffer: BufferAccess<TasksData>, ports: TaskPorts): InstrumentAttachment<TasksUI, TaskAnchor> {
  const scope = createLoadScope<TasksData>(EMPTY_FRAME);

  // The follow (lib/instruments/follow.ts): the shown list's page
  // content signature - the page has no revision, so its rows are
  // compared.
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
        return JSON.stringify(tasksPageOf(await source.instrumentPage(TASKS_PATH)));
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

  const actDeps = (): Parameters<typeof taskRowActs>[1] => ({
    machine: () => scope.frame().resolved?.machine ?? null,
    rowLink: (part) => taskRowOf(scope.frame().page ?? { groups: [] }, part)?.location ?? null,
    ports: () => ports,
  });

  const rowTargets = (page: TasksPageModel | null): TargetDefinition[] =>
    page === null
      ? []
      : tasksTargetsOf(page).map((row) => ({
          id: row.part,
          kind: row.kind,
          label: row.text,
          inside: null,
          clickable: true,
          acts: () => taskRowActs({ part: row.part, kind: row.kind, label: row.text }, actDeps()),
          anchor: row.task,
        }));

  const targetsSignature = (targets: readonly TargetDefinition[]): string =>
    JSON.stringify(targets.map((target) => [target.id, target.label, target.inside]));

  let publishedSignature: string | null = null;

  /** Publish a loaded frame with its targets: the domain data rides
   * the publication's value, the targets its own slot. */
  function publish(frame: TasksData): PreparedContent<ContentFrame<TasksData>> {
    scope.adopt(frame);
    const targets = rowTargets(frame.page);
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
    if (address.path !== TASKS_PATH) {
      // this surface serves a project's task list; the claims list
      // routes it here - a gap is named, never guessed
      return {
        kind: "failed",
        issue: failureIssue("asset-denied", `${asked} is not a project's task list; this surface serves task lists`),
      };
    }
    const project = address.machine.startsWith("tracker:") ? address.machine.slice("tracker:".length) : "";
    const base = (problem: string | null): TasksData => ({
      generation: token,
      asked: scope.asked(),
      location: asked,
      project,
      resolved: { machine: address.machine, path: TASKS_PATH },
      status: problem === null ? "loaded" : "invalid",
      problem,
      page: null,
      errors: [],
    });
    const problemOf = (error: unknown): string => (error instanceof Error ? error.message : String(error));
    let source: unknown;
    try {
      source = sourceFor(address.machine);
    } catch (error) {
      if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
      shownSignature = null;
      return publish(base(problemOf(error)));
    }
    if (!(source instanceof TrackerSource)) {
      shownSignature = null;
      return publish(base(`'${project}' is not a registered project's tracker`));
    }
    let viewPage;
    try {
      viewPage = await source.instrumentPage(TASKS_PATH);
    } catch (error) {
      if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
      // the project's tracker is unreachable: the list says why
      shownSignature = null;
      return publish(base(problemOf(error)));
    }
    if (scope.isStale(token)) return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
    const model = tasksPageOf(viewPage);
    shownSignature = JSON.stringify(model);
    return publish({
      generation: token,
      asked: scope.asked(),
      location: asked,
      project,
      resolved: { machine: address.machine, path: TASKS_PATH },
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
      // the address names the project's list: Back returns to it, a
      // relaunch reopens it. The anchor is the frame's: the published
      // target the reading position sits inside, named by its durable
      // anchor.
      const resolved = scope.frame().resolved;
      const address: Address = { machine: resolved.machine, path: resolved.path };
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
      // a shown list follows even while unreachable: it returns when
      // the project's tracker answers again
      if (!scope.isDestroyed() && intent === openIntent) {
        followAddress = address;
        follow.start(address);
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

  // The reading context is the frame's table alone: no extra keys.
  const reading = presentations.readingContext([]);

  // the frame rail's words, for the rows' agent marks: the door is
  // the window's injection (the rail is the frame's); absent, the
  // rows show no live marks
  const wordsObservers = new Set<(words: WordsSnapshot) => void>();
  const offWords = ports.railWords?.((words) => {
    for (const observer of [...wordsObservers]) observer(words);
  });

  return {
    presentation: {
      getSelection() {
        // the readout's selection fact is the frame's own: the anchor
        // of the label-selected target, cleared when no published
        // target carries the anchor any more (scenario 12's identity
        // rule)
        return buffer.presentations().selectedAnchor();
      },
      subscribeWords(observer) {
        wordsObservers.add(observer);
        return () => {
          wordsObservers.delete(observer);
        };
      },
    },
    input: reading,
    places,
    release() {
      offWords?.();
      scope.destroy();
      follow.stop();
    },
  };
}
