// The port's observation contract (t-36 slice 1): what a booted port
// exposes about already-existing state, for humans and for the
// harness's app-side adapter. Authored fresh - it projects the port's
// own model and doors, never the old app's inspection() shape or its
// selectors; old-app selector names are harness-adapter material, not
// contract.
//
// The surface is `window.__saccadePort.readout()`: a machine-readable
// projection of the same state the human UI renders. It is read-only:
// no setter, no test-only state, no product file edits. A field a
// build cannot produce is null - a check that needs it stops; it never
// guesses.
//
// Slice-1 shape (this tree's first bootable build):
// - app/window: identity and the native window label.
// - board: the shown location and its status - the location the open
//   ASKED for, the location that answered, the authored task reference
//   and the pinned revision when the path carries one.
// - errors: the structured issues the shown frame carries.
// - render: what actually happened per math expression, diagram,
//   figure and image - parsed never masquerades as rendered.
// - content: the decoded board's structural parts with their canonical
//   content paths, from the model - domain content is not derived from
//   HTML attributes. The rendered DOM mirrors these as data-port-part /
//   data-port-kind attributes for element-addressed observation.
// - diagnostics: the in-memory ring (the minimal code+message
//   catalogue).
// - doors: the suite doors the shared shell evals into every window -
//   the jump chip this window shows and the delivered open specs, with
//   whether each opened through the port's own model.

import type { BufferState } from "./contracts/buffer.ts";
import type { RenderPair } from "./instruments/registry.ts";
import type { BoardData, BoardUI, RenderOutcome } from "../instruments/board.ts";
import type { TasksData, TasksUI } from "../instruments/tasks.ts";
import type { StructuredIssue } from "./board/errors.ts";
import type { Board, Block } from "./board/model.ts";
import type { TaskRow } from "./tasks/page.ts";

export const PORT_SURFACE = "__saccadePort" as const;

export type PortObservation = {
  app: Readonly<{ name: "saccade-app"; slice: number }>;
  /** The surface whose buffer the window shows: the other surface's
   * retained buffer keeps its state and re-activates on its trail
   * return. */
  surface: "board" | "tasks";
  window: Readonly<{
    label: string | null;
    /** The live window carrying this window's link (the o act's
     * remembered receiver), or null (S18). */
    link: string | null;
    /** A live ask: the act that asked, the spec, the answerable
     * chips (S18). */
    ask: Readonly<{ act: string; spec: string; chips: readonly { window: string; chip: string }[] }> | null;
    /** Whether this window's STATE is bound to the store (the suite
     * worker's base defect 1: the acts' bounded wait and the tests'
     * stopgap both read this instead of log lines). */
    bound: boolean;
    /** GTK's own focus facts for this window (the shell's
     * window_focus_state), refreshed in the background; null with
     * no bridge or before the first answer (m-537). */
    focus: Readonly<{
      acceptsFocus: boolean;
      focusOnMap: boolean;
      active: boolean;
      /** The GTK focus widget's type ("WebKitWebView" when keys reach
       * the page), or null when the window has no focus widget. */
      focusWidget: string | null;
    }> | null;
  }>;
  board: Readonly<{
    status: "empty" | "loading" | "loaded" | "invalid" | "unavailable" | "destroyed";
    asked: string | null;
    location: string | null;
    task: string | null;
    pinnedRevision: number | null;
  }>;
  errors: readonly StructuredIssue[];
  render: readonly RenderOutcome[];
  content: Readonly<{ parts: readonly ContentPart[] }> | null;
  diagnostics: readonly DiagnosticEntryView[];
  doors: Readonly<{
    jumpChip: string | null;
    delivered: readonly DoorDelivery[];
  }>;
  /** The window's trail (slice 2): explicit openings only, the
   * cursor names the current entry. */
  trail: Readonly<{
    depth: number;
    cursor: number;
    entries: readonly { location: string; hasDeparture: boolean }[];
    canBack: boolean;
    canForward: boolean;
  }>;
  /** The attached viewport's live metrics. */
  viewport: Readonly<{ left: number; top: number; width: number; height: number }>;
  /** The find facility of the shown revision (scenario 16). */
  find: Readonly<{ open: boolean; query: string; count: number; active: number }>;
  /** The last reference-copy act's payload (scenario 13): the text
   * the ACTION produced (and attempted on the clipboard) - the
   * observation surface, like find; the clipboard itself is an
   * environment capability, never the source of truth. */
  copy: Readonly<{ payload: string; at: number }> | null;
  /** The figure inspection surface (S04): the inspected part and its
   * zoom while open, null when closed (the temporary surface adds no
   * trail entry; the board beneath keeps its updates). */
  inspection: Readonly<{ part: string; zoom: number; x: number; y: number }> | null;
  /** The board surface's follow state (the S15 diagnosis): whether
   * the follow-current poll is armed and for which task. */
  follow: Readonly<{ armed: boolean; task: number | null; address: string | null; requests: readonly string[] }> | null;
  /** The board view's own work since it was created (m-547 (5), the
   * mechanism the performance tests read): how many times it rendered
   * the board, and how many units (math, diagrams, figures, images) it
   * hydrated. A fold, a find, a selection must move neither. */
  presentation?: Readonly<{ renders: number; hydrated: number }> | null;
  /** The input surface (batch 3's contract): the pending strokes,
   * the which-key popup's content while shown, and the NAMED
   * tunables (read by the scenarios, never pinned there). */
  input: Readonly<{
    pending: readonly string[];
    whichKey: Readonly<{
      prefix: string;
      entries: readonly { keys: string; label: string; actionId: string }[];
    }> | null;
    tunables: Readonly<{
      whichKeyDelayMs: number;
      whichKeySuspendsTimeout: boolean;
      sequenceTimeoutMs: number;
    }>;
  }>;
  /** The window's board reading layout (lib/board/reading-layout.ts):
   * the text zoom and the column-width scale. */
  reading: Readonly<{ zoom: number; width: number }>;
  /** The tasks surface's shown page (the pipeline): the rows with
   * their tasks and links, the label selection, and the groups'
   * titles. Null on every other surface. */
  tasks: Readonly<{
    status: "empty" | "loading" | "loaded" | "invalid" | "unavailable" | "destroyed";
    asked: string | null;
    location: string | null;
    rows: readonly {
      part: string;
      task: string;
      location: string;
      title: string;
    }[];
    selection: string | null;
    groups: readonly { part: string; title: string }[];
  }> | null;
};

export type ContentPart = Readonly<{
  /** Canonical content path within the board: /content/0, nested as
   * /content/2/content/1. */
  part: string;
  kind: string;
  line: number | null;
}>;

export type DiagnosticEntryView = Readonly<{
  id: string;
  bufferId: string;
  code: string;
  message: string;
}>;

export type DoorDelivery = Readonly<{
  spec: string;
  opened: boolean;
  detail: string | null;
}>;

export type DoorsState = Readonly<{
  jumpChip: string | null;
  delivered: readonly DoorDelivery[];
}>;

/** The shown frame, extracted from the mapped buffer state: the
 * registry's own props carry the instrument's published domain data,
 * so the readout reads only what the window already holds. The
 * frames discriminate structurally - a board frame names a board, a
 * tasks frame names rows - so the Board instrument's module stays
 * untouched by the tasks slice. */
export function boardFrameOf(state: BufferState<RenderPair>): {
  frame: BoardData | null;
  status: PortObservation["board"]["status"];
} {
  const frame = frameOfPair(state);
  // the frames discriminate structurally: a board frame names a board
  return { frame: frame !== null && "board" in frame ? frame : null, status: statusOf(state) };
}

export function tasksFrameOf(state: BufferState<RenderPair>): {
  frame: TasksData | null;
  status: NonNullable<PortObservation["tasks"]>["status"];
} {
  const frame = frameOfPair(state);
  // a tasks frame names rows (the page model)
  return { frame: frame !== null && "page" in frame ? (frame as TasksData) : null, status: statusOf(state) };
}

function statusOf(state: BufferState<RenderPair>): "empty" | "loading" | "loaded" | "invalid" | "unavailable" | "destroyed" {
  if (state.kind === "destroyed") return "destroyed";
  switch (state.content.kind) {
    case "empty": return "empty";
    case "pending": return "loading";
    case "available": return "loaded";
    case "unavailable": return "unavailable";
  }
}

function frameOfPair(state: BufferState<RenderPair>): BoardData | TasksData | null {
  if (state.kind !== "alive") return null;
  const content = state.content;
  const pair = content.kind === "available" ? content.value
    : content.kind === "pending" ? content.previous : null;
  if (pair === null) return null;
  const value = pair.props["content"];
  return value === undefined ? null : (value as BoardData | TasksData);
}

/** The pinned revision a shown path carries: t-N@rK pins, bare t-N
 * stays null (it follows the tracker's current commits). */
export function pinnedRevisionOf(path: string | null): number | null {
  if (path === null) return null;
  const match = /@r(\d+)$/.exec(path.trim());
  return match && match[1] !== undefined ? Number(match[1]) : null;
}

/** The board's structural parts with canonical content paths, walked
 * from the decoded model - ground truth, never DOM-derived. */
export function contentPartsOf(board: Board): readonly ContentPart[] {
  const parts: ContentPart[] = [];
  const walk = (blocks: readonly Block[], prefix: string): void => {
    blocks.forEach((block, index) => {
      const part = `${prefix}/content/${index}`;
      parts.push({
        part,
        kind: block.kind,
        line: block.pos.start.line,
      });
      if (block.kind === "group" || block.kind === "incorporation") {
        walk(block.content, part);
      }
    });
  };
  walk(board.content, "");
  return parts;
}

function findViewOf(state: BufferState<RenderPair>): PortObservation["find"] {
  const content = state.kind === "alive" ? state.content : null;
  const pair = content?.kind === "available" ? content.value
    : content?.kind === "pending" ? content.previous : null;
  const controller = pair?.props["controller"] as Pick<BoardUI, "findView"> | undefined;
  if (typeof controller?.findView !== "function") return { open: false, query: "", count: 0, active: -1 };
  // the contract's four fields only: the view also carries the mark
  // plan (a Map), which is presentation, not observation
  const { open, query, count, active } = controller.findView();
  return { open, query, count, active };
}

export function tasksViewOf(
  frame: TasksData | null,
  state: BufferState<RenderPair>,
): NonNullable<PortObservation["tasks"]> {
  const rows: { part: string; task: string; location: string; title: string }[] = [];
  const groups: { part: string; title: string }[] = [];
  const page = frame?.page ?? null;
  if (page !== null) {
    for (const group of page.groups) {
      groups.push({ part: group.part, title: group.title });
      for (const row of group.rows) {
        rows.push({ part: row.part, task: row.task, location: row.location, title: row.title });
      }
    }
  }
  const controller = (state.kind === "alive" ? state.content : null);
  const pair = controller?.kind === "available" ? controller.value
    : controller?.kind === "pending" ? controller.previous : null;
  const face = pair?.props["controller"] as Pick<TasksUI, "getSelection"> | undefined;
  const selection = typeof face?.getSelection === "function" ? face.getSelection() : null;
  return {
    status: statusOf(state),
    asked: frame ? frame.asked : null,
    location: frame ? frame.location : null,
    rows,
    selection,
    groups,
  };
}

export function buildReadout(
  input: Readonly<{
    surface: "board" | "tasks";
    windowLabel: string | null;
    link: string | null;
    bound: boolean;
    focus?: PortObservation["window"]["focus"];
    ask: Readonly<{ act: string; spec: string; chips: readonly { window: string; chip: string }[] }> | null;
    state: BufferState<RenderPair>;
    diagnostics: readonly DiagnosticEntryView[];
    doors: DoorsState;
    trail: PortObservation["trail"];
    viewport: PortObservation["viewport"];
    copy: PortObservation["copy"];
    inspection: PortObservation["inspection"];
    follow: PortObservation["follow"];
    presentation?: PortObservation["presentation"];
    input: PortObservation["input"];
    reading: PortObservation["reading"];
  }>,
): PortObservation {
  const { frame, status } = boardFrameOf(input.state);
  const board = frame?.board ?? null;
  const tasksFrame = input.surface === "tasks" ? tasksFrameOf(input.state).frame : null;
  return {
    app: { name: "saccade-app", slice: 1 },
    surface: input.surface,
    window: { label: input.windowLabel, link: input.link, ask: input.ask, bound: input.bound, focus: input.focus ?? null },
    board: {
      status: input.surface === "board" ? status : "empty",
      asked: frame && input.surface === "board" ? frame.asked : null,
      location: frame && input.surface === "board" ? frame.location : null,
      task: board && board.task.length > 0 ? board.task : null,
      pinnedRevision: pinnedRevisionOf(frame && input.surface === "board" ? frame.location : null),
    },
    errors: frame && input.surface === "board" ? frame.errors : [],
    render: frame && input.surface === "board" ? frame.renderOutcomes : [],
    content:
      board && frame !== null && frame.status === "loaded" && input.surface === "board"
        ? { parts: contentPartsOf(board) }
        : null,
    diagnostics: input.diagnostics,
    doors: input.doors,
    trail: input.trail,
    viewport: input.viewport,
    find: input.surface === "board" ? findViewOf(input.state) : { open: false, query: "", count: 0, active: -1 },
    copy: input.copy,
    inspection: input.inspection,
    follow: input.follow,
    presentation: input.presentation ?? null,
    input: input.input,
    reading: input.reading,
    tasks: input.surface === "tasks"
      ? tasksViewOf(tasksFrame, input.state)
      : null,
  };
}

/** The native window label, read from the shell's own metadata. */
export function nativeWindowLabel(): string | null {
  const internals = (
    window as unknown as {
      __TAURI_INTERNALS__?: { metadata?: { currentWindow?: { label?: unknown } } };
    }
  ).__TAURI_INTERNALS__;
  const label = internals?.metadata?.currentWindow?.label;
  return typeof label === "string" ? label : null;
}
