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
import type { BoardData, RenderOutcome } from "../instruments/board.ts";
import type { StructuredIssue } from "./board/errors.ts";
import type { Board, Block } from "./board/model.ts";

export const PORT_SURFACE = "__saccadePort" as const;

export type PortObservation = {
  app: Readonly<{ name: "saccade-app-port"; slice: number }>;
  window: Readonly<{ label: string | null }>;
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

/** The shown board frame, extracted from the mapped buffer state: the
 * registry's own props carry the instrument's published domain data,
 * so the readout reads only what the window already holds. */
export function boardFrameOf(state: BufferState<RenderPair>): {
  frame: BoardData | null;
  status: PortObservation["board"]["status"];
} {
  if (state.kind === "destroyed") return { frame: null, status: "destroyed" };
  switch (state.content.kind) {
    case "empty":
      return { frame: null, status: "empty" };
    case "pending":
      return { frame: frameOfPair(state.content.previous), status: "loading" };
    case "available":
      return { frame: frameOfPair(state.content.value), status: "loaded" };
    case "unavailable":
      return { frame: null, status: "unavailable" };
  }
}

function frameOfPair(pair: RenderPair | null): BoardData | null {
  if (pair === null) return null;
  const content = pair.props["content"];
  return content === undefined ? null : (content as BoardData);
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

export function buildReadout(
  input: Readonly<{
    windowLabel: string | null;
    state: BufferState<RenderPair>;
    diagnostics: readonly DiagnosticEntryView[];
    doors: DoorsState;
    trail: PortObservation["trail"];
    viewport: PortObservation["viewport"];
    find: PortObservation["find"];
  }>,
): PortObservation {
  const { frame, status } = boardFrameOf(input.state);
  const board = frame?.board ?? null;
  return {
    app: { name: "saccade-app-port", slice: 1 },
    window: { label: input.windowLabel },
    board: {
      status,
      asked: frame ? frame.asked : null,
      location: frame ? frame.location : null,
      task: board && board.task.length > 0 ? board.task : null,
      pinnedRevision: pinnedRevisionOf(frame ? frame.location : null),
    },
    errors: frame ? frame.errors : [],
    render: frame ? frame.renderOutcomes : [],
    content:
      board && frame !== null && frame.status === "loaded"
        ? { parts: contentPartsOf(board) }
        : null,
    diagnostics: input.diagnostics,
    doors: input.doors,
    trail: input.trail,
    viewport: input.viewport,
    find: input.find,
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
