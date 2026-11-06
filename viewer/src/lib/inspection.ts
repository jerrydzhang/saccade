// Machine-readable projection of the same view model the human UI renders.
// Deliberately small: it grows with actual component needs, not as an
// up-front API surface (t-56 record 2520).

import type { ViewerModel, ViewerState } from "./viewmodel.ts";
import type { Block } from "./model.ts";
import type { CopyMode } from "./copy.ts";

export type InspectionOutline = {
  kind: string;
  title?: string;
  source: { line: number; column: number };
  children?: InspectionOutline[];
};

export type Inspection = {
  resource: {
    machine: string | null;
    path: string | null;
    status: ViewerState["status"];
  };
  identity: {
    /** Authored metadata; never a tracker association. */
    task: string | null;
    rawLength: number | null;
  };
  errors: ViewerState["errors"];
  diagnostics: ViewerState["diagnostics"];
  laneMessage: string | null;
  /** Actual rendering outcomes for the loaded generation: per math
   * expression, diagram, figure and image - ok/error/denied/missing.
   * A parsed board never masquerades as a rendered one. */
  renderOutcomes: ViewerState["renderOutcomes"];
  renderGeneration: ViewerState["renderGeneration"];
  copyMode: CopyMode;
  theme: ViewerState["theme"];
  outline: InspectionOutline[];
  actions: string[];
};

function outlineOf(blocks: Block[]): InspectionOutline[] {
  return blocks.map((block) => {
    if (block.kind === "group" || block.kind === "incorporation") {
      return {
        kind: block.kind,
        ...(block.kind === "group" && block.title ? { title: block.title } : {}),
        ...(block.kind === "incorporation" ? { title: block.id } : {}),
        source: { line: block.pos.start.line, column: block.pos.start.column },
        children: outlineOf(block.content),
      };
    }
    return {
      kind: block.kind,
      ...(block.kind === "figure" ? { title: block.path } : {}),
      source: { line: block.pos.start.line, column: block.pos.start.column },
    };
  });
}

export function inspect(model: ViewerModel): Inspection {
  const state = model.snapshot;
  return {
    resource: {
      machine: state.location?.machine ?? null,
      path: state.location?.path ?? null,
      status: state.status,
    },
    identity: {
      task: state.board?.task ?? null,
      rawLength: state.rawText?.length ?? null,
    },
    errors: state.errors,
    diagnostics: state.diagnostics,
    laneMessage: state.laneMessage,
    renderOutcomes: state.renderOutcomes,
    renderGeneration: state.renderGeneration,
    copyMode: state.copyMode,
    theme: state.theme,
    outline: state.board ? outlineOf(state.board.content) : [],
    actions: ["open", "setCopyMode", "setTheme", "search"],
  };
}
