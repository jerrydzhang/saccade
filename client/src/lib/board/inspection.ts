// Machine-readable projection of the same view model the human UI renders.
// Deliberately small: it grows with actual component needs, not as an
// up-front API surface (t-56 record 2520).

import type { ViewerModel, ViewerState } from "./viewmodel.ts";
import type { Block } from "./model.ts";
import { SUPPORTED_GRAMMAR_VERSIONS } from "./decode.ts";
import type { CopyMode } from "../copy.ts";

export type InspectionOutline = {
  kind: string;
  title?: string;
  source: { line: number; column: number };
  children?: InspectionOutline[];
};

/** A rendered element an operator can act on through real input. The id
 * is the element's own stable data-control attribute (diagram frames
 * derive theirs from the unit id, so ids follow the document, not a
 * registry). Bounds are the live viewport rect at inspection time. */
export type InspectionTarget = {
  id: string;
  role: string;
  label: string;
  enabled: boolean;
  bounds: { x: number; y: number; width: number; height: number };
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
    /** The grammar version the open board declares; decode accepted it,
     * so it is always inside the supported range. */
    grammar: number | null;
    /** The grammar versions this viewer decodes; a tested claim pinned
     * against the shared crate's range over the shared fixtures. */
    supportedGrammarVersions: number[];
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
  /** The controls and diagrams actually rendered right now, addressed
   * by their stable ids: the same elements real input acts on. */
  targets: InspectionTarget[];
};

function targetOf(element: Element): InspectionTarget {
  const rect = element.getBoundingClientRect();
  const disabled =
    ("disabled" in element && (element as HTMLButtonElement).disabled === true) ||
    element.getAttribute("aria-disabled") === "true";
  return {
    id: element.getAttribute("data-control") ?? "",
    role: element.getAttribute("role") ?? element.tagName.toLowerCase(),
    label:
      element.getAttribute("aria-label") ??
      element.textContent?.trim().slice(0, 60) ??
      "",
    enabled: !disabled,
    bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
  };
}

/** Enumerate the actionable rendered elements in document order. Only
 * what is actually mounted is listed: an unexpanded diagram contributes
 * its frame, the expanded overlay its controls and stage. */
function projectedTargets(): InspectionTarget[] {
  const targets: InspectionTarget[] = [];
  for (const element of document.querySelectorAll("[data-control]")) {
    const target = targetOf(element);
    if (target.id) targets.push(target);
  }
  return targets;
}

function outlineOf(blocks: Block[]): InspectionOutline[] {
  return blocks.map((block) => {
    if (block.kind === "group" || block.kind === "incorporation") {
      return {
        kind: block.kind,
        ...(block.kind === "group" && block.title ? { title: block.title } : {}),
        ...(block.kind === "incorporation" ? { title: block.source.task } : {}),
        source: { line: block.pos.start.line, column: block.pos.start.column },
        children: outlineOf(block.content),
      };
    }
    return {
      kind: block.kind,
      ...(block.kind === "figure" ? { title: block.input } : {}),
      ...(block.kind === "unknown" ? { title: block.name } : {}),
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
      grammar: state.board?.grammar ?? null,
      supportedGrammarVersions: [...SUPPORTED_GRAMMAR_VERSIONS],
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
    targets: projectedTargets(),
  };
}
