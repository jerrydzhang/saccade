// The viewer's single view model. The human UI and the machine-readable
// inspection projection read the same state; agent actions call the same
// methods the controls call. Plain class + listeners so pure tests and the
// Svelte app share it without a framework dependency in the core.

import type { Diagnostic, StructuredIssue } from "./errors.ts";
import { decodeBoard } from "./decode.ts";
import type { Board, FigureBlock } from "./model.ts";
import { sourceFor, SourceUnavailableError, type Location, type Machine } from "./sources.ts";
import type { CopyMode } from "./copy.ts";

export type ViewerStatus = "empty" | "loading" | "loaded" | "invalid";

export type RenderOutcome = {
  kind: "math" | "mermaid" | "figure" | "image" | "html" | "node";
  status: "ok" | "error" | "denied" | "missing" | "reported";
  message: string;
  line?: number;
};

export type ViewerState = {
  status: ViewerStatus;
  location: Location | null;
  board: Board | null;
  rawText: string | null;
  errors: StructuredIssue[];
  diagnostics: Diagnostic[];
  laneMessage: string | null;
  renderOutcomes: RenderOutcome[];
  renderGeneration: number | null;
  copyMode: CopyMode;
  theme: "system" | "light" | "dark";
};

export class ViewerModel {
  private state: ViewerState = {
    status: "empty",
    location: null,
    board: null,
    rawText: null,
    errors: [],
    diagnostics: [],
    laneMessage: null,
    renderOutcomes: [],
    renderGeneration: null,
    copyMode: "source",
    theme: "system",
  };

  /** Monotonic open generation: stale loads and hydrations are dropped
   * at their await boundaries, never written over a newer selection. */
  private generation = 0;

  private listeners = new Set<(state: ViewerState) => void>();

  subscribe(listener: (state: ViewerState) => void): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  get snapshot(): ViewerState {
    return this.state;
  }

  /** The current open generation; hydration and asset loads compare
   * against it after every await and drop themselves when stale. */
  get currentGeneration(): number {
    return this.generation;
  }

  private emit(): void {
    for (const listener of this.listeners) {
      listener(this.state);
    }
  }

  private set(patch: Partial<ViewerState>): void {
    this.state = { ...this.state, ...patch };
    this.emit();
  }

  /** Open an explicit location. Every lane and control funnels through here.
   * Stale responses (a slow A arriving after B was selected) are dropped:
   * the location on screen always names the board on screen. */
  async open(location: Location): Promise<void> {
    const token = ++this.generation;
    this.set({ status: "loading", location, errors: [], laneMessage: null });
    let text: string;
    try {
      const loaded = await sourceFor(location.machine).loadText(location);
      if (token !== this.generation) return;
      text = loaded.text;
    } catch (error) {
      if (token !== this.generation) return;
      const message = error instanceof SourceUnavailableError ? error.message : String(error);
      this.set({
        status: "invalid",
        board: null,
        rawText: null,
        errors: [{ code: "asset-missing", message }],
        laneMessage: message,
      });
      return;
    }
    const result = decodeBoard(text);
    if (token !== this.generation) return;
    if (result.status === "invalid") {
      this.set({
        status: "invalid",
        board: null,
        rawText: text,
        errors: result.errors,
        diagnostics: result.diagnostics,
        renderOutcomes: [],
        renderGeneration: null,
      });
      return;
    }
    this.set({
      status: "loaded",
      board: result.board,
      rawText: text,
      errors: [],
      diagnostics: result.board.diagnostics,
      renderOutcomes: [],
      renderGeneration: token,
    });
  }

  async loadFigureAsset(location: Location, block: FigureBlock): Promise<Uint8Array | string> {
    const token = this.generation;
    try {
      const asset = await sourceFor(location.machine).loadAsset(location, block.path);
      if (token !== this.generation) return "stale";
      return asset.bytes;
    } catch (error) {
      if (token !== this.generation) return "stale";
      if (error instanceof SourceUnavailableError) {
        if (error.code === "asset-denied") return `denied: ${error.message}`;
        return `missing: ${error.message}`;
      }
      return `missing: ${String(error)}`;
    }
  }

  /** Actual rendering outcomes, generation-scoped: hydration reports what
   * really happened per math expression, diagram, figure and image. The
   * inspection projection reads this, so a parsed board never masquerades
   * as a rendered one. */
  setRenderOutcomes(generation: number, outcomes: RenderOutcome[]): void {
    if (generation !== this.generation) return;
    this.set({ renderOutcomes: outcomes, renderGeneration: generation });
  }

  setCopyMode(mode: CopyMode): void {
    this.set({ copyMode: mode });
  }

  setTheme(theme: ViewerState["theme"]): void {
    this.set({ theme });
  }
}

export function machineLabel(machine: Machine): string {
  return machine === "fixture" ? "fixture" : "host";
}
