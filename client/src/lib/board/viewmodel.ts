// The viewer's single view model. The human UI and the machine-readable
// inspection projection read the same state; agent actions call the same
// methods the controls call. Plain class + listeners so pure tests and the
// Svelte app share it without a framework dependency in the core.

import type { Diagnostic, StructuredIssue } from "./errors.ts";
import { decodeBoard } from "./decode.ts";
import type { Board, FigureBlock } from "./model.ts";
import { crateVerdict } from "./validate.ts";
import {
  sourceFor,
  SourceUnavailableError,
  type Location,
  type Machine,
} from "../sources.ts";
import { parseLocation } from "../location.ts";
import type { CopyMode } from "../copy.ts";

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
    this.set({
      status: "loading",
      location,
      board: null,
      rawText: null,
      errors: [],
      diagnostics: [],
      laneMessage: null,
      renderOutcomes: [],
      renderGeneration: null,
    });
    let text: string;
    let resolvedLocation: Location = location;
    try {
      const loaded = await sourceFor(location.machine).loadText(location);
      if (token !== this.generation) return;
      text = loaded.text;
      // a bare tracker location resolved: the board on screen is the
      // revision that answered, and its figures read the same tree
      resolvedLocation = loaded.location;
    } catch (error) {
      if (token !== this.generation) return;
      const message =
        error instanceof SourceUnavailableError ? error.message : String(error);
      this.set({
        status: "invalid",
        board: null,
        rawText: null,
        errors: [{ code: "asset-missing", message }],
        laneMessage: message,
      });
      return;
    }
    const verdict = await this.validate(text);
    if (token !== this.generation) return;
    const result = decodeBoard(text);
    if (token !== this.generation) return;

    // The crate's refusal is final in the native lanes; the presentation
    // decoder never overrides it.
    if (verdict && verdict.verdict === "refused") {
      const issue: StructuredIssue = {
        code: "board-refused",
        message: verdict.message,
      };
      if (verdict.line !== undefined) {
        issue.line = verdict.line;
        issue.column = verdict.column ?? 0;
      }
      this.set({
        status: "invalid",
        board: null,
        rawText: text,
        errors: [issue],
        diagnostics: [],
        renderOutcomes: [],
        renderGeneration: null,
      });
      return;
    }
    if (result.status === "invalid") {
      if (verdict && verdict.verdict === "ok") {
        // The mirror refused text the crate accepted: a decoder bug.
        // Never render an unvalidated board; refuse with both accounts.
        const first = result.errors[0];
        const message = `decoder disagreement: the grammar validator accepted the board, the presentation decoder refused it (${first?.code}: ${first?.message})`;
        this.set({
          status: "invalid",
          board: null,
          rawText: text,
          errors: [{ code: "decoder-mismatch", message }, ...result.errors],
          diagnostics: [],
          renderOutcomes: [],
          renderGeneration: null,
          laneMessage: message,
        });
        return;
      }
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
      location: resolvedLocation,
      board: result.board,
      rawText: text,
      errors: [],
      diagnostics: result.board.diagnostics,
      renderOutcomes: [],
      renderGeneration: token,
    });
  }

  /** The native gate: admission through the shared board-grammar crate.
   * Null in the browser lane, where the mirror decoder alone decides. A
   * bridge that fails to answer refuses the board loudly; it never
   * degrades silently to mirror-only admission. */
  private async validate(
    text: string,
  ): Promise<Awaited<ReturnType<typeof crateVerdict>>> {
    try {
      return await crateVerdict(text);
    } catch (error) {
      const message =
        error instanceof SourceUnavailableError ? error.message : String(error);
      return { verdict: "refused", kind: "bridge-invalid", message };
    }
  }

  /** Parse an explicit location string and open it; a string that is
   * not a location is a loud structured refusal, never a guess. */
  async openInput(raw: string): Promise<void> {
    const parsed = parseLocation(raw);
    if (!parsed) {
      const message = `not a location: ${JSON.stringify(raw)}; use a tracker location (saccade/t-N), a path on the tracker's machine (/absolute/path), or machine:/absolute/path`;
      this.set({
        status: "invalid",
        board: null,
        rawText: null,
        errors: [{ code: "asset-denied", message }],
        diagnostics: [],
        laneMessage: message,
        renderOutcomes: [],
        renderGeneration: null,
      });
      return;
    }
    await this.open(parsed);
  }

  async loadFigureAsset(
    location: Location,
    block: FigureBlock,
  ): Promise<Uint8Array | string> {
    const token = this.generation;
    try {
      const asset = await sourceFor(location.machine).loadAsset(
        location,
        block.input,
      );
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
  return machine === "fixture" ? "fixture" : machine;
}
