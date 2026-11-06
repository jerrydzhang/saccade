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

export type ViewerState = {
  status: ViewerStatus;
  location: Location | null;
  board: Board | null;
  rawText: string | null;
  errors: StructuredIssue[];
  diagnostics: Diagnostic[];
  laneMessage: string | null;
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
    copyMode: "source",
    theme: "system",
  };

  private listeners = new Set<(state: ViewerState) => void>();

  subscribe(listener: (state: ViewerState) => void): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  get snapshot(): ViewerState {
    return this.state;
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

  /** Open an explicit location. Every lane and control funnels through here. */
  async open(location: Location): Promise<void> {
    this.set({ status: "loading", location, errors: [], laneMessage: null });
    let text: string;
    try {
      const loaded = await sourceFor(location.machine).loadText(location);
      text = loaded.text;
    } catch (error) {
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
    if (result.status === "invalid") {
      this.set({
        status: "invalid",
        board: null,
        rawText: text,
        errors: result.errors,
        diagnostics: result.diagnostics,
      });
      return;
    }
    this.set({
      status: "loaded",
      board: result.board,
      rawText: text,
      errors: [],
      diagnostics: result.board.diagnostics,
    });
  }

  async loadFigureAsset(location: Location, block: FigureBlock): Promise<Uint8Array | string> {
    try {
      const asset = await sourceFor(location.machine).loadAsset(location, block.path);
      return asset.bytes;
    } catch (error) {
      if (error instanceof SourceUnavailableError) {
        if (error.code === "asset-denied") return `denied: ${error.message}`;
        return `missing: ${error.message}`;
      }
      return `missing: ${String(error)}`;
    }
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
