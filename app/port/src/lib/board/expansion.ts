// Incorporation expansion (t-58; t-14 "The viewer": "Incorporations
// are collapsed to their source line and receiving text; expanding
// shows only the cited components"). An incorporation renders
// collapsed - its source line and its own receiving text, both the
// citing board's model. Its expand act reads the cited revision's
// board and shows exactly the components its select paths name,
// nothing around them; the same act collapses it again.
//
// The state is view state of the board's buffer, the way folds are:
// keyed by the incorporation's part AND its source, so a refresh or a
// return keeps an expansion while the same citation stands there, and
// a different citation at that part shows collapsed. It is never
// persisted with the window's place. Pure: the loader is injected.

import type { Block, Board } from "./model.ts";
import type { Location } from "../sources.ts";

/** The part path a cited component renders at: the incorporation's
 * own part plus its select's ordinal. Never a model target (the cited
 * component belongs to another task's revision). */
export function citedPart(incorporationPart: string, ordinal: number): string {
  return `${incorporationPart}/cited/${ordinal}`;
}

/** The component a select path names in a board, or null when the
 * path names nothing there. Select paths alternate content indices
 * with the container kind between: /content/1/group/content/0. */
export function selectBlock(board: Board, select: string): Block | null {
  const segments = select.split("/");
  if (segments[0] !== "" || segments[1] !== "content") return null;
  let blocks: readonly Block[] = board.content;
  let index = 2;
  for (;;) {
    const position = segments[index];
    if (position === undefined || !/^\d+$/.test(position)) return null;
    const block = blocks[Number(position)];
    if (block === undefined) return null;
    index += 1;
    if (index === segments.length) return block;
    const kind = segments[index];
    if (segments[index + 1] !== "content") return null;
    if ((kind === "group" || kind === "incorporation") && block.kind === kind) {
      blocks = block.content;
      index += 2;
      continue;
    }
    return null;
  }
}

export type CitedComponent = Readonly<{
  select: string;
  /** Where it renders (citedPart). */
  part: string;
  /** The named component, or null when the select names nothing in
   * the cited revision. */
  block: Block | null;
}>;

/** One expanded incorporation. */
export type Expansion = Readonly<{
  part: string;
  status: "loading" | "loaded" | "failed";
  /** The cited revision's location (its figures load there); null
   * until it answered. */
  location: Location | null;
  components: readonly CitedComponent[];
  /** Why the cited revision could not be read (status failed). */
  message: string;
}>;

/** An incorporation's authored source (the model's, as its target
 * carries it). */
export type CitedSource = Readonly<{ task: string; revision: number; select: readonly string[] }>;

export type LoadedBoard = Readonly<{ board: Board; location: Location }>;

export interface ExpansionDeps {
  /** Read and decode the cited revision (task@rN) on the shown
   * board's machine. Throws when it cannot. */
  load(machine: string, path: string): Promise<LoadedBoard>;
  /** The shown frame's machine, or null before one resolved. */
  machine(): string | null;
  /** The incorporation source standing at a part of the shown board
   * now, or null when the part names no incorporation. */
  sourceAt(part: string): CitedSource | null;
  onChange(): void;
}

export interface Expansions {
  isExpanded(part: string): boolean;
  /** Expand a collapsed incorporation, collapse an expanded one. */
  toggle(part: string): void;
  /** The expansions of the shown board, in no particular order. */
  view(): readonly Expansion[];
}

const keyOf = (part: string, machine: string, source: CitedSource): string =>
  JSON.stringify([part, machine, source.task, source.revision, source.select]);

export function createExpansions(deps: ExpansionDeps): Expansions {
  // keyed by part + source: a pinned revision is immutable, so what
  // answered once stays the answer
  const expanded = new Map<string, Expansion>();
  const currentKey = (part: string): string | null => {
    const source = deps.sourceAt(part);
    const machine = deps.machine();
    return source === null || machine === null ? null : keyOf(part, machine, source);
  };

  async function fetch(key: string, part: string, machine: string, source: CitedSource): Promise<void> {
    const path = `${source.task}@r${source.revision}`;
    let next: Expansion;
    try {
      const loaded = await deps.load(machine, path);
      next = {
        part,
        status: "loaded",
        location: loaded.location,
        components: source.select.map((select, ordinal) => ({
          select,
          part: citedPart(part, ordinal),
          block: selectBlock(loaded.board, select),
        })),
        message: "",
      };
    } catch (error) {
      next = {
        part,
        status: "failed",
        location: null,
        components: [],
        message: error instanceof Error ? error.message : String(error),
      };
    }
    // collapsed meanwhile: the answer is dropped
    if (!expanded.has(key)) return;
    expanded.set(key, next);
    deps.onChange();
  }

  return {
    isExpanded(part) {
      const key = currentKey(part);
      return key !== null && expanded.has(key);
    },
    toggle(part) {
      const source = deps.sourceAt(part);
      const machine = deps.machine();
      if (source === null || machine === null) return;
      const key = keyOf(part, machine, source);
      if (expanded.has(key)) {
        expanded.delete(key);
        deps.onChange();
        return;
      }
      expanded.set(key, { part, status: "loading", location: null, components: [], message: "" });
      deps.onChange();
      void fetch(key, part, machine, source);
    },
    view() {
      const out: Expansion[] = [];
      for (const [key, expansion] of expanded) {
        if (currentKey(expansion.part) === key) out.push(expansion);
      }
      return out;
    },
  };
}
