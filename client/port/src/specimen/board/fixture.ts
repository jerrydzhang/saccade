// The board specimens' fixtures (dev only): a board read from YAML by
// the app's own decoder, framed as the board instrument's content, and
// a controller that answers what BoardView asks and does nothing else.
// The sections mount the real BoardView, so what they show is the
// renderer, the hydration and the stylesheets the app ships.

import { decodeBoard } from "../../lib/board/decode.ts";
import { diffBoards, type ComponentChange } from "../../lib/board/history-diff.ts";
import type { Board } from "../../lib/board/model.ts";
import type { Expansion } from "../../lib/board/expansion.ts";
import type { BoardData, BoardUI, FindView, HistoryView } from "../../instruments/board.ts";
import { selectBlock } from "../../lib/board/expansion.ts";

export const SPECIMEN_LOCATION = { machine: "tracker:specimen", path: "t-2@r9" } as const;

/** A board from YAML, through the app's decoder. */
export function board(yaml: string): Board {
  const decoded = decodeBoard(yaml);
  if (decoded.status !== "ok") throw new Error(`specimen board did not decode: ${JSON.stringify(decoded.errors)}`);
  return decoded.board;
}

/** The board instrument's content for a shown board. */
export function frame(shown: Board, history: HistoryView | null = null): BoardData {
  return {
    generation: 1,
    asked: "specimen/t-2",
    location: "specimen/t-2@r9",
    resolved: { ...SPECIMEN_LOCATION },
    status: "loaded",
    board: shown,
    errors: [],
    renderOutcomes: [],
    targets: [],
    folded: [],
    history,
  };
}

/** History mode's content: `after` shown against `before`. */
export function historyFrame(
  before: Board,
  after: Board,
  changes: readonly ComponentChange[],
  view: Omit<HistoryView, "changed" | "at">,
): BoardData {
  const diff = diffBoards(before, after, changes);
  return frame(diff.board, { ...view, changed: diff.changed, at: -1 });
}

const FIGURE = `<svg xmlns="http://www.w3.org/2000/svg" width="520" height="160" viewBox="0 0 520 160">
<g fill="none" stroke="#5a5a5a" stroke-width="2"><rect x="20" y="40" width="120" height="70" rx="6"/><rect x="200" y="40" width="120" height="70" rx="6"/><rect x="380" y="40" width="120" height="70" rx="6"/><path d="M140 75h60M320 75h60"/></g>
<g font-family="serif" font-size="16" fill="#3a3a3a" text-anchor="middle"><text x="80" y="80">read</text><text x="260" y="80">offer</text><text x="440" y="80">accept</text></g></svg>`;

/** A controller for the specimen: the views it is given, inert acts. */
export function controller(
  options: { find?: FindView; expansions?: readonly Expansion[]; refusal?: string | null } = {},
): BoardUI {
  const answers: Record<string, unknown> = {
    displayedGeneration: () => 1,
    loadFigureAsset: async () => new TextEncoder().encode(FIGURE),
    subscribeFind: (observer: (view: FindView) => void) => {
      if (options.find) observer(options.find);
      return () => {};
    },
    subscribeHistoryRefusal: (observer: (refusal: string | null) => void) => {
      observer(options.refusal ?? null);
      return () => {};
    },
    expansions: () => options.expansions ?? [],
  };
  return new Proxy(answers, {
    get(target, key: string) {
      if (key in target) return target[key];
      // every other method: inert, and a subscription answers its own
      // unsubscribe
      return () => () => {};
    },
  }) as unknown as BoardUI;
}

/** An expanded incorporation at `part`, citing `cited`'s components. */
export function expanded(part: string, cited: Board, selects: readonly string[]): Expansion {
  return {
    part,
    status: "loaded",
    location: { machine: "tracker:specimen", path: "t-24@r23" },
    components: selects.map((select, ordinal) => ({ select, part: `${part}/cited/${ordinal}`, block: selectBlock(cited, select) })),
    message: "",
  };
}

/** A find view over the given parts: `query` matched `count` times in
 * each, the first occurrence of `activePart` active. */
export function findView(query: string, parts: readonly (readonly [string, number])[], activePart: string, open = true): FindView {
  const marks = new Map(parts.map(([part, count]) => [part, { count, active: part === activePart ? 0 : -1 }]));
  const count = parts.reduce((sum, [, n]) => sum + n, 0);
  const before = parts.findIndex(([part]) => part === activePart);
  const active = parts.slice(0, before).reduce((sum, [, n]) => sum + n, 0);
  return { open, query, count, active, unitParts: [], activeUnit: null, marks };
}
