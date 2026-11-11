// Selection first (t-14 "Interaction"): `s` puts labels on what can be
// selected - lowercase on text targets, uppercase on non-text targets -
// typing narrows the text targets by their text, a typed label selects
// its target, Enter is the primary act, and right-click opens the same
// menu at the pointer. The DOM-independent half lives here: label
// assignment and typed-input narrowing, so the contracts are pinned
// without a document.

export type TargetKind =
  | "text"
  | "annotation"
  | "figure"
  | "diagram"
  | "incorporation"
  | "unknown"
  | "citation"
  | "group"
  | "control";

/** Whether a target carries prose: text targets narrow by their text
 * and carry lowercase labels; everything else carries uppercase ones. */
export function isTextKind(kind: TargetKind): boolean {
  return kind === "text" || kind === "annotation";
}

const LOWER = "abcdefghijklmnopqrstuvwxyz".split("");
const UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

/** Two-character continuation, from the same letters. */
function combos(pool: string[]): string[] {
  const out: string[] = [];
  for (const first of pool) {
    for (const second of pool) {
      out.push(`${first}${second}`);
    }
  }
  return out;
}

/** The labels for `count` targets: the first targets of each namespace
 * get single letters, the rest get two-letter labels. Stable: target
 * order in the document is the label order. */
export function assignLabels(kinds: TargetKind[]): string[] {
  const textTargets = kinds.map((kind, index) => (isTextKind(kind) ? index : -1)).filter((index) => index >= 0);
  const otherTargets = kinds.map((kind, index) => (isTextKind(kind) ? -1 : index)).filter((index) => index >= 0);
  const labels: string[] = new Array(kinds.length).fill("");
  const textPool = [...LOWER, ...combos(LOWER)];
  const otherPool = [...UPPER, ...combos(UPPER)];
  textTargets.forEach((index, position) => {
    labels[index] = textPool[position] ?? "";
  });
  otherTargets.forEach((index, position) => {
    labels[index] = otherPool[position] ?? "";
  });
  return labels;
}

/** A typed step against the current selection state: a label that
 * completes selects its target; letters that no label needs either
 * extend a pending label or narrow the text targets by their text. */
export type NarrowState = {
  /** The targets still in play (indices into the target list). */
  alive: number[];
  /** The pending label keystrokes, if any label still matches. */
  pendingLabel: string;
  /** The typed text narrowing, when the step is a text search. */
  query: string;
};

export function startNarrow(count: number): NarrowState {
  return { alive: Array.from({ length: count }, (_, index) => index), pendingLabel: "", query: "" };
}

/** One typed character. Case decides the namespace: an uppercase
 * letter extends a non-text label, a lowercase letter a text label or
 * the text search. Returns the state; `selected` names a completed
 * label's target, if any. */
export function narrowStep(
  state: NarrowState,
  kinds: TargetKind[],
  texts: string[],
  labels: string[],
  typed: string,
): NarrowState & { selected: number | null } {
  if (typed.length !== 1) return { ...state, selected: null };
  const matching = (candidate: string): number[] =>
    state.alive.filter(
      (index) => labels[index] !== undefined && labels[index].length > 0 && labels[index].startsWith(candidate),
    );
  // a letter that names no label in its own case names one in the
  // other: no label ever needs Shift (round 16 folds the typed key to
  // the labels' case). Only a letter that names no label at all
  // narrows the text targets.
  let pending = state.pendingLabel + typed;
  let labelMatch = matching(pending);
  if (labelMatch.length === 0) {
    const flipped = typed === typed.toLowerCase() ? typed.toUpperCase() : typed.toLowerCase();
    const other = state.pendingLabel + flipped;
    const otherMatch = matching(other);
    if (otherMatch.length > 0) {
      pending = other;
      labelMatch = otherMatch;
    }
  }
  if (labelMatch.length > 0) {
    const first = labelMatch[0];
    if (first !== undefined && labelMatch.length === 1 && labels[first] === pending) {
      return { ...state, pendingLabel: "", selected: first };
    }
    return { ...state, pendingLabel: pending, selected: null };
  }
  // no label needs this character: it narrows the text targets
  const query = state.query + typed.toLowerCase();
  const alive = state.alive.filter((index) => {
    const kind = kinds[index];
    const text = texts[index];
    return kind !== undefined && text !== undefined && isTextKind(kind) && text.toLowerCase().includes(query);
  });
  return { alive, pendingLabel: "", query, selected: null };
}
