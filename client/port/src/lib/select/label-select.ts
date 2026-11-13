// The selection interaction's narrowing logic, lifted from
// client/port-staging/label-select.ts (t-36@r65) unchanged. Pure port:
// labels are assigned to the visible targets, typing a label selects
// its target, letters that no label needs narrow text targets by
// their text. Case-fold rule (b8240b5, scenario 12/19): a typed letter
// naming no label in its own case tries the folded case; only a
// letter that names no label at all narrows the text targets. Two
// simultaneous namespaces: lowercase on text targets, uppercase on
// everything else (the app's accepted shape).

export type TargetKind =
  | "text"
  | "annotation"
  | "figure"
  | "diagram"
  | "incorporation"
  | "unknown"
  | "citation"
  | "group"
  | "control"
  | "task";

/** Whether a target carries prose: text targets narrow by their text
 * and carry lowercase labels; everything else carries uppercase ones. */
export function isTextKind(kind: TargetKind): boolean {
  return kind === "text" || kind === "annotation";
}

const LOWER = "abcdefghijklmnopqrstuvwxyz".split("");
const UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

/** Two-character continuation, from the same letters (labelsFor's
 * second pool, inst16.html:101). */
function combos(pool: string[]): string[] {
  const out: string[] = [];
  for (const first of pool) {
    for (const second of pool) {
      out.push(`${first}${second}`);
    }
  }
  return out;
}

/** The labels for `count` targets, per namespace: up to 26 targets
 * take single letters; more take two-letter labels, ALL of them
 * (inst16.html:100 labelsFor), so no label prefixes another and every
 * one completes when typed. Stable: target order in the document is
 * the label order. */
export function assignLabels(kinds: TargetKind[]): string[] {
  const textTargets = kinds.map((kind, index) => (isTextKind(kind) ? index : -1)).filter((index) => index >= 0);
  const otherTargets = kinds.map((kind, index) => (isTextKind(kind) ? -1 : index)).filter((index) => index >= 0);
  const labels: string[] = new Array(kinds.length).fill("");
  const pool = (letters: string[], count: number): string[] =>
    count <= letters.length ? letters : combos(letters);
  const textPool = pool(LOWER, textTargets.length);
  const otherPool = pool(UPPER, otherTargets.length);
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
 * label's target, if any. Escape clearing the session is the host's
 * move - the session is dropped, no per-stroke state changes here. */
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
      (index) => labels[index] !== undefined && labels[index] !== "" && labels[index].startsWith(candidate),
    );
  // The case-fold rule: no label ever needs Shift.
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
  // Exact-single selection: one label left and it equals the typed
  // prefix exactly.
  if (labelMatch.length > 0) {
    const first = labelMatch[0];
    if (first !== undefined && labelMatch.length === 1 && labels[first] === pending) {
      return { ...state, pendingLabel: "", selected: first };
    }
    return { ...state, pendingLabel: pending, selected: null };
  }
  // No label needs this character: it narrows the text targets by
  // their text, case-insensitively.
  const query = state.query + typed.toLowerCase();
  const alive = state.alive.filter((index) => {
    const kind = kinds[index];
    const text = texts[index];
    return kind !== undefined && text !== undefined && isTextKind(kind) && text.toLowerCase().includes(query);
  });
  return { alive, pendingLabel: "", query, selected: null };
}
