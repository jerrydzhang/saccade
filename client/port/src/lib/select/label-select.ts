// The selection interaction's narrowing logic, lifted from
// client/port-staging/label-select.ts (t-36@r65) unchanged. Pure port:
// labels are assigned to the visible targets and typing a label
// selects its target (the prototype's hintKey, inst16.html:106; the
// text narrowing the staging carried is gone, human ruling m-541).
// ONE namespace (human ruling m-545, the prototype's labelsFor,
// inst16.html:99-100): the home-row letters in their order, painted
// uppercase whatever the target's kind; typing is case-insensitive.
// Case no longer tells targets apart (it made l and I ambiguous); how
// a label shows its target's kind is open for the design pass.

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

/** Whether a target carries prose (text and annotations). */
export function isTextKind(kind: TargetKind): boolean {
  return kind === "text" || kind === "annotation";
}

/** The prototype's label letters, in its order (inst16.html:99). */
const LETTERS = "ASDFGHJKLQWERTYUIOPZXCVBNM".split("");

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

/** The labels for `kinds.length` targets: up to 26 take single
 * letters in LETTERS order; more take two-letter pairs, ALL of them
 * (inst16.html:100 labelsFor), so no label prefixes another and every
 * one completes when typed. Stable: target order in the document is
 * the label order. */
export function assignLabels(kinds: readonly TargetKind[]): string[] {
  const pool = kinds.length <= LETTERS.length ? LETTERS : combos(LETTERS);
  return kinds.map((_, index) => pool[index] ?? "");
}

/** The selection's typed state: the pending label prefix. */
export type NarrowState = {
  /** The targets still in play (indices into the target list). */
  alive: number[];
  /** The label keystrokes typed so far. */
  pendingLabel: string;
};

export function startNarrow(count: number): NarrowState {
  return { alive: Array.from({ length: count }, (_, index) => index), pendingLabel: "" };
}

/** One typed letter, exactly as the prototype's hintKey
 * (inst16.html:106), human ruling m-541: letters match LABELS only -
 * the pending prefix filters the labels, one exact match selects its
 * target, and ZERO matches ends the session (`exit`). There is no text
 * narrowing (find, /, is the text search). Typing is case-insensitive
 * (no label ever needs Shift). The caller ignores non-letters. */
export function narrowStep(
  state: NarrowState,
  labels: string[],
  typed: string,
): NarrowState & { selected: number | null; exit: boolean } {
  const matching = (candidate: string): number[] =>
    state.alive.filter(
      (index) => labels[index] !== undefined && labels[index] !== "" && labels[index].startsWith(candidate),
    );
  // case-insensitive: every label is uppercase (hintKey folds the
  // typed letter to the namespace's case)
  const pending = state.pendingLabel + typed.toUpperCase();
  const alive = matching(pending);
  if (alive.length === 0) return { alive: [], pendingLabel: "", selected: null, exit: true };
  const first = alive[0];
  if (alive.length === 1 && first !== undefined && labels[first] === pending) {
    return { alive, pendingLabel: "", selected: first, exit: false };
  }
  return { alive, pendingLabel: pending, selected: null, exit: false };
}
