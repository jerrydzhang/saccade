// Module 2 of the t-36 slice-2 extraction: label-select narrowing.
//
// Pure port of the selection interaction's narrowing logic: labels are
// assigned to visible targets, typing a label selects its target,
// letters that no label needs narrow text targets by their text.
//
// Executed prototype source (round 16 reference):
//   - inst16.html:100-101: LETTERS (home-row order) and labelsFor -
//     single letters, then two-letter combinations, stable in target
//     order; one namespace per hint session (lower for find hints,
//     upper for `s` selection).
//   - inst16.html:106: hintKey - Escape clears; a typed letter extends
//     the pending label (folded to the labels' case); labels narrow by
//     prefix; a single remaining label equal to the typed prefix
//     selects its target (exact-single); the selected target's
//     callback fires after hints clear.
// Ratified differences carried here (the clean-room law allows
// prototype interaction code where ratified; these two come from the
// accepted fix and the accepted app behavior):
//   - The case-fold rule (commit b8240b5, scenario 19 line 439 "the
//     label case-fold fix", green as S12; the mandate's wording):
//     a typed letter naming no label in its own case tries the folded
//     case; only a letter that names no label at all narrows the text
//     targets. So no label ever needs Shift.
//   - Two simultaneous namespaces: lowercase labels on text targets,
//     uppercase on everything else, and typed letters that no label
//     needs narrow the text targets by their text - the app's accepted
//     selection behavior (select.ts@b8240b5, originally dd53b50 "Select
//     first"), which the prototype's one-namespace-per-session
//     hintKey does not have. Scenario 12 (lines 269-287): selecting by
//     label selects that target, never a silent neighbour.
// Staged deviations from the prototype's executed details are recorded
// in README.md (label pool order, digit handling).

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

/** Two-character continuation, from the same letters (labelsFor's
 * second pool, inst16.html:101: the double loop over the letters). */
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
 * order in the document is the label order (labelsFor, inst16.html:101;
 * the namespace split is the app's ratified shape, select.ts@b8240b5). */
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
 * label's target, if any. Escape clearing the session is the host's
 * move (hintKey's first branch, inst16.html:106) - the session is
 * dropped, no per-stroke state changes here. */
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
  // The case-fold rule (b8240b5, the mandate's wording): a letter that
  // names no label in its own case names one in the other; only a
  // letter that names no label at all narrows the text targets.
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
  // Exact-single selection (hintKey's last branch, inst16.html:106):
  // one label left and it equals the typed prefix exactly.
  if (labelMatch.length > 0) {
    const first = labelMatch[0];
    if (first !== undefined && labelMatch.length === 1 && labels[first] === pending) {
      return { ...state, pendingLabel: "", selected: first };
    }
    return { ...state, pendingLabel: pending, selected: null };
  }
  // No label needs this character: it narrows the text targets by
  // their text, case-insensitively (the app's accepted narrowing;
  // matching itself is the prototype's prefix filter, hintKey :106).
  const query = state.query + typed.toLowerCase();
  const alive = state.alive.filter((index) => {
    const kind = kinds[index];
    const text = texts[index];
    return kind !== undefined && text !== undefined && isTextKind(kind) && text.toLowerCase().includes(query);
  });
  return { alive, pendingLabel: "", query, selected: null };
}
