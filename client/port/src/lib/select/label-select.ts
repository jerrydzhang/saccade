// The selection interaction's narrowing logic, lifted from
// client/port-staging/label-select.ts (t-36@r65) unchanged. Pure port:
// labels are assigned to the visible targets and typing a label
// selects its target (the prototype's hintKey, inst16.html:106; the
// text narrowing the staging carried is gone, human ruling m-541).
// Case-fold rule (b8240b5): a typed letter naming no label in its own
// case tries the folded case. Two simultaneous namespaces: lowercase
// on text targets, uppercase on everything else (the app's accepted
// shape; the prototype paints one uppercase namespace per session).

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
 * narrowing (find, /, is the text search). Case-fold rule (no label
 * ever needs Shift): a letter naming no label in its own case tries
 * the other case. The caller ignores non-letters before calling. */
export function narrowStep(
  state: NarrowState,
  labels: string[],
  typed: string,
): NarrowState & { selected: number | null; exit: boolean } {
  const matching = (candidate: string): number[] =>
    state.alive.filter(
      (index) => labels[index] !== undefined && labels[index] !== "" && labels[index].startsWith(candidate),
    );
  let pending = state.pendingLabel + typed;
  let alive = matching(pending);
  if (alive.length === 0) {
    const flipped = typed === typed.toLowerCase() ? typed.toUpperCase() : typed.toLowerCase();
    pending = state.pendingLabel + flipped;
    alive = matching(pending);
  }
  if (alive.length === 0) return { alive: [], pendingLabel: "", selected: null, exit: true };
  const first = alive[0];
  if (alive.length === 1 && first !== undefined && labels[first] === pending) {
    return { alive, pendingLabel: "", selected: first, exit: false };
  }
  return { alive, pendingLabel: pending, selected: null, exit: false };
}
