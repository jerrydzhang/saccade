// The suite shell's pure half (t-14 "The suite"): the window jump's
// label assignment, the picker's entries (this device's attached
// windows first, then the detached states), and the names the HUD
// shows. The wire and the window acts ride the window store client and the
// bridge; what is here is only what a test can pin without either.

export type SuiteWindowInfo = { label: string; stateId?: string | null };

/** The jump labels: one chip per visible suite window, a, b, c ...
 * assigned in the window list's stable order. Beyond one alphabet the
 * labels grow a second character, exactly as the select-first labels
 * do. */
export function assignChips(windowLabels: string[]): Map<string, string> {
  const chips = new Map<string, string>();
  let next = 0;
  for (const label of windowLabels) {
    chips.set(label, chipFor(next));
    next += 1;
  }
  return chips;
}

function chipFor(index: number): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyz";
  let chip = "";
  let value = index;
  do {
    chip = alphabet[value % 26] + chip;
    value = Math.floor(value / 26) - 1;
  } while (value >= 0);
  return chip;
}

/** One picker entry: a window attached on this device (the jump
 * focuses it) or a detached state (Enter resumes one here). */
export type JumpEntry = {
  kind: "window" | "detached";
  chip: string | null;
  windowLabel: string | null;
  stateId: string;
  name: string;
};

export type WindowStateLite = {
  id: string;
  location: string | null;
  attachedHere: boolean;
};

/** The picker's entries: windows attached on this device, then the
 * detached states the human can resume here (t-14 "The suite"). A
 * detached state one of this device's windows shows is impossible:
 * attachedHere names this device. */
export function jumpPickerEntries(
  windows: SuiteWindowInfo[],
  states: WindowStateLite[],
): JumpEntry[] {
  const chips = assignChips(windows.map((window) => window.label));
  const stateById = new Map(states.map((state) => [state.id, state]));
  const attached: JumpEntry[] = [];
  for (const window of windows) {
    if (!window.stateId) continue;
    const state = stateById.get(window.stateId);
    if (!state || !state.attachedHere) continue;
    attached.push({
      kind: "window",
      chip: chips.get(window.label) ?? null,
      windowLabel: window.label,
      stateId: window.stateId,
      name: shortLocation(state.location),
    });
  }
  const attachedIds = new Set(attached.map((entry) => entry.stateId));
  const detached: JumpEntry[] = states
    .filter((state) => !state.attachedHere && !attachedIds.has(state.id))
    .map((state) => ({
      kind: "detached" as const,
      chip: null,
      windowLabel: null,
      stateId: state.id,
      name: shortLocation(state.location),
    }));
  return [...attached, ...detached];
}

/** The name a window or state shows: the tracker form drops the
 * surface prefix, everything else reads as written. */
export function shortLocation(location: string | null): string {
  if (location === null) return "(empty)";
  return location.replace(/^tracker\//, "");
}

/** The picker's narrowing: the query matches the name or the chip,
 * case-insensitive substring - the window jump narrows coarsely; the
 * document picker keeps the ranking ladder. */
export function entryMatches(entry: JumpEntry, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (needle.length === 0) return true;
  return (
    entry.name.toLowerCase().includes(needle) ||
    (entry.chip !== null && entry.chip.startsWith(needle))
  );
}
