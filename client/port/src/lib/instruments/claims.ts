// The claims seam, the frame's face over the registered instruments:
// which locations an instrument shows, and what it offers for a
// registered project's name. The frame reads these and never names an
// instrument; the instruments register themselves (lib/instruments/
// index.ts is the one file that names them).

import type { Address } from "../contracts/places.ts";
import { registeredInstruments } from "./registry.ts";

export type InstrumentClaim = Readonly<{
  /** The instrument's registration kind (the registry's one). */
  instrument: string;
  /** Whether this instrument shows the location. */
  claims(address: Address): boolean;
}>;

/** The instruments in claim order: the first claim that names an
 * address shows it. A project's task list is the task list's; a
 * window with no location is the frame's empty window; every other
 * address is the board's. */
export function instrumentClaims(): readonly InstrumentClaim[] {
  return registeredInstruments().map((entry) => ({
    instrument: entry.kind,
    claims: (address: Address) => entry.claims(address),
  }));
}

/** The first instrument whose claim names the address, in the
 * registered order; null when none does (an address no instrument
 * shows). */
export function instrumentForAddress(
  address: Address,
  claims: readonly InstrumentClaim[] = instrumentClaims(),
): string | null {
  const claimed = claims.find((claim) => claim.claims(address));
  return claimed === undefined ? null : claimed.instrument;
}

/** The location form an instrument offers for a registered project's
 * name (the task list claims `name/tasks`), or null when the
 * instrument offers none. The frame's rail reads this - the rail
 * knows nothing of tasks. */
export function projectViewFor(project: string): string | null {
  for (const entry of registeredInstruments()) {
    const view = entry.projectView(project);
    if (view !== null) return view;
  }
  return null;
}

/** The picker's task candidates on a tracker machine: the registered
 * instruments' first answer. The window's picker face reads this - the
 * window never reads a tracker's lib. */
export async function pickerCandidates(
  machine: string,
): Promise<readonly { id: string; title: string }[]> {
  for (const entry of registeredInstruments()) {
    if (entry.pickerCandidates === undefined) continue;
    const tasks = await entry.pickerCandidates(machine);
    if (tasks.length > 0) return tasks;
  }
  return [];
}

/** The targets the registered instruments publish beneath a project's
 * rail entry (the agents an instrument's tracker serves, its acts) or
 * at the rail's foot when the project is null (terminals no
 * registered project holds). The frame draws them without knowing
 * their kind. */
export function railTargetsFor(project: string | null): import("../contracts/buffer.ts").TargetDefinition[] {
  const out: import("../contracts/buffer.ts").TargetDefinition[] = [];
  for (const entry of registeredInstruments()) {
    for (const target of entry.railTargets(project)) out.push(target);
  }
  return out;
}

/** The unreachable note under a project's rail entry: the registered
 * instruments' first answer, or null (a plain entry stands). */
export function railNoteFor(project: string): string | null {
  for (const entry of registeredInstruments()) {
    const note = entry.railNote?.(project);
    if (note !== null && note !== undefined) return note;
  }
  return null;
}

/** The words feed's note, or null. */
export function railFootnote(): string | null {
  for (const entry of registeredInstruments()) {
    const note = entry.railFootnote?.();
    if (note !== null && note !== undefined) return note;
  }
  return null;
}
