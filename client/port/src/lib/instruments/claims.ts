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
