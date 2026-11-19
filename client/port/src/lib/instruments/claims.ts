// The claims list: which locations an instrument shows, one
// place listing the instruments - the window assembly reads it
// instead of a hard-coded choice between two.

import type { Address } from "../contracts/places.ts";

export type InstrumentClaim = Readonly<{
  /** The instrument's registration kind (lib/instruments/registry.ts
   * definitions register under it). */
  instrument: string;
  /** Whether this instrument shows the location. */
  claims(address: Address): boolean;
}>;

/** A project's task list: the tracker machine the location names, the
 * list as its path. */
function isTasksList(address: Address): boolean {
  return address.machine.startsWith("tracker:") && address.path === "tasks";
}

/** The instruments in claim order: the first claim that names an
 * address shows it. A project's task list is the task list's; a
 * window with no location is the frame's empty window; every other
 * address is the board's. */
export const instrumentClaims: readonly InstrumentClaim[] = [
  { instrument: "tasks", claims: isTasksList },
  { instrument: "board", claims: (address) => !isTasksList(address) },
];

/** The first instrument whose claim names the address, in the list's
 * order; null when none does (an address no instrument shows). */
export function instrumentForAddress(
  address: Address,
  claims: readonly InstrumentClaim[] = instrumentClaims,
): string | null {
  const claimed = claims.find((claim) => claim.claims(address));
  return claimed === undefined ? null : claimed.instrument;
}
