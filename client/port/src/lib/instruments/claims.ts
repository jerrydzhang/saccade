// The claims list: which locations an instrument shows, one
// place listing the instruments - the window assembly reads it
// instead of a hard-coded choice between two.

import type { Address } from "../contracts/places.ts";
import { isPlane } from "../location.ts";

export type InstrumentClaim = Readonly<{
  /** The instrument's registration kind (lib/instruments/registry.ts
   * definitions register under it). */
  instrument: string;
  /** Whether this instrument shows the location. */
  claims(address: Address): boolean;
}>;

/** The instruments in claim order: the first claim that names an
 * address shows it. The control plane's addresses are the task
 * list's; every other address is the board's. */
export const instrumentClaims: readonly InstrumentClaim[] = [
  { instrument: "tasks", claims: (address) => isPlane(address) },
  { instrument: "board", claims: (address) => !isPlane(address) },
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
