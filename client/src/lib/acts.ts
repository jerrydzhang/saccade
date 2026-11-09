// The two acts, never a mode (t-14 "The suite"): Enter (or a click) is
// the primary act; `o` (or shift- or middle-click) opens elsewhere
// through the link. The routing is the instrument's declaration: the
// viewer follows here - Enter opens the citation in this window - and
// the tasks and agents instruments (t-18) have no here, so their
// Enter goes through the link. They declare it and arrive without
// reworking the shell's routing; the shell does not build them.

export type PrimaryRouting = "here" | "link";

export type InstrumentRouting = {
  /** What the instrument's own Enter does with a target: opens it in
   * this window (`here`) or opens it through the link (`link`). */
  primary: PrimaryRouting;
};

/** The viewer's declaration (t-14 "The viewer"): Enter follows here. */
export const VIEWER_ROUTING: InstrumentRouting = { primary: "here" };

/** An elsewhere act: through the link (`o`, the answer becoming the
 * link when there is none), elsewhere once without binding (`i`), or
 * always a new window (`w`). */
export type ElsewhereAct = "link" | "once" | "new";

/** The full location spec a citation names: the parseable string form
 * the open surfaces all take, relative to the window's own tracker. */
export function citationSpec(
  location: { machine: string; path: string },
  path: string,
): string | null {
  if (!location.machine.startsWith("tracker:")) return null;
  const tracker = location.machine.slice("tracker:".length);
  return `${tracker}/${path}`;
}
