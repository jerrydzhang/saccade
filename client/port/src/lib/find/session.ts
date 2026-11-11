// The find session, lifted from client/port-staging/find-session.ts
// (t-36@r65) into the port unchanged.
//
// Find is a facility of the shown revision: entry, live typing, match
// movement as non-trail interaction, exit semantics. Scenario clauses
// (t-36@r50 reference/acceptance-scenarios.txt, scenario 16 lines
// 353-372): bare / enters Find on the displayed revision, in ordinary
// reading or history; typing updates matches; match movement is not
// trail navigation; leaving Find changes neither the revision nor the
// history facilities; a mouse reference activation ends Find and opens
// the reference, while a click alone does not close Find; Back does
// not resurrect Find.
//
// Executed sources:
//   - Prototype inst16.html:402-409 findText: the query is trimmed,
//     matched case-insensitively over the shown board's text; a hit
//     scrolls into view; the found marks take labels and selecting one
//     acts on the enclosing target. Entry in the prototype is space
//     then / through the leader (:71-79 commands table, :501 space
//     opens it) - the ratified difference is the bare / entry, carried
//     by input adapter A2 (harness t36_adapters.py:61-67; scenario 16
//     differential note, line 367).
//   - Baseline app (4d59c35/b8240b5, the S16-green carrier):
//     openSearch/closeSearch App.svelte:1055-1061; Escape closes at
//     :2836-2841; the search field's Enter/Shift-Enter move matches at
//     :3521; nextMatch wraps modulo at :3486-3491; occurrences are
//     derived from the live query and the shown board at :3187-3190;
//     searchUnits + collectMatches + normalizeQuery in
//     lib/board/projection.ts:202-238 (trim, lowercase, non-
//     overlapping hits, prose text first then the authored source).
// Staged deviations from executed sources are recorded in README.md
// (the prototype's 2-character minimum is not carried; the
// navigate-away exit is scenario-ratified but executed by neither
// source; the baseline leaves the search bar open through a citation
// activation, which scenario 16 forbids - this module follows the
// scenario).

export type FindUnit = Readonly<{
  id: string;
  /** The prose as shown (normalized). */
  text: string;
  /** The authored source, consulted only when the shown text did not
   * match (projection.ts searchUnits, :212-220). */
  source: string;
}>;

export type FindOccurrence = Readonly<{
  unit: FindUnit;
  start: number;
  end: number;
}>;

/** Strip math delimiters so `$x$` queries match the TeX value `x`
 * (projection.ts normalizeQuery, :236-238). */
export function normalizeQuery(text: string): string {
  return text.replace(/^\$\$?/, "").replace(/\$\$?$/, "");
}

/** Non-overlapping case-sensitive hits of `needle` in `haystack`,
 * advancing by the needle's length (projection.ts collectMatches,
 * :221-233). */
function collectMatches(haystack: string, needle: string): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  let from = 0;
  while (from <= haystack.length) {
    const idx = haystack.indexOf(needle, from);
    if (idx < 0) break;
    out.push([idx, idx + needle.length]);
    from = idx + Math.max(needle.length, 1);
  }
  return out;
}

/** The occurrences of `query` over `units`: trimmed, lowercase,
 * shown text first, authored source only as fallback
 * (projection.ts searchUnits, :202-220). An empty or blank query has
 * no occurrences (the app's "type to search" state). */
export function searchUnits(units: readonly FindUnit[], query: string): readonly FindOccurrence[] {
  const trimmed = query.trim();
  if (trimmed.length === 0) return [];
  const needle = normalizeQuery(trimmed).toLowerCase();
  const out: FindOccurrence[] = [];
  for (const unit of units) {
    const textMatches = collectMatches(normalizeQuery(unit.text).toLowerCase(), needle);
    if (textMatches.length > 0) {
      for (const [start, end] of textMatches) out.push({ unit, start, end });
      continue;
    }
    const sourceMatches = collectMatches(normalizeQuery(unit.source).toLowerCase(), needle);
    for (const [start, end] of sourceMatches) out.push({ unit, start, end });
  }
  return out;
}

export type FindSession = Readonly<{
  isOpen(): boolean;
  /** The shown query; it survives exit and re-entry (the baseline's
   * openSearch/closeSearch never clear it, App.svelte:1055-1061). */
  query(): string;
  /** The occurrences of the query over the CURRENT units of the shown
   * revision - recomputed per call, never a frozen copy. */
  occurrences(): readonly FindOccurrence[];
  /** The active match index; it persists while typing, exactly as the
   * baseline's activeMatch does (App.svelte:1043, no reset on input). */
  active(): number;
  /** Bare / enters Find (ratified entry; adapter A2). The prototype
   * enters with space then / (inst16.html leader) - the adapter, not
   * a normalization, carries that difference. */
  enter(): void;
  /** Typing updates the matches live (the derived occurrences,
   * App.svelte:3187-3190). Returns the new occurrences. */
  type(text: string): readonly FindOccurrence[];
  /** Enter is next, Shift-Enter previous (the contract's findExample,
   * t-36@r46; App.svelte:3521). Wraps around modulo the count
   * (nextMatch, App.svelte:3486-3491). Movement is not trail
   * navigation: the caller scrolls to the returned occurrence; the
   * session records no trail entry and holds no location. */
  move(delta: number): FindOccurrence | null;
  /** Escape ends Find (App.svelte:2836-2841 closeSearch): keys return
   * to the document; the revision and the history facilities are
   * untouched - the session holds neither. Back cannot resurrect Find:
   * Find is not trail content, so a trail return has nothing to
   * reopen. */
  escape(): void;
  /** A mouse reference activation ends Find and opens the reference
   * (scenario 16). The caller navigates; this ends the session. */
  navigateAway(): void;
  /** A click alone never closes Find (scenario 16): an activating
   * click is navigateAway, any other click is this, and it is a
   * no-op. */
  click(): void;
}>;

export function createFindSession(units: () => readonly FindUnit[]): FindSession {
  let open = false;
  let query = "";
  let active = 0;

  return {
    isOpen: () => open,
    query: () => query,
    occurrences: () => searchUnits(units(), query),
    active: () => active,
    enter: () => {
      // Entry clears nothing (baseline openSearch): a re-entered Find
      // shows the previous query and its matches.
      open = true;
    },
    type: (text: string) => {
      // active is deliberately not reset: typing updates the matches
      // while the active index persists (baseline behavior).
      query = text;
      return searchUnits(units(), query);
    },
    move: (delta: number) => {
      const occurrences = searchUnits(units(), query);
      if (occurrences.length === 0) return null;
      active = (active + delta + occurrences.length) % occurrences.length;
      return occurrences[active] ?? null;
    },
    escape: () => {
      open = false;
    },
    navigateAway: () => {
      open = false;
    },
    click: () => {
      // A click alone does not close Find (scenario 16).
    },
  };
}
