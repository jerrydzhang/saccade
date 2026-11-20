// Tests for the find session (find-session.ts): scenario 16 clauses -
// bare-/ entry (adapter A2), typing updates matches, match movement as
// non-trail interaction, exit semantics.
import test from "node:test";
import assert from "node:assert/strict";

import {
  createFindSession,
  normalizeQuery,
  searchUnits,
  type FindUnit,
} from "./find-session.ts";

const UNITS: readonly FindUnit[] = [
  { id: "u1", text: "Why does the tone arm resonate?", source: "Why does the tone arm resonate?" },
  { id: "u2", text: "See t-3@r14 and the sweep", source: "See t-3@r14 and the sweep" },
  { id: "u3", text: "The plinth rebuild replaced the plate", source: "The plinth rebuild replaced the plate" },
];

test("searchUnits: trim, lowercase, non-overlapping occurrences", () => {
  const hits = searchUnits(UNITS, "  the ");
  // Per unit, every non-overlapping hit is an occurrence, in unit
  // order (u3's prose carries "the" twice).
  assert.deepEqual(
    hits.map((hit) => [hit.unit.id, hit.start, hit.end]),
    [
      ["u1", 9, 12],
      ["u2", 16, 19],
      ["u3", 0, 3],
      ["u3", 28, 31],
    ],
  );
});

test("searchUnits: an empty or blank query has no occurrences", () => {
  assert.deepEqual(searchUnits(UNITS, ""), []);
  assert.deepEqual(searchUnits(UNITS, "   "), []);
});

test("searchUnits: no matches for an absent query (the 'no matches' state)", () => {
  assert.deepEqual(searchUnits(UNITS, "zebra"), []);
});

test("searchUnits: the authored source is consulted only as fallback", () => {
  // One authored expression is one logical search unit: matches in
  // the shown text count; the source is only consulted when the shown
  // text did not match (projection.ts searchUnits).
  const units: readonly FindUnit[] = [
    { id: "shown", text: "mass is 2 kg", source: "mass is $m = 2$ kg" },
    { id: "hidden", text: "nothing here", source: "the sweep in appendix" },
  ];
  const shown = searchUnits(units, "mass");
  assert.deepEqual(shown.map((hit) => hit.unit.id), ["shown"]);
  const fallback = searchUnits(units, "appendix");
  assert.deepEqual(fallback.map((hit) => hit.unit.id), ["hidden"]);
});

test("normalizeQuery strips math delimiters so a delimited query finds the value", () => {
  assert.equal(normalizeQuery("$x$"), "x");
  assert.equal(normalizeQuery("$$y$$"), "y");
  assert.equal(normalizeQuery("plain"), "plain");
  const units: readonly FindUnit[] = [{ id: "tex", text: "mass is m = 2 kg", source: "mass is $m = 2$ kg" }];
  assert.equal(searchUnits(units, "$m = 2$").length, 1);
});

test("bare / enters Find on the shown revision (adapter A2 entry)", () => {
  // The prototype enters with space then / (inst16.html leader);
  // the ratified app entry is the bare / - carried by the input
  // adapter, not a normalization (t36_adapters.py A2).
  const session = createFindSession(() => UNITS);
  assert.equal(session.isOpen(), false);
  session.enter();
  assert.equal(session.isOpen(), true);
});

test("typing updates matches live against the CURRENT units", () => {
  // The derived occurrences (App.svelte:3187-3190); the units
  // accessor models the shown revision's content changing under the
  // open session (scenario 11: content arriving while Find is open).
  let current: readonly FindUnit[] = UNITS;
  const session = createFindSession(() => current);
  session.enter();
  session.type("tone");
  assert.equal(session.occurrences().length, 1);
  current = [...UNITS, { id: "u4", text: "tone arm carriage", source: "tone arm carriage" }];
  assert.equal(session.occurrences().length, 2);
  session.type("tone arm");
  assert.equal(session.occurrences().length, 2);
});

test("Enter moves to the next match and wraps; Shift-Enter moves back", () => {
  // nextMatch modulo wrap (App.svelte:3486-3491); the search field's
  // Enter/Shift-Enter mapping (App.svelte:3521). The active index
  // starts at 0 and each move applies the delta first, exactly as the
  // baseline's (activeMatch + delta + n) % n does - so the first
  // Enter lands on index 1, the baseline's executed arithmetic.
  const session = createFindSession(() => UNITS);
  session.enter();
  session.type("the");
  // Occurrence order: u1, u2, u3, u3 (u3's prose holds "the" twice).
  assert.equal(session.occurrences().length, 4);
  assert.equal(session.move(1)?.unit.id, "u2");
  assert.equal(session.move(1)?.unit.id, "u3");
  assert.equal(session.move(-1)?.unit.id, "u2");
  assert.equal(session.move(-1)?.unit.id, "u1");
  // Wrapped backwards past the start.
  assert.equal(session.move(-1)?.unit.id, "u3");
});

test("movement with no occurrences acts nothing", () => {
  const session = createFindSession(() => UNITS);
  session.enter();
  session.type("zebra");
  assert.equal(session.move(1), null);
  assert.equal(session.active(), 0);
});

test("the active index persists while typing (baseline activeMatch)", () => {
  // App.svelte:1043: activeMatch is never reset by input; a later
  // movement wraps modulo the new count.
  const session = createFindSession(() => UNITS);
  session.enter();
  session.type("the");
  assert.equal(session.active(), 0);
  assert.equal(session.move(1)?.unit.id, "u2");
  assert.equal(session.active(), 1);
  session.type("plate");
  // The index is still 1; occurrences now hold one match.
  assert.equal(session.occurrences().length, 1);
  const moved = session.move(1);
  assert.equal(moved?.unit.id, "u3");
  assert.equal(session.active(), 0);
});

test("match movement is not trail navigation", () => {
  // Scenario 16: movement only moves between matches; the session
  // holds no location and records no trail entry - its whole state is
  // open/query/active, so a movement can change nothing else.
  const session = createFindSession(() => UNITS);
  session.enter();
  session.type("the");
  const before = session.query();
  session.move(1);
  session.move(-1);
  assert.equal(session.query(), before);
  assert.equal(session.isOpen(), true);
  assert.equal(session.occurrences().length, 4);
});

test("Escape ends Find; the revision and history facilities are untouched", () => {
  // closeSearch (App.svelte:1060-1061, reached from Escape at
  // :2836-2841): the session closes; it holds neither revision nor
  // history state, so leaving changes neither.
  const session = createFindSession(() => UNITS);
  session.enter();
  session.type("the");
  session.escape();
  assert.equal(session.isOpen(), false);
  // The query is not cleared: a re-entered Find shows it again
  // (openSearch never clears, App.svelte:1055-1058).
  session.enter();
  assert.equal(session.query(), "the");
  assert.equal(session.occurrences().length, 4);
});

test("a reference activation ends Find (navigateAway); a click alone never closes", () => {
  // Scenario 16: a mouse reference activation ends Find and opens the
  // reference, while a click alone does not close Find.
  const session = createFindSession(() => UNITS);
  session.enter();
  session.type("the");
  session.click();
  assert.equal(session.isOpen(), true);
  session.navigateAway();
  assert.equal(session.isOpen(), false);
});

test("Back cannot resurrect Find: exit leaves nothing to reopen", () => {
  // Find is not trail content; after exit the only way back in is a
  // fresh entry.
  const session = createFindSession(() => UNITS);
  session.enter();
  session.type("the");
  session.escape();
  assert.equal(session.isOpen(), false);
  session.navigateAway();
  session.click();
  assert.equal(session.isOpen(), false);
});
