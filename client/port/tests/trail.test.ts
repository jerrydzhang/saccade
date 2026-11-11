// The window's trail (chunk E): explicit openings only, departure
// capture on the entry being left, Back/Forward restore recorded
// places - never folds, zoom or drafts (scenario 14).

import assert from "node:assert/strict";
import test from "node:test";
import { createTrail } from "../src/lib/navigation/trail.ts";
import type { SavedPlace } from "../src/lib/contracts/places.ts";

const place = (tag: string): SavedPlace => ({
  bufferId: "b",
  address: { machine: "m", path: `/p-${tag}` },
  position: { kind: "default" },
});

function harness() {
  const log: string[] = [];
  const captures: SavedPlace[] = [];
  let next = 0;
  const trail = createTrail({
    capture: () => {
      const p = place(`cap-${next++}`);
      captures.push(p);
      return p;
    },
    restore: (p) => log.push(`restore:${p.address.path}`),
  });
  return { trail, log, captures };
}

test("explicit openings add entries; the departure capture lands on the entry being left", () => {
  const { trail, log, captures } = harness();
  trail.opened("a");
  assert.deepEqual(trail.view().entries, [{ location: "a", hasDeparture: false }]);
  trail.opened("b");
  // leaving "a" captured its departure place
  assert.deepEqual(trail.view().entries, [
    { location: "a", hasDeparture: true },
    { location: "b", hasDeparture: false },
  ]);
  assert.equal(captures.length, 1);
  assert.deepEqual(log, []);
});

test("Back restores the departure place and moves the cursor; Forward retraces", () => {
  const h = harness();
  h.trail.opened("a");
  h.trail.opened("b");
  assert.equal(h.trail.back(), true);
  assert.deepEqual(h.log, ["restore:/p-cap-0"]);
  assert.equal(h.trail.view().cursor, 0);
});

test("Back restores exactly the recorded departure; repeated backs need departures", () => {
  const h = harness();
  h.trail.opened("a");
  h.trail.opened("b");
  h.trail.opened("c");
  assert.equal(h.trail.back(), true);
  assert.deepEqual(h.log, ["restore:/p-cap-1"]);
  assert.equal(h.trail.back(), true);
  assert.deepEqual(h.log, ["restore:/p-cap-1", "restore:/p-cap-0"]);
  // at the first entry: no further back
  assert.equal(h.trail.back(), false);
  // forward retraces through the recorded departure of the next entry
  assert.equal(h.trail.forward(), true);
  assert.deepEqual(h.log, ["restore:/p-cap-1", "restore:/p-cap-0", "restore:/p-cap-1"]);
});

test("a new opening from the middle truncates forward history", () => {
  const h = harness();
  h.trail.opened("a");
  h.trail.opened("b");
  h.trail.opened("c");
  h.trail.back();
  h.trail.opened("d");
  assert.deepEqual(
    h.trail.view().entries.map((e) => e.location),
    ["a", "b", "d"],
  );
  assert.equal(h.trail.view().canForward, false);
});
