// The suite shell's pure contracts (t-14 "The suite"): the jump
// labels' assignment, the picker's entries (windows here, then
// detached), the names the HUD shows, and the coarse narrowing. The
// window acts themselves ride the bridge and the window store client.

import test from "node:test";
import assert from "node:assert/strict";
import {
  assignChips,
  entryMatches,
  jumpPickerEntries,
  shortLocation,
  type SuiteWindowInfo,
} from "../src/lib/suite.ts";

const WINDOWS: SuiteWindowInfo[] = [
  { label: "main", stateId: "s-1" },
  { label: "w-2", stateId: "s-2" },
  { label: "w-3", stateId: null },
];

const STATES = [
  { id: "s-1", location: "tracker/smoke/t-5", attachedHere: true },
  { id: "s-2", location: "tracker/smoke/t-9@r3", attachedHere: true },
  { id: "s-3", location: "tracker/smoke/t-7", attachedHere: false },
  { id: "s-4", location: null, attachedHere: false },
  { id: "s-5", location: "tracker/other/t-2", attachedHere: false },
];

test("the jump labels assign one chip per window, in the list's order", () => {
  const chips = assignChips(["main", "w-2", "w-3"]);
  assert.deepEqual([...chips], [
    ["main", "a"],
    ["w-2", "b"],
    ["w-3", "c"],
  ]);
  const many = assignChips(Array.from({ length: 27 }, (_, index) => `w-${index}`));
  assert.equal(many.get("w-0"), "a");
  assert.equal(many.get("w-26"), "aa", "beyond one alphabet the labels grow a character");
});

test("the picker lists windows on this device, then detached states", () => {
  const entries = jumpPickerEntries(WINDOWS, STATES);
  assert.deepEqual(
    entries.map((entry) => [entry.kind, entry.chip, entry.name]),
    [
      ["window", "a", "smoke/t-5"],
      ["window", "b", "smoke/t-9@r3"],
      ["detached", null, "smoke/t-7"],
      ["detached", null, "(empty)"],
      ["detached", null, "other/t-2"],
    ],
  );
  // a state attached but carried by no window of this process is
  // another instance's: this picker leaves it alone
  assert.equal(
    jumpPickerEntries(
      [{ label: "main", stateId: "s-1" }],
      [
        { id: "s-1", location: "tracker/x/t-1", attachedHere: true },
        { id: "s-9", location: "tracker/x/t-2", attachedHere: true },
      ],
    ).some((entry) => entry.stateId === "s-9"),
    false,
    "another instance's attached window is not this picker's jump",
  );
});

test("the names read without the surface prefix, and empties say so", () => {
  assert.equal(shortLocation("tracker/smoke/t-5"), "smoke/t-5");
  assert.equal(shortLocation("machine:/a/b.yaml"), "machine:/a/b.yaml");
  assert.equal(shortLocation("/bare/path"), "/bare/path");
  assert.equal(shortLocation(null), "(empty)");
});

test("the picker narrows coarsely by name or chip", () => {
  const entries = jumpPickerEntries(WINDOWS, STATES);
  assert.equal(entries.filter((entry) => entryMatches(entry, "t-5")).length, 1);
  assert.equal(entries.filter((entry) => entryMatches(entry, "T-5")).length, 1, "case folds");
  assert.equal(
    entries.filter((entry) => entryMatches(entry, "a")).length,
    1,
    "a chip matches only its own window",
  );
  assert.equal(entries.filter((entry) => entryMatches(entry, "")).length, entries.length);
  assert.equal(entries.filter((entry) => entryMatches(entry, "nothing")).length, 0);
});
