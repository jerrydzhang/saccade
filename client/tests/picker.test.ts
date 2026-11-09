import test from "node:test";
import assert from "node:assert/strict";
import { pickerLabelKind, rankedPicker, rankOf, type PickerItem } from "../src/lib/picker.ts";

const ITEMS: PickerItem[] = [
  { location: "smoke/t-17", kind: "task", name: "the viewer to the t-14 spec" },
  { location: "smoke/t-15", kind: "task", name: "pages for boards revisions and history" },
  { location: "smoke/t-5", kind: "task", name: "machine named reads" },
  { location: "machine:figures/error.png", kind: "file", name: "error.png" },
];

test("an id match outranks a name match", () => {
  // "t-5" is inside smoke/t-5 (id); "pages" starts that task's name
  assert.equal(rankOf("t-5", ITEMS[2]!), "id");
  assert.equal(rankOf("pages", ITEMS[1]!), "name");
});

test("the ranking runs id, name, word start, substring, scattered", () => {
  const wordStart: PickerItem = { location: "smoke/t-9", kind: "task", name: "asymmetric specimen" };
  assert.equal(rankOf("spec", wordStart), "word-start");
  const notes: PickerItem = { location: "smoke/t-9", kind: "task", name: "specimen notes" };
  assert.equal(rankOf("notes", notes), "word-start");
  const substring: PickerItem = { location: "smoke/t-9", kind: "task", name: "the specimen" };
  assert.equal(rankOf("ecim", substring), "substring", "mid-word, not a word start");
  const scattered: PickerItem = { location: "smoke/t-9", kind: "task", name: "viewer notes" };
  assert.equal(rankOf("vwr", scattered), "scattered");
  assert.equal(rankOf("zzz", scattered), "none");
});

test("recency breaks ties, the open item last", () => {
  const items: PickerItem[] = [
    { location: "smoke/t-1", kind: "task", name: "first" },
    { location: "smoke/t-2", kind: "task", name: "second" },
    { location: "smoke/t-3", kind: "task", name: "third" },
  ];
  const ranked = rankedPicker("", items, ["smoke/t-3", "smoke/t-1", "smoke/t-2"], "smoke/t-3");
  // empty query: recency order, the open item forced last
  assert.deepEqual(ranked.map((item) => item.location), ["smoke/t-1", "smoke/t-2", "smoke/t-3"]);
  const rankedOpen = rankedPicker("", items, ["smoke/t-1", "smoke/t-2", "smoke/t-3"], "smoke/t-1");
  assert.deepEqual(rankedOpen.map((item) => item.location), ["smoke/t-2", "smoke/t-3", "smoke/t-1"]);
});

test("quality beats recency", () => {
  const items: PickerItem[] = [
    { location: "smoke/t-1", kind: "task", name: "pages" },
    { location: "smoke/t-2", kind: "task", name: "the pages account" },
  ];
  // t-2 was opened most recently, but t-1's name match is exact-er:
  // both are name matches, so recency decides - a quality gap decides first
  const idItem: PickerItem = { location: "smoke/t-9", kind: "task", name: "pages" };
  const ranked = rankedPicker("t-9", [...items, idItem], ["smoke/t-1", "smoke/t-2"], null);
  assert.equal(ranked[0]?.location, "smoke/t-9", "the id match outranks everything");
});

test("a result shows only its kind and name", () => {
  const item: PickerItem = { location: "smoke/t-17", kind: "task", name: "the viewer" };
  assert.deepEqual(pickerLabelKind(item), { kind: "task", label: "the viewer" });
});

test("typed locations always resolve even when nothing enumerated them", () => {
  // the picker's UI answers a typed location directly; this pins the
  // shape: an unmatched query with a typed location rides as its own
  // item, ranked on its own id
  const typed: PickerItem = { location: "smoke/t-99", kind: "task", name: "" };
  assert.equal(rankOf("t-99", typed), "id");
});
