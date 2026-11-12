// The picker model's tiers and orders, pinned to the m-490 spec
// (the prototype's own rules, transcribed).

import assert from "node:assert/strict";
import test from "node:test";

import { isPathMode, pickList, scoreOf, type PickContext, type PickEntry } from "../src/lib/picker/model.ts";

const context: PickContext = {
  recent: ["hub:8811/seedtracker/t-3@r2", "hub:8811/seedtracker/t-7"],
  tasks: [
    { id: "t-3", title: "the landing seed account" },
    { id: "t-7", title: "group-made fixture" },
    { id: "t-12", title: "cited board apparatus" },
  ],
  specOf: (id) => `hub:8811/seedtracker/${id}`,
  openHere: null,
};

test("an empty query lists everything - the list is the discovery", () => {
  const list = pickList("", context);
  assert.equal(list.length, 3);
  // recent first, newest first: the recent array is newest-first, so
  // t-3 (recent[0], its pinned spec) then t-7, then the unopened t-12
  assert.deepEqual(list.map((entry) => entry.id), ["t-3", "t-7", "t-12"]);
});

test("the tiers score exactly as the prototype's (m-490 verbatim)", () => {
  const entry = (id: string, name: string, spec: string): PickEntry =>
    ({ kind: "board", id, name, spec, recentIndex: null });
  const e = entry("t-12", "cited board apparatus", "hub:8811/seedtracker/t-12");
  assert.equal(scoreOf("12", e), 1000, "all-digits equal to the task number");
  assert.equal(scoreOf("t-12", e), 950, "exact id");
  assert.equal(scoreOf("t-1", e), 800, "id starts with the query");
  assert.equal(scoreOf("cited", e), 600, "word start in id+name");
  assert.equal(scoreOf("ited", e), 450, "substring of id+name");
  assert.equal(scoreOf("seedtracker/t-1", e), 300, "in the path");
  assert.equal(scoreOf("cba", e), 100, "scattered letters (subsequence)");
  assert.equal(scoreOf("zzz", e), 0, "no match drops the entry");
});

test("a digits query ranks the exact task number above prefix matches", () => {
  const list = pickList("3", context);
  assert.equal(list[0]?.id, "t-3", "1000 beats the 800 of t-3x-style prefixes");
  // t-12 does not match '3' at any tier beyond subsequence absence
  assert.ok(list.every((entry) => entry.id !== "t-12") || scoreOf("3", list[list.length - 1]!) < 800);
});

test("fuzzy by title: a word of the title finds the task", () => {
  const list = pickList("apparatus", context);
  assert.deepEqual(list.map((entry) => entry.id), ["t-12"]);
});

test("ties: the location open here sorts last, then recency", () => {
  const here: PickContext = { ...context, openHere: "hub:8811/seedtracker/t-7" };
  const list = pickList("", here);
  assert.equal(list[list.length - 1]?.id, "t-7", "the open-here location sorts last on the empty query");
});

test("path mode enters on /, ~ or machine: prefixes", () => {
  assert.equal(isPathMode("/abs/path"), true);
  assert.equal(isPathMode("~/notes"), true);
  assert.equal(isPathMode("desk:/home/jerry"), true);
  assert.equal(isPathMode("t-3"), false);
  assert.equal(isPathMode("apparatus"), false);
});
