// The shown place (observation.ts shownPlaceOf): the readout's window
// place - the location AS ASKED (a bare open shows its resolved
// revision; the place keeps it bare) and the board part at the reading
// height, from the trail's own capture. The holder that wrote a
// window's place anywhere is gone (t-71@r3: the app keeps no window
// state between runs); what a reader sees is pinned here.

import assert from "node:assert/strict";
import { test } from "vitest";
import type { SavedPlace } from "../src/lib/contracts/places.ts";
import { shownPlaceOf } from "../src/lib/observation.ts";

const address = { machine: "tracker:repo", path: "t-1@r3" };
const captured = (position: SavedPlace["position"]) => (): SavedPlace => ({ bufferId: "main-board-1", address, position });

test("the shown place is the ASKED location and the board part at the reading height", () => {
  const anchored = captured({
    kind: "anchor",
    anchor: { kind: "board-part", path: "/content/7" },
    viewportOffset: { left: 0, top: -12 },
  });
  // a bare open shows its resolved revision; the place keeps it bare
  assert.deepEqual(shownPlaceOf({ asked: "repo/t-1", status: "loaded" }, anchored), {
    location: "repo/t-1",
    part: "/content/7",
  });
  assert.deepEqual(shownPlaceOf({ asked: "repo/t-1@r3", status: "loaded" }, captured({ kind: "default" })), {
    location: "repo/t-1@r3",
    part: null,
  });
});

test("the plane remembers no project: its place is the bare plane (t-54; the human, 2026-11-14)", () => {
  const plane = (path: string) => (): SavedPlace => ({
    bufferId: "main-tasks-1",
    address: { machine: "plane", path },
    position: { kind: "anchor", anchor: { kind: "task", task: "t-2" }, viewportOffset: { left: 0, top: 0 } },
  });
  assert.deepEqual(shownPlaceOf({ asked: "plane", status: "loaded" }, plane("pyrimel")), { location: "plane", part: null });
  assert.deepEqual(shownPlaceOf({ asked: "plane", status: "loaded" }, plane("")), { location: "plane", part: null });
});

test("a task row, a loading frame, an empty frame and a dead capture are no place or no part", () => {
  const row = captured({ kind: "anchor", anchor: { kind: "task", task: "t-2" }, viewportOffset: { left: 0, top: 0 } });
  assert.deepEqual(shownPlaceOf({ asked: "repo/t-1", status: "loaded" }, row), { location: "repo/t-1", part: null });
  assert.equal(shownPlaceOf({ asked: "repo/t-1", status: "loading" }, row), null);
  assert.equal(shownPlaceOf({ asked: "", status: "loaded" }, row), null);
  assert.equal(shownPlaceOf(undefined, row), null);
  const dead = () => {
    throw new Error("capture from a destroyed buffer");
  };
  assert.deepEqual(shownPlaceOf({ asked: "repo/t-1", status: "invalid" }, dead), { location: "repo/t-1", part: null });
});
