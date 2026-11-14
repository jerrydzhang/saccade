// The window's place (lib/navigation/window-place.ts): what the window
// shows and where it reads reach its window state once they settle -
// never before the state binds, never per change, never a place the
// state already holds (a follow refresh of the same address), and
// never while a placement is still in flight.

import assert from "node:assert/strict";
import { test } from "vitest";
import type { RestoreStatus, RestoringPlace, SavedPlace } from "../src/lib/contracts/places.ts";
import {
  createWindowPlaceHolder,
  placementDone,
  shownPlaceOf,
  type WindowPlace,
} from "../src/lib/navigation/window-place.ts";

function holder(initial: WindowPlace | null = null) {
  let shown: WindowPlace | null = initial;
  const written: WindowPlace[] = [];
  let job: (() => void) | null = null;
  let scheduled = 0;
  const place = createWindowPlaceHolder({
    current: () => shown,
    write: (next) => {
      written.push(next);
    },
    schedule: (_ms, run) => {
      job = run;
      scheduled += 1;
      return scheduled;
    },
    cancel: () => {
      job = null;
    },
  });
  return {
    place,
    written,
    show(next: WindowPlace | null) {
      shown = next;
      place.changed();
    },
    pending: () => job !== null,
    settle: () => {
      const run = job;
      job = null;
      run?.();
    },
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

test("an open writes its location once the window settles, not per change", () => {
  const h = holder();
  h.place.bound({ location: null, part: null });
  h.show({ location: "host/repo/t-1", part: null });
  h.show({ location: "host/repo/t-1", part: "board/content/0" });
  h.show({ location: "host/repo/t-1", part: "board/content/4" });
  assert.deepEqual(h.written, [], "nothing is written before the settle");
  h.settle();
  assert.deepEqual(h.written, [{ location: "host/repo/t-1", part: "board/content/4" }]);
});

test("nothing is written before the state binds; the bind writes a change made before it", () => {
  const h = holder();
  h.show({ location: "host/repo/t-1", part: null });
  h.settle();
  assert.deepEqual(h.written, [], "an unbound window has no state to write");
  h.place.bound({ location: null, part: null });
  h.settle();
  assert.deepEqual(h.written, [{ location: "host/repo/t-1", part: null }]);
});

test("a place the state already holds is never written again (a follow refresh)", () => {
  const h = holder();
  h.place.bound({ location: "host/repo/t-1", part: "board/content/2" });
  h.show({ location: "host/repo/t-1", part: "board/content/2" });
  h.settle();
  assert.deepEqual(h.written, [], "the stored place is the shown place");
  h.show({ location: "host/repo/t-1", part: "board/content/3" });
  h.settle();
  h.show({ location: "host/repo/t-1", part: "board/content/3" });
  h.settle();
  assert.deepEqual(h.written, [{ location: "host/repo/t-1", part: "board/content/3" }]);
});

test("a pinned and a bare location are different places", () => {
  const h = holder();
  h.place.bound({ location: "host/repo/t-1", part: null });
  h.show({ location: "host/repo/t-1@r2", part: null });
  h.settle();
  h.show({ location: "host/repo/t-1", part: null });
  h.settle();
  assert.deepEqual(h.written, [
    { location: "host/repo/t-1@r2", part: null },
    { location: "host/repo/t-1", part: null },
  ]);
});

test("a window showing nothing settled writes nothing", () => {
  const h = holder();
  h.place.bound({ location: "host/repo/t-1", part: "board/content/5" });
  h.show(null);
  h.settle();
  assert.deepEqual(h.written, [], "a loading or empty window keeps the stored place");
});

test("a placement in flight holds the write until it ends", async () => {
  const h = holder();
  h.place.bound({ location: "host/repo/t-1", part: "board/content/9" });
  let finish: () => void = () => {};
  h.place.placing(new Promise<void>((resolve) => (finish = resolve)));
  // the restored board shows its top before the anchor lands
  h.show({ location: "host/repo/t-1", part: "board/content/0" });
  h.settle();
  assert.deepEqual(h.written, [], "a half-placed window never overwrites the place it restores");
  h.show({ location: "host/repo/t-1", part: "board/content/9" });
  finish();
  await flush();
  assert.equal(h.pending(), true, "the placement's end schedules the check");
  h.settle();
  assert.deepEqual(h.written, [], "the placed window is at its stored place");
});

test("a failed placement still releases the hold", async () => {
  const h = holder();
  h.place.bound({ location: null, part: null });
  h.place.placing(Promise.reject(new Error("refused")));
  h.show({ location: "host/repo/t-2", part: null });
  await flush();
  h.settle();
  assert.deepEqual(h.written, [{ location: "host/repo/t-2", part: null }]);
});

test("destroy cancels the pending write", () => {
  const h = holder();
  h.place.bound({ location: null, part: null });
  h.show({ location: "host/repo/t-1", part: null });
  h.place.destroy();
  assert.equal(h.pending(), false);
  h.show({ location: "host/repo/t-3", part: null });
  assert.equal(h.pending(), false, "a destroyed holder schedules nothing");
});

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

test("the plane's part is the project it shows; an empty plane has none (t-54)", () => {
  const plane = (path: string) => (): SavedPlace => ({
    bufferId: "main-tasks-1",
    address: { machine: "plane", path },
    position: { kind: "anchor", anchor: { kind: "task", task: "t-2" }, viewportOffset: { left: 0, top: 0 } },
  });
  assert.deepEqual(shownPlaceOf({ asked: "plane", status: "loaded" }, plane("pyrimel")), { location: "plane", part: "pyrimel" });
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

function job(initial: RestoreStatus) {
  let status = initial;
  const observers = new Set<(status: RestoreStatus) => void>();
  const place: RestoringPlace = {
    snapshot: () => status,
    subscribe(observer) {
      observers.add(observer);
      observer(status);
      return () => observers.delete(observer);
    },
    cancel() {},
  };
  return {
    place,
    finish(next: RestoreStatus) {
      status = next;
      for (const observer of [...observers]) observer(next);
    },
    observers: () => observers.size,
  };
}

test("a placement's end is its first settled status, whatever it is", async () => {
  const running = job("pending-content");
  const done = placementDone(running.place);
  running.finish("pending-layout");
  running.finish("defaulted");
  assert.equal(await done, "defaulted");
  assert.equal(running.observers(), 0, "the end unsubscribes");
  assert.equal(await placementDone(job("superseded").place), "superseded");
});
