// The window store's client contracts (t-14 "The suite",
// t-24@r20): the route grammar, the echo rule, the no-op write rule,
// the merge and move rules, and the interim store's read. The wire
// itself is pinned on the server side by the tracker's api tests;
// here the client's own rules are what is under test.

import test from "node:test";
import assert from "node:assert/strict";
import {
  deviceId,
  fieldsDiffer,
  interimDraftKeys,
  isOwnEcho,
  markMoved,
  mergeDrafts,
  windowsUrl,
  readInterimDrafts,
  stateShowingTask,
  type WindowDraft,
  type WindowState,
} from "../src/lib/windows.ts";

function state(patch: Partial<WindowState>): WindowState {
  return {
    id: "w-x",
    location: null,
    part: null,
    zoom: null,
    drafts: null,
    lastDevice: null,
    changed: 1,
    attachedHere: true,
    link: null,
    ...patch,
  };
}

test("window store routes resolve against the base without doubling the slash", () => {
  const base = new URL("http://127.0.0.1:8811");
  assert.equal(windowsUrl(base, ""), "http://127.0.0.1:8811/api/v1/windows/");
  assert.equal(windowsUrl(base, "states"), "http://127.0.0.1:8811/api/v1/windows/states");
  assert.equal(
    windowsUrl(base, "states/w-1"),
    "http://127.0.0.1:8811/api/v1/windows/states/w-1",
  );
  assert.equal(
    windowsUrl(new URL("http://host.example:8811/"), "devices/dev-1/attach"),
    "http://host.example:8811/api/v1/windows/devices/dev-1/attach",
  );
});

test("a device id is generated once and persisted on the device", () => {
  const store = new Map<string, string>();
  const saved = globalThis.localStorage;
  (globalThis as Record<string, unknown>).localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
  };
  try {
    const first = deviceId();
    assert.ok(first.length > 0);
    assert.equal(deviceId(), first, "the id persists");
    assert.equal(store.get("saccade.device.id"), first);
  } finally {
    (globalThis as Record<string, unknown>).localStorage = saved;
  }
});

test("a watch answer at or below the newest own write is an echo", () => {
  assert.equal(isOwnEcho(10, 10), true, "the write's own landing is an echo");
  assert.equal(isOwnEcho(9, 10), true, "older than the newest own write is history");
  assert.equal(isOwnEcho(11, 10), false, "a newer answer is someone else's change");
  assert.equal(isOwnEcho(3, 0), false, "a client that never wrote applies everything");
});

test("a settled write that changes nothing is not sent", () => {
  const current = {
    location: "tracker/smoke/t-5",
    part: "/content/2",
    zoom: 1.2,
    drafts: [{ path: "/content/2", text: "note" }] as WindowDraft[] | null,
  };
  assert.equal(fieldsDiffer({ location: "tracker/smoke/t-5" }, current), false);
  assert.equal(fieldsDiffer({ part: "/content/2" }, current), false);
  assert.equal(fieldsDiffer({ zoom: 1.2 }, current), false);
  assert.equal(
    fieldsDiffer({ drafts: [{ path: "/content/2", text: "note" }] }, current),
    false,
  );
  assert.equal(fieldsDiffer({ location: "tracker/smoke/t-7" }, current), true);
  assert.equal(fieldsDiffer({ part: "/content/0" }, current), true);
  assert.equal(fieldsDiffer({ zoom: 1.3 }, current), true);
  assert.equal(fieldsDiffer({ zoom: null }, current), true, "clearing is a change");
  assert.equal(fieldsDiffer({ drafts: [] }, current), true);
  assert.equal(fieldsDiffer({ part: null }, current), true);
});

test("the move merges drafts by path, the state's own keeping untouched paths", () => {
  const existing: WindowDraft[] = [
    { path: "/content/0", text: "window store's own" },
    { path: "/content/5", text: "window store keeps this" },
  ];
  const moved: WindowDraft[] = [
    { path: "/content/0", text: "the interim draft wins its slot" },
    { path: "/content/2", text: "carried over" },
  ];
  const merged = mergeDrafts(existing, moved);
  assert.deepEqual(merged, [
    { path: "/content/0", text: "the interim draft wins its slot" },
    { path: "/content/2", text: "carried over" },
    { path: "/content/5", text: "window store keeps this" },
  ]);
  assert.deepEqual(mergeDrafts(null, moved), moved);
  assert.equal(mergeDrafts(existing, []).length, 2);
});

test("the move lands in this device's attached state showing the task, else nowhere", () => {
  const states = [
    state({ id: "w-1", location: "tracker/smoke/t-5", attachedHere: true }),
    state({ id: "w-2", location: "tracker/smoke/t-5", attachedHere: false }),
    state({ id: "w-3", location: "tracker/smoke/t-7", attachedHere: true }),
  ];
  const found = stateShowingTask(states, "tracker/smoke/t-5");
  assert.equal(found?.id, "w-1", "the attached state showing the task, never the detached one");
  assert.equal(stateShowingTask(states, "tracker/smoke/t-9"), null);
});

test("the interim store reads only well-shaped drafts, and the move marks only confirmed keys", () => {
  const store = new Map<string, string>([
    ["saccade.annotation.drafts.smoke.t-5", JSON.stringify([{ path: "/content/1", text: "kept" }])],
    ["saccade.annotation.drafts.smoke.t-6", JSON.stringify({ not: "an array" })],
    ["saccade.annotation.drafts.smoke.t-7", "not json at all"],
    ["saccade.other.key", "{}"],
  ]);
  const saved = globalThis.localStorage;
  (globalThis as Record<string, unknown>).localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    key: (index: number) => [...store.keys()][index] ?? null,
    get length() {
      return store.size;
    },
  };
  try {
    assert.deepEqual(interimDraftKeys(), [
      "saccade.annotation.drafts.smoke.t-5",
      "saccade.annotation.drafts.smoke.t-6",
      "saccade.annotation.drafts.smoke.t-7",
    ]);
    assert.deepEqual(readInterimDrafts("saccade.annotation.drafts.smoke.t-5"), [
      { path: "/content/1", text: "kept" },
    ]);
    assert.deepEqual(readInterimDrafts("saccade.annotation.drafts.smoke.t-6"), []);
    assert.deepEqual(readInterimDrafts("saccade.annotation.drafts.smoke.t-7"), []);
    markMoved(["saccade.annotation.drafts.smoke.t-5"]);
    assert.deepEqual(interimDraftKeys().filter((key) => !JSON.parse(store.get("saccade.annotation.moved") ?? "[]").includes(key)), [
      "saccade.annotation.drafts.smoke.t-6",
      "saccade.annotation.drafts.smoke.t-7",
    ]);
  } finally {
    (globalThis as Record<string, unknown>).localStorage = saved;
  }
});

import { normalizeState } from "../src/lib/windows.ts";

test("the wire's absent fields normalize to null: undefined never reads as a value", () => {
  // the answered doc omits every cleared field (t-24@r20's shape), so
  // a raw answer carries undefined where the client's rules read null
  const answered = normalizeState({
    id: "w-x",
    location: undefined as unknown as null,
    part: undefined as unknown as null,
    zoom: undefined as unknown as null,
    drafts: undefined as unknown as null,
    lastDevice: undefined as unknown as null,
    changed: 3,
    attachedHere: true,
    link: "w-y",
  } as WindowState);
  assert.equal(answered.location, null);
  assert.equal(answered.part, null);
  assert.equal(answered.zoom, null);
  assert.equal(answered.drafts, null);
  assert.equal(answered.lastDevice, null);
  assert.equal(answered.link, "w-y");
  assert.equal(
    answered.location !== null,
    false,
    "an absent location is null, never undefined that reads as a value",
  );
});
