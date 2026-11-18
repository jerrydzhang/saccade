// The device-local window store (t-71): the store's own contract in
// port idioms. The backend here is memory, a controlled port; there
// is no HTTP anywhere in the store (windows.test.ts still pins the
// old tracker client this wave).

import assert from "node:assert/strict";
import { test } from "vitest";

import { WindowStore } from "../src/lib/window-store.ts";

type Backend = { load(): Promise<string | null>; save(text: string): Promise<void> };

function memoryBackend(
  initial: string | null = null,
): { backend: Backend; text: () => string | null } {
  let stored: string | null = initial;
  return {
    backend: {
      load: () => Promise.resolve(stored),
      save: (text) => {
        stored = text;
        return Promise.resolve();
      },
    },
    text: () => stored,
  };
}

const liveOf =
  (...ids: string[]) =>
  (): Promise<ReadonlySet<string>> =>
    Promise.resolve(new Set(ids));

const full = { location: "t-71/board.yaml", part: "brief", zoom: 1.5, width: 0.8, drafts: [{ path: "notes", text: "hello" }] };
const FIELDS = ["location", "part", "zoom", "width", "drafts"] as const;

test("create then get round-trips the fields; the first save starts schema 1", async () => {
  const { backend, text } = memoryBackend();
  const store = new WindowStore({ backend, liveStates: liveOf() });
  const created = await store.create("s-1", full);
  assert.equal(created.id, "s-1");
  assert.equal(created.link, null);
  assert.equal(created.attachedHere, false);
  assert.deepEqual(await store.getState("s-1"), {
    id: "s-1",
    location: "t-71/board.yaml",
    part: "brief",
    zoom: 1.5,
    width: 0.8,
    drafts: [{ path: "notes", text: "hello" }],
    attachedHere: false,
    link: null,
  });
  assert.deepEqual(JSON.parse(text() ?? "{}"), {
    schema: 1,
    states: {
      "s-1": { location: "t-71/board.yaml", part: "brief", zoom: 1.5, width: 0.8, drafts: [{ path: "notes", text: "hello" }] },
    },
  });
});

test("create on an id that exists returns it as it stands", async () => {
  const { backend, text } = memoryBackend();
  const store = new WindowStore({ backend, liveStates: liveOf("s-1") });
  await store.create("s-1", { location: "first" });
  const again = await store.create("s-1", { location: "second" });
  assert.equal(again.location, "first");
  assert.deepEqual(await store.getState("s-1"), {
    id: "s-1",
    location: "first",
    part: null,
    zoom: null,
    width: null,
    drafts: null,
    attachedHere: true,
    link: null,
  });
  assert.deepEqual((JSON.parse(text() ?? "{}") as { states: Record<string, unknown> }).states["s-1"], {
    location: "first",
  });
});

test("patch: absent fields stay, null clears, for each field", async () => {
  const { backend } = memoryBackend();
  const store = new WindowStore({ backend, liveStates: liveOf() });
  await store.create("s-1", full);
  await store.patch("s-1", {});
  for (const field of FIELDS) {
    await store.patch("s-1", { [field]: null });
    const state = await store.getState("s-1");
    assert.equal(state[field] ?? null, null, `${field} clears`);
    for (const other of FIELDS) {
      if (other === field) continue;
      assert.deepEqual(state[other], full[other], `${other} stays when ${field} clears`);
    }
    await store.patch("s-1", { [field]: full[field] });
  }
  assert.deepEqual(await store.getState("s-1"), {
    id: "s-1",
    location: full.location,
    part: full.part,
    zoom: full.zoom,
    width: full.width,
    drafts: full.drafts,
    attachedHere: false,
    link: null,
  });
});

test("a state the store never held refuses a read and a patch", async () => {
  const { backend } = memoryBackend();
  const store = new WindowStore({ backend, liveStates: liveOf() });
  await assert.rejects(() => store.getState("s-404"), /holds no state s-404/);
  await assert.rejects(() => store.patch("s-404", { zoom: 2 }), /holds no state s-404/);
});

test("setLink keeps the link on the from-state alone", async () => {
  const { backend, text } = memoryBackend();
  const store = new WindowStore({ backend, liveStates: liveOf("s-1", "s-2") });
  await store.create("s-1");
  await store.create("s-2");
  await store.setLink("s-1", "s-2");
  assert.equal((await store.getState("s-1")).link, "s-2");
  assert.equal((await store.getState("s-2")).link, null);
  const states = (JSON.parse(text() ?? "{}") as { states: Record<string, { link?: string }> }).states;
  assert.deepEqual(states["s-1"], { link: "s-2" });
  assert.deepEqual(states["s-2"], {});
});

test("setLink refuses when a side carries no live window, and on a self-link", async () => {
  const { backend, text } = memoryBackend();
  const store = new WindowStore({ backend, liveStates: liveOf("s-1") });
  await store.create("s-1");
  await assert.rejects(() => store.setLink("s-9", "s-1"), /no live window carries s-9/);
  await assert.rejects(() => store.setLink("s-1", "s-9"), /no live window carries s-9/);
  await assert.rejects(() => store.setLink("s-1", "s-1"), /does not link to itself/);
  assert.deepEqual((JSON.parse(text() ?? "{}") as { states: Record<string, unknown> }).states, { "s-1": {} });
});

test("clearLink unlinks once and changes nothing when there is no link", async () => {
  const { backend, text } = memoryBackend();
  const store = new WindowStore({ backend, liveStates: liveOf("s-1", "s-2") });
  await store.create("s-1");
  await store.create("s-2");
  await store.setLink("s-1", "s-2");
  await store.clearLink("s-1");
  assert.equal((await store.getState("s-1")).link, null);
  const before = text();
  await store.clearLink("s-1");
  assert.equal(text(), before, "an unlink with no link writes nothing");
  await store.clearLink("s-3");
  assert.equal(text(), before, "an unlink of an unknown state writes nothing");
});

test("attachedHere is derived from the live states, never stored", async () => {
  const { backend, text } = memoryBackend();
  const live = new Set<string>(["s-1"]);
  const store = new WindowStore({ backend, liveStates: () => Promise.resolve(live) });
  await store.create("s-1");
  await store.create("s-2");
  assert.equal((await store.getState("s-1")).attachedHere, true);
  assert.equal((await store.getState("s-2")).attachedHere, false);
  live.delete("s-1");
  assert.equal((await store.getState("s-1")).attachedHere, false, "the answer follows the live set");
  assert.equal((text() ?? "").includes("attached"), false, "the document stores no attach fact");
});

test("a document whose schema is not 1 refuses and names the found version", async () => {
  const document = '{"schema":2,"states":{}}';
  const { backend, text } = memoryBackend(document);
  const store = new WindowStore({ backend, liveStates: liveOf() });
  await assert.rejects(() => store.getState("s-1"), /schema is 2, not 1/);
  await assert.rejects(() => store.create("s-9"), /schema is 2, not 1/);
  assert.equal(text(), document, "a refused document is not rewritten");
});

test("a malformed document refuses with its reason", async () => {
  const cases: [string, RegExp][] = [
    ["not json", /does not parse/],
    ["[]", /not a document object/],
    ['{"states":{}}', /names no schema/],
    ['{"schema":1}', /no states map/],
    ['{"schema":1,"states":{"s-1":7}}', /state s-1 is not an object/],
  ];
  for (const [document, reason] of cases) {
    const { backend } = memoryBackend(document);
    const store = new WindowStore({ backend, liveStates: liveOf() });
    await assert.rejects(() => store.getState("s-1"), reason, document);
  }
});

test("a second store over the same backend sees the first one's saved state", async () => {
  const { backend } = memoryBackend();
  const first = new WindowStore({ backend, liveStates: liveOf("s-1", "s-2") });
  await first.create("s-1", full);
  await first.create("s-2", { location: "elsewhere" });
  await first.setLink("s-1", "s-2");
  await first.patch("s-2", { zoom: 3, drafts: null });
  const second = new WindowStore({ backend, liveStates: liveOf("s-1") });
  const state = await second.getState("s-1");
  assert.deepEqual(state, {
    id: "s-1",
    location: full.location,
    part: full.part,
    zoom: full.zoom,
    width: full.width,
    drafts: full.drafts,
    attachedHere: true,
    link: "s-2",
  });
  const other = await second.getState("s-2");
  assert.equal(other.location, "elsewhere");
  assert.equal(other.zoom, 3);
  assert.equal(other.drafts, null);
  assert.equal(other.attachedHere, false);
  assert.equal(other.link, null);
});
