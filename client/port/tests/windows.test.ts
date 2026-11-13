// The window-store client (S18's elsewhere surface): the tracker's
// api/v1/windows wire contract in port idioms. The fakes here are
// controlled ports (the fetcher), never a re-stubbed policy.

import assert from "node:assert/strict";
import { test } from "vitest";

import { WindowStore, deviceId, windowsUrl } from "../src/lib/windows.ts";

type Call = { route: string; method: string; body: unknown };

function fakeStore(
  answers: Array<{ status: number; doc: unknown }>,
): { calls: Call[]; fetcher: (url: string, init: RequestInit) => Promise<Response> } {
  const calls: Call[] = [];
  let index = 0;
  return {
    calls,
    fetcher: (url, init) => {
      const route = url.replace("http://tracker.test/api/v1/windows/", "");
      let body: unknown = null;
      if (typeof init.body === "string") body = JSON.parse(init.body);
      calls.push({ route, method: init.method ?? "GET", body });
      const answer = answers[Math.min(index, answers.length - 1)]!;
      index += 1;
      return Promise.resolve(
        new Response(answer.doc === null ? "" : JSON.stringify(answer.doc), {
          status: answer.status,
        }),
      );
    },
  };
}

const stateDoc = (fields: Partial<{ id: string; link: string | null; location: string | null }> = {}) => ({
  state: {
    id: fields.id ?? "s-1",
    location: fields.location ?? null,
    part: null,
    zoom: null,
    drafts: null,
    last_device: null,
    changed: 3,
    attached_here: true,
    link: fields.link ?? null,
    ...fields,
  },
});

test("windowsUrl joins the store route on the tracker base", () => {
  assert.equal(
    windowsUrl("http://tracker.test", "states/s-1?device=d"),
    "http://tracker.test/api/v1/windows/states/s-1?device=d",
  );
});

test("deviceId answers a non-empty id on every call", () => {
  // in the node test env no localStorage exists, so each call is the
  // session-scoped fallback (a fresh id); persistence itself is a
  // browser capability, asserted through the store's explicit device
  assert.ok(deviceId().length > 0);
  assert.ok(deviceId().length > 0);
});

test("create posts the id and device; a 409 re-reads the state as it stands", async () => {
  const ok = fakeStore([{ status: 200, doc: stateDoc() }]);
  const store = new WindowStore({ base: "http://tracker.test", fetcher: ok.fetcher, device: "d-1" });
  const created = await store.create("s-1");
  assert.equal(created.id, "s-1");
  assert.deepEqual(ok.calls[0]!, {
    route: "states",
    method: "POST",
    body: { id: "s-1", device: "d-1" },
  });

  const conflict = fakeStore([
    { status: 409, doc: null },
    { status: 200, doc: stateDoc({ link: "s-2" }) },
  ]);
  const store2 = new WindowStore({ base: "http://tracker.test", fetcher: conflict.fetcher, device: "d-1" });
  const existing = await store2.create("s-1");
  assert.equal(existing.link, "s-2");
  assert.equal(conflict.calls[1]!.method, "GET");
});

test("setLink PUTs the device link; clearLink DELETEs it", async () => {
  const calls = fakeStore([{ status: 200, doc: stateDoc() }]);
  const store = new WindowStore({ base: "http://tracker.test", fetcher: calls.fetcher, device: "d-1" });
  await store.setLink("s-from", "s-to");
  assert.deepEqual(calls.calls[0]!, {
    route: "devices/d-1/links/s-from",
    method: "PUT",
    body: { to: "s-to" },
  });
  await store.clearLink("s-from");
  assert.equal(calls.calls[1]!.method, "DELETE");
});

test("a refused call names the route and status", async () => {
  const failing = fakeStore([{ status: 500, doc: { error: "boom" } }]);
  const store = new WindowStore({ base: "http://tracker.test", fetcher: failing.fetcher, device: "d-1" });
  await assert.rejects(
    () => store.getState("s-1"),
    /the window store refused states\/s-1 \(500\)/,
  );
});

test("attach and detach post the state against the device", async () => {
  const calls = fakeStore([{ status: 200, doc: stateDoc() }]);
  const store = new WindowStore({ base: "http://tracker.test", fetcher: calls.fetcher, device: "d-9" });
  await store.attach("s-1");
  await store.detach("s-1");
  assert.deepEqual(calls.calls[0]!, {
    route: "devices/d-9/attach",
    method: "POST",
    body: { state: "s-1" },
  });
  assert.equal(calls.calls[1]!.route, "devices/d-9/detach");
});

// The state document exactly as the tracker serializes it
// (tracker/src/serve/windows.rs StateDoc): snake_case names, and a
// field with no value is ABSENT, never null. The client read it
// as-is, so after a relaunch an absent location arrived as undefined
// and the restore threw on its length, and attached_here never
// matched attachedHere.
const wireState = {
  id: "s-1",
  last_device: "d-2",
  changed: 4,
  attached_here: true,
};
const normalized = {
  id: "s-1",
  location: null,
  part: null,
  zoom: null,
  width: null,
  drafts: null,
  lastDevice: "d-2",
  changed: 4,
  attachedHere: true,
  link: null,
};

test("a state read from the wire carries every field: absent ones null, names the client's", async () => {
  const wire = fakeStore([{ status: 200, doc: { windows: 1, seq: 4, state: wireState } }]);
  const store = new WindowStore({ base: "http://tracker.test", fetcher: wire.fetcher, device: "d-1" });
  assert.deepEqual(await store.getState("s-1"), normalized);
});

test("a listing from the wire is normalized state by state", async () => {
  const wire = fakeStore([{ status: 200, doc: { windows: 1, seq: 4, states: [wireState] } }]);
  const store = new WindowStore({ base: "http://tracker.test", fetcher: wire.fetcher, device: "d-1" });
  assert.deepEqual(await store.list(), [normalized]);
});
