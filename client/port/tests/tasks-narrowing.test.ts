// The task list's narrowing: its one spelling (lib/tasks/narrowing.ts),
// the frame's opaque carriage of it (the location grammar and the
// registry's canonical address), the echo the page must give back, and
// the tags the picker offers.

import assert from "node:assert/strict";
import { test, vi } from "vitest";

import { parseLocation, formatLocation } from "../src/lib/location.ts";
import { canonicalAddress } from "../src/lib/instruments/registry.ts";
import { instrumentForAddress } from "../src/lib/instruments/claims.ts";
import { tasks } from "../src/lib/instruments/index.ts";
import { echoMatches, narrowingOfPath, narrowingQuery, parseNarrowing } from "../src/lib/tasks/narrowing.ts";
import { pickList } from "../src/lib/picker/model.ts";
import { decodePage } from "../src/lib/pages.ts";
import { TrackerSource } from "../src/lib/tracker.ts";
import { sourceFor } from "../src/lib/sources.ts";

test("two spellings of one narrowing are one location", () => {
  const spelled = ["tag=UI,auth,ui", "tag=auth,ui", "tag=ui,AUTH", "tag=Auth,UI,ui,auth"];
  const canonical = spelled.map((opaque) => {
    const parsed = parseLocation(`proj/tasks?${opaque}`);
    assert.ok(parsed !== null, opaque);
    const verdict = canonicalAddress(parsed);
    assert.ok(verdict.ok, opaque);
    return formatLocation(verdict.address);
  });
  assert.deepEqual(new Set(canonical), new Set(["proj/tasks?tag=auth,ui"]));
});

test("the narrowing's spelling: tag names or untagged, nothing else", () => {
  const ok = parseNarrowing("untagged");
  assert.ok(ok.ok);
  assert.equal(ok.canonical, "untagged");
  for (const refused of ["tag=", "tag=a,", "tag=a,,b", "untagged,tag=a", "untaggedx", "tag=a&tag=b", "tags=a", "", "tag=a b", "tag=-a"]) {
    assert.equal(parseNarrowing(refused).ok, false, `'${refused}' is refused`);
  }
  assert.deepEqual(narrowingOfPath("tasks"), { ok: true, narrowing: null, path: "tasks" });
  const narrowed = narrowingOfPath("tasks?tag=B,a");
  assert.ok(narrowed.ok && narrowed.narrowing !== null);
  assert.equal(narrowed.path, "tasks?tag=a,b");
  assert.equal(narrowingQuery(narrowed.narrowing), "tag=a&tag=b");
  assert.equal(narrowingQuery({ tags: [], untagged: true }), "untagged");
});

test("the frame refuses a bad narrowing through the instrument, and one the instrument does not take", () => {
  const empty = canonicalAddress({ machine: "tracker:proj", path: "tasks?tag=" });
  assert.equal(empty.ok, false);
  const both = canonicalAddress({ machine: "tracker:proj", path: "tasks?tag=a&untagged" });
  assert.equal(both.ok, false);
  const board = canonicalAddress({ machine: "tracker:proj", path: "t-1?tag=a" });
  assert.equal(board.ok, false, "a board takes no narrowing");
  const plain = canonicalAddress({ machine: "tracker:proj", path: "tasks" });
  assert.deepEqual(plain, { ok: true, address: { machine: "tracker:proj", path: "tasks" } });
});

test("a narrowed list is the task list's, never the board's", () => {
  assert.equal(instrumentForAddress({ machine: "tracker:proj", path: "tasks?tag=auth" }), "tasks");
  assert.equal(instrumentForAddress({ machine: "tracker:proj", path: "tasks?untagged" }), "tasks");
});

test("the echo must be exactly the narrowing asked for", () => {
  const asked = { tags: ["auth", "ui"], untagged: false };
  assert.equal(echoMatches(asked, { tags: ["UI", "auth"], untagged: false }), true, "recorded names compare as the canonical form does");
  assert.equal(echoMatches(asked, undefined), false, "a tracker that does not narrow echoes nothing");
  assert.equal(echoMatches(asked, { tags: ["auth"], untagged: false }), false);
  assert.equal(echoMatches(asked, { tags: ["auth", "ui"], untagged: true }), false);
  assert.equal(echoMatches({ tags: [], untagged: true }, { tags: [], untagged: true }), true);
});

test("the page decodes its tag lines and its echo; a task page's names are not tag lines", () => {
  const root = decodePage({ view: { major: 1 }, kind: "root", location: "/", tags: [{ name: "auth", description: "sign-in" }] });
  assert.deepEqual(root.tags, [{ name: "auth", description: "sign-in" }]);
  const task = decodePage({ view: { major: 1 }, kind: "document", location: "t-0", tags: ["auth"] });
  assert.equal(task.tags, undefined);
  const page = decodePage({ view: { major: 1 }, kind: "document", location: "tasks", narrowing: { tags: ["auth"], untagged: false } });
  assert.deepEqual(page.narrowing, { tags: ["auth"], untagged: false });
  assert.equal(decodePage({ view: { major: 1 }, kind: "document", location: "tasks" }).narrowing, undefined);
});

test("the tasks instrument offers the repository's tags as picker candidates, each opening its narrowing", async () => {
  const source = sourceFor("tracker:tags-test") as TrackerSource;
  vi.spyOn(source, "instrumentPage").mockResolvedValue(
    decodePage({ view: { major: 1 }, kind: "document", location: "tasks", content: [] }),
  );
  vi.spyOn(source, "tags").mockResolvedValue([
    { name: "Auth", description: "sign-in and sessions" },
    { name: "ui", description: "" },
  ]);
  const candidates = await tasks.pickerCandidates!("tracker:tags-test");
  assert.deepEqual(candidates, [
    { id: "tasks?tag=auth", title: "Auth - sign-in and sessions" },
    { id: "tasks?tag=ui", title: "ui" },
  ]);
  const entries = pickList("auth", {
    recent: [],
    tasks: candidates,
    specOf: (id) => formatLocation({ machine: "tracker:tags-test", path: id }),
    openHere: null,
  });
  assert.equal(entries[0]?.spec, "tags-test/tasks?tag=auth");
});

test("a narrowed list opened before is the same picker entry as its candidate", () => {
  const entries = pickList("", {
    recent: ["tags-test/tasks?tag=auth"],
    tasks: [{ id: "tasks?tag=auth", title: "Auth - sign-in" }],
    specOf: (id) => `tags-test/${id}`,
    openHere: null,
  });
  assert.equal(entries.length, 1);
  assert.equal(entries[0]?.name, "Auth - sign-in");
});
