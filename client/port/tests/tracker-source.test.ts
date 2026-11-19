// The tracker source's contract: the port lives in two places - the
// declaration the app read and the address the server answers on - so
// the server is asked which repository it serves, and a mismatch with
// the registered path refuses naming both. The wire rides the asks'
// direct implementation with canned answers; no server here.

import assert from "node:assert/strict";
import { test, vi } from "vitest";
import { TrackerSource } from "../src/lib/tracker.ts";
import { directAsks, theProjects } from "../src/lib/asks.ts";

const REGISTERED = { machine: "box", path: "/srv/tasks-test" };

// this test file's own facts: the registered project's name (the
// declaration's), the repository the canned tracker says it serves,
// and a fresh project name per test so no earlier source's cached
// contract is ever handed back
let projectName = "";
let servedRepo: string | null = REGISTERED.path;
let counter = 0;

function stubAsks(): void {
  const root = (served: string | null): unknown => ({
    view: { major: 1 },
    kind: "document",
    location: "",
    seq: null,
    tracker: projectName,
    machine_id: "m-1",
    repo_path: served,
    board_grammar: { min: 1, max: 1 },
    view_majors: [1],
  });
  const tasks = (): unknown => ({
    view: { major: 1 },
    kind: "document",
    location: "tasks",
    seq: null,
    content: [],
  });
  const store = new Map<string, string>([["saccade.projects", JSON.stringify([`${REGISTERED.machine}:${REGISTERED.path}`])]]);
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => `name = "${projectName}"\n\n[tracker]\nport = 9\n`,
      runCommand: async () => {
        throw new Error("no commands here");
      },
      requestLocal: async (_machine, _address, route) => {
        const body = route === "/view/v1/" ? root(servedRepo) : tasks();
        return new TextEncoder().encode(JSON.stringify(body));
      },
    }),
  });
}

function freshSource(): TrackerSource {
  projectName = `tasks-test-${(counter += 1)}`;
  servedRepo = REGISTERED.path;
  return new TrackerSource(projectName);
}

test("a tracker serving another repository than the registered path refuses, naming both", async () => {
  stubAsks();
  const source = freshSource();
  await theProjects().register(REGISTERED);
  servedRepo = "/different/repo";
  await assert.rejects(
    source.instrumentPage("tasks"),
    (error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      assert.match(message, /\/different\/repo/);
      assert.match(message, /\/srv\/tasks-test/);
      assert.match(message, /different repositories/);
      return true;
    },
  );
});

test("a re-registration at another path re-checks, never replays the old contract", async () => {
  stubAsks();
  const source = freshSource();
  await theProjects().register(REGISTERED);
  assert.equal((await source.instrumentPage("tasks")).location, "tasks");
  // the same project name, now registered at another repository whose
  // tracker serves the old path: the cached contract must not stand in
  theProjects().forget(REGISTERED);
  await theProjects().register({ machine: "box", path: "/srv/other-repo" });
  servedRepo = REGISTERED.path;
  await assert.rejects(source.instrumentPage("tasks"), (error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    assert.match(message, /\/srv\/tasks-test/);
    assert.match(message, /\/srv\/other-repo/);
    return true;
  });
});

test("a tracker serving the registered repository's path loads", async () => {
  stubAsks();
  const source = freshSource();
  await theProjects().register(REGISTERED);
  const page = await source.instrumentPage("tasks");
  assert.equal(page.kind, "document");
  assert.equal(page.location, "tasks");
});
