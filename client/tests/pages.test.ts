import test from "node:test";
import assert from "node:assert/strict";
import {
  decodePage,
  fetchFileBytes,
  fetchPage,
  pageFact,
  type Fetcher,
} from "../src/lib/pages.ts";
import {
  TrackerSource,
  homeTracker,
  knownTrackers,
  registerTracker,
  serverBase,
  trackerBase,
  trackerIsThisMachine,
} from "../src/lib/tracker.ts";

// the one viewer setting rides localStorage in the app; node tests run
// with a minimal in-memory stand-in
const settingStore = new Map<string, string>();
(globalThis as Record<string, unknown>).localStorage ??= {
  getItem: (key: string) => settingStore.get(key) ?? null,
  setItem: (key: string, value: string) =>
    void settingStore.set(key, String(value)),
  removeItem: (key: string) => void settingStore.delete(key),
};

// -- the decoders ------------------------------------------------------

const ROOT = {
  view: { major: 1 },
  kind: "root",
  location: "/",
  seq: 9,
  tracker: "smoke",
  view_majors: [1],
  board_grammar: { min: 1, max: 1 },
  content: [{ kind: "fact", label: "family", value: "boards" }],
};

const BOARD_PAGE = {
  view: { major: 1 },
  kind: "document",
  location: "t-7@r2",
  seq: 9,
  task: "t-7",
  revision: 2,
  content: [
    { kind: "fact", label: "task", value: "t-7" },
    { kind: "fact", label: "name", value: "the cited account" },
    { kind: "fact", label: "state", value: "open" },
    { kind: "fact", label: "revision", value: "r2 of 2" },
    { kind: "fact", label: "commit", value: "a".repeat(40) },
    { kind: "ref", location: "t-7/history", label: "history" },
    { kind: "ref", location: "t-7@r2/file/board.yaml", label: "board.yaml" },
    {
      kind: "group",
      title: "incorporates t-8@r1",
      location: "t-7@r2:/content/1",
      content: [
        { kind: "ref", location: "t-8@r1", label: "source revision t-8@r1" },
        {
          kind: "text",
          text: "receiving",
          location: "t-7@r2:/content/1/incorporation/content/0",
        },
        {
          kind: "ref",
          location: "t-8@r1/content/0",
          label: "cites /content/0",
        },
      ],
    },
    {
      kind: "figure",
      caption: "a plot",
      file: "t-7@r2/file/figures/plot.png",
      location: "t-7@r2:/content/2",
    },
    { kind: "horizon", at: 5 },
  ],
};

const TIMELINE = {
  view: { major: 1 },
  kind: "timeline",
  location: "t-7/history",
  seq: 12,
  task: "t-7",
  revision: 2,
  entries: [
    {
      seq: 4,
      event: "task_created",
      actor: "smoke",
      tier: "agent",
      at: 1,
      revision: null,
      commit: null,
      message: null,
      grounds: null,
      explanation: null,
      session: null,
    },
    {
      seq: 8,
      event: "task_committed",
      actor: "smoke",
      tier: "agent",
      at: 2,
      revision: 1,
      commit: "b".repeat(40),
      message: "first",
      grounds: null,
      explanation: null,
      session: { kind: "ref", location: "i-9", label: "session i-9" },
      refs: [
        { kind: "ref", location: "t-7@r1", label: "revision r1" },
        { kind: "ref", location: "t-7@r1/diff", label: "diff" },
      ],
    },
  ],
};

function ok(body: unknown, type = "application/json"): Fetcher {
  return async () =>
    new Response(typeof body === "string" ? body : JSON.stringify(body), {
      status: 200,
      headers: { "content-type": type },
    });
}

test("the root page decodes with its contract fields", async () => {
  const page = await fetchPage(ok(ROOT), "http://x/view/v1/");
  assert.equal(page.kind, "root");
  assert.equal(page.tracker, "smoke");
  assert.deepEqual(page.boardGrammar, { min: 1, max: 1 });
  assert.deepEqual(page.viewMajors, [1]);
  assert.equal(page.seq, 9);
});

test("a refusal document raises with its code and its teaching text", async () => {
  const refusal = {
    view: { major: 1 },
    kind: "refusal",
    location: "t-99",
    seq: 3,
    code: "unknown_task",
    content: [
      { kind: "fact", label: "refusal", value: "unknown_task" },
      { kind: "text", text: "no task t-99 exists here; list the tasks first" },
    ],
  };
  await assert.rejects(
    fetchPage(ok(refusal), "http://x/view/v1/t-99"),
    (error: Error) =>
      error.message.includes("unknown_task") &&
      error.message.includes("no task t-99"),
  );
});

test("an unknown part kind rides through as raw, never refusing the page", async () => {
  const page = decodePage(BOARD_PAGE);
  const last = page.content?.[page.content.length - 1];
  assert.equal(last?.kind, "other");
  assert.equal(last && last.kind === "other" ? last.name : "", "horizon");
});

test("a board page keeps locations, figures and incorporation collapse", () => {
  const page = decodePage(BOARD_PAGE);
  const group = page.content?.[7];
  assert.equal(
    group && group.kind === "group" ? group.location : "",
    "t-7@r2:/content/1",
  );
  const figure = page.content?.[8];
  assert.equal(
    figure && figure.kind === "figure" ? figure.file : "",
    "t-7@r2/file/figures/plot.png",
  );
  const groups = (page.content ?? []).filter((part) => part.kind === "group");
  assert.equal(groups.length, 1);
});

test("pageFact reads the facts the picker and header need", () => {
  const page = decodePage(BOARD_PAGE);
  assert.equal(pageFact(page, "name"), "the cited account");
  assert.equal(pageFact(page, "revision"), "r2 of 2");
  assert.equal(pageFact(page, "absent"), null);
});

test("a timeline decodes entries with session refs and page refs", () => {
  const page = decodePage(TIMELINE);
  assert.equal(page.entries?.length, 2);
  const commit = page.entries?.[1];
  assert.equal(commit?.session?.location, "i-9");
  const firstRef = commit?.refs?.[0];
  assert.equal(firstRef === undefined ? "" : firstRef.location, "t-7@r1");
  assert.equal(commit?.commit, "b".repeat(40));
});

test("a non-JSON answer is a structured refusal, never a crash", async () => {
  const bad: Fetcher = async () =>
    new Response("<html>nope</html>", { status: 200 });
  await assert.rejects(fetchPage(bad, "http://x/view/v1/"), /non-JSON/);
});

test("a bare location re-resolves on every open; a later open never replays an old current", async () => {
  const log: string[] = [];
  let revision = 1;
  const routes: Record<string, Route> = {
    "/view/v1/": { body: () => ROOT },
    "/view/v1/t-7": {
      body: () => ({ ...BOARD_PAGE, location: `t-7@r${revision}`, revision }),
    },
    "/view/v1/t-7@r1/file/board.yaml": { body: () => "grammar: 1\ntask: t-7\ncontent: []\n", type: "text/yaml" },
    "/view/v1/t-7@r2/file/board.yaml": { body: () => "grammar: 1\ntask: t-7\ncontent:\n  - text: second\n", type: "text/yaml" },
  };
  const source = new TrackerSource("smoke", wire(routes, log));
  const restore = withHost("127.0.0.1:8811");
  try {
    const first = await source.loadText({ machine: "tracker:smoke", path: "t-7" });
    assert.equal(first.location.path, "t-7@r1");
    revision = 2;
    const second = await source.loadText({ machine: "tracker:smoke", path: "t-7" });
    assert.equal(second.location.path, "t-7@r2", "the second open answered the revision that landed");
    assert.match(second.text, /second/);
    const bareResolutions = log.filter((p) => p === "/view/v1/t-7");
    assert.equal(bareResolutions.length, 2, "every bare open resolved through the board page");
  } finally {
    restore();
  }
});

// -- the source over the wire -----------------------------------------

type Route = { body: (path: string) => unknown; type?: string };

function wire(routes: Record<string, Route>, log: string[]): Fetcher {
  return async (url: string) => {
    const path = new URL(url).pathname;
    log.push(path);
    // exact routes first: the root never swallows a longer route
    const exact = Object.entries(routes).find(([prefix]) => prefix === path);
    const prefix =
      exact ??
      Object.entries(routes).find(
        ([key]) => key !== "/view/v1/" && path.startsWith(key),
      );
    if (prefix !== undefined) {
      const body = prefix[1].body(path);
      if (body instanceof Uint8Array || typeof body === "string") {
        return new Response(body as BodyInit, {
          status: 200,
          headers: { "content-type": prefix[1].type ?? "application/json" },
        });
      }
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(
      JSON.stringify({
        view: { major: 1 },
        kind: "refusal",
        location: path,
        seq: 1,
        code: "unknown_location",
        content: [
          { kind: "fact", label: "refusal", value: "unknown_location" },
          { kind: "text", text: "no such route" },
        ],
      }),
      { status: 404, headers: { "content-type": "application/json" } },
    );
  };
}

function withHost(value: string | null): () => void {
  const saved = localStorage.getItem("saccade.server.host");
  if (value === null) localStorage.removeItem("saccade.server.host");
  else localStorage.setItem("saccade.server.host", value);
  return () => {
    if (saved === null) localStorage.removeItem("saccade.server.host");
    else localStorage.setItem("saccade.server.host", saved);
  };
}

test("a bare location resolves to its pinned revision and reads one immutable tree", async () => {
  const log: string[] = [];
  const yaml = "grammar: 1\ntask: t-7\ncontent:\n  - text: hello\n";
  const routes: Record<string, Route> = {
    "/view/v1/": { body: () => ROOT },
    "/view/v1/t-7": {
      body: (path) => (path === "/view/v1/t-7" ? BOARD_PAGE : BOARD_PAGE),
    },
    "/view/v1/t-7@r2/file/board.yaml": { body: () => yaml, type: "text/yaml" },
    "/view/v1/t-7@r2/file/figures/plot.png": {
      body: () => new Uint8Array([137, 80]),
      type: "image/png",
    },
  };
  const source = new TrackerSource("smoke", wire(routes, log));
  const restore = withHost("127.0.0.1:8811");
  try {
    const loaded = await source.loadText({
      machine: "tracker:smoke",
      path: "t-7",
    });
    assert.equal(
      loaded.location.path,
      "t-7@r2",
      "the bare location resolved to the pinned revision",
    );
    assert.match(loaded.text, /hello/);
    const asset = await source.loadAsset(
      { machine: "tracker:smoke", path: "t-7@r2" },
      "figures/plot.png",
    );
    assert.deepEqual([...asset.bytes], [137, 80]);
    // the bare page resolved exactly once; the files rode the pinned
    // routes; a pinned open's facts fetch runs out of band and is not
    // part of the open's critical path
    const pageLog = log.filter((p) => p === "/view/v1/t-7");
    assert.equal(
      pageLog.length,
      1,
      `the bare page resolved once: ${log.join(", ")}`,
    );
  } finally {
    restore();
  }
});

test("the contract refuses a wrong tracker name before any board read", async () => {
  const log: string[] = [];
  const routes: Record<string, Route> = {
    "/view/v1/": { body: () => ({ ...ROOT, tracker: "other" }) },
    "/view/v1/t-7": { body: () => BOARD_PAGE },
  };
  const source = new TrackerSource("smoke", wire(routes, log));
  const restore = withHost("127.0.0.1:8811");
  try {
    await assert.rejects(
      source.loadText({ machine: "tracker:smoke", path: "t-7" }),
      /serves tracker 'other'/,
    );
    assert.equal(
      log.filter((p) => p.includes("t-7")).length,
      0,
      "no board read happened",
    );
  } finally {
    restore();
  }
});

test("a grammar range excluding the viewer refuses before any board read", async () => {
  const log: string[] = [];
  const routes: Record<string, Route> = {
    "/view/v1/": {
      body: () => ({ ...ROOT, board_grammar: { min: 2, max: 2 } }),
    },
    "/view/v1/t-7": { body: () => BOARD_PAGE },
  };
  const source = new TrackerSource("smoke", wire(routes, log));
  const restore = withHost("127.0.0.1:8811");
  try {
    await assert.rejects(
      source.loadText({ machine: "tracker:smoke", path: "t-7" }),
      /move the binaries in step/,
    );
    assert.equal(
      log.filter((p) => p.includes("t-7")).length,
      0,
      "no board read happened",
    );
  } finally {
    restore();
  }
});

test("a refusal page answers a missing board as a structured error", async () => {
  const log: string[] = [];
  const refusal = {
    view: { major: 1 },
    kind: "refusal",
    location: "t-99",
    seq: 1,
    code: "unknown_task",
    content: [
      { kind: "fact", label: "refusal", value: "unknown_task" },
      { kind: "text", text: "no task t-99 exists here" },
    ],
  };
  const routes: Record<string, Route> = {
    "/view/v1/": { body: () => ROOT },
    "/view/v1/t-99": { body: () => refusal },
  };
  const source = new TrackerSource("smoke", wire(routes, log));
  const restore = withHost("127.0.0.1:8811");
  try {
    await assert.rejects(
      source.loadText({ machine: "tracker:smoke", path: "t-99" }),
      (error: Error) =>
        error.message.includes("unknown_task") &&
        error.message.includes("no task t-99"),
    );
  } finally {
    restore();
  }
});

test("the window remembers the tasks it opened, newest first", async () => {
  const log: string[] = [];
  const routes: Record<string, Route> = {
    "/view/v1/": { body: () => ROOT },
    "/view/v1/t-7": { body: () => BOARD_PAGE },
    "/view/v1/t-7@r2/file/board.yaml": {
      body: () => "grammar: 1\ntask: t-7\ncontent: []\n",
      type: "text/yaml",
    },
  };
  const source = new TrackerSource("smoke", wire(routes, log));
  const restore = withHost("127.0.0.1:8811");
  try {
    await source.loadText({ machine: "tracker:smoke", path: "t-7" });
    assert.equal(source.seenTasks.length, 1);
    assert.deepEqual(source.seenTasks[0], {
      spec: "t-7@r2",
      name: "the cited account",
    });
  } finally {
    restore();
  }
});

test("the file family serves bytes with the type the extension names", async () => {
  const raw: Fetcher = async () =>
    new Response(new Uint8Array([1, 2, 3]) as unknown as BodyInit, {
      status: 200,
      headers: { "content-type": "image/png" },
    });
  const bytes = await fetchFileBytes(raw, "http://x/f.png");
  assert.deepEqual([...bytes], [1, 2, 3]);
});

test("a file refusal raises with its code", async () => {
  const refusal: Fetcher = async () =>
    new Response(
      JSON.stringify({
        view: { major: 1 },
        kind: "refusal",
        location: "x",
        seq: 1,
        code: "file_missing",
        content: [
          { kind: "fact", label: "refusal", value: "file_missing" },
          { kind: "text", text: "no file" },
        ],
      }),
      { status: 404, headers: { "content-type": "application/json" } },
    );
  await assert.rejects(
    fetchFileBytes(refusal, "http://x/f.png"),
    /file_missing/,
  );
});

test("the hub proof is machine identity, never address shape", async () => {
  // t-24@r14: the root page's machine id compared with this device's
  // own; a loopback server address proves nothing, an unavailable or
  // empty id on either side proves nothing, and absent never equals
  // absent.
  const restore = withHost("127.0.0.1:8811");
  try {
    const equal: Fetcher = wire(
      { "/view/v1/": { body: () => ({ ...ROOT, machine_id: "id-9" }) } },
      [],
    );
    assert.equal(
      await trackerIsThisMachine(async () => "id-9", equal),
      true,
      "equal ids are the hub",
    );
    const other: Fetcher = wire(
      { "/view/v1/": { body: () => ({ ...ROOT, machine_id: "other" }) } },
      [],
    );
    assert.equal(
      await trackerIsThisMachine(async () => "id-9", other),
      false,
      "a different machine id is a spoke even over a loopback address",
    );
    const absent: Fetcher = wire({ "/view/v1/": { body: () => ({ ...ROOT }) } }, []);
    assert.equal(
      await trackerIsThisMachine(async () => "id-9", absent),
      false,
      "an absent tracker id never proves the hub",
    );
    const absentBoth: Fetcher = wire({ "/view/v1/": { body: () => ({ ...ROOT }) } }, []);
    assert.equal(
      await trackerIsThisMachine(async () => null, absentBoth),
      false,
      "null never equals null",
    );
    assert.equal(
      await trackerIsThisMachine(async () => "", equal),
      false,
      "an empty own id is never the hub",
    );
  } finally {
    restore();
  }
});

function withRegistry(): () => void {
  const keys = ["saccade.trackers", "saccade.trackers.home"];
  const saved = keys.map((key) => localStorage.getItem(key));
  for (const key of keys) localStorage.removeItem(key);
  return () => {
    keys.forEach((key, i) => {
      const value = saved[i];
      if (value === null || value === undefined) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    });
  };
}

test("serverBase refuses with the typed form when no tracker was reached", () => {
  const restore = withHost(null);
  const restoreRegistry = withRegistry();
  try {
    assert.throws(() => serverBase(), /no tracker reached yet; type a location with its host once/);
    assert.throws(() => trackerBase("saccade"), /host:8811\/saccade\/tasks/);
  } finally {
    restoreRegistry();
    restore();
  }
});

test("a reached tracker is remembered by name; the first is the home", () => {
  const restore = withHost(null);
  const restoreRegistry = withRegistry();
  try {
    registerTracker("saccade", "hub:8811");
    registerTracker("pyrimel", "10.0.0.5:8812");
    assert.deepEqual(knownTrackers(), { saccade: "hub:8811", pyrimel: "10.0.0.5:8812" });
    assert.equal(homeTracker(), "saccade");
    assert.equal(trackerBase("pyrimel").toString(), "http://10.0.0.5:8812/");
    assert.equal(serverBase().toString(), "http://hub:8811/", "the window store lives on the home");
    registerTracker("saccade", "hub.tail:8811");
    assert.equal(trackerBase("saccade").toString(), "http://hub.tail:8811/", "retyping a host replaces it");
  } finally {
    restoreRegistry();
    restore();
  }
});

test("a device set up before the registry keeps its one server", () => {
  const restore = withHost("127.0.0.1:8811");
  const restoreRegistry = withRegistry();
  try {
    assert.equal(homeTracker(), null);
    assert.equal(trackerBase("saccade").toString(), "http://127.0.0.1:8811/");
    assert.equal(serverBase().toString(), "http://127.0.0.1:8811/");
  } finally {
    restoreRegistry();
    restore();
  }
});

test("a page view type is never mistaken for a board", () => {
  const page = decodePage(TIMELINE);
  assert.equal(page.kind, "timeline");
});
