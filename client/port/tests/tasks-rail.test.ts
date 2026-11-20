// The control plane's rail (t-48), its pure half: an agents page's
// incarnations joined with Herdr core's words. Running is Herdr's say
// alone; the words shown are the program's own.

import assert from "node:assert/strict";
import { test } from "vitest";
import type { ViewPage } from "../src/lib/pages.ts";
import { NO_WORDS, agentFactsOf, railOf, wordFor, wordRunsOf, wordsByIncarnationOf, wordsOf, type WordsSnapshot } from "../src/lib/tasks/rail.ts";

function agentsPage(): ViewPage {
  const agent = (n: number, address: string | null, tasks: string | null, name: string | null = null) => ({
    kind: "group" as const,
    location: `i-${n}`,
    content: [
      { kind: "fact" as const, label: "incarnation", value: `i-${n}` },
      ...(name === null ? [] : [{ kind: "fact" as const, label: "name", value: name }]),
      ...(address === null ? [] : [{ kind: "fact" as const, label: "terminal address", value: address }]),
      { kind: "fact" as const, label: "harness", value: "claude-code" },
      ...(tasks === null ? [] : [{ kind: "fact" as const, label: "tasks", value: tasks }]),
    ],
  });
  return {
    view: { major: 1 },
    kind: "document",
    location: "agents",
    seq: null,
    content: [
      agent(1, "t10", "t-4 t-9"),
      agent(2, "t11", null),
      agent(3, "t12", "t-2", "theory"),
      agent(4, null, "t-1"),
      agent(5, "t13", "t-3"),
    ],
  };
}

const words: WordsSnapshot = {
  generation: 7,
  connected: true,
  words: {
    t10: { state: "working", msg: "running tests", at: 100 },
    t11: { state: "idle", at: 90 },
    t12: { state: "working", at: 80 },
    // ended: the reason, no word - not running
    t13: { finished: "exited", at: 70 },
    // a terminal no project's agent holds and no program named: never shown
    t99: { state: "working", at: 60 },
  },
};

test("an agents page restates its incarnations with their terminals and tasks", () => {
  const facts = agentFactsOf(agentsPage());
  assert.deepEqual(facts[0], { incarnation: "i-1", name: null, address: "t10", tasks: ["t-4", "t-9"] });
  assert.deepEqual(facts[1], { incarnation: "i-2", name: null, address: "t11", tasks: [] });
  assert.equal(facts[2]?.name, "theory", "a recorded name rides its own fact");
  assert.equal(facts[3]?.address, null);
});

test("an agent's tasks are the ones still open: the page's open tasks, else every owned task from an older tracker", () => {
  const page = (facts: { label: string; value: string }[]): ViewPage => ({
    view: { major: 1 },
    kind: "document",
    location: "agents",
    seq: null,
    content: [
      {
        kind: "group",
        location: "i-1",
        content: [{ kind: "fact", label: "incarnation", value: "i-1" }, ...facts.map((fact) => ({ kind: "fact" as const, ...fact }))],
      },
    ],
  });
  // ownership outlives a conclusion: only the open ones are shown
  assert.deepEqual(agentFactsOf(page([{ label: "tasks", value: "t-9 t-4 t-2" }, { label: "open tasks", value: "t-9" }]))[0]?.tasks, ["t-9"]);
  // every owned task has ended: an empty open list, never the owned one
  assert.deepEqual(agentFactsOf(page([{ label: "tasks", value: "t-4 t-2" }, { label: "open tasks", value: "" }]))[0]?.tasks, []);
  // a tracker before sac 0.20.8 sends no open tasks: the owned list stands
  assert.deepEqual(agentFactsOf(page([{ label: "tasks", value: "t-4 t-2" }]))[0]?.tasks, ["t-4", "t-2"]);
});

test("the rail shows running agents only, by Herdr's word in t-14's order, within a word by task", () => {
  const rail = railOf(
    [
      { name: "saccade", agents: agentFactsOf(agentsPage()), problem: null },
      { name: "jernerics", agents: [], problem: null },
    ],
    words,
    "saccade",
  );
  assert.equal(rail.herdr, null);
  assert.deepEqual(
    rail.projects.map((project) => [project.name, project.current]),
    [["jernerics", false], ["saccade", true]],
    "projects in name order, the shown one marked",
  );
  const saccade = rail.projects[1]!;
  assert.deepEqual(
    saccade.groups.map((group) => [group.state, group.agents.map((agent) => agent.incarnation)]),
    [["working", ["i-3", "i-1"]], ["idle", ["i-2"]]],
    "no terminal, an ended terminal, or a non-agent terminal never shows",
  );
  assert.equal(saccade.groups[0]?.agents[1]?.msg, "running tests", "the program's own words");
  assert.deepEqual(
    saccade.groups[0]?.agents.map((agent) => agent.name),
    ["theory", null],
    "a named agent carries its name; an unnamed one shows as its incarnation",
  );
  assert.equal(saccade.groups[0]?.agents[0]?.msg, null, "nothing shown when nothing was sent");
  assert.deepEqual(rail.projects[0]?.groups, []);
});

test("an unreadable project stays listed with its problem; no Herdr core says why", () => {
  const rail = railOf(
    [{ name: "pyrimel", agents: null, problem: "connection refused" }],
    { ...NO_WORDS, problem: "no socket at /run/user/1000/herdr" },
    null,
  );
  assert.equal(rail.projects[0]?.problem, "connection refused");
  assert.equal(rail.herdr, "no socket at /run/user/1000/herdr");
});

test("the shell's snapshot decodes field by field; anything malformed is no connection", () => {
  assert.deepEqual(wordsOf(null), NO_WORDS);
  const decoded = wordsOf({
    generation: 3,
    connected: true,
    words: { t1: { state: "blocked", msg: "permission", at: 5, extra: 1 }, bad: "x" },
  });
  assert.equal(decoded.generation, 3);
  assert.deepEqual(decoded.words.t1, { state: "blocked", msg: "permission", at: 5 }, "an absent field stays absent");
  const placed = wordsOf({
    generation: 4,
    connected: true,
    words: {
      // a where field (the retired desktop app's label) is not read
      t63: { state: "working", app: "claude-code", cwd: "/home/jerry/Projects/saccade", where: "saccade / general", terminal_title: "Envision", at: 6 },
    },
  });
  assert.deepEqual(placed.words.t63, {
    state: "working",
    app: "claude-code",
    cwd: "/home/jerry/Projects/saccade",
    terminalTitle: "Envision",
    at: 6,
  });
  assert.equal(decoded.words.bad, undefined);
});

// -- the rail as targets (t-48): selection, the act menu, forgetting ----

import { vi, onTestFinished } from "vitest";
import { agentPart, projectPart, railTargetsOf } from "../src/lib/tasks/rail.ts";
import { taskRowActs, tasksMenuLabel } from "../src/lib/tasks/menu.ts";
import { theProjects, directAsks } from "../src/lib/asks.ts";
import { parseRegisteredPath } from "../src/lib/projects.ts";
import { sourceFor } from "../src/lib/sources.ts";
import { TrackerSource } from "../src/lib/tracker.ts";
import { createTasksRail } from "../src/instruments/tasksRail.ts";
import { createFrameRail } from "../src/lib/frame/rail.ts";
import "../src/lib/instruments/index.ts";

function railWithAgent() {
  const words: WordsSnapshot = { generation: 1, connected: true, words: { t12: { state: "working", at: 1 } } };
  return railOf(
    [
      { name: "saccade", agents: agentFactsOf(agentsPage()), problem: null },
      { name: "pyrimel", agents: [], problem: null },
    ],
    words,
    "saccade",
  );
}

test("the rail's targets: each project, then its agents, carrying the shown mark", () => {
  const targets = railTargetsOf(railWithAgent());
  assert.deepEqual(
    targets.map((target) => [target.part, target.kind, target.text, target.task, target.current, target.within]),
    [
      [projectPart("pyrimel"), "project", "pyrimel", null, false, undefined],
      [projectPart("saccade"), "project", "saccade", null, true, undefined],
      // an agent sits inside its project (t-63: its label is nested)
      [agentPart("saccade", "i-3"), "agent", "theory (i-3) t-2", "t-2", true, projectPart("saccade")],
    ],
  );
});

test("a project's acts: f shows its tasks here first; a opens the add field; d unregisters any project, the shown one too", async () => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => 'name = "tasks-test"\n\n[tracker]\nport = 9\n',
      readAsset: async () => {
        throw new Error("no assets here");
      },
      requestLocal: async () => {
        throw new Error("no requests here");
      },
    }),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  const forgotten: string[] = [];
  const opened: string[] = [];
  const rail = createFrameRail({
    capture: () => ({ release: () => {} }),
    open: (spec) => opened.push(spec),
    shownProject: () => null,
  });
  rail.add.enter();
  rail.add.type("box:/srv/tasks-test");
  await rail.add.submit();
  const registered = rail.targets().find((entry) => entry.kind === "project");
  assert.ok(registered !== undefined, "the registered project publishes");
  const acts = registered.acts();
  assert.deepEqual(acts.map((act) => act.key), ["f", "a", "d"]);
  acts[0]!.action.run();
  assert.deepEqual(opened, ["tasks-test/tasks"], "f opens the project's task list");
  acts[1]!.action.run();
  assert.equal(rail.add.view().open, true, "a opens the add field and opens nothing");
  rail.add.close();
  acts[2]!.action.run();
  assert.deepEqual(rail.projectsNow().map((project) => project.name), [], "the unregistered project leaves the rail");
  // the store holds no path any more
  assert.deepEqual(JSON.parse(store.get("saccade.projects") ?? "[]"), []);
});

test("an agent's acts are its first task's, composed against its own project", async () => {
  // the agent's acts are the frame rail's now: the service resolves
  // them over its live view, the registry stubbed at the window seam
  // and the word feed answering the shell's own command
  const store = new Map<string, string>([["saccade.projects", JSON.stringify(["box:/srv/saccade"])]]);
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => 'name = "saccade"\n\n[tracker]\nport = 8814\n',
      readAsset: async () => {
        throw new Error("no assets here");
      },
      requestLocal: async () => {
        throw new Error("no requests here");
      },
    }),
    __TAURI_INTERNALS__: {
      invoke: (command: string) =>
        command === "words_wait"
          ? // the shell's bounded hold: the snapshot answers after its
            // hold, the same generation - a quiet feed, not a spin
            new Promise((resolve) =>
              setTimeout(
                () => resolve({
                  generation: 1,
                  connected: true,
                  words: { t10: { state: "working", app: "claude-code", at: 1 } },
                }),
                50,
              ),
            )
          : Promise.resolve(null),
    },
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  const source = sourceFor("tracker:saccade") as TrackerSource;
  vi.spyOn(source, "agentsPage").mockImplementation(async () => agentsPage());
  vi.spyOn(source, "identity").mockImplementation(async () => ({ machineId: null, repoPath: null }));
  const elsewhere: [string, string][] = [];
  const copied: string[] = [];
  const rail = createTasksRail({
    ports: () => ({
      open: () => {},
      elsewhere: (spec, act) => elsewhere.push([spec, act]),
      copy: (text) => copied.push(text),
    }),
  });
  rail.start();
  try {
    await vi.waitFor(() => {
      assert.ok(rail.words().words.t10 !== undefined, "the words read");
    });
    // the agents publish beneath the project's rail entry (the
    // instrument's own publication face; the ports come from the
    // rail's own deps)
    const targets = rail.railTargets("saccade");
    const agent = targets.find((entry) => entry.kind === "agent");
    assert.ok(agent !== undefined, "the rail's agents publish as targets");
    const acts = agent.acts();
    assert.deepEqual(acts.map((act) => act.key), ["y", "o", "i", "w"], "y the attach first, then the task's acts");
    acts[0]!.action.run();
    assert.deepEqual(copied, ["hc attach t10"]);
    acts[1]!.action.run();
    assert.deepEqual(elsewhere, [["saccade/t-4", "link"]]);
  } finally {
    rail.stop();
  }
});


test("a declared project with no tracker section is a plain rail entry: no beneath-targets, no problem line", async () => {
  const store = new Map<string, string>([["saccade.projects", JSON.stringify(["box:/srv/bare"])]]);
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => 'name = "bare"\n',
      readAsset: async () => {
        throw new Error("no assets here");
      },
      requestLocal: async () => {
        throw new Error("no requests here");
      },
    }),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  await theProjects().projects();
  const rail = createFrameRail({
    capture: () => ({ release: () => {} }),
    open: () => {},
    shownProject: () => null,
  });
  rail.start();
  try {
    await vi.waitFor(() => {
      assert.deepEqual(rail.projectsNow().map((project) => project.name), ["bare"], "the registry's name lists");
    });
    // the plain entry: a header with no problem line, and the seam
    // publishes nothing beneath it (the tasks instrument's tracker
    // refuses a repo it does not serve)
    const targets = rail.targets();
    const header = targets.find((target) => target.id === projectPart("bare"));
    assert.ok(header !== undefined, "the plain header draws");
    assert.equal(header.label, "bare");
    assert.deepEqual(
      targets.filter((target) => target.inside === projectPart("bare")),
      [],
      "nothing publishes beneath a project no instrument serves",
    );
    // no problem line anywhere under the entry
    assert.ok(!targets.some((target) => target.label.toLowerCase().includes("unreachable")));
  } finally {
    rail.stop();
  }
});

test("the backoff doubles per consecutive failure, caps at a minute, and resets on success (injected clock)", async () => {
  let moment = 1_000_000;
  const store = new Map<string, string>([["saccade.projects", JSON.stringify(["box:/srv/saccade"])]]);
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => 'name = "saccade"\n\n[tracker]\nport = 8814\n',
      readAsset: async () => {
        throw new Error("no assets here");
      },
      requestLocal: async () => {
        throw new Error("no requests here");
      },
    }),
  });
  const source = sourceFor("tracker:saccade") as TrackerSource;
  let answers: "fail" | "succeed" = "fail";
  const asked: number[] = [];
  vi.spyOn(source, "agentsPage").mockImplementation(async () => {
    asked.push(moment);
    if (answers === "fail") throw new Error("the tracker refused");
    return agentsPage();
  });
  vi.spyOn(source, "identity").mockImplementation(async () => ({ machineId: null, repoPath: null }));
  const rail = createTasksRail({
    ports: () => ({ open: () => {}, elsewhere: undefined, copy: undefined }),
    now: () => moment,
  });
  rail.start();
  const tick = async (): Promise<void> => {
    await rail.tick();
  };
  try {
    // first ask fails: the next ask is one interval doubled (6s away)
    await tick();
    const first = asked.length;
    assert.equal(first, 1, "the first ask ran");
    moment += 3000;
    await tick();
    assert.equal(asked.length, first, "the doubled interval holds the ask back at +3s");
    // past the doubled interval: the second ask runs and fails: the
    // next ask doubles again (12s away)
    moment += 4000;
    await tick();
    assert.equal(asked.length, first + 1, "the doubled interval released the ask");
    moment += 6000;
    await tick();
    assert.equal(asked.length, first + 1, "the second doubling holds the ask back at +6s");
    moment += 7000;
    await tick();
    assert.equal(asked.length, first + 2, "the second doubling released the ask");
    // the cap: many more failures never wait longer than a minute
    for (let round = 0; round < 6; round++) {
      moment += 61_000;
      await tick();
      assert.equal(asked.length, first + 2 + round + 1, "the capped ask released");
    }
    const cappedGap = asked[asked.length - 1]! - asked[asked.length - 2]!;
    assert.ok(cappedGap <= 61_000, "the cap keeps the wait under a minute");

    // a success resets: the next ask rides the live cadence again
    answers = "succeed";
    moment += 61_000;
    await tick();
    const afterSuccess = asked.length;
    moment += 3000;
    await tick();
    assert.equal(asked.length, afterSuccess + 1, "the reset returns the project to the live cadence");
  } finally {
    rail.stop();
  }
});

test("registering reads the declaration through the module; the store keeps paths alone; forgetting drops the path", async () => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => 'name = "pyrimel"\n\n[tracker]\nport = 8813\n',
      readAsset: async () => {
        throw new Error("no assets here");
      },
      requestLocal: async () => {
        throw new Error("no requests here");
      },
    }),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  const projects = theProjects();
  const registered = parseRegisteredPath("box:/srv/pyrimel");
  assert.ok(registered !== null);
  await projects.register(registered);
  assert.deepEqual(projects.registeredPaths(), [{ machine: "box", path: "/srv/pyrimel" }]);
  assert.deepEqual(JSON.parse(store.get("saccade.projects") ?? "[]"), ["box:/srv/pyrimel"], "the device store keeps the registered paths alone");
  const named = await projects.projectByName("pyrimel");
  assert.equal(named?.declaration.port, 8813);
  // registering again is a no-op, not a duplicate
  await projects.register(registered);
  assert.deepEqual(projects.registeredPaths(), [{ machine: "box", path: "/srv/pyrimel" }]);
  projects.forget(registered);
  assert.deepEqual(projects.registeredPaths(), []);
  assert.equal(store.get("saccade.projects"), "[]");
});

test("a declaration with no tracker section registers and lists; one that names nothing does not", async () => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => `name = "bare"\n`,
      readAsset: async () => {
        throw new Error("no assets here");
      },
      requestLocal: async () => {
        throw new Error("no requests here");
      },
    }),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  const projects = theProjects();
  const registered = parseRegisteredPath("box:/srv/bare");
  assert.ok(registered !== null);
  const declaration = await projects.register(registered);
  assert.deepEqual(declaration, { name: "bare", port: null }, "a declaration without a tracker section names the project alone");
  const listed = await projects.projects();
  assert.deepEqual(listed.map((project) => project.declaration.name), ["bare"]);
  assert.deepEqual(await projects.projectByName("bare"), listed[0]);
  assert.equal(await projects.projectByName("nope"), null);
});

test("a federating core's machine-first addresses join the tracker's bare ones; two machines with one address join neither", () => {
  const word = (state: string) => ({ state, at: 1 });
  assert.equal(wordFor({ "other/t81": word("done") }, "t81")?.state, "done");
  assert.equal(wordFor({ t81: word("working"), "other/t81": word("done") }, "t81")?.state, "working", "an exact key wins");
  assert.equal(wordFor({ "a/t81": word("done"), "b/t81": word("idle") }, "t81"), undefined);
  assert.equal(wordFor({ "other/xt81": word("done") }, "t81"), undefined, "only a whole address after the machine");
  const rail = railOf(
    [{ name: "pyrimel", agents: [{ incarnation: "i-9", name: null, address: "t88", tasks: ["t-4"] }], problem: null }],
    { generation: 1, connected: true, words: { "other/t88": word("working") } },
    "pyrimel",
  );
  assert.deepEqual(rail.projects[0]?.groups.map((group) => [group.state, group.agents.map((agent) => agent.incarnation)]), [["working", ["i-9"]]]);
});

// -- agents no tracker registered (t-55) ----------------------------------

import { railAgentText, stateSlugOf, terminalPart } from "../src/lib/tasks/rail.ts";

test("the worktree home's slug is the tracker's own: basename and the FNV-1a of the repo path", () => {
  // read off the live machine: pyrimel's worktrees live under
  // ~/.local/state/saccade/pyrimel-11f6a386/worktrees/
  assert.equal(stateSlugOf("/home/jerry/Projects/pyrimel"), "pyrimel-11f6a386");
  assert.equal(stateSlugOf("/home/jerry/Projects/pyrimel/"), "pyrimel-11f6a386", "a trailing slash is the same repo");
  // tracker/src/host/paths.rs pins the hue as the same hash mod 360
  assert.equal(Number.parseInt(stateSlugOf("/srv/hornet").split("-")[1]!, 16) % 360, 324);
});

function strayWords(): WordsSnapshot {
  return {
    generation: 9,
    connected: true,
    words: {
      // registered: i-3 of saccade (bare t12), never doubled as a stray
      t12: { state: "working", app: "claude-code", cwd: "/home/jerry/Projects/saccade", at: 1 },
      // in the saccade repo, unregistered
      t63: { state: "working", app: "claude-code", msg: "editing", cwd: "/home/jerry/Projects/saccade", terminalTitle: "saccade / general", at: 2 },
      // in a subdirectory of the repo
      t6r: { state: "idle", app: "pi", cwd: "/home/jerry/Projects/saccade/nix", terminalTitle: "pi", at: 3 },
      // in a pyrimel worktree, through a federating core
      "other/t4j": { state: "done", app: "claude-code", cwd: "/home/jerry/.local/state/saccade/pyrimel-11f6a386/worktrees/t-0", terminalTitle: "t-0 / 2", at: 4 },
      // nowhere a project knows: the foot group, by word then name
      t7x: { state: "done", app: "claude-code", cwd: "/srv/elsewhere", terminalTitle: "elsewhere / 1", at: 5 },
      t7r: { state: "working", app: "claude-code", cwd: "/srv/other", terminalTitle: "", at: 6 },
      t7q: { state: "error", app: "pi", terminalTitle: "pi run", at: 7 },
      // a shell: no app, never shown
      t3k: { cwd: "/home/jerry/Projects/saccade", terminalTitle: "saccade / service", at: 8 },
      // ended: not running
      t9: { app: "claude-code", finished: "exited", cwd: "/home/jerry/Projects/saccade", at: 9 },
      // a nested repo inside saccade: the deeper project holds it
      t80: { state: "idle", app: "claude-code", cwd: "/home/jerry/Projects/saccade/vendor/inner/src", at: 10 },
    },
  };
}

function strayRail() {
  return railOf(
    [
      { name: "saccade", agents: agentFactsOf(agentsPage()), problem: null, repoPath: "/home/jerry/Projects/saccade" },
      { name: "pyrimel", agents: [], problem: null, repoPath: "/home/jerry/Projects/pyrimel" },
      { name: "inner", agents: [], problem: null, repoPath: "/home/jerry/Projects/saccade/vendor/inner" },
      // a tracker whose repo path is unknown places nothing
      { name: "jernerics", agents: [], problem: null, repoPath: null },
    ],
    strayWords(),
    "saccade",
  );
}

test("a stray agent shows under the project whose repo or worktree holds its cwd, marked unregistered, after the registered", () => {
  const rail = strayRail();
  const project = (name: string) => rail.projects.find((entry) => entry.name === name)!;
  assert.deepEqual(
    project("saccade").groups.map((group) => [group.state, group.agents.map((agent) => [agent.incarnation, agent.address, agent.label])]),
    [
      ["working", [["i-3", "t12", null], [null, "t63", "saccade / general"]]],
      ["idle", [[null, "t6r", "pi"]]],
    ],
    "the terminal's title names it; the registered t12 is not doubled",
  );
  assert.equal(project("saccade").groups[0]?.agents[1]?.msg, "editing", "the program's own words");
  assert.deepEqual(
    project("pyrimel").groups.map((group) => [group.state, group.agents.map((agent) => agent.address)]),
    [["done", ["other/t4j"]]],
    "a worktree under the tracker's state slug is the project's; the address as the local core names it",
  );
  assert.deepEqual(project("inner").groups.map((group) => group.agents.map((agent) => agent.address)), [["t80"]], "the deepest repo wins");
  assert.deepEqual(project("jernerics").groups, []);
});

test("a stray agent no project holds shows at the foot, by word then name; shells and ended terminals never show", () => {
  const rail = strayRail();
  assert.deepEqual(
    rail.unregistered.map((agent) => [agent.state, agent.label, agent.address]),
    [
      ["error", "pi run", "t7q"],
      ["done", "elsewhere / 1", "t7x"],
      ["working", null, "t7r"],
    ],
  );
  const shown = [...rail.projects.flatMap((p) => p.groups.flatMap((g) => g.agents)), ...rail.unregistered].map((a) => a.address);
  assert.ok(!shown.includes("t3k") && !shown.includes("t9") && !shown.includes("t99"));
});

test("an address two machines hold is no stray: one of them is registered, so neither shows unregistered", () => {
  const rail = railOf(
    [{ name: "pyrimel", agents: [{ incarnation: "i-9", address: "t88", tasks: [], name: null }], problem: null, repoPath: "/p" }],
    {
      generation: 1,
      connected: true,
      words: {
        "a/t88": { state: "working", app: "pi", cwd: "/p", at: 1 },
        "b/t88": { state: "idle", app: "pi", cwd: "/p", at: 1 },
      },
    },
    null,
  );
  assert.deepEqual(rail.projects[0]?.groups, []);
  assert.deepEqual(rail.unregistered, []);
  // an exact registered address leaves another machine's terminal a stray
  const exact = railOf(
    [{ name: "pyrimel", agents: [{ incarnation: "i-9", address: "t5", tasks: [], name: null }], problem: null, repoPath: "/p" }],
    { generation: 1, connected: true, words: { t5: { state: "idle", app: "pi", at: 1 }, "other/t5": { state: "done", app: "pi", at: 1 } } },
    null,
  );
  assert.deepEqual(exact.unregistered.map((agent) => agent.address), ["other/t5"]);
});

test("every agent is a target with y copying hc attach <address> as the local core names it; a stray's only act is y", () => {
  const rail = strayRail();
  const targets = railTargetsOf(rail);
  const agents = targets.filter((target) => target.kind === "agent");
  assert.deepEqual(
    agents.map((target) => [target.part, target.text, target.project, target.task, target.address]),
    [
      [terminalPart("t80"), "t80", "inner", null, "t80"],
      [terminalPart("other/t4j"), "t-0 / 2 other/t4j", "pyrimel", null, "other/t4j"],
      [agentPart("saccade", "i-3"), "theory (i-3) t-2", "saccade", "t-2", "t12"],
      [terminalPart("t63"), "saccade / general t63", "saccade", null, "t63"],
      [terminalPart("t6r"), "pi t6r", "saccade", null, "t6r"],
      [terminalPart("t7q"), "pi run t7q", null, null, "t7q"],
      [terminalPart("t7x"), "elsewhere / 1 t7x", null, null, "t7x"],
      [terminalPart("t7r"), "t7r", null, null, "t7r"],
    ],
    "projects in name order, each followed by its agents; the foot group last",
  );
});
test("a named agent's menu heading reads by its name, the incarnation beside it (t-56)", () => {
  const base = { address: "t12", state: "working", msg: null, progress: null, tasks: ["t-2"], at: 1, label: null };
  assert.equal(railAgentText({ ...base, incarnation: "i-3", name: "theory" }), "theory (i-3) t-2");
  assert.equal(railAgentText({ ...base, incarnation: "i-3", name: null }), "i-3 t-2");
});

test("words in importance order: error, blocked, done, working, idle, any other word alphabetically, then no word", () => {
  const states = ["idle", "", "working", "zeal", "done", "blocked", "compacting", "error"];
  const agents = states.map((state, index) => ({ incarnation: `i-${index}`, name: null, address: `t${index}`, tasks: [] }));
  const rail = railOf(
    [{ name: "pyrimel", agents, problem: null }],
    {
      generation: 1,
      connected: true,
      words: Object.fromEntries(states.map((state, index) => [`t${index}`, state === "" ? { at: 1 } : { state, at: 1 }])),
    },
    "pyrimel",
  );
  assert.deepEqual(
    rail.projects[0]?.groups.map((group) => group.state),
    ["error", "blocked", "done", "working", "idle", "compacting", "zeal", ""],
  );
  // the pipeline's join: each registered agent's live word, none for no word
  const joined = wordsByIncarnationOf(rail);
  assert.equal(joined.get("i-7"), "error");
  assert.equal(joined.get("i-4"), "done");
  assert.equal(joined.has("i-1"), false, "an agent with no word carries no word");
  // only the shown project's agents join
  const unshown = railOf([{ name: "pyrimel", agents, problem: null }], { generation: 1, connected: true, words: {} }, null);
  assert.equal(wordsByIncarnationOf(unshown).size, 0);
});

test("the foot group reads as runs of one word, in its own order", () => {
  const runs = wordRunsOf(strayRail().unregistered);
  assert.deepEqual(
    runs.map((run) => [run.state, run.agents.map((agent) => agent.address)]),
    [
      ["error", ["t7q"]],
      ["done", ["t7x"]],
      ["working", ["t7r"]],
    ],
  );
});
