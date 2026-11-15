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
      t63: { state: "working", app: "claude-code", cwd: "/home/jerry/Projects/saccade", where: "saccade / general", terminal_title: "Envision", at: 6 },
    },
  });
  assert.deepEqual(placed.words.t63, {
    state: "working",
    app: "claude-code",
    cwd: "/home/jerry/Projects/saccade",
    where: "saccade / general",
    terminalTitle: "Envision",
    at: 6,
  });
  assert.equal(decoded.words.bad, undefined);
});

// -- the rail as targets (t-48): selection, the act menu, forgetting ----

import { vi, onTestFinished } from "vitest";
import { agentPart, projectPart, railTargetsOf } from "../src/lib/tasks/rail.ts";
import { taskRowActs, tasksMenuLabel } from "../src/lib/tasks/menu.ts";
import { forgetTracker, homeTracker, knownTrackers, registerTracker } from "../src/lib/tracker.ts";

function railWithAgent() {
  const words: WordsSnapshot = { generation: 1, connected: true, words: { t12: { state: "working", at: 1 } } };
  return railOf(
    [
      { name: "saccade", agents: agentFactsOf(agentsPage()), problem: null },
      { name: "pyrimel", agents: [], problem: null },
    ],
    words,
    "saccade",
    "pyrimel",
  );
}

test("the rail's targets: each project, then its agents, carrying home and shown", () => {
  const targets = railTargetsOf(railWithAgent());
  assert.deepEqual(
    targets.map((target) => [target.part, target.kind, target.text, target.task, target.home, target.current]),
    [
      [projectPart("pyrimel"), "project", "pyrimel", null, true, false],
      [projectPart("saccade"), "project", "saccade", null, false, true],
      [agentPart("saccade", "i-3"), "agent", "theory (i-3) t-2", "t-2", false, true],
    ],
  );
});

test("a project's acts: f shows it on this plane first; a adds a project; d forgets any project, the home and the shown one too", () => {
  const forgotten: string[] = [];
  const opened: [string, string | null | undefined][] = [];
  let adds = 0;
  const elsewhere: string[] = [];
  const target = (name: string, home: boolean, current: boolean) => ({
    part: projectPart(name),
    kind: "project" as const,
    text: name,
    project: name,
    task: null,
    home,
    current,
    host: name === "jernerics" ? "localhost:8814" : null,
    address: null,
  });
  const rail = [target("jernerics", false, false), target("pyrimel", true, false), target("saccade", false, true)];
  const deps = {
    machine: () => "tracker:saccade",
    rowLink: () => null,
    railTarget: (part: string) => rail.find((entry) => entry.part === part),
    forget: (name: string) => forgotten.push(name),
    addProject: () => {
      adds += 1;
    },
    ports: () => ({
      open: (spec: string, part?: string | null) => opened.push([spec, part]),
      elsewhere: (spec: string) => elsewhere.push(spec),
    }),
  };
  const acts = (name: string) => taskRowActs({ kind: "project", label: name, part: projectPart(name) }, deps);
  assert.deepEqual(acts("jernerics").map((act) => act.key), ["f", "a", "d"]);
  assert.deepEqual(acts("pyrimel").map((act) => act.key), ["f", "a", "d"]);
  assert.deepEqual(acts("saccade").map((act) => act.key), ["f", "a", "d"]);
  acts("jernerics")[0]!.action.run();
  acts("jernerics")[1]!.action.run();
  acts("jernerics")[2]!.action.run();
  // the plane names no tracker: the project rides as its part (t-54)
  assert.deepEqual(opened, [["plane", "jernerics"]]);
  assert.equal(adds, 1, "a opens the add field and opens nothing");
  assert.deepEqual(forgotten, ["jernerics"]);
  // the heading names the host it was reached at
  assert.equal(tasksMenuLabel({ kind: "project", label: "jernerics" }, rail[0]), "jernerics (localhost:8814)");
  assert.equal(tasksMenuLabel({ kind: "project", label: "pyrimel" }, rail[1]), "pyrimel");
});

test("an agent's acts are its first task's, composed against its own project", () => {
  const targets = railTargetsOf(railWithAgent());
  const elsewhere: [string, string][] = [];
  const acts = taskRowActs(
    { kind: "agent", label: "i-3 t-2", part: agentPart("saccade", "i-3") },
    {
      machine: () => "tracker:pyrimel",
      rowLink: () => null,
      railTarget: (part) => targets.find((entry) => entry.part === part),
      forget: () => {},
      addProject: () => {},
      ports: () => ({ open: () => {}, elsewhere: (spec, act) => elsewhere.push([spec, act]) }),
    },
  );
  assert.deepEqual(acts.map((act) => act.key), ["o", "i", "w"]);
  acts[0]!.action.run();
  assert.deepEqual(elsewhere, [["saccade/t-2", "link"]]);
});

test("forgetting drops a tracker from the registry; a forgotten home hands over to the next, the last leaves none", () => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  registerTracker("saccade", "localhost:8811");
  registerTracker("pyrimel", "localhost:8813");
  assert.equal(homeTracker(), "saccade");
  assert.equal(forgetTracker("nobody"), "unknown");
  assert.equal(forgetTracker("saccade"), "forgotten");
  assert.deepEqual(Object.keys(knownTrackers()), ["pyrimel"]);
  assert.equal(homeTracker(), "pyrimel");
  assert.equal(forgetTracker("pyrimel"), "forgotten");
  assert.deepEqual(knownTrackers(), {});
  assert.equal(homeTracker(), null);
  // the next tracker reached becomes the home
  registerTracker("jernerics", "localhost:8814");
  assert.equal(homeTracker(), "jernerics");
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
      t63: { state: "working", app: "claude-code", msg: "editing", cwd: "/home/jerry/Projects/saccade", where: "saccade / general", at: 2 },
      // in a subdirectory of the repo
      t6r: { state: "idle", app: "pi", cwd: "/home/jerry/Projects/saccade/nix", terminalTitle: "pi", at: 3 },
      // in a pyrimel worktree, through a federating core
      "other/t4j": { state: "done", app: "claude-code", cwd: "/home/jerry/.local/state/saccade/pyrimel-11f6a386/worktrees/t-0", where: "t-0 / 2", at: 4 },
      // nowhere a project knows: the foot group, by word then name
      t7x: { state: "done", app: "claude-code", cwd: "/srv/elsewhere", where: "elsewhere / 1", at: 5 },
      t7r: { state: "working", app: "claude-code", cwd: "/srv/other", terminalTitle: "", at: 6 },
      t7q: { state: "error", app: "pi", where: "", terminalTitle: "pi run", at: 7 },
      // a shell: no app, never shown
      t3k: { cwd: "/home/jerry/Projects/saccade", where: "saccade / service", at: 8 },
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
    "the where label names it, else the terminal's title; the registered t12 is not doubled",
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
  const copied: string[] = [];
  const elsewhere: [string, string][] = [];
  const deps = {
    machine: () => "tracker:saccade",
    rowLink: () => null,
    railTarget: (part: string) => targets.find((entry) => entry.part === part),
    forget: () => {},
    addProject: () => {},
    ports: () => ({
      open: () => {},
      elsewhere: (spec: string, act: string) => elsewhere.push([spec, act]),
      copy: (text: string) => copied.push(text),
    }),
  };
  const acts = (part: string, label: string) => taskRowActs({ kind: "agent", label, part }, deps);
  const registered = acts(agentPart("saccade", "i-3"), "i-3 t-2");
  assert.deepEqual(registered.map((act) => act.key), ["y", "o", "i", "w"], "y the attach first (Enter copies it), then the task's acts");
  registered[0]!.action.run();
  const stray = acts(terminalPart("other/t4j"), "t-0 / 2 other/t4j");
  assert.deepEqual(stray.map((act) => [act.key, act.action.label]), [["y", "copy hc attach other/t4j"]]);
  stray[0]!.action.run();
  acts(terminalPart("t7q"), "pi run t7q")[0]!.action.run();
  assert.deepEqual(copied, ["hc attach t12", "hc attach other/t4j", "hc attach t7q"]);
  assert.deepEqual(elsewhere, []);
  // with no clipboard injected, a stray has no act at all
  assert.deepEqual(
    taskRowActs({ kind: "agent", label: "t7q", part: terminalPart("t7q") }, { ...deps, ports: () => ({ open: () => {} }) }),
    [],
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
