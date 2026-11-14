// The control plane's rail (t-48), its pure half: an agents page's
// incarnations joined with Herdr core's words. Running is Herdr's say
// alone; the words shown are the program's own.

import assert from "node:assert/strict";
import { test } from "vitest";
import type { ViewPage } from "../src/lib/pages.ts";
import { NO_WORDS, agentFactsOf, railOf, wordsOf, type WordsSnapshot } from "../src/lib/tasks/rail.ts";

function agentsPage(): ViewPage {
  const agent = (n: number, address: string | null, tasks: string | null) => ({
    kind: "group" as const,
    location: `i-${n}`,
    content: [
      { kind: "fact" as const, label: "incarnation", value: `i-${n}` },
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
      agent(3, "t12", "t-2"),
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
    // a terminal no project's agent holds: never shown
    t99: { state: "working", at: 60 },
  },
};

test("an agents page restates its incarnations with their terminals and tasks", () => {
  const facts = agentFactsOf(agentsPage());
  assert.deepEqual(facts[0], { incarnation: "i-1", address: "t10", tasks: ["t-4", "t-9"] });
  assert.deepEqual(facts[1], { incarnation: "i-2", address: "t11", tasks: [] });
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
  assert.deepEqual(decoded.words.t1, {
    state: "blocked",
    msg: "permission",
    progress: undefined,
    title: undefined,
    finished: undefined,
    at: 5,
  });
  assert.equal(decoded.words.bad, undefined);
});
