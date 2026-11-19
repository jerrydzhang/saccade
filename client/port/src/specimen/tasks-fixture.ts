// The task list's specimen fixture: a rail built by the real railOf from
// fixture agents pages and Herdr words, a pipeline page model, and a
// controller that serves them and runs nothing. It covers every word
// (and one the core sends that the language has no shape for), an
// agent with no word, a named agent with several tasks, an
// unregistered agent under a project and at the foot, and every stage,
// one of them empty.

import { railOf, type AgentFacts, type HerdrWord, type Rail, type WordsSnapshot } from "../lib/tasks/rail.ts";
import { ADD_CLOSED, type AddView } from "../lib/tasks/add.ts";
import type { TaskRow, TasksPageModel } from "../lib/tasks/page.ts";
import type { TasksData, TasksUI } from "../instruments/tasksController.ts";

const REPO = "/home/jerry/Projects/pyrimel";

const agents: AgentFacts[] = [
  { incarnation: "i-3", name: "theory", address: "t3", tasks: ["t-0"] },
  { incarnation: "i-9", name: "empirical", address: "t9", tasks: ["t-5", "t-7"] },
  { incarnation: "i-6", name: null, address: "t6", tasks: ["t-2"] },
  { incarnation: "i-10", name: null, address: "t10", tasks: ["t-6"] },
  { incarnation: "i-4", name: null, address: "t4", tasks: [] },
  { incarnation: "i-11", name: "atlas", address: "t11", tasks: ["t-9"] },
  { incarnation: "i-12", name: null, address: "t12", tasks: ["t-10"] },
  { incarnation: "i-13", name: null, address: "t13", tasks: ["t-11"] },
  // an orchestrator answerable for many tasks at once: the list clips
  { incarnation: "i-14", name: "orchestrator of the overnight batches", address: "t14", tasks: Array.from({ length: 18 }, (_, n) => `t-${33 - n}`) },
];

const word = (fields: Omit<HerdrWord, "at">): HerdrWord => ({ at: 1, app: "claude-code", ...fields });

const words: WordsSnapshot = {
  generation: 1,
  connected: true,
  words: {
    t3: word({ state: "working", msg: "Reading tube_score.py; comparing against the hand-estimated walk" }),
    t9: word({ state: "working", msg: "Running the overnight PySR search (14/40 generations)" }),
    t6: word({ state: "blocked", msg: "Waiting for permission to run uv" }),
    t10: word({ state: "done", msg: "Offered t-6 for review" }),
    t4: word({ state: "idle" }),
    t14: word({ state: "working", msg: "Dispatching the next batch of checks to the three empirical agents, then waiting on all of their offers before the morning summary" }),
    t11: word({ state: "error", msg: "The search process exited with status 137" }),
    t12: word({ msg: "Started; no word yet" }),
    t13: word({ state: "compacting", msg: "Compacting the conversation" }),
    // unregistered: one in the project's repo, two nowhere known
    t70: word({ state: "working", cwd: `${REPO}/notes`, terminalTitle: "pyrimel / scratch", msg: "Drafting the atlas notes" }),
    t7q: word({ state: "error", app: "pi", terminalTitle: "pi run", msg: "Lost the model connection" }),
    t7r: word({ state: "idle", cwd: "/srv/elsewhere", terminalTitle: "elsewhere / 1" }),
  },
};

export const tasksRail: Rail = railOf(
  [
    { name: "jernerics", agents: [], problem: null, repoPath: "/home/jerry/Projects/jernerics" },
    { name: "pyrimel", agents, problem: null, repoPath: REPO },
    { name: "saccade", agents: [], problem: null, repoPath: "/home/jerry/Projects/saccade" },
  ],
  words,
  "pyrimel",
);

const now = Math.floor(Date.now() / 1000);
const HOUR = 3600;
let index = 0;
const row = (task: string, title: string, agents: string, commit: string, age: number): TaskRow => ({
  part: `/content/fixture/row/${index++}`,
  task,
  location: task,
  title,
  agents,
  commit,
  revision: commit === "" ? "" : "r1",
  at: now - age,
});
const group = (n: number, title: string, rows: TaskRow[]) => ({ part: `/content/${n}`, title, rows });

export const tasksPage: TasksPageModel = {
  groups: [
    group(0, "waiting", []),
    group(1, "in flight", [
      row("t-0", "derive a working theory from scratch", "i-3", "d192364a8e0f6c1b2d3e4f5a6b7c8d9e0f1a2b3c", 4 * 24 * HOUR),
      row("t-2", "numerically check the hand-estimated atlas walk", "i-6", "77be1f2c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a", 5 * HOUR),
      row("t-5", "empirical checks for the tube score, run alongside the t-0 theory", "i-9", "afbbde7aa1b2c3d4e5f60718293a4b5c6d7e8f90", 3 * HOUR),
      row("t-7", "tabulate the three disagreeing regions", "i-9", "3a91c0e00112233445566778899aabbccddeeff0", HOUR),
      row("t-9", "bound the search's memory before the next overnight run", "i-11", "", 2 * HOUR),
      row("t-10", "port the atlas walk to the new grid", "i-12", "", 20 * 60),
      row("t-11", "summarise the week's runs", "i-13", "", 50 * 60),
    ]),
    group(2, "awaiting judgment", [
      row("t-6", "bound the buffer maximum by a certified optimizer", "i-10", "c08e4a1ffeeddccbbaa99887766554433221100f", 40 * 60),
    ]),
    group(3, "concluded last 24h", [
      row("t-3", "rescore earlier search candidates with the tube score against the KL loss", "i-7", "8427e3a0123456789abcdef0123456789abcdef0", 21 * HOUR),
      row("t-4", "wire the search into the overnight runner", "i-8", "527afd0fedcba9876543210fedcba9876543210f", 21 * HOUR),
    ]),
  ],
};

/** The shown project's task list, loaded. */
export const tasksContent: TasksData = {
  generation: 1,
  asked: "pyrimel/tasks",
  location: "pyrimel/tasks",
  project: "pyrimel",
  resolved: { machine: "tracker:pyrimel", path: "tasks" },
  status: "loaded",
  problem: null,
  page: tasksPage,
  errors: [],
};

/** A controller that serves the fixture and runs nothing. The frame's
 * faces (select, menu) are the buffer's; the specimen hosts the view
 * bare, with no lease, so none appear here. The rail is the frame's,
 * not this instrument's. */
export function fixtureTasksUI(add: AddView = ADD_CLOSED): TasksUI {
  void add;
  return {
    getSelection: () => null,
    subscribeWords: (observer) => {
      observer(tasksRail);
      return () => {};
    },
  };
}
