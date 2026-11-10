// The instruments' pure contracts: the tasks page's shape, the
// agents' live join and sectioning, the activity lane's 6h/24h axis.
// Node --test, no framework, like the rest of the suite.

import test from "node:test";
import assert from "node:assert/strict";

import { decodePage, type ViewPage } from "../src/lib/pages.ts";
import {
  agentsSections,
  type AgentRow,
  attachCommand,
  mergeAgentSections,
  rowDetail,
  rowTask,
  rowTime,
  WORD_ORDER,
} from "../src/lib/instruments/agents.ts";
import { laneLayout } from "../src/lib/instruments/lane.ts";
import { taskAge, tasksSections } from "../src/lib/instruments/tasks.ts";
import { setWords, words } from "../src/lib/words.ts";

function pipelinePage(): ViewPage {
  return decodePage({
    view: { major: 1 },
    kind: "document",
    location: "tasks",
    seq: 42,
    content: [
      {
        kind: "group",
        title: "waiting",
        content: [
          {
            kind: "table",
            columns: ["task", "title", "agents", "latest commit"],
            rows: [
              { cells: ["t-2", "a fresh task", "", ""], location: "t-2", at: 1_000 },
            ],
          },
        ],
      },
      {
        kind: "group",
        title: "in flight",
        content: [
          {
            kind: "table",
            columns: ["task", "title", "agents", "latest commit"],
            rows: [
              {
                cells: ["t-1", "the proof", "i-9", "a".repeat(40)],
                location: "t-1",
                at: 2_000,
              },
            ],
          },
        ],
      },
      {
        kind: "group",
        title: "awaiting judgment",
        content: [
          {
            kind: "table",
            columns: ["task", "title", "agents", "latest commit"],
            rows: [],
          },
        ],
      },
      {
        kind: "group",
        title: "concluded last 24h",
        content: [
          {
            kind: "table",
            columns: ["task", "title", "agents", "latest commit"],
            rows: [],
          },
        ],
      },
    ],
  });
}

function agentsPage(): ViewPage {
  return decodePage({
    view: { major: 1 },
    kind: "document",
    location: "agents",
    seq: 43,
    content: [
      {
        kind: "group",
        location: "i-9",
        content: [
          { kind: "fact", label: "incarnation", value: "i-9" },
          { kind: "fact", label: "terminal address", value: "t3" },
          { kind: "fact", label: "harness", value: "pi" },
          { kind: "fact", label: "tasks", value: "t-5 t-1" },
          { kind: "fact", label: "workspace", value: "/home/hub/worktrees/one" },
          { kind: "fact", label: "last commit", value: "c".repeat(40) },
        ],
      },
      {
        kind: "group",
        location: "i-4",
        content: [
          { kind: "fact", label: "incarnation", value: "i-4" },
          { kind: "fact", label: "terminal address", value: "t8" },
          { kind: "fact", label: "harness", value: "claude code" },
          { kind: "fact", label: "tasks", value: "t-2" },
          { kind: "fact", label: "workspace", value: "/home/hub/worktrees/two" },
        ],
      },
      {
        kind: "group",
        location: "i-2",
        content: [
          { kind: "fact", label: "incarnation", value: "i-2" },
          { kind: "fact", label: "harness", value: "pi" },
        ],
      },
    ],
  });
}

test("the pipeline page restates as sections and rows in order", () => {
  const sections = tasksSections(pipelinePage());
  assert.deepEqual(
    sections.map((s) => s.title),
    ["waiting", "in flight", "awaiting judgment", "concluded last 24h"],
  );
  const inFlight = sections[1];
  assert.ok(inFlight, "in flight is the second section");
  assert.equal(inFlight.rows.length, 1);
  const row = inFlight.rows[0];
  assert.ok(row, "the in-flight section carries its row");
  assert.deepEqual(
    { ...row, latestCommit: row.latestCommit.slice(0, 1) },
    {
      task: "t-1",
      title: "the proof",
      agents: "i-9",
      latestCommit: "a",
      // a four-column page (a tracker older than the column) reads none
      latestRevision: "",
      at: 2_000,
      location: "t-1",
    },
  );
  assert.equal(sections[2]?.rows.length, 0, "the empty sections hold");
});

test("the age renders terse against now and empty without a moment", () => {
  const now = 3_600 * 24 * 1_000;
  assert.equal(taskAge(null, now), "");
  assert.equal(taskAge(now - 30, now), "now");
  assert.equal(taskAge(now - 5 * 60, now), "5m");
  assert.equal(taskAge(now - 3 * 3_600 - 12 * 60, now), "3h 12m");
  assert.equal(taskAge(now - 2 * 86_400 - 3 * 3_600, now), "2d 3h");
  assert.equal(taskAge(now + 10, now), "", "a moment ahead of now is not negative");
});

test("the agents view joins the live words by the terminal address", () => {
  setWords(
    {
      t3: { state: "working", app: "pi", msg: "editing src", progress: "42", at: 5_000 },
      t8: { state: "blocked", app: "claude", at: 5_100 },
    },
    1,
  );
  const sections = agentsSections(agentsPage(), words());
  const working = sections.find((s) => s.word === "working");
  assert.ok(working, "the working section exists");
  assert.equal(working.rows.length, 1);
  const row = working.rows[0];
  assert.ok(row, "the working row is i-9");
  assert.equal(row.incarnation, "i-9");
  assert.equal(row.word?.msg, "editing src");
  assert.equal(row.word?.progress, "42");
  assert.equal(rowDetail(row), "editing src \u00b7 42", "the detail exactly as sent, joined");
  assert.equal(rowTime(row), 5_000);
  const blocked = sections.find((s) => s.word === "blocked");
  const blockedRow = blocked?.rows[0];
  assert.equal(blockedRow?.word?.msg ?? null, null, "nothing when nothing was sent");
  assert.equal(rowDetail(blockedRow ?? rowTaskAgent()), "");
});

// a helper for the empty-detail assertion
function rowTaskAgent(): AgentRow {
  return {
    incarnation: "i-0",
    address: null,
    harness: null,
    tasks: [],
    workspace: null,
    lastCommit: null,
    word: null,
  };
}

test("the sections follow the spec's word order and the absent rides last", () => {
  setWords(
    {
      t3: { state: "working", at: 1 },
      t8: { state: "error", at: 2 },
    },
    2,
  );
  const sections = agentsSections(agentsPage(), words());
  const named = sections.filter((s) => s.word !== null).map((s) => s.word);
  assert.deepEqual(named.slice(0, WORD_ORDER.length), [...WORD_ORDER]);
  const populated = sections.filter((s) => s.rows.length > 0);
  assert.deepEqual(
    populated.map((s) => s.word),
    ["error", "working", null],
    "error, working, then the absent section last",
  );
  const absent = populated[2];
  assert.ok(absent, "the absent section rides last");
  assert.deepEqual(
    absent.rows.map((r) => r.incarnation),
    ["i-2"],
    "the session with no word and no address sits absent, shapeless",
  );
  assert.equal(rowTime(absent.rows[0] ?? rowTaskAgent()), null);
});

test("within a section rows order by task then workspace", () => {
  setWords({ t3: { state: "idle", at: 1 }, t8: { state: "idle", at: 2 }, t9: { state: "idle", at: 3 } }, 3);
  const page = decodePage({
    view: { major: 1 },
    kind: "document",
    location: "agents",
    seq: 1,
    content: [
      {
        kind: "group",
        location: "i-7",
        content: [
          { kind: "fact", label: "incarnation", value: "i-7" },
          { kind: "fact", label: "terminal address", value: "t3" },
          { kind: "fact", label: "tasks", value: "t-9" },
          { kind: "fact", label: "workspace", value: "/ws/b" },
        ],
      },
      {
        kind: "group",
        location: "i-3",
        content: [
          { kind: "fact", label: "incarnation", value: "i-3" },
          { kind: "fact", label: "terminal address", value: "t8" },
          { kind: "fact", label: "tasks", value: "t-2" },
          { kind: "fact", label: "workspace", value: "/ws/c" },
        ],
      },
      {
        kind: "group",
        location: "i-5",
        content: [
          { kind: "fact", label: "incarnation", value: "i-5" },
          { kind: "fact", label: "terminal address", value: "t9" },
          { kind: "fact", label: "tasks", value: "t-9" },
          { kind: "fact", label: "workspace", value: "/ws/a" },
        ],
      },
    ],
  });
  const sections = agentsSections(page, words());
  const idle = sections.find((s) => s.word === "idle");
  assert.deepEqual(
    (idle?.rows ?? []).map((r) => r.incarnation),
    ["i-3", "i-5", "i-7"],
    "t-2 before the t-9s; within t-9, workspace /ws/a before /ws/b",
  );
});

test("the row's task is the newest claim and the attach command rides the address", () => {
  setWords({}, 4);
  const sections = agentsSections(agentsPage(), words());
  const anyRow = sections.flatMap((s) => s.rows).find((r) => r.incarnation === "i-9");
  assert.ok(anyRow, "the joined session's row");
  assert.equal(rowTask(anyRow), "t-5", "latest claim first, the page's order");
  assert.equal(attachCommand(anyRow), "hc attach t3");
  const addressless = sections
    .flatMap((s) => s.rows)
    .find((r) => r.incarnation === "i-2");
  assert.ok(addressless, "the addressless session's row");
  assert.equal(attachCommand(addressless), null);
});

test("the lane's last six hours take the right two thirds", () => {
  const now = 100_000;
  const HOUR = 3_600;
  const layout = laneLayout(
    [
      { at: now - 24 * HOUR, label: "created", kind: "task_created" },
      { at: now - 12 * HOUR, label: "claimed", kind: "task_claimed" },
      { at: now - 6 * HOUR, label: "offered", kind: "task_offered" },
      { at: now - 3 * HOUR, label: "committed", kind: "task_committed" },
      { at: now - HOUR, label: "committed", kind: "task_committed" },
    ],
    now,
  );
  assert.equal(layout.sixHourMark, 1 / 3);
  assert.equal(layout.spanHours, 24);
  const [created, claimed, offered, threeHours, oneHour] = layout.positions;
  assert.ok(created && claimed && offered && threeHours && oneHour, "the five events laid out");
  assert.equal(created.x, 0, "the axis's own start sits at its left edge");
  assert.equal(offered.x, 1 / 3, "the six-hour mark sits at one third");
  assert.ok(claimed.x > 0 && claimed.x < 1 / 3, "the 12h event rides the left third");
  assert.ok(threeHours.x > 1 / 3 && threeHours.x < 1, "the 3h event rides the recent two thirds");
  assert.ok(oneHour.x > threeHours.x, "later events sit further right");
  assert.ok(oneHour.x <= 1);
  // the recent segment's midpoint (3h ago) sits at 2/3 of the axis
  const half = laneLayout([{ at: now - 3 * HOUR, label: "x", kind: "x" }], now);
  assert.ok(Math.abs((half.positions[0]?.x ?? 0) - 2 / 3) < 1 / 100, "3h ago lands at about 2/3");
});

test("events older than the axis clamp at the left edge, named so", () => {
  const now = 100_000;
  const HOUR = 3_600;
  const layout = laneLayout(
    [{ at: now - 30 * HOUR, label: "birth", kind: "task_created" }],
    now,
  );
  const [birth] = layout.positions;
  assert.ok(birth, "the clamped event laid out");
  assert.equal(birth.x, 0);
  assert.equal(birth.clamped, true, "the clamp is named, never silent");
});

test("the words store injects for the lanes and the join reads it", () => {
  setWords({ t1: { state: "done", at: 7 } }, 9);
  assert.deepEqual(words()["t1"], { state: "done", at: 7 });
  const page = agentsPage();
  const sections = agentsSections(page, words());
  // i-9 rides t3 (not injected): absent; no row in done
  const done = sections.find((s) => s.word === "done");
  assert.equal(done?.rows.length, 0);
});

test("agents span trackers: sections merge by word, every row names its project", () => {
  const row = (incarnation: string, task: string): AgentRow => ({
    incarnation, address: null, harness: null, tasks: [task], workspace: null, lastCommit: null, word: null,
  });
  const merged = mergeAgentSections([
    { project: "saccade", sections: [{ word: "working", rows: [row("i-3", "t-3")] }, { word: null, rows: [row("i-5", "t-5")] }] },
    { project: "pyrimel", sections: [{ word: "working", rows: [row("i-1", "t-1")] }] },
  ]);
  assert.deepEqual(merged.map((section) => section.word), ["working", null], "the absent-word section stays last");
  const working = merged[0]?.rows ?? [];
  assert.deepEqual(working.map((r) => `${r.project}/${r.incarnation}`), ["pyrimel/i-1", "saccade/i-3"]);
});
