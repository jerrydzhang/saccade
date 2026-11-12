// The Tasks instrument's pure model layer, unit-tested: the pipeline
// page restated as rows with canonical identities, the targets the
// select session serves, and the identity rules (a row leaving the
// page leaves the dataset).

import assert from "node:assert/strict";

import { ageOf } from "../src/lib/tasks/page.ts";
import test from "node:test";
import type { ViewPage } from "../src/lib/pages.ts";
import {
  taskRowOf,
  tasksPageOf,
  tasksTargetsOf,
} from "../src/lib/tasks/page.ts";

function tasksPage(): ViewPage {
  return {
    view: { major: 1 },
    kind: "document",
    location: "tasks",
    seq: null,
    content: [
      {
        kind: "group",
        title: "waiting",
        content: [
          {
            kind: "table",
            columns: ["task", "title", "agents", "latest commit", "latest revision"],
            rows: [
              {
                cells: ["t-0", "the cited seed account", "", "4c0e", "r1"],
                location: "t-0",
                at: 1794467029,
              },
              {
                cells: ["t-1", "the citing seed account", "i-1", "d0f2", "r1"],
                location: "t-1",
                at: 1794467038,
              },
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
            columns: ["task", "title", "agents", "latest commit", "latest revision"],
            rows: [],
          },
        ],
      },
    ],
  };
}

test("the tasks page restates groups and rows with canonical part paths", () => {
  const model = tasksPageOf(tasksPage());
  assert.equal(model.groups.length, 2);
  const waiting = model.groups[0];
  assert.equal(waiting?.title, "waiting");
  assert.equal(waiting?.part, "/content/0");
  const rows = waiting?.rows ?? [];
  assert.equal(rows.length, 2);
  assert.equal(rows[0]?.part, "/content/0/content/0/row/0");
  assert.equal(rows[0]?.task, "t-0");
  assert.equal(rows[0]?.title, "the cited seed account");
  assert.equal(rows[0]?.commit, "4c0e");
  assert.equal(rows[0]?.revision, "r1");
  assert.equal(rows[0]?.at, 1794467029);
  // an empty table still lands its group
  assert.deepEqual(model.groups[1]?.rows, []);
});

test("a row's link is its task; the link wins over the first cell", () => {
  const page = tasksPage();
  const group = page.content?.[0];
  if (group === undefined || group.kind !== "group") throw new Error("fixture shape");
  const table = group.content[0];
  if (table === undefined || table.kind !== "table") throw new Error("fixture shape");
  table.rows[0] = { cells: ["t-9", "renamed elsewhere", "", "", ""], location: "t-0", at: 1 };
  const model = tasksPageOf(page);
  assert.equal(model.groups[0]?.rows[0]?.task, "t-0", "the row's link carries the identity");
  assert.equal(model.groups[0]?.rows[0]?.title, "renamed elsewhere");
});

test("targets serve every row in document order with the row's label line", () => {
  const model = tasksPageOf(tasksPage());
  const targets = tasksTargetsOf(model);
  assert.equal(targets.length, 2);
  assert.deepEqual(
    targets.map((target) => target.part),
    ["/content/0/content/0/row/0", "/content/0/content/0/row/1"],
  );
  assert.equal(targets[0]?.kind, "task");
  assert.equal(targets[0]?.text, "t-0 the cited seed account");
  assert.equal(targets[0]?.location, "t-0");
  assert.equal(targets[1]?.text, "t-1 the citing seed account");
});

test("a row that leaves the page leaves the dataset (scenario 12 identity)", () => {
  const before = tasksPageOf(tasksPage());
  const target = tasksTargetsOf(before)[0];
  if (target === undefined) throw new Error("fixture shape");
  const after = tasksPageOf({ ...tasksPage(), content: [] });
  assert.equal(tasksTargetsOf(after).some((entry) => entry.part === target.part), false);
  assert.equal(taskRowOf(after, target.part), undefined);
  assert.equal(taskRowOf(before, target.part)?.task, "t-0");
});

test("a page without tables answers an empty model, never an error", () => {
  const model = tasksPageOf({
    view: { major: 1 },
    kind: "document",
    location: "tasks",
    seq: null,
    content: [{ kind: "text", text: "nothing runs" }],
  });
  assert.deepEqual(model.groups, []);
  assert.deepEqual(tasksTargetsOf(model), []);
});

// -- the age column (F4): the prototype's ago rules ------------------

test("ageOf renders minutes, hours and days from the activity moment", () => {
  const now = 1_700_000_000_000;
  assert.equal(ageOf(null, now), "", "no moment carries no age");
  assert.equal(ageOf(now - 5 * 60_000, now), "5m");
  assert.equal(ageOf(now - 59 * 60_000, now), "59m");
  assert.equal(ageOf(now - 2 * 3_600_000, now), "2h");
  assert.equal(ageOf(now - 47 * 3_600_000, now), "47h");
  assert.equal(ageOf(now - 3 * 86_400_000, now), "3d");
  assert.equal(ageOf(now + 60_000, now), "0m", "the future clamps to now");
});
