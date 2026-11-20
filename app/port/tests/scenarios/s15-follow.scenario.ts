// S15: a normally opened board follows the tracker's current
// revision; an explicitly opened revision holds
// (t36_scenario_bodies._s15_port). The scenario commits to its own
// fresh task, so the shared seeds stay as seeded for every other test.
import { expect, test } from "#scenario";
import { contentSignature, openLocation, spec } from "./kit/helpers.ts";

const board = (task: string, body: string): string => `grammar: 1\ntask: ${task}\ncontent:\n${body}`;

test("S15 current follows the tracker; a fixed revision holds", async ({ app, tracker }) => {
  const task = await tracker.createTask("s15: follow current versus a fixed revision");
  await tracker.commit(task, board(task, "  - text: the follow board, first pass\n"), "s15: first pass");

  const before = contentSignature(await openLocation(app, spec(tracker, task)));
  const { revision } = await tracker.commit(
    task,
    board(task, "  - text: revised live mid-scenario\n  - group:\n      title: Live update\n      content:\n        - text: the live-update group child\n"),
    "s15: the live-update revision",
  );
  expect(revision).toBe(2);
  // the current board follows: the shown content changes to r2
  await expect
    .poll(async () => (await app.readout()).board.location, { timeout: 20_000 })
    .toBe(`${tracker.repo}/${task}@r2`);
  expect(contentSignature(await app.readout())).not.toBe(before);

  // the explicit revision is pinned: a later commit leaves it alone
  const pinned = await openLocation(app, spec(tracker, `${task}@r1`));
  expect(pinned.board.pinnedRevision).toBe(1);
  const held = contentSignature(pinned);
  const third = await tracker.commit(task, board(task, "  - text: a third revision while pinned\n"), "s15: the pinned-board commit");
  expect(third.revision).toBe(3);
  // a hold is an absence: it is asserted over a window, not awaited
  await new Promise((resolve) => setTimeout(resolve, 3_000));
  expect(contentSignature(await app.readout())).toBe(held);
});
