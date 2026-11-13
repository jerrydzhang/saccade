// The task list follows the record as a board follows its current
// revision (S15): a task created while the list is shown appears
// without reopening it. The human hit it on the native app - an open
// board refreshed, the task list never did.
import { expect, test } from "#scenario";
import { spec, submitLocation } from "./kit/helpers.ts";

test("a task created while the list is shown appears in it", async ({ app, tracker }) => {
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  const task = await tracker.createTask("tasks-follow: born while the list is shown");
  await expect
    .poll(async () => (await app.readout()).tasks?.rows.map((row) => row.task) ?? [], { timeout: 20_000 })
    .toContain(task);
});
