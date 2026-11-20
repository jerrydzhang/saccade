// A remote project's task list, chosen from the rail of a window that
// began empty, draws its rows - including a row whose agents cell
// names its owner. The project's machine is not the lane's own and
// every tracker request answers late, as a remote machine reached
// through the core does. The human's list stayed on "loading..." while
// the readout said loaded: drawing a row that names an agent threw,
// and the frame never left its pending line.
import { expect, test } from "./kit/browser.ts";

test("empty window, a remote project's rail entry: the list draws its rows, an owned task's too", async ({ laned, tracker }) => {
  const owned = await tracker.createTask("rail-open-remote: a task someone works");
  await tracker.claimTask(owned);
  const app = await laned({ machine: "other", latencyMs: 60 });
  await expect.poll(() => app.shows("empty-surface")).toBe(true);
  await app.press(" ", "r");
  const entry = `rail/project/${tracker.repo}`;
  await expect.poll(() => app.partText(entry), { timeout: 20_000 }).not.toBe(null);
  await app.clickPart(entry);
  await expect
    .poll(async () => {
      const tasks = (await app.readout()).tasks;
      return { project: tasks?.project ?? null, status: tasks?.status ?? null };
    }, { timeout: 20_000 })
    .toEqual({ project: tracker.repo, status: "loaded" });
  // the surface draws the rows the readout names, not the pending line
  const row = (await app.readout()).tasks?.rows.find((entry) => entry.task === owned)?.part ?? null;
  expect(row).not.toBe(null);
  await expect.poll(() => app.partText(row!)).toContain("port-suite");
  expect(await app.controlText("content")).not.toContain("loading...");
});
