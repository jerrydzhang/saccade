// The task list in a narrow window (t-41): the human keeps the list
// narrow, and the fixed five-column grid squeezed the title until long
// titles wrapped over many lines. A title stays on one line, cut with
// an ellipsis; as the window narrows the lesser columns (agents, then
// latest commit) drop out first, from the header as from the rows; the
// task id, title and age stay; the row's act menu names the full title.
// Browser-only: the native lane's kiosk-shell holds the window's size.
import { expect, test } from "#scenario";
import { openPlane } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

const LONG = "a task whose title runs long enough to wrap over many lines in a narrow task list, "
  + "far past what one line of any window the human keeps holds, and then some more words";
// the latest commit column names the commit by its sha
const SHA = /\b[0-9a-f]{40}\b/;

type Seeded = { short: string; long: string[]; committed: string };

async function seed(tracker: Tracker): Promise<Seeded> {
  const short = await tracker.createTask("narrow short");
  const long: string[] = [];
  for (let index = 0; index < 3; index++) long.push(await tracker.createTask(`${LONG} (${index + 1})`));
  const committed = long[0]!;
  await tracker.commit(committed, `grammar: 1\ntask: ${committed}\ncontent:\n  - text: one block\n`, "tasks-narrow: the one commit");
  return { short, long, committed };
}

async function partOf(app: AppDriver, task: string): Promise<string> {
  const row = (await app.readout()).tasks?.rows.find((entry) => entry.task === task);
  if (row === undefined) throw new Error(`${task} is not listed`);
  return row.part;
}

/** Each row's height, the short row's first. */
async function heights(app: AppDriver, seeded: Seeded): Promise<number[]> {
  const out: number[] = [];
  for (const task of [seeded.short, ...seeded.long]) {
    const box = await app.partBox(await partOf(app, task));
    if (box === null) throw new Error(`${task} is not rendered`);
    out.push(Math.round(box.bottom - box.top));
  }
  return out;
}

/** The visible lines of the group that lists `task`: its column
 * header's labels each stand on a line of their own (grid items). */
async function groupLines(app: AppDriver, task: string): Promise<string[]> {
  const part = await partOf(app, task);
  const obs = await app.readout();
  const group = obs.tasks?.groups.find((entry) => part.startsWith(`${entry.part}/`));
  if (group === undefined) throw new Error(`no group holds ${part}`);
  const text = (await app.partText(group.part)) ?? "";
  return text.split("\n").map((line) => line.trim());
}

async function expectOneLine(app: AppDriver, seeded: Seeded): Promise<void> {
  const [short, ...long] = await heights(app, seeded);
  for (const height of long) expect(Math.abs(height - short!), `a long row is ${height}px, a one-line row ${short}px`).toBeLessThanOrEqual(1);
}

test("long titles stay on one line, and the lesser columns drop out as the window narrows", async ({ app, tracker }) => {
  const seeded = await seed(tracker);
  await openPlane(app, tracker);
  await expect.poll(async () => (await app.readout()).tasks?.rows.some((row) => row.task === seeded.committed) ?? false).toBe(true);
  await app.press("Escape");
  const committed = await partOf(app, seeded.committed);

  // a wide window: every column, each row one line
  await app.resize(1100, 800);
  await expectOneLine(app, seeded);
  expect(await groupLines(app, seeded.committed)).toContain("agents");
  expect(await groupLines(app, seeded.committed)).toContain("latest commit");
  expect(await app.partText(committed)).toMatch(SHA);

  // narrower: agents drop out first; the latest commit stays
  await app.resize(560, 800);
  await expectOneLine(app, seeded);
  await expect.poll(() => groupLines(app, seeded.committed)).not.toContain("agents");
  expect(await groupLines(app, seeded.committed)).toContain("latest commit");
  expect(await app.partText(committed)).toMatch(SHA);

  // narrow: the latest commit drops out too; id, title and age stay
  await app.resize(340, 800);
  await expectOneLine(app, seeded);
  await expect.poll(() => groupLines(app, seeded.committed)).not.toContain("latest commit");
  expect(await groupLines(app, seeded.committed)).toEqual(expect.arrayContaining(["task", "title", "age"]));
  const text = (await app.partText(committed)) ?? "";
  expect(text).not.toMatch(SHA);
  expect(text).toContain(seeded.committed);
  // the row's right edge stays inside the window: nothing scrolls sideways
  const box = await app.partBox(committed);
  expect(box!.right).toBeLessThanOrEqual(340);

  // the full title stays reachable: the row's act menu names it
  await app.clickPart(committed, { button: "right" });
  await expect.poll(() => app.menuLabel()).toBe(`${seeded.committed} ${LONG} (1)`);
});
