// The task list's highlight (m-547 (4)): j/k scroll the list with no
// row cursor (S17), yet a row stayed highlighted - the one that scrolled
// under a pointer resting there (CSS :hover). A row is highlighted by
// the pointer MOVING onto it, and looks like its neighbours otherwise
// (the prototype's ledger rows: ink only, no shading). Each row is
// compared with its own look before any pointer moved: rows of
// different states may differ among themselves.
import { expect, test } from "#scenario";
import { openPlane } from "./kit/helpers.ts";
import type { AppDriver } from "./kit/driver.ts";

async function visibleRows(app: AppDriver): Promise<string[]> {
  const obs = await app.readout();
  const rows = obs.tasks?.rows ?? [];
  const out: string[] = [];
  for (const row of rows) {
    const look = await app.partLook(row.part);
    if (look !== null) out.push(row.part);
  }
  return out;
}

test("j/k under a resting pointer highlight no row; a pointer move highlights its row", { tags: ["native"] }, async ({ app, tracker }) => {
  for (let index = 0; index < 60; index++) await tracker.createTask(`highlight row ${index + 1}`);
  await openPlane(app, tracker);
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  await app.press("Escape");
  const rows = await visibleRows(app);
  expect(rows.length).toBeGreaterThan(10);
  // each row's own look with no pointer over the list: rows may differ
  // among themselves (their states), never from their own baseline
  const baseline = new Map<string, string>();
  for (const part of rows) baseline.set(part, JSON.stringify(await app.partLook(part)));
  // the pointer moves onto a row: that row is highlighted
  const under = rows[5]!;
  await app.pointAt(under);
  await expect.poll(async () => JSON.stringify(await app.partLook(under))).not.toBe(baseline.get(under));
  // then the keyboard scrolls the list under the resting pointer
  await app.press("j", "j", "j", "j", "j");
  await expect.poll(async () => (await app.readout()).viewport.top).toBeGreaterThan(200);
  // the mark clears on the scroll event, which the engine dispatches a
  // frame after the scroll position the readout already shows: wait for
  // it, bounded - a mark that never clears still fails here. The bound
  // carries the read's own cost: one evaluation is a WebDriver round
  // trip per mounted row (64 rows here; the list does not virtualize),
  // 834ms on a quiet machine and 1218-1341ms under the lane's load,
  // while vitest's default 1000ms deadline runs once from the poll's
  // start - it aborted the single evaluation mid-read even with every
  // row already back at its baseline look, which was the flake (t-72:
  // the native lane's failures tracked the machine's day, the dumps
  // never showed a row marked; the clear itself was never late)
  const changed = async (): Promise<string[]> => {
    const out: string[] = [];
    for (const part of rows) {
      if (JSON.stringify(await app.partLook(part)) !== baseline.get(part)) out.push(part);
    }
    return out;
  };
  await expect
    .poll(changed, { timeout: 10_000, message: "after a keyboard scroll every row looks as it did with no pointer" })
    .toEqual([]);
});
