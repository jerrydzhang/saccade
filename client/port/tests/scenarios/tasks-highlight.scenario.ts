// The task list's highlight (m-547 (4)): j/k scroll the list with no
// row cursor (S17), yet a row stayed highlighted - the one that scrolled
// under a pointer resting there (CSS :hover). A row is highlighted by
// the pointer MOVING onto it, and looks like its neighbours otherwise
// (the prototype's ledger rows: ink only, no shading).
import { expect, test } from "#scenario";
import { submitLocation, spec } from "./kit/helpers.ts";
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
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  await app.press("Escape");
  const rows = await visibleRows(app);
  expect(rows.length).toBeGreaterThan(10);
  // the pointer moves onto a row: that row is highlighted
  const under = rows[5]!;
  await app.pointAt(under);
  const plain = (await app.partLook(rows[1]!))!;
  await expect.poll(async () => JSON.stringify(await app.partLook(under))).not.toBe(JSON.stringify(plain));
  // then the keyboard scrolls the list under the resting pointer
  await app.press("j", "j", "j", "j", "j");
  await expect.poll(async () => (await app.readout()).viewport.top).toBeGreaterThan(200);
  const looks = new Set<string>();
  for (const part of rows) looks.add(JSON.stringify(await app.partLook(part)));
  expect([...looks], "every row looks the same after a keyboard scroll").toHaveLength(1);
});
