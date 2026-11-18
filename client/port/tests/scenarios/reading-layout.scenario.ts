// The board's reading layout (lib/board/reading-layout.ts): the human
// found boards small with no way to zoom, and wide margins even on a
// vertical monitor. Two scales: + - = zoom the text (the column grows
// with it), Ctrl-+ Ctrl-- Ctrl-= change the column's width alone.
// Per window, for the run (t-71@r3): they start at their defaults and
// nothing carries them across a quit.
import { expect, test } from "#scenario";
import { longBoard, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Box } from "./kit/driver.ts";

async function firstParagraph(app: AppDriver): Promise<string> {
  const parts = (await app.readout()).content?.parts ?? [];
  const paragraph = parts.find((part) => part.kind === "text");
  if (paragraph === undefined) throw new Error("the board shows no paragraph");
  return paragraph.part;
}

const width = (box: Box | null): number => (box === null ? 0 : box.right - box.left);
const height = (box: Box | null): number => (box === null ? 0 : box.bottom - box.top);

test("+ enlarges the board's text and its column together; = restores", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker, 3)));
  const part = await firstParagraph(app);
  const before = await app.partBox(part);
  expect((await app.readout()).reading).toEqual({ zoom: 1, width: 1 });

  await app.press("+", "+");
  await expect.poll(async () => (await app.readout()).reading?.zoom ?? null).toBe(1.2);
  await expect.poll(async () => height(await app.partBox(part))).toBeGreaterThan(height(before) * 1.1);
  expect(width(await app.partBox(part))).toBeGreaterThan(width(before) * 1.1);

  await app.press("-");
  await expect.poll(async () => (await app.readout()).reading?.zoom ?? null).toBe(1.1);

  await app.press("=");
  await expect.poll(async () => (await app.readout()).reading?.zoom ?? null).toBe(1);
  await expect.poll(async () => width(await app.partBox(part))).toBeCloseTo(width(before), 0);
});

test("Ctrl-+ widens the column without enlarging the text; Ctrl-= restores", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, await longBoard(tracker, 3)));
  const part = await firstParagraph(app);
  const before = await app.partBox(part);

  await app.press("Ctrl-+", "Ctrl-+", "Ctrl-+");
  await expect.poll(async () => (await app.readout()).reading?.width ?? null).toBe(1.3);
  await expect.poll(async () => width(await app.partBox(part))).toBeGreaterThan(width(before) * 1.2);
  // the text is the same size: a one-line paragraph keeps its height
  expect(height(await app.partBox(part))).toBeCloseTo(height(before), 0);
  expect((await app.readout()).reading?.zoom).toBe(1);

  await app.press("Ctrl--");
  await expect.poll(async () => (await app.readout()).reading?.width ?? null).toBe(1.2);

  await app.press("Ctrl-=");
  await expect.poll(async () => (await app.readout()).reading?.width ?? null).toBe(1);
  await expect.poll(async () => width(await app.partBox(part))).toBeCloseTo(width(before), 0);
});
