// S17's remaining oracle, migrated from the t36 Python port lane
// (t36_differential.py _s17_port): the task row's menu carries o i w y
// in that order (F6); the strip reads unlinked, then names the link
// the row's Enter recorded; a second Enter REUSES the link (no further
// window); the list holds its surface and gains no trail entry for a
// delivery elsewhere. (One window's Enter births a linked window - the
// birth needs a second live webview, native S18's; here a second window
// exists, so the first Enter asks and the answer records the link.)
import { expect, test } from "./kit/browser.ts";
import { answerAsk, spec, submitLocation, whenBound } from "./kit/helpers.ts";
import type { AppDriver } from "./kit/driver.ts";

async function rowMenu(app: AppDriver): Promise<void> {
  await app.press("Escape");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  const [chip] = await app.labels();
  await app.type(chip!.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
}

test("S17 the row menu is o i w y; a second Enter reuses the link; the list holds", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ others: ["w-b"] });
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  await whenBound(app);
  const depth = (await app.readout()).trail.depth;
  await rowMenu(app);
  expect(await app.menuActs()).toEqual(["o", "i", "w", "y"]);
  expect((await app.strip()).link).toBe("unlinked");
  await app.press("Enter");
  await answerAsk(app, "w-b");
  await expect.poll(async () => (await app.readout()).window.link).toBe("w-b");
  const linked = "w-b";
  const delivered = shell.argsOf("open_in_window").length;
  expect((await app.strip()).link).toBe(`-> ${linked}`);
  const windows = shell.windows().length;
  await rowMenu(app);
  await app.press("Enter");
  await expect.poll(() => shell.argsOf("open_in_window").length).toBe(delivered + 1);
  expect(shell.argsOf("open_in_window").at(-1)!.label).toBe(linked);
  expect(shell.argsOf("broadcast_jump_labels").length, "the second Enter does not ask").toBe(1);
  expect(shell.windows().length, "no further window").toBe(windows);
  const obs = await app.readout();
  expect(obs.surface).toBe("tasks");
  expect(obs.trail.depth).toBe(depth);
});
