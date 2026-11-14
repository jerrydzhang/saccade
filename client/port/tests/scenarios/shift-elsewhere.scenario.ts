// Shift opens elsewhere (t-47; t-14 "Two acts, never a mode"): a
// shifted or middle click on a citation runs its o act (open at the
// link - with one window, a new window that becomes the link), and
// Shift+Enter in the picker opens the location in a new window while
// this one keeps what it shows. Native windows only. Each pointer
// gesture is the first act of its own test: a WebDriver pointer
// action into a window another has taken focus from was sometimes
// never delivered (kit/helpers.ts citationActByLabel).
import { expect, test } from "#scenario";
import { openLocation, openPicker, spec, whenBound } from "./kit/helpers.ts";

for (const gesture of ["shift", "middle"] as const) {
  test(`a ${gesture} click on a citation opens it at the link`, { tags: ["multi-window"] }, async ({ app, tracker }) => {
    const { citingChild: citing, cited } = tracker.tasks;
    await openLocation(app, spec(tracker, citing));
    await whenBound(app);
    await app.click("citation", gesture === "shift" ? { shift: true } : { button: "middle" });
    await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
    const receiver = (await app.windows()).find((label) => label !== "main")!;
    await expect
      .poll(async () => (await app.readout(receiver)).board.location, { timeout: 20_000 })
      .toBe(`${tracker.repo}/${cited}@r1`);
    const source = await app.readout("main");
    expect(source.window.link, "o's new window becomes the link").toBe(receiver);
    expect(source.board.asked, "the source still shows its own board").toBe(`${tracker.repo}/${citing}`);
  });
}

test("Shift+Enter in the picker opens a new window; this one keeps its board", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  const { citingChild: citing, cited } = tracker.tasks;
  await openLocation(app, spec(tracker, citing));
  await whenBound(app);
  await openPicker(app);
  await app.type(spec(tracker, cited));
  await app.press("Shift-Enter");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const opened = (await app.windows()).find((label) => label !== "main")!;
  await expect
    .poll(async () => (await app.readout(opened)).board.location, { timeout: 20_000 })
    .toBe(`${tracker.repo}/${cited}@r1`);
  const main = await app.readout("main");
  expect(main.board.asked).toBe(`${tracker.repo}/${citing}`);
  expect(main.window.link, "a new window never becomes the link").toBeNull();
});
