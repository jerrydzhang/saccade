// The figure viewer, ported exactly from the prototype's enlarge
// (inst16.html:241, 393-401; components16.css #lightbox): it covers the
// content area, the stage centred on it (translate -w*z/2, -h*z/2);
// z starts at 1.2; + and - zoom x1.2, = resets to 1.2 and the centre,
// the wheel zooms x1.12 a notch, 0.3 to 8; a drag pans; Escape closes;
// the hint reads "wheel or + - zoom . = resets . drag pans . Esc
// closes". The human's m-547 (1): zoom worked, nothing else did, and a
// diagram opened stuck at the top like a header.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Box, Tracker } from "./kit/driver.ts";

const PNG = Uint8Array.from(
  atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="),
  (ch) => ch.charCodeAt(0),
);

async function viewerBoard(app: AppDriver, tracker: Tracker): Promise<void> {
  const task = await tracker.createTask("viewer: a figure and a diagram");
  await tracker.commit(
    task,
    [
      "grammar: 1",
      `task: ${task}`,
      "content:",
      "  - figure:",
      "      input: figures/plot.png",
      "      caption: the plot",
      "  - text: |",
      "      ```mermaid",
      "      flowchart LR",
      "        a[one] --> b[two]",
      "      ```",
      "",
    ].join("\n"),
    "viewer: figure and diagram",
    [{ path: "figures/plot.png", content: PNG }],
  );
  await openLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).render.filter((entry) => entry.status === "ok").length, { timeout: 15_000 }).toBe(2);
  await app.press("Escape");
}

const center = (box: Box) => ({ x: (box.left + box.right) / 2, y: (box.top + box.bottom) / 2 });

async function opened(app: AppDriver, part: string): Promise<void> {
  await app.clickPart(part);
  await expect.poll(async () => (await app.readout()).inspection?.part ?? null).toBe(part);
}

async function centred(app: AppDriver): Promise<void> {
  const viewer = (await app.controlBox("inspection"))!;
  const stage = (await app.controlBox("inspection-stage"))!;
  // the viewer covers the content area, not a header strip
  expect(viewer.bottom - viewer.top).toBeGreaterThan(300);
  expect(Math.abs(center(stage).x - center(viewer).x)).toBeLessThan(2);
  expect(Math.abs(center(stage).y - center(viewer).y)).toBeLessThan(2);
}

for (const [what, part] of [["a figure", "/content/0"], ["a diagram", "/content/1/diagram/0"]] as const) {
  test(`${what} opens centred at 1.2, with the hint`, { tags: ["native"] }, async ({ app, tracker }) => {
    await viewerBoard(app, tracker);
    await opened(app, part);
    expect((await app.readout()).inspection).toMatchObject({ zoom: 1.2, x: 0, y: 0 });
    await centred(app);
    expect(await app.controlText("inspection-hint")).toBe("wheel or + - zoom \u00b7 = resets \u00b7 drag pans \u00b7 Esc closes");
  });
}

test("+ - and = zoom by the prototype's steps; Escape closes", { tags: ["native"] }, async ({ app, tracker }) => {
  await viewerBoard(app, tracker);
  await opened(app, "/content/0");
  await app.press("+");
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.44, 6);
  await app.press("+");
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.728, 6);
  await app.press("-");
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.44, 6);
  await centred(app);
  await app.press("=");
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.2, 6);
  await app.press("Escape");
  await expect.poll(async () => (await app.readout()).inspection).toBeNull();
});

test("a drag pans the stage; = re-centres it", { tags: ["native"] }, async ({ app, tracker }) => {
  await viewerBoard(app, tracker);
  await opened(app, "/content/1/diagram/0");
  const before = center((await app.controlBox("inspection-stage"))!);
  await app.drag("inspection", 60, 40);
  await expect.poll(async () => {
    const view = (await app.readout()).inspection;
    return view === null ? null : [Math.round(view.x), Math.round(view.y)];
  }).toEqual([60, 40]);
  const after = center((await app.controlBox("inspection-stage"))!);
  expect(after.x - before.x).toBeCloseTo(60, -1);
  expect(after.y - before.y).toBeCloseTo(40, -1);
  await app.press("=");
  await expect.poll(async () => (await app.readout()).inspection).toMatchObject({ zoom: 1.2, x: 0, y: 0 });
  await centred(app);
});

// browser-only: the native stack delivers no wheel events
test("the wheel zooms x1.12 a notch, about the stage's centre", async ({ app, tracker }) => {
  await viewerBoard(app, tracker);
  await opened(app, "/content/0");
  await app.wheel("inspection", -100);
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.2 * 1.12, 6);
  await app.wheel("inspection", 100);
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.2, 6);
  await centred(app);
});

// m-550: h/j/k/l pan the viewer, a step the size of j/k's line (the
// tunable INSPECTION_PAN_PX), in the viewer's own context: the view
// moves as j/k move it over a board (j: the view goes down, so the
// stage goes up), and no stroke reaches the board beneath.
test("h j k l pan the view by the named step; the board beneath never scrolls", { tags: ["native"] }, async ({ app, tracker }) => {
  await viewerBoard(app, tracker);
  await opened(app, "/content/0");
  const board = (await app.readout()).viewport.top;
  const pan = async () => {
    const view = (await app.readout()).inspection;
    return view === null ? null : [Math.round(view.x), Math.round(view.y)];
  };
  await app.press("l");
  await expect.poll(pan).toEqual([-60, 0]);
  await app.press("h", "h");
  await expect.poll(pan).toEqual([60, 0]);
  await app.press("j");
  await expect.poll(pan).toEqual([60, -60]);
  await app.press("k", "k");
  await expect.poll(pan).toEqual([60, 60]);
  expect((await app.readout()).viewport.top).toBe(board);
  await app.press("=");
  await expect.poll(pan).toEqual([0, 0]);
});

// m-552: zoom in failed for the human: a real "+" is Shift+= and the
// binding demanded no Shift. The driver now presses "+" as a person
// does (Shift held); the keypad's + arrives without Shift; both zoom in.
test("a real Shift+= and the keypad + both zoom in", { tags: ["native"] }, async ({ app, tracker }) => {
  await viewerBoard(app, tracker);
  await opened(app, "/content/0");
  await app.press("+");
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.44, 6);
  await app.press("NumpadAdd");
  await expect.poll(async () => (await app.readout()).inspection?.zoom ?? 0).toBeCloseTo(1.728, 6);
});

// S04, migrated from the t36 Python port lane (t36_scenario_bodies.py
// check_s04_deactivation): deactivation separates input from content -
// a figure's plain click opens the viewer; board keys (G, gg) do
// nothing to the board beneath while it is open; Escape closes it and
// adds no trail entry; the board is still loaded. (The lane's zoom
// steps were the old quarter ladder; the viewer's own keys are pinned
// above.)
test("S04 the open viewer takes every key; Escape closes it with the trail unchanged", { tags: ["native"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, `${tracker.tasks.landing}@r2`));
  await expect.poll(async () => (await app.readout()).render.some((entry) => entry.kind === "figure" && entry.status === "ok")).toBe(true);
  await app.press("Escape");
  const figure = ((await app.readout()).content?.parts ?? []).find((part) => part.kind === "figure")!.part;
  await app.clickPart(figure);
  await expect.poll(async () => (await app.readout()).inspection?.part ?? null).toBe(figure);
  const before = await app.readout();
  await app.press("G");
  await app.press("g", "g");
  // a hold: board keys must leave the board beneath where it was
  await new Promise((resolve) => setTimeout(resolve, 400));
  const beneath = await app.readout();
  expect(beneath.viewport.top).toBe(before.viewport.top);
  expect(beneath.inspection?.part).toBe(figure);
  await app.press("Escape");
  await expect.poll(async () => (await app.readout()).inspection).toBeNull();
  const closed = await app.readout();
  expect(closed.trail).toEqual(before.trail);
  expect(closed.board.status).toBe("loaded");
});
