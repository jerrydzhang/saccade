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
