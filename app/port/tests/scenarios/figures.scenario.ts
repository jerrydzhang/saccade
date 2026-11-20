// D3/D4 (the human's first-use defects): a figure is a target whatever
// its file type. s labels it, its label opens its acts, x enlarges it,
// and a plain click enlarges it too (the prototype: inst16.html:47 the
// figure kind's acts, :211-214 a figure of any type, :392-400 the
// lightbox enlarges what is shown). The human's boards carry mostly
// non-image figures (.py, .ts, .txt, .json, .diff ...); the port
// rendered every non-.svg figure as an image/png data URL
// (lib/board/hydrate.ts), so a text figure was a broken image.
// Read through the labels, the menu and the readout's inspection.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

// a 1x1 PNG (the landing seed's own bytes, client/harness/t36_seeds.py)
const PNG = Uint8Array.from(
  atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="),
  (ch) => ch.charCodeAt(0),
);
const SCRIPT = "def sweep(hz):\n    return hz * 2\n";

// the human's boards nest their figures in groups (t-35, t-36): the
// same two figures, top level and inside a group
async function figureBoard(app: AppDriver, tracker: Tracker): Promise<void> {
  const task = await tracker.createTask("figures: an image and a script, top level and grouped");
  await tracker.commit(
    task,
    [
      "grammar: 1",
      `task: ${task}`,
      "content:",
      "  - text: Two figures follow.",
      "  - figure:",
      "      input: figures/plot.png",
      "      caption: the plot",
      "  - figure:",
      "      input: figures/sweep.py",
      "      caption: the sweep script",
      "  - group:",
      "      title: The same figures, grouped",
      "      content:",
      "        - text: Inside a group, as the real boards carry them.",
      "        - figure:",
      "            input: figures/plot.png",
      "            caption: the grouped plot",
      "        - figure:",
      "            input: figures/sweep.py",
      "            caption: the grouped sweep script",
      "",
    ].join("\n"),
    "figures: png and py",
    [
      { path: "figures/plot.png", content: PNG },
      { path: "figures/sweep.py", content: SCRIPT },
    ],
  );
  await openLocation(app, spec(tracker, task));
  // the figures displayed, as a person sees them before acting: the
  // board reports each figure's outcome once its asset is shown
  await expect.poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "figure").length).toBe(4);
  await app.press("Escape");
}

const FIGURES = ["/content/1", "/content/2", "/content/3/content/1", "/content/3/content/2"];

for (const part of FIGURES) {
  test(`s labels the figure ${part}; its label opens its acts and x enlarges it`, { tags: ["native"] }, async ({ app, tracker }) => {
    await figureBoard(app, tracker);
    await app.press("s");
    await expect.poll(async () => (await app.labels()).map((label) => label.part)).toContain(part);
    const chip = (await app.labels()).find((label) => label.part === part)!;
    await app.type(chip.label);
    await expect.poll(() => app.menuOpen()).toBe(true);
    await app.press("x");
    await expect.poll(async () => (await app.readout()).inspection?.part ?? null).toBe(part);
    expect(await app.shows("inspection")).toBe(true);
    await app.press("Escape");
    await expect.poll(async () => (await app.readout()).inspection).toBeNull();
  });

  test(`a plain click on the figure ${part} enlarges it`, { tags: ["native"] }, async ({ app, tracker }) => {
    await figureBoard(app, tracker);
    await app.clickPart(part);
    await expect.poll(async () => (await app.readout()).inspection?.part ?? null).toBe(part);
    expect(await app.shows("inspection")).toBe(true);
  });
}

test("a text figure shows its text, not a broken image", { tags: ["native"] }, async ({ app, tracker }) => {
  await figureBoard(app, tracker);
  await expect.poll(async () => (await app.partText("/content/2")) ?? "").toContain("def sweep(hz)");
});



// D3/D4 on the human's own boards (a read-only copy of t-36, 37
// figures): before the fix not one figure finished displaying - the
// board re-rendered every ~150ms, forever. Its markdown carries raw
// HTML, which the renderer reports as a diagnostic; the render effect
// pushed each diagnostic into reactive state it also read, so every
// render scheduled the next, and every figure was back to its
// placeholder before its asset landed. Nothing could be enlarged, by
// click or by s/x. Raw HTML beside a figure: the figure displays and
// enlarges.
test("a board with raw HTML still displays and enlarges its figures", { tags: ["native"] }, async ({ app, tracker }) => {
  const task = await tracker.createTask("figures: raw HTML beside a figure");
  await tracker.commit(
    task,
    [
      "grammar: 1",
      `task: ${task}`,
      "content:",
      "  - text: |",
      "      Authored <b>raw HTML</b> the viewer reports and does not render.",
      // a board of the real boards' weight: each render costs enough
      // that the loop never settles (t-36 rendered every ~150ms)
      ...Array.from({ length: 200 }, (_, index) => `  - text: paragraph ${index + 1}, with *emphasis* and \`code\`, of a long board`),
      "  - figure:",
      "      input: figures/plot.png",
      "      caption: the plot",
      "",
    ].join("\n"),
    "figures: raw HTML and a figure",
    [{ path: "figures/plot.png", content: PNG }],
  );
  await openLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "figure").length, { timeout: 10_000 }).toBe(1);
  await app.press("Escape");
  await app.clickPart("/content/201");
  await expect.poll(async () => (await app.readout()).inspection?.part ?? null).toBe("/content/201");
});
