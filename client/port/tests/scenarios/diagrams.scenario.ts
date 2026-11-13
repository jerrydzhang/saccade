// Mermaid diagrams are select targets (the human's m-545 item 1). The
// prototype makes every rendered diagram a figure-kind target
// (inst16.html:224, data-kind figure, label "diagram"): s labels it,
// its acts are the figure's - x Enlarge first (:47) - and a plain
// click runs that first act. The port gave diagrams no label at all.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

const DIAGRAM = "/content/1/diagram/0";

async function diagramBoard(app: AppDriver, tracker: Tracker): Promise<void> {
  const task = await tracker.createTask("diagrams: one mermaid fence");
  await tracker.commit(
    task,
    [
      "grammar: 1",
      `task: ${task}`,
      "content:",
      "  - text: A paragraph before the diagram.",
      "  - text: |",
      "      The flow:",
      "",
      "      ```mermaid",
      "      flowchart LR",
      "        cli[sac CLI] --> api[the API]",
      "      ```",
      "",
    ].join("\n"),
    "diagrams: a mermaid fence",
  );
  await openLocation(app, spec(tracker, task));
  // the diagram rendered (its outcome reported)
  await expect.poll(async () => (await app.readout()).render.some((entry) => entry.kind === "mermaid" && entry.status === "ok"), { timeout: 15_000 }).toBe(true);
  await app.press("Escape");
}

test("s labels a rendered diagram; its acts are enlarge then copy; x enlarges it", { tags: ["native"] }, async ({ app, tracker }) => {
  await diagramBoard(app, tracker);
  await app.press("s");
  await expect.poll(async () => (await app.labels()).map((label) => label.part)).toContain(DIAGRAM);
  const chip = (await app.labels()).find((label) => label.part === DIAGRAM)!;
  await app.type(chip.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect((await app.menuGeometry())?.part).toBe(DIAGRAM);
  await app.press("x");
  await expect.poll(async () => (await app.readout()).inspection?.part ?? null).toBe(DIAGRAM);
  expect(await app.shows("inspection")).toBe(true);
});

test("a plain click on a diagram enlarges it", { tags: ["native"] }, async ({ app, tracker }) => {
  await diagramBoard(app, tracker);
  await app.clickPart(DIAGRAM);
  await expect.poll(async () => (await app.readout()).inspection?.part ?? null).toBe(DIAGRAM);
});
