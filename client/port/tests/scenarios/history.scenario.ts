// History mode (t-49; t-14 "The viewer", History): space h shows the
// shown revision against the one before it - the words it removed and
// added marked in place, a removed component placed back where it
// stood, the commit that landed it beside the board; h and l step
// time; n jumps to a change; space h again leaves, showing the board
// as it was asked for. History is a mode, never a trail entry.
import { expect, test } from "#scenario";
import { spec, submitLocation } from "./kit/helpers.ts";

const R1 = (task: string) => `grammar: 1
task: ${task}
content:
  - text: |
      # Account

      The loop holds.
  - group:
      title: An idea that fell away
      content:
        - text: |
            The long argument.
`;

const R2 = (task: string) => `grammar: 1
task: ${task}
content:
  - text: |
      # Account

      The loop breaks.
`;

test("history shows a revision's changes, steps time, and leaves in place", async ({ app, tracker }) => {
  const task = await tracker.createTask("history: two revisions");
  await tracker.commit(task, R1(task), "first account");
  await tracker.commit(task, R2(task), "rewrite: the loop breaks; the idea folds away");
  await submitLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).board.location, { timeout: 20_000 }).toBe(`${tracker.repo}/${task}@r2`);
  const trailBefore = (await app.readout()).trail.depth;

  await app.press(" ", "h");
  await expect.poll(async () => (await app.readout()).board.history?.revision ?? null, { timeout: 20_000 }).toBe(2);
  const shown = await app.readout();
  expect(shown.board.history).toMatchObject({ latest: 2, changed: ["/content/0", "/content/1"] });
  expect(await app.controlText("history-message")).toBe("rewrite: the loop breaks; the idea folds away");
  // the changed text reads both words, painted (no sentinel survives);
  // the removed group stands where it stood
  const changedText = (await app.partText("/content/0")) ?? "";
  expect(changedText).toContain("holds");
  expect(changedText).toContain("breaks");
  expect(changedText).not.toMatch(/[\uE000-\uE003]/);
  expect((await app.partText("/content/1")) ?? "").toContain("The long argument.");

  // h: the revision before - the first, everything added
  await app.press("h");
  await expect.poll(async () => (await app.readout()).board.history?.revision ?? null).toBe(1);
  expect(await app.controlText("history-message")).toBe("first account");
  // l: back to the newest; l past it stays
  await app.press("l");
  await expect.poll(async () => (await app.readout()).board.history?.revision ?? null).toBe(2);
  await app.press("l");
  expect((await app.readout()).board.history?.revision).toBe(2);
  // n: the first change
  await app.press("n");
  await expect.poll(async () => (await app.readout()).board.history?.at ?? null).toBe(0);

  // space h: leave - the board as asked, following again; no trail entry
  await app.press(" ", "h");
  await expect.poll(async () => (await app.readout()).board.history ?? null).toBeNull();
  const left = await app.readout();
  expect(left.board.location).toBe(`${tracker.repo}/${task}@r2`);
  expect(left.board.asked).toBe(`${tracker.repo}/${task}`);
  expect(left.content?.parts.length).toBe(1);
  expect(left.trail.depth).toBe(trailBefore);
});
