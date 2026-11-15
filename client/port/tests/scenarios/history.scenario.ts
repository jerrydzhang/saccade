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

/** A board of headed sections (each text its own key, so the tracker
 * aligns them by heading): `lead` sections first, then 30 sections
 * whose bodies read `word`, then a tail when asked. */
function sections(task: string, word: string, lead: number, tail = false): string {
  const parts = [
    ...Array.from({ length: lead }, (_, index) => `  - text: |\n      ## Lead ${index + 1}\n\n      A lead section.\n`),
    ...Array.from({ length: 30 }, (_, index) => `  - text: |\n      ## Section ${index + 1}\n\n      The ${word} body of section ${index + 1}.\n`),
    ...(tail ? ["  - text: |\n      ## Tail\n\n      The tail.\n"] : []),
  ];
  return `grammar: 1\ntask: ${task}\ncontent:\n${parts.join("")}`;
}

test("the part being read holds its screen position; t tracks it and H L walk its changes", async ({ app, tracker }) => {
  const task = await tracker.createTask("history: the read part holds still");
  await tracker.commit(task, sections(task, "first", 0), "r1: thirty sections");
  await tracker.commit(task, sections(task, "second", 0), "r2: every body rewritten");
  await tracker.commit(task, sections(task, "second", 3), "r3: three leads above");
  await tracker.commit(task, sections(task, "fourth", 3), "r4: every body rewritten again");
  await tracker.commit(task, sections(task, "fourth", 3, true), "r5: a tail below");
  await submitLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).board.location, { timeout: 20_000 }).toBe(`${tracker.repo}/${task}@r5`);

  // read into the middle: the part at the reading height and its place
  await app.press("Ctrl-d", "Ctrl-d");
  await expect.poll(async () => (await app.readout()).window.place?.part ?? null).toMatch(/^\/content\/([4-9]|[1-3]\d)$/);
  const read = (await app.readout()).window.place!.part!;
  const heading = ((await app.partText(read)) ?? "").split("\n")[0]!.trim();
  expect(heading).toMatch(/^Section \d+$/);
  const { top, left } = (await app.partBox(read))!;
  /** The shown part reading `heading` stands where it stood, and the
   * board column never moved sideways. */
  const holds = async (path: string) => {
    expect(((await app.partText(path)) ?? "").split("\n")[0]!.trim()).toBe(heading);
    await expect.poll(async () => Math.abs(((await app.partBox(path))?.top ?? Infinity) - top)).toBeLessThanOrEqual(2);
    expect(Math.abs((await app.partBox(path))!.left - left)).toBeLessThanOrEqual(1);
  };
  const revision = async (expected: number) =>
    expect.poll(async () => (await app.readout()).board.history?.revision ?? null, { timeout: 20_000 }).toBe(expected);
  const index = Number(/\d+$/.exec(read)![0]);
  const before = `/content/${index - 3}`; // the same section without the leads

  // entering keeps it in place; r5 and r4 and r3 hold the same paths
  await app.press(" ", "h");
  await revision(5);
  await holds(read);
  await app.press("h");
  await revision(4);
  await holds(read);
  await app.press("h");
  await revision(3);
  await holds(read);
  // r2 has no leads: the section moves up the board, never on screen
  await app.press("h");
  await revision(2);
  await holds(before);

  // t: the part at the reading height is tracked
  await app.press("t");
  await expect.poll(async () => (await app.readout()).board.history?.tracked ?? null).toBe(before);
  expect(await app.controlText("history-tracked")).toBe(`tracking ${heading}`);

  // H walks to the revisions that changed it: r1 added it; nothing before
  await app.press("H");
  await revision(1);
  await holds(before);
  await app.press("H");
  await new Promise((resolve) => setTimeout(resolve, 300));
  expect((await app.readout()).board.history?.revision).toBe(1);
  // L: r2 rewrote it, r3 left it alone, r4 rewrote it; r5 left it alone
  await app.press("L");
  await revision(2);
  await holds(before);
  await app.press("L");
  await revision(4);
  await holds(read);
  expect((await app.readout()).board.history?.tracked).toBe(read);
  await app.press("L");
  await new Promise((resolve) => setTimeout(resolve, 300));
  expect((await app.readout()).board.history?.revision).toBe(4);

  // Escape stops tracking; leaving keeps the part in place
  await app.press("Escape");
  await expect
    .poll(async () => {
      const shown = (await app.readout()).board.history;
      return shown === null ? "left history" : shown.tracked;
    })
    .toBeNull();
  await app.press(" ", "h");
  await expect.poll(async () => (await app.readout()).board.history ?? null).toBeNull();
  expect((await app.readout()).board.location).toBe(`${tracker.repo}/${task}@r4`);
  await holds(read);
});

test("history reads the tracker's component changes: a moved section shows where it went", async ({ app, tracker }) => {
  const task = await tracker.createTask("history: a moved section");
  const board = (order: string[]) =>
    `grammar: 1\ntask: ${task}\ncontent:\n` + order.map((name) => `  - text: |\n      ## ${name}\n\n      The ${name} section.\n`).join("");
  await tracker.commit(task, board(["One", "Two", "Three"]), "three sections");
  await tracker.commit(task, board(["Two", "Three", "One"]), "One moves to the end");
  await submitLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).board.location, { timeout: 20_000 }).toBe(`${tracker.repo}/${task}@r2`);
  await app.press(" ", "h");
  await expect.poll(async () => (await app.readout()).board.history?.revision ?? null, { timeout: 20_000 }).toBe(2);
  // changes.rs names one move; One is not placed back where it stood
  expect((await app.readout()).board.history?.changed).toEqual(["/content/2"]);
  expect(((await app.partText("/content/2")) ?? "").split("\n")[0]).toBe("One");
  expect(await app.partText("/content/3")).toBeNull();
});
