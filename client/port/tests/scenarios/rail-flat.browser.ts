// t-63's flat rail: no word headings - each agent row carries its own
// state (its glyph, drawn from data-port-word) - in importance order
// of the word; an unregistered terminal reads its address in the id
// column, then Herdr's label in the name column. In select mode a
// label inside another labelled target is drawn nested (outlined): a
// project's agent, never the project or a foot-group terminal.
import { expect, test } from "./kit/browser.ts";
import { openPlane } from "./kit/helpers.ts";

// three running terminals no tracker registered and no cwd places: the
// rail's foot section holds them (lib/tasks/rail.ts)
const words = {
  generation: 1,
  connected: true,
  words: {
    t7a: { app: "claude", state: "idle", where: "elsewhere / 1", at: 1 },
    t7b: { app: "claude", state: "working", msg: "Drafting the notes", where: "notes run", at: 1 },
    t7c: { app: "pi", state: "blocked", where: "pi run", at: 1 },
  },
};

test("the rail is flat: rows in importance order, no word headings, a stray's label in the name column", async ({ bridged, tracker }) => {
  const { app } = await bridged({ words });
  await openPlane(app, tracker);
  await expect.poll(() => app.shows("unregistered")).toBe(true);
  const foot = (await app.controlText("unregistered")) ?? "";
  // blocked before working before idle (wordRank), each row once
  const order = ["pi run", "notes run", "elsewhere / 1"].map((label) => foot.indexOf(label));
  expect(order.every((at) => at >= 0)).toBe(true);
  expect([...order].sort((a, b) => a - b)).toEqual(order);
  // no row of the rail is a word: the words ride the rows' glyphs
  for (const heading of ["blocked", "working", "idle"]) {
    expect(foot.split("\n").map((line) => line.trim())).not.toContain(heading);
  }
  // the id column holds the address, the name column Herdr's label
  const row = (await app.partText("rail/terminal/t7c")) ?? "";
  expect(row.indexOf("t7c")).toBeGreaterThanOrEqual(0);
  expect(row.indexOf("t7c")).toBeLessThan(row.indexOf("pi run"));
});

test("select mode: an agent inside a project is labelled nested; the project and a foot terminal are not", async ({ bridged, tracker }) => {
  // the stray is placed in the project by its cwd, the tracker's repo
  const root = (await (await fetch(`${tracker.base}/view/v1/`)).json()) as { repo_path?: string };
  expect(root.repo_path).toBeTruthy();
  const placed = {
    ...words,
    words: { ...words.words, t7d: { app: "claude", state: "working", where: "inside", cwd: root.repo_path, at: 1 } },
  };
  const { app } = await bridged({ words: placed });
  await openPlane(app, tracker);
  await expect.poll(() => app.partText("rail/terminal/t7d")).not.toBe(null);
  await app.press("Escape");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  const nestedOf = new Map((await app.labels()).map((chip) => [chip.part, chip.nested]));
  expect(nestedOf.get(`rail/project/${tracker.repo}`)).toBe(false);
  expect(nestedOf.get("rail/terminal/t7d")).toBe(true);
  expect(nestedOf.get("rail/terminal/t7c")).toBe(false);
});
