// t-40: a fenced code block is coloured by language (lib/highlight.ts),
// and the colours are presentation only - the block's text is its
// source and find matches across the coloured spans as it does in prose.
// The colours themselves are not in the readout; the run tokenizing is
// unit-tested (tests/highlight.test.ts).
import { expect, test } from "#scenario";
import { find, openLocation, spec } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";

const SOURCE = ["fn main() {", "    // count the rows", '    let rows = vec!["a", "b"];', '    println!("{} rows", rows.len());', "}"];

async function rustFence(app: AppDriver, tracker: Tracker): Promise<void> {
  const task = await tracker.createTask("code: a rust fence");
  const lines = ["grammar: 1", `task: ${task}`, "content:", "  - text: |", "      Prose before the fence.", "", "      ```rust"];
  for (const line of SOURCE) lines.push(`      ${line}`);
  lines.push("      ```", "", "      Prose after the fence.");
  await tracker.commit(task, lines.join("\n") + "\n", "code: a rust fence");
  await openLocation(app, spec(tracker, task));
}

test("a highlighted code block shows its source between its prose", { tags: ["native"] }, async ({ app, tracker }) => {
  await rustFence(app, tracker);
  const text = await app.partText("/content/0");
  expect(text).toContain(SOURCE.join("\n"));
  expect(text).toContain("Prose before the fence.");
  expect(text).toContain("Prose after the fence.");
});

test("find matches across a code block's colours", { tags: ["native"] }, async ({ app, tracker }) => {
  await rustFence(app, tracker);
  // "fn main" spans a keyword, a space and a title
  expect((await find(app, "fn main")).count).toBe(1);
  await expect.poll(async () => (await app.findMarks()).map((mark) => mark.text).join("")).toBe("fn main");
});

test("find counts every occurrence in a code block, whatever its colour", { tags: ["native"] }, async ({ app, tracker }) => {
  await rustFence(app, tracker);
  // "rows" sits in a comment, a binding, a string and a receiver
  expect((await find(app, "rows")).count).toBe(4);
  await expect.poll(async () => (await app.findMarks()).map((mark) => mark.text)).toEqual(["rows", "rows", "rows", "rows"]);
});
