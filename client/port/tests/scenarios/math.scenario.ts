// t-27: valid LaTeX from stiff-sindy's migrated t-0 board renders.
// t-39's triage reproduced two causes on main 484038f, both fixed:
// - a display block whose $$ shares its line with the body
//   ("$$\begin{aligned}" ... "\end{aligned}$$") parsed as a fenced
//   block whose opening line is meta: \begin{aligned} was dropped, the
//   closing line never closed, and the block swallowed the text after it
//   (src/lib/markdown.ts, bareMathFence);
// - \mathbb and \mathcal need MathJax 4's dynamic font ranges, which
//   the synchronous tex2svg could not wait for ("MathJax retry"); the
//   ranges are served from the bundle (vite.config.ts) and conversion is
//   the promise API (src/lib/mathjax.ts). Native too: the font ranges
//   load under the app's own protocol and CSP.
// The stored text is not rewritten.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";

const BLOCKS = [
  // inline: blackboard and calligraphic
  "Inline $(\\mathbb{Z}_2)^p$ and $\\mathcal{F}_S$.",
  // display, delimiters on their own lines
  "$$\n\\begin{aligned}\n-X_S^\\top(y-X_S\\beta_S)+\\sigma\\odot w_S &= 0\\\\\n\\beta_S &= G^{-1}\\big(X_S^\\top y-\\sigma\\odot w_S\\big)\n\\end{aligned}\n$$",
  // display, delimiters on the body's first and last lines, with a
  // paragraph line before and after (the stored board's layout)
  "On-support:\n$$\\begin{aligned}\n0 &< \\sigma\\odot\\beta_S\\\\\n0 &< \\sigma\\odot G^{-1}(X_S^\\top y-\\sigma\\odot w_S)\n\\end{aligned}$$\nAfter the block.",
];

test("t-27 valid LaTeX renders inline and in both display layouts", { tags: ["native"] }, async ({ app, tracker }) => {
  const task = await tracker.createTask("math: the t-27 examples");
  const lines = ["grammar: 1", `task: ${task}`, "content:"];
  for (const block of BLOCKS) {
    lines.push("  - text: |");
    for (const line of block.split("\n")) lines.push(`      ${line}`);
  }
  await tracker.commit(task, lines.join("\n") + "\n", "math: the t-27 examples");
  await openLocation(app, spec(tracker, task));
  // two inline, two display expressions, every one rendered
  await expect
    .poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "math").map((entry) => entry.status), { timeout: 15_000 })
    .toEqual(["ok", "ok", "ok", "ok"]);
  // the text after a same-line display block stays text
  expect(await app.partText("/content/2")).toContain("After the block.");
});

test("invalid math fails visibly with its TeX message, the text around it intact", { tags: ["native"] }, async ({ app, tracker }) => {
  const task = await tracker.createTask("math: an invalid expression");
  const board = ["grammar: 1", `task: ${task}`, "content:", "  - text: |", "      Before $\\frac{a$ after."].join("\n") + "\n";
  await tracker.commit(task, board, "math: an invalid expression");
  await openLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "math").map((entry) => entry.status), { timeout: 15_000 }).toEqual(["error"]);
  const [outcome] = (await app.readout()).render.filter((entry) => entry.kind === "math");
  expect(outcome!.message).not.toContain("[object Object]");
  expect(outcome!.message).toMatch(/[Mm]issing close brace|Extra open brace/);
  const text = await app.partText("/content/0");
  expect(text).toContain("Before");
  expect(text).toContain("after.");
});
