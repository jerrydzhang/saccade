// t-27: valid LaTeX from stiff-sindy's migrated t-0 board renders.
// Pending (t-39's triage reproduced two causes on main 484038f):
// - a display block whose $$ shares its line with the body
//   ("$$\begin{aligned}" ... "\end{aligned}$$") parses as a fenced
//   block whose opening line is meta: \begin{aligned} is dropped, the
//   closing line never closes, and the block swallows the text after it;
// - \mathbb and \mathcal need MathJax 4's dynamic font ranges, which
//   the synchronous tex2svg cannot wait for ("MathJax retry").
// The tag comes off when both render; the stored text is not rewritten.
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

test("t-27 valid LaTeX renders inline and in both display layouts", { tags: ["pending"] }, async ({ app, tracker }) => {
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
