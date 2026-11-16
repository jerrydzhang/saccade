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
// t-61: MathJax 4's a11y suite (explorer, speech) boots on by default;
// it is configured off through the menu settings (src/lib/mathjax.ts),
// so rendered math carries no tabindex - a click cannot steal focus and
// an ordinary keystroke cannot open the explorer's help dialog with its
// docs.mathjax.org link. Browser-only: the assertion reads the app
// document's mjx-containers, which the readout does not carry.
import { expect, test } from "#scenario";
import type { AppDriver, Tracker } from "./kit/driver.ts";
import { openLocation, spec } from "./kit/helpers.ts";

// the kit's browser driver mounts the app as a sibling iframe; the app
// window's own constructors dispatch events inside its document (a
// DOM window carries the constructors; TS's Window type just does not
// declare them)
function appWindow(): Window & { KeyboardEvent: typeof KeyboardEvent } {
  const frame = document.querySelector<HTMLIFrameElement>("[data-port-suite=app]");
  if (!frame?.contentWindow) throw new Error("the app frame is not mounted");
  return frame.contentWindow as Window & { KeyboardEvent: typeof KeyboardEvent };
}

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

test("t-61 rendered math is inert: no tabindex, no explorer keys, no docs link", async ({ app, tracker }) => {
  const task = await tracker.createTask("math: the a11y suite stays off");
  const board = ["grammar: 1", `task: ${task}`, "content:", "  - text: |", "      Inline $(\\mathbb{Z}_2)^p$ and plain $x^2$."].join("\n") + "\n";
  await tracker.commit(task, board, "math: the a11y suite stays off");
  await openLocation(app, spec(tracker, task));
  await expect
    .poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "math").map((entry) => entry.status), { timeout: 15_000 })
    .toEqual(["ok", "ok"]);
  const win = appWindow();
  const containers = win.document.querySelectorAll("mjx-container");
  expect(containers.length).toBe(2);
  // a click on math lands on the container; with the a11y suite off it
  // must not be in the tab order at all
  for (const container of containers) expect(container.getAttribute("tabindex")).toBeNull();
  // force the focus a click would give, then ordinary keys: the
  // explorer's help dialog carries a docs.mathjax.org anchor
  const container = containers[0] as HTMLElement;
  container.focus();
  for (const key of ["h", "Enter"]) {
    container.dispatchEvent(new win.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
  }
  // an absence: the dialog and its docs anchor would have appeared
  // within this window had the explorer been on
  await new Promise((resolve) => setTimeout(resolve, 400));
  expect(win.document.querySelector('a[href*="docs.mathjax"]')).toBeNull();
  expect(win.document.querySelector("dialog, mjx-help, [role=dialog]")).toBeNull();
  // the keys left the rendered math untouched
  await expect.poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "math").map((entry) => entry.status)).toEqual(["ok", "ok"]);
});

// t-62: rendered math is a part of its own - <block>/math/<n> in
// document order, the TeX riding the target - and its menu's one act,
// y, copies the expression's LaTeX source: the TeX a paste into notes
// wants, not the authored $...$ form and not rendered markup. The
// journeys are the machinery's own (scenario 9): select mode and a
// right-click open the menu, a plain click runs the first act
// directly - so a plain click on math copies.
const TEX_INLINE = "(\\mathbb{Z}_2)^p";
const TEX_DISPLAY = "E = mc^2";

async function mathPartsBoard(app: AppDriver, tracker: Tracker): Promise<void> {
  const task = await tracker.createTask("math: parts and their copy act");
  const board = [
    "grammar: 1",
    `task: ${task}`,
    "content:",
    "  - text: |",
    "      Inline $(\\mathbb{Z}_2)^p$ and a display:",
    "",
    "      $$",
    "      E = mc^2",
    "      $$",
  ].join("\n") + "\n";
  await tracker.commit(task, board, "math: parts and their copy act");
  await openLocation(app, spec(tracker, task));
  await expect
    .poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "math").map((entry) => entry.status), { timeout: 15_000 })
    .toEqual(["ok", "ok"]);
  await app.press("Escape");
}

test("t-62 a math part's menu is y copy LaTeX: the label journey and the right-click journey copy the TeX exactly", async ({ app, tracker }) => {
  await mathPartsBoard(app, tracker);
  // the label journey: s labels the expressions, typing a label
  // selects the part and opens its menu
  await app.press("s");
  await expect.poll(async () => (await app.labels()).map((label) => label.part)).toContain("/content/0/math/0");
  const chip = (await app.labels()).find((label) => label.part === "/content/0/math/0")!;
  await app.type(chip.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  // one act, the copy key every kind shares
  expect(await app.menuActs()).toEqual(["y"]);
  await app.clickMenuAct("y");
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toBe(TEX_INLINE);
  // close before execute: the act ran, the menu is gone
  await expect.poll(() => app.menuOpen()).toBe(false);
  // the right-click journey: the same act list at the pointer, the
  // display expression's own TeX
  await app.clickPart("/content/0/math/1", { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect(await app.menuActs()).toEqual(["y"]);
  await app.clickMenuAct("y");
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toBe(TEX_DISPLAY);
});

test("t-62 a plain click on math copies its TeX directly (the primary act), no menu", async ({ app, tracker }) => {
  await mathPartsBoard(app, tracker);
  expect((await app.readout()).copy).toBeNull();
  await app.clickPart("/content/0/math/0");
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toBe(TEX_INLINE);
  expect(await app.menuOpen()).toBe(false);
  await app.clickPart("/content/0/math/1");
  await expect.poll(async () => (await app.readout()).copy?.payload ?? null).toBe(TEX_DISPLAY);
  expect(await app.menuOpen()).toBe(false);
});

test("t-62 an invalid expression is no math part: no math menu, the authored source stays as text", async ({ app, tracker }) => {
  const task = await tracker.createTask("math: an invalid expression is no part");
  const board = ["grammar: 1", `task: ${task}`, "content:", "  - text: |", "      Before $\\frac{a$ after."].join("\n") + "\n";
  await tracker.commit(task, board, "math: an invalid expression is no part");
  await openLocation(app, spec(tracker, task));
  await expect.poll(async () => (await app.readout()).render.filter((entry) => entry.kind === "math").map((entry) => entry.status), { timeout: 15_000 }).toEqual(["error"]);
  const win = appWindow();
  // no math part anywhere in the document: the menu machinery has no
  // math target to open for the error span
  expect(win.document.querySelector('[data-port-kind="math"]')).toBeNull();
  const errors = win.document.querySelectorAll<HTMLElement>(".math-error");
  expect(errors.length).toBe(1);
  expect(errors[0]!.hasAttribute("data-port-part")).toBe(false);
  // the authored source stands as inspectable, selectable text
  expect(errors[0]!.textContent).toContain("\\frac{a");
  const [outcome] = (await app.readout()).render.filter((entry) => entry.kind === "math");
  expect(outcome!.status).toBe("error");
});
