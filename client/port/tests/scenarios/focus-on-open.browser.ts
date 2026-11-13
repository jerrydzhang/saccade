// Opening through the link FOCUSES the receiving window (human ruling
// m-547 (3), reversing t-14's "a new window never asks for focus" for
// the elsewhere acts): o, i, w and the task row's Enter each end with
// the platform's activation request for the receiver - the shell's
// focus_suite_window, the window jump's own door. Read through the
// mocked shell's call log (kit/shell-mock.ts); that the request really
// activates a window on the person's compositor is the native lane's
// and the human's trace's (window-focus.scenario.ts).
import { expect, test } from "./kit/browser.ts";
import { answerAsk, citationAct, openLocation, spec, submitLocation, whenBound } from "./kit/helpers.ts";
import type { Shell } from "./kit/browser.ts";

const focused = (shell: Shell): unknown[] => shell.argsOf("focus_suite_window").map((args) => args.label);

test("o into a new window focuses that window", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged();
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await citationAct(app, "o");
  await expect.poll(() => shell.windows().length).toBe(2);
  const born = shell.windows().find((entry) => entry.label !== "main")!.label;
  await expect.poll(() => focused(shell)).toEqual([born]);
});

test("the ask's answer, a delivery through the link, and w each focus their receiver", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ others: ["w-b"] });
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  // o with no link asks; the answer delivers to w-b and links it
  await citationAct(app, "o");
  await answerAsk(app, "w-b");
  await expect.poll(() => focused(shell)).toEqual(["w-b"]);
  // o through the live link
  await citationAct(app, "o");
  await expect.poll(() => focused(shell)).toEqual(["w-b", "w-b"]);
  // i answered with w-b
  await citationAct(app, "i");
  await answerAsk(app, "w-b");
  await expect.poll(() => focused(shell)).toEqual(["w-b", "w-b", "w-b"]);
  // w: a new viewer
  await citationAct(app, "w");
  await expect.poll(() => shell.windows().length).toBe(3);
  const third = shell.windows().find((entry) => entry.label !== "main" && entry.label !== "w-b")!.label;
  await expect.poll(() => focused(shell)).toEqual(["w-b", "w-b", "w-b", third]);
});

test("the task row's Enter, opening elsewhere, focuses the receiver", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged();
  await submitLocation(app, spec(tracker, "tasks"));
  await expect.poll(async () => (await app.readout()).tasks?.status ?? null).toBe("loaded");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  const [chip] = await app.labels();
  await app.type(chip!.label.toLowerCase());
  await expect.poll(() => app.menuOpen()).toBe(true);
  await whenBound(app);
  await app.press("Enter");
  await expect.poll(() => shell.windows().length).toBe(2);
  const born = shell.windows().find((entry) => entry.label !== "main")!.label;
  await expect.poll(() => focused(shell)).toEqual([born]);
});
