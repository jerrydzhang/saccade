// The app's half of the multi-window behaviour, against the mocked
// shell (kit/shell-mock.ts): the link state and the strip, the o/i/w
// decision (src/lib/elsewhere.ts elsewhereDecision), the dead-link
// fallback, and which shell command fires with which arguments. The
// window store is the real throwaway tracker's. What needs a second
// LIVE webview (the receiver's boot, the delivery landing there, the
// link a new window completes once it attaches) is the native
// project's: s18-linked-delivery.scenario.ts.
// Sources: t36_checks._s18_port, t36_differential._s17_port.
import { expect, test } from "./kit/browser.ts";
import { answerAsk, citationAct, openLocation, spec, submitLocation, whenBound } from "./kit/helpers.ts";
import type { AppDriver, Tracker } from "./kit/driver.ts";
import type { Shell } from "./kit/browser.ts";

// the citing seed's incorporation names the cited board at r1
const citedAt = (tracker: Tracker): string => `${tracker.repo}/${tracker.tasks.cited}@r1`;

async function onCiting(app: AppDriver, tracker: Tracker): Promise<void> {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
}

// this window's own window-store state, as it bound it to the shell
function ownState(shell: Shell): string {
  const bound = shell.argsOf("bind_window_state").at(-1)?.stateId;
  if (typeof bound !== "string") throw new Error("the app never bound its window state");
  return bound;
}

// o through an ask answered with `window`'s chip: the link is set
async function linkThroughAsk(app: AppDriver, window: string): Promise<void> {
  await citationAct(app, "o");
  await answerAsk(app, window);
  await expect.poll(async () => (await app.readout()).window.link).toBe(window);
}

test("one window, no link: o opens a new window born at the citation", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged();
  await onCiting(app, tracker);
  expect((await app.readout()).window).toMatchObject({ label: "main", link: null, ask: null });
  expect((await app.strip()).link).toBe("unlinked");

  await citationAct(app, "o");
  await expect.poll(() => shell.argsOf("open_suite_window").length).toBe(1);
  const stateId = shell.argsOf("open_suite_window")[0]!.stateId as string;
  // the delivery rides the new state's birth location, never an eval
  // into the new webview (App.ts openInNewWindow)
  expect((await tracker.windowState(stateId, shell.device)).location).toBe(citedAt(tracker));
  expect(shell.windows().map((entry) => entry.label)).toHaveLength(2);
  expect(shell.argsOf("open_in_window")).toEqual([]);
  expect(shell.argsOf("broadcast_jump_labels")).toEqual([]);
  // this window still shows the citing board
  expect((await app.readout()).board.asked).toBe(`${tracker.repo}/${tracker.tasks.citingChild}`);
});

test("several windows, no link: o asks; the answer delivers there and becomes the link", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ others: ["w-b"] });
  await onCiting(app, tracker);

  await citationAct(app, "o");
  await expect.poll(() => shell.argsOf("broadcast_jump_labels").length).toBe(1);
  expect(shell.argsOf("broadcast_jump_labels")[0]).toEqual({
    labels: [{ window: "main", chip: "a" }, { window: "w-b", chip: "b" }],
  });
  const ask = (await app.readout()).window.ask;
  expect(ask).toMatchObject({ act: "link", chips: [{ window: "main", chip: "a" }, { window: "w-b", chip: "b" }] });
  expect(ask!.spec).toBe(citedAt(tracker));
  expect(shell.argsOf("open_suite_window")).toEqual([]);

  await answerAsk(app, "w-b");
  await expect.poll(() => shell.argsOf("open_in_window")).toEqual([{ label: "w-b", location: ask!.spec }]);
  expect(shell.argsOf("broadcast_jump_labels_clear").length).toBeGreaterThanOrEqual(1);
  await expect.poll(async () => (await app.readout()).window).toMatchObject({ link: "w-b", ask: null });
  expect((await app.strip()).link).toBe("-> w-b");
  // the link is the store's fact: this window's state points at w-b's
  const receiver = shell.windows().find((entry) => entry.label === "w-b")!.stateId;
  expect((await tracker.windowState(ownState(shell), shell.device)).link).toBe(receiver);
});

test("a live link: o delivers without asking; i asks and leaves the link; w opens a new viewer", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ others: ["w-b"] });
  await onCiting(app, tracker);
  await linkThroughAsk(app, "w-b");
  const asks = shell.argsOf("broadcast_jump_labels").length;
  const spec = shell.argsOf("open_in_window")[0]!.location;

  // o: the link delivers, no ask, no new window
  await citationAct(app, "o");
  await expect.poll(() => shell.argsOf("open_in_window").length).toBe(2);
  expect(shell.argsOf("open_in_window")[1]).toEqual({ label: "w-b", location: spec });
  expect(shell.argsOf("broadcast_jump_labels").length).toBe(asks);
  expect(shell.argsOf("open_suite_window")).toEqual([]);

  // i: several windows ask, once; Escape cancels, the link stays
  await citationAct(app, "i");
  await expect.poll(async () => (await app.readout()).window.ask?.act ?? null).toBe("once");
  expect(shell.argsOf("broadcast_jump_labels").length).toBe(asks + 1);
  await app.press("Escape");
  await expect.poll(async () => (await app.readout()).window.ask).toBeNull();
  expect((await app.readout()).window.link).toBe("w-b");

  // w: always a new viewer, born at the citation; the link stays
  await citationAct(app, "w");
  await expect.poll(() => shell.argsOf("open_suite_window").length).toBe(1);
  const born = shell.argsOf("open_suite_window")[0]!.stateId as string;
  expect((await tracker.windowState(born, shell.device)).location).toBe(spec);
  expect((await app.readout()).window.link).toBe("w-b");
  expect((await app.strip()).link).toBe("-> w-b");
});

test("a dead link among several windows: o falls back to asking, without the dead window", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ others: ["w-b", "w-c"] });
  await onCiting(app, tracker);
  await linkThroughAsk(app, "w-b");
  const delivered = shell.argsOf("open_in_window").length;

  shell.close("w-b");
  await citationAct(app, "o");
  await expect.poll(() => shell.argsOf("broadcast_jump_labels").length).toBe(2);
  expect(shell.argsOf("broadcast_jump_labels")[1]).toEqual({
    labels: [{ window: "main", chip: "a" }, { window: "w-c", chip: "b" }],
  });
  expect(shell.argsOf("open_in_window").length).toBe(delivered);
  expect(shell.argsOf("open_suite_window")).toEqual([]);
});

test("a dead link with one window left: o opens a new window, never delivers into the dead one", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ others: ["w-b"] });
  await onCiting(app, tracker);
  await linkThroughAsk(app, "w-b");
  const delivered = shell.argsOf("open_in_window").length;

  shell.close("w-b");
  await citationAct(app, "o");
  await expect.poll(() => shell.argsOf("open_suite_window").length).toBe(1);
  expect(shell.argsOf("open_in_window").length).toBe(delivered);
});

test("S17 the task row's Enter delivers through the link and the list holds its surface", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged();
  await submitLocation(app, spec(tracker, "tasks"));
  await expect
    .poll(async () => {
      const obs = await app.readout();
      return { surface: obs.surface, status: obs.tasks?.status ?? null };
    })
    .toEqual({ surface: "tasks", status: "loaded" });
  const rows = (await app.readout()).tasks!.rows;
  expect(rows.length).toBeGreaterThanOrEqual(2);
  expect((await app.strip()).link).toBe("unlinked");

  // a real label keystroke selects the row and opens its menu
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  const [chip] = await app.labels();
  const row = rows.find((entry) => entry.part === chip!.part)!;
  await app.type(chip!.label);
  await expect.poll(() => app.menuOpen()).toBe(true);
  expect((await app.readout()).tasks!.selection).not.toBeNull();

  // Enter runs the primary, o: one window and no link opens a new
  // window born at the row's board (once the window state is bound)
  await whenBound(app);
  await app.press("Enter");
  await expect.poll(() => shell.argsOf("open_suite_window").length).toBe(1);
  const born = shell.argsOf("open_suite_window")[0]!.stateId as string;
  expect((await tracker.windowState(born, shell.device)).location).toContain(`/${row.task}`);
  const obs = await app.readout();
  expect(obs.surface).toBe("tasks");
  expect(obs.trail.entries.map((entry) => entry.location).join(" ")).not.toContain(`/${row.task}@`);
});
