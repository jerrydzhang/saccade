// The shared scenario helpers: the journeys every scenario repeats,
// written once against the AppDriver interface, so both drivers run
// them unchanged. Waiting is vitest's own expect.poll, never a
// hand-written deadline loop.
import { expect } from "vitest";

import { inputOf, type AppDriver, type InputView, type PortObservation, type SeededTracker, type Tracker } from "./driver.ts";

/** A location spec on the seeded tracker: host/repo/path. */
export function spec(tracker: SeededTracker, path: string): string {
  return `${tracker.host}/${tracker.repo}/${path}`;
}

/** Open the picker by its chord (space space) from a clear state. */
export async function openPicker(app: AppDriver): Promise<void> {
  await app.press("Escape");
  await app.press(" ", " ");
  await expect.poll(() => app.shows("picker-input")).toBe(true);
}

/** Open a location through the picker with real typing and Enter,
 * then wait until the shown board answers it. */
export async function openLocation(app: AppDriver, location: string): Promise<PortObservation> {
  await submitLocation(app, location);
  const path = location.slice(location.indexOf("/") + 1);
  return waitShown(app, path);
}

/** Submit a location through the picker (real typing and Enter);
 * the caller waits for whatever surface it opens. */
export async function submitLocation(app: AppDriver, location: string): Promise<void> {
  await openPicker(app);
  await app.type(location);
  await app.press("Enter");
}

/** Wait until the window shows the open it asked for, `path`
 * (repo/t-N or repo/t-N@rK), loaded; answers that readout. The shown
 * location is the revision that answered: repo/t-N@rK. */
export async function waitShown(app: AppDriver, path: string, window?: string): Promise<PortObservation> {
  await expect
    .poll(async () => {
      const board = (await app.readout(window)).board;
      return { status: board.status, asked: board.asked };
    })
    .toEqual({ status: "loaded", asked: path });
  return app.readout(window);
}

/** The shown content's signature: location plus every part's path
 * and kind (a part count cannot tell two same-sized boards apart). */
export function contentSignature(obs: PortObservation): string {
  const parts = (obs.content?.parts ?? []).map((part) => `${part.part}:${part.kind}`).sort();
  return `${obs.board.location ?? ""} ${parts.join(" ")}`;
}

/** Open find with its key and type a query into the find field. */
export async function find(app: AppDriver, query: string): Promise<PortObservation["find"]> {
  await app.press("Escape");
  await app.press("/");
  await expect.poll(async () => (await app.readout()).find.open).toBe(true);
  await app.type(query);
  await expect.poll(async () => (await app.readout()).find.query).toBe(query);
  return (await app.readout()).find;
}

/** Wait until this window has bound its window state: an elsewhere
 * act (o, i, w, a task row's Enter) pressed before that is dropped
 * (src/App.ts openElsewhere; the readout carries no binding fact). */
export async function whenBound(app: AppDriver): Promise<void> {
  await expect.poll(() => app.windowBound(), { timeout: 15_000 }).toBe(true);
}

/** An act from a citation's act menu: the real right-click opens the
 * menu, the act's key runs it (o, i, w, f, y). The elsewhere acts wait
 * for this window's binding: an o pressed while the window state is
 * still binding is dropped (src/App.ts openElsewhere). */
export async function citationAct(app: AppDriver, key: string): Promise<void> {
  if ("oiw".includes(key)) await whenBound(app);
  await app.click("citation", { button: "right" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press(key);
}

/** Answer a live ask with a window's chip, typed into the ask field. */
export async function answerAsk(app: AppDriver, window: string): Promise<void> {
  await expect.poll(() => app.shows("ask-input")).toBe(true);
  const chip = (await app.readout()).window.ask?.chips.find((entry) => entry.window === window)?.chip;
  if (chip === undefined) throw new Error(`the ask offers no chip for ${window}`);
  await app.type(chip);
  await app.press("Enter");
}

/** The readout's PROPOSED input field (kit/driver.ts InputView); a
 * build without it fails the read, naming the field. */
export async function readInput(app: AppDriver): Promise<InputView> {
  const input = inputOf(await app.readout());
  if (input === undefined) throw new Error("the readout carries no input field (the batch 3 proposed contract, kit/driver.ts InputView)");
  return input;
}

/** A board long enough to page through: `blocks` short text blocks on
 * a fresh task. Answers the task id. */
export async function longBoard(tracker: Tracker, blocks = 150): Promise<string> {
  const task = await tracker.createTask(`a long board: ${blocks} blocks`);
  const body = Array.from({ length: blocks }, (_, index) => `  - text: paragraph ${index + 1} of a board long enough to page through\n`).join("");
  await tracker.commit(task, `grammar: 1\ntask: ${task}\ncontent:\n${body}`, "a long board");
  return task;
}
