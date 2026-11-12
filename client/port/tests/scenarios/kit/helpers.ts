// The shared scenario helpers: the journeys every scenario repeats,
// written once against the AppDriver interface, so both drivers run
// them unchanged. Waiting is vitest's own expect.poll, never a
// hand-written deadline loop.
import { expect } from "vitest";

import type { AppDriver, PortObservation, SeededTracker } from "./driver.ts";

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

/** An act from a citation's act menu: the real right-click opens the
 * menu, the act's key runs it (o, i, w, f, y). The elsewhere acts wait
 * for this window's binding: an o pressed while the window state is
 * still binding is dropped (src/App.ts openElsewhere). */
export async function citationAct(app: AppDriver, key: string): Promise<void> {
  if ("oiw".includes(key)) await expect.poll(() => app.windowBound(), { timeout: 15_000 }).toBe(true);
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
