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
  await openPicker(app);
  await app.type(location);
  await app.press("Enter");
  const path = location.slice(location.indexOf("/") + 1);
  return waitShown(app, path);
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
