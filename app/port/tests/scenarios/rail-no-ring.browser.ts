// Choosing a project in the rail of a window that began empty, then
// pressing Escape, draws no focus ring on the entry: the entry is no
// keyboard focus target the key could light up.
import { expect, test } from "./kit/browser.ts";

test("empty window, the rail's project entry chosen, then Escape: no focus ring on the entry", async ({ app, tracker }) => {
  await expect.poll(() => app.shows("empty-surface")).toBe(true);
  await app.press(" ", "r");
  await expect.poll(() => app.shows(`project-${tracker.repo}`)).toBe(true);
  await app.click(`project-${tracker.repo}`);
  await expect.poll(async () => (await app.readout()).tasks?.project ?? null).toBe(tracker.repo);
  await app.press("Escape");
  const doc = (document.querySelector('iframe[data-port-suite="app"]') as HTMLIFrameElement).contentDocument!;
  const entry = doc.querySelector(`[data-port-control="project-${tracker.repo}"]`) as HTMLElement;
  expect(entry).not.toBe(null);
  const style = entry.ownerDocument.defaultView!.getComputedStyle(entry);
  expect(entry.matches(":focus-visible")).toBe(false);
  expect(style.outlineStyle === "none" || style.outlineWidth === "0px").toBe(true);
});
