// The D1 diagnostic's app half (src/lib/input/key-trace.ts): with the
// shell's SACCADE_KEY_TRACE set, every keydown sends one `key` line
// through key_trace carrying the routing answer and the scroller's
// top/height/client before and after; unset, the app sends nothing but
// the one startup query. The shell half (stderr) is commands.rs
// key_trace. Read through the mocked shell's call log.
import { expect, test } from "./kit/browser.ts";
import { longBoard, openLocation, spec } from "./kit/helpers.ts";
import type { Shell } from "./kit/browser.ts";

function traceLines(shell: Shell): string[] {
  return shell.argsOf("key_trace").flatMap((args) => (args.lines as string[] | undefined) ?? []);
}

test("with the trace on, j sends a key line naming the route and the scroll it made", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ keyTrace: true });
  await openLocation(app, spec(tracker, await longBoard(tracker)));
  await app.press("Escape");
  await expect.poll(() => traceLines(shell).some((line) => line.startsWith("init "))).toBe(true);
  const before = (await app.readout()).viewport.top;
  await app.press("j");
  // the capture-phase `seen` line names j too; the `key` line is the
  // routed record
  const keyLine = (entry: string): boolean => /^key #\d+ .* key=j /.test(entry);
  await expect.poll(() => traceLines(shell).filter(keyLine).length).toBe(1);
  const line = traceLines(shell).find(keyLine)!;
  expect(traceLines(shell).some((entry) => /^seen #\d+ .* key=j /.test(entry))).toBe(true);
  expect(line).toContain("route=window-then-buffer");
  expect(line).toContain("buffer=invoke:board.scroll-down");
  expect(line).toContain("disposition=consume");
  expect(line).toContain("surface=board");
  // the scroller's top/height/client before, and 150ms later moved by
  // the line step
  const before150 = /s\(top\/height\/client\)=(\d+)\//.exec(line);
  const later = /later=(\d+)\//.exec(line);
  expect(Number(before150?.[1])).toBe(before);
  expect(Number(later?.[1])).toBeGreaterThan(before);
});

test("a bare modifier key is never consumed (m-536: key=Super disposition=consume)", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ keyTrace: true });
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  await app.press("Escape");
  for (const modifier of ["Shift", "Control", "Alt", "Meta"]) {
    await app.press(modifier);
    const line = (entry: string): boolean => new RegExp(`^key #\\d+ .* key=${modifier} `).test(entry);
    await expect.poll(() => traceLines(shell).filter(line).length).toBeGreaterThan(0);
    expect(traceLines(shell).find(line)).toContain("route=modifier");
    expect(traceLines(shell).find(line)).toContain("preventedAfter=0");
  }
});

test("in select mode the key line carries the labels and the typed buffer", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged({ keyTrace: true });
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  await app.press("Escape");
  await app.press("s");
  await expect.poll(async () => (await app.labels()).length).toBeGreaterThan(0);
  const first = (await app.labels())[0]!;
  // the picker's typing sent digits too: only lines after this point
  const line = (entry: string): boolean => /^key #\d+ .* key=1 /.test(entry);
  const earlier = traceLines(shell).filter(line).length;
  await app.press("1");
  await expect.poll(() => traceLines(shell).filter(line).length).toBe(earlier + 1);
  const text = traceLines(shell).filter(line).at(-1)!;
  expect(text).toContain("buf.ctx=captured/commands");
  expect(text).toContain(`${first.label}=${first.part}`);
  expect(text).toMatch(/select\.typed=/);
});

test("with the trace off, no key line is sent", async ({ bridged, tracker }) => {
  const { app, shell } = await bridged();
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  await app.press("Escape", "j", "k");
  // a hold: the trace would have sent within its 150ms window
  await new Promise((resolve) => setTimeout(resolve, 400));
  expect(traceLines(shell)).toEqual([]);
});
