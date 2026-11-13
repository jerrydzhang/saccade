// The mechanisms behind m-547 (5)'s slowness, pinned without timing:
// find's units are derived once per board, each block's markdown is
// parsed once whoever reads it, and figures load concurrently.
import assert from "node:assert/strict";
import { test } from "vitest";
import type { Board } from "../src/lib/board/model.ts";

const board = (): Board =>
  ({
    content: [
      { kind: "text", markdown: "first block, t-3@r14 cited", pos: { start: { line: 1 }, end: { line: 1 } } },
      { kind: "text", markdown: "second block\n\n```mermaid\nflowchart LR\n  a --> b\n```\n", pos: { start: { line: 2 }, end: { line: 6 } } },
    ],
  }) as unknown as Board;

test("find's units are derived once per board (every keystroke re-derived them)", async () => {
  const { findUnitsOf } = await import("../src/lib/find/units.ts");
  const shown = board();
  assert.equal(findUnitsOf(shown), findUnitsOf(shown));
});

test("each block's markdown is parsed once across targets and find", async () => {
  const { markdownParseCount } = await import("../src/lib/markdown.ts");
  const { boardTargetsOf } = await import("../src/lib/board/targets.ts");
  const { findUnitsOf } = await import("../src/lib/find/units.ts");
  const shown = board();
  const before = markdownParseCount();
  boardTargetsOf(shown);
  findUnitsOf(shown);
  boardTargetsOf(shown);
  // two blocks, two parses - however many readers
  assert.ok(markdownParseCount() - before <= 2, `parsed ${markdownParseCount() - before} times`);
});

test("the figure pool keeps up to its limit in flight, in item order", async () => {
  const { inPool } = await import("../src/lib/board/pool.ts");
  let inFlight = 0;
  let most = 0;
  const started: number[] = [];
  await inPool([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6, async (item) => {
    started.push(item);
    inFlight += 1;
    most = Math.max(most, inFlight);
    await new Promise((resolve) => setTimeout(resolve, 5));
    inFlight -= 1;
  });
  assert.equal(most, 6);
  assert.deepEqual(started, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
});
