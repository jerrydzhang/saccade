// The key surfaces' projections (t-59, lib/input/help.ts): which-key
// shows the pending prefix and its next strokes; help sets every key
// in columns by the context it comes from. Both read the matcher's own
// views, so they cannot drift from what the keys run.

import assert from "node:assert/strict";
import { test } from "vitest";

import { createMatcher } from "../src/lib/input/key-matcher.ts";
import { boardReadingContext, windowContext } from "../src/lib/input/contexts.ts";
import { displayKeys, helpGroups, pendingKeys } from "../src/lib/input/help.ts";

const windowKeys = () =>
  windowContext({
    openPicker: () => {}, closeOverlays: () => {}, pickerIsOpen: () => false,
    trailBack: (): boolean => true, trailForward: (): boolean => true,
    toggleHistory: () => {}, linkByLabels: () => {}, unlink: () => {}, toggleRail: () => {}, registerProject: () => {}, keepView: () => {}, showHelp: () => {},
  });

const boardKeys = () =>
  boardReadingContext({
    scrollToTop: () => {}, scrollToBottom: () => {}, scrollDown: () => {}, scrollUp: () => {},
    scrollFraction: () => {}, enterSelect: () => {}, enterFind: () => {}, nextMatch: () => {},
    prevMatch: () => {}, clearSearch: () => {}, readingLayout: () => {},
    historyStep: () => {}, historyTrack: () => {}, historyWalk: () => {},
  });

const stroke = (key: string) => ({ key, ctrl: false, shift: false, alt: false, meta: false });

test("which-key: the head is the pending prefix, the rows its next strokes, case kept", () => {
  const matcher = createMatcher(windowKeys());
  matcher.resolve(stroke(" "));
  const view = pendingKeys(matcher.pendingView()!);
  assert.equal(view.prefix, "space");
  assert.deepEqual(
    view.rows.map((row) => row.key),
    ["space", "h", "l", "u", "r", "a", "k"],
    "space's continuations, in declaration order; no space /",
  );
  assert.equal(view.rows[0]!.label, "open a location");
});

test("help: one column per context, titled, the instrument's first", () => {
  const help = helpGroups([
    { title: "this board", view: createMatcher(boardKeys()).allView() },
    { title: "the window", view: createMatcher(windowKeys()).allView() },
  ]);
  assert.deepEqual(help.groups.map((group) => group.title), ["this board", "the window"]);
  const board = help.groups[0]!.rows;
  const window = help.groups[1]!.rows;
  assert.ok(board.some((row) => row.key === "/" && row.label === "find in this revision"));
  assert.ok(board.some((row) => row.key === "g g"));
  assert.ok(window.some((row) => row.key === "space space" && row.label === "open a location"));
  assert.ok(window.some((row) => row.key === "space r" && row.label === "show or hide the rail"));
  assert.ok(window.some((row) => row.key === "space a" && row.label === "register a project (machine:/path/to/repository)"));
  assert.ok(window.some((row) => row.key === "?"));
  assert.ok(!window.some((row) => row.key === "space /"), "find is / alone");
});

test("help: an action's alternative keys share one row", () => {
  const help = helpGroups([{ title: "this board", view: createMatcher(boardKeys()).allView() }]);
  const half = help.groups[0]!.rows.filter((row) => row.label === "page down half the view");
  assert.deepEqual(half, [{ key: "d, Ctrl-d", label: "page down half the view" }]);
});

test("help: a context with no keys shows no column", () => {
  const empty = { prefix: [], completions: [], entries: [] };
  assert.deepEqual(helpGroups([{ title: "this board", view: empty }]).groups, []);
});

test("spc reads space", () => {
  assert.equal(displayKeys("spc spc"), "space space");
  assert.equal(displayKeys("Ctrl-o"), "Ctrl-o");
});
