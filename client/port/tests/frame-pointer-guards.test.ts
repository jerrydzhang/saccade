// The frame's pointer guards and the middle button's lifecycle,
// extracted from the board's element-by-element wiring and tested
// over a controlled clock: the right-gesture window drops the
// synthesized post-contextmenu click, the middle press arms and the
// release over the element acts, leaving cancels, and the stray click
// a middle release synthesizes drops within its window.

import assert from "node:assert/strict";
import { test } from "vitest";

import { createPointerGuards, RIGHT_GESTURE_WINDOW_MS, STRAY_CLICK_WINDOW_MS } from "../src/lib/frame/pointer-guards.ts";

function clock(): { now: () => number; advance(ms: number): void } {
  let t = 1000;
  return { now: () => t, advance: (ms: number) => void (t += ms) };
}

test("a plain click within the right gesture's window drops; past it, it acts", () => {
  const t = clock();
  const guards = createPointerGuards(t.now);
  guards.contextmenu();
  t.advance(RIGHT_GESTURE_WINDOW_MS - 1);
  assert.equal(guards.rightGestureArtifact(), true, "the synthesized post-contextmenu click is the artifact");
  t.advance(2);
  assert.equal(guards.rightGestureArtifact(), false, "past the window a click is a user's again");
  assert.equal(guards.rightGestureArtifact(), false, "with no right gesture at all nothing drops");
});

test("a middle press acts on the release over the element; the release acts once", () => {
  const t = clock();
  const guards = createPointerGuards(t.now);
  assert.equal(guards.middleRelease(), false, "a release with no press acts nothing");
  guards.middlePress();
  t.advance(10);
  assert.equal(guards.middleRelease(), true, "the release over the element acts");
  assert.equal(guards.middleRelease(), false, "the release acts once");
});

test("leaving the surface cancels an armed middle press", () => {
  const t = clock();
  const guards = createPointerGuards(t.now);
  guards.middlePress();
  guards.middleLeave();
  assert.equal(guards.middleRelease(), false, "the cancelled press acts nothing");
});

test("the stray click a middle release synthesizes drops within the window", () => {
  const t = clock();
  const guards = createPointerGuards(t.now);
  guards.middlePress();
  guards.middleRelease();
  t.advance(STRAY_CLICK_WINDOW_MS - 1);
  assert.equal(guards.middleStrayClick(), true, "the stray click right after the release drops");
  t.advance(2);
  assert.equal(guards.middleStrayClick(), false, "past the window a click is a user's again");
  assert.equal(guards.middleAuxClick(1), true, "the middle's auxclick preventDefaults");
  assert.equal(guards.middleAuxClick(0), false, "a left auxclick is not the frame's to prevent");
});
