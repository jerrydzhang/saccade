// The deck (t-14, t-3, t-24@r21): the document follows, the cue waits
// unless the human is on the latest slide, h l step, digits jump, and
// the copied command is the exact cue command the agent runs.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  applyDocument,
  cueCommand,
  deckNameFromBytes,
  isValidDeckName,
  jumpDeck,
  newDeck,
  stepDeck,
  waitingSlide,
} from "../src/lib/deck.ts";

const NAME = "deck-" + "ab".repeat(12);

test("the address is deck- plus 96 bits in the core's name alphabet", () => {
  const random = new Uint8Array(12);
  for (let i = 0; i < 12; i++) random[i] = 0xab;
  const name = deckNameFromBytes(random);
  assert.equal(name, NAME);
  assert.ok(isValidDeckName(name));
  assert.ok(name.length <= 32, "inside the core's 32-character name rule");
  assert.ok(!isValidDeckName("deck-" + "AB".repeat(12)), "uppercase is not the deck's form");
  assert.ok(!isValidDeckName("deck-" + "ab".repeat(11)), "shorter is not the deck's form");
});

test("a landing appends the new slides and follows the list", () => {
  const deck = newDeck(NAME);
  const first = applyDocument(deck, false, ["tracker/x/t-1"], 1);
  assert.deepEqual(first.arrived, [0]);
  assert.deepEqual(first.state.slides, ["tracker/x/t-1"]);
  const second = applyDocument(first.state, false, ["tracker/x/t-1", "/tmp/a.yaml"], 2);
  assert.deepEqual(second.arrived, [1]);
  assert.deepEqual(second.state.slides, ["tracker/x/t-1", "/tmp/a.yaml"]);
});

test("a cue waits unless the human is on the latest slide", () => {
  const deck = newDeck(NAME);
  const one = applyDocument(deck, false, ["tracker/x/t-1"], 1);
  assert.equal(one.show, true, "the deck's first cue lands on whatever is shown");
  assert.equal(one.state.position, 0);
  assert.equal(one.state.entered, false, "showing a cue is not the human entering");
  // the human is showing slide 1 (the latest): a new cue shows
  const two = applyDocument(one.state, true, ["tracker/x/t-1", "tracker/x/t-2"], 2);
  assert.equal(two.show, true);
  // the human followed a ref away: the view is not the slide
  const three = applyDocument(two.state, false, [
    "tracker/x/t-1",
    "tracker/x/t-2",
    "tracker/x/t-3",
  ], 3);
  assert.equal(three.show, false, "a cue waits when the window shows something else");
  assert.deepEqual(three.arrived, [2]);
  assert.equal(three.state.position, 1, "the pointer stays where the deck stood");
  // and once waiting has begun, a human who never engaged is not
  // pulled away either: slide 1 auto-shown, the human opened their
  // own file, the next cue waits
  const opened = { ...one.state, position: 0 };
  const waited = applyDocument(opened, false, ["tracker/x/t-1", "tracker/x/t-2"], 2);
  assert.equal(waited.show, false);
});

test("a wait never pulls the human's reading away", () => {
  const deck = newDeck(NAME);
  const one = applyDocument(deck, true, ["tracker/x/t-1"], 1);
  const shown = { ...one.state, entered: true, position: 0 };
  const two = applyDocument(shown, true, ["tracker/x/t-1", "tracker/x/t-2"], 2);
  assert.equal(two.show, true, "on the latest slide, the cue shows");
  assert.equal(two.state.position, 1);
  // stepped back to slide 1, showing it; a cue waits
  const stepped = { ...two.state, position: 0 };
  const three = applyDocument(stepped, true, [
    "tracker/x/t-1",
    "tracker/x/t-2",
    "tracker/x/t-3",
  ], 3);
  assert.equal(three.show, false);
});

test("a deleted document is the address dying", () => {
  const deck = newDeck(NAME);
  const one = applyDocument(deck, false, ["tracker/x/t-1"], 1);
  const dead = applyDocument(one.state, false, null, 2);
  assert.equal(dead.dead, true);
  assert.equal(dead.show, false);
});

test("a landing that is not a location list is ignored", () => {
  const deck = newDeck(NAME);
  for (const value of ["nope", { slides: [] }, 3]) {
    const event = applyDocument(deck, false, value, 1);
    assert.equal(event.dead, false);
    assert.deepEqual(event.arrived, []);
    assert.deepEqual(event.state.slides, []);
  }
});

test("h steps back and l steps forward, stopping at the ends", () => {
  const deck = { ...newDeck(NAME), slides: ["a", "b", "c"], position: 1, entered: true };
  const back = stepDeck(deck, "older");
  assert.equal(back.target, 0);
  const backEnd = stepDeck({ ...back.state, position: 0 }, "older");
  assert.equal(backEnd.target, 0, "the start holds");
  const forward = stepDeck(deck, "newer");
  assert.equal(forward.target, 2);
  const forwardEnd = stepDeck({ ...forward.state, position: 2 }, "newer");
  assert.equal(forwardEnd.target, 2, "the latest holds");
});

test("l from outside the deck enters at the latest; h does nothing", () => {
  const outside = { ...newDeck(NAME), slides: ["a", "b"], position: null, entered: false };
  const forward = stepDeck(outside, "newer");
  assert.equal(forward.target, 1, "next means newest");
  assert.equal(forward.state.entered, true);
  const back = stepDeck(outside, "older");
  assert.equal(back.target, null);
  const empty = stepDeck(newDeck(NAME), "newer");
  assert.equal(empty.target, null, "an empty deck steps nowhere");
});

test("digits jump to slide numbers, past-the-end does nothing", () => {
  const deck = { ...newDeck(NAME), slides: ["a", "b"], position: null, entered: false };
  const two = jumpDeck(deck, 2);
  assert.equal(two.target, 1);
  assert.equal(two.state.entered, true);
  const nine = jumpDeck(deck, 9);
  assert.equal(nine.target, null, "beyond the last slide is no jump");
  assert.equal(jumpDeck(deck, 0).target, null);
});

test("the waiting slide is the one the next forward step lands on", () => {
  const deck = { ...newDeck(NAME), slides: ["a", "b", "c"], position: 2, entered: true };
  assert.equal(waitingSlide(deck, true), null, "on the latest, nothing waits");
  // followed a ref away: the pointer still names the latest
  assert.equal(waitingSlide(deck, false), null);
  const behind = { ...deck, position: 0 };
  assert.equal(waitingSlide(behind, true), 2, "slide 3 waits (1-based)");
  const away = { ...deck, position: 0 };
  assert.equal(waitingSlide(away, false), 2);
  const away2 = { ...newDeck(NAME), slides: ["a", "b"], position: 0, entered: false };
  assert.equal(waitingSlide(away2, false), 2, "a cue can wait before the human ever engages");
});

test("the copied string is the exact cue command the agent runs", () => {
  const command = cueCommand(NAME, "tracker/saccade/t-14@r3");
  assert.match(command, new RegExp(`name = "${NAME}"`));
  assert.match(command, /doc\.get/);
  assert.match(command, /doc\.set/);
  assert.match(command, /version.*doc\["version"\]/);
  assert.match(command, /slides\.append\(sys\.argv\[1\]\)/);
  assert.match(command, /hc", "connect"/);
  assert.match(command, /<<'DECK'/, "a quoted heredoc: nothing expands in the shell");
  // the location rides as the argument, not interpolated into the body
  assert.match(command, /^python3 - 'tracker\/saccade\/t-14@r3' <<'DECK'/);
});
