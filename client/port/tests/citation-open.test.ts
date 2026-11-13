// The citation open composition (the S08 live finding): a citation
// activation carries the CURRENT board's machine and the cited task
// path; the open must take the parseable location form. A colon join
// of a tracker machine ("tracker:seedtracker" + ":" + "t-0@r1")
// parses as nothing and the open is refused "not a location" - the
// native lane caught the citation click doing exactly that. These
// tests pin the round-trip the citation handoff relies on:
// formatLocation produces a spec parseOpenSpec accepts, and the
// composed address keeps the cited task's pinned revision.

import assert from "node:assert/strict";
import { test } from "vitest";
import { formatLocation, parseLocation } from "../src/lib/location.ts";
import { parseOpenSpec } from "../src/instruments/boardController.ts";

test("a tracker machine's location form parses back to the same address", () => {
  const machine = "tracker:seedtracker";
  const path = "t-0@r1";
  const spec = formatLocation({ machine, path });
  // the tracker form is name/spec - never a colon join
  assert.equal(spec, "seedtracker/t-0@r1");
  assert.deepEqual(parseLocation(spec), { machine, path });
  assert.deepEqual(parseOpenSpec(spec), { machine, path });
});

test("a plain machine's location form still colon-joins", () => {
  const spec = formatLocation({ machine: "host", path: "/board/board.yaml" });
  assert.equal(spec, "host:/board/board.yaml");
  assert.deepEqual(parseOpenSpec(spec), { machine: "host", path: "/board/board.yaml" });
});

test("the colon join of a tracker machine is not a location", () => {
  // the exact broken composition the citation handoff used: it must
  // stay refusable so a regression to hand-rolled joins cannot pass
  // as a silent wrong open
  assert.equal(parseOpenSpec("tracker:seedtracker:t-0@r1"), null);
});

// The citation span's attributes (the S08 anchor finding): the select
// path rides the span so hydration's link opens the citation anchored
// at the named part - display text alone loses the anchor.
import { citationAttributes } from "../src/lib/render.ts";

test("a citation with a select carries it as a data attribute", () => {
  assert.deepEqual(
    citationAttributes({ task: "t-0", revision: 1, select: ["/content/0/group/content/0"] }),
    {
      "data-cite-task": "t-0",
      "data-cite-revision": "1",
      "data-cite-select": "/content/0/group/content/0",
    },
  );
});

test("a citation without a select carries no select attribute", () => {
  assert.deepEqual(citationAttributes({ task: "t-2", revision: 3, select: null }), {
    "data-cite-task": "t-2",
    "data-cite-revision": "3",
  });
});
