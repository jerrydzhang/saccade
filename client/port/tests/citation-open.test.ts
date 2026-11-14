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

// The control plane's address (t-54): "plane" names no tracker; a
// tracker's pipeline is no location of its own, with no alias; a
// machine named plane is still reached as plane:/path.
import { isPlane, projectAddress } from "../src/lib/location.ts";

test("plane is the control plane's location and names no tracker", () => {
  assert.deepEqual(parseLocation("plane"), { machine: "plane", path: "" });
  assert.equal(formatLocation({ machine: "plane", path: "" }), "plane");
  // the project it shows rides the address, never the location string
  assert.equal(formatLocation({ machine: "plane", path: "saccade" }), "plane");
  assert.ok(isPlane({ machine: "plane", path: "saccade" }));
  assert.ok(!isPlane({ machine: "plane", path: "/etc/board.yaml" }), "a machine named plane keeps its paths");
  assert.deepEqual(parseLocation("plane:/x/board.yaml"), { machine: "plane", path: "/x/board.yaml" });
});

test("a tracker's tasks page is no location, bare or with its host", () => {
  assert.equal(parseLocation("saccade/tasks"), null);
  assert.equal(parseOpenSpec("localhost:8811/saccade/tasks"), null);
  // board locations keep both forms
  assert.deepEqual(parseLocation("localhost:8813/pyrimel/t-3"), { machine: "tracker:pyrimel", path: "t-3" });
});

test("the plane's add field takes host[:port]/name; the port defaults to sac serve's", () => {
  assert.deepEqual(projectAddress("localhost:8813/pyrimel"), { host: "localhost:8813", tracker: "pyrimel" });
  assert.deepEqual(projectAddress(" t81/saccade/ "), { host: "t81:8811", tracker: "saccade" });
  assert.deepEqual(projectAddress("[::1]:8812/repo"), { host: "[::1]:8812", tracker: "repo" });
  assert.equal(projectAddress("saccade"), null);
  assert.equal(projectAddress("localhost:8813/pyrimel/t-3"), null, "a board location is the picker's");
  assert.equal(projectAddress(""), null);
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
