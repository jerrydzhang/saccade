// The two acts' pure contracts (t-14 "The suite"): the instrument
// routing's declaration and the citation's full spec. The routing
// itself rides the shell; what is here is the declaration's shape and
// the spec naming.

import test from "node:test";
import assert from "node:assert/strict";
import { citationSpec, VIEWER_ROUTING } from "../src/lib/acts.ts";

test("the viewer's declaration is Enter follows here; t-18's instruments declare link", () => {
  assert.equal(VIEWER_ROUTING.primary, "here");
});

test("a citation names its full tracker spec, relative to the window's own tracker", () => {
  assert.equal(
    citationSpec({ machine: "tracker:smoke", path: "t-5" }, "t-1@r2"),
    "smoke/t-1@r2",
  );
  assert.equal(
    citationSpec({ machine: "tracker:saccade", path: "t-19" }, "t-19@r3"),
    "saccade/t-19@r3",
  );
  assert.equal(
    citationSpec({ machine: "b", path: "/board/x.yaml" }, "t-1"),
    null,
    "a citation rides a tracker location; the machine form has no tracker to name",
  );
});
