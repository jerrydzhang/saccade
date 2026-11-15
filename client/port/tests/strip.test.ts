// The strip's load status (t-59): shown only when the view is not
// simply loaded - loading, or a failure - never as permanent chrome.

import assert from "node:assert/strict";
import { test } from "vitest";

import { stripStatus } from "../src/lib/strip.ts";

test("a loaded or empty window says nothing about its load", () => {
  assert.equal(stripStatus("loaded"), null);
  assert.equal(stripStatus("empty"), null);
});

test("loading shows, and is not a failure", () => {
  assert.deepEqual(stripStatus("loading"), { text: "loading", failed: false });
});

test("every other status is a failure the strip names", () => {
  for (const status of ["unavailable", "invalid", "destroyed"]) {
    assert.deepEqual(stripStatus(status), { text: status, failed: true });
  }
});
