// The instrument scaffold's load-scope state machine (order of work
// step 1): the generation/release bookkeeping both controllers
// copied. Direct tests for the pure state; the controllers' rewiring
// rides their existing suites and the focused probes.

import assert from "node:assert/strict";
import { test } from "vitest";

import { createLoadScope, issueOf, readyFrame } from "../src/lib/instruments/scaffold.ts";
import { SourceUnavailableError } from "../src/lib/sources.ts";

test("a token is current until a newer begin supersedes it", () => {
  const scope = createLoadScope<{ n: number }>({ n: 0 });
  const first = scope.begin();
  assert.equal(scope.isStale(first), false);
  const second = scope.begin();
  assert.equal(scope.isStale(first), true, "the older token is stale");
  assert.equal(scope.isStale(second), false, "the newest token is current");
});

test("destroy makes every token stale, current and future", () => {
  const scope = createLoadScope<{ n: number }>({ n: 0 });
  const token = scope.begin();
  scope.destroy();
  assert.equal(scope.isStale(token), true, "the current token is stale after destroy");
  const later = scope.begin();
  assert.equal(scope.isStale(later), true, "even a fresh begin is stale once destroyed");
  assert.equal(scope.isDestroyed(), true);
});

test("frames adopt over the empty; the asked spec round-trips", () => {
  const scope = createLoadScope<{ n: number }>({ n: 0 });
  assert.deepEqual(scope.frame(), { n: 0 }, "the empty holds until adoption");
  scope.adopt({ n: 7 });
  assert.deepEqual(scope.frame(), { n: 7 });
  scope.setAsked("host:1/tracker/t-1");
  assert.equal(scope.asked(), "host:1/tracker/t-1");
  // adoption is plain replacement, never merge
  scope.adopt({ n: 9 });
  assert.deepEqual(scope.frame(), { n: 9 });
});

test("readyFrame wraps domain data with the empty target list", () => {
  const prepared = readyFrame({ value: 1 });
  assert.equal(prepared.kind, "ready");
  if (prepared.kind === "ready") {
    assert.deepEqual(prepared.value.value, { value: 1 });
    assert.deepEqual(prepared.value.targets, []);
  }
});

test("issueOf carries a source's unavailable verdict verbatim", () => {
  const error = new SourceUnavailableError("asset-denied", "no such file");
  const issue = issueOf(error);
  assert.equal(issue.code, "asset-denied");
  assert.equal(issue.message, "no such file");
});

test("issueOf classifies foreign failures as generic reads", () => {
  const issue = issueOf(new Error("socket hiccup"));
  assert.equal(issue.code, "read-failed");
  assert.equal(issue.message, "Error: socket hiccup");
});
