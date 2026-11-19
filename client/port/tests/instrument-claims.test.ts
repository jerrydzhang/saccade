// The claims list: one place listing which locations each
// instrument shows. Tested on fake addresses per claim - a project's
// task list is the task list's, every other address the board's; the
// first claim that names an address wins; none is null.

import assert from "node:assert/strict";
import { test } from "vitest";

import { instrumentClaims, instrumentForAddress, type InstrumentClaim } from "../src/lib/instruments/claims.ts";
import type { Address } from "../src/lib/contracts/places.ts";

test("a project's task list claims the task list", () => {
  assert.equal(instrumentForAddress({ machine: "tracker:saccade", path: "tasks" }), "tasks");
  assert.equal(instrumentForAddress({ machine: "tracker:pyrimel", path: "tasks" }), "tasks");
});

test("every other address claims the board", () => {
  assert.equal(instrumentForAddress({ machine: "somehost:repo", path: "t-1" }), "board");
  assert.equal(instrumentForAddress({ machine: "somehost:repo", path: "t-1@r3" }), "board");
  assert.equal(instrumentForAddress({ machine: "tracker:saccade", path: "t-1" }), "board", "a task board is the board's");
  assert.equal(instrumentForAddress({ machine: "plane", path: "/absolute" }), "board", "a machine named plane keeps its paths");
});

test("the first claim that names the address wins", () => {
  const claims: readonly InstrumentClaim[] = [
    { instrument: "first", claims: () => true },
    { instrument: "second", claims: () => true },
  ];
  const address: Address = { machine: "tracker:saccade", path: "tasks" };
  assert.equal(instrumentForAddress(address, claims), "first");
});

test("an address no claim names is null", () => {
  const claims: readonly InstrumentClaim[] = [{ instrument: "narrow", claims: () => false }];
  assert.equal(instrumentForAddress({ machine: "plane", path: "" }, claims), null);
});

test("the shipped list names both instruments in their order", () => {
  assert.deepEqual(
    instrumentClaims.map((claim) => claim.instrument),
    ["tasks", "board"],
  );
});
