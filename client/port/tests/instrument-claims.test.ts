// The claims list (t-73): one place listing which locations each
// instrument shows. Tested on fake addresses per claim - the plane's
// addresses are the task list's, every other address the board's; the
// first claim that names an address wins; none is null.

import assert from "node:assert/strict";
import { test } from "vitest";

import { instrumentClaims, instrumentForAddress, type InstrumentClaim } from "../src/lib/instruments/claims.ts";
import type { Address } from "../src/lib/contracts/places.ts";

test("the plane's addresses claim the task list", () => {
  assert.equal(instrumentForAddress({ machine: "plane", path: "" }), "tasks");
  assert.equal(instrumentForAddress({ machine: "plane", path: "pyrimel" }), "tasks");
});

test("every other address claims the board", () => {
  assert.equal(instrumentForAddress({ machine: "somehost:repo", path: "t-1" }), "board");
  assert.equal(instrumentForAddress({ machine: "somehost:repo", path: "t-1@r3" }), "board");
});

test("a plane-shaped machine with a path-leading slash is not the plane (isPlane's own rule)", () => {
  assert.equal(instrumentForAddress({ machine: "plane", path: "/absolute" }), "board");
});

test("the first claim that names the address wins", () => {
  const claims: readonly InstrumentClaim[] = [
    { instrument: "first", claims: () => true },
    { instrument: "second", claims: () => true },
  ];
  const address: Address = { machine: "plane", path: "" };
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
