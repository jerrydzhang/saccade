// The browser lane's asks install: this lane has no native shell, so
// the app's one asks face (window.__saccadeAsks, read at boot by
// lib/asks.ts) is answered directly, before main.ts evaluates.
//
// - readFile is answered by the lane's declaration reader: a vite
//   middleware that serves the seeded repository's declaration file
//   for any request whose path ends /workbench.toml. That is the same
//   test-only trust the host reader carries - never human
//   functionality; the lane's tests are the only caller.
// - requestLocal is a real fetch of the seeded tracker, which allows
//   this page's origin (tracker-setup.ts).
// - runCommand refuses: the lane runs no commands.
//
// Registration rides the device store seeding: the lane seeds the one
// registered path the test page names (the seeded repository's real
// path), so a scenario registers nothing by itself unless it drives
// the real add field. The test page also gets a per-ask counter
// (window.__saccadeLaneCounts) for the timing scenarios.

import { directAsks } from "../../../src/lib/asks.ts";

type LaneSeed = { machine: string; repoPath: string; base: string };

type LaneWindow = Window & {
  __saccadeAsks?: unknown;
  __saccadeLaneSeed?: LaneSeed;
  __saccadeLaneCounts?: { readFile: number; runCommand: number; requestLocal: number };
};

const seed = (window.parent as LaneWindow).__saccadeLaneSeed;
const counts = { readFile: 0, runCommand: 0, requestLocal: 0 };
(window as LaneWindow).__saccadeLaneCounts = counts;

if (seed !== undefined) {
  // the device store keeps the registered paths alone; the lane's one
  // seed is the seeded repository's real path on this machine
  try {
    localStorage.setItem("saccade.projects", JSON.stringify([`${seed.machine}:${seed.repoPath}`]));
  } catch {
    // storage refused: the scenarios register through the add field
  }
}

(window as LaneWindow).__saccadeAsks = directAsks({
  async readFile() {
    counts.readFile += 1;
    const reply = await fetch("/workbench.toml");
    if (!reply.ok) {
      throw new Error(`the lane's declaration reader answered ${reply.status} for /workbench.toml`);
    }
    return reply.text();
  },
  async runCommand() {
    counts.runCommand += 1;
    throw new Error("the lane runs no commands");
  },
  async requestLocal(_machine, _address, route) {
    counts.requestLocal += 1;
    if (seed === undefined) {
      throw new Error(`the lane has no seeded tracker to answer ${route}`);
    }
    if (!route.startsWith("/")) {
      throw new Error(`the route must start with '/': ${route}`);
    }
    // a real fetch of the seeded tracker, whose loopback address the
    // ask carries and whose origin allowlist names this page
    const reply = await fetch(`${seed.base}${route}`);
    if (!reply.ok) {
      throw new Error(`the tracker answered ${reply.status} for ${route}`);
    }
    return new Uint8Array(await reply.arrayBuffer());
  },
});
