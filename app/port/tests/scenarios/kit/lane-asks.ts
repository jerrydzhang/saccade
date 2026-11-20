// The browser lane's asks install: this lane has no native shell, so
// the app's one asks face (window.__saccadeAsks, read at boot by
// lib/asks.ts) is answered directly, before main.ts evaluates.
//
// - readFile is answered by the lane's declaration reader: a vite
//   middleware that serves the seeded repository's declaration file
//   for any request whose path ends /workbench.toml. That is the same
//   test-only trust the host reader carries - never human
//   functionality; the lane's tests are the only caller.
// - readAsset refuses: the lane serves no machine assets (its boards
//   ride the fixture source, whose bytes never leave the page).
// - requestLocal is a real fetch of the seeded tracker, which allows
//   this page's origin (tracker-setup.ts).
//
// Registration rides the device store seeding: the lane seeds the one
// registered path the test page names (the seeded repository's real
// path), so a scenario registers nothing by itself unless it drives
// the real add field. The test page also gets a per-ask counter
// (window.__saccadeLaneCounts) for the timing scenarios. A seed may
// carry a latency: every tracker request then answers that many
// milliseconds late, as a remote machine's would.

import { directAsks } from "../../../src/lib/asks.ts";
import { RequestRefused } from "../../../src/lib/board/errors.ts";

type LaneSeed = { machine: string; repoPath: string; base: string; latencyMs?: number };

type LaneWindow = Window & {
  __saccadeAsks?: unknown;
  __saccadeLaneSeed?: LaneSeed;
  __saccadeLaneCounts?: { readFile: number; readAsset: number; requestLocal: number };
};

const seed = (window.parent as LaneWindow).__saccadeLaneSeed;
const counts = { readFile: 0, readAsset: 0, requestLocal: 0 };
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
  async readAsset() {
    counts.readAsset += 1;
    throw new Error("the lane serves no machine assets");
  },
  async requestLocal(_machine, port, route) {
    counts.requestLocal += 1;
    if (seed === undefined) {
      throw new Error(`the lane has no seeded tracker to answer ${route}`);
    }
    if (!route.startsWith("/")) {
      throw new Error(`the route must start with '/': ${route}`);
    }
    const latency = seed.latencyMs ?? 0;
    if (latency > 0) await new Promise((resolve) => setTimeout(resolve, latency));
    // a real fetch of the seeded tracker at the loopback port the ask
    // carries - the same address the shell composes on a machine; the
    // tracker's origin allowlist names this page
    const reply = await fetch(`http://127.0.0.1:${port}${route}`);
    if (!reply.ok) {
      const message = `the tracker answered ${reply.status} for ${route}`;
      const text = await reply.text();
      let refusal = false;
      try {
        refusal = typeof (JSON.parse(text) as { code?: unknown }).code === "string";
      } catch {
        // a non-JSON body is no refusal document
      }
      // the shell hands a refusal document through with the failure;
      // the lane does the same
      throw refusal ? new RequestRefused("machine-unavailable", message, text) : new Error(message);
    }
    return new Uint8Array(await reply.arrayBuffer());
  },
});
