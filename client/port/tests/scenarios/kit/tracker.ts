// The scenarios' throwaway tracker (node side only): one real
// `sac serve` on a fresh temp db and repo, on a free loopback port,
// seeded with the t36 seed set by client/harness/t36_seeds.py itself
// (the seed boards stay single-sourced there until the Python harness
// cuts over). Never the production tracker: the db is always a fresh
// file under a new temp root, and the server is spawned and stopped
// by its own handle. The run root (db, repo, serve log) is kept.
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { mkdirSync, mkdtempSync, openSync, accessSync, constants } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { SeededTracker } from "./driver.ts";

const HARNESS = fileURLToPath(new URL("../../../../harness", import.meta.url));
const REPO_NAME = "seedtracker";

export type RunningTracker = { tracker: SeededTracker; root: string; stop: () => Promise<void> };

export async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      const port = typeof address === "object" && address !== null ? address.port : 0;
      probe.close(() => resolve(port));
    });
  });
}

function sacBin(): string {
  const explicit = process.env.SACCADE_SERVE_BIN ?? "";
  const found = explicit.length > 0
    ? explicit
    : (spawnSync("sh", ["-c", "command -v sac"], { encoding: "utf8" }).stdout ?? "").trim();
  if (!isAbsolute(found)) {
    throw new Error("the scenarios need the real sac binary: set SACCADE_SERVE_BIN to an absolute path (just port-test-browser builds it)");
  }
  accessSync(found, constants.X_OK);
  return found;
}

async function waitUp(base: string, child: ChildProcess, log: string): Promise<void> {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`sac serve exited during startup (log: ${log})`);
    try {
      await fetch(`${base}/api/v1/version`);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
  throw new Error(`sac serve never answered within 60s (log: ${log})`);
}

/** Start the throwaway tracker and seed it. `origins` are the page
 * origins allowed to read it (the vitest browser server's). */
export async function startTracker(origins: readonly string[]): Promise<RunningTracker> {
  const root = mkdtempSync(join(tmpdir(), "saccade-port-suite-"));
  const repo = join(root, REPO_NAME);
  mkdirSync(repo);
  mkdirSync(join(root, "logs"));
  const log = join(root, "logs", "seed-serve.log");
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  const out = openSync(log, "a");
  const args = ["serve", "--db", join(root, "seed.db"), "--repo", repo, "--bind", "127.0.0.1", "--port", String(port)];
  for (const origin of origins) args.push("--allow-origin", origin);
  const child = spawn(sacBin(), args, { stdio: ["ignore", out, out] });
  const stop = async (): Promise<void> => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const exited = new Promise((resolve) => child.once("exit", resolve));
    child.kill("SIGTERM");
    const timer = setTimeout(() => child.kill("SIGKILL"), 5_000);
    await exited;
    clearTimeout(timer);
  };
  try {
    await waitUp(base, child, log);
    const tasks = seed(base);
    console.log(`[port-suite] seeded tracker ${base} (run root ${root})`);
    return { tracker: { base, host: `127.0.0.1:${port}`, repo: REPO_NAME, tasks }, root, stop };
  } catch (error) {
    await stop();
    throw error;
  }
}

// the seed set, committed by the harness's own seed_via over the wire
const SEED_SCRIPT = `
import json, sys, types
sys.path.insert(0, sys.argv[1])
import t36_seeds
server = types.SimpleNamespace(base=sys.argv[2])
commits = t36_seeds.seed_via(lambda route, payload: t36_seeds.SeedServer.post(server, route, payload))
print(json.dumps(commits))
`;

function seed(base: string): SeededTracker["tasks"] {
  const run = spawnSync("python3", ["-I", "-c", SEED_SCRIPT, HARNESS, base], { encoding: "utf8" });
  if (run.status !== 0) throw new Error(`seeding failed: ${run.stderr}`);
  const commits = JSON.parse(run.stdout) as { task: string; message: string }[];
  const by = (needle: string): string => {
    const hit = commits.find((commit) => commit.message.includes(needle));
    if (hit === undefined) throw new Error(`no seed commit names ${JSON.stringify(needle)}`);
    return hit.task;
  };
  return {
    cited: by("group-made"),
    citing: by("incorporation into a group child"),
    citingChild: by("the other group child"),
    landing: by("landing board, text only"),
  };
}
