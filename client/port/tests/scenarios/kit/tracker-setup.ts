// The scenarios' globalSetup: one throwaway seeded tracker per
// project run (tracker.ts), provided to every test as `tracker`, and
// stopped by its own handle when the run ends.
import type { TestProject } from "vitest/node";

import { startTracker } from "./tracker.ts";

export default async function setup(project: TestProject): Promise<() => Promise<void>> {
  if (project.name === "native") {
    const { pinned } = await import("./native.ts");
    pinned();
  }
  const origins = (process.env.SAC_PORT_SUITE_ORIGINS ?? "").split(" ").filter((origin) => origin.length > 0);
  const running = await startTracker(origins);
  project.provide("tracker", running.tracker);
  return running.stop;
}
