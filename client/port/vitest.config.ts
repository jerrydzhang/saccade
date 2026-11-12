// The port's one test runner: vitest, three projects, one scenario
// shape (tests/scenarios/README.md).
//
// - unit: the pure and component tests in tests/*.test.ts, in node.
// - browser: tests/scenarios/*.scenario.ts in WebKit (vitest browser
//   mode, Playwright provider) against the port served by this vite
//   config, with a throwaway seeded tracker from globalSetup.
// - native: the same scenario files through tauri-driver and
//   WebdriverIO against the built port binary, inside the herdr-ui
//   sandbox; only scenarios tagged native or multi-window run there
//   (the recipe passes the tag filter).
import { fileURLToPath } from "node:url";
import { defineConfig, mergeConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";
import viteConfig from "./vite.config.ts";
import { freePort, setupTrackerWrites } from "./tests/scenarios/kit/tracker.ts";

const here = (path: string): string => fileURLToPath(new URL(path, import.meta.url));

// the vitest browser server's port, pinned per run so the throwaway
// tracker can allow exactly its origin (tracker-setup.ts reads it);
// the config may load more than once per run, so the first pick holds
process.env.SAC_PORT_SUITE_BROWSER_PORT ??= String(await freePort());
const browserPort = Number(process.env.SAC_PORT_SUITE_BROWSER_PORT);
process.env.SAC_PORT_SUITE_ORIGINS = [
  `http://localhost:${browserPort}`,
  `http://127.0.0.1:${browserPort}`,
].join(" ");

// the scenario tags: a scenario that needs real native windows says
// so, and the browser project skips it by its tag
const tags = [
  { name: "native", description: "also runs on the native project (the built app through tauri-driver)" },
  { name: "multi-window", description: "needs real native windows: native project only" },
];

// WebKit's display-less EGL: the saccade-app devshell's pinned mesa
// vendor dir with forced software rendering (the browser lane's own
// recipe, client/harness/entry.py and browser_lane.py)
function webkitEnv(): Record<string, string> {
  const mesa = process.env.SAC_APP_MESA_EGL_VENDOR_DIR ?? "";
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined) env[key] = value;
  }
  if (mesa.length > 0) {
    env.__EGL_VENDOR_LIBRARY_DIRS = mesa;
    env.LIBGL_ALWAYS_SOFTWARE = "1";
  }
  env.EGL_PLATFORM = "surfaceless";
  env.WEBKIT_FORCE_COMPLEX_TEXT = "1";
  return env;
}

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      projects: [
        {
          extends: true,
          test: {
            name: "unit",
            environment: "node",
            include: ["tests/*.test.ts"],
            restoreMocks: true,
          },
        },
        {
          extends: true,
          resolve: { alias: { "#scenario": here("./tests/scenarios/kit/browser.ts") } },
          // pre-bundled up front: a mid-run optimize reloads the page
          optimizeDeps: {
            include: ["yaml", "remark-gfm", "remark-math", "remark-parse", "unified", "mermaid", "mathjax/tex-svg.js"],
          },
          test: {
            name: "browser",
            include: ["tests/scenarios/*.scenario.ts"],
            tags: [tags[0]!, { ...tags[1]!, skip: true }],
            globalSetup: ["tests/scenarios/kit/tracker-setup.ts"],
            testTimeout: 30_000,
            api: { port: browserPort, strictPort: true },
            browser: {
              enabled: true,
              headless: true,
              provider: playwright({ launchOptions: { env: webkitEnv() } }),
              instances: [{ browser: "webkit" }],
              viewport: { width: 1100, height: 800 },
              // the port's controls carry data-port-control ids
              locators: { testIdAttribute: "data-port-control" },
              commands: { trackerCommit, trackerCreateTask },
            },
          },
        },
        {
          extends: true,
          resolve: { alias: { "#scenario": here("./tests/scenarios/kit/native.ts") } },
          test: {
            name: "native",
            environment: "node",
            include: ["tests/scenarios/*.scenario.ts"],
            tags,
            globalSetup: ["tests/scenarios/kit/tracker-setup.ts"],
            // one native app at a time: each test owns a fresh app
            fileParallelism: false,
            testTimeout: 120_000,
            hookTimeout: 120_000,
          },
        },
      ],
    },
  }),
);

// the browser's write door into the throwaway tracker: browser-mode
// code cannot reach node, so a mid-scenario commit rides a vitest
// browser command (it runs in the vitest process, where the tracker
// setup lives)
async function trackerCommit(
  _context: unknown,
  task: string,
  board: string,
  message: string,
): Promise<{ revision: number }> {
  return setupTrackerWrites().commit(task, board, message);
}

async function trackerCreateTask(_context: unknown, name: string): Promise<string> {
  return setupTrackerWrites().createTask(name);
}
