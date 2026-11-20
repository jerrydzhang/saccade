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
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { defineConfig, mergeConfig } from "vitest/config";
import type { Plugin } from "vite";
import { playwright } from "@vitest/browser-playwright";
import viteConfig from "./vite.config.ts";
import { freePort } from "./tests/scenarios/kit/tracker.ts";
import { portDrag, portSetViews, portWheel } from "./tests/scenarios/kit/pointer-commands.ts";

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

// pending: a scenario written RED, before the feature it gates. Skipped
// in the ordinary runs (the gates stay green meanwhile); `just
// app-test-pending` runs exactly these. The tag comes off when the
// feature lands and the test passes.
const pending = {
  name: "pending",
  description: "gates an unbuilt feature: red until it lands (just app-test-pending)",
  skip: !process.env.SAC_PORT_SUITE_PENDING,
};

// WebKit's display-less EGL: the saccade-app devshell's pinned mesa
// vendor dir with forced software rendering (the browser lane's own
// recipe, kept here when that lane retired)
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

// The lane's declaration reader: the browser lane's readFile ask is
// answered here - any request whose path ends /workbench.toml is
// served the seeded repository's declaration file. That is the same
// test-only trust the host reader carries, never human functionality;
// the lane's pages are the only callers (kit/lane-asks.ts). The
// seeded repository is named when the globalSetup starts the tracker
// (kit/tracker-setup.ts, SAC_PORT_SUITE_REPO); the middleware reads
// it lazily, since the config loads before the setup runs.
const laneDeclarationReader = (): Plugin => ({
  name: "lane-declaration-reader",
  configureServer(server) {
    server.middlewares.use((request, response, next) => {
      const path = (request.url ?? "").split("?")[0]!;
      if (!path.endsWith("/workbench.toml")) return next();
      const repo = process.env.SAC_PORT_SUITE_REPO ?? "";
      if (repo.length === 0) {
        response.statusCode = 503;
        response.end("the lane has no seeded repository");
        return;
      }
      try {
        const body = readFileSync(join(repo, "workbench.toml"));
        response.setHeader("Content-Type", "text/plain; charset=utf-8");
        response.end(body);
      } catch {
        response.statusCode = 404;
        response.end("the seeded repository carries no declaration file");
      }
    });
  },
});

export default mergeConfig(
  viteConfig,
  defineConfig({
    // the loopback literal, never the name: the session lane's sandbox
    // has no /etc/hosts, so "localhost" does not resolve there
    server: { host: "127.0.0.1" },
    plugins: [laneDeclarationReader()],
    // the lane's suite mode binds the client tree read-only and names a
    // cache under its run root (app/harness/suite_payload.py)
    ...(process.env.SUITE_CACHE_DIR ? { cacheDir: process.env.SUITE_CACHE_DIR } : {}),
    test: {
      projects: [
        {
          extends: true,
          test: {
            name: "unit",
            environment: "node",
            include: ["tests/*.test.ts"],
            restoreMocks: true,
            tags: [pending],
          },
        },
        {
          extends: true,
          resolve: { alias: { "#scenario": here("./tests/scenarios/kit/browser.ts") } },
          // pre-bundled up front: a mid-run optimize reloads the page
          optimizeDeps: {
            include: [
              "yaml",
              "remark-gfm",
              "remark-math",
              "remark-parse",
              "unified",
              "mermaid",
              "mathjax/tex-svg.js",
              // src/lib/highlight.ts's core and its language set
              "highlight.js/lib/core",
              ...["bash", "css", "diff", "ini", "javascript", "json", "markdown", "nix", "python", "rust", "shell", "sql", "typescript", "xml", "yaml"].map(
                (language) => `highlight.js/lib/languages/${language}`,
              ),
            ],
          },
          test: {
            name: "browser",
            // *.browser.ts: browser-only scenarios (the mocked shell)
            include: ["tests/scenarios/*.scenario.ts", "tests/scenarios/*.browser.ts"],
            tags: [tags[0]!, { ...tags[1]!, skip: true }, pending],
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
              // real drag and wheel input (kit/pointer-commands.ts)
              commands: { portDrag, portSetViews, portWheel },
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
            tags: [...tags, pending],
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
