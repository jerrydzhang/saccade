import { cpSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// The port tree's vite root is app/port: the config anchors every
// path to its own directory, so the build behaves the same from any
// working directory. The npm tree itself stays at the client root
// (reused, never duplicated); the config resolves it from there.
const root = fileURLToPath(new URL(".", import.meta.url));
const clientRoot = fileURLToPath(new URL("..", import.meta.url));

// MathJax 4 loads two kinds of files at run time, each copied from the
// pinned packages into the build so it loads offline from the same
// origin - never an external fetch (the app's CSP allows script-src
// 'self' only):
// - the speech worker (worker + mathmaps), resolved against
//   loader.paths.mathjax: the package's sre directory at <origin>/sre;
// - the font's dynamic ranges (double-struck, calligraphic, script,
//   fraktur, ...), resolved against loader.paths.fonts: the newcm font's
//   svg/dynamic directory at <origin>/fonts (src/lib/mathjax.ts).
// The dev server (the browser lane) serves the same files from
// node_modules: the promise conversion waits for the font ranges and for
// the speech worker, so without them no expression would ever finish.
const MATHJAX_ASSETS: readonly { served: string; source: string }[] = [
  { served: "sre", source: `${clientRoot}node_modules/mathjax/sre` },
  {
    served: "fonts/mathjax-newcm-font/svg/dynamic",
    source: `${clientRoot}node_modules/@mathjax/mathjax-newcm-font/svg/dynamic`,
  },
];

const CONTENT_TYPES: Record<string, string> = { js: "text/javascript", json: "application/json" };

const mathjaxOfflineAssets = (): Plugin => ({
  name: "mathjax-offline-assets",
  configureServer(server) {
    for (const { served, source } of MATHJAX_ASSETS) {
      server.middlewares.use(`/${served}/`, (request, response, next) => {
        const name = (request.url ?? "").split("?")[0]!.replace(/^\//, "");
        const type = CONTENT_TYPES[name.split(".").pop() ?? ""];
        if (type === undefined || !/^[a-z0-9_.-]+(\/[a-z0-9_.-]+)*$/i.test(name) || name.includes("..")) return next();
        try {
          const body = readFileSync(`${source}/${name}`);
          response.setHeader("Content-Type", type);
          response.end(body);
        } catch {
          next();
        }
      });
    }
  },
  closeBundle() {
    for (const { served, source } of MATHJAX_ASSETS) {
      cpSync(source, `${root}dist/${served}`, { recursive: true });
    }
  },
});

export default defineConfig({
  root,
  base: "./",
  plugins: [svelte(), mathjaxOfflineAssets()],
  build: { target: "es2022", sourcemap: true },
});
