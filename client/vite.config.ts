import { cpSync } from "node:fs";
import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// MathJax 4's speech worker resolves against loader.paths.mathjax; the
// pinned package's sre directory (worker + mathmaps) is copied into the
// build so it loads offline from the same origin - never an external
// fetch.
const mathjaxSreAssets = (): Plugin => ({
  name: "mathjax-sre-assets",
  closeBundle() {
    cpSync("node_modules/mathjax/sre", "dist/sre", { recursive: true });
  },
});

export default defineConfig({
  base: "./",
  plugins: [svelte(), mathjaxSreAssets()],
  build: { target: "es2022", sourcemap: true },
  server: { port: 1420, strictPort: true },
  preview: { port: 1421, strictPort: true },
});
