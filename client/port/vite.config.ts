import { cpSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// The port tree's vite root is client/port: the config anchors every
// path to its own directory, so the build behaves the same from any
// working directory. The npm tree itself stays at the client root
// (reused, never duplicated); the config resolves it from there.
const root = fileURLToPath(new URL(".", import.meta.url));
const clientRoot = fileURLToPath(new URL("..", import.meta.url));

// MathJax 4's speech worker resolves against loader.paths.mathjax; the
// pinned package's sre directory (worker + mathmaps) is copied into the
// build so it loads offline from the same origin - never an external
// fetch.
const mathjaxSreAssets = (): Plugin => ({
  name: "mathjax-sre-assets",
  closeBundle() {
    cpSync(`${clientRoot}node_modules/mathjax/sre`, `${root}dist/sre`, { recursive: true });
  },
});

export default defineConfig({
  root,
  base: "./",
  plugins: [svelte(), mathjaxSreAssets()],
  build: { target: "es2022", sourcemap: true },
});
