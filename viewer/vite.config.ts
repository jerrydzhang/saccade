import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  base: "./",
  plugins: [svelte()],
  build: { target: "es2022", sourcemap: true },
  server: { port: 1420, strictPort: true },
  preview: { port: 1421, strictPort: true },
});
