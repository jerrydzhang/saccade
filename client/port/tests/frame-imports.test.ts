// The frame/instrument boundary, held by reading the source: the
// frame's files (the assembly, its component, the frame components
// and the frame's lib) import nothing from the instruments' home, and
// the assembly names no instrument. Adding an instrument touches the
// instruments' own module and the instruments' index - never a frame
// file.

import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { test } from "vitest";

const ROOT = join(import.meta.dirname, "..", "src");

// the frame's files: the assembly, its component tree, and the
// frame's own lib (the rail service among them) - the instruments
// live in src/instruments, their registry home in src/lib/instruments
const frameFiles: string[] = [join(ROOT, "App.ts"), join(ROOT, "App.svelte")];
for (const dir of [join(ROOT, "components"), join(ROOT, "lib", "frame")]) {
  (function walk(d: string): void {
    for (const entry of readdirSync(d)) {
      const path = join(d, entry);
      if (statSync(path).isDirectory()) walk(path);
      else if (/\.(ts|svelte)$/.test(entry)) frameFiles.push(path);
    }
  })(dir);
}
const assembly = frameFiles.filter((file) => file.endsWith("App.ts"));

test("the frame files exist (the check reads the tree it guards)", () => {
  assert.ok(frameFiles.length > 10, `frame files found: ${frameFiles.length}`);
  assert.ok(frameFiles.some((file) => file.endsWith("App.ts")));
  assert.ok(frameFiles.some((file) => file.endsWith("App.svelte")));
});

test("no frame file imports an instrument", () => {
  const offenders: string[] = [];
  for (const file of frameFiles) {
    const source = readFileSync(file, "utf8");
    for (const match of source.matchAll(/from\s+"([^"]+)"/g)) {
      const specifier = match[1]!;
      // resolve the specifier against the importing file's directory:
      // a frame file's instruments/ imports resolve into
      // src/lib/instruments (the registry home - allowed) or into
      // src/instruments (the instruments themselves - refused)
      if (!specifier.includes("instruments/")) continue;
      const dir = file.slice(0, file.lastIndexOf("/"));
      const parts = specifier.split("/");
      const resolved: string[] = dir.split("/");
      for (const part of parts) {
        if (part === "." || part === "") continue;
        if (part === "..") resolved.pop();
        else resolved.push(part);
      }
      const resolvedPath = resolved.join("/");
      if (resolvedPath.includes("lib/instruments/")) continue;
      offenders.push(`${file}: ${specifier}`);
    }
  }
  assert.deepEqual(offenders, [], "the frame imports an instrument");
});


test("the window assembly names no instrument", () => {
  for (const file of assembly) {
    const source = readFileSync(file, "utf8");
    for (const name of ["boardInstrument", "tasksInstrument", "connectBoard", "connectTasks", "registrations"]) {
      assert.ok(!source.includes(name), `${file} names ${name}`);
    }
  }
});
