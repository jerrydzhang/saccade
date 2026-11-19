// The frame/instrument boundary, held by reading the source: the
// frame's files (the assembly, its component, the frame components
// and the frame's lib) import nothing from the instruments' home,
// nothing from the instruments' own lib, and the assembly names no
// instrument. Adding an instrument touches the instruments' own
// module and the instruments' index - never a frame file.
//
// The instruments' OWN lib (the named set below): only the tracker's
// instruments use it - the task list's lib (rail, page, menu), the
// tracker's wire client, the board's model and rendering machinery,
// the board's find, and the /view/v1 wire client. Everything else
// under lib/ is shared with the frame (the contracts, the input and
// key machinery, the select and menu sessions, the location grammar,
// the picker, the trail, the elsewhere acts, the observation, the
// sources face, the herdr-core module, the registry home) or is the
// window's own (the figure viewer's inspection lib), or is shared by
// both sides by shape (the structured issue in lib/board/errors, the
// reading layout the window holds for the board).

import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { test } from "vitest";

const ROOT = join(import.meta.dirname, "..", "src");

/** The instruments' own lib, named: none of these a frame file
 * imports. lib/board/errors.ts is deliberately left out (the
 * structured-issue shape both instruments carry) and lib/board/
 * reading-layout.ts (the window holds the layout state for the
 * board), as is lib/figures/inspection.ts with shape.ts (the window's
 * own figure viewer). */
const INSTRUMENT_LIB = [
  join(ROOT, "lib", "tasks"),
  join(ROOT, "lib", "tracker.ts"),
  join(ROOT, "lib", "pages.ts"),
  join(ROOT, "lib", "markdown.ts"),
  join(ROOT, "lib", "highlight.ts"),
  join(ROOT, "lib", "mathjax.ts"),
  join(ROOT, "lib", "mermaid.ts"),
  join(ROOT, "lib", "render.ts"),
  join(ROOT, "lib", "find"),
  join(ROOT, "lib", "board"),
];

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

/** The shared leavings inside the named set: the frame carries these
 * by shape or by state, and the check allows exactly them. */
const SHARED = [
  join(ROOT, "lib", "board", "errors.ts"),
  join(ROOT, "lib", "board", "reading-layout.ts"),
  join(ROOT, "lib", "figures", "inspection.ts"),
];

function resolvesIntoInstrumentLib(from: string, specifier: string): string | null {
  // relative specifiers resolve against the importing file's
  // directory; anything else rides the bundler's own resolution (no
  // instrument lib is reachable bare)
  if (!specifier.startsWith(".")) return null;
  const dir = from.slice(0, from.lastIndexOf("/"));
  const parts = specifier.split("/");
  const resolved: string[] = dir.split("/");
  for (const part of parts) {
    if (part === "." || part === "") continue;
    if (part === "..") resolved.pop();
    else resolved.push(part);
  }
  const resolvedPath = resolved.join("/");
  for (const root of INSTRUMENT_LIB) {
    if (resolvedPath === root || resolvedPath.startsWith(root + "/")) {
      if (SHARED.includes(resolvedPath)) return null;
      return root;
    }
  }
  return null;
}

test("no frame file imports the instruments' own lib", () => {
  const offenders: string[] = [];
  for (const file of frameFiles) {
    const source = readFileSync(file, "utf8");
    for (const match of source.matchAll(/from\s+"([^"]+)"/g)) {
      const hit = resolvesIntoInstrumentLib(file, match[1]!);
      if (hit !== null) offenders.push(`${file}: ${match[1]} (into ${hit})`);
    }
  }
  assert.deepEqual(offenders, [], "the frame imports the instruments' own lib");
});

test("the check refuses an instrument import into a frame file (the guard's own proof)", () => {
  // lib/frame/presentations.ts sits in the frame and imports
  // lib/select only; its own face is clean. The proof: a synthetic
  // specifier reaching lib/tasks from a frame file is flagged.
  assert.equal(
    resolvesIntoInstrumentLib(join(ROOT, "lib", "frame", "rail.ts"), "../../lib/tasks/rail.ts"),
    join(ROOT, "lib", "tasks"),
    "a synthetic tracker-lib import from the frame is flagged",
  );
  assert.equal(
    resolvesIntoInstrumentLib(join(ROOT, "lib", "frame", "rail.ts"), "../tasks/rail.ts"),
    join(ROOT, "lib", "tasks"),
    "the relative form is flagged too",
  );
  assert.equal(
    resolvesIntoInstrumentLib(join(ROOT, "App.ts"), "../lib/board/errors.ts"),
    null,
    "the shared structured-issue shape is allowed",
  );
});


test("the window assembly names no instrument", () => {
  for (const file of assembly) {
    const source = readFileSync(file, "utf8");
    for (const name of ["boardInstrument", "tasksInstrument", "connectBoard", "connectTasks", "registrations"]) {
      assert.ok(!source.includes(name), `${file} names ${name}`);
    }
  }
});
