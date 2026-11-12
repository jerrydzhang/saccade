// The theming contract (r93/r94) as an executable test: the theme is
// ONE overlay that only assigns :root token values; the base stands
// alone; nothing consumes a token the base does not define; tear-out
// is deleting one file and one import.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const styles = path.resolve(import.meta.dirname, "../src/styles");
const read = (name: string): string => fs.readFileSync(path.join(styles, name), "utf8");

function rootVariables(css: string): Set<string> {
  const declared = new Set<string>();
  for (const block of css.match(/:root[^{]*\{([^}]*)\}/g) ?? []) {
    for (const match of block.matchAll(/--port-[a-z0-9-]+\s*:/g)) {
      declared.add(match[0]!.replace(/\s*:/, ""));
    }
  }
  return declared;
}

function stripComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

function consumedVariables(css: string): Set<string> {
  const consumed = new Set<string>();
  for (const match of css.matchAll(/var\((--port-[a-z0-9-]+)/g)) {
    consumed.add(match[1]!);
  }
  return consumed;
}

test("the theme overlay only assigns :root tokens - no selectors, no component rules", () => {
  const theme = stripComments(read("theme.css"));
  // every rule is a bare :root block
  const rules = theme.split("}").map((part) => part.trim()).filter((part) => part.length > 0);
  for (const rule of rules) {
    assert.match(rule, /^:root\s*\{/, `the theme carries a non-:root rule: ${rule.slice(0, 60)}`);
  }
  // and no !important anywhere (the overlay never fights the base)
  assert.ok(!theme.includes("!important"), "the theme must never use !important");
});

test("every token the app consumes is declared in the base (the base stands alone)", () => {
  const base = read("tokens.css");
  const declared = rootVariables(base);
  for (const consumed of consumedVariables(read("app.css"))) {
    assert.ok(
      declared.has(consumed),
      `app.css consumes ${consumed} but tokens.css does not declare it`,
    );
  }
});

test("every token the theme defines is a base token (the overlay swaps, never invents structure)", () => {
  const base = rootVariables(read("tokens.css"));
  for (const themed of rootVariables(read("theme.css"))) {
    assert.ok(base.has(themed), `theme.css defines ${themed}, unknown to the base`);
  }
});

test("tear-out is one import: main.ts names theme.css exactly once, after the base", () => {
  const main = fs.readFileSync(path.resolve(import.meta.dirname, "../src/main.ts"), "utf8");
  const occurrences = main.split("theme.css").length - 1;
  assert.equal(occurrences, 1, "theme.css rides exactly one import");
  const themeAt = main.indexOf("theme.css");
  const appAt = main.indexOf("app.css");
  assert.ok(appAt < themeAt, "the overlay imports after the base styles");
});

test("the theme carries the prototype's core palette (transcribed from the compiled r16 tokens)", () => {
  const theme = stripComments(read("theme.css"));
  // the extracted values the visual review will judge; the test pins
  // the transcription against silent drift
  for (const value of [
    "#0d0f13", // app-bg
    "#d6dae2", // ink
    "#8b93a3", // ink-2
    "#4c8dff", // accent
    "#f4f1ea", // the paper world
    "JetBrains Mono",
    "Source Serif 4",
  ]) {
    assert.ok(theme.includes(value), `the theme lost the r16 value ${value}`);
  }
});
