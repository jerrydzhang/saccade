// The theming contract (t-59, "The language so far") as an executable
// test: tokens.css is the one vocabulary and stands alone; theme.css
// only assigns tokens, light on :root and dark on :root[data-theme=
// "dark"]; every block stylesheet reads tokens only; components carry
// no styling of their own; the theme mode follows the system.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "vitest";
import { followThemePreference, type PreferenceQuery } from "../src/lib/theme-mode.ts";

const src = path.resolve(import.meta.dirname, "../src");
const styles = path.join(src, "styles");
const read = (name: string): string => fs.readFileSync(path.join(styles, name), "utf8");

function stripComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

/** The token assignments of every rule whose selector matches. */
function assigned(css: string, selector: RegExp): Set<string> {
  const declared = new Set<string>();
  for (const match of stripComments(css).matchAll(/([^{}]+)\{([^}]*)\}/g)) {
    if (!selector.test(match[1]!.trim())) continue;
    for (const token of match[2]!.matchAll(/--port-[a-z0-9-]+(?=\s*:)/g)) declared.add(token[0]);
  }
  return declared;
}

const ROOT = /^:root$/;
const DARK = /^:root\[data-theme="dark"\]$/;

function blockFiles(): string[] {
  return fs
    .readdirSync(path.join(styles, "blocks"))
    .filter((name) => name.endsWith(".css"))
    .map((name) => `blocks/${name}`);
}

function consumed(css: string): Set<string> {
  const out = new Set<string>();
  for (const match of css.matchAll(/var\((--port-[a-z0-9-]+)/g)) out.add(match[1]!);
  return out;
}

function svelteFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...svelteFiles(full));
    else if (entry.name.endsWith(".svelte")) out.push(full);
  }
  return out;
}

test("the theme only assigns tokens: a light :root and a dark :root[data-theme=dark], nothing else", () => {
  const theme = stripComments(read("theme.css"));
  for (const match of theme.matchAll(/([^{}]+)\{/g)) {
    const selector = match[1]!.trim();
    assert.ok(ROOT.test(selector) || DARK.test(selector), `the theme carries a rule for ${selector}`);
  }
  for (const match of theme.matchAll(/\{([^}]*)\}/g)) {
    for (const declaration of match[1]!.split(";").map((part) => part.trim()).filter((part) => part.length > 0)) {
      assert.match(declaration, /^--port-[a-z0-9-]+\s*:/, `the theme declares more than a token: ${declaration}`);
    }
  }
  assert.ok(!theme.includes("!important"), "the theme never uses !important");
});

test("every token a stylesheet reads is declared in the base, which stands alone", () => {
  const base = assigned(read("tokens.css"), ROOT);
  for (const file of ["app.css", ...blockFiles()]) {
    for (const token of consumed(read(file))) {
      assert.ok(base.has(token), `${file} reads ${token}, which tokens.css does not declare`);
    }
  }
});

test("the theme assigns only base tokens, and dark assigns only what light assigns", () => {
  const base = assigned(read("tokens.css"), ROOT);
  const light = assigned(read("theme.css"), ROOT);
  const dark = assigned(read("theme.css"), DARK);
  for (const token of light) assert.ok(base.has(token), `theme.css assigns ${token}, unknown to the base`);
  for (const token of dark) assert.ok(light.has(token), `dark assigns ${token}, which light does not`);
});

test("every block stylesheet reads tokens only: no literal colour in a block or the frame", () => {
  for (const file of ["app.css", ...blockFiles()]) {
    const css = stripComments(read(file));
    const literal = css.match(/#[0-9a-f]{3,8}\b|rgba?\(|hsla?\(/i);
    assert.equal(literal, null, `${file} carries a literal colour (${literal?.[0]}); name a token instead`);
  }
});

test("components carry no styling of their own: no <style>, no inline colour or type", () => {
  for (const file of svelteFiles(src)) {
    if (file.includes(`${path.sep}specimen${path.sep}`)) continue;
    const text = fs.readFileSync(file, "utf8");
    const name = path.relative(src, file);
    assert.ok(!/<style[\s>]/.test(text), `${name} carries a <style> block; its look belongs in styles/blocks`);
    for (const match of text.matchAll(/style="([^"]*)"/g)) {
      assert.ok(
        !/(^|;)\s*(color|background|font|border|box-shadow)/.test(match[1]!),
        `${name} sets colour or type inline: ${match[1]}`,
      );
    }
  }
});

test("main.ts imports the language once, after the base and the blocks, and bundles the faces", () => {
  const main = fs.readFileSync(path.join(src, "main.ts"), "utf8");
  assert.equal(main.split("theme.css").length - 1, 1, "theme.css rides exactly one import");
  assert.ok(main.indexOf("app.css") < main.indexOf("theme.css"), "the language imports after the base styles");
  const fonts = fs.readFileSync(path.join(src, "fonts.ts"), "utf8");
  assert.match(fonts, /@fontsource\/source-serif-4/);
  assert.match(fonts, /@fontsource\/ibm-plex-mono/);
});

test("the theme carries the language's chosen values (t-59), pinned against silent drift", () => {
  const theme = stripComments(read("theme.css"));
  for (const value of [
    "#1f6b6e", // the accent, light: pale teal's ink
    "#a2c3c4", // the accent, dark: the human's base0C
    "#100f0e", // dark: the human's base00
    "#fbf8f3", // light: the board's paper
    "Source Serif 4",
    "IBM Plex Mono",
  ]) {
    assert.ok(theme.includes(value), `the theme lost ${value}`);
  }
});

test("the theme mode follows the system preference and stops when asked", () => {
  const root = { dataset: {} as DOMStringMap };
  let listener: (() => void) | null = null;
  const query: PreferenceQuery & { matches: boolean } = {
    matches: false,
    addEventListener: (_type, next) => {
      listener = next;
    },
    removeEventListener: (_type, gone) => {
      if (listener === gone) listener = null;
    },
  };
  const stop = followThemePreference(root, query);
  assert.equal(root.dataset.theme, "light");
  query.matches = true;
  listener!();
  assert.equal(root.dataset.theme, "dark");
  stop();
  assert.equal(listener, null, "stopping removes the listener");
});
