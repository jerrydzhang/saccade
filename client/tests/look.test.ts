// The look pipeline's contracts: scheme parsing, the contrast floor,
// the scheme's own polarity, the token map's coverage, and the
// configuration resolution (t-14 "The design language"; t-24@r3).

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { floor, hueDistance, parseScheme, ratio, schemeName, toOklchHue } from "../src/lib/look/scheme.ts";
import { DEFAULT_SCHEME_TEXT, lookVars, polarity, WEIGHTS } from "../src/lib/look/lookVars.ts";
import {
  resolveLook,
  setLookOverride,
  type LookSource,
} from "../src/lib/look/store.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const STYLES = join(HERE, "../src/styles");

const MOUNTAIN = DEFAULT_SCHEME_TEXT;

// A light base16 scheme (light base00, dark inks) - polarity is the
// scheme's own, so this must give a light suite with no mode anywhere.
const LIGHT = `scheme: "Paper"
base00: "f4f1ea"
base01: "e6e1d6"
base02: "d8d2c4"
base03: "8a8478"
base04: "5e5a52"
base05: "443f38"
base06: "2e2b26"
base07: "1c1b19"
base08: "a65b5b"
base09: "a67f4d"
base0A: "9a905c"
base0B: "6f8f6f"
base0C: "5f8f90"
base0D: "6a688f"
base0E: "8f6a8f"
base0F: "a65b5b"`;

// A deliberately hostile scheme: near-background hues and a
// near-grayscale accent, to measure what the floor can and cannot do.
const EXTREME = `scheme: "Hostile"
base00: "101010"
base01: "161616"
base02: "1d1d1d"
base03: "212121"
base04: "9c9c9c"
base05: "c8c8c8"
base06: "dedede"
base07: "efefef"
base08: "141a14"
base09: "181813"
base0A: "121212"
base0B: "0e1410"
base0C: "101414"
base0D: "141418"
base0E: "181214"
base0F: "141414"`;

test("a complete base16 declaration parses, quotes and hashes optional", () => {
  const scheme = parseScheme(MOUNTAIN);
  assert.ok(scheme);
  assert.equal(scheme.base00, "#100f0e");
  assert.equal(scheme.base07, "#fafafa");
  assert.equal(scheme.base0D, "#aaa8c4");
});

test("an incomplete declaration refuses", () => {
  assert.equal(parseScheme("base00: \"000000\"\nbase01: \"111111\""), null);
  assert.equal(parseScheme(""), null);
});

test("a JSON-form declaration parses", () => {
  const json = JSON.stringify(Object.fromEntries([...Array(16)].map((_, i) => [`base0${i.toString(16)}`, "abcdef"])));
  const scheme = parseScheme(json);
  assert.ok(scheme);
  assert.equal(scheme.base0A, "#abcdef");
});

test("the scheme's own name reads for notices and evidence", () => {
  assert.equal(schemeName(MOUNTAIN), "Mountain (Boosted)");
  assert.equal(schemeName(LIGHT), "Paper");
  assert.equal(schemeName("base00: \"000000\""), null);
});

test("the floor reaches its ratio with the hue kept, dark scheme", () => {
  const scheme = parseScheme(MOUNTAIN)!;
  const bg = scheme.base00 as string;
  const roles: [string, string, number][] = [
    ["working", scheme.base0D as string, 3.2],
    ["done", scheme.base0B as string, 3.2],
    ["blocked", scheme.base0A as string, 3.2],
    ["error", scheme.base08 as string, 3.2],
    ["text", scheme.base0C as string, 4.5],
  ];
  for (const [name, hex, minRatio] of roles) {
    const out = floor(hex, bg, minRatio, 0.1);
    assert.ok(out.ratio >= minRatio, `${name}: measured ${out.ratio.toFixed(2)} < ${minRatio}`);
    const hueMoved = hueDistance(toOklchHue(hex), toOklchHue(out.hex));
    assert.ok(hueMoved < 0.05, `${name}: hue moved`);
  }
});

test("the floor walks lightness away from the background's side", () => {
  const scheme = parseScheme(MOUNTAIN)!;
  const source = scheme.base0C;
  assert.ok(source, "the default scheme names base0C");
  const dark = floor(source, scheme.base00 as string, 4.5, 0);
  const light = floor(source, "#fafafa", 4.5, 0);
  // against the dark background the colour must lighten...
  assert.ok(
    Number.parseInt(dark.hex.slice(1, 3), 16) >= Number.parseInt(source.slice(1, 3), 16),
    "a floored role on a dark background must not darken",
  );
  // ...and against a light one it must darken
  assert.ok(
    Number.parseInt(light.hex.slice(1, 3), 16) <= Number.parseInt(source.slice(1, 3), 16),
    "a floored role on a light background must not lighten",
  );
});

test("the token map reaches every floored role on the default and light schemes", () => {
  for (const schemeText of [MOUNTAIN, LIGHT]) {
    const look = lookVars(schemeText, "regular");
    assert.deepEqual(look.misses, [], `floor misses on ${schemeName(schemeText) ?? "scheme"}: ${JSON.stringify(look.misses)}`);
  }
});

test("extreme schemes reach every floor", () => {
  const look = lookVars(EXTREME, "regular");
  assert.deepEqual(look.misses, [], `hostile scheme misses: ${JSON.stringify(look.misses)}`);
  const bg = parseScheme(EXTREME)!.base00 as string;
  for (const token of ["--s-working", "--s-done", "--c-task", "--ref"]) {
    const measured = ratio(look.vars[token] as string, bg);
    assert.ok(measured >= 3.0, `${token} measured ${measured.toFixed(2)}`);
  }
});

// The nadir background, measured: the natural-direction walk (against a
// background near the contrast nadir, relative luminance about 0.19)
// caps white-side contrast at about 4.31:1, so the floor falls back to
// the opposite direction, where black side clears the floor. The floor
// is always reachable; a miss would be a defect (t-24@r10).
const MID = `scheme: "Mid"
base00: "7a7a76"
base01: "6e6e6a"
base02: "85857f"
base03: "5c5c58"
base04: "9a9a94"
base05: "a8a8a2"
base06: "b8b8b2"
base07: "c8c8c2"
base08: "7a5c5c"
base09: "7a6e4c"
base0A: "74744c"
base0B: "4c7a4c"
base0C: "4c7a78"
base0D: "4c4c7a"
base0E: "7a4c7a"
base0F: "7a5c5c"`;

test("the nadir background reaches every floor via the opposite walk", () => {
  const look = lookVars(MID, "regular");
  assert.deepEqual(look.misses, [], `the nadir scheme must reach: ${JSON.stringify(look.misses)}`);
  const bg = parseScheme(MID)!.base00 as string;
  // the text floors hold (reached through the fallback direction)
  for (const token of ["--c-task", "--c-sess", "--ref", "--w-ui-error"]) {
    const measured = ratio(look.vars[token] as string, bg);
    assert.ok(measured >= 4.5, `${token} measured ${measured.toFixed(2)}`);
  }
  // and the shape floors hold
  assert.ok(ratio(look.vars["--s-working"] as string, bg) >= 3.2);
  assert.ok(ratio(look.vars["--s-error"] as string, bg) >= 3.2);
});

test("white or black clears 4.5 against any background; the sweep reaches everywhere", () => {
  // the arithmetic r10 names: the worse side of white vs black bottoms
  // out at about 4.58:1 at the luminance crossover (about 0.179)
  for (const bg of ["#000000", "#101010", "#303030", "#4a4a46", "#5c5c58", "#6e6e6a", "#7a7a76", "#85857f", "#909090", "#9a9a94", "#c8c8c2", "#d8d2c4", "#e6e1d6", "#f2f2f0", "#f4f1ea", "#ffffff"]) {
    const white = ratio("#ffffff", bg);
    const black = ratio("#000000", bg);
    assert.ok(Math.max(white, black) >= 4.58, `${bg}: white ${white.toFixed(2)} black ${black.toFixed(2)}`);
  }
  // and the whole token map reaches every floor across broad schemes:
  // dark and light, mid-luminance, hostile hues, grayscale
  for (const schemeText of [MOUNTAIN, LIGHT, EXTREME, MID]) {
    const look = lookVars(schemeText, "regular");
    assert.deepEqual(
      look.misses,
      [],
      `${schemeName(schemeText) ?? "scheme"}: ${JSON.stringify(look.misses)}`,
    );
  }
});

test("light or dark is the scheme's own; nothing reverses", () => {
  const dark = lookVars(MOUNTAIN, "regular");
  const light = lookVars(LIGHT, "regular");
  assert.equal(dark.polarity, "dark");
  assert.equal(light.polarity, "light");
  assert.equal(dark.vars["--app-bg"], "#100f0e");
  assert.equal(light.vars["--app-bg"], "#f4f1ea");
  // a light scheme's own inks are dark: the default ink is base05
  const scheme = parseScheme(LIGHT)!;
  assert.equal(light.vars["--ink"], scheme.base05 as string);
  // and the polarity function agrees on both
  assert.equal(polarity(scheme), "light");
  assert.equal(polarity(parseScheme(MOUNTAIN)!), "dark");
});

test("weight maps to the ui and world weights, defaulting to regular", () => {
  assert.deepEqual(WEIGHTS.regular, ["400", "400"]);
  const look = lookVars(MOUNTAIN, "medium");
  assert.equal(look.vars["--wt-ui"], "500");
  assert.equal(look.vars["--wt-world"], "450");
});

test("every colour variable the stylesheets read, the map produces", () => {
  const look = lookVars(MOUNTAIN, "regular");
  const produced = new Set(Object.keys(look.vars));
  const statics = new Set<string>();
  const tokensCss = readFileSync(join(STYLES, "tokens.css"), "utf8");
  for (const match of tokensCss.matchAll(/^\s*(--[a-z0-9-]+)\s*:/gm)) {
    if (match[1] !== undefined) statics.add(match[1]);
  }
  const consumed = new Set<string>();
  for (const name of ["tokens.css", "board.css"]) {
    const css = readFileSync(join(STYLES, name), "utf8");
    for (const match of css.matchAll(/var\((--[a-z0-9-]+)[),]/g)) {
      if (match[1] !== undefined) consumed.add(match[1]);
    }
  }
  // tokens.css defines the statics it consumes; board.css may consume
  // only map colours or tokens.css statics
  const missing = [...consumed].filter((token) => !produced.has(token) && !statics.has(token));
  assert.deepEqual(missing, [], "styles read variables nobody defines");
  // and the map produces nothing the styles have abandoned, except the
  // named tokens waiting for their components: the later instruments'
  // rows and menus (pop-line, hover, sel, the identifier hues), the
  // viewer's history and annotation marks (d-add, draft, warn, hint),
  // and the paper pair the steps 2/3 surfaces will read
  const future = new Set([
    "--hud",
    "--bar",
    "--pop-line",
    "--line-soft",
    "--hover",
    "--sel",
    "--c-sess",
    "--c-msg",
    "--c-ws",
    "--c-key",
    "--c-link",
    "--w-ui-working",
    "--w-ui-blocked",
    "--d-add",
    "--d-add-bg",
    "--d-mod",
    "--d-del",
    "--d-del-bg",
    "--draft",
    "--draft-bg",
    "--warn-bg",
    "--warn-ink",
    "--hint-bg",
    "--hint-ink",
    "--hint-2",
    "--w-bg",
    "--w-ink",
    "--w-code",
    "--chip",
    "--lane-bg",
    "--lane-c",
  ]);
  const unused = [...produced].filter((token) => !consumed.has(token) && !future.has(token));
  assert.deepEqual(unused, [], "the map produces variables no style reads");
});

test("configuration resolution falls back with one plain notice", () => {
  // no host at all (the browser lane): the built-in default, silently
  // - there is nothing here the human can fix (t-24@r7)
  const none = resolveLook(null);
  assert.equal(none.scheme.base00, "#100f0e");
  assert.equal(none.weight, "regular");
  assert.equal(none.notice, null);

  // a missing config file
  const missing: LookSource = {
    generation: 1,
    configFound: false,
    configPath: "/home/me/.config/dev.saccade.board-viewer/config.toml",
    schemeText: null,
    weight: null,
    problem: null,
  };
  const missed = resolveLook(missing);
  assert.match(missed.notice ?? "", /no look configuration at \/home\/me\/.*; the built-in scheme is in use/);

  // a stated problem from the host
  const broken: LookSource = { generation: 2, configFound: true, configPath: "/x/config.toml", schemeText: null, weight: null, problem: "scheme file missing: /x/scheme.yaml" };
  assert.match(resolveLook(broken).notice ?? "", /scheme file missing/);

  // a complete valid configuration resolves cleanly, no notice
  const good: LookSource = { generation: 3, configFound: true, configPath: "/x/config.toml", schemeText: LIGHT, weight: "medium", problem: null };
  const resolved = resolveLook(good);
  assert.equal(resolved.notice, null);
  assert.equal(resolved.weight, "medium");
  assert.equal(resolved.polarity, "light");
});

test("an invalid configured scheme falls back and says so", () => {
  const bad: LookSource = { generation: 4, configFound: true, configPath: "/x/config.toml", schemeText: "base00: \"000000\"", weight: null, problem: null };
  const look = resolveLook(bad);
  assert.equal(look.scheme.base00, "#100f0e");
  assert.match(look.notice ?? "", /not a complete base16 scheme/);
});

test("the inspection override is a test surface, in memory only", () => {
  setLookOverride({ schemeText: LIGHT });
  const overridden = resolveLook(null);
  assert.equal(overridden.polarity, "light");
  assert.equal(overridden.fromOverride, true);
  assert.equal(overridden.notice, null);
  setLookOverride({ weight: "semibold" });
  assert.equal(resolveLook(null).weight, "semibold");
  setLookOverride(null);
  assert.equal(resolveLook(null).weight, "regular");
  assert.equal(resolveLook(null).fromOverride, false);
});
