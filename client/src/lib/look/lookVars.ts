// The token map: a user's base16 scheme and weight in, the design
// language's colour tokens out (t-14 "The design language", as
// clarified by t-24@r3). Light or dark is the scheme's own: the ramp
// reads in base16 order and a light scheme (a light base00) gives a
// light suite on its own; the mockup's ramp reversal was a comparison
// device, not a feature, and nothing here reverses anything. Meaning
// bearing roles keep their hue and are floored in OKLCH against the
// background (about 3:1 for state shapes, 4.5:1 for text, per role
// minimums ratified with the round 16 reference; t-22@r3).

import { floor, lum, mix, parseScheme, ratio as ratioVs, rgba, type Base16 } from "./scheme.ts";

/** The built-in default scheme, used when no configuration names one or
 * the named one is missing or invalid. It is the user's own Mountain
 * (Boosted), the scheme the round 16 reference was drawn against. */
export const DEFAULT_SCHEME_TEXT = `scheme: "Mountain (Boosted)"
author: "gnsfujiwara (https://github.com/gnsfujiwara), modified by jerry"
base00: "100f0e"
base01: "1b1918"
base02: "292624"
base03: "6d6562"
base04: "c4a6a8"
base05: "d4ceca"
base06: "e8e2dd"
base07: "fafafa"
base08: "c4a6a8"
base09: "dac09a"
base0A: "c4c090"
base0B: "a2c4a3"
base0C: "a2c3c4"
base0D: "aaa8c4"
base0E: "c4a6c4"
base0F: "c4a6a8"`;

/** Text weight is a user preference (t-24@r3); regular by default. The
 * pair is [ui, world] weight, matching the round 16 reference. */
export type LookWeight = "regular" | "medium" | "semibold";

export const WEIGHTS: Record<LookWeight, [string, string]> = {
  regular: ["400", "400"],
  medium: ["500", "450"],
  semibold: ["600", "500"],
};

/** The [ui, world] weights for a preference; regular when unknown. */
export function weightPair(weight: LookWeight): [string, string] {
  const pair = WEIGHTS[weight];
  return pair ?? WEIGHTS.regular;
}

export function isLookWeight(value: string): value is LookWeight {
  return value === "regular" || value === "medium" || value === "semibold";
}

/** Light or dark is the scheme's own: read from the background's
 * luminance, never a user mode. */
export function polarity(scheme: Base16): "dark" | "light" {
  const bg = scheme.base00;
  return bg !== undefined && lum(bg) < 0.2 ? "dark" : "light";
}

export type FloorMiss = {
  token: string;
  requested: number;
  measured: number;
};

export type LookTokens = {
  vars: Record<string, string>;
  /** Roles whose floor could not be reached within the walk bound.
   * The caller discloses misses; nothing here paper over one. */
  misses: FloorMiss[];
  polarity: "dark" | "light";
  weight: LookWeight;
};

/**
 * Map a scheme onto every colour token the design language reads.
 * The ramp reads in base16 order (bg base00, surfaces 01-02, muted 03,
 * inks 04-07); polarity, floor direction and shadows follow the scheme.
 */
export function lookVars(schemeText: string, weight: LookWeight): LookTokens {
  const scheme = parseScheme(schemeText) ?? parseScheme(DEFAULT_SCHEME_TEXT);
  if (!scheme) throw new Error("the built-in default scheme must parse");
  const b = (k: string): string => {
    const hex = scheme["base0" + k];
    if (hex === undefined) throw new Error(`the scheme lacks base0${k}`);
    return hex;
  };
  const misses: FloorMiss[] = [];

  const bg = b("0");
  const [s1, s2, s3] = [b("1"), b("2"), b("3")];
  const ink = b("5");
  const inkHi = b("6");
  const dark = polarity(scheme) === "dark";
  const mode = dark ? "dark" : "light";

  const F = (token: string, hex: string, minRatio: number, minChroma: number): string => {
    const out = floor(hex, bg, minRatio, minChroma);
    if (!out.reached) misses.push({ token, requested: minRatio, measured: out.ratio });
    return out.hex;
  };
  const text = (token: string, hex: string): string => F(token, hex, 4.5, 0);

  const working = F("s-working", b("D"), 3.2, 0.1);
  const done = F("s-done", b("B"), 3.2, 0.12);
  const blocked = F("s-blocked", b("A"), 3.2, 0.12);
  const error = F("s-error", b("8"), 3.2, 0.14);
  const mod = F("d-mod", b("9"), 3.2, 0.11);

  const hint = F("hint-bg", b("A"), 3.0, 0.13);
  const hintInk = ratioVs(hint, b("0")) > ratioVs(hint, b("7")) ? b("0") : b("7");
  const acc = F("acc", b("D"), 3.0, 0.08);
  const accInk = ratioVs(acc, b("0")) > ratioVs(acc, b("7")) ? b("0") : b("7");

  const vars: Record<string, string> = {
    "--app-bg": bg,
    "--hud": "transparent",
    "--bar": rgba(s1, 0.55),
    "--pop": rgba(s1, 0.97),
    "--pop-line": s2,
    "--line": rgba(s2, 1),
    "--line-strong": mix(s2, s3, 0.35),
    "--line-soft": rgba(s2, 0.55),
    "--ink": ink,
    "--ink-hi": inkHi,
    "--ink-2": mix(ink, bg, 0.3),
    "--ink-3": F("ink-3", s3, 3.0, 0),
    "--acc": acc,
    "--acc-ink": accInk,
    "--hover": rgba(acc, 0.08),
    "--sel": rgba(acc, 0.2),
    "--chip": rgba(ink, 0.07),
    "--c-task": text("c-task", b("D")),
    "--c-sess": text("c-sess", b("B")),
    "--c-msg": text("c-msg", b("E")),
    "--c-ws": text("c-ws", b("C")),
    "--c-key": text("c-key", b("A")),
    "--c-link": text("c-link", b("9")),
    "--s-working": working,
    "--s-done": done,
    "--s-blocked": blocked,
    "--s-error": error,
    "--s-idle": F("s-idle", s3, 3.0, 0),
    "--w-ui-working": text("w-ui-working", b("D")),
    "--w-ui-blocked": text("w-ui-blocked", b("A")),
    "--w-ui-error": text("w-ui-error", b("8")),
    "--d-add": done,
    "--d-add-bg": rgba(done, dark ? 0.24 : 0.2),
    "--d-mod": mod,
    "--d-del": error,
    "--d-del-bg": rgba(error, dark ? 0.22 : 0.17),
    "--draft": mod,
    "--draft-bg": rgba(mod, 0.1),
    "--warn-bg": rgba(mod, 0.18),
    "--warn-ink": text("warn-ink", b("9")),
    "--hint-bg": hint,
    "--hint-ink": hintInk,
    "--hint-2": done,
    "--w-bg": "transparent",
    "--w-ink": ink,
    "--w-code": rgba(s2, 0.7),
    "--w-line": rgba(s2, 1),
    "--ref": text("ref", b("D")),
    "--sh-pop": dark ? "0 14px 34px rgba(0,0,0,.4)" : "0 10px 28px rgba(0,0,0,.14)",
    "--lane-bg": rgba(ink, 0.05),
    "--lane-c": mod,
    "--scroll": rgba(ink, 0.2),
    "--wt-ui": weightPair(weight)[0],
    "--wt-world": weightPair(weight)[1],
  };
  return { vars, misses, polarity: mode, weight };
}
