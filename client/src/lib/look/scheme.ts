// Base16 scheme parsing and the contrast floor (t-14, "The design
// language"). The colour pipeline is pure: scheme text in, sixteen
// colours out; a role colour and a background in, a floored colour out.
// Meaning-bearing roles keep their hue and gain lightness/saturation in
// OKLCH until they reach their contrast floor (~3:1 shapes, 4.5:1
// text), against the background they render on. Ported from the round
// 16 reference's lookVars (t-22@r3), which is the ratified algorithm;
// this module adds no policy of its own.

export type Base16 = Record<string, string>;

const HEX_LINE = /^\s*(base0[0-9A-Fa-f])\s*:\s*"?#?([0-9a-fA-F]{6})"?/;

/** Parse a base16 scheme declaration. Accepts the canonical YAML-ish
 * form (`baseXX: "hex"` lines, optional quotes and leading `#`) and the
 * JSON object form used by some base16 tooling; returns null unless all
 * sixteen colours are present. */
export function parseScheme(text: string): Base16 | null {
  if (/^\s*\{/.test(text)) {
    try {
      const raw = JSON.parse(text) as Record<string, unknown>;
      const out: Base16 = {};
      for (const [key, value] of Object.entries(raw)) {
        if (!/^base0[0-9A-Fa-f]$/.test(key) || typeof value !== "string") continue;
        const hex = /^#?([0-9a-fA-F]{6})$/.exec(value.trim());
        if (hex && hex[1] !== undefined) out[key.slice(0, 5) + key.charAt(5).toUpperCase()] = "#" + hex[1].toLowerCase();
      }
      return Object.keys(out).length === 16 ? out : null;
    } catch {
      return null;
    }
  }
  const out: Base16 = {};
  for (const line of text.split("\n")) {
    const match = HEX_LINE.exec(line);
    if (match && match[1] !== undefined && match[2] !== undefined) {
      out[match[1].slice(0, 5) + match[1].charAt(5).toUpperCase()] = "#" + match[2].toLowerCase();
    }
  }
  return Object.keys(out).length === 16 ? out : null;
}

/** The scheme's own name (its `scheme:` line), for plain notices and
 * evidence; null when the declaration names none. */
export function schemeName(text: string): string | null {
  const match = /^\s*scheme\s*:\s*"?([^"\n]+)"?\s*$/m.exec(text);
  return match && match[1] !== undefined ? match[1].trim() : null;
}

// ---------- sRGB <-> OKLCH ----------

type Rgb = [number, number, number];

const channel = (hex: string, at: number): number => parseInt(hex.slice(at, at + 2), 16) / 255;

const hexRgb = (hex: string): Rgb => [channel(hex, 1), channel(hex, 3), channel(hex, 5)];

const byte = (v: number): string =>
  Math.round(Math.max(0, Math.min(1, v)) * 255)
    .toString(16)
    .padStart(2, "0");

const rgbHex = (rgb: Rgb): string => `#${byte(rgb[0])}${byte(rgb[1])}${byte(rgb[2])}`;

const lin = (v: number): number => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const unlin = (v: number): number => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);

function toOklch(hex: string): [number, number, number] {
  const [r, g, b] = hexRgb(hex);
  const lr = lin(r);
  const lg = lin(g);
  const lb = lin(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  return [L, Math.hypot(A, B), Math.atan2(B, A)];
}

/** The colour's OKLCH hue in radians, for hue-preservation evidence. */
export function toOklchHue(hex: string): number {
  return toOklch(hex)[2];
}

function fromOklch(L: number, C: number, H: number): Rgb {
  const A = C * Math.cos(H);
  const B = C * Math.sin(H);
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
  return [
    unlin(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    unlin(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    unlin(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ];
}

const inGamut = (rgb: Rgb): boolean => rgb.every((v) => v >= -0.0005 && v <= 1.0005);

// ---------- contrast ----------

/** WCAG 2 relative luminance of a hex colour. */
export function lum(hex: string): number {
  const [r, g, b] = hexRgb(hex);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** WCAG 2 contrast ratio between two hex colours. */
export function ratio(a: string, b: string): number {
  const x = lum(a);
  const y = lum(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

/** Shortest angular distance between two hues, in radians. */
export function hueDistance(a: number, b: number): number {
  const d = Math.abs(a - b) % (2 * Math.PI);
  return Math.min(d, 2 * Math.PI - d);
}

export type FloorOutcome = {
  hex: string;
  /** Measured contrast of the outcome against the background. */
  ratio: number;
  /** True when the outcome reached the requested ratio. */
  reached: boolean;
  /** True when the hue survives (the request is hue-preserving; a
   * near-zero-chroma source has no hue to keep and reports false). */
  hueKept: boolean;
};

/**
 * The contrast floor. Keep the hue; raise saturation to at least
 * `minChroma` (OKLCH units) and walk lightness in the direction that
 * increases contrast against `bg` (dark backgrounds push lightness up,
 * light backgrounds push it down) until the colour meets `minRatio`.
 * Out of gamut colours give up chroma first, never hue. When the bound
 * is exhausted the best colour found is returned and `reached` says so:
 * the caller discloses the miss, it is never papered over.
 */
export function floor(hex: string, bg: string, minRatio: number, minChroma: number): FloorOutcome {
  const source = toOklch(hex);
  let L = source[0];
  let C = source[1];
  const H = source[2];
  const sourceChroma = C;
  const dark = lum(bg) < 0.2;
  if (C < minChroma) C = minChroma;
  const make = (): string => {
    let c = C;
    let rgb = fromOklch(L, c, H);
    while (!inGamut(rgb) && c > 0) {
      c -= 0.004;
      rgb = fromOklch(L, c, H);
    }
    return rgbHex(rgb);
  };
  let out = make();
  for (let i = 0; i < 60 && ratio(out, bg) < minRatio; i++) {
    L += dark ? 0.012 : -0.012;
    L = Math.max(0, Math.min(1, L));
    out = make();
  }
  const measured = ratio(out, bg);
  return {
    hex: out,
    ratio: measured,
    reached: measured >= minRatio,
    hueKept: sourceChroma >= 0.02 ? hueDistance(H, toOklch(out)[2]) < 0.05 : false,
  };
}

/** Hex with alpha over a hex colour, as a CSS rgba() string. */
export function rgba(hex: string, alpha: number): string {
  const [r, g, b] = hexRgb(hex);
  return `rgba(${byte(r)},${byte(g)},${byte(b)},${alpha})`;
}

/** Linear-in-sRGB mix of two hex colours; t=0 is a, t=1 is b. */
export function mix(a: string, b: string, t: number): string {
  const ca = hexRgb(a);
  const cb = hexRgb(b);
  return rgbHex([ca[0] * (1 - t) + cb[0] * t, ca[1] * (1 - t) + cb[1] * t, ca[2] * (1 - t) + cb[2] * t]);
}
