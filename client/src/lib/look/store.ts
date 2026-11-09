// The look store: configuration in, tokens on the document root.
// The scheme and weight are configuration read from the platform's
// local config file (t-24@r3), read at start and re-read when the file
// (or the scheme file it names) changes; a missing or invalid scheme
// falls back to the built-in default and the client says so once,
// plainly. There is no preference UI and no persistence of our own:
// the file is the preference, local to the device. The inspection
// setters are test surfaces only, named as such.

import {
  DEFAULT_SCHEME_TEXT,
  isLookWeight,
  lookVars,
  polarity as schemePolarity,
  type FloorMiss,
  type LookWeight,
} from "./lookVars.ts";
import { parseScheme, schemeName, type Base16 } from "./scheme.ts";

/** What the host answers about the local configuration. `generation`
 * advances on every reload the host performs; a look wait against a
 * stale generation answers with the current payload either way. */
export type LookSource = {
  generation: number;
  configFound: boolean;
  /** The config file's own path, for a plain notice. */
  configPath: string | null;
  /** Scheme text as read (inline or from the named file); null when
   * none could be read. */
  schemeText: string | null;
  weight: LookWeight | null;
  problem: string | null;
};

export type ResolvedLook = {
  scheme: Base16;
  schemeText: string;
  weight: LookWeight;
  polarity: "dark" | "light";
  vars: Record<string, string>;
  /** Every floored role that could not reach its floor. Surfaced for
   * evidence and tests; the caller decides disclosure. */
  misses: FloorMiss[];
  /** The one plain notice, when the built-in default is in use. */
  notice: string | null;
  fromOverride: boolean;
};

// The test override: set through the inspection surface, in memory
// only, never written anywhere.
let override: { schemeText?: string; weight?: LookWeight } | null = null;

/** The resolved look for a host answer. Pure: no DOM, no calls. */
export function resolveLook(source: LookSource | null): ResolvedLook {
  const fromOverride = override !== null && (override.schemeText !== undefined || override.weight !== undefined);
  const weight: LookWeight =
    override?.weight ?? (source?.weight !== null && source?.weight !== undefined && isLookWeight(source.weight) ? source.weight : "regular");
  const overrideText = override?.schemeText;
  const schemeText = overrideText ?? source?.schemeText ?? DEFAULT_SCHEME_TEXT;
  const parsed = parseScheme(schemeText);
  const useText = parsed === null ? DEFAULT_SCHEME_TEXT : schemeText;
  const look = lookVars(useText, weight);

  let notice: string | null = null;
  if (parsed === null && overrideText !== undefined) {
    notice = "the overridden scheme is not a complete base16 scheme; the built-in scheme is in use";
  } else if (parsed === null) {
    notice = "the configured scheme is not a complete base16 scheme; the built-in scheme is in use";
  } else if (source === null) {
    notice = "no local look configuration (this window has no host); the built-in scheme is in use";
  } else if (!source.configFound) {
    notice = `no look configuration at ${source.configPath ?? "the platform configuration directory"}; the built-in scheme is in use`;
  } else if (source.problem !== null) {
    notice = `${source.problem}; the built-in scheme is in use`;
  }
  return {
    scheme: parseScheme(useText) as Base16,
    schemeText: useText,
    weight,
    polarity: look.polarity,
    vars: look.vars,
    misses: look.misses,
    notice,
    fromOverride,
  };
}

/** Apply the token map to an element (the document root). */
export function applyLook(root: HTMLElement, look: ResolvedLook): void {
  for (const [token, value] of Object.entries(look.vars)) root.style.setProperty(token, value);
  root.dataset.mode = look.polarity;
}

/** Set the test override (inspection only, never human functionality). */
export function setLookOverride(o: { schemeText?: string; weight?: LookWeight } | null): void {
  override = o;
}

export function lookOverrideActive(): boolean {
  return override !== null;
}

/** The human-facing summary for plain notices and evidence. */
export function lookSummary(look: ResolvedLook): {
  scheme: string | null;
  weight: LookWeight;
  polarity: string;
  notice: string | null;
} {
  return {
    scheme: schemeName(look.schemeText),
    weight: look.weight,
    polarity: schemePolarity(look.scheme),
    notice: look.notice,
  };
}
