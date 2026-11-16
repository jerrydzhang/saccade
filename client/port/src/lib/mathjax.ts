// MathJax 4 adapter. The bundled tex-svg component is configured before
// load; conversion failures throw so the caller keeps authored source
// visible. Conversion is the promise API: MathJax 4 loads a font's
// dynamic ranges (\mathbb, \mathcal, ...) on first use, which the
// synchronous tex2svg cannot wait for ("MathJax retry"). Glyphs
// inherit surrounding text color via currentColor; sizes stay natural
// (no shrinking) per the ratified direction. MathJax 4's a11y suite
// (explorer, speech, braille) boots on by default and is disabled at
// configure time through the menu settings' enrich switch (t-61):
// rendered math carries no tabindex, so a click cannot steal focus and
// an ordinary keystroke cannot drive the explorer's help dialog with
// its docs.mathjax.org link; no overlays, no SRE worker lane.

import type { MathNode } from "./markdown.ts";

type Tex2Svg = (tex: string, options?: { display?: boolean }) => Promise<HTMLElement>;

interface MathJaxGlobal {
  loader?: { paths?: { mathjax?: string; fonts?: string } };
  startup?: { promise?: Promise<unknown>; document?: unknown; typeset?: boolean };
  tex2svgPromise?: Tex2Svg;
  typesetPromise?: () => Promise<void>;
  tex?: Record<string, unknown>;
  svg?: Record<string, unknown>;
  options?: Record<string, unknown>;
}

declare global {
  interface Window {
    MathJax?: MathJaxGlobal;
  }
}

let readyPromise: Promise<Tex2Svg> | null = null;

function configureBeforeLoad(): void {
  // The speech worker resolves against loader.paths.mathjax; the built
  // frontend carries the pinned sre assets at <origin>/sre, so the
  // offline same-origin path keeps speech from fetching anything
  // external. location.origin covers both the http browser lane and the
  // Tauri protocol (tauri://localhost under WebKitGTK) - the native lane
  // verifies this empirically rather than assuming it. The font's
  // dynamic ranges resolve against loader.paths.fonts the same way
  // (<origin>/fonts/mathjax-newcm-font/svg/dynamic, vite.config.ts);
  // left unset, MathJax would fetch them from a CDN.
  const mathjaxRoot = typeof location !== "undefined" ? location.origin : "";
  window.MathJax = {
    loader: { paths: { mathjax: mathjaxRoot, fonts: `${mathjaxRoot}/fonts` } },
    tex: {
      inlineMath: [["$", "$"]],
      displayMath: [["$$", "$$"]],
      // A failed conversion must surface, never masquerade as content.
      formatError: (_jax: unknown, err: Error) => {
        throw err;
      },
      packages: { "[+]": ["ams", "newcommand", "configmacros"] },
    },
    svg: {
      fontCache: "none",
      scale: 1,
      // stroke/fill follow currentColor so theme changes flow through
      mtextInheritFont: true,
    },
    options: {
      enableMenu: false,
      // MathJax 4's a11y suite is governed by the contextual menu
      // settings, not by the top-level options: the document
      // constructor derives enableSpeech, enableBraille,
      // enableComplexity and enableExplorer from
      // menuOptions.settings.enrich at boot, overwriting any
      // top-level enableExplorer, so enrich: false is the only path
      // that lands (the 4.1 docs call disabling enrichment the way to
      // turn the accessibility tools off at once). enableMenu: false
      // still governs the context menu itself.
      menuOptions: {
        settings: {
          enrich: false,
        },
      },
    },
    startup: {
      typeset: false,
    },
  };
}

async function loadEngine(): Promise<Tex2Svg> {
  configureBeforeLoad();
  await import("mathjax/tex-svg.js");
  const mj = window.MathJax;
  if (!mj) throw new Error("MathJax did not start");
  await mj.startup?.promise;
  if (typeof mj.tex2svgPromise !== "function") {
    throw new Error("MathJax tex2svgPromise is unavailable in this component bundle");
  }
  return mj.tex2svgPromise;
}

export function ensureMathEngine(): Promise<Tex2Svg> {
  if (readyPromise === null) {
    readyPromise = loadEngine().catch((error: unknown) => {
      readyPromise = null;
      throw error;
    });
  }
  return readyPromise;
}

export type MathRender = { ok: true; container: HTMLElement } | { ok: false; message: string };

export async function renderMath(node: MathNode): Promise<MathRender> {
  let tex2svg: Tex2Svg;
  try {
    tex2svg = await ensureMathEngine();
  } catch (error) {
    return { ok: false, message: `math engine unavailable: ${String(error)}` };
  }
  try {
    const container = await tex2svg(node.value, { display: node.type === "math" });
    return { ok: true, container };
  } catch (error) {
    return { ok: false, message: errorMessage(error) };
  }
}

/** A TeX error is not an Error instance (MathJax's TexError carries a
 * message of its own); String() of it would read "[object Object]". */
function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "object" && error !== null && "message" in error) {
    const message = (error as { message: unknown }).message;
    if (typeof message === "string" && message.length > 0) return message;
  }
  return String(error);
}
