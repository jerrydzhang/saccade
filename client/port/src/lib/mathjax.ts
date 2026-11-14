// MathJax 4 adapter. The bundled tex-svg component is configured before
// load; conversion failures throw so the caller keeps authored source
// visible. Conversion is the promise API: MathJax 4 loads a font's
// dynamic ranges (\mathbb, \mathcal, ...) on first use, which the
// synchronous tex2svg cannot wait for ("MathJax retry"). Glyphs
// inherit surrounding text color via currentColor; sizes stay natural
// (no shrinking) per the ratified direction.

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
