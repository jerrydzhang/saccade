// MathJax 4 adapter. The bundled tex-svg component is configured before
// load; conversion failures throw so the caller keeps authored source
// visible. Glyphs inherit surrounding text color via currentColor; sizes
// stay natural (no shrinking) per the ratified direction.

import type { MathNode } from "./markdown.ts";

type Tex2Svg = (tex: string, options?: { display?: boolean }) => HTMLElement;

interface MathJaxGlobal {
  loader?: { paths?: { mathjax?: string } };
  startup?: { promise?: Promise<unknown>; document?: unknown; typeset?: boolean };
  tex2svg?: Tex2Svg;
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
  // verifies this empirically rather than assuming it.
  const mathjaxRoot = typeof location !== "undefined" ? location.origin : "";
  window.MathJax = {
    loader: { paths: { mathjax: mathjaxRoot } },
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
  if (typeof mj.tex2svg !== "function") {
    throw new Error("MathJax tex2svg is unavailable in this component bundle");
  }
  return mj.tex2svg;
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

export type MathRender =
  | { ok: true; container: HTMLElement }
  | { ok: false; message: string };

export async function renderMath(node: MathNode): Promise<MathRender> {
  let tex2svg: Tex2Svg;
  try {
    tex2svg = await ensureMathEngine();
  } catch (error) {
    return { ok: false, message: `math engine unavailable: ${String(error)}` };
  }
  try {
    const container = tex2svg(node.value, { display: node.type === "math" });
    return { ok: true, container };
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : String(error) };
  }
}
