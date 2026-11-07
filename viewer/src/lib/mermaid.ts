// Mermaid 11 adapter. A mermaid fence is an ordinary Markdown fence of the
// same source-positioned tree; rendering failure keeps the authored source
// visible with an explicit error. Diagrams behave as figures: fitted to the
// available width, expanded only on activation.

export type MermaidRender =
  | { ok: true; svg: string }
  | { ok: false; message: string };

type MermaidRenderFn = (id: string, code: string) => Promise<{ svg: string }>;

interface MermaidGlobal {
  initialize: (config: Record<string, unknown>) => void;
  render: MermaidRenderFn;
}

let renderFn: Promise<MermaidRenderFn> | null = null;
let renderCount = 0;

async function loadMermaid(): Promise<MermaidRenderFn> {
  const mod = (await import("mermaid")) as { default: MermaidGlobal };
  const mermaid = mod.default;
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    htmlLabels: false,
    fontFamily: "sans-serif",
    flowchart: { htmlLabels: false },
    class: { htmlLabels: false },
  });
  return (id: string, code: string) => mermaid.render(id, code);
}

function engine(): Promise<MermaidRenderFn> {
  if (renderFn === null) {
    renderFn = loadMermaid().catch((error: unknown) => {
      renderFn = null;
      throw error;
    });
  }
  return renderFn;
}

export async function renderMermaid(code: string): Promise<MermaidRender> {
  let render: MermaidRenderFn;
  try {
    render = await engine();
  } catch (error) {
    return { ok: false, message: `mermaid engine unavailable: ${String(error)}` };
  }
  // label wrapping measures DOM text; the first render after a font
  // load can measure before the fonts settle
  await document.fonts.ready;
  const id = `mmd-${++renderCount}`;
  try {
    const { svg } = await render(id, code);
    return { ok: true, svg };
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : String(error) };
  }
}
