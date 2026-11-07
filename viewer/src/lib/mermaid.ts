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

// Diagnostic evidence for the native lane only: what the produced SVG
// actually contains, so a rendering gap is attributable (text nodes
// present but empty, absent entirely, or font resolution). Gated by
// the same env var the shell reads for console observability.
function logSvgDiagnostics(id: string, svg: string): void {
  const gated = (window as unknown as { __BOARD_VIEWER_DIAG__?: boolean }).__BOARD_VIEWER_DIAG__;
  if (!gated) return;
  const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
  const texts = doc.querySelectorAll("text");
  const first = texts[0];
  console.log(
    `[viewer-diag] ${id}: svg=${svg.length}b textNodes=${texts.length} ` +
      `firstText=${JSON.stringify(first ? first.textContent : null)} ` +
      `firstFont=${JSON.stringify(first ? first.getAttribute("font-family") : null)}`,
  );
  const live = document.querySelector(`#${id} text, #${id} span`);
  if (live) {
    const style = getComputedStyle(live);
    console.log(
      `[viewer-diag] ${id}: liveText=${JSON.stringify(live.textContent)} ` +
        `computedFont=${JSON.stringify(style.fontFamily)} ` +
        `fontSize=${JSON.stringify(style.fontSize)} fill=${JSON.stringify(style.fill)}`,
    );
  } else {
    console.log(`[viewer-diag] ${id}: no live text/span under #${id}`);
  }
  const probe = document.createElement("span");
  probe.textContent = "mermaid-font-probe";
  probe.style.fontFamily = 'sans-serif';
  document.body.appendChild(probe);
  const rect = probe.getBoundingClientRect();
  console.log(`[viewer-diag] font probe sans-serif width=${rect.width} height=${rect.height}`);
  probe.remove();
}

export async function renderMermaid(code: string): Promise<MermaidRender> {
  let render: MermaidRenderFn;
  try {
    render = await engine();
  } catch (error) {
    return { ok: false, message: `mermaid engine unavailable: ${String(error)}` };
  }
  const id = `mmd-${++renderCount}`;
  try {
    const { svg } = await render(id, code);
    logSvgDiagnostics(id, svg);
    return { ok: true, svg };
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : String(error) };
  }
}
