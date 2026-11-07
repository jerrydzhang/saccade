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
function diagEnabled(): boolean {
  return (window as unknown as { __BOARD_VIEWER_DIAG__?: boolean }).__BOARD_VIEWER_DIAG__ === true;
}

function logSvgDiagnostics(id: string, svg: string): void {
  // only where the lane's explicit env signal arrived over the bridge;
  // a human run never logs, never probes
  if (!diagEnabled()) return;
  const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
  const texts = doc.querySelectorAll("text");
  const first = texts[0];
  const nonEmpty = [...texts].filter((t) => (t.textContent ?? "").length > 0).length;
  console.log(
    `[viewer-diag] ${id}: svg=${svg.length}b textNodes=${texts.length} nonEmpty=${nonEmpty} ` +
      `firstText=${JSON.stringify(first ? first.textContent : null)} ` +
      `firstFont=${JSON.stringify(first ? first.getAttribute("font-family") : null)}`,
  );
  if (first) {
    console.log(`[viewer-diag] ${id}: firstText.outerHTML=${JSON.stringify(first.outerHTML.slice(0, 300))}`);
  }
  // the raw measurement facts under the broken geometry, with the shim
  // state visible: does the prototype carry the viewer marker, and what
  // does an isolated text element actually report?
  const proto = (window as unknown as { SVGTextContentElement?: { prototype: Record<string, unknown> } })
    .SVGTextContentElement?.prototype;
  console.log(`[viewer-diag] ${id}: shimInstalled=${proto ? Object.hasOwn(proto, "__viewerShim") : "no-proto"}`);
  try {
    const probeSvg = `<svg xmlns="http://www.w3.org/2000/svg"><text x="0" y="0">sac CLI</text></svg>`;
    const probeDoc = new DOMParser().parseFromString(probeSvg, "image/svg+xml");
    const probeText = probeDoc.querySelector("text");
    document.body.appendChild(probeDoc.documentElement);
    const raw = (probeText as SVGTextContentElement).getComputedTextLength();
    const rect = probeText!.getBoundingClientRect();
    probeDoc.documentElement.remove();
    console.log(`[viewer-diag] ${id}: isolated probe length=${raw} rect=${rect.width}`);
  } catch (error) {
    console.log(`[viewer-diag] ${id}: isolated probe FAILED: ${String(error)}`);
  }
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
  // Label wrapping depends on DOM text measurement. In the headless
  // lane the first renders ran before fonts and window geometry were
  // ready and measurement returned nonsense (negative widths), so
  // every render waits for the fonts; the diagnostics record the
  // viewport facts for the evidence trail.
  if (document.fonts.status !== "loaded") {
    if (diagEnabled()) console.log(`[viewer-diag] fonts status=${document.fonts.status}; awaiting ready`);
    await document.fonts.ready;
    if (diagEnabled()) console.log(`[viewer-diag] fonts ready status=${document.fonts.status}`);
  }
  if (diagEnabled()) {
    const vv = window.visualViewport;
    console.log(
      `[viewer-diag] viewport=${window.innerWidth}x${window.innerHeight} ` +
        `outer=${window.outerWidth}x${window.outerHeight} dpr=${window.devicePixelRatio} ` +
        `vv=${vv ? `${vv.width}x${vv.height} scale=${vv.scale}` : "absent"} ` +
        `docFontSize=${getComputedStyle(document.documentElement).fontSize}`,
    );
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
