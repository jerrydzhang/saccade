// Diagram expansion: activating an embedded diagram expands it into the
// available viewing space where zoom/pan and fit/reset are available; closing
// returns to the prior reading position (the board underneath never moves,
// so no scroll restoration is needed). The embedded diagram never captures
// zoom/pan gestures.

import { el } from "./render.ts";

let activeOverlay: HTMLElement | null = null;

function close(): void {
  if (activeOverlay) {
    activeOverlay.remove();
    activeOverlay = null;
  }
}

export function expandDiagram(frame: HTMLElement): void {
  if (activeOverlay) close();
  const svg = frame.querySelector("svg");
  if (!svg) return;
  const cloned = svg.cloneNode(true) as SVGElement;
  cloned.setAttribute("class", "diagram-svg expanded");

  const stage = el("div", { class: "diagram-stage" });
  stage.append(cloned);

  const zoomLabel = el("span", { class: "diagram-zoom-label" }, "100%");
  let scale = 1;
  let panX = 0;
  let panY = 0;
  let dragging: { x: number; y: number; baseX: number; baseY: number } | null = null;

  const apply = (): void => {
    // CSS transform on the cloned <svg> itself, not the SVG transform
    // ATTRIBUTE: WebKit ignores transform on the outermost svg element
    // (Chromium honors it per SVG2), so the attribute broke zoom/pan on
    // the real client while the browser lane passed. transform-origin
    // keeps the translate/scale math identical in both engines, and
    // getBoundingClientRect reflects CSS transforms, so fit stays
    // consistent.
    cloned.style.transformOrigin = "0 0";
    cloned.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    zoomLabel.textContent = `${Math.round(scale * 100)}%`;
  };

  const controls = el(
    "div",
    { class: "diagram-controls" },
    el("button", { type: "button", "data-action": "zoom-in" }, "+"),
    el("button", { type: "button", "data-action": "zoom-out" }, "-"),
    el("button", { type: "button", "data-action": "fit" }, "fit"),
    el("button", { type: "button", "data-action": "reset" }, "reset"),
    zoomLabel,
    el("button", { type: "button", "data-action": "close" }, "close"),
  );

  const dialog = el(
    "div",
    { class: "diagram-overlay", role: "dialog", "aria-label": "expanded diagram" },
    controls,
    stage,
  );

  controls.addEventListener("click", (event) => {
    const target = (event.target as HTMLElement).closest("[data-action]");
    if (!target) return;
    const action = target.getAttribute("data-action");
    if (action === "zoom-in") scale = Math.min(scale * 1.25, 8);
    else if (action === "zoom-out") scale = Math.max(scale / 1.25, 0.2);
    else if (action === "reset") {
      scale = 1;
      panX = 0;
      panY = 0;
    } else if (action === "fit") {
      const stageRect = stage.getBoundingClientRect();
      const svgRect = cloned.getBoundingClientRect();
      if (svgRect.width > 0 && svgRect.height > 0) {
        scale = Math.min(stageRect.width / svgRect.width, stageRect.height / svgRect.height, 8);
      }
      panX = 0;
      panY = 0;
    } else if (action === "close") {
      close();
      return;
    }
    apply();
  });

  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    scale = Math.min(Math.max(scale * factor, 0.2), 8);
    apply();
  });

  stage.addEventListener("pointerdown", (event) => {
    dragging = { x: event.clientX, y: event.clientY, baseX: panX, baseY: panY };
    stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = (event.clientX - dragging.x) / scale;
    const dy = (event.clientY - dragging.y) / scale;
    panX = dragging.baseX + dx;
    panY = dragging.baseY + dy;
    apply();
  });
  stage.addEventListener("pointerup", () => {
    dragging = null;
  });
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      close();
    }
  });

  apply();
  document.body.append(dialog);
  activeOverlay = dialog;
}

export function bindDiagramExpansion(root: HTMLElement): void {
  root.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const frame = target.closest<HTMLElement>(".diagram-frame");
    if (frame && !target.closest("button")) {
      expandDiagram(frame);
    }
  });
  root.addEventListener("keydown", (event) => {
    const target = event.target as HTMLElement;
    if ((event.key === "Enter" || event.key === " ") && target.classList.contains("diagram-frame")) {
      event.preventDefault();
      expandDiagram(target);
    }
  });
}
