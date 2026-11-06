// Async hydration of placeholders the DOM renderer emits: math (MathJax),
// mermaid diagrams, markdown images and figure assets (through the model's
// source adapters). A conversion failure always leaves authored source
// visible with an explicit error; it never leaves an unexplained blank.

import { renderMath } from "./mathjax.ts";
import { renderMermaid } from "./mermaid.ts";
import type { ViewerModel } from "./viewmodel.ts";
import type { Location } from "./sources.ts";
import { el } from "./render.ts";

export async function hydrate(container: HTMLElement, model: ViewerModel, location: Location): Promise<void> {
  const mathPending = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="math"]'));
  await Promise.all(mathPending.map((span) => hydrateMath(span)));

  const mermaidPending = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="mermaid"]'));
  await Promise.all(mermaidPending.map((div) => hydrateMermaid(div)));

  const figures = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="figure"]'));
  await Promise.all(figures.map((frame) => hydrateFigure(frame, model, location)));

  const images = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="image"]'));
  await Promise.all(images.map((span) => hydrateImage(span, model, location)));
}

async function hydrateMath(span: HTMLElement): Promise<void> {
  const tex = span.dataset.unitTex ?? "";
  const result = await renderMath({ type: span.dataset.unitDisplay === "true" ? "math" : "inlineMath", value: tex, pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } });
  if (!result.ok) {
    span.classList.add("math-error");
    span.title = `math conversion failed: ${result.message}`;
    span.textContent = span.dataset.unitSource ?? `$${tex}$`;
    return;
  }
  span.textContent = "";
  span.append(result.container);
}

async function hydrateMermaid(div: HTMLElement): Promise<void> {
  const source = div.dataset.unitSource ?? "";
  const result = await renderMermaid(source);
  const pre = div.querySelector("pre");
  if (!result.ok) {
    div.classList.add("mermaid-error");
    const note = el("p", { class: "render-error", role: "alert" }, `mermaid rendering failed: ${result.message}`);
    if (pre) {
      div.insertBefore(note, pre);
    } else {
      div.append(note);
    }
    // The authored fence stays inspectable.
    return;
  }
  if (pre) pre.remove();
  const figure = el("div", { class: "diagram-frame", "data-diagram-source": source, role: "button", tabindex: "0", "aria-label": "diagram, activate to expand" });
  figure.innerHTML = result.svg;
  const svg = figure.querySelector("svg");
  if (svg) {
    svg.setAttribute("class", "diagram-svg");
    svg.removeAttribute("style");
  }
  const caption = el("p", { class: "diagram-caption" }, "diagram - click to expand, source preserved");
  div.append(figure, caption);
}

async function hydrateFigure(frame: HTMLElement, model: ViewerModel, location: Location): Promise<void> {
  const figureEl = frame.closest<HTMLElement>(".block-figure");
  const path = figureEl?.dataset.figurePath ?? "";
  const block = { kind: "figure", path, pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } } as const;
  const outcome = await model.loadFigureAsset(location, block);
  frame.textContent = "";
  if (typeof outcome === "string") {
    frame.classList.add(outcome.startsWith("denied") ? "asset-denied" : "asset-missing");
    frame.append(el("p", { class: "render-error", role: "alert" }, `figure unavailable: ${outcome}`));
    return;
  }
  const mime = path.endsWith(".svg") ? "image/svg+xml" : "image/png";
  const b64 = bytesToBase64(outcome);
  const img = el("img", { class: "figure-img", alt: path, src: `data:${mime};base64,${b64}` });
  frame.append(img);
}

async function hydrateImage(span: HTMLElement, model: ViewerModel, location: Location): Promise<void> {
  const src = span.dataset.imageSrc ?? "";
  const outcome = await model.loadFigureAsset(location, { kind: "figure", path: src, pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } });
  span.textContent = "";
  if (typeof outcome === "string") {
    span.classList.add("asset-missing");
    span.append(el("span", { class: "render-error", role: "alert" }, `image unavailable: ${outcome}`));
    return;
  }
  const mime = src.endsWith(".svg") ? "image/svg+xml" : "image/png";
  const img = el("img", { class: "md-img", alt: src, src: `data:${mime};base64,${bytesToBase64(outcome)}` });
  span.append(img);
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
