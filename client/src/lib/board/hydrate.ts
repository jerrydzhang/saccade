// Async hydration of placeholders the DOM renderer emits: math (MathJax),
// mermaid diagrams, markdown images and figure assets (through the model's
// source adapters). A conversion failure always leaves authored source
// visible with an explicit error; it never leaves an unexplained blank.
//
// Every await is generation-checked: a stale hydration drops itself at the
// boundary and writes nothing into a container it no longer belongs to.
// Outcomes are reported generation-scoped so the model carries actual
// rendering results, never an implied parsed-equals-rendered claim.

import { renderMath } from "../mathjax.ts";
import { renderMermaid } from "../mermaid.ts";
import type { ViewerModel, RenderOutcome } from "./viewmodel.ts";
import { sourceFor, type Location } from "../sources.ts";
import { TrackerSource } from "../tracker.ts";
import { el } from "../render.ts";

export async function hydrate(
  container: HTMLElement,
  model: ViewerModel,
  location: Location,
  generation: number,
  collected: RenderOutcome[],
): Promise<RenderOutcome[]> {
  const stale = (): boolean => model.currentGeneration !== generation;

  const mathPending = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="math"]'));
  for (const span of mathPending) {
    await hydrateMath(span, collected);
    if (stale()) return collected;
  }

  const mermaidPending = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="mermaid"]'));
  for (const div of mermaidPending) {
    await hydrateMermaid(div, collected);
    if (stale()) return collected;
  }

  const figures = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="figure"]'));
  for (const frame of figures) {
    await hydrateFigure(frame, model, location, collected);
    if (stale()) return collected;
  }

  const images = Array.from(container.querySelectorAll<HTMLElement>('[data-pending="image"]'));
  for (const span of images) {
    await hydrateImage(span, model, location, collected);
    if (stale()) return collected;
  }

  await hydrateCitations(container, model, location);
  if (stale()) return collected;

  return collected;
}

/** The incorporation citations: on the tracker lane the dense revision
 * resolves through the tracker's own history and the citation becomes a
 * link to that revision; anywhere else the authored claim and the hash
 * stay as rendered - a provenance fact, never a guessed number. */
async function hydrateCitations(
  container: HTMLElement,
  model: ViewerModel,
  location: Location,
): Promise<void> {
  if (!location.machine.startsWith("tracker:")) return;
  const source = sourceFor(location.machine);
  if (!(source instanceof TrackerSource)) return;
  const spans = Array.from(container.querySelectorAll<HTMLElement>(".citation[data-cite-task]"));
  for (const span of spans) {
    const task = span.dataset.citeTask ?? "";
    const commit = span.dataset.citeCommit ?? "";
    const resolved = await source.citationLabel(task, commit).catch(() => null);
    if (resolved) {
      const link = el(
        "a",
        {
          class: "citation-link",
          href: "#",
          "data-control": "board:open-citation",
          "data-open-path": resolved.path,
        },
        resolved.label,
      );
      link.addEventListener("click", (event) => {
        event.preventDefault();
        void model.open({ machine: location.machine, path: resolved.path });
      });
      span.replaceChildren(link);
    } else {
      span.append(el("span", { class: "citation-unresolved" }, ` (${commit.slice(0, 8)})`));
    }
  }
}

async function hydrateMath(span: HTMLElement, outcomes: RenderOutcome[]): Promise<void> {
  const tex = span.dataset.unitTex ?? "";
  const line = span.dataset.line ? Number(span.dataset.line) : undefined;
  const result = await renderMath({ type: span.dataset.unitDisplay === "true" ? "math" : "inlineMath", value: tex, authored: span.dataset.unitSource ?? "", pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } });
  if (!result.ok) {
    span.classList.add("math-error");
    span.title = `math conversion failed: ${result.message}`;
    span.textContent = span.dataset.unitSource ?? `$${tex}$`;
    outcomes.push({ kind: "math", status: "error", message: result.message, line });
    return;
  }
  span.textContent = "";
  span.append(result.container);
  outcomes.push({ kind: "math", status: "ok", message: "", line });
}

async function hydrateMermaid(div: HTMLElement, outcomes: RenderOutcome[]): Promise<void> {
  const source = div.dataset.unitSource ?? "";
  const line = div.dataset.line ? Number(div.dataset.line) : undefined;
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
    outcomes.push({ kind: "mermaid", status: "error", message: result.message, line });
    // The authored fence stays inspectable.
    return;
  }
  if (pre) pre.remove();
  const figure = el(
    "div",
    {
      class: "diagram-frame",
      "data-diagram-source": source,
      role: "button",
      tabindex: "0",
      "aria-label": "diagram, activate to expand",
      // stable target id: the frame is addressed through its unit, so
      // the projection lists the same id the rendered document carries
      ...(div.dataset.unitId ? { "data-control": `diagram:${div.dataset.unitId}` } : {}),
    },
  );
  figure.innerHTML = result.svg;
  const svg = figure.querySelector("svg");
  if (svg) {
    svg.setAttribute("class", "diagram-svg");
    svg.removeAttribute("style");
  }
  const caption = el("p", { class: "diagram-caption" }, "diagram - click to expand, source preserved");
  div.append(figure, caption);
  outcomes.push({ kind: "mermaid", status: "ok", message: "", line });
}

async function hydrateFigure(frame: HTMLElement, model: ViewerModel, location: Location, outcomes: RenderOutcome[]): Promise<void> {
  const figureEl = frame.closest<HTMLElement>(".block-figure");
  const input = figureEl?.dataset.figureInput ?? "";
  const block = { kind: "figure", input, caption: "", pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } } as const;
  const outcome = await model.loadFigureAsset(location, block);
  if (outcome === "stale") return;
  frame.textContent = "";
  if (typeof outcome === "string") {
    const denied = outcome.startsWith("denied");
    frame.classList.add(denied ? "asset-denied" : "asset-missing");
    frame.append(el("p", { class: "render-error", role: "alert" }, `figure unavailable: ${outcome}`));
    outcomes.push({ kind: "figure", status: denied ? "denied" : "missing", message: outcome });
    return;
  }
  const mime = input.endsWith(".svg") ? "image/svg+xml" : "image/png";
  const b64 = bytesToBase64(outcome);
  const img = el("img", { class: "figure-img", alt: input, src: `data:${mime};base64,${b64}` });
  frame.append(img);
  outcomes.push({ kind: "figure", status: "ok", message: input });
}

async function hydrateImage(span: HTMLElement, model: ViewerModel, location: Location, outcomes: RenderOutcome[]): Promise<void> {
  const src = span.dataset.imageSrc ?? "";
  const outcome = await model.loadFigureAsset(location, { kind: "figure", input: src, caption: "", pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } });
  if (outcome === "stale") return;
  span.textContent = "";
  if (typeof outcome === "string") {
    span.classList.add("asset-missing");
    span.append(el("span", { class: "render-error", role: "alert" }, `image unavailable: ${outcome}`));
    outcomes.push({ kind: "image", status: outcome.startsWith("denied") ? "denied" : "missing", message: outcome });
    return;
  }
  const mime = src.endsWith(".svg") ? "image/svg+xml" : "image/png";
  const img = el("img", { class: "md-img", alt: src, src: `data:${mime};base64,${bytesToBase64(outcome)}` });
  span.append(img);
  outcomes.push({ kind: "image", status: "ok", message: src });
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
