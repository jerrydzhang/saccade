// The expansions painted onto a rendered board (t-58): an expanded
// incorporation carries data-port-expanded and, after its source
// line, a container holding exactly its cited components; a
// collapsed one carries neither. Like a fold, an expansion is applied
// to the board as it stands, never a rebuild of the whole board.

import { el, renderBlock, type RenderHooks } from "../render.ts";
import type { Expansion } from "./expansion.ts";

/** The cited components of one expansion, rendered. */
export function renderCited(expansion: Expansion, hooks: RenderHooks): HTMLElement {
  const container = el("div", {
    class: "incorporation-cited",
    "data-port-control": "cited",
    "data-port-cited-of": expansion.part,
    "data-port-cited-status": expansion.status,
  });
  if (expansion.status === "loading") {
    container.append(el("p", { class: "provenance" }, "reading the cited components"));
    return container;
  }
  if (expansion.status === "failed") {
    container.append(
      el("p", { class: "render-error", role: "alert" }, `the cited revision could not be read: ${expansion.message}`),
    );
    return container;
  }
  for (const component of expansion.components) {
    if (component.block === null) {
      container.append(
        el("p", { class: "render-error", role: "alert" }, `the select ${component.select} names nothing in the cited revision`),
      );
      continue;
    }
    container.append(renderBlock(component.block, component.part, hooks));
  }
  return container;
}

/** Bring the rendered board in line with the expansions; each freshly
 * rendered container is handed to `rendered` (its hydration). */
export function paintExpansions(
  root: HTMLElement,
  expansions: readonly Expansion[],
  hooks: () => RenderHooks,
  rendered: (container: HTMLElement, expansion: Expansion) => void,
): void {
  const byPart = new Map(expansions.map((expansion) => [expansion.part, expansion]));
  for (const block of root.querySelectorAll<HTMLElement>(".block-incorporation[data-port-part]")) {
    const part = block.getAttribute("data-port-part") ?? "";
    const expansion = byPart.get(part);
    const existing = Array.from(block.children).find((child) => child.classList.contains("incorporation-cited"));
    if (expansion !== undefined && existing?.getAttribute("data-port-cited-status") === expansion.status) continue;
    existing?.remove();
    if (expansion === undefined) {
      block.removeAttribute("data-port-expanded");
      continue;
    }
    block.setAttribute("data-port-expanded", "");
    const container = renderCited(expansion, hooks());
    const line = Array.from(block.children).find((child) => child.classList.contains("provenance"));
    if (line !== undefined) line.after(container);
    else block.prepend(container);
    rendered(container, expansion);
  }
}
