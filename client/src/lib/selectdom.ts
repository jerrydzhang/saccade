// The DOM half of select-first (t-14 "Interaction"): what can be
// selected in this window, the label chips over the targets, and the
// primary acts the document already carries. The pure half (label
// assignment, typed narrowing) lives in select.ts and is pinned by the
// node suite; nothing here decides, it only collects and paints.

import { assignLabels, isTextKind, type TargetKind } from "./select.ts";

export type SelectableTarget = {
  /** Stable id: the data-control or data-block-id the target carries. */
  id: string;
  kind: TargetKind;
  element: HTMLElement;
  /** The visible text, for typed narrowing of text targets. */
  text: string;
};

function kindOf(element: HTMLElement): TargetKind {
  if (element.classList.contains("block-text")) return "text";
  if (element.classList.contains("block-annotation")) return "annotation";
  if (element.classList.contains("block-figure")) return "figure";
  if (element.classList.contains("block-incorporation")) return "incorporation";
  if (element.classList.contains("block-unknown")) return "unknown";
  if (element.classList.contains("diagram-frame")) return "diagram";
  if (element.classList.contains("citation-link")) return "citation";
  return "control";
}

/** Every selectable target in document order: rendered blocks, the
 * chrome controls, the citation links. */
export function collectTargets(root: HTMLElement): SelectableTarget[] {
  const targets: SelectableTarget[] = [];
  const seen = new Set<HTMLElement>();
  for (const node of root.querySelectorAll<HTMLElement>(".block, .diagram-frame, .citation-link, [data-control]")) {
    if (seen.has(node)) continue;
    seen.add(node);
    targets.push({
      id: node.getAttribute("data-control") ?? node.getAttribute("data-block-id") ?? "",
      kind: kindOf(node),
      element: node,
      text: (node.textContent ?? "").trim(),
    });
  }
  return targets;
}

/** The labels for a collected target list: lowercase on text targets,
 * uppercase on the rest. */
export function labelsFor(targets: SelectableTarget[]): string[] {
  return assignLabels(targets.map((target) => target.kind));
}

/** Show the label chips over the targets. Returns the hide function;
 * the caller hides on selection, Escape, scroll, or re-render. */
export function showLabels(targets: SelectableTarget[], labels: string[]): () => void {
  const layer = document.createElement("div");
  layer.className = "select-labels";
  targets.forEach((target, index) => {
    const label = labels[index];
    if (!label) return;
    const rect = target.element.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return;
    const chip = document.createElement("span");
    chip.className = `select-label select-label-${isTextKind(target.kind) ? "text" : "other"}`;
    chip.textContent = label;
    chip.style.left = `${Math.max(4, rect.left - 2)}px`;
    chip.style.top = `${Math.max(4, rect.top - 2)}px`;
    layer.append(chip);
  });
  document.body.append(layer);
  return () => layer.remove();
}

/** Mark the selected target and scroll it into view. */
export function highlight(target: SelectableTarget | null, previous: SelectableTarget | null): void {
  if (previous && previous.element !== target?.element) {
    previous.element.classList.remove("select-selected");
  }
  if (target) {
    target.element.classList.add("select-selected");
    target.element.scrollIntoView({ block: "center", behavior: "auto" });
  }
}

/** The scroll container's stepping, in lines and pages: the motion keys
 * ride the content host's own scroller. */
export function scrollByLines(root: HTMLElement | undefined, lines: number): void {
  if (!root) return;
  root.scrollBy({ top: lines * 24, behavior: "auto" });
}

export function scrollByPages(root: HTMLElement | undefined, pages: number): void {
  if (!root) return;
  root.scrollBy({ top: pages * root.clientHeight * 0.9, behavior: "auto" });
}

export function scrollTop(root: HTMLElement | undefined): void {
  if (!root) return;
  root.scrollTo({ top: 0, behavior: "auto" });
}

export function scrollBottom(root: HTMLElement | undefined): void {
  if (!root) return;
  root.scrollTo({ top: root.scrollHeight, behavior: "auto" });
}

/** True while the event rides a text field: the viewer's keymap must
 * not steal the human's typing. */
export function inTextField(event: Event): boolean {
  const target = event.target;
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  );
}
