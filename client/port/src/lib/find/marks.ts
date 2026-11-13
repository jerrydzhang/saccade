// Find's marks on the rendered text (D2): every occurrence of the
// committed or live query is wrapped in a <mark data-port-find-match>,
// the active one also carries data-port-find-active. Painting walks a
// matched part's rendered text in document order, skipping what the
// shown text (shown.ts) skips - math, mermaid, figures - so the k-th
// hit here is the session's k-th occurrence in that part. When the
// counts disagree (rendered text the model did not predict), the part
// itself takes the mark instead: a coarser mark, never a wrong one.
//
// The painter owns only its own <mark> elements and the attributes it
// sets; unpainting restores the text nodes exactly.

import { normalizeQuery } from "./session.ts";

export type MarkPlan = Readonly<{
  query: string;
  /** Per part, the occurrence count the session found there, and the
   * index (within that part) of the active occurrence, or -1. */
  parts: ReadonlyMap<string, { count: number; active: number }>;
}>;

const SKIP = "[data-unit-tex], [data-copy-unit='mermaid'], .block-figure, svg, script, style, input, textarea, [data-port-find-skip]";

function partElement(root: HTMLElement, part: string): HTMLElement | null {
  const element = root.querySelector(`[data-port-part=${JSON.stringify(part)}]`);
  return element instanceof HTMLElement ? element : null;
}

/** The text nodes that belong to `part` itself, in document order: not
 * those inside a nested part (a group's children are their own parts)
 * nor inside skipped renderings. */
function ownTextNodes(element: HTMLElement): Text[] {
  const out: Text[] = [];
  const walker = element.ownerDocument.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (parent === null) return NodeFilter.FILTER_REJECT;
      const skipped = parent.closest(SKIP);
      if (skipped !== null && element.contains(skipped)) return NodeFilter.FILTER_REJECT;
      // a prose ref is its own target but its text is its block's
      const owner = parent.closest('[data-port-part]:not([data-port-kind="ref"])');
      if (owner !== element) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) out.push(node as Text);
  return out;
}

export function unpaintMarks(root: HTMLElement): void {
  const parents = new Set<Node>();
  for (const mark of root.querySelectorAll("mark[data-port-find-match]")) {
    const parent = mark.parentNode;
    if (parent === null) continue;
    while (mark.firstChild !== null) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    parents.add(parent);
  }
  for (const parent of parents) parent.normalize();
  for (const marked of root.querySelectorAll("[data-port-find-match], [data-port-find-active]")) {
    marked.removeAttribute("data-port-find-match");
    marked.removeAttribute("data-port-find-active");
  }
}

function wrap(node: Text, start: number, end: number, active: boolean): Text {
  // split off [start, end) of this node into a mark; answers the
  // remainder after the mark (the next piece to search)
  const middle = start > 0 ? node.splitText(start) : node;
  const rest = middle.splitText(end - start);
  const mark = node.ownerDocument.createElement("mark");
  mark.className = "port-find-hit";
  mark.setAttribute("data-port-find-match", "");
  if (active) mark.setAttribute("data-port-find-active", "");
  middle.parentNode?.insertBefore(mark, middle);
  mark.appendChild(middle);
  return rest;
}

function paintPart(element: HTMLElement, needle: string, count: number, active: number): boolean {
  const nodes = ownTextNodes(element);
  const text = nodes.map((node) => node.data).join("");
  // the session lowercases the same way (session.ts searchUnits)
  const haystack = text.toLowerCase();
  const hits: Array<[number, number]> = [];
  for (let from = 0; ; ) {
    const index = haystack.indexOf(needle, from);
    if (index < 0) break;
    hits.push([index, index + needle.length]);
    from = index + Math.max(needle.length, 1);
  }
  if (hits.length !== count) return false;
  // map each hit onto the text nodes it spans, last first so earlier
  // offsets stay valid while nodes split
  const starts: number[] = [];
  let offset = 0;
  for (const node of nodes) {
    starts.push(offset);
    offset += node.data.length;
  }
  for (let hit = hits.length - 1; hit >= 0; hit--) {
    const [from, to] = hits[hit]!;
    for (let index = nodes.length - 1; index >= 0; index--) {
      const node = nodes[index]!;
      const nodeStart = starts[index]!;
      const nodeEnd = nodeStart + node.data.length;
      if (nodeEnd <= from || nodeStart >= to) continue;
      wrap(node, Math.max(from, nodeStart) - nodeStart, Math.min(to, nodeEnd) - nodeStart, hit === active);
    }
  }
  return true;
}

/** Repaint every mark under `root` for `plan` (the previous marks go
 * first). Answers the active mark (its first piece), or the active
 * part when that part fell back to a part-level mark, or null. */
export function paintMarks(root: HTMLElement, plan: MarkPlan): HTMLElement | null {
  unpaintMarks(root);
  const trimmed = plan.query.trim();
  if (trimmed.length === 0) return null;
  const needle = normalizeQuery(trimmed).toLowerCase();
  let activeElement: HTMLElement | null = null;
  for (const [part, { count, active }] of plan.parts) {
    const element = partElement(root, part);
    if (element === null) continue;
    if (paintPart(element, needle, count, active)) {
      if (active >= 0) activeElement = element.querySelector<HTMLElement>("mark[data-port-find-active]") ?? activeElement;
      continue;
    }
    element.setAttribute("data-port-find-match", "");
    if (active >= 0) {
      element.setAttribute("data-port-find-active", "");
      activeElement = element;
    }
  }
  return activeElement;
}
